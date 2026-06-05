"""polars-metal: Metal-backed execution engine for Polars on Apple Silicon."""

from __future__ import annotations

import contextlib
import inspect
from functools import partial, wraps
from typing import Any

import polars.lazyframe.frame as _plf

from polars_metal import _native
from polars_metal import _rolling_detect as _rolling_detect_module  # noqa: F401
from polars_metal import (
    _vector_detect as _vector_detect_module,  # noqa: F401  (installs with_columns patch eagerly)
)
from polars_metal import (
    _vector_namespace as _vector_namespace_module,  # noqa: F401  (registers .metal)
)
from polars_metal._callback import execute_with_metal
from polars_metal._engine import MetalEngine

__version__ = _native.version_string()


_PATCH_ATTR = "_polars_metal_original_gpu_engine_callback"
_COLLECT_PATCH_ATTR = "_polars_metal_original_collect"
_EXPECTED_PARAMS = {"engine", "streaming", "background", "new_streaming", "_eager"}

# Optimization flags forwarded verbatim when we rebuild a user-supplied
# `optimizations` object with CSE forced off (see `_opt_flags_without_cse`).
_OPT_FLAG_FIELDS = (
    "predicate_pushdown",
    "projection_pushdown",
    "simplify_expression",
    "slice_pushdown",
    "comm_subplan_elim",
    "cluster_with_columns",
    "collapse_joins",
    "check_order_observe",
    "fast_projection",
    "sort_collapse",
)


def _opt_flags_without_cse(user_opt: Any) -> Any:
    """Return a fresh ``QueryOptFlags`` with ``comm_subexpr_elim`` forced off.

    Polars' common-subexpression-elimination pass hoists shared subexpressions
    into ``__POLARS_CSER_*`` temp columns. For a Metal-routed compute subtree
    that fragments one fused MLX subgraph into several — each temp column
    becomes its own dispatch, with its result round-tripping Series→Metal→Series
    between dispatches. That is exactly the per-dispatch fragmentation
    CLAUDE.md's principle #1 warns against; MLX does its own kernel-level CSE,
    so Polars-side CSE only hurts us here. Measured: the 10M haversine collapses
    from 3 dispatches to 1 with CSE off.

    ``QueryOptFlags.update`` mutates in place and the module-level default is a
    shared singleton, so we never mutate — we construct a new object, copying
    any caller-supplied flags through and overriding only CSE.
    """
    from polars.lazyframe.opt_flags import QueryOptFlags

    kwargs: dict[str, Any] = {}
    if user_opt is not None:
        for field in _OPT_FLAG_FIELDS:
            value = getattr(user_opt, field, None)
            if value is not None:
                kwargs[field] = value
    return QueryOptFlags(comm_subexpr_elim=False, **kwargs)


# Transcendental tokens that signal a fusion-eligible compute chain in
# `lf.explain()` output (e.g. `col("lat").sin()`). We only force CSE off for
# such queries — see `_is_fusion_candidate`. Plain arithmetic / non-compute
# queries keep Polars' default CSE so that fall-back queries are byte-identical
# to pure CPU (forcing CSE off unconditionally exposed a Polars CSE-off
# correctness bug on e.g. `value_counts`/struct-expansion plans).
_FUSION_FUNC_TOKENS = (
    # Transcendentals (haversine / Black-Scholes chains).
    ".sin(",
    ".cos(",
    ".tan(",
    ".sinh(",
    ".cosh(",
    ".tanh(",
    ".arcsin(",
    ".arccos(",
    ".arctan(",
    "arctan2",
    ".arcsinh(",
    ".arccosh(",
    ".arctanh(",
    ".exp(",
    ".exp2(",
    ".log(",
    ".log1p(",
    ".log10(",
    ".log2(",
    ".sqrt(",
    ".cbrt(",
    # Fused reductions / scans over a compute chain. A shared sub-expression
    # feeding two of these (e.g. (x*2).sum() and (x*2).std()) is hoisted by CSE
    # into a temp the fused HStack dispatch can't consume — so these also need
    # CSE off. value_counts / struct-expansion plans contain none of these.
    # (Broadening the set is safe: before this gate CSE was *always* forced off,
    # so more tokens just preserves the old behavior for more queries; only
    # token-free, non-fusion plans take the new CSE-on path.)
    ".sum(",
    ".mean(",
    ".std(",
    ".var(",
    ".cum_sum(",
    ".cum_prod(",
    ".cum_max(",
    ".cum_min(",
)


def _is_fusion_candidate(lf: Any) -> bool:
    """True iff the LazyFrame's plan contains a transcendental compute op — the
    strong signal that a fused MLX subgraph (haversine / Black-Scholes / etc.)
    will run and would be fragmented by Polars CSE. Conservative: any failure or
    no-match returns False, leaving CSE at Polars' (correct) default. Uses
    ``explain()`` (cheap; does not serialize the DataFrame)."""
    try:
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("ignore")
            txt = lf.explain()
        return any(tok in txt for tok in _FUSION_FUNC_TOKENS)
    except Exception:
        return False


def _verify_patch_site() -> None:
    """Assert both Polars functions we're about to patch have the signatures we expect.

    If Polars refactors and renames or reorders parameters, the assertions fire
    with a clear message rather than us silently failing to intercept anything.
    """
    fn = getattr(_plf, "_gpu_engine_callback", None)
    if fn is None:
        raise RuntimeError(
            "polars_metal: polars.lazyframe.frame._gpu_engine_callback is missing. "
            "Polars version may be unsupported; pin to a known-good rev."
        )
    sig = inspect.signature(fn)
    actual_params = set(sig.parameters.keys())
    if actual_params != _EXPECTED_PARAMS:
        raise RuntimeError(
            f"polars_metal: _gpu_engine_callback signature changed. "
            f"Expected params {_EXPECTED_PARAMS}, got {actual_params}. "
            f"Pin to a supported Polars rev or update the patch."
        )

    # LazyFrame.collect patch site: we rely on `post_opt_callback` being
    # accepted (it flows through Polars' internal **_kwargs catch-all, not
    # as a named parameter). Probe at import time by calling collect on a
    # trivial frame with a no-op callback. If Polars renames or removes the
    # internal hook, this raises with a clear message rather than letting
    # our patch silently fail in production.
    collect_fn = getattr(_plf.LazyFrame, "collect", None)
    if collect_fn is None:
        raise RuntimeError(
            "polars_metal: polars.lazyframe.frame.LazyFrame.collect is missing. "
            "Polars version may be unsupported."
        )
    import polars as _pl_top  # local import; avoid circulars at module load

    try:
        _pl_top.LazyFrame({"_pm_probe": [1]}).collect(
            engine="cpu",
            post_opt_callback=lambda *_a, **_kw: None,
        )
    except TypeError as e:
        if "post_opt_callback" in str(e):
            raise RuntimeError(
                "polars_metal: LazyFrame.collect no longer accepts `post_opt_callback`. "
                "Our patch uses this internal hook to inject the engine callback; "
                "without it the patch can't dispatch to MetalEngine. "
                "Pin to a supported Polars rev or rework the patch shape."
            ) from e
        raise


def _patch_gpu_engine_callback() -> None:
    """Wrap Polars' _gpu_engine_callback and LazyFrame.collect for MetalEngine dispatch.

    When engine=MetalEngine() is passed to df.collect():
    - LazyFrame.collect is intercepted: we build the Metal callback, then
      re-call collect with engine="cpu" and inject the callback via
      post_opt_callback (Polars' internal test-only bypass).  In M0 our
      callback returns None, so Polars' CPU executor takes over — correct
      behaviour that exercises the full dispatch path.
    - _gpu_engine_callback is wrapped so that if MetalEngine somehow reaches
      it (e.g., via collect_lazy / other entrypoints), it returns our callback
      rather than raising ValueError.

    Both patches are idempotent — re-importing polars_metal does not double-patch.
    """
    if hasattr(_plf, _PATCH_ATTR):
        return  # already patched

    _verify_patch_site()

    # --- Patch 1: _gpu_engine_callback ---
    # Guards against MetalEngine reaching the function's engine-type validator.
    original_callback = _plf._gpu_engine_callback
    setattr(_plf, _PATCH_ATTR, original_callback)

    def callback_wrapper(engine: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        if isinstance(engine, MetalEngine):
            return partial(execute_with_metal, config=engine)
        return original_callback(engine, **kwargs)

    _plf._gpu_engine_callback = callback_wrapper  # type: ignore[assignment]

    # --- Patch 2: LazyFrame.collect ---
    # Polars' collect() passes `engine` raw to ldf.collect() (Rust), which only
    # accepts strings.  After _gpu_engine_callback it converts GPUEngine → "gpu"
    # but has no branch for MetalEngine.  We intercept before the original so
    # MetalEngine never reaches the Rust boundary as a Python object.
    original_collect = _plf.LazyFrame.collect
    setattr(_plf.LazyFrame, _COLLECT_PATCH_ATTR, original_collect)

    @wraps(original_collect)
    def collect_wrapper(self: Any, *, engine: Any = "auto", **kwargs: Any) -> Any:
        if isinstance(engine, MetalEngine):
            cb = partial(execute_with_metal, config=engine)
            # If the caller already passed a post_opt_callback (Polars'
            # internal hook), chain ours before theirs so both run on the
            # same NodeTraverser. In M0 our callback never modifies the
            # plan (it walks-and-falls-back), so theirs sees the same
            # state Polars would have given them.
            existing_cb = kwargs.pop("post_opt_callback", None)
            if existing_cb is not None:
                ours = cb

                def chained(nt: Any, *args: Any, **kw: Any) -> Any:
                    ours(nt, *args, **kw)
                    return existing_cb(nt, *args, **kw)

                cb = chained
            # Force CSE off for fusion-candidate plans so a compute subtree
            # stays a single fused MLX subgraph rather than fragmenting across
            # one dispatch per hoisted temp column (see `_opt_flags_without_cse`).
            # Gated on `_is_fusion_candidate`: forcing CSE off unconditionally
            # changed results for fall-back queries that hit a Polars CSE-off
            # correctness bug (e.g. `value_counts` / struct expansion). Non-fusion
            # queries keep Polars' default CSE so they stay byte-identical to CPU.
            # A deprecated direct `comm_subexpr_elim` kwarg, if present, is dropped.
            if _is_fusion_candidate(self):
                kwargs.pop("comm_subexpr_elim", None)
                kwargs["optimizations"] = _opt_flags_without_cse(kwargs.pop("optimizations", None))
            # M5 rolling: serialize-detected rolling_* run on a custom Metal kernel.
            # Skip under streaming (adapter is in-memory only) and when nothing matches.
            from polars_metal import _rolling_detect, _rolling_dispatch

            streaming = bool(kwargs.get("streaming") or kwargs.get("new_streaming"))
            rolling_bindings = [] if streaming else _rolling_detect.find_rolling_bindings(self)
            if rolling_bindings:

                def _collect_rest(rest_lf: Any) -> Any:
                    return original_collect(rest_lf, engine="cpu", post_opt_callback=cb, **kwargs)

                return _rolling_dispatch.apply_rolling(self, rolling_bindings, _collect_rest)

            # M6 vector search: serialize-detected .metal.cosine_topk/.knn sentinels
            # run on the GPU via the same M5 collect-and-stitch template. Placed after
            # the rolling block; the two don't co-occur in one outermost layer in
            # practice (and if they did, rolling consumes first — acceptable for MVP).
            from polars_metal import _vector_detect, _vector_dispatch

            vector_bindings = [] if streaming else _vector_detect.find_vector_bindings(self)
            if vector_bindings:

                def _collect_rest_vs(rest_lf: Any) -> Any:
                    return original_collect(rest_lf, engine="cpu", post_opt_callback=cb, **kwargs)

                return _vector_dispatch.apply_vector_search(self, vector_bindings, _collect_rest_vs)
            # post_opt_callback is an internal bypass that injects a callback
            # directly, skipping _gpu_engine_callback. We run the query on
            # the CPU engine; in M0 our callback falls through, so the result
            # is identical to plain engine="cpu".
            return original_collect(self, engine="cpu", post_opt_callback=cb, **kwargs)
        return original_collect(self, engine=engine, **kwargs)

    _plf.LazyFrame.collect = collect_wrapper  # type: ignore[method-assign]


_patch_gpu_engine_callback()


def _warmup_kernels() -> None:
    """Pre-compile common fused-agg signatures at import time (Task 18).

    Cost: ~100-500ms one-time per process. Benefit: first user query of
    common shapes (single F32 Sum, F32 Mean, Q1-shape 10-agg, Q1 disc_price
    expression) doesn't pay MSL compile (~100-300ms each).

    Best-effort: any error (missing Metal device, compile failure) is
    swallowed so module import never breaks. Real failures resurface when
    a query of that shape actually runs.
    """
    with contextlib.suppress(Exception):
        _native.warmup_common_fused_signatures()


_warmup_kernels()


__all__ = ["MetalEngine", "__version__"]
