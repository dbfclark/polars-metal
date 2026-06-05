"""polars-metal: Metal-backed execution engine for Polars on Apple Silicon."""

from __future__ import annotations

import contextlib
import inspect
from functools import partial, wraps
from typing import Any

import polars.lazyframe.frame as _plf

from polars_metal import _native
from polars_metal import _rolling_detect as _rolling_detect_module  # noqa: F401
from polars_metal import _vector_namespace as _vector_namespace_module  # noqa: F401  (registers .metal)
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
            # Force CSE off for Metal-routed plans so a compute subtree stays
            # a single fused MLX subgraph rather than fragmenting across one
            # dispatch per hoisted temp column (see `_opt_flags_without_cse`).
            # A deprecated direct `comm_subexpr_elim` kwarg, if present, is
            # dropped — the `optimizations` object is the supported channel and
            # we override CSE regardless of what the caller requested.
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
