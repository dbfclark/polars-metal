"""polars-metal: Metal-backed execution engine for Polars on Apple Silicon."""

from __future__ import annotations

import inspect
from functools import partial, wraps
from typing import Any

import polars.lazyframe.frame as _plf

from polars_metal import _native
from polars_metal._callback import execute_with_metal
from polars_metal._engine import MetalEngine

__version__ = _native.version_string()


_PATCH_ATTR = "_polars_metal_original_gpu_engine_callback"
_COLLECT_PATCH_ATTR = "_polars_metal_original_collect"
_EXPECTED_PARAMS = {"engine", "streaming", "background", "new_streaming", "_eager"}


def _verify_patch_site() -> None:
    """Assert the Polars callback we're about to patch has the signature we expect.

    If Polars refactors and renames or reorders parameters, the assertion fires
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
            # post_opt_callback is an internal bypass that injects a callback
            # directly, skipping _gpu_engine_callback. We run the query on
            # the CPU engine; in M0 our callback falls through, so the result
            # is identical to plain engine="cpu".
            return original_collect(self, engine="cpu", post_opt_callback=cb, **kwargs)
        return original_collect(self, engine=engine, **kwargs)

    _plf.LazyFrame.collect = collect_wrapper  # type: ignore[method-assign]


_patch_gpu_engine_callback()


__all__ = ["MetalEngine", "__version__"]
