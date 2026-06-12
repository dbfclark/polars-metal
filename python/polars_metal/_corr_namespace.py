"""M6 corr: lf.metal.corr() LazyFrame verb — sentinel builder + capture cache.

Registers a LazyFrame `.metal` namespace (separate registry from the Expr-level
`.metal` namespace in _vector_namespace.py). corr() adds a struct sentinel via
with_columns (keeping the input columns so dispatch can read them) carrying a
tagged Int64 handle + a CPU-raising map_batches field. The column list and
force_gpu flag live in a module cache keyed by the handle, popped at dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from polars_metal._detect_common import CaptureCache, sentinel_fields

CORR_SENTINEL_TAG = "__pm_corr__"
CORR_SENTINEL_COL = "__pm_corr_sentinel"


@dataclass(frozen=True)
class CorrSpec:
    columns: tuple[str, ...]
    force_gpu: bool


_CACHE = CaptureCache()


def _capture(columns: tuple[str, ...], force_gpu: bool) -> int:
    return _CACHE.capture(CorrSpec(columns, force_gpu))


get_capture = _CACHE.get
evict_capture = _CACHE.evict


def _raise_cpu(_s: pl.Series) -> pl.Series:
    raise pl.exceptions.ComputeError(
        "polars_metal: .metal.corr() requires collect(engine='metal'); "
        "it has no plain-CPU implementation. Use df.corr() for CPU."
    )


def build_corr_sentinel(any_col: str, handle: int) -> pl.Expr:
    """Struct sentinel: tagged Int64 handle + CPU-raising field. Added (not
    selected) so the input columns survive for dispatch; dropped before the
    rest-collect under engine='metal'."""
    return pl.struct(
        sentinel_fields(
            pl.col(any_col),
            tag=CORR_SENTINEL_TAG,
            payload=handle,
            raise_alias="__pm_corr_raise",
            tag_exact=True,
            raise_fn=_raise_cpu,
        )
    ).alias(CORR_SENTINEL_COL)


@pl.api.register_lazyframe_namespace("metal")
class MetalLazyNamespace:
    def __init__(self, lf: pl.LazyFrame) -> None:
        self._lf = lf

    def corr(self, force_gpu: bool = False) -> pl.LazyFrame:
        """Pearson correlation matrix of ALL columns of this frame, on the GPU.

        Returns a sentinel-bearing LazyFrame; collect(engine='metal') replaces
        it with the p x p Float32 correlation matrix. Narrow columns upstream with
        .select(...). Float32 output (documented divergence from df.corr()'s F64).
        """
        cols = tuple(self._lf.collect_schema().names())
        if len(cols) == 0:
            raise ValueError("polars_metal: .metal.corr() requires at least one column.")
        handle = _capture(cols, bool(force_gpu))
        return self._lf.with_columns(build_corr_sentinel(cols[0], handle))
