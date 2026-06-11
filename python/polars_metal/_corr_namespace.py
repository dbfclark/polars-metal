"""M6 corr: lf.metal.corr() LazyFrame verb — sentinel builder + capture cache.

Registers a LazyFrame `.metal` namespace (separate registry from the Expr-level
`.metal` namespace in _vector_namespace.py). corr() adds a struct sentinel via
with_columns (keeping the input columns so dispatch can read them) carrying a
tagged Int64 handle + a CPU-raising map_batches field. The column list and
force_gpu flag live in a module cache keyed by the handle, popped at dispatch.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import polars as pl

_HANDLE_COUNTER = itertools.count(1)

CORR_SENTINEL_TAG = "__pm_corr__"
CORR_SENTINEL_COL = "__pm_corr_sentinel"


@dataclass(frozen=True)
class CorrSpec:
    columns: tuple[str, ...]
    force_gpu: bool


_CORR_CACHE: dict[int, CorrSpec] = {}


def _capture(columns: tuple[str, ...], force_gpu: bool) -> int:
    handle = next(_HANDLE_COUNTER)
    _CORR_CACHE[handle] = CorrSpec(columns, force_gpu)
    return handle


def pop_capture(handle: int) -> CorrSpec | None:
    return _CORR_CACHE.pop(handle, None)


def _raise_cpu(_s: pl.Series) -> pl.Series:
    raise RuntimeError(
        "polars_metal: .metal.corr() requires collect(engine='metal'); "
        "it has no plain-CPU implementation. Use df.corr() for CPU."
    )


def build_corr_sentinel(any_col: str, handle: int) -> pl.Expr:
    """Struct sentinel: tagged Int64 handle + CPU-raising field. Added (not
    selected) so the input columns survive for dispatch; dropped before the
    rest-collect under engine='metal'."""
    return pl.struct(
        [
            pl.lit(handle, dtype=pl.Int64).alias(CORR_SENTINEL_TAG),
            pl.col(any_col)
            .map_batches(_raise_cpu, return_dtype=pl.Float32)
            .alias("__pm_corr_raise"),
        ]
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
