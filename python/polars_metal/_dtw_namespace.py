"""M6 A4 DTW: .metal.dtw capture + CPU-raising sentinel builder.

Mirrors the vector-search sentinel: a struct carrying the input column + an
Int64 handle-id (tagged) + an opaque map_batches(_raise) field. The reference
sequence, window, and allow_cpu_fallback flag are held by-reference in a
module-global cache keyed by the handle, popped at dispatch.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import polars as pl

_HANDLE_COUNTER = itertools.count(1)

DTW_SENTINEL_TAG = "__pm_dtw__"


@dataclass(frozen=True)
class DtwSpec:
    reference: Any  # length-L sequence (numpy/list), by-reference
    window: int | None
    allow_cpu_fallback: bool
    query_col: str


_DTW_CACHE: dict[int, DtwSpec] = {}


def _capture(reference: Any, window: int | None, allow_cpu_fallback: bool, query_col: str) -> int:
    handle = next(_HANDLE_COUNTER)
    _DTW_CACHE[handle] = DtwSpec(reference, window, allow_cpu_fallback, query_col)
    return handle


def pop_capture(handle: int) -> DtwSpec | None:
    return _DTW_CACHE.pop(handle, None)


def _raise_cpu(_s: pl.Series) -> pl.Series:
    raise RuntimeError(
        "polars_metal: .metal.dtw requires collect(engine='metal'); "
        "it has no plain-CPU implementation."
    )


def build_dtw_sentinel(seq_expr: pl.Expr, query_col: str, handle: int) -> pl.Expr:
    """A struct sentinel: input col + tagged Int64 handle + CPU-raising field.
    Dispatch drops this output column before the CPU collect (the map_batches
    never runs) and replaces it with the GPU F32 distance column."""
    return pl.struct(
        [
            seq_expr.alias("__pm_dtw_seq"),
            pl.lit(handle, dtype=pl.Int64).alias(f"{DTW_SENTINEL_TAG}{query_col}"),
            seq_expr.map_batches(_raise_cpu, return_dtype=pl.Float32).alias("__pm_dtw_raise"),
        ]
    )


def make_dtw_expr(
    seq_expr: pl.Expr,
    reference: Any,
    window: int | None,
    allow_cpu_fallback: bool,
) -> pl.Expr:
    roots = seq_expr.meta.root_names()
    if len(roots) != 1:
        raise ValueError(
            f"polars_metal: .metal.dtw must apply to a single column (got roots {roots})."
        )
    if window is not None and window < 0:
        raise ValueError("polars_metal: .metal.dtw window must be >= 0 or None.")
    qcol = roots[0]
    handle = _capture(reference, window, allow_cpu_fallback, qcol)
    return build_dtw_sentinel(seq_expr, qcol, handle)
