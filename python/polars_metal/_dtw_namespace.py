"""M6 A4 DTW: .metal.dtw capture + CPU-raising sentinel builder.

Mirrors the vector-search sentinel: a struct carrying the input column + an
Int64 handle-id (tagged) + an opaque map_batches(_raise) field. The reference
sequence, window, and allow_cpu_fallback flag are held by-reference in a
module-global cache keyed by the handle, popped at dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from polars_metal._detect_common import CaptureCache, sentinel_fields

DTW_SENTINEL_TAG = "__pm_dtw__"


@dataclass(frozen=True)
class DtwSpec:
    reference: Any  # length-L sequence (numpy/list), by-reference
    window: int | None
    allow_cpu_fallback: bool
    query_col: str


_CACHE = CaptureCache()


def _capture(reference: Any, window: int | None, allow_cpu_fallback: bool, query_col: str) -> int:
    return _CACHE.capture(DtwSpec(reference, window, allow_cpu_fallback, query_col))


get_capture = _CACHE.get
evict_capture = _CACHE.evict


def _raise_cpu(_s: pl.Series) -> pl.Series:
    raise pl.exceptions.ComputeError(
        "polars_metal: .metal.dtw requires collect(engine='metal'); "
        "it has no plain-CPU implementation."
    )


def build_dtw_sentinel(seq_expr: pl.Expr, query_col: str, handle: int) -> pl.Expr:
    """A struct sentinel: input col + tagged Int64 handle + CPU-raising field.
    Dispatch drops this output column before the CPU collect (the map_batches
    never runs) and replaces it with the GPU F32 distance column."""
    return pl.struct(
        sentinel_fields(
            seq_expr,
            tag=DTW_SENTINEL_TAG,
            payload=handle,
            col=query_col,
            in_alias="__pm_dtw_seq",
            raise_alias="__pm_dtw_raise",
            raise_fn=_raise_cpu,
        )
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
