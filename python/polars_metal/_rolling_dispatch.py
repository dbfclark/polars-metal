"""Execute detected rolling bindings via the custom Metal kernel and stitch
results onto the collected frame. Collect-and-stitch over whole, materialized
columns (chunk-safe): no map_batches, no streaming.

The split strategy relies on Polars projection pushdown to elide the rolling
computation from the CPU path (verified: lf.drop(out_names).explain() shows
no RollingExpr in the plan).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from polars_metal import _native
from polars_metal._rolling_detect import RollingBinding

_OP_CODE = {"sum": 0, "mean": 1, "var": 2, "std": 3}


def _rolling_series(src: pl.Series, b: RollingBinding) -> pl.Series:
    """Run the Metal rolling kernel on *src* and return a named Float32 Series.

    The returned Series has the same length as *src*: the first ``b.window - 1``
    values are null (structurally, matching Polars' own rolling semantics) and
    the remainder are the kernel output.
    """
    # Null-bearing input: the Metal kernel can't reproduce Polars' null-skipping
    # rolling semantics (it would propagate NaN). Fall back to Polars CPU for
    # this column — exact match, just unaccelerated.
    if src.null_count() > 0:
        expr = getattr(pl.col(b.column), f"rolling_{b.op}")(b.window)
        return src.rename(b.column).to_frame().select(expr.alias(b.out_name)).to_series()

    s = src.rechunk()  # contiguous F32 buffer (zero-copy contract)
    x = np.ascontiguousarray(s.to_numpy(), dtype=np.float32)
    out = np.empty(x.shape[0], dtype=np.float32)
    _native.execute_rolling(
        inp=(x.ctypes.data, x.size),
        out=(out.ctypes.data, out.size),
        w=b.window,
        op=_OP_CODE[b.op],
        ddof=b.ddof,
    )

    # Build a Float32 Series with the first w-1 values null.
    # Use concat([nulls_prefix, valid_suffix]) — the cleanest Polars idiom
    # that handles w=1 (zero-length prefix, no nulls) cleanly.
    n_null = b.window - 1
    if n_null > 0:
        nulls = pl.Series(b.out_name, [None] * n_null, dtype=pl.Float32)
        valid = pl.Series(b.out_name, out[n_null:], dtype=pl.Float32)
        res = pl.concat([nulls, valid])
    else:
        res = pl.Series(b.out_name, out, dtype=pl.Float32)

    return res


def apply_rolling(
    lf: pl.LazyFrame,
    bindings: list[RollingBinding],
    collect_fn,
) -> pl.DataFrame:
    """Dispatch rolling bindings to the Metal kernel and stitch into a DataFrame.

    *collect_fn(rest_lf) -> DataFrame* runs the existing in-memory collect
    path on the non-rolling columns. Projection pushdown on *rest_lf* (obtained
    via ``lf.drop(out_names)``) eliminates the rolling computation from the CPU
    path so that each rolling column is computed exactly once — on the GPU.

    Column order is restored to match the schema declared by the original
    LazyFrame.
    """
    out_names = [b.out_name for b in bindings]
    # Capture original output column order before dropping anything.
    order = lf.collect_schema().names()
    # Drop rolling output columns; projection pushdown eliminates their
    # computation from the CPU plan (verified via explain()).
    rest_lf = lf.drop(out_names)
    df = collect_fn(rest_lf)

    # Build a dict from column name → Series so we can stitch in order.
    cols: dict[str, pl.Series] = {c: df.get_column(c) for c in df.columns}

    for b in bindings:
        cols[b.out_name] = _rolling_series(df.get_column(b.column), b)

    return pl.DataFrame([cols[c] for c in order])
