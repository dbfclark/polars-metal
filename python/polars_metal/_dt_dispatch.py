"""Execute detected dt bindings via the gregorian Metal kernel and stitch
results onto the collected frame. Collect-and-stitch over whole, materialized
columns (chunk-safe), mirroring _rolling_dispatch.

Datetime columns are converted to days-since-1970 host-side via integer
floor-division (numpy `//` floors toward -inf, matching Polars for pre-epoch
values); Date columns feed their physical i32 directly. The kernel computes
every field in Int32; month/day are narrowed to Int8 to match Polars, and
nulls are restored positionally.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from polars_metal import _native
from polars_metal._dt_detect import DtBinding

_FIELD_CODE = {"year": 0, "month": 1, "day": 2}
_FIELD_DTYPE = {"year": pl.Int32, "month": pl.Int8, "day": pl.Int8}
_FIELD_NUMPY_DTYPE = {"year": np.int32, "month": np.int8, "day": np.int8}


def _dt_series(src: pl.Series, b: DtBinding) -> pl.Series:
    """Run the gregorian kernel on *src* and return a named Series of the
    Polars-matching dtype (Int32 for year, Int8 for month/day), with nulls
    restored positionally."""
    n = src.len()
    out_dtype = _FIELD_DTYPE[b.field]
    if n == 0:
        return pl.Series(b.out_name, [], dtype=out_dtype)

    mask = src.is_not_null()
    phys = src.to_physical().fill_null(0).to_numpy()
    if b.is_date:
        days = np.ascontiguousarray(phys, dtype=np.int32)
    else:
        # Floor-div the i64 since-epoch value to days (numpy `//` floors toward
        # -inf, matching Polars for pre-epoch values). The day count fits i32
        # for every representable Polars Datetime (its year range bounds days to
        # ~±3.7M), so the int32 narrowing never overflows in practice.
        days = np.ascontiguousarray((phys // b.units_per_day).astype(np.int32))

    out = np.empty(days.shape[0], dtype=np.int32)
    _native.execute_dt(
        inp=(days.ctypes.data, days.size),
        out=(out.ctypes.data, out.size),
        field=_FIELD_CODE[b.field],
    )

    # Narrow on the numpy side (out is int32): astype(int8) is ~30x cheaper than
    # pl.Series(int32).cast(pl.Int8) (~19ms -> ~0.6ms at 10M rows). year stays int32.
    narrowed = out.astype(_FIELD_NUMPY_DTYPE[b.field], copy=False)
    dense = pl.Series(b.out_name, narrowed, dtype=out_dtype)
    if src.null_count() == 0:
        return dense
    # Restore nulls positionally. `Series.set(filter, None)` is vectorized
    # (~0.8ms at 2M rows) vs an O(n) `[None]*n` + zip_with (~45ms) — keeping the
    # surrounding op at CPU-parity so it never swamps the GPU kernel.
    return dense.set(~mask, None)


def apply_dt(lf: pl.LazyFrame, bindings: list[DtBinding], collect_fn) -> pl.DataFrame:
    """Dispatch dt bindings to the Metal kernel and stitch into a DataFrame.

    *collect_fn(rest_lf)* runs the existing collect path on the non-dt
    columns; projection pushdown on ``lf.drop(out_names)`` elides the dt
    computation from the CPU path so each dt column is computed once, on GPU.
    Column order matches the original LazyFrame's schema.
    """
    out_names = [b.out_name for b in bindings]
    # Capture original output column order before dropping anything.
    order = lf.collect_schema().names()
    # Drop dt output columns; projection pushdown eliminates their computation
    # from the CPU plan (verified: lf.drop([...]).explain() shows only the bare
    # DataFrameScan, no dt expression).
    rest_lf = lf.drop(out_names)
    df = collect_fn(rest_lf)

    # Build a dict from column name → Series so we can stitch in order.
    cols: dict[str, pl.Series] = {c: df.get_column(c) for c in df.columns}
    for b in bindings:
        cols[b.out_name] = _dt_series(df.get_column(b.column), b)
    return pl.DataFrame([cols[c] for c in order])
