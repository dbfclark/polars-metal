"""M6 A4 DTW: execute detected bindings on the GPU and stitch the F32 column in.

Collect-and-stitch over whole materialized columns (mirrors _vector_dispatch):
  1. drop sentinel output cols -> CPU-collect the rest (the opaque
     map_batches(_raise) never runs),
  2. per binding: stage the Array(F32,L) column + reference, run execute_dtw,
     restore null rows positionally, build the F32 column,
  3. reassemble in schema order.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from polars_metal import _native
from polars_metal._dtw_detect import DtwBinding
from polars_metal._dtw_namespace import DtwSpec, pop_capture


def _reference_vec(reference) -> np.ndarray:
    r = np.ascontiguousarray(np.asarray(reference, dtype=np.float32)).reshape(-1)
    return r


def _seq_matrix(s: pl.Series) -> tuple[np.ndarray, int, int]:
    if not isinstance(s.dtype, pl.Array) or s.dtype.inner != pl.Float32:
        raise ValueError(f"polars_metal: .metal.dtw requires Array(Float32, L); got {s.dtype}")
    L = s.dtype.size
    n = s.len()
    flat = s.to_numpy()  # (n, L); nulls render as NaN rows — masked below
    m = np.ascontiguousarray(flat, dtype=np.float32).reshape(n, L)
    return m, n, L


def _run_binding(frame: pl.DataFrame, b: DtwBinding) -> pl.Series:
    spec: DtwSpec | None = pop_capture(b.handle)
    if spec is None:
        raise RuntimeError("polars_metal: dtw spec handle missing (already consumed?)")
    s = frame.get_column(b.query_col).rechunk()
    mat, n, L = _seq_matrix(s)
    ref = _reference_vec(spec.reference)
    if ref.shape[0] != L:
        raise ValueError(f"polars_metal: .metal.dtw reference length {ref.shape[0]} != L {L}")
    window = -1 if spec.window is None else int(spec.window)

    null_mask = s.is_null().to_numpy()  # rows to null out afterward
    if n == 0:
        return pl.Series(b.out_name, [], dtype=pl.Float32)
    # Neutralize genuinely-null rows (restored to None afterward). Driven by the
    # ROW null mask, not cell-level NaN: a non-null row that contains a legitimate
    # NaN cell must NOT be silently zeroed (the kernel's fmin drops NaN, so it
    # would return a wrong finite value rather than the oracle's NaN) — raise.
    safe = mat.copy()
    safe[null_mask] = 0.0
    if np.isnan(safe).any():
        raise ValueError(
            "polars_metal: .metal.dtw: a non-null sequence contains NaN, which the "
            "GPU kernel cannot match against the oracle (nulls are supported; NaN cells are not)."
        )
    qflat = np.ascontiguousarray(safe, dtype=np.float32).reshape(-1)
    out = np.empty(n, dtype=np.float32)
    _native.execute_dtw(
        (qflat.ctypes.data, qflat.size),
        (ref.ctypes.data, ref.size),
        (out.ctypes.data, out.size),
        n,
        L,
        window,
    )
    res = pl.Series(b.out_name, out, dtype=pl.Float32)
    if null_mask.any():
        res = res.scatter(np.nonzero(null_mask)[0], None)
    return res


def apply_dtw(lf: pl.LazyFrame, bindings: list[DtwBinding], collect_fn) -> pl.DataFrame:
    out_names = [b.out_name for b in bindings]
    order = lf.collect_schema().names()
    rest_lf = lf.drop(out_names)
    df = collect_fn(rest_lf)
    cols: dict[str, pl.Series] = {c: df.get_column(c) for c in df.columns}
    for b in bindings:
        cols[b.out_name] = _run_binding(df, b)
    return pl.DataFrame([cols[c] for c in order])
