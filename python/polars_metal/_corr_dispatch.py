"""M6 corr: execute a detected corr binding -> p x p Float32 correlation matrix.

Frame-replacing (unlike dtw/fft/vector which stitch a column into the N-row
frame): collect the input columns, then return a fresh p x p frame. Routing:
  - non-numeric column            → raise
  - any null, or N < 2            → CPU fallback (df.corr() cast F32)
  - p < CORR_P_MIN and not force  → CPU fallback (cast F32)
  - else                          → GPU (execute_corr), result cast F32
F32 output regardless of path (documented divergence from df.corr()'s F64).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from polars_metal import _native
from polars_metal._corr_detect import CorrBinding
from polars_metal._corr_namespace import CorrSpec, pop_capture

CORR_P_MIN = 8  # spike crossover: p>=~8 GPU wins; below, CPU df.corr() is faster.


def _cpu_corr_f32(df: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    """Polars CPU correlation, cast to Float32 (the oracle + the small-p path)."""
    return df.select(columns).corr().cast(pl.Float32)


def _gpu_corr_f32(df: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    n = df.height
    p = len(columns)
    f32 = df.select([pl.col(c).cast(pl.Float32) for c in columns])
    mat = np.ascontiguousarray(f32.to_numpy(), dtype=np.float32)  # (n, p) row-major
    flat = mat.reshape(-1)
    out = _native.execute_corr((flat.ctypes.data, int(flat.size)), int(n), int(p))
    cmat = np.asarray(out, dtype=np.float32).reshape(p, p)
    return pl.DataFrame(
        {columns[j]: pl.Series(columns[j], cmat[:, j], dtype=pl.Float32) for j in range(p)}
    )


def _run_corr(df: pl.DataFrame, spec: CorrSpec) -> pl.DataFrame:
    columns = spec.columns
    # dtype gate (numeric only) + null check, fetching each column once.
    has_null = False
    for c in columns:
        s = df.get_column(c)
        if not s.dtype.is_numeric():
            raise ValueError(
                f"polars_metal: .metal.corr() requires numeric columns; column {c!r} is {s.dtype}."
            )
        if s.null_count() > 0:
            has_null = True
    p = len(columns)
    if has_null or df.height < 2:
        return _cpu_corr_f32(df, columns)
    if p < CORR_P_MIN and not spec.force_gpu:
        return _cpu_corr_f32(df, columns)
    return _gpu_corr_f32(df, columns)


def apply_corr(lf: pl.LazyFrame, binding: CorrBinding, collect_fn) -> pl.DataFrame:
    spec: CorrSpec | None = pop_capture(binding.handle)
    if spec is None:
        raise RuntimeError("polars_metal: corr spec handle missing (already consumed?)")
    rest_lf = lf.drop(binding.out_name)  # drop sentinel; input columns remain
    df = collect_fn(rest_lf)
    return _run_corr(df, spec)
