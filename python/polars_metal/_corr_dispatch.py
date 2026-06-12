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

import weakref

import numpy as np
import polars as pl

from polars_metal import _native
from polars_metal._corr_namespace import CorrSpec, evict_capture, get_capture
from polars_metal._detect_common import SentinelBinding

CORR_P_MIN = 8  # spike crossover: p>=~8 GPU wins; below, CPU df.corr() is faster.


def _cpu_corr_f32(df: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    """Polars CPU correlation, cast to Float32 (the oracle + the small-p path)."""
    return df.select(columns).corr().cast(pl.Float32)


def _column_f32_1d(s: pl.Series) -> np.ndarray:
    """One column → contiguous 1-D F32 numpy. F32 single-chunk no-null columns
    round-trip zero-copy; others pay one cast/rechunk copy (still cheap)."""
    if s.dtype != pl.Float32:
        s = s.cast(pl.Float32)
    return s.rechunk().to_numpy()


def _gpu_corr_f32(df: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    n = df.height
    p = len(columns)
    # Build a (p, n) variable-major buffer the cheap way: each Polars column is a
    # contiguous Arrow buffer, so per-column to_numpy() is zero-copy (F32, single
    # chunk, no nulls — the GPU path only reaches here for such columns); stacking
    # them is one sequential 200MB memcpy. This is ~12x faster than df.to_numpy()
    # (which yields a column-major array) + the ascontiguousarray transpose that a
    # sample-major (n, p) layout would force. The kernel computes corr in (p, n).
    rows = [_column_f32_1d(df.get_column(c)) for c in columns]
    pn = np.ascontiguousarray(np.stack(rows), dtype=np.float32)  # (p, n) row-major
    flat = pn.reshape(-1)
    out = _native.execute_corr((flat.ctypes.data, int(flat.size)), int(p), int(n))
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
    if df.height < 2:
        raise pl.exceptions.ComputeError(
            "polars_metal: .metal.corr() needs at least 2 rows to compute a "
            f"correlation (got {df.height})."
        )
    if has_null:
        return _cpu_corr_f32(df, columns)
    if p < CORR_P_MIN and not spec.force_gpu:
        return _cpu_corr_f32(df, columns)
    return _gpu_corr_f32(df, columns)


def apply_corr(lf: pl.LazyFrame, binding: SentinelBinding, collect_fn) -> pl.DataFrame:
    spec: CorrSpec | None = get_capture(binding.payload)
    if spec is None:
        raise pl.exceptions.ComputeError(
            "polars_metal: corr spec handle missing (already consumed?)"
        )
    # Tie eviction to the lf lifetime: when this lf is GC'd the cache entry is
    # freed. Registering twice (two collects of the same lf) is harmless —
    # both do an idempotent dict.pop.
    weakref.finalize(lf, evict_capture, binding.payload)
    rest_lf = lf.drop(binding.out_name)  # drop sentinel; input columns remain
    df = collect_fn(rest_lf)
    return _run_corr(df, spec)
