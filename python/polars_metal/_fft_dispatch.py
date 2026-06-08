"""M6 A3: execute detected FFT bindings on the GPU and stitch Struct{real,imag} columns in.

Collect-and-stitch over whole, materialized columns (chunk-safe), mirroring _vector_dispatch:
  1. drop the sentinel output columns → CPU-collect the rest (projection pushdown elides them,
     so the opaque map_batches(_raise) field never runs),
  2. for each binding: materialize the input column (real F32 or Struct{real,imag}), run the GPU
     FFT/iFFT, build the Struct{real,imag} column,
  3. reassemble in original schema order.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from polars_metal import _native
from polars_metal._fft_detect import FftBinding
from polars_metal._fft_namespace import OP_IFFT


def _input_streams(s: pl.Series) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Return (real f32 contiguous, imag f32 contiguous or None, n) for a signal column.

    Real Float32 column → (real, None, n). Struct{real:F32, imag:F32} → (real, imag, n).
    Nulls (outer or in either field) → raise (FFT over nulls is ill-defined).
    """
    n = s.len()
    if s.null_count() != 0:
        raise ValueError("polars_metal: .metal.fft/.ifft input contains nulls")
    s = s.rechunk()
    if s.dtype == pl.Float32:
        re = np.ascontiguousarray(s.to_numpy(), dtype=np.float32)
        return re, None, n
    if isinstance(s.dtype, pl.Struct):
        fields = {f.name: f.dtype for f in s.dtype.fields}
        if fields.get("real") != pl.Float32 or fields.get("imag") != pl.Float32:
            raise ValueError(
                "polars_metal: .metal.ifft struct input must be Struct{real:Float32, imag:Float32}; "
                f"got {s.dtype}"
            )
        re = np.ascontiguousarray(s.struct.field("real").to_numpy(), dtype=np.float32)
        im = np.ascontiguousarray(s.struct.field("imag").to_numpy(), dtype=np.float32)
        return re, im, n
    raise ValueError(
        "polars_metal: .metal.fft/.ifft requires a Float32 column or Struct{real,imag}; "
        f"got {s.dtype}"
    )


def _run_binding(df: pl.DataFrame, b: FftBinding) -> pl.Series:
    re, im, n = _input_streams(df.get_column(b.input_col))
    if n == 0:
        return pl.DataFrame(
            {"real": np.empty(0, np.float32), "imag": np.empty(0, np.float32)}
        ).to_struct(b.out_name)
    inverse = b.op == OP_IFFT
    imag_arg = None if im is None else (im.ctypes.data, im.size)
    real_out, imag_out = _native.execute_fft((re.ctypes.data, re.size), imag_arg, n, inverse)
    return pl.DataFrame(
        {
            "real": np.asarray(real_out, dtype=np.float32),
            "imag": np.asarray(imag_out, dtype=np.float32),
        }
    ).to_struct(b.out_name)


def apply_fft(lf: pl.LazyFrame, bindings: list[FftBinding], collect_fn) -> pl.DataFrame:
    """Dispatch FFT bindings to the GPU and stitch struct columns in. *collect_fn(rest_lf)*
    collects the non-sentinel columns; dropping the sentinel output columns elides their
    computation (including the opaque map_batches(_raise)) via projection pushdown."""
    out_names = [b.out_name for b in bindings]
    order = lf.collect_schema().names()
    rest_lf = lf.drop(out_names)
    df = collect_fn(rest_lf)
    cols: dict[str, pl.Series] = {c: df.get_column(c) for c in df.columns}
    for b in bindings:
        cols[b.out_name] = _run_binding(df, b)
    return pl.DataFrame([cols[c] for c in order])
