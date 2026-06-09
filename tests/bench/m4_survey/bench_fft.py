"""1-D FFT: metal engine path vs numpy.fft, at MLX's reliable size.

Honest, apples-to-apples — the metal number is the full engine collect path (detect + dispatch +
readback + struct build), NOT bare FFI, per the A2 honest-baseline discipline.

Size note: MLX's Metal FFT is correct only up to 2^20 (ml-explore/mlx#1800 — open upstream,
silent garbage above 2^23). The engine guards to power-of-2 <= 2^20 on the GPU and falls back to
CPU above that, so this bench measures the regime where the GPU path actually runs: N = 2^20.
The larger-N "headline" awaits the hand-rolled MSL FFT; benching 8M here would just time the CPU
fallback. At 2^20 the GPU path is ~7-8x faster than numpy.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_metal  # noqa: F401  (registers engine + .metal namespace)
from polars_metal import MetalEngine
from tests.bench.m4_survey._timing import time_callable

N = 1 << 20  # 2^20 = 1,048,576 — the largest size MLX's Metal FFT computes correctly.


def bench_engine_path() -> float:
    sig = np.random.default_rng(0xF7).standard_normal(N).astype(np.float32)
    df = pl.DataFrame({"sig": sig}, schema={"sig": pl.Float32})
    expr = pl.col("sig").metal.fft().alias("spec")

    def metal() -> object:
        return df.lazy().with_columns(expr).collect(engine=MetalEngine())

    def numpy_fft() -> object:
        return np.fft.fft(sig)

    metal()  # warmup (MSL/MLX compile + first-dispatch cost)
    metal_res = time_callable(f"metal.fft[N={N:,}]", metal)
    numpy_res = time_callable(f"numpy.fft[N={N:,}]", numpy_fft)
    ratio = metal_res.median_ms / numpy_res.median_ms
    print(
        f"\nFFT engine-path gate (N=2^20, MLX's reliable ceiling):\n"
        f"  metal.fft median_ms={metal_res.median_ms:.3f}\n"
        f"  numpy.fft median_ms={numpy_res.median_ms:.3f}\n"
        f"  ratio_metal_over_numpy={ratio:.4f}"
    )
    assert ratio < 1.0, f"fft gate: metal/numpy={ratio:.4f} not < 1.0"
    return ratio


@pytest.mark.benchmark(group="fft")
def test_bench_fft_engine_gate() -> None:
    """2^20-point fft, metal engine path must beat numpy.fft (ratio_metal_over_numpy < 1.0)."""
    ratio = bench_engine_path()
    assert ratio < 1.0
