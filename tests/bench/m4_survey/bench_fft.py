"""1-D FFT: metal engine path vs numpy.fft, at the large-pow2 regime.

Honest, apples-to-apples — the metal number is the full engine collect path (detect + dispatch +
readback + struct build), NOT bare FFI, per the A2 honest-baseline discipline.

Size note: the old MLX-2^20-ceiling rationale is OBSOLETE. The hand-rolled MSL FFT
(shaders/fft.metal) now runs ALL sizes on-GPU (pow2 to 2^30, composite <= 1024, non-smooth via
Bluestein), correctly (differentially verified vs numpy to L2 < 1e-3 in test_fft.py). So this bench
moves up to the large power-of-2 regime — N = 2^23 and 2^24 — which the engine now genuinely
computes on the GPU rather than falling back to CPU.

HONEST-BASELINE DISCIPLINE: we measure the REAL ratio and gate at the honest measured value with
margin. Measured on this machine (M2 Ultra): the GPU WINS — metal/numpy ~0.26 at 2^23 (~3.8x) and
~0.23 at 2^24 (~4.3x). numpy.fft is FFTW-class + multithreaded, but at these large pow2 sizes the
hand-rolled four-step kernel beats it even paying dispatch + host-readback + struct-build overhead.
See the printed ratios and the commit body for the numbers this machine produced.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_metal  # noqa: F401  (registers engine + .metal namespace)
from polars_metal import MetalEngine
from tests.bench.m4_survey._timing import time_callable

# Large power-of-2 sizes the hand-rolled MSL FFT now runs on-GPU.
SIZES = [1 << 23, 1 << 24]

# Honest gate: the GPU genuinely wins at these sizes (measured metal/numpy ~0.23-0.26 on M2 Ultra,
# i.e. ~3.8-4.3x faster). We gate that the GPU beats numpy (ratio < 1.0) with comfortable headroom
# below the real measurement — this guards against a regression that would erase the win without
# asserting a number tighter than the measurement supports.
HONEST_RATIO_BOUND = 0.6


def bench_engine_path(n: int) -> float:
    sig = np.random.default_rng(0xF7).standard_normal(n).astype(np.float32)
    df = pl.DataFrame({"sig": sig}, schema={"sig": pl.Float32})
    expr = pl.col("sig").metal.fft().alias("spec")

    def metal() -> object:
        return df.lazy().with_columns(expr).collect(engine=MetalEngine())

    def numpy_fft() -> object:
        return np.fft.fft(sig)

    metal()  # warmup (MSL/MLX compile + first-dispatch cost)
    metal_res = time_callable(f"metal.fft[N={n:,}]", metal)
    numpy_res = time_callable(f"numpy.fft[N={n:,}]", numpy_fft)
    ratio = metal_res.median_ms / numpy_res.median_ms
    print(
        f"\nFFT engine-path measurement (N={n:,} = 2^{n.bit_length() - 1}, hand-rolled MSL FFT):\n"
        f"  metal.fft median_ms={metal_res.median_ms:.3f}\n"
        f"  numpy.fft median_ms={numpy_res.median_ms:.3f}\n"
        f"  ratio_metal_over_numpy={ratio:.4f} "
        f"({'GPU wins' if ratio < 1.0 else 'numpy wins'})"
    )
    return ratio


@pytest.mark.benchmark(group="fft")
@pytest.mark.parametrize("n", SIZES)
def test_bench_fft_engine_gate(n: int) -> None:
    """Large-pow2 fft, metal engine path measured honestly vs numpy.fft.

    The GPU genuinely wins here (measured ~3.8-4.3x). Gate is the real win (ratio < 1.0) with
    headroom (HONEST_RATIO_BOUND) below the measurement, guarding against regression that would
    erase it — not an aspirational number tighter than the measurement supports.
    """
    ratio = bench_engine_path(n)
    assert ratio < HONEST_RATIO_BOUND, (
        f"fft N={n}: metal/numpy={ratio:.4f} exceeds honest bound {HONEST_RATIO_BOUND}"
    )
