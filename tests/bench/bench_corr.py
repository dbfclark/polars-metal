"""Bench: GPU correlation matrix vs Polars CPU df.corr().

The spike (scripts/spike_corr_crossover.py) saw ~5-20x at p>=10 for N>=100K
(up to ~20x at N=1M p=50), using raw MLX against single-threaded numpy. This
records the HONEST engine-path number: full collect() overhead (serialize-detect
+ dispatch + MLX ingest + readback + DataFrame build) vs Polars' native
multithreaded df.corr().

HONEST-BASELINE DISCIPLINE (matching the repo's M6 pattern): we measure the
real engine ratio and gate at the honest measured value with margin. If the GPU
does not win at N=1M p=50 (speedup <= 1x) that is a regression relative to the
spike — the task plan explicitly flags this as a DONE_WITH_CONCERNS case, not
something to silently accept.

Run standalone: python tests/bench/bench_corr.py
Run gate only:  pytest tests/bench/bench_corr.py::test_bench_corr_engine_gate
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_metal as pm
from tests.bench.m4_survey._timing import time_callable

# The canonical gate shape: large N, moderate-high p.  After the (p,n) zero-copy
# ingest optimization the honest engine number is ~9.9x here (ratio ~0.10); the
# residual gap to the spike's raw-MLX ~20x is the host->Metal staging copy.  We
# gate at a conservative 0.2 (GPU must be >5x faster) — well clear of the ~0.10
# measurement, so it catches an ingest/dispatch-cliff regression (e.g. a return
# to the df.to_numpy()+transpose path, which dropped this to ~2x) without
# asserting the exact number.
GATE_N = 1_000_000
GATE_P = 50
HONEST_RATIO_BOUND = 0.2  # metal_ms / cpu_ms must be < this to pass


def _make_df(n: int, p: int, *, seed: int = 0xC1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p)).astype(np.float32)
    return pl.DataFrame(x, schema=[f"c{i}" for i in range(p)])


def bench_corr_sweep() -> None:
    """Print the full N x p sweep table (standalone, informational)."""
    eng = pm.MetalEngine()
    print(f"\n{'N':>9} {'p':>4} {'cpu_ms':>9} {'gpu_ms':>9} {'speedup':>8}")
    for n in (100_000, 1_000_000):
        for p in (10, 25, 50):
            df = _make_df(n, p)
            lf = df.lazy()
            it = 5 if n <= 100_000 else 3
            # warmup
            df.corr()
            lf.metal.corr(force_gpu=True).collect(engine=eng)

            cpu = time_callable(
                f"cpu.df.corr[N={n:,} p={p}]",
                lambda df=df: df.corr(),
                n_warmup=1,
                n_measure=it,
            )
            gpu = time_callable(
                f"metal.corr[N={n:,} p={p}]",
                lambda lf=lf, eng=eng: lf.metal.corr(force_gpu=True).collect(engine=eng),
                n_warmup=1,
                n_measure=it,
            )
            speedup = cpu.median_ms / gpu.median_ms
            print(f"{n:>9,} {p:>4} {cpu.median_ms:>9.3f} {gpu.median_ms:>9.3f} {speedup:>7.2f}x")
    print()


def bench_engine_gate(n: int = GATE_N, p: int = GATE_P) -> float:
    """Engine-path gate: metal corr matrix must beat Polars CPU df.corr().

    Returns ratio_metal_over_cpu (metal_ms / cpu_ms). The gate asserts < HONEST_RATIO_BOUND
    (i.e. GPU must be >2x faster at the canonical N=1M p=50 shape).
    """
    df = _make_df(n, p)
    lf = df.lazy()
    eng = pm.MetalEngine()

    # warmup: first collect builds the MLX pipeline + MSL kernels
    df.corr()
    lf.metal.corr(force_gpu=True).collect(engine=eng)

    it = 3
    cpu_res = time_callable(
        f"cpu.df.corr[N={n:,} p={p}]",
        lambda: df.corr(),
        n_warmup=1,
        n_measure=it,
    )
    gpu_res = time_callable(
        f"metal.corr[N={n:,} p={p}]",
        lambda: lf.metal.corr(force_gpu=True).collect(engine=eng),
        n_warmup=1,
        n_measure=it,
    )
    ratio = gpu_res.median_ms / cpu_res.median_ms
    speedup = cpu_res.median_ms / gpu_res.median_ms
    print(
        f"\n=== engine gate (baseline.json: phase8_corr_matrix_200x200k proxy) ===\n"
        f"  N={n:,}  p={p}\n"
        f"  cpu_ms={cpu_res.median_ms:.2f}  gpu_ms={gpu_res.median_ms:.2f}\n"
        f"  ratio_metal_over_cpu={ratio:.4f}  speedup={speedup:.1f}x\n"
        f"  gate: ratio < {HONEST_RATIO_BOUND}  "
        f"({'PASS' if ratio < HONEST_RATIO_BOUND else 'FAIL'})\n"
    )
    return ratio


@pytest.mark.benchmark(group="corr")
def test_bench_corr_engine_gate() -> None:
    """Engine-path gate: metal corr matrix must beat Polars CPU df.corr() by >2x.

    Shape: N=1M rows x p=50 F32 columns (the spike's canonical case).
    Gate is conservative (>2x) vs the spike's raw-MLX ~20x ceiling — the engine
    path adds serialize-detect + dispatch + ingest + readback overhead. If this
    gate fails, the dispatch path has regressed significantly.
    """
    ratio = bench_engine_gate()
    assert ratio < HONEST_RATIO_BOUND, (
        f"corr gate: metal/cpu={ratio:.4f} not < {HONEST_RATIO_BOUND} "
        f"(GPU must be >{1 / HONEST_RATIO_BOUND:.0f}x faster at N={GATE_N:,} p={GATE_P})"
    )


def main() -> None:
    print("=== GPU correlation matrix benchmark (engine path vs df.corr()) ===")
    print("Honest engine-path numbers; not raw MLX.")
    bench_corr_sweep()
    bench_engine_gate()


if __name__ == "__main__":
    main()
