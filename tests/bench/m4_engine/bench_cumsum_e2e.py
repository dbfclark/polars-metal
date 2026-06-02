"""End-to-end cumulative-sum via engine='metal' (M4 Phase 7, Task 28).

`with_columns(cs=pl.col("x").cum_sum())` is an HStack binding, so it reuses the
Phase 6 fused dispatch path (zero-copy I/O). A scan is more bandwidth-bound
than the transcendental chains (it reads + writes the whole column with little
compute per element), so the speedup is lower than haversine / Black-Scholes —
this is the expected shape per the project's bandwidth framing, not a defect.

Survey number (M2 Ultra, 10M F32): cumsum 6.6x vs Polars-native.
"""

from __future__ import annotations

import gc
import statistics
import time

import numpy as np
import polars as pl

import polars_metal


def _make_floats(n: int, seed: int = 0xC57) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"x": rng.standard_normal(n).astype(np.float32)})


def _time(fn, n_iters: int = 8):
    fn()  # warmup
    gc.collect()
    samples = []
    for _ in range(n_iters):
        gc.collect()
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)
    return statistics.median(samples), min(samples), max(samples)


def main() -> None:
    n = 10_000_000
    print(f"Generating {n:,} F32 rows...")
    df = _make_floats(n)

    print("=== Correctness check (100k subset) ===")
    df_small = df.head(100_000)
    cpu_small = df_small.lazy().with_columns(cs=pl.col("x").cum_sum()).collect()
    metal_small = (
        df_small.lazy()
        .with_columns(cs=pl.col("x").cum_sum())
        .collect(engine=polars_metal.MetalEngine())
    )
    max_abs_err = float(np.max(np.abs(cpu_small["cs"].to_numpy() - metal_small["cs"].to_numpy())))
    print(f"  max abs error vs CPU: {max_abs_err:.2e}")
    print()

    def cpu_fn() -> pl.DataFrame:
        return df.lazy().with_columns(cs=pl.col("x").cum_sum()).collect()

    engine = polars_metal.MetalEngine()

    def metal_fn() -> pl.DataFrame:
        return df.lazy().with_columns(cs=pl.col("x").cum_sum()).collect(engine=engine)

    cpu_med, _, _ = _time(cpu_fn)
    metal_med, metal_min, metal_max = _time(metal_fn)

    print("=== Summary ===")
    print(f"  Polars CPU     : {cpu_med:>8.1f} ms")
    print(f"  engine='metal' : {metal_med:>8.1f} ms  (min {metal_min:.1f}, max {metal_max:.1f})")
    print(f"  speedup        : {cpu_med / metal_med:.2f}x")


if __name__ == "__main__":
    main()
