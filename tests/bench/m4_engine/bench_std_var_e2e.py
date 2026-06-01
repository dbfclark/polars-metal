"""End-to-end std/var reductions via engine='metal' (M4 Phase 7, Task 26).

`select(pl.col("x").std(), pl.col("x").var())` over a bare Float32 column routes
each reduction through the fused MLX path (population variance), then applies a
Bessel correction at the dispatch boundary to match Polars' sample default
(ddof=1). The speedup is modest: std and var dispatch separately, and Polars CPU
std/var is already well-optimized — but it's a correct, real win.
"""

from __future__ import annotations

import gc
import statistics
import time

import numpy as np
import polars as pl

import polars_metal


def _make_floats(n: int, seed: int = 0xFA57) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"x": rng.standard_normal(n).astype(np.float32)})


def _exprs() -> list[pl.Expr]:
    return [pl.col("x").std().alias("std"), pl.col("x").var().alias("var")]


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
    engine = polars_metal.MetalEngine()

    cpu = df.lazy().select(_exprs()).collect()
    metal = df.lazy().select(_exprs()).collect(engine=engine)
    print("=== Correctness ===")
    for col in ("std", "var"):
        print(f"  {col}: cpu={cpu[col][0]:.6f}  metal={metal[col][0]:.6f}")
    print()

    cpu_med, _, _ = _time(lambda: df.lazy().select(_exprs()).collect())
    metal_med, metal_min, metal_max = _time(
        lambda: df.lazy().select(_exprs()).collect(engine=engine)
    )

    print("=== Summary ===")
    print(f"  Polars CPU     : {cpu_med:>8.1f} ms")
    print(f"  engine='metal' : {metal_med:>8.1f} ms  (min {metal_min:.1f}, max {metal_max:.1f})")
    print(f"  speedup        : {cpu_med / metal_med:.2f}x")


if __name__ == "__main__":
    main()
