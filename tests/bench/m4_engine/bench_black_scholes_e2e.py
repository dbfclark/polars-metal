"""End-to-end Black-Scholes-shape pricing via engine='metal'.

Survey numbers (M2 Ultra, 10M F32):
  Polars CPU: 242 ms
  MLX direct: 3.86 ms   (architectural ceiling)
"""

from __future__ import annotations

import gc
import statistics
import time

import numpy as np
import polars as pl

import polars_metal


def _make_options(n: int, seed: int = 0xCAFE) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "s": rng.uniform(50.0, 150.0, size=n).astype(np.float32),
            "k": rng.uniform(50.0, 150.0, size=n).astype(np.float32),
            "t": rng.uniform(0.1, 2.0, size=n).astype(np.float32),
        }
    )


def _bs_expr(sigma: float = 0.2, r: float = 0.05) -> pl.Expr:
    s, k, t = pl.col("s"), pl.col("k"), pl.col("t")
    sigma_sqrt_t = sigma * t.sqrt()
    d1 = ((s / k).log() + (r + 0.5 * sigma * sigma) * t) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    # CDF approx: 0.5 * (1 + tanh(0.7978845608 * x))
    coef = 0.7978845608
    cdf_d1 = 0.5 * (1.0 + (coef * d1).tanh())
    cdf_d2 = 0.5 * (1.0 + (coef * d2).tanh())
    return s * cdf_d1 - k * (-r * t).exp() * cdf_d2


def _time(fn, n_iters: int = 5):
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
    N = 10_000_000
    print(f"Generating {N:,} option-shaped F32 rows...")
    df = _make_options(N)
    expr = _bs_expr()
    print()

    print("=== Correctness check (100k subset) ===")
    df_small = df.head(100_000)
    cpu_small = df_small.lazy().with_columns(call=expr).collect()
    metal_small = df_small.lazy().with_columns(call=expr).collect(engine=polars_metal.MetalEngine())
    cpu_v = cpu_small["call"].to_numpy()
    metal_v = metal_small["call"].to_numpy()
    max_abs_err = float(np.max(np.abs(cpu_v - metal_v)))
    max_rel_err = float(np.max(np.abs((cpu_v - metal_v) / np.maximum(np.abs(cpu_v), 1e-6))))
    print(f"  max abs error vs CPU: {max_abs_err:.2e}")
    print(f"  max rel error vs CPU: {max_rel_err:.2e}")
    print()

    print(f"=== Polars CPU @ N={N:,} (median of 5) ===")

    def cpu_fn() -> pl.DataFrame:
        return df.lazy().with_columns(call=expr).collect()

    cpu_med, cpu_min, cpu_max = _time(cpu_fn)
    print(f"  {cpu_med:.1f} ms  (min {cpu_min:.1f}, max {cpu_max:.1f})")
    print()

    engine = polars_metal.MetalEngine()
    print(f"=== engine='metal' @ N={N:,} (median of 5) ===")

    def metal_fn() -> pl.DataFrame:
        return df.lazy().with_columns(call=expr).collect(engine=engine)

    metal_med, metal_min, metal_max = _time(metal_fn)
    print(f"  {metal_med:.1f} ms  (min {metal_min:.1f}, max {metal_max:.1f})")
    print()

    print("=== Summary ===")
    print(f"  Polars CPU         : {cpu_med:>8.1f} ms")
    print(f"  engine='metal'     : {metal_med:>8.1f} ms")
    print(f"  MLX-direct ceiling : {3.86:>8.1f} ms  (survey number)")
    print()
    ratio = metal_med / cpu_med
    speedup = cpu_med / metal_med
    print(f"  metal/cpu  = {ratio:.2f}x  ({'speedup' if ratio < 1 else 'slowdown'} {speedup:.2f}x)")
    print(f"  metal/ceiling = {metal_med / 3.86:.1f}x  (overhead vs MLX direct)")


if __name__ == "__main__":
    main()
