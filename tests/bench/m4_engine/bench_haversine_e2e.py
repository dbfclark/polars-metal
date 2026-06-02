"""End-to-end haversine via engine='metal'.

Survey numbers (M2 Ultra, 10M F32):
  Polars CPU: 181 ms
  MLX direct: 3.49 ms   (architectural ceiling, not engine-bound)

This bench measures the current engine path including:
  - Polars Series -> numpy.astype(F32) -> bytes (copy)
  - PyBytes -> Rust &[u8] -> &[f32] (reinterpret, no copy)
  - MetalBuffer::from_f32_slice (Metal allocation + copy)
  - mlx_array_view_metal_buffer (zero-copy view)
  - eval + readback as Vec<f32> (copy)
  - PyBytes -> numpy.frombuffer -> Series

Three of those steps each copy 40MB of F32 per column at 10M rows. Until
input zero-copy direct from the Polars Arrow buffer + output zero-copy
via the MLX allocator FFI land, the engine path carries 2-3x overhead
versus the MLX-direct ceiling. We're still expected to beat Polars CPU
on this workload by a healthy margin.
"""

from __future__ import annotations

import gc
import statistics
import time

import numpy as np
import polars as pl

import polars_metal


def _make_taxi(n: int, seed: int = 0xCAB) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "pickup_lat": rng.uniform(40.6, 40.9, size=n).astype(np.float32),
            "pickup_lon": rng.uniform(-74.05, -73.7, size=n).astype(np.float32),
            "drop_lat": rng.uniform(40.6, 40.9, size=n).astype(np.float32),
            "drop_lon": rng.uniform(-74.05, -73.7, size=n).astype(np.float32),
        }
    )


def _haversine_expr() -> pl.Expr:
    R = 6371.0
    d2r = float(np.pi / 180.0)
    pla = pl.col("pickup_lat") * d2r
    dla = pl.col("drop_lat") * d2r
    dlat = (dla - pla) / 2.0
    dlon = (pl.col("drop_lon") - pl.col("pickup_lon")) * d2r / 2.0
    a = dlat.sin() ** 2 + pla.cos() * dla.cos() * dlon.sin() ** 2
    return 2.0 * R * a.sqrt().arcsin()


def _time(fn, n_iters: int = 5) -> tuple[float, float, float]:
    """Returns (median_ms, min_ms, max_ms)."""
    # Warmup
    fn()
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
    print(f"Generating {N:,} taxi-shaped F32 rows...")
    df = _make_taxi(N)
    expr = _haversine_expr()
    print(f"  schema: {df.schema}")
    print(f"  bytes (4 F32 cols * {N:,} rows * 4): {4 * N * 4 / 1e6:.1f} MB")
    print()

    # Correctness check on a small subset (full 10M comparison is slow on CPU)
    print("=== Correctness check (100k subset) ===")
    df_small = df.head(100_000)
    cpu_small = df_small.lazy().with_columns(d=expr).collect()
    metal_small = df_small.lazy().with_columns(d=expr).collect(engine=polars_metal.MetalEngine())
    cpu_d = cpu_small["d"].to_numpy()
    metal_d = metal_small["d"].to_numpy()
    max_abs_err = float(np.max(np.abs(cpu_d - metal_d)))
    print(f"  max abs error vs CPU: {max_abs_err:.2e}")
    print()

    print(f"=== Polars CPU @ N={N:,} (median of 5) ===")

    def cpu_fn() -> pl.DataFrame:
        return df.lazy().with_columns(d=expr).collect()

    cpu_med, cpu_min, cpu_max = _time(cpu_fn)
    print(f"  {cpu_med:.1f} ms  (min {cpu_min:.1f}, max {cpu_max:.1f})")
    print()

    engine = polars_metal.MetalEngine()
    print(f"=== engine='metal' @ N={N:,} (median of 5) ===")

    def metal_fn() -> pl.DataFrame:
        return df.lazy().with_columns(d=expr).collect(engine=engine)

    metal_med, metal_min, metal_max = _time(metal_fn)
    print(f"  {metal_med:.1f} ms  (min {metal_min:.1f}, max {metal_max:.1f})")
    print()

    print("=== Summary ===")
    print(f"  Polars CPU         : {cpu_med:>8.1f} ms")
    print(f"  engine='metal'     : {metal_med:>8.1f} ms")
    print(f"  MLX-direct ceiling : {3.49:>8.1f} ms  (survey number)")
    print()
    ratio = metal_med / cpu_med
    speedup = cpu_med / metal_med
    print(f"  metal/cpu  = {ratio:.2f}x  ({'speedup' if ratio < 1 else 'slowdown'} {speedup:.2f}x)")
    print(f"  metal/ceiling = {metal_med / 3.49:.1f}x  (overhead vs MLX direct)")


if __name__ == "__main__":
    main()
