"""Rolling-window quantile and mean on F32 time-series.

A more typical Polars workload. Rolling aggregations are everywhere in
panel data / time-series feature engineering, and Polars' CPU implementation
is well-tuned.

For rolling_quantile with window W=500, p50:
  Each output row computes a median over W input rows.
  Polars uses an order-statistic data structure (skip list / partial sort)
  giving O(N log W) total ops for naive, much less with their custom kernel.
  Estimated ops per row: ~log2(500) = 9 comparisons + insertions.
  Memory: read N rows once, write N rows; bytes = N * 4 * 2 = 8 MB at N=1M.
  Density: ~9 ops/byte — borderline.

For rolling_mean with W=500:
  Each output is a sum of W consecutive F32s. With Polars' SIMD-friendly
  ring-buffer kernel, this is O(N) with maybe 4-8 ops per row.
  Density: very low — memory-bandwidth bound.

This is the "moderate compute, moderate memory" case. If Metal can't win
here either, the only winning workloads are the deep-compute ones
(cosine, edit distance).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from tests.bench.m4_survey._timing import time_callable


def make_series(n: int, seed: int = 0xDEAD) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"x": rng.standard_normal(n).astype(np.float32)})


def main() -> None:
    N = 10_000_000
    df = make_series(N)

    print(f"\n=== rolling-window benchmarks ===  N={N:,}")
    print()

    for w in (100, 1000, 10_000):
        time_callable(
            f"rolling_mean[w={w}]",
            lambda w=w: df.select(pl.col("x").rolling_mean(window_size=w)),
            extra={"window": w},
        )

    for w in (100, 1000):
        time_callable(
            f"rolling_quantile_p50[w={w}]",
            lambda w=w: df.select(
                pl.col("x").rolling_quantile(quantile=0.5, window_size=w)
            ),
            extra={"window": w},
        )

    for w in (100, 1000):
        time_callable(
            f"rolling_std[w={w}]",
            lambda w=w: df.select(pl.col("x").rolling_std(window_size=w)),
            extra={"window": w},
        )


if __name__ == "__main__":
    main()
