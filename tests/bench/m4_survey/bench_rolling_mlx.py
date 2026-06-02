"""Rolling-window operations via MLX cumsum-diff trick.

In the first pass I declared rolling_mean bandwidth-bound from the
Polars CPU side (~115ms at N=10M, W=any → ~350MB/s). But MLX has a
fast cumsum (5.7ms at N=10M, measured in bench_extra_ops.py) and
rolling_mean can be expressed as cumsum followed by sliding subtraction.

If that works, rolling_mean / rolling_sum become a GPU win after all.
Let's check.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import polars as pl

from tests.bench.m4_survey._timing import time_callable


def main() -> None:
    N = 10_000_000
    rng = np.random.default_rng(0xCAFE)
    arr = rng.standard_normal(N).astype(np.float32)
    df = pl.DataFrame({"x": arr})

    a_mx = mx.array(arr)
    mx.eval(a_mx)

    print(f"\n=== rolling via cumsum-diff trick ===  N={N:,}")
    print()

    for W in (100, 1000, 10_000):
        polars_res = time_callable(
            f"polars.rolling_mean[W={W}]",
            lambda W=W: df.select(pl.col("x").rolling_mean(window_size=W)),
        )

        def mlx_rolling_mean(W=W):
            # cumsum approach: y[i] = (cumsum[i] - cumsum[i-W]) / W
            cs = mx.cumsum(a_mx)
            # shift by W: prepend W zeros, drop last W
            cs_shifted = mx.concatenate([mx.zeros(W, dtype=mx.float32), cs[:-W]])
            out = (cs - cs_shifted) / float(W)
            mx.eval(out)
            return out

        mlx_res = time_callable(f"mlx.rolling_mean_via_cumsum[W={W}]", mlx_rolling_mean)
        print(f"  ratio Polars/MLX = {polars_res.median_ms / mlx_res.median_ms:.2f}x\n")


if __name__ == "__main__":
    main()
