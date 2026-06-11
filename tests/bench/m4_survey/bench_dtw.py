"""End-to-end DTW benchmark (M6 A4): GPU .metal.dtw vs dtaidistance (fair C bar).

Run: python -m tests.bench.m4_survey.bench_dtw
"""

from __future__ import annotations

import numpy as np
import polars as pl
from dtaidistance import dtw as _dtw

import polars_metal
from tests.bench.m4_survey._timing import time_callable


def _frame(Q):
    L = Q.shape[1]
    return pl.DataFrame({"seq": [list(r) for r in Q]}, schema={"seq": pl.Array(pl.Float32, L)})


def main() -> None:
    eng = polars_metal.MetalEngine()
    rng = np.random.default_rng(0xA4)
    for L in (64, 256):
        r32 = rng.standard_normal(L).astype(np.float32)
        r64 = r32.astype(np.float64)
        for N in (10_000, 100_000):
            Q = rng.standard_normal((N, L)).astype(np.float32)
            base = _frame(Q).lazy()
            series = [r64] + [Q[i].astype(np.float64) for i in range(N)]

            # The .metal.dtw expr captures a single-use handle (popped at dispatch),
            # so build it fresh inside the timed callable (mirrors bench_cosine_topk).
            def metal(base=base, r32=r32, eng=eng):
                return base.with_columns(pl.col("seq").metal.dtw(r32).alias("d")).collect(
                    engine=eng
                )

            metal()  # warmup: first call builds the kernel/pipeline.
            cpu = time_callable(
                f"dtaidistance N={N} L={L}",
                lambda series=series, N=N: _dtw.distance_matrix_fast(
                    series, block=((0, 1), (1, N + 1)), compact=True, parallel=True
                ),
            )
            gpu = time_callable(f"metal.dtw  N={N} L={L}", metal)
            print(
                f"  N={N:>7,} L={L:>4}  gpu/cpu={gpu.median_ms / cpu.median_ms:6.3f}  "
                f"(speedup {cpu.median_ms / gpu.median_ms:6.2f}x)"
            )


if __name__ == "__main__":
    main()
