"""End-to-end bare-reduction routing benchmark (B4).

Honest in-engine numbers behind the B4 routing decision: bare sum/min/max/mean
are bandwidth-bound and stay on CPU (the host->MLX ingest alone exceeds Polars'
multithreaded SIMD scan); only std/var clear the dispatch floor and route to
GPU (5-9x). Reports, per case:
  - cpu        : lf.collect()                       (Polars CPU oracle)
  - routed     : lf.collect(engine=metal)           (what the engine actually does)
  - forced_gpu : same, with the gate widened        (informational — the loss avoided)

Run: python -m tests.bench.m4_survey.bench_reductions
"""

from __future__ import annotations

import numpy as np
import polars as pl

import polars_metal
from polars_metal import _native, _walker
from tests.bench.m4_survey._timing import time_callable


def _make(dtype: pl.DataType, n: int, rng: np.random.Generator) -> pl.DataFrame:
    if dtype == pl.Float32:
        a = rng.standard_normal(n).astype(np.float32)
    elif dtype == pl.Int32:
        a = rng.integers(-1_000_000, 1_000_000, size=n, dtype=np.int32)
    elif dtype == pl.Int64:
        a = rng.integers(-1_000_000_000, 1_000_000_000, size=n, dtype=np.int64)
    else:
        raise ValueError(dtype)
    return pl.DataFrame({"x": pl.Series(a, dtype=dtype)})


def _dispatches(lf, eng) -> int:
    n = {"c": 0}
    orig = _native.execute_fused_expr

    def cnt(scope, inputs, out):
        n["c"] += 1
        return orig(scope=scope, inputs=inputs, out=out)

    _native.execute_fused_expr = cnt
    try:
        lf.collect(engine=eng)
    finally:
        _native.execute_fused_expr = orig
    return n["c"]


def main() -> None:
    eng = polars_metal.MetalEngine()
    rng = np.random.default_rng(0xB4)
    bare = [
        (pl.Int32, ("sum", "min", "max")),
        (pl.Int64, ("sum", "min", "max")),
        (pl.Float32, ("sum", "mean", "min", "max")),
    ]
    for n in (10_000_000, 100_000_000):
        print(f"\n=== bare reductions (route to CPU) ===  N={n:,}")
        for dtype, ops in bare:
            df = _make(dtype, n, rng)
            for op in ops:
                lf = df.lazy().select(getattr(pl.col("x"), op)().alias("r"))
                assert _dispatches(lf, eng) == 0, f"bare {op} {dtype} should be CPU"
                cpu = time_callable(f"{dtype}.{op} cpu", lambda lf=lf: lf.collect())
                routed = time_callable(f"{dtype}.{op} routed", lambda lf=lf: lf.collect(engine=eng))
                saved = _walker._BARE_GPU_WORTHY_REDUCTIONS
                _walker._BARE_GPU_WORTHY_REDUCTIONS = frozenset(
                    {"std", "var", "sum", "mean", "min", "max"}
                )
                try:
                    assert _dispatches(lf, eng) == 1, "forced GPU should dispatch"
                    forced = time_callable(
                        f"{dtype}.{op} forced_gpu", lambda lf=lf: lf.collect(engine=eng)
                    )
                finally:
                    _walker._BARE_GPU_WORTHY_REDUCTIONS = saved
                print(
                    f"  {dtype!s:>8}.{op:<5} routed/cpu={routed.median_ms / cpu.median_ms:5.2f}x "
                    f"forced_gpu/cpu={forced.median_ms / cpu.median_ms:5.2f}x"
                )

        print(f"\n=== std/var (route to GPU) ===  N={n:,}")
        df = _make(pl.Float32, n, rng)
        for op in ("std", "var"):
            lf = df.lazy().select(getattr(pl.col("x"), op)().alias("r"))
            assert _dispatches(lf, eng) == 1, f"{op} should dispatch to GPU"
            cpu = time_callable(f"f32.{op} cpu", lambda lf=lf: lf.collect())
            gpu = time_callable(f"f32.{op} gpu", lambda lf=lf: lf.collect(engine=eng))
            print(
                f"  f32.{op:<5} gpu/cpu={gpu.median_ms / cpu.median_ms:5.2f}x "
                f"(speedup {cpu.median_ms / gpu.median_ms:4.1f}x)"
            )


if __name__ == "__main__":
    main()
