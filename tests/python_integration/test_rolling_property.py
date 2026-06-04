"""
Task 8 — Correctness gate: randomized differential test comparing engine="metal"
against Polars CPU across many random shapes and all four rolling ops.

The Metal path uses a tile-blocked kernel (TG_SIZE=256, so windows near/over 256
stress the tile-boundary path) for null-free F32 columns; everything else falls
back to CPU (also correct). assert_frame_equal uses rel_tol/abs_tol (not rtol/atol).
"""

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

import polars_metal


def test_rolling_matches_cpu_random():
    eng = polars_metal.MetalEngine()
    rng = np.random.default_rng(7)
    for _ in range(50):
        n = int(rng.integers(1, 5000))
        w = int(rng.integers(1, max(2, min(n, 512))))
        x = rng.standard_normal(n).astype(np.float32)
        df = pl.DataFrame({"x": x})
        for op in ("mean", "sum", "var", "std"):
            lf = df.lazy().with_columns(r=getattr(pl.col("x"), f"rolling_{op}")(w))
            assert_frame_equal(
                lf.collect(engine=eng),
                lf.collect(),
                check_exact=False,
                rel_tol=1e-3,
                abs_tol=1e-4,
            )


def test_rolling_boundary_windows():
    # Windows around the TG_SIZE=256 tile boundary, plus tiny and full windows.
    eng = polars_metal.MetalEngine()
    rng = np.random.default_rng(11)
    for n in (1, 2, 255, 256, 257, 512, 513, 1000):
        x = rng.standard_normal(n).astype(np.float32)
        df = pl.DataFrame({"x": x})
        for w in {1, 2, 255, 256, 257, n}:
            if w < 1 or w > n:
                continue
            for op in ("mean", "sum", "var", "std"):
                if op in ("var", "std") and w < 2:
                    continue  # var/std need w>=2 (ddof=1); w=1 falls back to CPU anyway
                lf = df.lazy().with_columns(r=getattr(pl.col("x"), f"rolling_{op}")(w))
                assert_frame_equal(
                    lf.collect(engine=eng),
                    lf.collect(),
                    check_exact=False,
                    rel_tol=1e-3,
                    abs_tol=1e-4,
                )
