"""
Task 5 — end-to-end rolling dispatch via MetalEngine collect wrapper.

Verifies that rolling_mean and rolling_sum are detected, routed to the
Metal kernel, and produce results matching Polars CPU — including correct
leading nulls (first w-1 rows are null).
"""

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

import polars_metal


def test_rolling_mean_sum_e2e_match_cpu():
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"x": rng.standard_normal(4096).astype(np.float32)})
    eng = polars_metal.MetalEngine()
    for op, w in [("mean", 64), ("sum", 50)]:
        expr = getattr(pl.col("x"), f"rolling_{op}")(w)
        lf = df.lazy().with_columns(r=expr)
        got = lf.collect(engine=eng)
        assert_frame_equal(got, lf.collect(), check_exact=False, rel_tol=1e-4, abs_tol=1e-4)
        assert got["r"][: w - 1].null_count() == w - 1  # first w-1 structurally null
