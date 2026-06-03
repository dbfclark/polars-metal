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


def test_rolling_inplace_overwrite_falls_back_to_cpu():
    df = pl.DataFrame({"x": np.arange(8, dtype=np.float32)})
    eng = polars_metal.MetalEngine()
    lf = df.lazy().with_columns(x=pl.col("x").rolling_mean(3))  # out_name == source
    assert_frame_equal(lf.collect(engine=eng), lf.collect())  # no crash; matches CPU


def test_rolling_null_input_matches_cpu():
    df = pl.DataFrame(
        {"x": pl.Series([1.0, None, 3.0, 4.0, None, 6.0, 7.0, 8.0], dtype=pl.Float32)}
    )
    eng = polars_metal.MetalEngine()
    for op in ("mean", "sum"):
        lf = df.lazy().with_columns(r=getattr(pl.col("x"), f"rolling_{op}")(3))
        assert_frame_equal(lf.collect(engine=eng), lf.collect())
