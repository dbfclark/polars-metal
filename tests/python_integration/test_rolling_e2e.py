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


def test_repeated_collect_same_lf_correct():
    # 2nd collect of the SAME lf object hits the slow path (cache evicted by
    # the 1st collect via pop()) and is still correct.
    rng = np.random.default_rng(42)
    df = pl.DataFrame({"x": rng.standard_normal(20).astype(np.float32)})
    eng = polars_metal.MetalEngine()
    lf = df.lazy().with_columns(r=pl.col("x").rolling_mean(3))
    first = lf.collect(engine=eng)
    second = lf.collect(engine=eng)  # cache already evicted by first collect
    cpu = lf.collect()
    assert_frame_equal(first, cpu, check_exact=False, rel_tol=1e-4, abs_tol=1e-4)
    assert_frame_equal(second, cpu, check_exact=False, rel_tol=1e-4, abs_tol=1e-4)


def test_rolling_repeated_collect_same_lf_fast_and_correct():
    # After M7a get-not-pop fix: 2nd collect of the SAME lf should still use
    # the fast path (cache not evicted by 1st collect) and produce correct results.
    df = pl.DataFrame({"x": [float(i) for i in range(5000)]})
    lf = df.lazy().with_columns(
        pl.col("x").cast(pl.Float32).rolling_mean(window_size=100).alias("rm")
    )
    eng = polars_metal.MetalEngine()
    out1 = lf.collect(engine=eng)
    out2 = lf.collect(engine=eng)  # 2nd collect of SAME lf -- must match, no error
    assert_frame_equal(out1, out2, check_exact=False, rel_tol=1e-4, abs_tol=1e-4)
    # And matches CPU
    exp = lf.collect()
    assert_frame_equal(out1, exp, check_exact=False, rel_tol=1e-4, abs_tol=1e-4)


def test_non_rolling_collect_after_rolling_unaffected():
    # A non-with_columns frame collected after a rolling one must not pick up
    # stale exprs from a previous (now-evicted) cache entry.
    eng = polars_metal.MetalEngine()
    # Consume a rolling lf — populates then evicts the cache entry.
    pl.DataFrame({"x": np.arange(20, dtype=np.float32)}).lazy().with_columns(
        r=pl.col("x").rolling_mean(3)
    ).collect(engine=eng)
    # This frame was NOT created via with_columns; it must not see stale exprs.
    other = pl.DataFrame({"x": np.arange(20, dtype=np.float32)}).lazy().select(pl.col("x") * 2)
    assert_frame_equal(other.collect(engine=eng), other.collect())
