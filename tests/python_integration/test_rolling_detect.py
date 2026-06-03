"""
Task 4 — find_rolling_bindings: detect handleable rolling_* in serialized plan.

Tests cover:
  - all four ops (sum/mean/var/std) detected for F32 with default options
  - rejection of F64 inputs
  - rejection of non-default options (center, min_samples, weights)
"""

import polars as pl

from polars_metal._rolling_detect import find_rolling_bindings


def test_detects_rolling_mean_f32_default():
    lf = (
        pl.DataFrame({"x": pl.Series([1.0, 2, 3], dtype=pl.Float32)})
        .lazy()
        .with_columns(r=pl.col("x").rolling_mean(3))
    )
    found = find_rolling_bindings(lf)
    assert len(found) == 1
    b = found[0]
    assert (b.op, b.column, b.window, b.out_name) == ("mean", "x", 3, "r")


def test_detects_all_four_ops_f32():
    f32 = pl.DataFrame({"x": pl.Series([1.0, 2, 3, 4, 5], dtype=pl.Float32)}).lazy()
    for fn, op in [
        ("rolling_sum", "sum"),
        ("rolling_mean", "mean"),
        ("rolling_var", "var"),
        ("rolling_std", "std"),
    ]:
        lf = f32.with_columns(r=getattr(pl.col("x"), fn)(3))
        found = find_rolling_bindings(lf)
        assert len(found) == 1 and found[0].op == op and found[0].window == 3


def test_rejects_non_f32_and_options():
    f64 = pl.DataFrame({"x": pl.Series([1.0, 2, 3], dtype=pl.Float64)}).lazy()
    assert find_rolling_bindings(f64.with_columns(r=pl.col("x").rolling_mean(3))) == []  # F64

    f32 = pl.DataFrame({"x": pl.Series([1.0, 2, 3], dtype=pl.Float32)}).lazy()
    assert find_rolling_bindings(f32.with_columns(r=pl.col("x").rolling_mean(3, center=True))) == []
    assert (
        find_rolling_bindings(f32.with_columns(r=pl.col("x").rolling_mean(3, min_samples=1))) == []
    )
    assert (
        find_rolling_bindings(
            f32.with_columns(r=pl.col("x").rolling_mean(3, weights=[1.0, 2.0, 3.0]))
        )
        == []
    )
