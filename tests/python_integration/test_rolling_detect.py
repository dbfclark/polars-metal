"""
Task 4 — find_rolling_bindings: detect handleable rolling_* in serialized plan.

Tests cover:
  - all four ops (sum/mean/var/std) detected for F32 with default options
  - rejection of F64 inputs
  - rejection of non-default options (center, min_samples, weights)
  - slow-path: mixed bare (shadows source) + aliased rolling keeps aliased binding
"""

import os
import tempfile

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


def _make_scan_lf(exprs):
    """Return a scan_parquet-backed LazyFrame that exercises the SLOW detect path.

    scan_parquet is not captured in the with_columns cache (it's a file-backed
    source, not a DataFrameScan), so find_rolling_bindings falls through to the
    slow serialize+parse path.  Used by slow-path regression tests."""
    df = pl.DataFrame({"x": pl.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype=pl.Float32)})
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmppath = f.name
    try:
        df.write_parquet(tmppath)
        return pl.scan_parquet(tmppath).with_columns(exprs), tmppath
    except Exception:
        os.unlink(tmppath)
        raise


def test_slow_path_bare_plus_aliased_rolling_keeps_aliased():
    """Regression: bare rolling_mean (out_name shadows source) alongside an aliased
    rolling_mean in the SAME with_columns must NOT poison the aliased binding.

    Old slow path silently skipped bare exprs (only Alias nodes were accepted).
    After the M7 A-2 scaffold migration, iter_candidate_nodes yields bare exprs
    too.  Without the per-binding skip fix the shadowing guard would fire on the
    whole result list and return [], losing the valid 'rm' binding.

    The expected result is identical to old behaviour: only the aliased binding
    is dispatched; the bare one is quietly dropped (not an error)."""
    lf, tmppath = _make_scan_lf(
        [
            pl.col("x").rolling_mean(window_size=3),  # bare — shadows "x"
            pl.col("x").rolling_mean(window_size=3).alias("rm"),  # aliased — valid
        ]
    )
    try:
        result = find_rolling_bindings(lf)
        assert len(result) == 1, f"expected 1 binding, got {result}"
        b = result[0]
        assert b.out_name == "rm"
        assert b.column == "x"
        assert b.op == "mean"
        assert b.window == 3
    finally:
        os.unlink(tmppath)


def test_slow_path_bare_rolling_alone_returns_empty():
    """Slow path: a single bare rolling (shadows source) returns [] — no dispatch."""
    lf, tmppath = _make_scan_lf([pl.col("x").rolling_mean(window_size=3)])
    try:
        assert find_rolling_bindings(lf) == []
    finally:
        os.unlink(tmppath)


def test_slow_path_aliased_rolling_alone_dispatches():
    """Slow path: a single aliased rolling is dispatched normally."""
    lf, tmppath = _make_scan_lf([pl.col("x").rolling_mean(window_size=3).alias("rm")])
    try:
        result = find_rolling_bindings(lf)
        assert len(result) == 1
        assert result[0].out_name == "rm"
    finally:
        os.unlink(tmppath)
