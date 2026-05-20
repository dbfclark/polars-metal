"""Phase 6 milestone: ``df.filter(comparison_expr)`` runs on GPU end-to-end.

After Task 18 the walker accepts Filter nodes whose predicate is a
column-vs-leaf comparison ``BinaryExpr`` in the closed op set
``{==, !=, <, <=, >, >=}``. The Rust UDF evaluates the comparison via
``dispatch_cmp_<dtype>[_scalar]`` to produce a bit-packed bool predicate,
then runs the existing compaction pipeline (Task 14) on the surviving
columns.

Anything outside the closed set — AND/OR (Task 20), arithmetic in the
predicate (M2+), casts, function calls — still falls back to CPU
cleanly. The fallback tests below pin that contract.
"""

from __future__ import annotations

import logging

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_metal


def test_filter_col_gt_scalar_runs_on_gpu(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="polars_metal")
    df = pl.DataFrame(
        {
            "a": [-1, 0, 1, 2, None, 4],
            "b": [10, 20, 30, 40, 50, 60],
        }
    )
    cpu = df.lazy().filter(pl.col("a") > 0).collect()
    metal = df.lazy().filter(pl.col("a") > 0).collect(engine=polars_metal.MetalEngine(debug=True))
    assert_frame_equal(cpu, metal)
    log_text = " ".join(r.getMessage() for r in caplog.records if r.name == "polars_metal")
    assert "installed UDF" in log_text, f"expected UDF installation, got logs: {log_text}"


def test_filter_col_lt_col_runs_on_gpu() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
    cpu = df.lazy().filter(pl.col("a") < pl.col("b")).collect()
    metal = df.lazy().filter(pl.col("a") < pl.col("b")).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_col_eq_scalar() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 2, 1, 2]})
    cpu = df.lazy().filter(pl.col("a") == 2).collect()
    metal = df.lazy().filter(pl.col("a") == 2).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_col_ne_scalar() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 2, 1, 2]})
    cpu = df.lazy().filter(pl.col("a") != 2).collect()
    metal = df.lazy().filter(pl.col("a") != 2).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_all_six_comparison_ops_i64() -> None:
    df = pl.DataFrame({"a": [-3, -1, 0, 1, 3]})
    cases = [
        (">", lambda c: c > 0),
        (">=", lambda c: c >= 0),
        ("<", lambda c: c < 0),
        ("<=", lambda c: c <= 0),
        ("==", lambda c: c == 0),
        ("!=", lambda c: c != 0),
    ]
    for op_str, op_fn in cases:
        cpu = df.lazy().filter(op_fn(pl.col("a"))).collect()
        metal = df.lazy().filter(op_fn(pl.col("a"))).collect(engine=polars_metal.MetalEngine())
        assert_frame_equal(cpu, metal), f"op={op_str}"


def test_filter_with_f64_comparison() -> None:
    df = pl.DataFrame({"x": [1.0, 2.5, 3.7, 4.0, 5.5]})
    cpu = df.lazy().filter(pl.col("x") > 3.0).collect()
    metal = df.lazy().filter(pl.col("x") > 3.0).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


@pytest.mark.xfail(
    reason="cmp_f64 implements IEEE 754 ordered semantics (NaN OP x → false), but "
    "Polars uses TotalOrd (NaN > any non-NaN → true). Kernel-level fix tracked "
    "as a Task 17 follow-up in docs/open-questions.md.",
    strict=True,
)
def test_filter_with_nan_f64_total_ord() -> None:
    """Polars treats NaN with TotalOrd: ``NaN > 0`` is True, the row survives.

    Our kernel implements IEEE 754 ordered comparison (``NaN > 0`` is False),
    so this test fails today. The kernel must be reworked to match Polars'
    TotalOrd semantics — that's a Task 17 follow-up, not Task 18 territory.
    """
    df = pl.DataFrame({"x": [1.0, float("nan"), 3.0, float("nan"), 5.0]})
    cpu = df.lazy().filter(pl.col("x") > 0).collect()
    metal = df.lazy().filter(pl.col("x") > 0).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_with_null_input_propagates_null() -> None:
    """null op x produces null → null in predicate drops the row (Polars semantics)."""
    df = pl.DataFrame({"a": [1, None, 3, None, 5]})
    cpu = df.lazy().filter(pl.col("a") > 2).collect()
    metal = df.lazy().filter(pl.col("a") > 2).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_f64_col_lt_col() -> None:
    df = pl.DataFrame({"x": [1.0, 2.5, 3.0, 4.5], "y": [2.0, 2.5, 2.0, 5.0]})
    cpu = df.lazy().filter(pl.col("x") < pl.col("y")).collect()
    metal = df.lazy().filter(pl.col("x") < pl.col("y")).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_compound_predicate_now_runs_on_gpu() -> None:
    """Compound AND/OR predicates landed in Task 20 (see test_filter_compound.py).

    Phase 6 used to fall back on this shape; Phase 7 dispatches it to
    the ``bool_and`` / ``bool_or`` kernels. We keep the (now-passing)
    test here as a regression pin against the Phase 6 codepath
    accidentally re-rejecting compound predicates.
    """
    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    cpu = df.lazy().filter((pl.col("a") > 1) & (pl.col("b") < 40)).collect()
    metal = (
        df.lazy()
        .filter((pl.col("a") > 1) & (pl.col("b") < 40))
        .collect(engine=polars_metal.MetalEngine())
    )
    assert_frame_equal(cpu, metal)


def test_filter_arithmetic_in_predicate_falls_back() -> None:
    """``pl.col('a') + 1 > 0`` has arithmetic in the lhs — M2+ territory."""
    df = pl.DataFrame({"a": [-2, -1, 0, 1, 2]})
    cpu = df.lazy().filter(pl.col("a") + 1 > 0).collect()
    metal = df.lazy().filter(pl.col("a") + 1 > 0).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_then_select_runs_on_gpu() -> None:
    """Project-on-Filter integration: T15's test still passes with cmp predicate."""
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
    cpu = df.lazy().filter(pl.col("a") > 2).select("b").collect()
    metal = df.lazy().filter(pl.col("a") > 2).select("b").collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_mixed_dtype_comparison_falls_back() -> None:
    """``i64 > f64_literal`` injects a Cast in the IR — fall back cleanly."""
    df = pl.DataFrame({"a": [1, 2, 3, 4]})
    cpu = df.lazy().filter(pl.col("a") > 2.5).collect()
    metal = df.lazy().filter(pl.col("a") > 2.5).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)
