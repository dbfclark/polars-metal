"""Phase 7 milestone: compound predicates with AND/OR run on GPU end-to-end.

After Task 20 the walker accepts ``BinaryExpr(And|Or, lhs_pred, rhs_pred)``
where both sides are predicate expressions from the M1 closed set (Column,
Literal, Compare, And, Or — arbitrarily nested). The Rust UDF dispatches
``bool_and`` / ``bool_or`` kernels to combine sub-predicates.

The truth-table cases (``test_filter_or_with_null_3valued`` and
``test_filter_and_with_null_3valued``) pin Polars' 3-valued logic against
the kernel's Kleene-style semantics: true dominates OR, false dominates
AND, otherwise null propagates.
"""

from __future__ import annotations

import logging

import polars as pl
from polars.testing import assert_frame_equal

import polars_metal


def test_filter_compound_and_runs_on_gpu(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="polars_metal")
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, None],
            "b": [10, 20, 30, 40, 50, 60],
            "c": [50, 40, 30, 20, 10, 5],
        }
    )
    cpu = df.lazy().filter((pl.col("a") > 0) & (pl.col("b") < pl.col("c"))).collect()
    metal = (
        df.lazy()
        .filter((pl.col("a") > 0) & (pl.col("b") < pl.col("c")))
        .collect(engine=polars_metal.MetalEngine(debug=True))
    )
    assert_frame_equal(cpu, metal)
    log_text = " ".join(r.getMessage() for r in caplog.records if r.name == "polars_metal")
    # M2 cost model: filter→CPU always. No UDF is installed; the router logs the CPU route.
    assert "router routes entire query to CPU" in log_text, (
        f"expected router-to-CPU log, got: {log_text}"
    )


def test_filter_compound_or_runs_on_gpu() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 8, 6, 4, 2]})
    cpu = df.lazy().filter((pl.col("a") > 4) | (pl.col("b") < 5)).collect()
    metal = (
        df.lazy()
        .filter((pl.col("a") > 4) | (pl.col("b") < 5))
        .collect(engine=polars_metal.MetalEngine())
    )
    assert_frame_equal(cpu, metal)


def test_filter_three_way_and() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
            "c": [1, 1, 1, 1, 1],
        }
    )
    cpu = df.lazy().filter((pl.col("a") > 1) & (pl.col("b") > 1) & (pl.col("c") == 1)).collect()
    metal = (
        df.lazy()
        .filter((pl.col("a") > 1) & (pl.col("b") > 1) & (pl.col("c") == 1))
        .collect(engine=polars_metal.MetalEngine())
    )
    assert_frame_equal(cpu, metal)


def test_filter_mixed_and_or() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
    cpu = df.lazy().filter((pl.col("a") > 4) | ((pl.col("a") < 3) & (pl.col("b") > 3))).collect()
    metal = (
        df.lazy()
        .filter((pl.col("a") > 4) | ((pl.col("a") < 3) & (pl.col("b") > 3)))
        .collect(engine=polars_metal.MetalEngine())
    )
    assert_frame_equal(cpu, metal)


def test_filter_or_with_null_3valued() -> None:
    """Polars OR is 3-valued: True | null = True, null | False = null."""
    df = pl.DataFrame({"a": [True, False, None], "b": [None, False, True]})
    cpu = df.lazy().filter(pl.col("a") | pl.col("b")).collect()
    metal = df.lazy().filter(pl.col("a") | pl.col("b")).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_and_with_null_3valued() -> None:
    """Polars AND is 3-valued: False & null = False, null & True = null."""
    df = pl.DataFrame({"a": [True, False, None], "b": [None, True, True]})
    cpu = df.lazy().filter(pl.col("a") & pl.col("b")).collect()
    metal = df.lazy().filter(pl.col("a") & pl.col("b")).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_compound_with_f64() -> None:
    df = pl.DataFrame({"x": [1.0, 2.5, 3.7, 4.0, 5.5], "y": [10.0, 8.0, 6.0, 4.0, 2.0]})
    cpu = df.lazy().filter((pl.col("x") > 2.0) & (pl.col("y") < 9.0)).collect()
    metal = (
        df.lazy()
        .filter((pl.col("x") > 2.0) & (pl.col("y") < 9.0))
        .collect(engine=polars_metal.MetalEngine())
    )
    assert_frame_equal(cpu, metal)


def test_filter_select_after_compound() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
    cpu = df.lazy().filter((pl.col("a") > 1) & (pl.col("a") < 5)).select("b").collect()
    metal = (
        df.lazy()
        .filter((pl.col("a") > 1) & (pl.col("a") < 5))
        .select("b")
        .collect(engine=polars_metal.MetalEngine())
    )
    assert_frame_equal(cpu, metal)


def test_filter_not_function_falls_back() -> None:
    """``pl.col(x).not_()`` lowers to a Function node, not a BinaryExpr — fall back cleanly to CPU.

    (``~(col > x)`` is rewritten by Polars into the flipped comparison
    ``col <= x`` at IR construction time, so it stays on GPU. The only
    way to land a true NOT in the predicate is the ``.not_()`` form on a
    Boolean column.)
    """
    df = pl.DataFrame({"a": [True, False, True, False]})
    cpu = df.lazy().filter(pl.col("a").not_()).collect()
    metal = df.lazy().filter(pl.col("a").not_()).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)
