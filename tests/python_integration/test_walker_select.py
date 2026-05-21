"""Tests for the bottom-up IR walker (Task 7, M1 Phase 4).

In this phase, the walker recognizes:
- DataFrameScan (with optional .projection column-name list)
- SimpleProjection / Select (column re-selection only)

Filter still falls back to CPU; arbitrary expressions fall back.

Each test asserts the engine="metal" result is identical to the CPU result.
Additional tests assert (via debug-log capture) that the walker actually
fires for the supported cases — without this we couldn't distinguish a
correct UDF dispatch from a silent CPU fallback.
"""

from __future__ import annotations

import logging

import polars as pl
from polars.testing import assert_frame_equal

import polars_metal


def test_select_only_query_returns_correct_result() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
    cpu = df.lazy().select(["b", "a"]).collect()
    metal = df.lazy().select(["b", "a"]).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_select_with_unsupported_string_dtype_falls_back_cleanly() -> None:
    # String dtype is not in our M1 closed dtype set; the walker must FallBack
    # cleanly (no exception, no wrong result).
    df = pl.DataFrame({"a": [1, 2, 3], "s": ["x", "y", "z"]})
    cpu = df.lazy().select(["s", "a"]).collect()
    metal = df.lazy().select(["s", "a"]).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_select_reorders_columns() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": [5, 6]})
    cpu = df.lazy().select(["c", "a", "b"]).collect()
    metal = df.lazy().select(["c", "a", "b"]).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_scan_only_query_returns_correct_result() -> None:
    # No projection — just a bare scan/collect.
    df = pl.DataFrame({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]})
    cpu = df.lazy().collect()
    metal = df.lazy().collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_still_falls_back_correctly() -> None:
    # Filter is not implemented in Phase 4. The walker must FallBack and
    # Polars CPU produces the result — same as M0.
    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [10.0, 20.0, 30.0, 40.0]})
    cpu = df.lazy().filter(pl.col("a") > 2).collect()
    metal = df.lazy().filter(pl.col("a") > 2).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_select_with_bool_column() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "flag": [True, False, True]})
    cpu = df.lazy().select(["flag", "a"]).collect()
    metal = df.lazy().select(["flag", "a"]).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_walker_actually_installs_udf_for_supported_query(caplog) -> None:
    """Proves the walker fires (rather than silently falling back) when the
    plan is in its closed set. Without this, a regression that broke the
    walker but kept results correct via CPU fallback would slip through."""
    caplog.set_level(logging.DEBUG, logger="polars_metal")
    df = pl.DataFrame({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]})
    _ = df.lazy().select(["b", "a"]).collect(engine=polars_metal.MetalEngine(debug=True))
    msgs = [r.getMessage() for r in caplog.records if r.name == "polars_metal"]
    assert any("installed UDF for plan kind=Scan" in m for m in msgs), msgs


def test_walker_falls_back_for_unsupported_dtype(caplog) -> None:
    """Mirror of the previous test: with String dtype we must FallBack —
    no UDF install line in the log."""
    caplog.set_level(logging.DEBUG, logger="polars_metal")
    df = pl.DataFrame({"a": [1, 2, 3], "s": ["x", "y", "z"]})
    _ = df.lazy().select(["s", "a"]).collect(engine=polars_metal.MetalEngine(debug=True))
    msgs = [r.getMessage() for r in caplog.records if r.name == "polars_metal"]
    assert any("falling back" in m for m in msgs), msgs
    assert not any("installed UDF" in m for m in msgs), msgs


def test_walker_falls_back_for_filter_with_unsupported_predicate(caplog) -> None:
    """A Filter predicate outside the M1 closed set must FallBack.

    As of Task 18 (Phase 6) the walker accepts comparison ``BinaryExpr``
    predicates on i64/f64 columns. So we exercise the unsupported shape:
    arithmetic in the predicate (``pl.col('a') + 1 > 0``), which Polars
    encodes as ``BinaryExpr(Gt, BinaryExpr(Plus, ...), Literal)``. The
    inner arithmetic BinaryExpr fails the comparison-op check and the
    whole predicate falls back.
    """
    caplog.set_level(logging.DEBUG, logger="polars_metal")
    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [10.0, 20.0, 30.0, 40.0]})
    _ = df.lazy().filter(pl.col("a") + 1 > 2).collect(engine=polars_metal.MetalEngine(debug=True))
    msgs = [r.getMessage() for r in caplog.records if r.name == "polars_metal"]
    assert any("falling back" in m for m in msgs), msgs
    assert not any("installed UDF" in m for m in msgs), msgs
