"""Tests for Rust-side ``_native.execute_plan``: scan + project dispatch.

After Task 8 the Python UDF closure body is a thin call into Rust. The
walker captures the underlying ``PyDataFrame`` from the ``DataFrameScan``
node and passes it (plus a serialized plan dict) into Rust, which
interprets the plan and reassembles a ``PyDataFrame`` for return.

These tests exercise the Rust entry point directly *and* the end-to-end
walker path, so a regression in either layer surfaces.

Important: the Rust side must NOT call into ``LazyFrame.collect`` or any
path that re-enters the Polars engine plugin mechanism (else the
intercepted ``collect`` recurses through MetalEngine forever). It calls
``PyDataFrame.select`` directly via PyO3 ``call_method1``.
"""

from __future__ import annotations

import logging

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_metal
from polars_metal import _native


def test_native_execute_plan_round_trips_scan() -> None:
    """A Scan-only plan returns the input df unchanged."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]})
    plan = {"kind": "Scan", "n_rows": 3, "columns": [["a", "I64"], ["b", "F64"]]}
    result_pydf = _native.execute_plan(df._df, plan)
    result = pl.DataFrame._from_pydf(result_pydf)
    assert_frame_equal(df, result)


def test_native_execute_plan_round_trips_scan_then_project() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]})
    plan = {
        "kind": "Project",
        "columns": ["b"],
        "input": {
            "kind": "Scan",
            "n_rows": 3,
            "columns": [["a", "I64"], ["b", "F64"]],
        },
    }
    result_pydf = _native.execute_plan(df._df, plan)
    result = pl.DataFrame._from_pydf(result_pydf)
    assert_frame_equal(df.select("b"), result)


def test_native_execute_plan_project_reorders_columns() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": [5, 6]})
    plan = {
        "kind": "Project",
        "columns": ["c", "a", "b"],
        "input": {
            "kind": "Scan",
            "n_rows": 2,
            "columns": [["a", "I64"], ["b", "F64"], ["c", "I64"]],
        },
    }
    result_pydf = _native.execute_plan(df._df, plan)
    result = pl.DataFrame._from_pydf(result_pydf)
    assert_frame_equal(df.select(["c", "a", "b"]), result)


def test_native_execute_plan_filter_raises_not_implemented() -> None:
    """Filter dispatch arrives in Phase 5+; today it must error cleanly,
    not panic, not silently fall back."""
    df = pl.DataFrame({"a": [1, 2, 3], "mask": [True, False, True]})
    plan = {
        "kind": "Filter",
        "predicate": {"kind": "Column", "name": "mask", "dtype": "Bool"},
        "input": {
            "kind": "Scan",
            "n_rows": 3,
            "columns": [["a", "I64"], ["mask", "Bool"]],
        },
    }
    with pytest.raises(NotImplementedError) as excinfo:
        _native.execute_plan(df._df, plan)
    msg = str(excinfo.value).lower()
    assert "phase 5" in msg or "filter" in msg or "not implemented" in msg, (
        f"Filter rejection message unclear: {excinfo.value!r}"
    )


def test_native_execute_plan_unknown_kind_raises() -> None:
    """Bogus kinds must surface as a ValueError, not silently produce wrong
    output."""
    df = pl.DataFrame({"a": [1, 2, 3]})
    plan = {"kind": "NonsenseNode"}
    with pytest.raises(ValueError) as excinfo:
        _native.execute_plan(df._df, plan)
    assert "NonsenseNode" in str(excinfo.value) or "unknown" in str(excinfo.value).lower()


def test_end_to_end_walker_select_uses_rust_dispatch(caplog) -> None:
    """End-to-end smoke test: a select query routes through the walker and the
    Rust router. Under M2's cost model (project inherits scan→CPU), the router
    routes this to CPU without installing a UDF. The result must still match
    CPU Polars."""
    caplog.set_level(logging.DEBUG, logger="polars_metal")
    df = pl.DataFrame({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]})
    cpu = df.lazy().select(["b", "a"]).collect()
    metal = df.lazy().select(["b", "a"]).collect(engine=polars_metal.MetalEngine(debug=True))
    assert_frame_equal(cpu, metal)
    msgs = [r.getMessage() for r in caplog.records if r.name == "polars_metal"]
    # M2 cost model: Scan→CpuLeave, Project inherits→CpuLeave. No UDF installed.
    assert any("router routes entire query to CPU" in m for m in msgs), msgs


def test_end_to_end_walker_scan_only_uses_rust_dispatch() -> None:
    """End-to-end test for the bare-scan path (no projection node)."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]})
    cpu = df.lazy().collect()
    metal = df.lazy().collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)
