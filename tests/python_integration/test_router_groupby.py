# tests/python_integration/test_router_groupby.py
"""End-to-end router test for GroupBy.

After Task 12 the wire format accepts GroupBy nodes and the router
returns a per-node decision dict. We don't run a UDF yet — Phase 2's
goal is plumbing, not execution.
"""

from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

import polars_metal
from polars_metal import _native


def test_compute_lifting_plan_recognizes_groupby_at_high_row_count() -> None:
    plan = {
        "kind": "GroupBy",
        "input": {
            "kind": "Scan",
            "n_rows": 1_000_000,
            "columns": [["k", "I64"], ["v", "I64"]],
        },
        "keys": [["k", "I64"]],
        "aggs": [{"input_col": "v", "op": "Sum", "output_alias": "sum_v"}],
    }
    lifting = _native.compute_lifting_plan(plan)
    assert lifting["Scan#0"] == "cpu_leave"
    assert lifting["GroupBy#1"] == "gpu_lift"


def test_compute_lifting_plan_routes_small_groupby_to_cpu() -> None:
    plan = {
        "kind": "GroupBy",
        "input": {
            "kind": "Scan",
            "n_rows": 10_000,
            "columns": [["k", "I64"], ["v", "I64"]],
        },
        "keys": [["k", "I64"]],
        "aggs": [{"input_col": "v", "op": "Sum", "output_alias": "sum_v"}],
    }
    lifting = _native.compute_lifting_plan(plan)
    assert lifting["GroupBy#1"] == "cpu_leave"


def test_groupby_over_filter_uses_filter_input_row_count() -> None:
    plan = {
        "kind": "GroupBy",
        "input": {
            "kind": "Filter",
            "input": {
                "kind": "Scan",
                "n_rows": 1_000_000,
                "columns": [["k", "I64"]],
            },
            "predicate": {"kind": "Column", "name": "_mask", "dtype": "Bool"},
        },
        "keys": [["k", "I64"]],
        "aggs": [],
    }
    lifting = _native.compute_lifting_plan(plan)
    assert lifting["Scan#0"] == "cpu_leave"
    assert lifting["Filter#1"] == "cpu_leave"
    assert lifting["GroupBy#2"] == "gpu_lift"


def test_groupby_end_to_end_still_correct_via_cpu_fallback() -> None:
    df = pl.DataFrame({"k": [1, 1, 2, 2, 3], "v": [10, 20, 30, 40, 50]})
    cpu = df.lazy().group_by("k").agg(pl.col("v").sum().alias("s")).sort("k").collect()
    metal = (
        df.lazy()
        .group_by("k")
        .agg(pl.col("v").sum().alias("s"))
        .sort("k")
        .collect(engine=polars_metal.MetalEngine())
    )
    assert_frame_equal(cpu, metal)
