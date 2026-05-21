# tests/python_integration/test_router_application.py
"""Phase 1 router-application tests.

After Task 7 the walker no longer makes routing decisions on its own.
It builds a MetalPlanNode tree, hands it to _native.compute_lifting_plan,
receives a lifting plan dict, and applies it. With M2's starting cost
model (filter→CPU always), filter queries that used to install a UDF
should now leave the IR untouched and route to CPU.

The result correctness contract is unchanged from M1: any query routed
to CPU by the router must still produce the same DataFrame as
`engine="cpu"`. Tests assert byte-exact equality.
"""

from __future__ import annotations

import logging

import polars as pl
from polars.testing import assert_frame_equal

import polars_metal


def test_filter_with_router_routes_to_cpu_no_udf_installed(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="polars_metal")
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
    cpu = df.lazy().filter(pl.col("a") > 2).collect()
    metal = df.lazy().filter(pl.col("a") > 2).collect(engine=polars_metal.MetalEngine(debug=True))
    assert_frame_equal(cpu, metal)
    log_text = " ".join(r.getMessage() for r in caplog.records if r.name == "polars_metal")
    # New: with M2's filter→CPU cost rule, no UDF is installed for filter.
    assert "installed UDF" not in log_text, f"expected no UDF, got: {log_text}"


def test_select_only_query_still_routes_to_cpu_via_router(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="polars_metal")
    df = pl.DataFrame({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]})
    # Select with no filter still routes to CPU (no GPU-beneficial op).
    cpu = df.lazy().select(["b", "a"]).collect()
    metal = df.lazy().select(["b", "a"]).collect(engine=polars_metal.MetalEngine(debug=True))
    assert_frame_equal(cpu, metal)


def test_fallback_unrecognized_op_still_routes_to_cpu() -> None:
    # Sort isn't recognised by the walker — should fall back cleanly.
    df = pl.DataFrame({"a": [3, 1, 2]})
    cpu = df.lazy().sort("a").collect()
    metal = df.lazy().sort("a").collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)
