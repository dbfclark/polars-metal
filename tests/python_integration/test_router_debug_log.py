# tests/python_integration/test_router_debug_log.py
"""Verify MetalEngine(debug=True) emits a parseable per-node decision log.

Spec § "Layer 1.5: Router behavior" expects test_routing.py (this file)
to assert decisions per node by parsing debug logs. We pin the log
format here so future log-format changes surface loudly.

Log format
----------
Per query, one DEBUG record on logger 'polars_metal' of the form:

    "router decisions: {Kind#seq: decision, ...}"

where `decision` is one of `gpu_lift`, `cpu_leave`, or `fallback:<reason>`.
"""

from __future__ import annotations

import ast
import logging

import polars as pl

import polars_metal


def _decisions_from_logs(caplog) -> dict[str, str]:
    for r in caplog.records:
        if r.name == "polars_metal" and r.getMessage().startswith("router decisions: "):
            payload = r.getMessage()[len("router decisions: ") :]
            # Eval'd back to dict; the log uses repr().
            return ast.literal_eval(payload)
    return {}


def test_filter_query_logs_cpu_leave_for_all_nodes(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="polars_metal")
    df = pl.DataFrame({"a": [1, 2, 3]})
    df.lazy().filter(pl.col("a") > 1).collect(engine=polars_metal.MetalEngine(debug=True))
    decisions = _decisions_from_logs(caplog)
    assert "Scan#0" in decisions
    assert "Filter#1" in decisions
    assert decisions["Filter#1"] == "cpu_leave"
    assert decisions["Scan#0"] == "cpu_leave"


def test_sort_query_logs_fallback_for_unrecognized_node(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="polars_metal")
    df = pl.DataFrame({"a": [3, 1, 2]})
    df.lazy().sort("a").collect(engine=polars_metal.MetalEngine(debug=True))
    # The walker itself emits FallBack on Sort (it's not in the walker's
    # accepted set yet); the router is never called. Confirm the walker
    # log says so.
    log_text = " ".join(r.getMessage() for r in caplog.records if r.name == "polars_metal")
    assert "walker fallback" in log_text
