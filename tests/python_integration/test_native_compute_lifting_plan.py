# tests/python_integration/test_native_compute_lifting_plan.py
"""Python ↔ Rust round-trip for compute_lifting_plan.

The Python walker (Task 7) will produce a plan dict in this shape and
consume the lifting plan dict in this shape. This test pins the wire
format independently of the walker so changes to one side surface
loudly in the other.
"""

from __future__ import annotations

from polars_metal import _native


def test_filter_over_scan_routes_to_cpu_leave() -> None:
    plan = {
        "kind": "Filter",
        "input": {
            "kind": "Scan",
            "n_rows": 1_000_000,
            "columns": [["a", "I64"]],
        },
        "predicate": {"kind": "Column", "name": "mask", "dtype": "Bool"},
    }
    lifting = _native.compute_lifting_plan(plan)
    # Wire format: dict[str, str], where the key is "Kind#seq" and the
    # value is "gpu_lift" | "cpu_leave" | "fallback:<reason>".
    assert lifting["Scan#0"] == "cpu_leave"
    assert lifting["Filter#1"] == "cpu_leave"


def test_project_over_scan_routes_to_cpu_leave() -> None:
    plan = {
        "kind": "Project",
        "input": {
            "kind": "Scan",
            "n_rows": 1_000,
            "columns": [["a", "I64"]],
        },
        "columns": ["a"],
    }
    lifting = _native.compute_lifting_plan(plan)
    assert lifting["Scan#0"] == "cpu_leave"
    assert lifting["Project#1"] == "cpu_leave"


def test_unknown_kind_yields_fallback() -> None:
    # Use a genuinely-unsupported IR kind. (Sort moved out of the "unknown"
    # set in Phase 10 — it's now a CPU-passthrough wrapper that inherits
    # its child's decision.)
    plan = {"kind": "UnknownXyz", "input": {"kind": "Scan", "n_rows": 100, "columns": []}}
    lifting = _native.compute_lifting_plan(plan)
    decision = lifting["UnknownXyz#1"]
    assert decision.startswith("fallback:"), f"expected fallback, got {decision!r}"


def test_sort_passes_through_child_decision() -> None:
    """Sort is a CPU-passthrough wrapper since Phase 10; it inherits the
    decision of its inner subtree. Below the groupby cost-model threshold,
    that's CpuLeave; above, it's GpuLift via the inner GroupBy."""
    plan = {"kind": "Sort", "input": {"kind": "Scan", "n_rows": 100, "columns": []}}
    lifting = _native.compute_lifting_plan(plan)
    # Inner Scan is cpu_leave by initial cost; Sort inherits.
    assert lifting["Sort#1"] == "cpu_leave"
