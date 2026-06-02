"""Per-entry perf-gate check.

`baseline.json` entries may include a `_gate` block:

    {
        "ratio_metal_over_cpu": 0.914,
        "_gate": {"ratio_lt": 1.0}
    }

If `_gate.ratio_lt` is present, the actual ratio must be strictly less.
Entries without a `_gate` block are informational (no check).
"""

from __future__ import annotations

from typing import Any


def check_baseline(baseline: dict[str, Any]) -> list[str]:
    """Return a list of failure messages; empty list = pass.

    Iterates baseline["queries"]; top-level keys (_units, _notes, machine,
    git_sha, date) are metadata and skipped.
    """
    failures: list[str] = []
    queries = baseline.get("queries", {})
    for name, entry in queries.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("_pending"):
            continue
        gate = entry.get("_gate")
        if gate is None:
            continue
        if "ratio_lt" in gate:
            actual = entry.get("ratio_metal_over_cpu")
            if actual is None:
                failures.append(f"{name}: missing ratio_metal_over_cpu (gate requires it)")
                continue
            limit = gate["ratio_lt"]
            if not actual < limit:
                failures.append(f"{name}: ratio_metal_over_cpu={actual} not < {limit}")
    return failures
