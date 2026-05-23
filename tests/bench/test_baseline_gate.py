"""End-of-bench-session gate verification.

Runs after pytest-benchmark fixtures have updated baseline.json.
Fails the session if any _gate-ed entry violates its threshold.
"""

import json
from pathlib import Path

import pytest

from tests.bench._gate_check import check_baseline


def test_baseline_gate_thresholds_met():
    path = Path(__file__).parent / "baseline.json"
    baseline = json.loads(path.read_text())
    failures = check_baseline(baseline)
    if failures:
        msg = "Perf gate failures:\n" + "\n".join(f"  - {f}" for f in failures)
        pytest.fail(msg)
