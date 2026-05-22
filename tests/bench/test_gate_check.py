"""Verify the gate-check helper enforces per-entry ratio thresholds."""
from tests.bench._gate_check import check_baseline


def _baseline_with(queries):
    """Build a fixture mirroring real baseline.json: queries nested under top-level key."""
    return {
        "_notes": "test fixture",
        "git_sha": "deadbeef",
        "date": "2026-05-22",
        "queries": queries,
    }


def test_check_passes_when_all_ratios_meet_threshold():
    baseline = _baseline_with({
        "tpch_q1_modified": {"ratio_metal_over_cpu": 0.914, "_gate": {"ratio_lt": 1.0}},
    })
    failures = check_baseline(baseline)
    assert failures == []


def test_check_fails_when_ratio_exceeds_threshold():
    baseline = _baseline_with({
        "tpch_q1_modified": {"ratio_metal_over_cpu": 1.05, "_gate": {"ratio_lt": 1.0}},
    })
    failures = check_baseline(baseline)
    assert len(failures) == 1
    assert "tpch_q1_modified" in failures[0]
    assert "1.05" in failures[0] and "1.0" in failures[0]


def test_check_skips_entries_without_gate_metadata():
    baseline = _baseline_with({
        "informational_entry": {"ratio_metal_over_cpu": 99.0},
    })
    failures = check_baseline(baseline)
    assert failures == []


def test_check_reports_missing_required_key():
    baseline = _baseline_with({
        "tpch_q1_modified": {"_gate": {"ratio_lt": 1.0}},  # ratio_metal_over_cpu absent
    })
    failures = check_baseline(baseline)
    assert any("missing ratio_metal_over_cpu" in f for f in failures)
