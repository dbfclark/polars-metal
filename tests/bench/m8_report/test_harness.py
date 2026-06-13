from __future__ import annotations

import json
import time

from tests.bench.m8_report._timing import Stats, measure
from tests.bench.m8_report.emit import build_header, to_json, to_markdown
from tests.bench.m8_report.registry import ENTRIES, BenchEntry
from tests.bench.m8_report.run import Row


def test_measure_excludes_warmup_and_reports_median():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        time.sleep(0.01)

    stats = measure(fn, warmup=2, iters=5)
    # 2 warmup + 5 measured = 7 total calls
    assert calls["n"] == 7
    assert isinstance(stats, Stats)
    # median should be ~10ms; allow generous slack for scheduler noise
    assert 8.0 <= stats.median_ms <= 40.0
    assert stats.min_ms <= stats.median_ms


def test_registry_entries_are_well_formed():
    assert len(ENTRIES) >= 1
    names = set()
    for e in ENTRIES:
        assert isinstance(e, BenchEntry)
        assert e.name and e.name not in names, f"dup/empty name {e.name!r}"
        names.add(e.name)
        assert e.category
        assert e.sizes and all(isinstance(s, int) for s in e.sizes)
        assert callable(e.make_input)
        assert callable(e.engine_fn)
        assert callable(e.cpu_fn)
        assert e.ceiling_fn is None or callable(e.ceiling_fn)
        assert e.check is None or callable(e.check)


def test_fusion_chain_category_present():
    cats = {e.category for e in ENTRIES}
    assert "fusion-chain" in cats


def _sample_rows():
    return [
        Row("haversine", "fusion-chain", 10_000_000, 12.0, 180.0, 3.5, 15.0, 51.4, 3.43, "✅ ≥10×"),  # noqa: RUF001
        Row(
            "tpch_q1",
            "conformance-loser",
            10_000_000,
            300.0,
            60.0,
            None,
            0.2,
            None,
            None,
            "🔴 loss",
        ),
    ]


def test_to_markdown_has_columns_and_rows():
    md = to_markdown(_sample_rows(), build_header())
    assert "engine ×CPU" in md  # noqa: RUF001
    assert "tax" in md
    assert "haversine" in md
    assert "🔴 loss" in md
    assert "fusion-chain" in md
    assert "conformance-loser" in md


def test_to_json_roundtrips():
    payload = to_json(_sample_rows(), build_header())
    parsed = json.loads(payload)
    assert parsed["rows"][0]["name"] == "haversine"
    assert parsed["rows"][0]["engine_vs_cpu"] == 15.0
    assert "polars_version" in parsed["header"]
