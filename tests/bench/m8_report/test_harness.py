from __future__ import annotations

import time

from tests.bench.m8_report._timing import Stats, measure
from tests.bench.m8_report.registry import ENTRIES, BenchEntry


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
