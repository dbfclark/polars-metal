from __future__ import annotations

import time

from tests.bench.m8_report._timing import Stats, measure


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
