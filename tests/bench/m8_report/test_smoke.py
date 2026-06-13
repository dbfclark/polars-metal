"""Smoke + correctness gate — runs every registry entry at its smallest size,
iters=1, and asserts engine output matches the CPU baseline. A fast wrong
answer is not a win, so the report harness doubles as a differential check.

Runs in `make test-unit` (smallest sizes only, fast).
"""

from __future__ import annotations

import pytest

from tests.bench.m8_report.registry import ENTRIES
from tests.bench.m8_report.run import smoke_one


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e.name)
def test_smoke_correctness(entry):
    smoke_one(entry)  # builds smallest input, runs engine+cpu, asserts match
