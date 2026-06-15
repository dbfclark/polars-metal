"""The single timing path for the M8 report.

Wraps the proven m4_survey.time_callable (warmup + median-of-N with gc between
runs) so every callable in the report is measured identically. The engine path
is always timed as a closure that includes host->Metal ingest, compute, and
fold-back — no carve-outs. That tax is the whole point of the report.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tests.bench.m4_survey._timing import time_callable

DEFAULT_WARMUP = 2
DEFAULT_ITERS = 7


@dataclass
class Stats:
    median_ms: float
    min_ms: float
    p90_ms: float


def measure(
    fn: Callable[[], object],
    *,
    warmup: int = DEFAULT_WARMUP,
    iters: int = DEFAULT_ITERS,
) -> Stats:
    """Warm `fn` `warmup` times (discarded), time it `iters` times, return Stats."""
    res = time_callable("m8", fn, n_warmup=warmup, n_measure=iters)
    # time_callable exposes median/min/max; for our small iters, max is the p90 proxy.
    return Stats(median_ms=res.median_ms, min_ms=res.min_ms, p90_ms=res.max_ms)
