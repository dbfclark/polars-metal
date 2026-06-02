"""Shared timing harness for M4 survey CPU baselines.

Methodology (matches docs/m4-benchmark-survey.md):
  - 1 warmup iteration (discarded)
  - 5 measured iterations
  - Report median wall-clock in milliseconds
  - Use perf_counter_ns for sub-microsecond resolution
  - Force GC between runs to avoid amortized free() noise
"""

from __future__ import annotations

import gc
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter_ns


@dataclass
class BenchResult:
    name: str
    median_ms: float
    min_ms: float
    max_ms: float
    n_iters: int
    extra: dict[str, object]

    def __str__(self) -> str:
        return (
            f"{self.name:50s} median={self.median_ms:9.2f}ms  "
            f"min={self.min_ms:9.2f}ms  max={self.max_ms:9.2f}ms  "
            f"n={self.n_iters}  {self.extra}"
        )


def time_callable(
    name: str,
    fn: Callable[[], object],
    *,
    n_warmup: int = 1,
    n_measure: int = 5,
    extra: dict[str, object] | None = None,
) -> BenchResult:
    """Time `fn` with warmup and report median wall-clock.

    Returns BenchResult. Prints a one-line summary to stdout.
    """
    for _ in range(n_warmup):
        result = fn()
        del result
        gc.collect()

    samples_ns: list[int] = []
    for _ in range(n_measure):
        gc.collect()
        t0 = perf_counter_ns()
        result = fn()
        t1 = perf_counter_ns()
        samples_ns.append(t1 - t0)
        del result

    samples_ms = [s / 1e6 for s in samples_ns]
    res = BenchResult(
        name=name,
        median_ms=statistics.median(samples_ms),
        min_ms=min(samples_ms),
        max_ms=max(samples_ms),
        n_iters=n_measure,
        extra=extra or {},
    )
    print(res)
    return res
