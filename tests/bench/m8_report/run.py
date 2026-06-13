"""Driver: smoke-correctness mode and full-timing mode.

No timing logic of its own (delegates to _timing.measure); no formatting
(delegates to emit).
"""

from __future__ import annotations

from dataclasses import dataclass

from tests.bench.m8_report._timing import measure
from tests.bench.m8_report.registry import ENTRIES, BenchEntry


@dataclass
class Row:
    name: str
    category: str
    size: int
    engine_ms: float
    cpu_ms: float
    ceiling_ms: float | None
    engine_vs_cpu: float  # cpu_ms / engine_ms  (>1 = engine win)
    ceiling_vs_cpu: float | None  # cpu_ms / ceiling_ms
    tax: float | None  # engine_ms / ceiling_ms  (>=1 overhead)
    verdict: str


def _default_check(engine_out, cpu_out) -> None:
    import numpy as np
    import polars as pl

    if isinstance(engine_out, pl.DataFrame) and isinstance(cpu_out, pl.DataFrame):
        assert engine_out.columns == cpu_out.columns
        for col in engine_out.columns:
            a, b = engine_out[col].to_numpy(), cpu_out[col].to_numpy()
            if np.issubdtype(a.dtype, np.number):
                np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3, err_msg=col)
    else:  # scalars / arrays
        np.testing.assert_allclose(
            np.asarray(engine_out), np.asarray(cpu_out), rtol=1e-3, atol=1e-3
        )


def smoke_one(entry: BenchEntry) -> None:
    """Run engine + cpu at the smallest size, assert correctness."""
    size = min(entry.sizes)
    data = entry.make_input(size)
    engine_out = entry.engine_fn(data)
    cpu_out = entry.cpu_fn(data)
    check = entry.check or _default_check
    check(engine_out, cpu_out)


def _verdict(engine_vs_cpu: float) -> str:
    if engine_vs_cpu >= 10.0:
        return "✅ ≥10×"  # noqa: RUF001
    if engine_vs_cpu > 1.15:
        return "🟢 win"
    if engine_vs_cpu >= 0.85:
        return "🟡 tie"
    return "🔴 loss"


def run(entries: list[BenchEntry] = ENTRIES) -> list[Row]:
    """Full sweep: every entry x every size, timed. Returns rows."""
    rows: list[Row] = []
    for e in entries:
        for size in e.sizes:
            data = e.make_input(size)
            # bind loop vars (e, data) as defaults to silence ruff B023 — these
            # closures are called synchronously inside the loop, so binding is safe.
            engine_ms = measure(lambda e=e, data=data: e.engine_fn(data)).median_ms
            cpu_ms = measure(lambda e=e, data=data: e.cpu_fn(data)).median_ms
            ceiling_ms = (
                measure(lambda e=e, data=data: e.ceiling_fn(data)).median_ms
                if e.ceiling_fn is not None
                else None
            )
            engine_vs_cpu = cpu_ms / engine_ms
            ceiling_vs_cpu = (cpu_ms / ceiling_ms) if ceiling_ms else None
            tax = (engine_ms / ceiling_ms) if ceiling_ms else None
            rows.append(
                Row(
                    name=e.name,
                    category=e.category,
                    size=size,
                    engine_ms=engine_ms,
                    cpu_ms=cpu_ms,
                    ceiling_ms=ceiling_ms,
                    engine_vs_cpu=engine_vs_cpu,
                    ceiling_vs_cpu=ceiling_vs_cpu,
                    tax=tax,
                    verdict=_verdict(engine_vs_cpu),
                )
            )
            print(
                f"{e.name:24s} N={size:>12,}  engine={engine_ms:8.2f}ms "
                f"cpu={cpu_ms:8.2f}ms  {engine_vs_cpu:6.2f}× {_verdict(engine_vs_cpu)}"  # noqa: RUF001
            )
    return rows
