# M8 — Honest Perf Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified, regenerable benchmark harness that measures every shipped op end-to-end through `engine="metal"` and emits one honest perf report (engine-vs-CPU + raw-ceiling + ingest/fold-back tax), then run it and write the workload-mapping verdict.

**Architecture:** Four focused modules under `tests/bench/m8_report/` — `_timing.py` (the single timing path, wrapping the proven `time_callable`), `registry.py` (pure-data `BenchEntry` list + callables, reusing existing `m4_*` fixtures), `run.py` (driver: smoke-correctness mode + full-timing mode), `emit.py` (rows → `docs/perf-report.md` + `perf-report.json`). Timing logic lives in exactly one place; the report is a pure function of collected rows.

**Tech Stack:** Python 3.11, Polars, polars_metal (`engine="metal"` + `.metal` namespace), NumPy, MLX (raw ceilings), pytest. Spec: `docs/superpowers/specs/2026-06-12-m8-honest-perf-report-design.md`.

---

## File Structure

| File | Responsibility |
|---|---|
| `tests/bench/m8_report/__init__.py` | package marker |
| `tests/bench/m8_report/_timing.py` | `Stats` + `measure(fn, *, warmup, iters)` — the only timing path; wraps `m4_survey._timing.time_callable` |
| `tests/bench/m8_report/registry.py` | `BenchEntry` dataclass + `ENTRIES: list[BenchEntry]`; pure data + callables, no timing/formatting |
| `tests/bench/m8_report/run.py` | `smoke()` (smallest size, iters=1, assert engine==cpu) and `run()` (full sweep → `Row` list) |
| `tests/bench/m8_report/emit.py` | `to_markdown(rows, header)` + `to_json(rows, header)`; writes report artifacts |
| `tests/bench/m8_report/test_harness.py` | unit tests for `measure`, `emit`, and registry validation |
| `tests/bench/m8_report/test_smoke.py` | the smoke+correctness gate (runs in `make test-unit`) |
| `docs/perf-report.md` | generated artifact (committed) |
| `perf-report.json` | generated machine-readable twin (committed) |
| `Makefile` | new `perf-report` target; smoke gate wired into `test-unit` |

**Reused fixtures (import, do not rebuild):**
- `tests/bench/m4_survey/_timing.py::time_callable`, `BenchResult`
- `tests/bench/m4_engine/bench_haversine_e2e.py::_make_taxi`, `_haversine_expr`
- existing `.metal` verb signatures: `corr(force_gpu=False)`, `cosine_topk(corpus, k, corpus_col="emb")`, `knn(corpus, k, corpus_col="emb")`, `fft()`, `dtw(ref, window=..., allow_cpu_fallback=...)`

---

## Task 1: Timing core (`_timing.py`)

**Files:**
- Create: `tests/bench/m8_report/__init__.py`
- Create: `tests/bench/m8_report/_timing.py`
- Test: `tests/bench/m8_report/test_harness.py`

- [ ] **Step 1: Create the package marker**

Create `tests/bench/m8_report/__init__.py`:

```python
"""M8 honest perf report harness."""
```

- [ ] **Step 2: Write the failing test**

Create `tests/bench/m8_report/test_harness.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/bench/m8_report/test_harness.py::test_measure_excludes_warmup_and_reports_median -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.bench.m8_report._timing'`

- [ ] **Step 4: Write minimal implementation**

Create `tests/bench/m8_report/_timing.py`:

```python
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
    """Warm `fn` `warmup` times (discarded), time it `iters` times, return Stats.

    Delegates to the proven time_callable for the warmup/measure/gc loop, then
    derives p90 from the same samples via a second timed pass is NOT done — we
    reuse time_callable's median/min and approximate p90 as max for small N.
    """
    res = time_callable("m8", fn, n_warmup=warmup, n_measure=iters)
    # time_callable exposes median/min/max; for our small iters, max == p90 proxy.
    return Stats(median_ms=res.median_ms, min_ms=res.min_ms, p90_ms=res.max_ms)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/bench/m8_report/test_harness.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/bench/m8_report/__init__.py tests/bench/m8_report/_timing.py tests/bench/m8_report/test_harness.py
git commit -m "M8: timing core wrapping the proven time_callable"
```

---

## Task 2: Registry skeleton + `BenchEntry` (fusion-chain seed entries)

**Files:**
- Create: `tests/bench/m8_report/registry.py`
- Modify: `tests/bench/m8_report/test_harness.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/bench/m8_report/test_harness.py`:

```python
from tests.bench.m8_report.registry import ENTRIES, BenchEntry


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
        # ceiling_fn and check are optional
        assert e.ceiling_fn is None or callable(e.ceiling_fn)
        assert e.check is None or callable(e.check)


def test_fusion_chain_category_present():
    cats = {e.category for e in ENTRIES}
    assert "fusion-chain" in cats
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/bench/m8_report/test_harness.py::test_registry_entries_are_well_formed -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.bench.m8_report.registry'`

- [ ] **Step 3: Write minimal implementation**

Create `tests/bench/m8_report/registry.py`:

```python
"""The op registry: pure data + callables, no timing/formatting logic.

Each BenchEntry carries up to three callables:
  - engine_fn: full engine="metal" wall-clock (ingest+compute+fold-back). REQUIRED.
  - cpu_fn:    the mission baseline — Polars CPU where a native expr exists,
               else the idiomatic CPU tool (numpy/scipy/dtaidistance). REQUIRED.
  - ceiling_fn: raw MLX/numpy with no engine overhead. OPTIONAL (None where no
               meaningful raw form exists).
  - check:     optional correctness comparator (engine_out, cpu_out) -> None,
               raises on mismatch. None => default numeric-allclose on result.

Fixtures are imported from existing m4_* benches, not rebuilt.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

import polars_metal as pm
from tests.bench.m4_engine.bench_haversine_e2e import _haversine_expr, _make_taxi

_ENGINE = pm.MetalEngine()


@dataclass
class BenchEntry:
    name: str
    category: str
    sizes: list[int]
    make_input: Callable[[int], Any]
    engine_fn: Callable[[Any], Any]
    cpu_fn: Callable[[Any], Any]
    ceiling_fn: Callable[[Any], Any] | None = None
    check: Callable[[Any, Any], None] | None = None


# ---- helpers -------------------------------------------------------------

def _black_scholes_expr() -> pl.Expr:
    # F32 transcendental chain on a single price column.
    s = pl.col("s")
    k, r, t, vol = 100.0, 0.02, 1.0, 0.3
    d1 = ((s / k).log() + (r + 0.5 * vol * vol) * t) / (vol * (t ** 0.5))
    d2 = d1 - vol * (t ** 0.5)
    # crude normal-CDF proxy via tanh approx — kept identical on both paths.
    ncdf = lambda x: 0.5 * (1.0 + (x * 0.7978845608).tanh())
    return s * ncdf(d1) - k * (-r * t).exp() * ncdf(d2)


def _make_prices(n: int, seed: int = 0xB5) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"s": rng.uniform(50, 150, size=n).astype(np.float32)})


def _frame_allclose(engine_out: pl.DataFrame, cpu_out: pl.DataFrame, *, rtol=1e-3, atol=1e-3) -> None:
    """Default check: every numeric column close between engine and CPU output."""
    assert engine_out.columns == cpu_out.columns, (engine_out.columns, cpu_out.columns)
    for col in engine_out.columns:
        a = engine_out[col].to_numpy()
        b = cpu_out[col].to_numpy()
        if np.issubdtype(a.dtype, np.number):
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=f"col {col}")


# ---- registry ------------------------------------------------------------

ENTRIES: list[BenchEntry] = [
    BenchEntry(
        name="haversine",
        category="fusion-chain",
        sizes=[1_000_000, 10_000_000, 100_000_000],
        make_input=_make_taxi,
        engine_fn=lambda df: df.lazy().with_columns(d=_haversine_expr()).collect(engine=_ENGINE),
        cpu_fn=lambda df: df.lazy().with_columns(d=_haversine_expr()).collect(),
        ceiling_fn=None,
        check=_frame_allclose,
    ),
    BenchEntry(
        name="black_scholes",
        category="fusion-chain",
        sizes=[1_000_000, 10_000_000, 100_000_000],
        make_input=_make_prices,
        engine_fn=lambda df: df.lazy().with_columns(c=_black_scholes_expr()).collect(engine=_ENGINE),
        cpu_fn=lambda df: df.lazy().with_columns(c=_black_scholes_expr()).collect(),
        ceiling_fn=None,
        check=_frame_allclose,
    ),
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/bench/m8_report/test_harness.py -v`
Expected: PASS (all three tests)

- [ ] **Step 5: Commit**

```bash
git add tests/bench/m8_report/registry.py tests/bench/m8_report/test_harness.py
git commit -m "M8: BenchEntry registry + fusion-chain seed entries"
```

---

## Task 3: Smoke+correctness driver + gate

**Files:**
- Create: `tests/bench/m8_report/run.py`
- Create: `tests/bench/m8_report/test_smoke.py`

- [ ] **Step 1: Write the failing test**

Create `tests/bench/m8_report/test_smoke.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/bench/m8_report/test_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.bench.m8_report.run'`

- [ ] **Step 3: Write minimal implementation**

Create `tests/bench/m8_report/run.py`:

```python
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
    engine_vs_cpu: float          # cpu_ms / engine_ms  (>1 = engine win)
    ceiling_vs_cpu: float | None  # cpu_ms / ceiling_ms
    tax: float | None             # engine_ms / ceiling_ms  (>=1 overhead)
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
        return "✅ ≥10×"
    if engine_vs_cpu > 1.15:
        return "🟢 win"
    if engine_vs_cpu >= 0.85:
        return "🟡 tie"
    return "🔴 loss"


def run(entries: list[BenchEntry] = ENTRIES) -> list[Row]:
    """Full sweep: every entry × every size, timed. Returns rows."""
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
                    name=e.name, category=e.category, size=size,
                    engine_ms=engine_ms, cpu_ms=cpu_ms, ceiling_ms=ceiling_ms,
                    engine_vs_cpu=engine_vs_cpu, ceiling_vs_cpu=ceiling_vs_cpu,
                    tax=tax, verdict=_verdict(engine_vs_cpu),
                )
            )
            print(f"{e.name:24s} N={size:>12,}  engine={engine_ms:8.2f}ms "
                  f"cpu={cpu_ms:8.2f}ms  {engine_vs_cpu:6.2f}× {_verdict(engine_vs_cpu)}")
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/bench/m8_report/test_smoke.py -v`
Expected: PASS (haversine + black_scholes smoke at 1M, engine==cpu within tol)

- [ ] **Step 5: Commit**

```bash
git add tests/bench/m8_report/run.py tests/bench/m8_report/test_smoke.py
git commit -m "M8: smoke+correctness driver + gate; full-timing run()"
```

---

## Task 4: Report emitter (`emit.py`)

**Files:**
- Create: `tests/bench/m8_report/emit.py`
- Modify: `tests/bench/m8_report/test_harness.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/bench/m8_report/test_harness.py`:

```python
import json

from tests.bench.m8_report.emit import build_header, to_json, to_markdown
from tests.bench.m8_report.run import Row


def _sample_rows():
    return [
        Row("haversine", "fusion-chain", 10_000_000, 12.0, 180.0, 3.5,
            15.0, 51.4, 3.43, "✅ ≥10×"),
        Row("tpch_q1", "conformance-loser", 10_000_000, 300.0, 60.0, None,
            0.2, None, None, "🔴 loss"),
    ]


def test_to_markdown_has_columns_and_rows():
    md = to_markdown(_sample_rows(), build_header())
    assert "engine ×CPU" in md
    assert "tax" in md
    assert "haversine" in md
    assert "🔴 loss" in md
    # category section headers present
    assert "fusion-chain" in md
    assert "conformance-loser" in md


def test_to_json_roundtrips():
    payload = to_json(_sample_rows(), build_header())
    parsed = json.loads(payload)
    assert parsed["rows"][0]["name"] == "haversine"
    assert parsed["rows"][0]["engine_vs_cpu"] == 15.0
    assert "polars_version" in parsed["header"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/bench/m8_report/test_harness.py::test_to_markdown_has_columns_and_rows -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.bench.m8_report.emit'`

- [ ] **Step 3: Write minimal implementation**

Create `tests/bench/m8_report/emit.py`:

```python
"""Artifact emitters: rows -> markdown report + JSON twin. No measurement."""

from __future__ import annotations

import json
import platform
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import polars as pl

from tests.bench.m8_report._timing import DEFAULT_ITERS, DEFAULT_WARMUP


def build_header() -> dict[str, str]:
    try:
        import mlx.core as mx
        mlx_version = getattr(mx, "__version__", "unknown")
    except Exception:
        mlx_version = "unavailable"
    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "polars_version": pl.__version__,
        "numpy_version": np.__version__,
        "mlx_version": mlx_version,
        "methodology": f"warmup={DEFAULT_WARMUP}, iters={DEFAULT_ITERS}, "
                       "median reported, engine path includes ingest+fold-back",
    }


def _fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def to_markdown(rows, header) -> str:
    lines = ["# polars-metal — honest perf report", ""]
    lines.append("> Internal decision-input. Numbers are machine-specific (see header). "
                 "Engine path is full `engine=\"metal\"` wall-clock incl. ingest + fold-back.")
    lines.append("")
    lines.append("## Environment")
    for k, v in header.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    # Executive scorecard
    ge10 = sum(1 for r in rows if r.engine_vs_cpu >= 10.0)
    wins = sum(1 for r in rows if r.engine_vs_cpu > 1.15)
    losses = sum(1 for r in rows if r.engine_vs_cpu < 0.85)
    lines += [
        "## Executive scorecard",
        "",
        f"- Rows clearing **≥10× vs CPU** (order-of-magnitude bar): **{ge10}** / {len(rows)}",
        f"- Rows that win (>1.15×): **{wins}** / {len(rows)}",
        f"- Rows that tie/lose: **{len(rows) - wins}** (losses: {losses})",
        "",
    ]

    # Per-category tables
    by_cat: dict[str, list] = defaultdict(list)
    for r in rows:
        by_cat[r.category].append(r)
    for cat, crows in by_cat.items():
        lines.append(f"## {cat}")
        lines.append("")
        lines.append("| op | size | engine ms | CPU ms | engine ×CPU | ceiling ms | ceiling ×CPU | tax | verdict |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in crows:
            lines.append(
                f"| {r.name} | {r.size:,} | {_fmt(r.engine_ms)} | {_fmt(r.cpu_ms)} | "
                f"{_fmt(r.engine_vs_cpu)}× | {_fmt(r.ceiling_ms)} | "
                f"{_fmt(r.ceiling_vs_cpu)} | {_fmt(r.tax)} | {r.verdict} |"
            )
        lines.append("")

    # Reconciliation + verdict are written by hand in Task 13 (appended below a marker).
    lines.append("<!-- VERDICT: filled in Task 13 after a real run -->")
    lines.append("")
    return "\n".join(lines)


def to_json(rows, header) -> str:
    return json.dumps(
        {"header": header, "rows": [asdict(r) for r in rows]},
        indent=2,
    )


def write_report(rows, *, md_path: str, json_path: str) -> None:
    header = build_header()
    with open(md_path, "w") as f:
        f.write(to_markdown(rows, header))
    with open(json_path, "w") as f:
        f.write(to_json(rows, header))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/bench/m8_report/test_harness.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/bench/m8_report/emit.py tests/bench/m8_report/test_harness.py
git commit -m "M8: report emitter (markdown + json) with synthetic-row tests"
```

---

## Task 5: CLI entrypoint + `make perf-report` + wire smoke gate into `test-unit`

**Files:**
- Modify: `tests/bench/m8_report/run.py` (add `main()`)
- Modify: `Makefile`

- [ ] **Step 1: Add `main()` to `run.py`**

Append to `tests/bench/m8_report/run.py`:

```python
def main() -> None:
    from tests.bench.m8_report.emit import write_report

    rows = run()
    write_report(rows, md_path="docs/perf-report.md", json_path="perf-report.json")
    print(f"\nWrote docs/perf-report.md + perf-report.json ({len(rows)} rows)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Inspect the current Makefile targets**

Run: `grep -n "test-unit\|^bench\|^perf" Makefile`
Expected: shows the existing `test-unit` and `bench` recipes so the new target matches their style.

- [ ] **Step 3: Add the `perf-report` target and wire the smoke gate**

In `Makefile`, add `perf-report` to `.PHONY` and add this target near `bench`:

```makefile
perf-report:
	python -m tests.bench.m8_report.run
```

And add the smoke gate to the `test-unit` recipe (append as a new line at the end of the existing `test-unit` block — keep `--test-threads=1` semantics for the Rust side unchanged; this is the Python smoke gate):

```makefile
	pytest tests/bench/m8_report/test_smoke.py tests/bench/m8_report/test_harness.py -q
```

- [ ] **Step 4: Verify the smoke gate runs under test-unit**

Run: `pytest tests/bench/m8_report/test_smoke.py tests/bench/m8_report/test_harness.py -q`
Expected: PASS (all smoke + harness tests green)

- [ ] **Step 5: Commit**

```bash
git add tests/bench/m8_report/run.py Makefile
git commit -m "M8: make perf-report target + smoke gate wired into test-unit"
```

---

## Task 6: Add rolling category

**Files:**
- Modify: `tests/bench/m8_report/registry.py`

- [ ] **Step 1: Add rolling entries to the registry**

In `registry.py`, add helper + entries (append before the closing `]` of `ENTRIES`):

```python
def _make_signal_1col(n: int, seed: int = 0x501) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"x": rng.standard_normal(n).astype(np.float32)})


def _rolling_entry(stat: str, window: int) -> BenchEntry:
    expr = getattr(pl.col("x"), f"rolling_{stat}")(window_size=window)
    return BenchEntry(
        name=f"rolling_{stat}_w{window}",
        category="rolling",
        sizes=[1_000_000, 10_000_000],
        make_input=_make_signal_1col,
        engine_fn=lambda df, e=expr: df.lazy().with_columns(r=e).collect(engine=_ENGINE),
        cpu_fn=lambda df, e=expr: df.lazy().with_columns(r=e).collect(),
        ceiling_fn=None,
        check=_frame_allclose,
    )
```

Then extend `ENTRIES`:

```python
ENTRIES += [
    _rolling_entry("mean", 1000),
    _rolling_entry("sum", 1000),
    _rolling_entry("var", 1000),
    _rolling_entry("std", 1000),
]
```

Note: rolling produces leading nulls (window-1 head). `_frame_allclose` uses `to_numpy()`, which renders nulls as NaN on both paths identically, so the comparison holds. If a NaN-vs-NaN mismatch surfaces, change `_frame_allclose` to pass `equal_nan=True` to `assert_allclose`.

- [ ] **Step 2: Run the smoke gate (now covers rolling)**

Run: `pytest tests/bench/m8_report/test_smoke.py -v -k rolling`
Expected: PASS (4 rolling entries, engine==cpu at 1M)

If a null/NaN mismatch fails: edit `_frame_allclose` to add `equal_nan=True` to the `assert_allclose` call, re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/bench/m8_report/registry.py
git commit -m "M8: add rolling category (mean/sum/var/std, window=1000)"
```

---

## Task 7: Add vector-search category (cosine_topk, knn)

**Files:**
- Modify: `tests/bench/m8_report/registry.py`

- [ ] **Step 1: Add vector-search entries**

In `registry.py`, add helpers + entries. The `.metal` verbs take a numpy corpus and return an Expr over a query embedding column. The CPU baseline is a numpy brute-force top-k.

```python
_VEC_D = 768
_VEC_K = 10
_VEC_CORPUS_N = 50_000


def _make_queries(n: int, seed: int = 0x7EC) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, _VEC_D)).astype(np.float32)
    return pl.DataFrame({"emb": emb.tolist()}, schema={"emb": pl.Array(pl.Float32, _VEC_D)})


def _vec_corpus(seed: int = 0xC0A) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((_VEC_CORPUS_N, _VEC_D)).astype(np.float32)


_CORPUS = _vec_corpus()


def _cpu_cosine_topk(df: pl.DataFrame) -> np.ndarray:
    q = np.asarray(df["emb"].to_list(), dtype=np.float32)
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    cn = _CORPUS / np.linalg.norm(_CORPUS, axis=1, keepdims=True)
    sims = qn @ cn.T                      # (Q, corpusN)
    idx = np.argsort(-sims, axis=1)[:, :_VEC_K]
    return np.sort(idx, axis=1)          # sort indices per row for order-stable compare


def _engine_cosine_topk(df: pl.DataFrame) -> np.ndarray:
    out = (
        df.lazy()
        .with_columns(pl.col("emb").metal.cosine_topk(_CORPUS, k=_VEC_K).alias("h"))
        .collect(engine=_ENGINE)
    )
    hits = np.asarray(out["h"].to_list(), dtype=np.int64)
    return np.sort(hits, axis=1)


def _check_topk(engine_out, cpu_out) -> None:
    # Compare the SETS of returned indices per row (cosine ties may reorder).
    assert engine_out.shape == cpu_out.shape
    for i in range(engine_out.shape[0]):
        assert set(engine_out[i].tolist()) == set(cpu_out[i].tolist()), f"row {i}"


ENTRIES += [
    BenchEntry(
        name="cosine_topk",
        category="vector-search",
        sizes=[1_000, 100_000],
        make_input=_make_queries,
        engine_fn=_engine_cosine_topk,
        cpu_fn=_cpu_cosine_topk,
        ceiling_fn=None,
        check=_check_topk,
    ),
]
```

Note on `knn`: `knn` returns nearest neighbours by L2 distance; it shares the corpus/query fixture. Add a second entry mirroring `cosine_topk` but using `.metal.knn(...)` and an L2 CPU baseline (`np.argsort` of squared distances). If `knn`'s return shape differs from `cosine_topk` (verify by running it once at size 1000 in a REPL), adapt `_engine_knn`/`_cpu_knn` accordingly before adding the entry. Keep `_check_topk` as the comparator.

```python
def _cpu_knn(df: pl.DataFrame) -> np.ndarray:
    q = np.asarray(df["emb"].to_list(), dtype=np.float32)
    # squared L2 distance to each corpus row, take k smallest
    d2 = ((q[:, None, :] - _CORPUS[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d2, axis=1)[:, :_VEC_K]
    return np.sort(idx, axis=1)


def _engine_knn(df: pl.DataFrame) -> np.ndarray:
    out = (
        df.lazy()
        .with_columns(pl.col("emb").metal.knn(_CORPUS, k=_VEC_K).alias("h"))
        .collect(engine=_ENGINE)
    )
    return np.sort(np.asarray(out["h"].to_list(), dtype=np.int64), axis=1)


ENTRIES += [
    BenchEntry(
        name="knn",
        category="vector-search",
        sizes=[1_000, 100_000],
        make_input=_make_queries,
        engine_fn=_engine_knn,
        cpu_fn=_cpu_knn,
        ceiling_fn=None,
        check=_check_topk,
    ),
]
```

The `_cpu_knn` brute force is O(Q·N·D) and memory-heavy; at the smoke size (Q=1000) it is fine. At the full size (Q=100k) it may be slow — that is acceptable (CPU baseline being slow is the point). If it OOMs at 100k, chunk the query loop; do not reduce the corpus.

- [ ] **Step 2: Run the smoke gate (vector-search)**

Run: `pytest tests/bench/m8_report/test_smoke.py -v -k "cosine_topk or knn"`
Expected: PASS (engine top-k set == CPU top-k set at Q=1000)

If knn's actual return differs from the assumed `Array`/list-of-indices shape, fix `_engine_knn` to match its real output (inspect with a one-off `python -c`), then re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/bench/m8_report/registry.py
git commit -m "M8: add vector-search category (cosine_topk, knn) vs numpy brute force"
```

---

## Task 8: Add FFT + DTW categories

**Files:**
- Modify: `tests/bench/m8_report/registry.py`

- [ ] **Step 1: Add FFT entries (engine `.metal.fft` vs numpy.fft)**

```python
def _make_fft_signal(n: int, seed: int = 0xFF7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"sig": rng.standard_normal(n).astype(np.float32)})


def _engine_fft(df: pl.DataFrame):
    return (
        df.lazy()
        .with_columns(pl.col("sig").metal.fft().alias("spec"))
        .collect(engine=_ENGINE)
    )


def _cpu_fft(df: pl.DataFrame):
    spec = np.fft.fft(df["sig"].to_numpy().astype(np.float64))
    return pl.DataFrame({"spec": spec})


def _check_fft(engine_out, cpu_out) -> None:
    # engine "spec" column dtype: inspect once. If it's a struct/list of (re, im)
    # or interleaved, reduce both to complex arrays before comparing magnitudes.
    e = engine_out["spec"]
    c = cpu_out["spec"].to_numpy()
    # Convert engine output to a complex numpy array `ev` (shape-dependent — see note).
    ev = _engine_fft_to_complex(e)
    np.testing.assert_allclose(np.abs(ev), np.abs(c), rtol=1e-2, atol=1e-1)


def _engine_fft_to_complex(series) -> np.ndarray:
    """Adapt the engine fft output Series to a complex numpy array.

    The engine fft returns planar (SoA) output. Inspect the actual Series dtype
    once (python -c "...fft()...; print(out.schema)") and implement the exact
    conversion. Likely shapes:
      - a struct column {re: F32, im: F32}  -> ev = re + 1j*im
      - two columns spec_re / spec_im       -> rename the entry's alias accordingly
    Pick the real one; this stub documents the contract, not a guess.
    """
    raise NotImplementedError("fill in after inspecting fft() output schema")


ENTRIES += [
    BenchEntry(
        name="fft",
        category="fft",
        sizes=[1 << 20, 1 << 23, 1 << 25],
        make_input=_make_fft_signal,
        engine_fn=_engine_fft,
        cpu_fn=_cpu_fft,
        ceiling_fn=None,  # numpy IS the bar; raw MLX fft broken >2^20
        check=_check_fft,
    ),
]
```

**Before running:** resolve `_engine_fft_to_complex` by inspecting the real output schema:

Run: `python -c "import numpy as np, polars as pl, polars_metal as pm; df=pl.DataFrame({'sig':np.arange(8,dtype=np.float32)}); print(df.lazy().with_columns(pl.col('sig').metal.fft().alias('spec')).collect(engine=pm.MetalEngine()).schema)"`

Implement `_engine_fft_to_complex` to match that schema (struct → `re + 1j*im`, or adjust the entry's output columns). Replace the `raise`.

- [ ] **Step 2: Add DTW entry (engine `.metal.dtw` vs dtaidistance)**

```python
_DTW_L = 256
_DTW_REF_SEED = 0xD7


def _make_dtw_seqs(n: int, seed: int = 0xD75) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    seqs = rng.standard_normal((n, _DTW_L)).astype(np.float32)
    return pl.DataFrame({"seq": seqs.tolist()}, schema={"seq": pl.Array(pl.Float32, _DTW_L)})


def _dtw_ref() -> np.ndarray:
    return np.random.default_rng(_DTW_REF_SEED).standard_normal(_DTW_L).astype(np.float32)


_DTW_REF = _dtw_ref()


def _engine_dtw(df: pl.DataFrame) -> np.ndarray:
    out = (
        df.lazy()
        .with_columns(pl.col("seq").metal.dtw(_DTW_REF, window=16).alias("d"))
        .collect(engine=_ENGINE)
    )
    return out["d"].to_numpy()


def _cpu_dtw(df: pl.DataFrame) -> np.ndarray:
    from dtaidistance import dtw

    seqs = np.asarray(df["seq"].to_list(), dtype=np.float64)
    ref = _DTW_REF.astype(np.float64)
    # window semantics: engine window=w  <->  dtaidistance window=w+1 (see dtw memory)
    return np.array([dtw.distance(s, ref, window=16 + 1) for s in seqs])


ENTRIES += [
    BenchEntry(
        name="dtw",
        category="dtw",
        sizes=[1_000, 50_000],
        make_input=_make_dtw_seqs,
        engine_fn=_engine_dtw,
        cpu_fn=_cpu_dtw,
        ceiling_fn=None,
        check=lambda e, c: np.testing.assert_allclose(e, c, rtol=1e-2, atol=1e-2),
    ),
]
```

- [ ] **Step 3: Run the smoke gate (fft + dtw)**

Run: `pytest tests/bench/m8_report/test_smoke.py -v -k "fft or dtw"`
Expected: PASS. (FFT magnitude matches numpy within tol; DTW distance matches dtaidistance within tol.)

If `dtaidistance` is not installed: `pip install dtaidistance` (it is the M6 A4 oracle, already a dev dep — confirm with `python -c "import dtaidistance"`).

- [ ] **Step 4: Commit**

```bash
git add tests/bench/m8_report/registry.py
git commit -m "M8: add fft (vs numpy.fft) + dtw (vs dtaidistance) categories"
```

---

## Task 9: Add corr category

**Files:**
- Modify: `tests/bench/m8_report/registry.py`

- [ ] **Step 1: Add corr entries (engine `lf.metal.corr` vs `df.corr()`)**

Reuse the proven shape from `tests/bench/bench_corr.py::_make_df`.

```python
def _make_corr_df(n: int, p: int, seed: int = 0xC1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p)).astype(np.float32)
    return pl.DataFrame(x, schema=[f"c{i}" for i in range(p)])


def _corr_entry(p: int) -> BenchEntry:
    return BenchEntry(
        name=f"corr_p{p}",
        category="corr",
        sizes=[100_000, 1_000_000],
        make_input=lambda n, p=p: _make_corr_df(n, p),
        engine_fn=lambda df: df.lazy().metal.corr(force_gpu=True).collect(engine=_ENGINE),
        cpu_fn=lambda df: df.corr(),
        ceiling_fn=None,
        check=_frame_allclose,
    )


ENTRIES += [_corr_entry(10), _corr_entry(50)]
```

Note: `df.corr()` and the engine corr both return a p×p F32 matrix frame with identical column names, so `_frame_allclose` applies directly. corr at small N may differ more in the last digits; if the smoke check fails at N=100k, loosen this entry's tolerance by giving it `check=lambda e, c: _frame_allclose(e, c, rtol=1e-2, atol=1e-2)` — but try the default first.

- [ ] **Step 2: Run the smoke gate (corr)**

Run: `pytest tests/bench/m8_report/test_smoke.py -v -k corr`
Expected: PASS (engine corr matrix == df.corr() within tol at N=100k)

- [ ] **Step 3: Commit**

```bash
git add tests/bench/m8_report/registry.py
git commit -m "M8: add corr category (p=10,50) vs df.corr()"
```

---

## Task 10: Add Track B category (dt + integer reductions)

**Files:**
- Modify: `tests/bench/m8_report/registry.py`

- [ ] **Step 1: Add dt + int-reduction entries**

```python
def _make_datetimes(n: int, seed: int = 0xD7E) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    # epoch-ms spread across ~40 years
    ms = rng.integers(0, 1_262_304_000_000, size=n)
    return pl.DataFrame({"ts": ms}).with_columns(
        ts=pl.col("ts").cast(pl.Datetime(time_unit="ms"))
    )


def _make_ints(n: int, seed: int = 0x1A7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"v": rng.integers(-1_000_000, 1_000_000, size=n).astype(np.int32)})


ENTRIES += [
    BenchEntry(
        name="dt_year",
        category="temporal-int",
        sizes=[1_000_000, 10_000_000, 50_000_000],
        make_input=_make_datetimes,
        engine_fn=lambda df: df.lazy().with_columns(y=pl.col("ts").dt.year()).collect(engine=_ENGINE),
        cpu_fn=lambda df: df.lazy().with_columns(y=pl.col("ts").dt.year()).collect(),
        ceiling_fn=None,
        check=_frame_allclose,
    ),
    BenchEntry(
        name="int_sum",
        category="temporal-int",
        sizes=[1_000_000, 10_000_000, 100_000_000],
        make_input=_make_ints,
        engine_fn=lambda df: df.lazy().select(s=pl.col("v").sum()).collect(engine=_ENGINE),
        cpu_fn=lambda df: df.lazy().select(s=pl.col("v").sum()).collect(),
        ceiling_fn=None,
        check=_frame_allclose,
    ),
]
```

Note: `dt.year()` returns Int32, `int_sum` returns a 1-row Int64. `_frame_allclose` handles both (integer columns are `np.number`). int dtypes compare exactly within the default rtol/atol.

- [ ] **Step 2: Run the smoke gate (temporal-int)**

Run: `pytest tests/bench/m8_report/test_smoke.py -v -k "dt_year or int_sum"`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/bench/m8_report/registry.py
git commit -m "M8: add temporal-int category (dt.year, int sum)"
```

---

## Task 11: Add conformance-only losers (TPC-H Q1/Q6, bare reduction)

**Files:**
- Modify: `tests/bench/m8_report/registry.py`

- [ ] **Step 1: Locate the existing TPC-H fixtures**

Run: `grep -n "def " tests/bench/_canonical_q1_fixture_f32.py tests/bench/_q6_fixture_f32.py | head`
Expected: shows the fixture builders + the query exprs (e.g. `make_lineitem_f32`, `q1_expr` / `q6_expr` — note the exact names returned).

- [ ] **Step 2: Add loser entries reusing those fixtures**

Using the actual function names found in Step 1 (shown here as `make_q1_frame_f32` / `apply_q1` and `make_q6_frame_f32` / `apply_q6` — substitute the real names):

```python
from tests.bench._canonical_q1_fixture_f32 import (  # adjust to real names
    apply_q1,
    make_q1_frame_f32,
)
from tests.bench._q6_fixture_f32 import apply_q6, make_q6_frame_f32  # adjust


ENTRIES += [
    BenchEntry(
        name="tpch_q1",
        category="conformance-loser",
        sizes=[10_000_000],
        make_input=make_q1_frame_f32,
        engine_fn=lambda df: apply_q1(df.lazy()).collect(engine=_ENGINE),
        cpu_fn=lambda df: apply_q1(df.lazy()).collect(),
        ceiling_fn=None,
        check=_frame_allclose,
    ),
    BenchEntry(
        name="tpch_q6",
        category="conformance-loser",
        sizes=[10_000_000],
        make_input=make_q6_frame_f32,
        engine_fn=lambda df: apply_q6(df.lazy()).collect(engine=_ENGINE),
        cpu_fn=lambda df: apply_q6(df.lazy()).collect(),
        ceiling_fn=None,
        check=_frame_allclose,
    ),
    BenchEntry(
        name="bare_sum_f32",
        category="conformance-loser",
        sizes=[1_000_000, 100_000_000],
        make_input=lambda n: pl.DataFrame(
            {"x": np.random.default_rng(0xBA5).standard_normal(n).astype(np.float32)}
        ),
        engine_fn=lambda df: df.lazy().select(s=pl.col("x").sum()).collect(engine=_ENGINE),
        cpu_fn=lambda df: df.lazy().select(s=pl.col("x").sum()).collect(),
        ceiling_fn=None,
        # bare F32 sum at 1e8 magnitude diverges in low digits (known: prop_gpu_sum_f32
        # 1e11 flake). Use a relative tolerance scaled to the sum magnitude.
        check=lambda e, c: _frame_allclose(e, c, rtol=1e-2, atol=1.0),
    ),
]
```

If the TPC-H Q1 output column order/names differ between engine and CPU (groupby ordering), the smoke check may fail on column comparison. If so, sort both outputs by their group key before comparing inside a bespoke `check` for `tpch_q1` (e.g. `lambda e, c: _frame_allclose(e.sort(e.columns[0]), c.sort(c.columns[0]))`).

- [ ] **Step 3: Run the smoke gate (losers)**

Run: `pytest tests/bench/m8_report/test_smoke.py -v -k "tpch or bare"`
Expected: PASS (engine output == CPU output — these are *correct* but slow; the report will show the 🔴 loss verdict on timing, not correctness).

- [ ] **Step 4: Commit**

```bash
git add tests/bench/m8_report/registry.py
git commit -m "M8: add conformance-loser category (TPC-H Q1/Q6, bare F32 sum)"
```

---

## Task 12: Full gate run — confirm the whole smoke suite is green

**Files:** none (verification task)

- [ ] **Step 1: Run the entire smoke + harness suite**

Run: `pytest tests/bench/m8_report/ -v`
Expected: PASS for every entry across all categories + all harness unit tests. This proves the engine path equals the CPU path on the full op surface at smallest sizes.

- [ ] **Step 2: Run `make lint`**

Run: `make lint`
Expected: clean (ruff + clippy + fmt). Fix any ruff findings in the new modules (unused imports, line length) before proceeding.

- [ ] **Step 3: Commit any lint fixes**

```bash
git add -A && git commit -m "M8: lint fixes across the report harness" || echo "nothing to fix"
```

---

## Task 13: Run the real report + write the workload-mapping verdict

**Files:**
- Generate: `docs/perf-report.md`, `perf-report.json`
- Modify: `docs/perf-report.md` (append verdict + reconciliation by hand)

- [ ] **Step 1: Generate the report (the full timed sweep)**

Run: `make perf-report`
Expected: prints a per-row table to stdout and writes `docs/perf-report.md` + `perf-report.json`. This is slow (large sizes × 7 iters across all ops) — minutes, not seconds. Capture the stdout.

- [ ] **Step 2: Sanity-check the generated tables**

Run: `head -60 docs/perf-report.md`
Expected: environment header populated (machine, polars/mlx versions), executive scorecard with the ≥10× count, per-category tables with engine ×CPU and verdict columns filled.

- [ ] **Step 3: Write the survey-reconciliation table**

Replace the `<!-- VERDICT: filled in Task 13 ... -->` marker in `docs/perf-report.md` with a reconciliation section. For each op whose measured engine ×CPU differs from a previously-claimed figure, add a row mapping claim → measured → reason. Source the prior claims from CLAUDE.md (the M4 survey block) and the memory notes. Concrete template (fill with the run's real numbers):

```markdown
## Survey reconciliation

| op | previously claimed | measured (engine ×CPU) | why the gap |
|---|---|---|---|
| haversine | 22× (M4 survey) | <measured> | survey was engine-path; confirm/adjust |
| black_scholes | 28× | <measured> | — |
| fft | 77× (raw MLX vs numpy) | <measured> | claim was raw-MLX-vs-numpy; engine adds planar host I/O |
| dt_year | 30–40× (survey) | <measured> | dt is bandwidth-shaped; Polars CPU is SIMD-fast |
| corr_p50 | 7.8× (survey) / ~9.9× (M6) | <measured> | — |
| dtw | 13.4× (vs dtaidistance) | <measured> | — |
| tpch_q1/q6 | 2.8–19.6× SLOWER | <measured> | bandwidth-bound, expected loss |
```

- [ ] **Step 4: Write the mission verdict + workload-mapping synthesis**

Append the §5-block-5 section to `docs/perf-report.md`. Three parts, grounded in the measured rows:

```markdown
## Mission verdict & workload map

### Mission verdict
- Order-of-magnitude bar (≥10× vs Polars CPU) cleared by: <list the ops + sizes from the scorecard>.
- Ties / losses: <list>, all bandwidth-shaped (TPC-H, bare reductions, dt/int) — consistent with the roofline.
- Is the bar still right? <one honest paragraph>.

### Workloads we can win at today
For each measured win, name the real-world data challenge it serves:
- **F32 transcendental feature pipelines** (finance Black-Scholes, geo haversine, scientific) → fusion chains, measured <X>× at <N>.
- **Embedding similarity / retrieval** → vector search (cosine_topk/knn), measured <X>×.
- **Spectral / signal batch** → FFT, measured <X>×.
- **Time-series alignment** → DTW, measured <X>×.
- **Correlation / covariance analytics** → corr, measured <X>×.
Note size dependence and the tax each carries (from the tax column where measured).

### Where the user still has to think too hard
- <bandwidth-shaped ops, mid-pipeline CPU-fallback boundaries — name them>.

### Next-direction trigger (not a decision)
Does any winning workload's natural query shape want to cross a join (or other
currently-CPU-fallback) boundary mid-pipeline? <yes/no + which>. If yes → the next
milestone should build a mixed-pipeline crossing-tax benchmark and reconsider GPU
joins (to keep the pipeline resident and erase the fold-back/re-ingest copies). If
no → joins stay deferred. State the trigger; do not pre-decide it.
```

- [ ] **Step 5: Commit the report**

```bash
git add docs/perf-report.md perf-report.json
git commit -m "M8: generate honest perf report + workload-mapping verdict"
```

- [ ] **Step 6: Final gate**

Run: `make gate`
Expected: green (the slow `perf-report` is NOT part of `gate`; the smoke gate under `test-unit` is). Confirm conformance shows only the documented pre-existing deviations (Mean F32→F32, prop_gpu_sum_f32 1e11 flake, lazyframe/group_by).

---

## Self-review notes (addressed)

- **Spec coverage:** §3 four modules → Tasks 1,2,3,4 (+ run.main Task 5). §4 eight categories → Tasks 2 (fusion), 6 (rolling), 7 (vector), 8 (fft/dtw), 9 (corr), 10 (temporal-int), 11 (losers). §5 report blocks → Task 4 (header/scorecard/tables) + Task 13 (reconciliation/verdict). §6 reproducibility → Task 5 (`make perf-report`). §7 testing → Task 1 (measure), 4 (emit), 3+all (smoke). §8 guardrails: no new ops (✓), reuse fixtures (✓ imports), one timing path (✓ `_timing.measure`), ingest included (✓ engine_fn closures collect through engine), losers in (✓ Task 11).
- **Sequencing:** safety net (smoke gate, Task 3) lands before categories 6–11 extend it, per spec §6.
- **Known fill-ins flagged, not hidden:** fft output-schema adapter (Task 8 Step 1, with the exact inspection command), knn return shape (Task 7), TPC-H fixture names (Task 11 Step 1, with the grep to resolve them). Each has a concrete resolution command, not a vague "TBD".
