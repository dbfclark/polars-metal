# M8 — Honest Perf Report (design)

Date: 2026-06-12
Branch: `m8-honest-perf-report` (off `main`; M0–M7 all merged)
Predecessor: M7 (PR #7) — consolidation & hardening (namespace spine, `udf.rs` decomposition, differential harness)
Seed: `docs/superpowers/specs/2026-06-11-m6-consolidation-audit.md` (§4, "Prove it" fork)

## 1. Purpose

M4–M6 piled up perf claims fast, and the honest-perf reconciliation throughout M6 showed
the **engine-path numbers run well below the survey's headline figures** — the survey
compared raw MLX/NumPy, while the real `engine="metal"` path pays host→Metal ingest +
fold-back. Examples already known: FFT measured **3–4.6×** vs the survey's **77×**; dt
**10–27×** vs **30–40×**; corr **~9.9×** vs the survey's **7.8×** (corr went *up*). No single
rigorous, reproducible pass has ever confronted this gap across the whole engine.

M8 is that pass. It is a **measurement & validation milestone**: no new ops, no new kernels.
The deliverable is one committed, regenerable **honest perf report** that measures every
shipped op end-to-end, leads with the mission's literal bar (faster than Polars CPU), and
makes the ingest/fold-back tax visible as an explicit column. A number you cannot reproduce
is not proof; the whole milestone is about producing numbers we can stand behind.

This scope was chosen deliberately (brainstorm 2026-06-12): on the M8 fork the architect chose
**"Prove it / release" → "Honest perf report"** over new compute-bound ops, the `build_index()`
statefulness subsystem, or coverage breadth.

**The report is internal decision-input, not a published artifact.** Its explicit job is to
answer the architect's next-direction question: **do we pick up bandwidth-shaped work
(joins, etc.) or stay compute-focused?** CLAUDE.md currently has hash join "deferred
indefinitely unless a non-TPC-H workload demands it"; the architect wants to revisit that on
data. We cannot benchmark a join we have not built — but the report's **bandwidth-vs-compute
scorecard is the evidence base for that call.** Our existing bandwidth-shaped ops (TPC-H
Q1/Q6, bare reductions) already lose 2.8–19.6×, and a hash join is the same roofline shape;
the report makes that current and concrete so the join decision is made on measurement, not
vibes. Presentation is tuned for one reader (the architect) — clarity over polish, no
marketing framing.

## 2. Decisions locked in the brainstorm

1. **Headline baseline = both columns, side by side.** Per op, report **engine vs Polars CPU**
   (the mission bar: "faster than Polars CPU by an order of magnitude") *and* **engine vs raw
   MLX/NumPy ceiling** (the survey's framing). The gap between them is the ingest/fold-back
   tax, surfaced as its own column. For `.metal` verbs with no Polars-native equivalent (fft,
   dtw, knn) the CPU column falls back to the **idiomatic CPU tool a user would actually reach
   for** (numpy/scipy/dtaidistance); corr uses real `df.corr()`.
2. **Coverage = everything, losers included.** Every shipped op — M4 fusion chains, M5 rolling,
   all M6 namespace verbs, Track B int/dt — **and** the conformance-only losers (TPC-H Q1/Q6,
   bare reductions). A loss documented honestly is as valuable as a win. This is the complete
   scorecard, not the marketing suite.
3. **Approach = unified report-generating harness** (over completing the scattered benches in
   place, or a one-shot script). Reproducibility is the point: a registry + regenerable report
   is the only approach where re-running yields the same report and every op goes through
   identical timing rules.

## 3. Architecture

New harness under `tests/bench/m8_report/`, four pieces, each one job:

```
tests/bench/m8_report/
  _timing.py      # shared timing core — the single source of timing truth
  registry.py     # the op registry: list of BenchEntry, pure data + callables, no logic
  run.py          # the driver: iterate registry, time each callable, collect rows
  emit.py         # rows -> docs/perf-report.md + perf-report.json
```

**The seam that matters: timing logic lives in exactly one place (`_timing.py`), data lives in
`registry.py`, and the report is a pure function of the collected rows.** That is what makes
the numbers reproducible and cross-op comparisons fair.

### 3.1 `_timing.py` — shared timing core

One function: `measure(fn, *, warmup, iters) -> Stats`. Runs `warmup` discarded calls, then
`iters` timed calls, returns `Stats(median, min, p90)`. Every callable in the report goes
through this one function, so timing rules are identical across all ops.

- **The engine path is timed as a closure that includes host→Metal ingest, compute, and
  fold-back — no carve-outs.** That tax is the whole point of the report.
- `warmup`/`iters` are report-level constants (proposed `warmup=2, iters=7`, median reported),
  recorded in the report header so the methodology is self-describing.

### 3.2 `registry.py` — the op registry

A flat list of `BenchEntry` dataclasses. Each is pure data + up to three callables:

```python
BenchEntry(
    name="haversine", category="fusion-chain",
    sizes=[1_000_000, 10_000_000, 100_000_000],
    make_input=lambda n: ...,                                    # fixture builder
    engine_fn=lambda df: df.lazy()...collect(engine="metal"),   # full wall-clock, REQUIRED
    cpu_fn=lambda df: df.lazy()...collect(),                     # Polars CPU / best-CPU, REQUIRED
    ceiling_fn=lambda arr: mlx_haversine(arr),                   # raw MLX/numpy, OPTIONAL
)
```

Reuses existing fixtures rather than rebuilding them: `m4_engine/` closures feed `engine_fn`,
`m4_survey/` closures feed `ceiling_fn`. `ceiling_fn` is `None` where no meaningful raw form
exists (numpy already *is* the bar for FFT; dt/int are bandwidth-shaped with no raw ceiling).

### 3.3 `run.py` — the driver

Pure orchestration. For each entry × each size: build the input once, run `measure()` on each
non-`None` callable, assemble a result row:

```
(name, category, size, engine_ms, cpu_ms, ceiling_ms,
 engine_vs_cpu, ceiling_vs_cpu, tax, verdict)
```

No timing logic of its own (delegates to `_timing.measure`). No formatting (delegates to
`emit`).

### 3.4 `emit.py` — artifact emitters

Takes the rows, writes `docs/perf-report.md` (human) and `perf-report.json` (machine-readable,
for future regression diffing). Zero measurement, zero timing.

## 4. Coverage (the registry contents)

Eight categories. `cpu_fn` is the mission baseline (Polars CPU where a native expression
exists, else the idiomatic CPU tool); `ceiling_fn` optional.

| Category | Ops | engine_fn | cpu_fn (mission baseline) | ceiling_fn |
|---|---|---|---|---|
| **Fusion chains (M4)** | haversine, black-scholes, std/var, cumsum, sort, top-k | `collect(engine="metal")` | Polars CPU | raw MLX |
| **Rolling (M5)** | rolling_{mean,sum,var,std} × window sweep | engine path | Polars CPU `.rolling_*` | MLX/numpy |
| **Vector search (M6)** | cosine_topk, knn | `.metal.cosine_topk/.knn` | numpy/sklearn brute-force | raw MLX |
| **FFT (M6)** | fft, 2²⁰…2²⁵ | `.metal.fft` | `numpy.fft` | — (numpy *is* the bar) |
| **DTW (M6)** | dtw | `.metal.dtw` | dtaidistance | raw MSL kernel (no host I/O) |
| **Corr (M6)** | corr, p sweep | `lf.metal.corr()` | `df.corr()` / `numpy.corrcoef` | raw MLX standardize+GEMM |
| **Temporal + int (Track B)** | dt.{year,month,day}, int sum/min/max | engine path | Polars CPU | — (bandwidth-shaped) |
| **🔴 Conformance-only losers** | TPC-H Q1, Q6, bare reductions | engine path | Polars CPU | — |

Each op sweeps sizes (e.g. fusion chains at 1M/10M/100M) because several ops **change verdict
with N** — that crossover is itself a finding (bandwidth-shaped ops win only at scale, or never).

## 5. The report artifact (`docs/perf-report.md`)

Self-describing, five blocks:

1. **Header** — auto-captured: date, chip (M2 Ultra / cores / RAM), OS, Polars + MLX + Python
   versions, methodology (`warmup=2, iters=7, median reported, engine path includes
   ingest+fold-back`). Numbers are machine-specific; the header names the machine.

2. **Executive scorecard** — the mission bar up front: count of ops clearing **≥10× vs CPU**
   (the "order of magnitude" claim), count clearing any win (>1×), count tying/losing.

3. **Per-category tables**, columns:

   | op | size | engine ms | CPU ms | **engine ×CPU** | ceiling ms | ceiling ×CPU | **tax** | verdict |

   - **engine ×CPU** = `CPU_ms / engine_ms` — the mission number.
   - **ceiling ×CPU** = `CPU_ms / ceiling_ms` — the survey's theoretical max.
   - **tax** = `engine_ms / ceiling_ms` (≥1) — the ingest/fold-back multiplier the survey never
     showed (how much the FFI/host path costs vs the raw kernel).
   - **verdict** — ✅ ≥10× / 🟢 win (>1×) / 🟡 tie / 🔴 loss `(bandwidth-bound)`.

4. **Survey reconciliation** — a table mapping each op's *previously-claimed* figure
   (CLAUDE.md / survey / memory) → *measured* engine ×CPU, one line per gap. Example: "FFT:
   claimed 77× was raw-MLX-vs-numpy; measured 3–4.6× engine path, tax ≈1.5× from planar host I/O."

5. **Mission verdict + next-direction read (prose)** — where we clear the order-of-magnitude
   bar, where we tie/lose, and an honest read on whether the bar is still the right bar for
   this workload class. **Then the decision the report exists to inform:** the
   bandwidth-vs-compute split as measured, and what it implies for whether joins (and other
   bandwidth-shaped work) are worth building. The TPC-H Q1/Q6 + bare-reduction losses are the
   join proxy — if those lose by 3–20× on the same roofline a hash join would sit on, that is
   the data point. State it plainly so the join go/no-go is answerable from this report.

## 6. Reproducibility

`make perf-report` → `run.py` → `emit.py`. Hardware + library versions captured into the report
header automatically, so the artifact is self-describing and a re-run on the same machine
reproduces it. The JSON twin enables future report-to-report diffing.

## 7. Testing strategy

The **numbers** are not testable (they are measurements); the **machinery** is. Three nets:

- **`measure()` unit test** — feed a known-duration fn (e.g. `time.sleep`-based), assert the
  returned median is within tolerance and warmup runs are excluded.
- **`emit.py` unit test** — feed synthetic rows, assert well-formed markdown + valid JSON with
  the expected columns/keys.
- **Smoke + correctness gate** — run every registry entry at its **smallest size with
  `iters=1`**, asserting (a) no exception and (b) **engine output equals CPU output** (byte-exact
  where order is defined, within documented tolerance for float reductions per the existing
  conformance deviations). A fast wrong answer is not a win, so the report harness doubles as a
  differential check on the full op surface.

The smoke+correctness gate runs in `make test-unit` (fast, smallest sizes). The **full timed
report (`make perf-report`) stays out of `make gate`** — it is slow (large sizes × iters) and
its output is a measurement artifact, not a pass/fail. Per CLAUDE.md: a perf regression is not a
failing test; a correctness regression is.

## 8. Guardrails

1. **No new ops, no new kernels.** M8 measures the existing surface; it does not extend it.
2. **Reuse existing fixtures.** `m4_survey/` and `m4_engine/` fixtures feed the registry; do not
   rebuild input generators that already exist.
3. **One timing path.** All callables go through `_timing.measure`. No per-op bespoke timing.
4. **Engine path includes ingest + fold-back, always.** No carve-outs that flatter the number.
5. **Losers stay in.** TPC-H Q1/Q6 and bare reductions are measured and reported as losses; the
   report's honesty depends on it.

## 9. Out of scope (deferred)

- **Release/packaging/usability hardening** (wheel, install story, public API docs) — the other
  half of the "prove it / release" fork; not the driver here (the report is internal
  decision-input). Revisit if/when the next-direction call points at release.
- **Acting on the next-direction decision** — M8 *produces the evidence* for the joins-vs-compute
  call; actually building joins, new compute-bound ops, cooperative-wavefront DTW, or
  `build_index()` is M9+, decided after reading this report.
- **Multi-machine portability matrix** — the report is M2 Ultra primary, self-describing by
  machine; a base-M1/M2 sweep is a future add, not this milestone.

## 10. Definition of done

- `tests/bench/m8_report/` harness exists with the four pieces; `make perf-report` regenerates
  `docs/perf-report.md` + `perf-report.json`.
- The report covers every category in §4 (losers included), with both the engine-×CPU and
  ceiling-×CPU columns and the tax column populated.
- The survey-reconciliation table accounts for every previously-claimed figure.
- The smoke+correctness gate runs under `make test-unit` and is green; `make gate` green.
- The mission-verdict prose states plainly where the order-of-magnitude bar is cleared and where
  it is not.
