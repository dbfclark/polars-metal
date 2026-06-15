# M9 — Crossing-Tax Benchmark (design)

Date: 2026-06-15
Branch: `m9-crossing-tax-benchmark` (off `main`; M0–M8 merged)
Predecessor: M8 (PR #8) — the honest perf report, whose verdict fired the "resident pipeline" trigger
Seed: the M8 next-direction trigger ([[m8-perf-report-findings]]) + the ideation that followed (this conversation)

## 1. Purpose

M8 proved the engine wins an order of magnitude on the compute-bound class and loses on the
bandwidth/irregular class (TPC-H, joins) — a hardware roofline, not an effort gap. The follow-on
ideation established the real opportunity and its precise framing:

**The M-series superpower is cheap CPU↔GPU switching.** On a discrete GPU a crossing means a PCIe
transfer, so the doctrine is "stay resident, cross as little as possible" (cuDF). On Apple Silicon
the CPU and GPU read the *same* physical RAM — there is no transfer — so the doctrine flips: **route
every op to whichever processor is best for it and switch whenever it pays.** This is the project's
own M2 founding pivot ("unified memory means per-op routing has zero transfer cost"). The distinctive
capability is doing the compute-bound op on the GPU and the join/scan on the CPU *in the same query*
without paying to move between them — something neither a CPU-only nor a discrete-GPU engine can do.

**The one honest caveat** (already discovered, the StagingPool finding): the crossing is not *literally*
free. Metal's zero-copy buffer wrap (`bytesNoCopy`) needs 16 KB page alignment, but Polars Arrow
buffers are 64-byte aligned — so a crossing costs **one RAM→RAM memcpy of the crossed bytes** (not a
reinterpret, but also not a PCIe transfer). So crossing cost ≈ `bytes_crossed / bandwidth + dispatch`:
cheap for small volumes, real for large. That is *why* keeping crossing volume small (push reducers
before the crossing) matters — not to avoid crossing, but to let free routing win handily.

**M9 is a measure-first milestone — no engine changes.** Today the engine is all-or-nothing: a `Join`
node makes the walker `FallBack` and the *entire* query runs on Polars CPU (verified in
`_callback.py::execute_with_metal`), so no crossings happen today and there is nothing to measure in
situ. M9 **hand-builds** representative mixed pipelines and times candidate execution strategies to
answer one question with data, then emits a go/no-go for M10.

## 2. The central question

> On M-series, is CPU↔GPU switching cheap enough that **free per-op routing** beats both all-CPU and
> naive execution on mixed compute+join pipelines — and how small must the crossing volume be for that
> to hold?

The deliverable is not just four timings; it is a **fitted cost model** (§5) that makes the verdict
*generalize* to any pipeline from its (bytes-crossed, crossing-count) profile.

## 3. Pipelines — broad sweep across the join spectrum

Each is a realistic compute → join → compute shape. Resident-constructibility decreases down the list
(a "gather" join degenerates to `take(indices)` — GPU-able via `mx.take`; a "hash" join must discover
matches by key equality — irregular, no MLX primitive, CPU-bound).

| # | Pipeline | "Join" shape | Resident path? |
|---|----------|--------------|----------------|
| P1 | **retrieve→rerank**: `cosine_topk(q, corpus, k)` → take metadata rows by hit-index → transcendental rerank (similarity + a metadata feature) | gather by index | **Yes** (`mx.take`) |
| P2 | **fact→dim lookup**: fact table with dense `id` → gather `dim[id]` features → Black-Scholes-shaped chain on fact+dim columns | gather by dense id | **Yes** |
| P3 | **as-of join**: two time-series, match each left row to the latest right row ≤ its timestamp → compute on joined columns | sort-merge | **Partial** (sort is GPU-able; merge sequential) |
| P4 | **hash-equi-join**: relational equi-join on a key → compute chain on joined columns (TPC-H-shaped) | hash build+probe | **No** (documents the boundary) |

## 4. Execution paths (timed per pipeline)

Routing is identical in paths 2–4 (compute→GPU, join→CPU unless resident); what differs is *how* you
cross.

1. **all-CPU** — the whole pipeline in Polars CPU. *Baseline = what `engine="metal"` does today on a
   join query.*
2. **partial-naive** — compute on GPU, join on CPU, crossing *dumbly*: all columns, no pre-reduction,
   per-op `mx.eval` barriers. *Isolates the cost of crossing badly.*
3. **partial-smart (free-routing)** — same routing, crossing *minimally*: project to only-needed
   columns, push reducers (`topk`/filter) before the crossing so volume is tiny, batch the sync.
   **The "switch when convenient" thesis path.**
4. **resident** *(P1/P2 only)* — gather on GPU too (`mx.take`); a single final fold-back. *The ceiling
   for gather-shaped joins.*

**What each comparison establishes:**
- **3 vs 1** — does smart free-routing beat all-CPU? (the headline; expect yes for P1/P2, unknown for
  P3, likely no for P4)
- **3 vs 2** — how much does crossing *smart* (small volume) matter vs crossing dumb? (isolates the
  volume lever)
- **4 vs 3** *(gather)* — does keeping the gather *resident* add anything over a cheap small CPU
  crossing? (if ≈, free routing suffices even for gather — no GPU join ever needed)

## 5. Measurement — the α/β cost-model decomposition

Beyond end-to-end path timings, instrument a single crossing to fit:

> `crossing_cost ≈ α · bytes_crossed + β · n_crossings`

- **α** — the page-aligned memcpy rate (StagingPool copy-in + readback copy-out, the irreducible
  16 KB-alignment cost), measured by sweeping bytes crossed at fixed crossing count.
- **β** — the fixed per-crossing dispatch/sync overhead (kernel launch, command-buffer submit,
  `mx.eval` barrier), measured by sweeping crossing count at fixed tiny volume.

Knowing (α, β) lets us **predict** whether free routing wins for any pipeline from its (bytes,
crossings) profile — turning "we measured these four" into "here is the rule for when to route to GPU."

**Sweeps:** corpus/table size, `k`, columns crossed (volume), number of crossings. **Honest baselines:**
CPU = Polars CPU (competent, multithreaded); GPU compute = our actual `.metal` ops (`cosine_topk`,
fusion chains) + the buffer bridge / `StagingPool` for the crossing; resident gather = `mx.take`.

## 6. Components

Mirror the M8 harness; reuse what exists. No engine changes — everything is hand-built bench code.

```
tests/bench/m9_crossing/
  __init__.py
  _crossing.py    # the instrumented crossing primitive: page-aligned memcpy (reuse StagingPool) +
                  # sync; exposes the α/β probes (bytes-sweep, count-sweep)
  _pipelines.py   # P1–P4, each as 4 (or 3) path-callables that return an IDENTICAL result
  run.py          # drive pipelines × paths × sweeps -> rows; fit α, β
  emit.py         # rows + fitted cost model -> docs/crossing-tax-report.md + crossing-tax.json
  test_smoke.py   # the smoke+correctness gate (all paths agree)
  test_harness.py # unit tests for the crossing primitive + emit
```

Reuse `tests/bench/m8_report/_timing.py::measure` (the proven warmup + median-of-N harness). Reuse
`polars_metal_buffer::StagingPool` for the page-aligned crossing. GPU compute uses the existing
`.metal` namespace ops and MLX directly where a pipeline needs a raw kernel.

## 7. Testing strategy

It is a measurement, so the M8 discipline applies: a **smoke+correctness gate** asserts **all paths
produce the identical result** for each pipeline at the smallest size. This both validates the
hand-built resident/partial paths are correct *and* guarantees apples-to-apples timing — a path that
is fast but wrong is caught. Plus unit tests for the crossing primitive (α/β probes return sane
monotonic numbers on a known-size buffer) and `emit` (well-formed report + JSON from synthetic rows).

- Smoke + harness run in `make test-unit` (smallest sizes, fast).
- The full timed sweep is `make crossing-report`, **out of `make gate`** (slow; its output is a
  measurement artifact, not pass/fail).
- A perf regression is not a failing test; a correctness divergence between paths is.

## 8. Guardrails

1. **No engine changes.** M9 hand-builds bench code only; the walker/router/UDF are untouched. Building
   partial-GPU / resident execution / a boundary-aware router is M10, gated on this report.
2. **Reuse fixtures and harness.** `_timing.measure`, `StagingPool`, the `.metal` ops — do not rebuild.
3. **Honest baselines.** CPU = competent multithreaded Polars; GPU = our real ops. No strawman CPU.
   (Heed the M8 lesson: in-process CPU/BLAS baselines showed run-to-run variance — report medians,
   treat ratios as ±30%, and re-run if a number is non-monotonic in a sweep.)
4. **All paths must agree** (correctness gate) before any timing is trusted.

## 9. Definition of done

- `tests/bench/m9_crossing/` exists with the components in §6; `make crossing-report` regenerates
  `docs/crossing-tax-report.md` + `crossing-tax.json`.
- The report contains, per pipeline (P1–P4), the path timings (paths 1–4 where they exist) across the
  sweeps, **and** the fitted (α, β) cost model with the bytes/count probes behind it.
- The smoke+correctness gate (all paths agree) runs under `make test-unit` and is green; `make gate`
  green.
- A **verdict** that resolves to exactly one of:
  - **(a)** free-routing (path 3) beats all-CPU broadly → **M10 = build the boundary-aware router**;
  - **(b)** resident (path 4) ≫ smart-partial only for gather → **M10 = narrower resident-gather build**;
  - **(c)** crossings dominate irreducibly (path 3 ≤ path 1) → **drop the direction**, re-center on
    hardening existing wins.
- The verdict states the (α, β)-derived rule for *when* free routing wins (the crossing-volume
  threshold), so M10's scope follows from the cost model, not a single data point.

## 10. Out of scope (M10+ / deferred)

- **All engine changes** — the boundary-aware lifting pass / router, partial-GPU execution, a resident
  GPU gather-join. M9 sizes whether they are worth building; M10 builds the one the verdict selects.
- **A GPU hash join** — the roofline (and M3/M8) already say it loses on M-series; P4 only documents
  the boundary, it does not motivate building one.
- **Polars-optimizer changes** — we consume the post-optimization plan; we do not modify Polars'
  predicate/projection pushdown.
