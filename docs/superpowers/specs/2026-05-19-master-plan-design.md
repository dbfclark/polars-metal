# polars-metal master plan: skeleton → narrow-but-fast (M0–M3)

**Status.** Draft, approved 2026-05-19.
**Scope of this document.** The master plan, not any milestone's design.

## What this plan covers

The path from empty repo to **"hash groupby on GPU strictly beats CPU Polars on M2 Ultra"** — CLAUDE.md roadmap items 1–5, organized into four milestones (M0–M3). After M3 we re-plan the next slice.

Each milestone is a separate spec → implementation plan → ship cycle. This document is the index; each milestone produces its own design doc under `docs/superpowers/specs/` when its turn comes.

## Out of scope (defer to post-M3 planning)

- Radix sort, hash join, window functions, strings, regex (CLAUDE.md roadmap items 6–10).
- Public release, PyPI distribution.
- Perf parity with cuDF-Polars on a 4090. That is the mission, not the M3 bar.
- Multi-Mac portability work beyond a per-milestone gate.
- Spill-to-CPU policy.
- GitHub Actions CI. Mac runners on hosted CI are too expensive to justify at this stage; gates run locally.

## Constraints carried from CLAUDE.md

All conventions in CLAUDE.md apply unchanged. In particular: no `unwrap()` outside tests, no `unsafe` outside `*-sys` crates and the buffer bridge (each with a `// SAFETY:` comment), errors propagate as `PolarsError::ComputeError` at the engine boundary, null semantics match Polars exactly. The roadmap order is binding — M0 first, M3 last, no skipping ahead.

## Milestones

### M0 — End-to-end skeleton, CPU fallback only

**Deliverable.** `df.collect(engine="metal")` runs end to end and returns results indistinguishable from CPU Polars on any Polars query — because every IR node falls back to CPU. No GPU code path is exercised yet. Python wheel buildable via `maturin develop`.

**Substance.**

- **Buffer bridge** (`polars-metal-buffer`): Arrow `Buffer` ↔ `MTLBuffer`, zero-copy where alignment permits, null-bitmap helpers (load/store 8 rows at a time, validated end-to-end), offset-buffer support, dictionary-column support. Heavy proptest coverage.
- **MLX FFI crate** (`polars-metal-mlx-sys`): bindings to one trivial MLX op end to end, sufficient to validate the chosen FFI strategy. Not yet used by the engine.
- **Engine adapter** (`polars-metal-core`): registers with Polars' plugin/engine API, accepts a physical plan, walks the IR, dispatches by node type. Every IR node currently routes to a CPU-fallback stub that round-trips the subtree through CPU Polars.
- **Kernel crate** (`polars-metal-kernels`): empty scaffolding, ready for M1.
- **Python package** (`python/polars_metal/`): minimal — imports the Rust extension and registers the engine.

**M0-defining decisions (decided in M0's design spec).**

- **MLX FFI strategy.** Spike `cxx` and "C shim + `bindgen`". Pick one, document why. Weak prior: `cxx`.
- **Polars rev pin.** Pick the Polars rev this plan will use through M3.
- **Buffer-bridge alignment policy.** When must we copy; when can we share an `MTLBuffer` over an Arrow `Buffer` directly.
- **Error propagation shape.** Path from MSL/MLX through Rust to `PolarsError::ComputeError`.
- **Per-query arena strategy.** How scratch buffers are reused across kernels within a query.

Decisions made in M0 are inherited by every subsequent milestone, so M0's spec deserves correspondingly more care.

### M1 — First real GPU path: scan / project / filter

**Deliverable.** `df.filter(...)` over numeric columns runs entirely on GPU, returns Polars-identical results on null-heavy inputs, and does not regress vs. CPU on representative inputs.

**Substance.**

- **Scan.** Polars columns → Arrow → `MTLBuffer`, zero-copy where alignment permits, copy otherwise.
- **Project.** Column selection (mostly metadata manipulation).
- **Filter.** Null-aware boolean-mask compaction. First real MSL kernel. Threadgroup sizes queried at runtime from `MTLDevice`.
- Conformance harness expanded to include filter tests; perf benchmark for filter added to the baseline.

"No regression" is the M1 bar, not "beats CPU". M2 Ultra's CPU Polars is already very fast on simple filters; the point of M1 is proving the bridge under real GPU load and validating the null-aware kernel pattern.

### M2 — Elementwise + reductions

**Deliverable.** Arithmetic, comparison, logical ops, and `sum/mean/min/max/count` aggregations run on GPU with Polars-identical null semantics.

**Substance.**

- Elementwise: arithmetic, comparison, logical, bitwise. Mostly MLX passthroughs.
- Reductions: `sum/mean/min/max/count`. Null-aware variants required.
- Type matrix: `int8/16/32/64`, `uint8/16/32/64`, `float32/64`. Bool/date/datetime handled per Polars conventions.

**M2-defining decision (in M2's design spec).** Null-aware reductions via (a) custom MSL null-aware kernel or (b) MLX op on data + separate validity reduction. Pick one after a spike; do not defer the decision into the implementation.

### M3 — Hash groupby with sum/mean/count/min/max

**Deliverable.** `df.group_by(key).agg(...)` runs on GPU and **strictly beats CPU Polars** on a documented representative analytical query on M2 Ultra. This is the "narrow-but-fast" goal.

**Substance.**

- First custom MSL kernel of real algorithmic substance.
- **Fixed-width keys only** (int / float / bool). Variable-width and multi-key deferred to post-M3.
- Algorithm: two-pass count-then-fill. Reference implementation: cuDF's hash groupby kernel in `cudf/cpp/src/`. Port; do not reinvent. Atomics tempting but two-pass is usually right.
- Aggregations: `sum/mean/count/min/max` with null propagation matching Polars.
- Threadgroup sizing portable across M1/M2/M3/M4 (queried at runtime, not hardcoded).

**Failure path.** If correctness lands but the perf gate fails, M3 is **not** complete. The explicit choice at that point is "tune (extend M3)" vs. "accept slower-than-CPU and renegotiate the gate." That decision happens in a PR description, not by drift.

## Gate criteria

Each milestone must pass these gates before the next begins. Gates run as local commands; CLAUDE.md is the source of truth for exact invocations.

| Gate | M0 | M1 | M2 | M3 |
|---|---|---|---|---|
| **Conformance.** Polars' Python test suite under `engine="metal"`, with per-milestone skip/xfail registry. A test passing on CPU and failing on Metal is a release-blocker. | ✓ | ✓ | ✓ | ✓ |
| **Differential.** proptest (Rust) and Hypothesis (Python) against CPU Polars, byte-equal on null-heavy / empty / single-row / random inputs for every op the milestone claims. | ✓ | ✓ | ✓ | ✓ |
| **Portability.** Full local suite passes on small M2 and M1 in addition to M2 Ultra. Run manually at milestone boundary, before declaring done. | ✓ | ✓ | ✓ | ✓ |
| **Lint/format.** `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check`, `ruff check python/`, `ruff format --check python/`. | every commit | every commit | every commit | every commit |
| **Perf-regression.** criterion + pytest-benchmark, tracked against a checked-in baseline. | informational | ✓ | ✓ | ✓ |
| **Decisive perf.** Documented benchmark query strictly faster than CPU Polars on M2 Ultra. | — | — | — | **✓** |
| **cuDF-Polars diff-test** on AWS NVIDIA instance, manual cadence. | — | — | — | **✓** |

## Cross-cutting tracks

Built incrementally alongside milestones; used by every milestone after their introduction.

**Conformance harness (M0 build).** Python harness running Polars' own test suite with `engine="metal"`. Per-test skip/xfail registry checked into the repo; adding a skip requires a PR. As GPU paths land, items move out of skip — this is the visible measure of coverage growth.

**Local gates (no hosted CI).** All gates run as local commands. The user runs the portability gate on small M2 and M1 at milestone boundaries; Claude reminds before declaring a milestone done. Mac runners on GitHub Actions are too expensive to justify at this stage of the project.

**Benchmark harness (M0 stub, M1 first real benchmark, M3 decisive).** criterion at the Rust crate level for per-kernel microbenches. pytest-benchmark at the Python level for end-to-end queries. Numbers tracked across commits via a checked-in baseline file so regressions surface in diff review.

**cuDF-Polars diff-test (M3 build, on-demand).** Scripted run on an AWS NVIDIA instance. Same Polars query against cuDF-Polars and against the M2 Ultra build, asserts byte-equal output. Manual cadence — invoked at M3's gate, not per-commit.

**Polars rev pinning policy.** Pin to a known-good rev in `Cargo.toml` chosen during M0. Bump only at milestone boundaries. Every bump runs the full conformance + diff-test sweep before merge. Plugin API changes handled per-incident — adapt locally; upstream a fix if the change is unintentional.

**Documentation (continuous).** `docs/architecture.md`, `docs/kernel-authoring.md`, and `docs/open-questions.md` (all named in CLAUDE.md) updated as decisions land in each milestone. Per-milestone design docs live in `docs/superpowers/specs/`.

## Risks & open questions

Each milestone's spec addresses the risks owned by it. The rest are tracked in `docs/open-questions.md`.

**Plugin API stability.** Polars' engine plugin surface is not stable across versions. Mitigation: pin rev; bump only at milestone boundaries. *Owned by M0.*

**Buffer-bridge null-bitmap correctness.** Arrow's bit-packed little-endian validity layout is fiddly; a wrong helper here is inherited by every later kernel. Mitigation: aggressive proptest coverage in M0, beyond what the surface area would normally warrant. *Owned by M0.*

**MLX null story.** MLX is dense-numeric-first with no first-class nulls. M2's reductions need either a custom MSL null-aware kernel or MLX op + separate validity reduction. Mitigation: pick after a spike in M2's spec. *Owned by M2.*

**Hash groupby GPU performance.** The biggest single unknown in M0–M3. Two-pass count-then-fill is the textbook approach; cuDF is the reference. Mitigation: M3's spec includes "read cuDF's hash groupby kernel first" as a prerequisite task. *Owned by M3.*

**Memory pressure on base M1.** M2 Ultra has plenty of memory; M1 does not. Mitigation: no explicit spill-to-CPU in M0–M3, but portability gate catches OOM regressions. *Tracked in `docs/open-questions.md` from M0 onward.*

**FFI to MLX.** Open until M0's spike resolves it. Top two candidates: `cxx` crate (best ergonomics for C++-shaped APIs, more build complexity) vs. hand-written C shim + `bindgen` (more control, linear maintenance cost in op coverage). Switching cost grows with kernel count, so picking before kernels are written matters. Weak prior: `cxx`. *Owned by M0.*

## Working with Claude on this plan

This project is staffed primarily by Claude under direction. Each milestone's implementation plan should bias toward:

- **Short, well-specified sub-tasks** — one kernel, one type, one null variant at a time.
- **Strong test gates per sub-task** — Polars conformance + differential proptest are the safety net that lets Claude work confidently.
- **CLAUDE.md's "Working with Claude Code in this repo" section applies in full**: read the matching cuDF kernel before writing MSL, check MLX before writing custom, do not speculatively optimize, do not add files to `shaders/` without a matching test in `tests/kernel/`.

## What this document does not do

- It does not specify any milestone's design — that is the per-milestone spec's job.
- It does not specify any milestone's implementation plan — that is the per-milestone plan's job.
- It does not lock gate commands — CLAUDE.md is the source of truth for exact commands.
- It does not commit to a calendar — milestones complete when their gates pass.
