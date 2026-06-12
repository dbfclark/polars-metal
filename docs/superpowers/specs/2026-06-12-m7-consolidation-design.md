# M7 — Consolidation & Hardening (design)

Date: 2026-06-12
Branch: `m7-consolidation` (off `m6-vector-search`; M7 consolidates M6 code, which is not yet merged)
Predecessor: M6 (PR #6, `m6-vector-search`) — the `.metal` namespace suite + Track B + memory pass
Seed: `docs/superpowers/specs/2026-06-11-m6-consolidation-audit.md` (§4 M7 candidate pool)

## 1. Purpose

M4–M6 piled up a lot of surface fast — 6 `.metal` namespace verbs, integer parity, the
`dt` kernel, FFT, DTW, corr, vector search — by copy-pasting the previous op's template
and never refactoring back. M7 pays that debt down. It is a **consolidation & hardening
milestone**: no new ops, no new perf kernels. The deliverable is a clean, trustworthy base
that the next flagship (M8) can build on without fighting accumulated duplication and
inconsistent contracts.

This scope was chosen deliberately (brainstorm 2026-06-12): the architect's read was
"probably sharpening and coverage; it's been a lot of code so far and we could use a big
cleanup pass," and on the explicit fork chose **pure consolidation + coverage** over
opportunistic perf or a perf flagship.

The cleanup targets are not guesses — they come from a four-slice grounded survey
(Python namespace, Rust engine, kernel/buffer/FFI, tests/gate/docs) run 2026-06-12.

## 2. What the survey found (the debt map)

The debt is **concentrated in two hotspots**; the kernel/buffer/FFI and test/doc layers are
largely healthy.

**🔴 Hotspot 1 — Python `.metal` namespace machinery.** The 6 verbs (cosine_topk, knn, fft,
dtw, corr; plus rolling/dt) each carry a near-identical copy of the detect → dispatch →
cache → serialize template:
- 4–6 detect modules of ~90–110 lines each; 3 duplicated sentinel builders; 3 identical
  cache/capture triplets; per-verb binding dataclasses; 5 copy-paste dispatch blocks in
  `__init__.py`. Estimate **≈525 lines → ≈165** via one parameterized spine + generic cache.
- Worse than duplication: **inconsistent contracts.** Null handling — vector/fft *raise*, dtw
  *masks+restores*, corr *falls back to CPU*. Boundary errors — `RuntimeError` vs
  `ComputeError` vs `ValueError`. Streaming — some raise, rolling silently returns `[]`. No
  documented per-verb contract; FFT is missing the handle-evicted guard the others have.

**🔴 Hotspot 2 — Rust `crates/polars-metal-core/src/udf.rs` (2,881 lines).** A monolith mixing
plan deserialization, predicate parsing, comparisons, groupby parsing, value-column building,
packing, and the whole groupby executor. Plus:
- **Dtype-dispatch duplication**: `build_agg_kind_and_vcol` (21 arms),
  `eval_to_metal_buffers` (10 arms), and 4 near-identical `cmp_*` fns — ≈250 lines foldable.
- The M2 per-agg groupby (conformance-only) is **tangled into the live path**.
- 5–8 `unsafe` blocks missing inline `// SAFETY:` comments (CLAUDE.md violation).
- `decide_groupby_dispatch` is a second routing point divorced from `router/cost.rs`.

**🟡 Mild — kernel/buffer/FFI.** Largely clean. Two candidates exist (FFT dual-core fold ~187
dup lines; 8 per-width MLX copy wrappers → one dispatcher) but both are **deferred to M8** —
see §5. StagingPool-only-in-dt is confirmed *correct* (rolling/dtw are genuinely zero-copy via
`MetalBuffer::from_borrowed_f32`); it just lacks a doc note.

**🟢 Healthy — tests/gate/lint/docs.** `cargo fmt`/clippy/ruff all clean at survey time.
Conformance baseline verified clean (the 8 M6 fixes hold). Real gaps: `tests/diff/` was
**retired** (no dedicated random-input differential harness remains; `make test-diff` is a
hole), and F64/integer rolling fallback is untested. Stale `test-kernel` Makefile comment.

## 3. Scope — three workstreams

### Workstream A — Python `.metal` namespace spine

**A1. Contract first (de-risks everything else).** Write down the intended per-verb contract on
three axes — **null handling, boundary error type, streaming** — and add characterization
tests pinning each verb's *current* behavior before any refactor. The goal is **intentional
and documented, not forced-uniform**: corr falling back to CPU on nulls is a legitimate choice
(pairwise-complete correlation is well-defined); dtw masking+restoring is its row-semantics;
vector/fft raising on nulls may be correct (a null in an embedding/signal is meaningless). We
correct only the *accidentally*-wrong ones, each with a test pinning the change:
- Boundary errors → `PolarsError.ComputeError` (`pl.exceptions.ComputeError`) per the engine
  conventions, replacing stray `RuntimeError`.
- Add FFT's missing handle-evicted guard (the other verbs have it).
- Decide and document streaming behavior uniformly (raise vs silent-skip).
Output: a committed **verb-contract doc** (`docs/metal-namespace-contracts.md`).

**A2. Collapse to one spine.** In `python/polars_metal/_detect_common.py`:
- one **parameterized detect factory** (binding schema = field names, sentinel tag, parser fn)
  replacing the 4–6 detect modules;
- one **generic capture-cache** (`CaptureCache`) replacing the 3 triplets;
- one **sentinel builder** parameterized by tag prefix + field list;
- a generic **`Binding`** replacing the per-verb dataclasses;
- a **loop-driven dispatch registry** in `__init__.py` replacing the 5 copy-paste blocks.

Rolling and dt detectors diverge (rolling uses `_parse_rolling_expr`; dt needs schema). Bring
them onto the spine where clean; where they genuinely need extra (dt's optional `schema`
param), the factory accepts it and other verbs ignore it. Document any deliberate divergence.

Target: ≈525 → ≈165 lines. Behavior-preserving except the A1 contract corrections.

### Workstream B — Rust `udf.rs` decomposition

**B1. Split the monolith** into focused modules under `crates/polars-metal-core/src/`:
- `parser` (plan/predicate/groupby deserialization),
- `cmp` (the comparison kernels + the generic helper from B2),
- `groupby_core` (the live fused aggregation path),
- `groupby_legacy` (the M2 per-agg conformance path, isolated *out* of the live path; shared
  value-column builders stay shared).

**B2. `per_dtype!` macro** folding the duplicated dtype-dispatch:
`build_agg_kind_and_vcol` (21 arms), `eval_to_metal_buffers` (10 arms), and the 4 `cmp_*`
functions (~250 lines). **Behavior-identical** — pure mechanical fold. Pin with characterization
tests / existing kernel tests before folding.

**B3. Tidy:** add the 5–8 missing `// SAFETY:` comments (fft.rs, udf.rs unsafe slices);
consolidate the second routing point — move `decide_groupby_dispatch` toward `router/`, or, if
it must stay at execution time (expressions have no per-agg fallback), document why at the call
site so it can't silently drift from `router/cost.rs`.

**B4. MLX FFI wrapper parametrization (pure DRY).** Collapse the 8 per-width integer copy
functions in `crates/polars-metal-mlx-sys/src/lib.rs` (`mlx_array_copy_to_i32`, `_i8`, `_i16`,
`_i64`, `_u8`, `_u16`, `_u32`, `_u64`) into one parametric `mlx_array_copy_to_dtype(arr,
dtype_tag, out_ptr, n)` FFI entry + a Rust-side dispatcher on `MlxDtype`. Reduces the FFI
boundary from 8 near-identical wrappers to 1. Behavior-identical; pin with existing
vector/fft/int-readback tests before folding. (Pulled into M7 from the M8 deferral list — it is
pure consolidation with no perf implication; the FFT dual-core fold stays in M8 because it
carries behavior risk, B4 does not.)

### Workstream C — Coverage & hardening

**C1. Restore the differential harness (highest-leverage; do first).** Build a lean
`hypothesis`-based random-input oracle sweeping the namespace verbs + fused F32/int chains
against CPU Polars (byte-exact, including null-heavy / empty / single-row inputs, per the
testing strategy). Wire `make test-diff` to actually run it — today it's a hole (`tests/diff/`
was retired). This is the safety net the A/B refactors lean on.

**C2. Rolling F64/integer = CPU-fallback (hardware constraint, not a TODO).** Apple Silicon GPUs
have **no FP64** — MSL has no `double` type (same root constraint as the lack of 64-bit atomics
/ `atomic_float`). That is *why* the whole engine is F32-first; F64 inputs route to CPU Polars
by necessity. There is no F64 rolling kernel to write (software double-double would be slower
than the CPU fallback). M7's job: add explicit tests confirming F64/integer rolling falls back
to CPU cleanly and correctly, and add a one-line note in the rolling docs that F64→CPU is a
hardware constraint, not an unfinished feature.

**C3. Doc/Makefile tidy:** document the StagingPool-only-in-dt rationale (survey confirmed
rolling/dtw are genuinely zero-copy); fix the stale `test-kernel` Makefile comment; land the
A1 verb-contract doc.

## 4. Guardrails

1. **Behavior-preserving** except where a contract is *deliberately* corrected — and every such
   correction has a test pinning the new behavior. No silent behavior changes.
2. **`make gate` green at every step.** Per the subagent-fmt-drift lesson, implementers run
   `cargo fmt` / `ruff` per-task, not only at the final gate.
3. **Do not extend conformance-only code.** Groupby / sort / TPC-H stay conformance-only (Mission
   non-goal). B2 *folds* the existing dtype arms; it adds **no new groupby dtypes** (the survey's
   "integer groupby parity gap" is intentionally *not* filled — extending groupby violates the
   non-goal).
4. **No new perf kernels.** The FFT dual-core fold is deferred to M8 (see §5). (The MLX-wrapper
   parametrization, originally bundled with it, was pulled into M7 as B4 — it is pure DRY with
   no perf or behavior implication.)

## 5. Explicitly out of scope (deferred to M8)

- **FFT dual-core fold** (~187 lines planar duplicating interleaved). Real consolidation, but
  delicate — Bluestein (prime/non-smooth sizes) and the differential oracle depend on the
  interleaved core — so it carries behavior risk disproportionate to a cleanup milestone.
- **All perf-deepening** (cooperative-wavefront DTW, fused single-command-buffer pipelines,
  custom `corr.metal`) — these are M8+ flagship candidates, not M7.
- **Large Python files** (`_walker.py`, `_udf.py`, `_fusion_analyzer.py`, ~1.5k lines each):
  flagged by the survey but orthogonal to the namespace consolidation; revisit if they obstruct
  M8.

## 6. Sequencing & definition of done

**Order:** C1 (differential harness) **first** — it is the safety net. Then **A and B in
parallel** (disjoint files: Python vs Rust). Coverage/doc tidy (C2, C3) last.

**Definition of done:**
- `make gate` green; the differential harness runs under `make test-diff`.
- Line-count reductions realized: Python namespace ≈525 → ≈165; the three Rust dtype-dispatch
  sites folded via `per_dtype!`; `udf.rs` split into the four named modules; the 8 MLX integer
  copy wrappers folded to one parametric FFI entry.
- The verb-contract doc and the rolling-F64-is-CPU note are committed.
- Zero behavior changes except the deliberately-corrected contracts, each test-pinned.

## 7. Testing strategy

- **Characterization tests before every refactor** (A1 for the verbs; existing kernel tests +
  added cases for B2) — pin behavior, then refactor under green.
- **Differential oracle** (C1) — random inputs vs CPU Polars, the broad regression net.
- **Conformance suite** (`engine="metal"`) — must stay clean (the 8 M6 fixes hold).
- A perf regression is *not* a failing test; a correctness regression is. Benches stay separate.
