# M7 C1 — Differential Safety Net (Rust-first) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the differential/property safety net that the M7 A/B refactors lean on — Rust `proptest` for the kernel/dtype-dispatch layer (where most of M7's risk lives), a lean Python `hypothesis` slice for the irreducibly-Python plan/namespace surface, and a discoverable `make test-diff` target wired into the gate.

**Architecture:** The Rust proptest net already exists and is gated (cmp/subgraph/agg/readback). C1 is a **targeted top-up**, not a rebuild: (1) extend the buffer-path subgraph proptest from I32-only to **all 10 numeric dtypes** so B2's `per_dtype!` macro fold is fully pinned before it lands; (2) confirm the cmp + agg fold sites cover their full op/dtype matrix; (3) restore the retired Python plan-level differential slice (engine=metal vs CPU Polars) from git and extend it with F32 fused-chain + reduction strategies; (4) add `make test-diff` aggregating the suite and wire it into `gate`.

**Tech Stack:** Rust + `proptest` (kernel/dispatch layer); Python + `hypothesis` + `polars.testing.assert_frame_equal` (plan/namespace layer); GNU make.

**Note on TDD framing for a safety net:** these are regression/characterization tests over an *already-correct* engine. The honest cycle is: write the test, run it, expect **PASS** against current `main`. A FAIL is not "red-then-green" — it means we surfaced a latent bug; stop and report it rather than papering over it. Each task calls this out where it applies.

---

### Task 1: Pin the `per_dtype!` readback surface — all-10-dtype buffer-path subgraph tests

The `per_dtype!` macro fold in Workstream B2 collapses the 10-arm `write_back!` match in `eval_to_metal_buffers` (`crates/polars-metal-core/src/fusion/subgraph.rs:225+`). Today only **I32** is tested at the buffer level (`test_subgraph_int.rs`). This task pins all 10 dtypes via identity round-trips + a per-dtype arithmetic proptest, so the fold cannot silently break a dtype.

**Files:**
- Create: `crates/polars-metal-core/tests/test_subgraph_dtype_readback.rs`
- Reference (read, do not modify): `crates/polars-metal-core/tests/test_subgraph_int.rs` (the I32 template), `crates/polars-metal-core/src/fusion/scope.rs:8-22` (the `InputDtype` variants), `crates/polars-metal-core/src/fusion/subgraph.rs:225-235` (the `write_back!` arms)

- [ ] **Step 1: Confirm the macro-generated per-dtype buffer API names**

The `MetalBuffer` per-dtype constructors/readbacks are macro-generated (only `from_f32_slice` appears in source text). Confirm the exact method names exist for every numeric dtype before writing tests against them.

Run:
```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
python3 - <<'PY'
import re, pathlib
src = "\n".join(p.read_text() for p in pathlib.Path("crates/polars-metal-buffer/src").rglob("*.rs"))
for dt in ["i8","i16","i32","i64","u8","u16","u32","u64"]:
    has_from = (f"from_{dt}_slice" in src)
    has_to   = (f"to_{dt}_vec" in src)
    print(f"{dt:>4}: from_{dt}_slice={has_from}  to_{dt}_vec={has_to}")
PY
```
Expected: every dtype prints `True` / `True` (the methods are macro-generated). If any dtype is missing a constructor or readback, note it — Step 3 covers only the dtypes that have both; a missing one is a real buffer-crate gap to add in a follow-up (do **not** invent a method name).

- [ ] **Step 2: Write the failing test file (identity round-trips + arithmetic proptest)**

Mirror the I32 pattern from `test_subgraph_int.rs` for every dtype that passed Step 1. Identity round-trips pin the pure readback dispatch; the per-dtype `Add` proptest pins integer (not float-reinterpreted) semantics through `eval_to_metal_buffers`.

```rust
//! M7 C1 Task 1: pin every numeric dtype through the buffer-path subgraph
//! (`eval_to_metal_buffers` readback) so B2's `per_dtype!` macro fold cannot
//! silently break a dtype arm. Mirrors `test_subgraph_int.rs` (I32) for all
//! 10 numeric dtypes.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use polars_metal_buffer::{MetalBuffer, MetalDevice};
use polars_metal_native::fusion::scope::{FusionScope, InputDtype};
use polars_metal_native::fusion::subgraph::MlxSubgraph;
use polars_metal_native::fusion::supported_ops::OpId;
use proptest::prelude::*;

/// Identity-subgraph round-trip: stage a typed buffer, run a no-op subgraph,
/// assert the readback equals the input. Generated per dtype by the macro below.
macro_rules! identity_round_trip {
    ($name:ident, $rs:ty, $from:ident, $to:ident, $input_dt:expr, $vals:expr) => {
        #[test]
        fn $name() {
            let device = MetalDevice::system_default().expect("metal");
            let vals: Vec<$rs> = $vals;
            let buf = Arc::new(MetalBuffer::$from(&device, &vals).expect("stage"));

            let mut scope = FusionScope::new();
            let a = scope.add_input("a", $input_dt);
            scope.mark_output(a);

            let sg = MlxSubgraph::from_fusion_scope_buffers(&scope, &[buf]).expect("build");
            let outs = sg.eval_to_metal_buffers(&device).expect("eval");
            assert_eq!(outs.len(), 1);
            assert_eq!(outs[0].$to(), vals);
        }
    };
}

identity_round_trip!(i8_identity_round_trips,  i8,  from_i8_slice,  to_i8_vec,  InputDtype::I8,  vec![-128, -1, 0, 1, 127]);
identity_round_trip!(i16_identity_round_trips, i16, from_i16_slice, to_i16_vec, InputDtype::I16, vec![-32768, -1, 0, 1, 32767]);
identity_round_trip!(i32_identity_round_trips, i32, from_i32_slice, to_i32_vec, InputDtype::I32, vec![-7, 0, 1, 100, 2_000_000_000]);
identity_round_trip!(i64_identity_round_trips, i64, from_i64_slice, to_i64_vec, InputDtype::I64, vec![-1, 0, 1, 9_000_000_000_000]);
identity_round_trip!(u8_identity_round_trips,  u8,  from_u8_slice,  to_u8_vec,  InputDtype::U8,  vec![0, 1, 127, 255]);
identity_round_trip!(u16_identity_round_trips, u16, from_u16_slice, to_u16_vec, InputDtype::U16, vec![0, 1, 65535]);
identity_round_trip!(u32_identity_round_trips, u32, from_u32_slice, to_u32_vec, InputDtype::U32, vec![0, 1, 4_000_000_000]);
identity_round_trip!(u64_identity_round_trips, u64, from_u64_slice, to_u64_vec, InputDtype::U64, vec![0, 1, 18_000_000_000_000_000_000]);

proptest! {
    #![proptest_config(ProptestConfig { cases: 32, .. ProptestConfig::default() })]

    /// Per-dtype `Add` through the buffer path uses integer (not float-
    /// reinterpreted) semantics — the assertion fails loudly if a fold arm
    /// reads the bit-pattern as the wrong dtype. Ranges are narrowed to avoid
    /// wrap (MLX integer add wraps; the Rust reference would diverge on overflow).
    #[test]
    fn i32_add_buffer_path_matches_scalar(
        a in prop::collection::vec(-1_000_000i32..1_000_000, 1..32),
        b in prop::collection::vec(-1_000_000i32..1_000_000, 1..32),
    ) {
        let device = MetalDevice::system_default().expect("metal");
        let len = a.len().min(b.len());
        let (a, b) = (a[..len].to_vec(), b[..len].to_vec());
        let expect: Vec<i32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

        let a_buf = Arc::new(MetalBuffer::from_i32_slice(&device, &a).expect("stage a"));
        let b_buf = Arc::new(MetalBuffer::from_i32_slice(&device, &b).expect("stage b"));

        let mut scope = FusionScope::new();
        let ai = scope.add_input("a", InputDtype::I32);
        let bi = scope.add_input("b", InputDtype::I32);
        let m = scope.push_op(OpId::Add, vec![ai, bi]);
        scope.mark_output(m);

        let sg = MlxSubgraph::from_fusion_scope_buffers(&scope, &[a_buf, b_buf]).expect("build");
        let outs = sg.eval_to_metal_buffers(&device).expect("eval");
        prop_assert_eq!(outs[0].to_i32_vec(), expect);
    }
}
```

If Step 1 reported a dtype with no `from_*_slice`/`to_*_vec`, delete its `identity_round_trip!` line and note the gap in the commit message — do not fabricate the method.

- [ ] **Step 3: Run the new tests — expect PASS (a FAIL is a latent bug, report it)**

Run:
```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
cargo test -p polars-metal-core --test test_subgraph_dtype_readback -- --test-threads=1
```
Expected: all `*_identity_round_trips` + `i32_add_buffer_path_matches_scalar` PASS. If any FAIL, the current engine has a latent dtype-readback bug — **stop and report it**, do not adjust the assertion to make it pass.

- [ ] **Step 4: Lint**

Run:
```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
cargo fmt && cargo clippy -p polars-metal-core --all-targets -- -D warnings
```
Expected: clean (no warnings).

- [ ] **Step 5: Commit**

```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
git add crates/polars-metal-core/tests/test_subgraph_dtype_readback.rs
git commit -m "M7 C1: pin all 10 dtypes through eval_to_metal_buffers (per_dtype! fold net)"
```

---

### Task 2: Confirm the cmp + agg fold-site op/dtype matrix; top up any gap

B2 also folds the 4 `cmp_*` functions and the `build_agg_kind_and_vcol` arms. Their proptest nets exist (`test_cmp_i64`, `test_cmp_f64`, `test_fused_vs_per_agg`, `test_groupby_aggregate`). This task **audits** that those nets cover the full op/dtype matrix the fold touches, and adds arms only where a gap is found — no speculative tests.

**Files:**
- Reference (read): `crates/polars-metal-kernels/tests/test_cmp_i64.rs`, `crates/polars-metal-kernels/tests/test_cmp_f64.rs`, `crates/polars-metal-kernels/tests/test_fused_vs_per_agg.rs`
- Modify only if a gap is found (exact file determined by the audit).

- [ ] **Step 1: Audit cmp op coverage**

Run:
```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
grep -n "proptest!\|CompareOp::\|ops()\|fn cpu_cmp" crates/polars-metal-kernels/tests/test_cmp_i64.rs crates/polars-metal-kernels/tests/test_cmp_f64.rs
```
Expected: both files have a `proptest!` block driving all 6 `CompareOp` variants (Eq/Ne/Lt/Le/Gt/Ge) over both column-column and column-scalar shapes against a `cpu_cmp_*` reference. **Pass criterion:** the proptest covers all 6 ops × {cc, cs} shapes with nulls. If it does, cmp is fully pinned — record "cmp covered, no change" and skip to Step 3.

- [ ] **Step 2: Audit agg dispatch dtype coverage**

Run:
```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
grep -n "proptest!\|Sum\|Mean\|Min\|Max\|Count\|I32\|I64\|F32\|F64\|dtype" crates/polars-metal-kernels/tests/test_fused_vs_per_agg.rs | head -40
```
Expected: a proptest comparing the fused agg path against the per-agg reference across the agg kinds (Sum/Mean/Min/Max/Count) and the dtypes the live path supports (per CLAUDE.md, groupby is conformance-only — the live fused dtypes are I32/F32; do **not** add new groupby dtypes, that violates the non-goal). **Pass criterion:** every agg kind on the supported dtypes is exercised against the reference. If a supported (agg, dtype) pair is missing, add it to the existing proptest's strategy in Step 4; otherwise record "agg covered, no change."

- [ ] **Step 3: If both audits pass — record and move on (no code change)**

If Steps 1–2 both met their pass criteria, this task is a no-op confirmation. Write a one-line note in the C1 progress log: "cmp + agg fold sites: full matrix covered by existing proptest; no top-up needed." Skip to Task 3. (This is a legitimate outcome — the survey found these nets already strong.)

- [ ] **Step 4: (Only if a gap was found) Add the missing arm, run, lint, commit**

Add the missing (op, shape) or (agg, dtype) case to the *existing* proptest strategy in the relevant file (mirror its current arms exactly — same reference fn, same tolerance). Then:
```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
cargo test -p polars-metal-kernels --test <file_stem> -- --test-threads=1   # expect PASS
cargo fmt && cargo clippy -p polars-metal-kernels --all-targets -- -D warnings
git add crates/polars-metal-kernels/tests/<file>.rs
git commit -m "M7 C1: top up <cmp|agg> proptest to cover <the missing case>"
```
Expected: PASS, clean lint.

---

### Task 3: Restore the lean Python plan-level differential slice

The retired `tests/diff/` harness covered the engine=metal-vs-CPU **plan surface** (scan / filter / predicate / projection over random null-heavy frames) — the irreducibly-Python net for the routing/walker that A and the walker touch. Restore it from git and extend it with F32 fused-chain + reduction strategies. Keep it lean: kernels are Rust's job (Tasks 1–2).

**Files:**
- Restore from git `f669dd4^`: `tests/diff/__init__.py`, `tests/diff/conftest.py`, `tests/diff/strategies.py`
- Create: `tests/diff/test_differential.py` (restored + extended)
- Reference (read): `tests/python_integration/test_rolling_property.py` (the `assert_frame_equal(..., rel_tol=, abs_tol=)` pattern)

- [ ] **Step 1: Restore the retired harness files from git**

Run:
```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
git show 'f669dd4^:tests/diff/__init__.py'   > tests/diff/__init__.py
git show 'f669dd4^:tests/diff/conftest.py'   > tests/diff/conftest.py
git show 'f669dd4^:tests/diff/strategies.py' > tests/diff/strategies.py
ls tests/diff/
```
Expected: `__init__.py`, `conftest.py`, `strategies.py` present. (`strategies.py` provides `numeric_frame`, `null_heavy_frame`, `m1_null_density_dataframe`, `m1_predicate_expr`, `m1_projection_subset`.)

- [ ] **Step 2: Write the restored + extended differential test**

The first three properties are the restored plan-surface net (engine vs CPU over random frames). The fourth is the M7 extension: a bounded random F32 compute chain (the M4 fusion path), optionally terminated by a reduction, asserted equal to CPU within tolerance.

```python
"""M7 C1: plan-level differential properties — collect(engine=metal) matches CPU.

Plan surface only (scan / filter / predicate / projection / fused F32 chains).
Kernel-level numerics are covered Rust-side by proptest (Tasks 1-2); this slice
exists for the irreducibly-Python engine-plugin + routing surface.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

import polars_metal
from tests.diff.strategies import (
    m1_null_density_dataframe,
    null_heavy_frame,
    numeric_frame,
)

_ENG = polars_metal.MetalEngine()


@given(numeric_frame())
@settings(max_examples=100, deadline=None)
def test_numeric_collect_matches_cpu(lf) -> None:  # type: ignore[no-untyped-def]
    assert lf.collect(engine=_ENG).equals(lf.collect())


@given(null_heavy_frame())
@settings(max_examples=100, deadline=None)
def test_null_heavy_collect_matches_cpu(lf) -> None:  # type: ignore[no-untyped-def]
    assert lf.collect(engine=_ENG).equals(lf.collect())


@given(m1_null_density_dataframe())
@settings(max_examples=100, deadline=None)
def test_m1_frame_scan_matches_cpu(df) -> None:  # type: ignore[no-untyped-def]
    lf = df.lazy()
    assert lf.collect(engine=_ENG).equals(lf.collect())


# --- M7 extension: random F32 fused compute chains vs CPU --------------------

# A "safe" op set whose F32 output is finite for bounded inputs in [-10, 10],
# so engine and CPU agree to tolerance (no NaN/Inf divergence). Mirrors the
# Rust proptest_subgraph "safe set" philosophy on the Python plan surface.
def _apply(expr: pl.Expr, op: str) -> pl.Expr:
    if op == "neg":
        return -expr
    if op == "abs":
        return expr.abs()
    if op == "square":
        return expr * expr
    if op == "sin":
        return expr.sin()
    if op == "cos":
        return expr.cos()
    if op == "tanh":
        return expr.tanh()
    raise AssertionError(f"unknown op {op!r}")


_OPS = ("neg", "abs", "square", "sin", "cos", "tanh")


@given(
    n=st.integers(min_value=1, max_value=2000),
    ops=st.lists(st.sampled_from(_OPS), min_size=1, max_size=5),
    reducer=st.sampled_from((None, "sum", "mean", "std", "var")),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=60, deadline=None)
def test_fused_f32_chain_matches_cpu(n, ops, reducer, seed) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(n) * 3.0).astype(np.float32)
    df = pl.DataFrame({"x": x}, schema={"x": pl.Float32})

    expr = pl.col("x")
    for op in ops:
        expr = _apply(expr, op)
    if reducer is not None:
        if reducer in ("std", "var") and n < 2:
            return  # ddof=1 undefined for n<2; CPU and engine both skip
        expr = getattr(expr, reducer)()

    lf = df.lazy().select(r=expr)
    assert_frame_equal(
        lf.collect(engine=_ENG),
        lf.collect(),
        check_exact=False,
        rel_tol=1e-3,
        abs_tol=1e-4,
    )
```

- [ ] **Step 3: Run the Python differential slice — expect PASS (a FAIL is a latent bug)**

Run:
```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
pytest tests/diff/test_differential.py -q
```
Expected: all properties PASS. A FAIL means an engine-vs-CPU divergence on the plan surface — **stop and report it** (it may be a known deviation; check `docs/open-questions.md` and the conformance known-failures before treating it as new).

- [ ] **Step 4: Lint**

Run:
```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
ruff check tests/diff/ && ruff format --check tests/diff/
```
Expected: clean. (If `ruff format --check` flags the restored files, run `ruff format tests/diff/` and re-check.)

- [ ] **Step 5: Commit**

```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
git add tests/diff/
git commit -m "M7 C1: restore lean Python plan-level differential slice + F32 fused-chain property"
```

---

### Task 4: Add `make test-diff`, wire into the gate, fix the stale comment

Make the differential suite discoverable and gated. `test-diff` runs the Rust proptest subset (Tasks 1–2 + the existing nets) plus the Python slice (Task 3).

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Add the `test-diff` target and update `.PHONY`**

In `Makefile`, add `test-diff` to the `.PHONY` line and add the target. The Rust differential tests are named proptest/differential test files; run them by name plus the Python slice. Replace the `.PHONY` line:

```makefile
.PHONY: build wheel test-unit test-kernel test-conformance test-diff bench lint gate refresh-refs help
```

Add this target (place it after `test-conformance`):

```makefile
test-diff:
	# Rust differential/property nets (kernel + dispatch layer)
	cargo test -p polars-metal-core --test proptest_subgraph --test test_subgraph_int --test test_subgraph_dtype_readback -- --test-threads=1
	cargo test -p polars-metal-kernels --test test_cmp_i64 --test test_cmp_f64 --test test_fused_vs_per_agg -- --test-threads=1
	# Python plan-level differential slice (engine=metal vs CPU)
	pytest tests/diff -q
```

- [ ] **Step 2: Fix the stale `test-kernel` comment (C3 overlap, cheap here)**

Replace the stale M0 placeholder line in the `test-kernel` target:

```makefile
test-kernel:
	# kernel correctness suite (proptest + differential, --test-threads=1 for Metal queue safety)
	cargo test -p polars-metal-kernels -- --test-threads=1
```

- [ ] **Step 3: Run `make test-diff` standalone — expect PASS**

Run:
```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
make test-diff
```
Expected: all Rust differential tests + the Python slice PASS. (Requires the wheel built — if pytest errors on import, run `make wheel` first.)

- [ ] **Step 4: Wire `test-diff` into the gate**

Replace the `gate` target line to include `test-diff` (after `test-conformance`):

```makefile
gate: lint test-unit test-kernel wheel test-conformance test-diff
	@echo "M0 gate passed."
```

- [ ] **Step 5: Run the full gate — expect PASS**

Run:
```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
make gate
```
Expected: gate passes end to end. (This is the slow full run; allow several minutes.)

- [ ] **Step 6: Commit**

```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
git add Makefile
git commit -m "M7 C1: add make test-diff target, wire into gate, fix stale test-kernel comment"
```

---

## Self-Review

**Spec coverage (against §3 C1 of the design):**
- "Pin the `per_dtype!` fold surface (all 10 dtypes through `eval_to_metal_buffers`)" → Task 1. ✓
- "Confirm cmp + agg fold sites cover their full op/dtype matrix; add missing arms" → Task 2. ✓
- "Restore the lean Python plan-level slice + extend with F32 fused-chain + reduction strategies" → Task 3. ✓
- "Add a discoverable `make test-diff` target and wire into `gate`" → Task 4. ✓
- Guardrail "don't extend conformance-only groupby" → Task 2 Step 2 explicitly forbids adding new groupby dtypes. ✓

**Placeholder scan:** No TBD/TODO. Task 2 is deliberately audit-then-conditionally-act (a legitimate "confirm the existing net is sufficient" task, not a placeholder) — its no-op outcome is explicitly allowed and recorded.

**Type/name consistency:** `InputDtype::{I8,I16,I32,I64,U8,U16,U32,U64}` match `scope.rs:8-22`; `MetalBuffer::from_*_slice`/`to_*_vec` confirmed macro-generated and verified in Task 1 Step 1 before use; `MlxSubgraph::{from_fusion_scope_buffers, eval_to_metal_buffers}` match `test_subgraph_int.rs`; the restored strategy names (`numeric_frame`, `null_heavy_frame`, `m1_null_density_dataframe`) match the recovered `strategies.py`.

**Out of scope (deferred to A/B plans):** namespace-verb randomized sweeps vs numpy/external oracles (fft/corr/dtw/vector) — those verbs already have hand-picked differential tests; their randomized-sweep net belongs with the Workstream A namespace-spine plan (the refactor it protects), not C1.
