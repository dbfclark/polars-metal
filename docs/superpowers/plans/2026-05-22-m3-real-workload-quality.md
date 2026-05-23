# M3 — Real-workload quality: TPC-H slice, dual-mode GPU build, strings — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `polars-metal` at real-workload quality on the analytical surface M2 staked out. Land canonical TPC-H Q1 (Utf8 keys + inline expressions), high-cardinality Q1 (64K and 1M groups), Q6, and 100M-row Q1 with each strictly faster than CPU Polars on M2 Ultra.

**Architecture:** M2's three-layer flow (walker → router → walker applies → dispatch) is preserved. M3 extends each layer: walker recognizes Utf8 keys and binary-expression aggregations; router selects build-phase mode (partitioned hash A1 / sort-segment-reduce A2) by cardinality estimate and recognizes filter+GroupBy fused subtrees; kernel layer adds an MSL template engine that emits one fused aggregation kernel per query signature.

**Tech Stack:** Rust 2021 (workspace), `objc2-metal` (Metal API), `cxx` for MLX FFI (unchanged from M2), `pyo3 0.22` + `maturin` (unchanged), `polars` pinned to `py-1.40.1` (unchanged), `proptest` for kernel + reference comparison, `pytest-benchmark` + `criterion` for perf.

**Spec:** [`docs/superpowers/specs/2026-05-22-m3-design.md`](../specs/2026-05-22-m3-design.md). All decisions there are binding; this plan does not relitigate them.

**Conventions** (per CLAUDE.md): No `unwrap()` outside tests. No `unsafe` outside `*-sys` crates and the buffer bridge — each with a `// SAFETY:` comment. One MSL kernel family per file. Errors propagate as `polars.exceptions.ComputeError` at the engine boundary. Null semantics match Polars exactly. Don't add files to `shaders/` without a matching test. Read the matching cuDF kernel before writing MSL.

**Pre-task reading.** Before starting Phase 3 (fused agg template), read:
- `crates/polars-metal-kernels/src/groupby.rs` (M2's CPU-finalize aggregation pattern; M3's fused kernel replaces this for 32-bit hot path)
- `shaders/aggregate.metal` (M2's per-agg kernels; M3 emits equivalent stanzas inside one fused kernel)
- `crates/polars-metal-core/src/plan/mod.rs` (AggSpec; M3 adds `AggSpec::Expression`)

Before Phase 4 (partitioned hash A1), read:
- `references/cudf/cpp/src/groupby/hash/groupby.cu` — cuDF's hash-table-in-shared-memory design; we adapt to Apple Silicon's TGSM
- `docs/kernel-authoring.md` § "Apple Silicon Metal atomic ops constraint" — why 32-bit atomics only
- `shaders/_validity.metal` — null-bitmap helpers (M1/M2)

Before Phase 5 (sort-segment-reduce A2), read:
- `references/cudf/cpp/src/groupby/sort/groupby.cu` — the algorithm we port
- `references/cudf/cpp/src/sort/radix_sort.cuh` — radix-sort primitive (for our restricted-scope u128 sort)

Before Phase 7 (string dictionary), read:
- `references/polars/crates/polars-core/src/chunked_array/builder/string.rs` — Polars' string column layout
- `crates/polars-metal-buffer/src/lib.rs` — current buffer bridge

Before Phase 10 (filter into GPU), read:
- `crates/polars-metal-core/src/router/affinity.rs` — M2's smoothing pass; M3 extends with fused-subtree recognition

---

## Phase 0 — Preflight + perf gate (capability H)

### Task 1: Confirm M2 gates green on the new branch

**Files:** none (verification only).

- [ ] **Step 1: Branch off main**

```bash
git checkout main && git pull --ff-only origin main
git checkout -b m3-realworkload
git rev-parse --abbrev-ref HEAD && git log -1 --oneline
```

Expected: branch `m3-realworkload`; HEAD at the M2 merge commit (`3568262` or later).

- [ ] **Step 2: Run the M2 gate**

```bash
make gate
```

Expected: all phases pass (`lint`, `test-unit`, `test-kernel`, `wheel`, `test-conformance`). Wall-clock ~6-8 min on M2 Ultra.

If anything fails: stop and fix on a separate branch before M3 work; do not pile new work on top of a broken baseline.

- [ ] **Step 3: Verify Metal toolchain + MLX still present**

```bash
xcrun metal --version && python -c "import polars_metal; print(polars_metal._native.version_string())"
```

Expected: Metal toolchain reports a version; the Python import succeeds.
If `polars_metal` import fails: `make wheel` to rebuild.

- [ ] **Step 4: Record M2 baseline values**

```bash
python -c "import json; d=json.load(open('tests/bench/baseline.json')); print(d['queries']['tpch_q1_modified'])"
```

Expected: prints `ratio_metal_over_cpu ≈ 0.914` (or whatever the current M2 value is). Record this value; M3's perf gate verifies M2 doesn't regress.

**Note:** `baseline.json` nests query entries under a top-level `queries` key. Top-level fields are `_units`, `_notes`, `machine`, `git_sha`, `date`, `queries`. Task 2's gate-check helper iterates `baseline["queries"]`, not the top-level dict.

Nothing to commit in Task 1.

### Task 2: Extend `baseline.json` schema for per-entry gate thresholds (capability H)

**Files:**
- Modify: `tests/bench/baseline.json`
- Create: `tests/bench/_gate_check.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/bench/test_gate_check.py
"""Verify the gate-check helper enforces per-entry ratio thresholds."""
import json
from pathlib import Path

import pytest

from tests.bench._gate_check import check_baseline


def _baseline_with(queries):
    """Build a fixture mirroring real baseline.json: queries nested under top-level key."""
    return {
        "_notes": "test fixture",
        "git_sha": "deadbeef",
        "date": "2026-05-22",
        "queries": queries,
    }


def test_check_passes_when_all_ratios_meet_threshold():
    baseline = _baseline_with({
        "tpch_q1_modified": {"ratio_metal_over_cpu": 0.914, "_gate": {"ratio_lt": 1.0}},
    })
    failures = check_baseline(baseline)
    assert failures == []


def test_check_fails_when_ratio_exceeds_threshold():
    baseline = _baseline_with({
        "tpch_q1_modified": {"ratio_metal_over_cpu": 1.05, "_gate": {"ratio_lt": 1.0}},
    })
    failures = check_baseline(baseline)
    assert len(failures) == 1
    assert "tpch_q1_modified" in failures[0]
    assert "1.05" in failures[0] and "1.0" in failures[0]


def test_check_skips_entries_without_gate_metadata():
    baseline = _baseline_with({
        "informational_entry": {"ratio_metal_over_cpu": 99.0},
    })
    failures = check_baseline(baseline)
    assert failures == []


def test_check_reports_missing_required_key():
    baseline = _baseline_with({
        "tpch_q1_modified": {"_gate": {"ratio_lt": 1.0}},  # ratio_metal_over_cpu absent
    })
    failures = check_baseline(baseline)
    assert any("missing ratio_metal_over_cpu" in f for f in failures)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/bench/test_gate_check.py -v
```

Expected: collection error (`_gate_check` does not exist).

- [ ] **Step 3: Implement `_gate_check.py`**

```python
# tests/bench/_gate_check.py
"""Per-entry perf-gate check.

`baseline.json` entries may include a `_gate` block:

    {
        "ratio_metal_over_cpu": 0.914,
        "_gate": {"ratio_lt": 1.0}
    }

If `_gate.ratio_lt` is present, the actual ratio must be strictly less.
Entries without a `_gate` block are informational (no check).
"""
from __future__ import annotations

from typing import Any


def check_baseline(baseline: dict[str, Any]) -> list[str]:
    """Return a list of failure messages; empty list = pass.

    Iterates baseline["queries"]; top-level keys (_units, _notes, machine,
    git_sha, date) are metadata and skipped.
    """
    failures: list[str] = []
    queries = baseline.get("queries", {})
    for name, entry in queries.items():
        if not isinstance(entry, dict):
            continue
        gate = entry.get("_gate")
        if not gate:
            continue
        if "ratio_lt" in gate:
            actual = entry.get("ratio_metal_over_cpu")
            if actual is None:
                failures.append(
                    f"{name}: missing ratio_metal_over_cpu (gate requires it)"
                )
                continue
            limit = gate["ratio_lt"]
            if not actual < limit:
                failures.append(
                    f"{name}: ratio_metal_over_cpu={actual} not < {limit}"
                )
    return failures
```

- [ ] **Step 4: Add `_gate` block to M2's existing entries**

Edit `tests/bench/baseline.json`. The file structure is:

```json
{
  "_units": "...",
  "machine": "...",
  "git_sha": "...",
  "date": "...",
  "queries": {
    "tpch_q1_modified": { "cpu_ms": ..., "metal_ms": ..., "ratio_metal_over_cpu": 0.914 },
    "tpch_q1_modified_32bit": { ..., "ratio_metal_over_cpu": 0.988 },
    "tpch_q1_modified_32bit_high_card": { ..., "ratio_metal_over_cpu": 0.991 },
    "filter_*": { ... }   // these are filter-only baselines; no _gate
  }
}
```

Add `"_gate": {"ratio_lt": 1.0}` inside each of the three `tpch_q1_modified*` entries (under `queries.<name>`). Leave the `filter_*` entries without a `_gate` — they're informational. Preserve all existing `cpu_ms`/`metal_ms`/`ratio_metal_over_cpu` values exactly.

- [ ] **Step 5: Run tests again to verify pass**

```bash
pytest tests/bench/test_gate_check.py -v
```

Expected: 4 passes.

- [ ] **Step 6: Commit**

```bash
git add tests/bench/_gate_check.py tests/bench/test_gate_check.py tests/bench/baseline.json
git commit -m "Bench: per-entry _gate thresholds; check helper

Capability H. Existing M2 entries get _gate.ratio_lt = 1.0; M3
will add new entries with tighter thresholds per spec § Performance.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 3: Wire gate check into `make bench` workflow

**Files:**
- Modify: `tests/bench/conftest.py` or create `tests/bench/test_baseline_gate.py`

- [ ] **Step 1: Add a session-end test that runs the gate**

```python
# tests/bench/test_baseline_gate.py
"""End-of-bench-session gate verification.

Runs after pytest-benchmark fixtures have updated baseline.json.
Fails the session if any _gate-ed entry violates its threshold.
"""
import json
from pathlib import Path

import pytest

from tests.bench._gate_check import check_baseline


def test_baseline_gate_thresholds_met():
    path = Path(__file__).parent / "baseline.json"
    baseline = json.loads(path.read_text())
    failures = check_baseline(baseline)
    if failures:
        msg = "Perf gate failures:\n" + "\n".join(f"  - {f}" for f in failures)
        pytest.fail(msg)
```

- [ ] **Step 2: Run to verify pass on current baseline**

```bash
pytest tests/bench/test_baseline_gate.py -v
```

Expected: pass (M2 entries meet their `ratio_lt: 1.0` gates).

- [ ] **Step 3: Commit**

```bash
git add tests/bench/test_baseline_gate.py
git commit -m "Bench: session-end gate-threshold check

Runs in pytest tests/bench/, fails if any _gate-ed entry violates
its ratio_lt threshold.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 1 — Smaller-integer composite key dtypes (capability F)

M2's composite-key encoder supports `i64`, `f64`, `bool`, `i32`, `f32` (5 variants). M3 extends to `i8`, `i16`, `u8`, `u16`, `u32`. This is a contained extension to `crates/polars-metal-kernels/src/groupby.rs` (where `KeyDtype` actually lives — the plan originally referenced `polars-metal-core/src/plan/mod.rs` but the real definition is in the kernels crate), with proptest coverage per new dtype.

This phase is foundational for downstream phases — A's hash kernels, B's fused agg, and D's string-dict path all consume the encoder.

**Note on sort-order preservation.** M2's encoder does *not* sort-bias signed integers (I32/I64 cast through their unsigned bit pattern: `i32 as u32 as u128`). Groupby is hash-based — sort order doesn't matter for finding duplicates. M3 follows M2's convention: signed-int variants pass through their unsigned bit pattern.

### Task 4: Extend `KeyDtype` enum

**Files:**
- Modify: `crates/polars-metal-kernels/src/groupby.rs` (the `KeyDtype` enum + `data_bits()` + encoder dispatch in `encode_keys` + decoder dispatch in `decode_keys` — all in one file)

- [ ] **Step 1: Read M2's existing `KeyDtype`**

```bash
grep -n "enum KeyDtype" crates/polars-metal-core/src/plan/mod.rs
```

Note the existing variants and their `width_bits()` impl. M3 adds variants; the pattern is identical.

- [ ] **Step 2: Add new variants**

In `crates/polars-metal-core/src/plan/mod.rs`, find `enum KeyDtype` and add:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyDtype {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
}

impl KeyDtype {
    /// Bit-width consumed in the composite key (data bits, not including
    /// the null indicator bit added separately by the encoder).
    pub fn width_bits(self) -> u32 {
        match self {
            KeyDtype::Bool => 1,
            KeyDtype::I8  | KeyDtype::U8  => 8,
            KeyDtype::I16 | KeyDtype::U16 => 16,
            KeyDtype::I32 | KeyDtype::U32 | KeyDtype::F32 => 32,
            KeyDtype::I64 | KeyDtype::U64 | KeyDtype::F64 => 64,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, KeyDtype::F32 | KeyDtype::F64)
    }

    pub fn is_signed(self) -> bool {
        matches!(self, KeyDtype::I8 | KeyDtype::I16 | KeyDtype::I32 | KeyDtype::I64)
    }
}
```

- [ ] **Step 3: Update encoder dispatch in `groupby.rs`**

Find the `encode_keys` function (or equivalent — M2 named it `composite_encode` per spec § "Composite key encoding"). Add match arms for each new dtype that read N bytes from the Polars Series chunk and pack into the u128 at the schema-recorded bit offset. The bit-packing helper from M2 (`pack_bits` or similar) takes (value: u64, width: u32, offset: u32) and is reused.

For signed integers, zero-extend after a bias by `2^(width-1)` so the sort order matches Polars' total ordering (M2's i64 path uses this; M3's i8/i16/i32 follow identical pattern but with smaller widths).

- [ ] **Step 4: Update decoder dispatch**

Mirror the encoder: each new dtype's `decode_keys` arm reads N bits at the schema-recorded offset, reverses the signed-int bias, and writes to a typed `Vec<T>` for result reconstruction.

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-core/src/plan/mod.rs crates/polars-metal-kernels/src/groupby.rs
git commit -m "Plan: KeyDtype gains i8/i16/i32/u8/u16/u32

Capability F. Encoder/decoder dispatch arms added for each new dtype
using the same bit-packing pattern M2 established for i64/f64/bool.
Signed integers biased to preserve sort order.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 5: Proptest for new dtypes — encoder roundtrip

**Files:**
- Create: `crates/polars-metal-kernels/tests/test_key_encoding_small_ints.rs`

- [ ] **Step 1: Write the test**

```rust
// crates/polars-metal-kernels/tests/test_key_encoding_small_ints.rs
//! Proptest: encode → decode roundtrip for the M3-added KeyDtype variants.
//! Follows M2's pattern: struct-literal KeyColumn with `data: &[u8]` and
//! `valid: &[u8]` (no constructor helpers exist).

use polars_metal_kernels::groupby::{
    decode_keys, encode_keys, DecodedColumn, KeyColumn, KeyDtype,
};
use proptest::prelude::*;

fn bytes_i8(values: &[i8]) -> Vec<u8> {
    values.iter().map(|v| *v as u8).collect()
}
fn bytes_i16(values: &[i16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn bytes_i32(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn bytes_u8(values: &[u8]) -> Vec<u8> {
    values.to_vec()
}
fn bytes_u16(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn bytes_u32(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn all_valid(n_rows: usize) -> Vec<u8> {
    vec![0xFFu8; (n_rows + 7) / 8]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn i8_roundtrip(values in proptest::collection::vec(any::<i8>(), 1..256)) {
        let data = bytes_i8(&values);
        let valid = all_valid(values.len());
        let col = KeyColumn { name: "k".into(), dtype: KeyDtype::I8, data: &data, valid: &valid, n_rows: values.len() };
        let (encoded, schema) = encode_keys(&[col]).expect("encode");
        let decoded = decode_keys(&encoded, &schema);
        match &decoded[0] {
            DecodedColumn::I8 { values: out, valid: ovalid } => {
                prop_assert_eq!(out, &values);
                prop_assert!(ovalid.iter().all(|&v| v));
            }
            _ => prop_assert!(false, "expected DecodedColumn::I8"),
        }
    }

    #[test]
    fn i16_roundtrip(values in proptest::collection::vec(any::<i16>(), 1..256)) {
        let data = bytes_i16(&values);
        let valid = all_valid(values.len());
        let col = KeyColumn { name: "k".into(), dtype: KeyDtype::I16, data: &data, valid: &valid, n_rows: values.len() };
        let (encoded, schema) = encode_keys(&[col]).expect("encode");
        let decoded = decode_keys(&encoded, &schema);
        match &decoded[0] {
            DecodedColumn::I16 { values: out, .. } => prop_assert_eq!(out, &values),
            _ => prop_assert!(false, "expected DecodedColumn::I16"),
        }
    }

    #[test]
    fn u8_roundtrip(values in proptest::collection::vec(any::<u8>(), 1..256)) {
        let data = bytes_u8(&values);
        let valid = all_valid(values.len());
        let col = KeyColumn { name: "k".into(), dtype: KeyDtype::U8, data: &data, valid: &valid, n_rows: values.len() };
        let (encoded, schema) = encode_keys(&[col]).expect("encode");
        let decoded = decode_keys(&encoded, &schema);
        match &decoded[0] {
            DecodedColumn::U8 { values: out, .. } => prop_assert_eq!(out, &values),
            _ => prop_assert!(false, "expected DecodedColumn::U8"),
        }
    }

    #[test]
    fn u16_roundtrip(values in proptest::collection::vec(any::<u16>(), 1..256)) {
        let data = bytes_u16(&values);
        let valid = all_valid(values.len());
        let col = KeyColumn { name: "k".into(), dtype: KeyDtype::U16, data: &data, valid: &valid, n_rows: values.len() };
        let (encoded, schema) = encode_keys(&[col]).expect("encode");
        let decoded = decode_keys(&encoded, &schema);
        match &decoded[0] {
            DecodedColumn::U16 { values: out, .. } => prop_assert_eq!(out, &values),
            _ => prop_assert!(false, "expected DecodedColumn::U16"),
        }
    }

    #[test]
    fn u32_roundtrip(values in proptest::collection::vec(any::<u32>(), 1..256)) {
        let data = bytes_u32(&values);
        let valid = all_valid(values.len());
        let col = KeyColumn { name: "k".into(), dtype: KeyDtype::U32, data: &data, valid: &valid, n_rows: values.len() };
        let (encoded, schema) = encode_keys(&[col]).expect("encode");
        let decoded = decode_keys(&encoded, &schema);
        match &decoded[0] {
            DecodedColumn::U32 { values: out, .. } => prop_assert_eq!(out, &values),
            _ => prop_assert!(false, "expected DecodedColumn::U32"),
        }
    }

    #[test]
    fn multi_dtype_composite_under_128_bits(
        i8_vals  in proptest::collection::vec(any::<i8>(),  4..32),
        i16_vals in proptest::collection::vec(any::<i16>(), 4..32),
        u32_vals in proptest::collection::vec(any::<u32>(), 4..32),
    ) {
        let n = i8_vals.len().min(i16_vals.len()).min(u32_vals.len());
        let i8_vals = &i8_vals[..n];
        let i16_vals = &i16_vals[..n];
        let u32_vals = &u32_vals[..n];
        let valid = all_valid(n);
        let d8 = bytes_i8(i8_vals);
        let d16 = bytes_i16(i16_vals);
        let d32 = bytes_u32(u32_vals);
        // Total: 3*1 + 8 + 16 + 32 = 59 bits (well under 128)
        let cols = vec![
            KeyColumn { name: "a".into(), dtype: KeyDtype::I8,  data: &d8,  valid: &valid, n_rows: n },
            KeyColumn { name: "b".into(), dtype: KeyDtype::I16, data: &d16, valid: &valid, n_rows: n },
            KeyColumn { name: "c".into(), dtype: KeyDtype::U32, data: &d32, valid: &valid, n_rows: n },
        ];
        let (encoded, schema) = encode_keys(&cols).expect("encode");
        let decoded = decode_keys(&encoded, &schema);
        match (&decoded[0], &decoded[1], &decoded[2]) {
            (DecodedColumn::I8 { values: out_a, .. },
             DecodedColumn::I16 { values: out_b, .. },
             DecodedColumn::U32 { values: out_c, .. }) => {
                prop_assert_eq!(out_a.as_slice(), i8_vals);
                prop_assert_eq!(out_b.as_slice(), i16_vals);
                prop_assert_eq!(out_c.as_slice(), u32_vals);
            }
            _ => prop_assert!(false, "unexpected decoded variants"),
        }
    }

    #[test]
    fn duplicate_signed_values_encode_identically(values in proptest::collection::vec(any::<i32>(), 2..64)) {
        // Groupby relies on identical keys producing identical encoded u128s.
        // Sort order is NOT preserved (M2's convention; groupby is hash-based).
        let data = bytes_i32(&values);
        let valid = all_valid(values.len());
        let col = KeyColumn { name: "k".into(), dtype: KeyDtype::I32, data: &data, valid: &valid, n_rows: values.len() };
        let (encoded, _schema) = encode_keys(&[col]).expect("encode");
        for i in 0..values.len() {
            for j in 0..values.len() {
                if values[i] == values[j] {
                    prop_assert_eq!(encoded[i], encoded[j]);
                }
            }
        }
    }
}
```

- [ ] **Step 2: No constructor helpers needed**

The test uses M2's existing struct-literal pattern. `KeyColumn` already has all required fields (`name`, `dtype`, `data`, `valid`, `n_rows`). `DecodedColumn` variants are pattern-matched directly (no accessor methods).

- [ ] **Step 3: Run the test**

```bash
cargo test -p polars-metal-kernels --test test_key_encoding_small_ints -- --test-threads=1
```

Expected: 5 properties pass at 256 cases each.

- [ ] **Step 4: Commit**

```bash
git add crates/polars-metal-kernels/tests/test_key_encoding_small_ints.rs crates/polars-metal-kernels/src/groupby.rs
git commit -m "Kernel: proptest roundtrip + sort-order for i8/i16/i32/u8/u16/u32 keys

Capability F. 5 properties at 256 cases each; covers encode→decode
identity and sort-order preservation for signed ints.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 6: Walker — recognize Polars Int8/Int16/Int32/UInt8/UInt16/UInt32 dtypes as keys

**Files:**
- Modify: `python/polars_metal/_walker.py`

- [ ] **Step 1: Find the dtype-to-KeyDtype mapping**

```bash
grep -n "KeyDtype" python/polars_metal/_walker.py
```

M2 has a map from `pl.DataType` to the string used in the wire format. Find it.

- [ ] **Step 2: Extend the map**

```python
# python/polars_metal/_walker.py (excerpt)
_POLARS_DTYPE_TO_KEY_DTYPE = {
    pl.Boolean: "Bool",
    pl.Int8:    "I8",
    pl.Int16:   "I16",
    pl.Int32:   "I32",
    pl.Int64:   "I64",
    pl.UInt8:   "U8",
    pl.UInt16:  "U16",
    pl.UInt32:  "U32",
    pl.UInt64:  "U64",
    pl.Float32: "F32",
    pl.Float64: "F64",
}
```

If M2's map is keyed differently (e.g. `pl.dtype` instance vs class), follow the existing pattern.

- [ ] **Step 3: Verify width check still uses sum of `width_bits`**

The walker's fallback-on-overflow check (composite key > 128 bits) should now correctly account for the new dtypes by reading from the Rust side via `_native.key_dtype_width_bits(name)` or by inlining the same widths. M2 either has a Python copy or queries Rust — match the existing pattern.

- [ ] **Step 4: Commit**

```bash
git add python/polars_metal/_walker.py
git commit -m "Walker: route i8/i16/i32/u8/u16/u32 keys through native encoder

Capability F. Polars dtype map extended; width-budget check stays in
place. Composite > 128 bits still falls back.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 7: Python integration test — Q1-i8 fixture

**Files:**
- Create: `tests/python_integration/test_groupby_small_int_keys.py`

- [ ] **Step 1: Write the test**

```python
# tests/python_integration/test_groupby_small_int_keys.py
"""Verify groupby on i8/i16/i32 key columns matches CPU byte-exact."""
import polars as pl
from polars.testing import assert_frame_equal

import polars_metal as pm


def _make_df(n=10_000, n_groups=4, dtype=pl.Int8):
    keys = pl.Series("k", [(i % n_groups) for i in range(n)], dtype=dtype)
    vals = pl.Series("v", [i * 1.5 for i in range(n)], dtype=pl.Float64)
    return pl.DataFrame([keys, vals])


def _check(df):
    q = df.lazy().group_by("k").agg(pl.col("v").sum(), pl.len())
    cpu = q.collect(engine="cpu").sort("k")
    metal = q.collect(engine=pm.MetalEngine()).sort("k")
    assert_frame_equal(cpu, metal)


def test_groupby_i8_key():
    _check(_make_df(dtype=pl.Int8))


def test_groupby_i16_key():
    _check(_make_df(dtype=pl.Int16))


def test_groupby_i32_key():
    _check(_make_df(dtype=pl.Int32))


def test_groupby_u8_key():
    _check(_make_df(dtype=pl.UInt8))


def test_groupby_u16_u32_mixed():
    """Multi-key composite with smaller integers fits in 128 bits."""
    n = 10_000
    k1 = pl.Series("k1", [(i % 4)  for i in range(n)], dtype=pl.UInt16)
    k2 = pl.Series("k2", [(i % 8)  for i in range(n)], dtype=pl.UInt32)
    v  = pl.Series("v",  [i * 1.5  for i in range(n)], dtype=pl.Float64)
    df = pl.DataFrame([k1, k2, v])
    q = df.lazy().group_by("k1", "k2").agg(pl.col("v").sum())
    assert_frame_equal(
        q.collect(engine="cpu").sort(["k1", "k2"]),
        q.collect(engine=pm.MetalEngine()).sort(["k1", "k2"]),
    )
```

- [ ] **Step 2: Build + run**

```bash
make wheel
pytest tests/python_integration/test_groupby_small_int_keys.py -v
```

Expected: 5 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/python_integration/test_groupby_small_int_keys.py
git commit -m "Tests: groupby on i8/i16/i32/u8/u16/u32 keys matches CPU

Capability F. 5 integration tests using assert_frame_equal against
engine='cpu'.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 2 — Binary expression unfolding inside `.agg()` (capability G)

`sum(a * b)` and similar shapes inside `.agg()` currently force the user to pre-project the intermediate column. M3 supports `*`, `+`, `-`, `/` on column/column and column/literal operands inline. The expression is captured in a new `AggSpec::Expression` IR variant; the fused-kernel template engine (Phase 3) consumes it and emits the elementwise math inside the per-row loop, so no intermediate column materializes.

This phase only lands the *plan-time representation* and walker rewriter. The *kernel-side consumption* lands in Phase 3.

**Plan-vs-code starting state (M2):**
- `AggSpec` in `crates/polars-metal-core/src/plan/mod.rs` is a **struct**: `pub struct AggSpec { pub input_col: String, pub op: AggOp, pub output_alias: String }`. `AggOp::Len` is a variant of `AggOp` (input_col convention: empty string).
- `ParsedAgg` in `crates/polars-metal-core/src/udf.rs` lines ~918–924 is the wire-format-parsed twin (same fields).
- The dispatch hot path (`dispatch_groupby` ~lines 1572–1659 in `udf.rs`) reads `agg.input_col`, `agg.op`, `agg.output_alias` directly.
- The walker (`python/polars_metal/_walker.py` `_walk_agg_expression`) emits `{"input_col", "op", "output_alias"}` per agg; `op=="Len"` with `input_col==""` for `pl.len()`.
- `output_dtype` does **not** exist on AggSpec/ParsedAgg in M2; per-agg output type is computed at dispatch time from the input column's dtype + op semantics. **Phase 2 preserves this — we do not add `output_dtype` to the IR.** Phase 3's `AggSignature` will compute it from the input dtypes already carried in the kernel layer (see Phase 3 patch).

Phase 2 converts `AggSpec` and `ParsedAgg` to **enums** with three variants — `Simple`, `Expression`, `Length` — and threads the change through the wire format, the parser, the dispatch loop, and the existing tests. This is the same "enum with three variants, no `output_dtype`" choice agreed for Phase 2; Phase 3 patches its own `AggSignature` to match.

### Task 8: Convert `AggSpec` to enum + define `AggExpr` IR

**Files:**
- Modify: `crates/polars-metal-core/src/plan/mod.rs` — add `BinaryOp`/`AggExpr`, convert `AggSpec` struct → enum
- Modify: `crates/polars-metal-core/src/udf.rs` — convert `ParsedAgg` struct → enum, update `parse_groupby_plan` to dispatch on `"kind"`, update `dispatch_groupby` field accesses to match arms
- Modify: `crates/polars-metal-core/tests/test_plan_groupby.rs` — existing struct-literal `AggSpec { ... }` constructions become `AggSpec::Simple { ... }` / `AggSpec::Length { ... }`
- Modify: `python/polars_metal/_walker.py` — emit a `"kind"` discriminator on agg dicts (`"Simple"` for column inputs, `"Length"` for `op=="Len"`; `"Expression"` lands in Task 9)
- Create: `crates/polars-metal-core/tests/test_agg_expr.rs` — new tests for the AggExpr IR

**Note on `output_dtype`:** intentionally omitted from every variant. Phase 3 computes output dtype from the input column dtype (already carried in the wire format as the column's metadata) + op semantics. Do not add an `output_dtype` field even though earlier drafts of the plan included one.

- [ ] **Step 1: Write the failing AggExpr test**

```rust
// crates/polars-metal-core/tests/test_agg_expr.rs
use polars_metal_native::plan::{AggExpr, AggOp, AggSpec, BinaryOp};

#[test]
fn agg_expr_column_literal_constructs() {
    let expr = AggExpr::Binary {
        op: BinaryOp::Mul,
        lhs: Box::new(AggExpr::Column("l_extendedprice".into())),
        rhs: Box::new(AggExpr::Binary {
            op: BinaryOp::Sub,
            lhs: Box::new(AggExpr::LiteralF64(1.0)),
            rhs: Box::new(AggExpr::Column("l_discount".into())),
        }),
    };
    let cols = expr.referenced_columns();
    assert_eq!(cols, vec!["l_extendedprice".to_string(), "l_discount".to_string()]);
}

#[test]
fn agg_spec_expression_carries_op_and_alias() {
    let spec = AggSpec::Expression {
        expr: AggExpr::Column("v".into()),
        op: AggOp::Sum,
        output_alias: "sum_v".into(),
    };
    match &spec {
        AggSpec::Expression { op, output_alias, .. } => {
            assert_eq!(*op, AggOp::Sum);
            assert_eq!(output_alias, "sum_v");
        }
        _ => panic!("expected Expression variant"),
    }
}

#[test]
fn agg_spec_length_carries_alias_only() {
    let spec = AggSpec::Length { output_alias: "n".into() };
    match &spec {
        AggSpec::Length { output_alias } => assert_eq!(output_alias, "n"),
        _ => panic!("expected Length variant"),
    }
}

#[test]
fn agg_spec_simple_carries_input_col() {
    let spec = AggSpec::Simple {
        input_col: "v".into(),
        op: AggOp::Sum,
        output_alias: "v_sum".into(),
    };
    match &spec {
        AggSpec::Simple { input_col, op, output_alias } => {
            assert_eq!(input_col, "v");
            assert_eq!(*op, AggOp::Sum);
            assert_eq!(output_alias, "v_sum");
        }
        _ => panic!("expected Simple variant"),
    }
}

#[test]
fn agg_expr_depth_check_rejects_overdeep_nesting() {
    // M3 caps expression depth at 4 to keep MSL emission bounded.
    let mut e = AggExpr::Column("v".into());
    for _ in 0..5 {
        e = AggExpr::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(e),
            rhs: Box::new(AggExpr::LiteralF64(0.0)),
        };
    }
    assert!(e.depth() > 4);
    assert!(matches!(e.validate(), Err(_)));
}

#[test]
fn agg_expr_depth_4_passes_validation() {
    // depth-4 expression: ((a + b) * (c - d)) — the kind of shape Q1 needs.
    let e = AggExpr::Binary {
        op: BinaryOp::Mul,
        lhs: Box::new(AggExpr::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(AggExpr::Column("a".into())),
            rhs: Box::new(AggExpr::Column("b".into())),
        }),
        rhs: Box::new(AggExpr::Binary {
            op: BinaryOp::Sub,
            lhs: Box::new(AggExpr::Column("c".into())),
            rhs: Box::new(AggExpr::Column("d".into())),
        }),
    };
    assert_eq!(e.depth(), 2);
    assert!(e.validate().is_ok());
}
```

- [ ] **Step 2: Convert `AggSpec` to enum + add `BinaryOp`/`AggExpr`**

```rust
// crates/polars-metal-core/src/plan/mod.rs (replaces the existing struct AggSpec)

/// Binary operations supported in inline aggregation expressions.
/// Capability G's scope: arithmetic only; no comparison / boolean / function calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp { Add, Sub, Mul, Div }

/// Expression tree consumed by the fused aggregation kernel.
/// Operands are columns or literals; operations are binary arithmetic.
#[derive(Debug, Clone, PartialEq)]
pub enum AggExpr {
    Column(String),
    LiteralF64(f64),
    LiteralI64(i64),
    Binary { op: BinaryOp, lhs: Box<AggExpr>, rhs: Box<AggExpr> },
}

impl AggExpr {
    /// All columns referenced anywhere in the tree, in left-to-right order,
    /// deduplicated. Used by the kernel template engine to know which
    /// buffers to bind.
    pub fn referenced_columns(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        self.walk(&mut |e| {
            if let AggExpr::Column(name) = e {
                if !out.iter().any(|n| n == name) { out.push(name.clone()); }
            }
        });
        out
    }

    pub fn depth(&self) -> usize {
        match self {
            AggExpr::Column(_) | AggExpr::LiteralF64(_) | AggExpr::LiteralI64(_) => 0,
            AggExpr::Binary { lhs, rhs, .. } => 1 + lhs.depth().max(rhs.depth()),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.depth() > 4 {
            return Err(format!("expression depth {} exceeds M3 cap of 4", self.depth()));
        }
        Ok(())
    }

    fn walk<F: FnMut(&AggExpr)>(&self, f: &mut F) {
        f(self);
        if let AggExpr::Binary { lhs, rhs, .. } = self {
            lhs.walk(f);
            rhs.walk(f);
        }
    }
}

/// Aggregation specification. Three variants:
/// - `Simple` — aggregate one input column (M2 shape).
/// - `Expression` — aggregate the value of an inline binary-arithmetic expression (M3, capability G).
/// - `Length` — `pl.len()`, counts rows per group; no input column read.
///
/// Output dtype is **not** carried here. The kernel layer derives it from
/// the input column dtype(s) + op semantics at dispatch / signature time.
#[derive(Debug, Clone, PartialEq)]
pub enum AggSpec {
    Simple {
        input_col: String,
        op: AggOp,
        output_alias: String,
    },
    Expression {
        expr: AggExpr,
        op: AggOp,
        output_alias: String,
    },
    Length {
        output_alias: String,
    },
}
```

**Migration:** delete the existing `pub struct AggSpec { ... }` and its doc comment. The `pub enum MetalPlanNode::GroupBy { aggs: Vec<AggSpec>, ... }` field type is unchanged (still `Vec<AggSpec>`). Replace the implementation note above the old struct with the new doc comment.

- [ ] **Step 3: Convert `ParsedAgg` to enum (mirror of AggSpec) in `udf.rs`**

`crates/polars-metal-core/src/udf.rs` ~lines 918–924 currently has:

```rust
pub struct ParsedAgg {
    pub input_col: String,
    pub op: AggOp,
    pub output_alias: String,
}
```

Replace with an enum that mirrors `AggSpec`:

```rust
#[derive(Debug, Clone)]
pub enum ParsedAgg {
    Simple { input_col: String, op: AggOp, output_alias: String },
    Expression { expr: AggExpr, op: AggOp, output_alias: String },
    Length { output_alias: String },
}

impl ParsedAgg {
    /// Convenience: the output alias regardless of variant (every variant
    /// has one; dispatch reads this for result-column naming).
    pub fn output_alias(&self) -> &str {
        match self {
            ParsedAgg::Simple { output_alias, .. }
            | ParsedAgg::Expression { output_alias, .. }
            | ParsedAgg::Length { output_alias } => output_alias,
        }
    }
}
```

Add `AggExpr` to the `use crate::plan::{...}` import at the top of `udf.rs` (it already imports `AggOp`, `MetalDtype`, `AggSpec`).

- [ ] **Step 4: Update `parse_groupby_plan` to dispatch on `"kind"`**

In `udf.rs` ~lines 992–1022, the loop that reads each agg dict currently extracts `input_col`/`op`/`output_alias` directly. Rewrite it to read an optional `"kind"` discriminator and dispatch:

```rust
// Inside the `for item in aggs_list.iter()` loop (replaces the body):
let entry: Bound<PyDict> = item
    .downcast_into()
    .map_err(|_| GroupByParseError::WrongType("agg entry"))?;

// Backwards-compatible read: missing "kind" means M2-shape Simple/Length
// (the existing wire format). Explicit "kind" means M3-shape; "Expression"
// requires an "expr" field whose parser lands in Task 9.
let kind: String = entry
    .get_item("kind")
    .ok()
    .flatten()
    .and_then(|v| v.extract().ok())
    .unwrap_or_else(|| {
        // Legacy shape: infer from op=="Len".
        let op_str: String = entry
            .get_item("op").ok().flatten()
            .and_then(|v| v.extract().ok())
            .unwrap_or_default();
        if op_str == "Len" { "Length".into() } else { "Simple".into() }
    });

let output_alias: String = entry
    .get_item("output_alias").ok().flatten()
    .and_then(|v| v.extract().ok())
    .ok_or(GroupByParseError::WrongType("output_alias"))?;

let parsed = match kind.as_str() {
    "Length" => ParsedAgg::Length { output_alias },
    "Simple" => {
        let input_col: String = entry
            .get_item("input_col").ok().flatten()
            .and_then(|v| v.extract().ok())
            .unwrap_or_default();
        let op_str: String = entry
            .get_item("op").ok().flatten()
            .and_then(|v| v.extract().ok())
            .ok_or(GroupByParseError::WrongType("op"))?;
        let op = AggOp::from_wire(&op_str).ok_or(GroupByParseError::UnknownOp(op_str))?;
        ParsedAgg::Simple { input_col, op, output_alias }
    }
    "Expression" => {
        // Expression-shape parser lands in Task 9; Phase 2 Task 8 leaves a stub
        // that returns an error so callers can detect it without a panic.
        return Err(GroupByParseError::WrongType(
            "AggSpec::Expression parsing not implemented until Task 9",
        ));
    }
    other => return Err(GroupByParseError::UnknownOp(format!("kind={other}"))),
};
aggs.push(parsed);
```

Task 9 fills in the `"Expression"` arm (parses the `expr` sub-tree from the dict).

- [ ] **Step 5: Update `dispatch_groupby` field accesses in `udf.rs`**

Around lines 1572–1659, the dispatch loop reads `agg.op`, `agg.input_col`, `agg.output_alias` directly on the old `ParsedAgg` struct. Convert to a `match` per agg. The current logic — early-exit for `Len`, otherwise look up `by_name.get(&agg.input_col)`, build a kind+vcol, push the operation — becomes:

```rust
// Inside the for-loop over `aggs`:
let (op, output_alias, input_col_opt) = match agg {
    ParsedAgg::Length { output_alias } => {
        // Len path: no value column read.
        ops.push(AggregateOp {
            kind: AggregateKind::Len,
            input_col_idx: i,
            output_alias: output_alias.clone(),
        });
        continue;
    }
    ParsedAgg::Simple { input_col, op, output_alias } => {
        (*op, output_alias.clone(), Some(input_col.as_str()))
    }
    ParsedAgg::Expression { .. } => {
        // Phase 3 wires this branch; Phase 2's router gate (Task 10) ensures
        // we never reach here at runtime — but defensively reject if we do.
        return Err(/* polars-error; mirror the surrounding error style */
            polars_error::polars_err!(
                ComputeError: "AggSpec::Expression dispatch awaits Phase 3 fused-kernel consumer"
            ));
    }
};
let input_col = input_col_opt.expect("Simple variant always has input_col");
// ... rest of M2's existing logic (look up dtype/data/valid, build kind+vcol,
// push the op into `ops`) goes here unchanged ...
```

The exact error-construction pattern (`polars_err!` vs. the local error type) — match what the rest of `dispatch_groupby` returns. If unsure, grep nearby for `return Err(` to see the established style.

- [ ] **Step 6: Update existing tests in `test_plan_groupby.rs`**

`crates/polars-metal-core/tests/test_plan_groupby.rs` currently has ~7 `AggSpec { input_col: ..., op: ..., output_alias: ... }` struct literals. Convert each:

- `AggSpec { input_col: "v".into(), op: AggOp::Sum, output_alias: "v_sum".into() }`
  → `AggSpec::Simple { input_col: "v".into(), op: AggOp::Sum, output_alias: "v_sum".into() }`
- Literals where `op == AggOp::Len` and `input_col == ""`:
  → `AggSpec::Length { output_alias: "n".into() }`

No semantic change; pure mechanical rewrite. Run `cargo test -p polars-metal-core --test test_plan_groupby -- --test-threads=1` and confirm all pre-existing tests still pass.

- [ ] **Step 7: Update walker emit to include `"kind"` discriminator**

In `python/polars_metal/_walker.py` `_walk_agg_expression` (~line 447), the function currently returns dicts like `{"input_col": "...", "op": "...", "output_alias": "..."}`. Add a `"kind"` key on both successful return paths:

```python
# Len branch (~line 482-487):
return {
    "kind": "Length",
    "output_alias": output_alias or "len",
}

# Simple branch (~line 525-529):
return {
    "kind": "Simple",
    "input_col": str(col_name),
    "op": op,
    "output_alias": output_alias or f"{col_name}_{op.lower()}",
}
```

The `"input_col"` and `"op"` fields are dropped from the Length branch (the parser no longer reads them for Length). Task 9 adds the `"Expression"` branch.

- [ ] **Step 8: Run the suite**

```bash
cargo test -p polars-metal-core --test test_agg_expr -- --test-threads=1
cargo test -p polars-metal-core --test test_plan_groupby -- --test-threads=1
cargo test -p polars-metal-core --test test_router_cost -- --test-threads=1
make wheel
pytest tests/python_integration/ -k "groupby" -v
```

Expected:
- `test_agg_expr` — 6 passes (new file).
- `test_plan_groupby` — same pre-existing tests pass (struct literals migrated).
- `test_router_cost` — unchanged; `Vec<AggSpec>::new()` is variant-agnostic.
- Existing Python groupby integration tests — unchanged behavior (walker emits `kind` discriminator; parser handles both old and new shapes; dispatch unaffected for Simple/Length).

- [ ] **Step 9: Commit**

```bash
git add crates/polars-metal-core/src/plan/mod.rs \
        crates/polars-metal-core/src/udf.rs \
        crates/polars-metal-core/tests/test_agg_expr.rs \
        crates/polars-metal-core/tests/test_plan_groupby.rs \
        python/polars_metal/_walker.py
git commit -m "Plan: AggSpec → enum + AggExpr IR (no output_dtype)

Capability G — IR layer. Three variants: Simple (M2 shape), Expression
(M3, populated in Task 9), Length (was AggOp::Len with empty input_col).
ParsedAgg mirrors the enum. Wire format gains a 'kind' discriminator;
parser is backwards-compatible (missing kind → infer Simple/Length).
output_dtype omitted by design — kernel layer derives it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 9: Walker — pattern-match `Agg(BinaryExpr(...))` and emit `kind: "Expression"`

**Files:**
- Modify: `python/polars_metal/_walker.py`
- Modify: `crates/polars-metal-core/src/udf.rs` (fill in the `"Expression"` arm of `parse_groupby_plan` left as a stub in Task 8)
- Create: `tests/python_integration/test_walker_expression_unfolding.py`

- [ ] **Step 1: Find M2's verified attribute names**

```bash
grep -n "view_expression\|BinaryExpr\|node_id\|getattr(.*,\s*['\"](op|left|right|value)" python/polars_metal/_walker.py
```

M2 already pattern-matches `BinaryExpr` for filter predicates (`_walk_predicate` and friends). Reuse the same attribute access pattern there (likely `inner.left`, `inner.right`, and either `inner.op` or an `Operator` lookup table) — do **not** invent new names.

- [ ] **Step 2: Implement the agg-expression extractor**

In `python/polars_metal/_walker.py`, alongside `_walk_agg_expression` (the function that currently returns Simple/Length dicts), add:

```python
# Polars' BinaryExpr op names. Mirrors the table M2 uses for predicate
# binary ops; agg unfolding only supports the four arithmetic ops.
# Use the same op-name discovery that the predicate path uses (M2 keys
# off str(op) yielding "Operator.Multiply" / "Operator.Plus" etc., or
# a function-name attribute — match the existing pattern).
_AGG_BINARY_OP_NAMES: dict[str, str] = {
    "Operator.Multiply": "Mul",
    "Operator.Plus":     "Add",
    "Operator.Minus":    "Sub",
    "Operator.Divide":   "Div",
    "Operator.TrueDivide": "Div",  # py-1.40.1 emits this for `/` on floats
}

_AGG_EXPR_MAX_DEPTH = 4


def _walk_agg_expr_node(nt, node_id, in_schema, depth):
    """Recursively lower one Polars expression sub-node to an AggExpr dict.

    Returns the dict on success; returns None if the shape is outside
    capability G (function calls, comparisons, deeper than _AGG_EXPR_MAX_DEPTH,
    unsupported binary ops, unknown literal types). Returning None causes
    the caller to fall back the whole agg.
    """
    if depth < 0:
        return None
    try:
        node = nt.view_expression(node_id)
    except Exception:
        return None
    cls = type(node).__name__

    if cls == "Column":
        col_name = getattr(node, "name", None)
        if col_name is None:
            return None
        # Validate the column exists in input schema (same check as Simple path).
        if in_schema.get(str(col_name)) is None:
            return None
        return {"kind": "Column", "name": str(col_name)}

    if cls == "Literal":
        val = getattr(node, "value", None)
        if isinstance(val, bool):  # bool is a subclass of int; reject explicitly
            return None
        if isinstance(val, float):
            return {"kind": "LiteralF64", "value": float(val)}
        if isinstance(val, int):
            return {"kind": "LiteralI64", "value": int(val)}
        return None

    if cls == "BinaryExpr":
        op = getattr(node, "op", None)
        op_key = str(op) if op is not None else ""
        op_tag = _AGG_BINARY_OP_NAMES.get(op_key)
        if op_tag is None:
            return None
        left_id = getattr(node, "left", None)
        right_id = getattr(node, "right", None)
        if left_id is None or right_id is None:
            return None
        lhs = _walk_agg_expr_node(nt, left_id, in_schema, depth - 1)
        if lhs is None:
            return None
        rhs = _walk_agg_expr_node(nt, right_id, in_schema, depth - 1)
        if rhs is None:
            return None
        return {"kind": "Binary", "op": op_tag, "lhs": lhs, "rhs": rhs}

    return None
```

**Implementation note:** M2's predicate walker uses `str(op)` to key into a name table (see `_CMP_OP_NAMES` and `_LOGICAL_OP_NAMES`). The exact strings (`"Operator.Multiply"` etc.) need to be verified against the running py-1.40.1 — if py-1.40.1 emits something different (e.g. `"Operator.Mul"`), update `_AGG_BINARY_OP_NAMES` to match. Run `python -c "import polars as pl; e = pl.col('a') * pl.col('b'); print(...)" ` style probes if uncertain.

- [ ] **Step 3: Wire the Expression branch into `_walk_agg_expression`**

The existing `_walk_agg_expression` function (the one that returns Simple/Length dicts) currently bails out (`return None`) for any non-`Column` inner expression. Extend it: when the inner expression is `BinaryExpr` (and the outer `Agg` `name` is one of the supported ops Sum/Mean/Count/Min/Max — `Len` doesn't take expressions), try the expression extractor:

```python
# In _walk_agg_expression, after the existing Column branch (~line 511-513) fails,
# before the final `return None`:
if inner_cls in ("BinaryExpr",):
    expr_dict = _walk_agg_expr_node(nt, arg_id, in_schema, _AGG_EXPR_MAX_DEPTH)
    if expr_dict is None:
        return None
    # Synthesize a default alias if the user didn't provide one. Polars'
    # default for `(a * b).sum()` is something like "a", but we can't
    # rely on that — fall back to a stable synthesised name.
    return {
        "kind": "Expression",
        "expr": expr_dict,
        "op": op,
        "output_alias": output_alias or f"expr_{op.lower()}",
    }
```

The Simple branch's existing `return {...}` dict gains `"kind": "Simple"` (added in Task 8 Step 7). The Length branch already has `"kind": "Length"` (Task 8 Step 7). The new branch above adds `"kind": "Expression"`.

- [ ] **Step 4: Fill in the `"Expression"` arm of `parse_groupby_plan` in `udf.rs`**

Task 8 left this arm returning `WrongType`. Replace with a recursive parser that consumes the `"expr"` sub-dict the walker emits:

```rust
// crates/polars-metal-core/src/udf.rs — extend the imports at top:
use crate::plan::{AggExpr, AggOp, AggSpec, BinaryOp, MetalDtype};

// Helper: recursively parse an AggExpr dict.
fn parse_agg_expr_dict(d: &Bound<PyDict>) -> Result<AggExpr, GroupByParseError> {
    let kind: String = d
        .get_item("kind").ok().flatten()
        .and_then(|v| v.extract().ok())
        .ok_or(GroupByParseError::WrongType("expr.kind"))?;
    match kind.as_str() {
        "Column" => {
            let name: String = d
                .get_item("name").ok().flatten()
                .and_then(|v| v.extract().ok())
                .ok_or(GroupByParseError::WrongType("expr.name"))?;
            Ok(AggExpr::Column(name))
        }
        "LiteralF64" => {
            let v: f64 = d
                .get_item("value").ok().flatten()
                .and_then(|v| v.extract().ok())
                .ok_or(GroupByParseError::WrongType("expr.value(f64)"))?;
            Ok(AggExpr::LiteralF64(v))
        }
        "LiteralI64" => {
            let v: i64 = d
                .get_item("value").ok().flatten()
                .and_then(|v| v.extract().ok())
                .ok_or(GroupByParseError::WrongType("expr.value(i64)"))?;
            Ok(AggExpr::LiteralI64(v))
        }
        "Binary" => {
            let op_str: String = d
                .get_item("op").ok().flatten()
                .and_then(|v| v.extract().ok())
                .ok_or(GroupByParseError::WrongType("expr.op"))?;
            let op = match op_str.as_str() {
                "Add" => BinaryOp::Add,
                "Sub" => BinaryOp::Sub,
                "Mul" => BinaryOp::Mul,
                "Div" => BinaryOp::Div,
                _ => return Err(GroupByParseError::UnknownOp(format!("binary op {op_str}"))),
            };
            let lhs_dict: Bound<PyDict> = d
                .get_item("lhs").ok().flatten()
                .ok_or(GroupByParseError::WrongType("expr.lhs"))?
                .downcast_into()
                .map_err(|_| GroupByParseError::WrongType("expr.lhs(dict)"))?;
            let rhs_dict: Bound<PyDict> = d
                .get_item("rhs").ok().flatten()
                .ok_or(GroupByParseError::WrongType("expr.rhs"))?
                .downcast_into()
                .map_err(|_| GroupByParseError::WrongType("expr.rhs(dict)"))?;
            Ok(AggExpr::Binary {
                op,
                lhs: Box::new(parse_agg_expr_dict(&lhs_dict)?),
                rhs: Box::new(parse_agg_expr_dict(&rhs_dict)?),
            })
        }
        other => Err(GroupByParseError::UnknownOp(format!("expr kind={other}"))),
    }
}
```

Then in `parse_groupby_plan`'s `"Expression"` arm (the stub from Task 8 Step 4), replace the error return with:

```rust
"Expression" => {
    let op_str: String = entry
        .get_item("op").ok().flatten()
        .and_then(|v| v.extract().ok())
        .ok_or(GroupByParseError::WrongType("op"))?;
    let op = AggOp::from_wire(&op_str).ok_or(GroupByParseError::UnknownOp(op_str))?;
    let expr_dict: Bound<PyDict> = entry
        .get_item("expr").ok().flatten()
        .ok_or(GroupByParseError::WrongType("expr"))?
        .downcast_into()
        .map_err(|_| GroupByParseError::WrongType("expr(dict)"))?;
    let expr = parse_agg_expr_dict(&expr_dict)?;
    // Apply the same depth cap as the IR; defence-in-depth in case the
    // walker emits something deeper than _AGG_EXPR_MAX_DEPTH.
    expr.validate().map_err(|_| GroupByParseError::WrongType("expr(too deep)"))?;
    ParsedAgg::Expression { expr, op, output_alias }
}
```

- [ ] **Step 5: Add the integration test**

```python
# tests/python_integration/test_walker_expression_unfolding.py
"""Verify Polars binary-expression aggregations lift to MetalPlanNode."""
import polars as pl
from polars.testing import assert_frame_equal

import polars_metal as pm


def test_sum_a_times_b_routes_gpu():
    """The Q1-shape sum(a * b) goes through Expression-rewriter."""
    df = pl.DataFrame({
        "k": [0, 0, 1, 1, 2, 2] * 1000,
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 1000,
        "b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] * 1000,
    })
    q = df.lazy().group_by("k").agg(
        (pl.col("a") * pl.col("b")).sum().alias("sum_ab"),
    )
    cpu = q.collect(engine="cpu").sort("k")
    metal = q.collect(engine=pm.MetalEngine()).sort("k")
    assert_frame_equal(cpu, metal)


def test_sum_a_times_one_minus_b():
    """Q1's disc_price shape: sum(a * (1 - b))."""
    df = pl.DataFrame({
        "k": [0, 0, 1, 1] * 5000,
        "a": [10.0, 20.0, 30.0, 40.0] * 5000,
        "b": [0.05, 0.1, 0.15, 0.2] * 5000,
    })
    q = df.lazy().group_by("k").agg(
        (pl.col("a") * (1.0 - pl.col("b"))).sum().alias("disc"),
    )
    cpu = q.collect(engine="cpu").sort("k")
    metal = q.collect(engine=pm.MetalEngine()).sort("k")
    assert_frame_equal(cpu, metal)


def test_unsupported_function_call_falls_back():
    """abs() inside agg → falls back to CPU; result still correct."""
    df = pl.DataFrame({"k": [0, 1, 0, 1], "v": [-1.0, 2.0, -3.0, 4.0]})
    q = df.lazy().group_by("k").agg(pl.col("v").abs().sum().alias("s"))
    cpu = q.collect(engine="cpu").sort("k")
    metal = q.collect(engine=pm.MetalEngine()).sort("k")
    assert_frame_equal(cpu, metal)


def test_depth_5_falls_back():
    df = pl.DataFrame({"k": [0]*100, "v": [1.0]*100})
    expr = pl.col("v")
    for _ in range(5):
        expr = expr + 1.0
    q = df.lazy().group_by("k").agg(expr.sum().alias("s"))
    # Should still produce correct result; either GPU (if router decides
    # depth-5 is ok despite our cap) or CPU fallback.
    cpu = q.collect(engine="cpu").sort("k")
    metal = q.collect(engine=pm.MetalEngine()).sort("k")
    assert_frame_equal(cpu, metal)
```

- [ ] **Step 6: Build + run**

```bash
make wheel
pytest tests/python_integration/test_walker_expression_unfolding.py -v
```

Expected: 4 passes. (The kernel-side consumption is Phase 3; at this point, `AggSpec::Expression` may still route as Fallback per Task 10's gate. Tests must still pass via CPU fallback.)

- [ ] **Step 7: Commit**

```bash
git add python/polars_metal/_walker.py \
        crates/polars-metal-core/src/udf.rs \
        tests/python_integration/test_walker_expression_unfolding.py
git commit -m "Walker + udf: extract binary expressions inside .agg() as Expression specs

Capability G — Python + wire-format side. Walker pattern-matches
BinaryExpr(Mul/Add/Sub/Div) recursively up to depth 4; emits
{kind: 'Expression', expr: {...}, op, output_alias}. udf.rs parses
the expr sub-tree into AggExpr. Unsupported shapes return None →
router falls back. Kernel-side consumption lands in Phase 3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 10: Router decision — Expression specs route GPU only when Phase 3 lands

**Files:**
- Modify: `crates/polars-metal-core/src/router/cost.rs`
- Modify: `crates/polars-metal-core/tests/test_router_cost.rs` (add a test that proves the gate fires)

- [ ] **Step 1: Add a temporary gate in `decide_groupby_with_keys`**

`decide_groupby_with_keys(n_rows, keys, _aggs)` already accepts an `_aggs: &[AggSpec]` parameter (currently unused — leading underscore). Rename to `aggs` and add a pre-check that fails any GroupBy whose agg list contains an Expression variant:

```rust
// crates/polars-metal-core/src/router/cost.rs

pub fn decide_groupby_with_keys(
    n_rows: usize,
    keys: &[(String, MetalDtype)],
    aggs: &[AggSpec],
) -> NodeDecision {
    // Phase 2 gate: Expression specs require the Phase 3 fused-kernel
    // consumer. Until that lands, fall back at plan time so callers
    // see a deterministic CPU path rather than a runtime panic.
    if aggs.iter().any(|a| matches!(a, AggSpec::Expression { .. })) {
        return NodeDecision::Fallback(
            "AggSpec::Expression awaiting Phase 3 fused-kernel consumer".into(),
        );
    }

    let total_bits: usize = keys.iter().map(|(_, d)| key_width_bits(*d)).sum();
    if total_bits > 128 {
        return NodeDecision::Fallback(format!(
            "composite key total {total_bits} bits; M2 supports ≤ 128"
        ));
    }
    decide_groupby(n_rows)
}
```

- [ ] **Step 2: Add a router test that proves the gate fires**

In `crates/polars-metal-core/tests/test_router_cost.rs`, add (alongside the existing tests):

```rust
use polars_metal_native::plan::{AggExpr, AggOp, AggSpec, BinaryOp, MetalDtype};
use polars_metal_native::router::cost::decide_groupby_with_keys;
use polars_metal_native::router::NodeDecision;

#[test]
fn router_falls_back_when_any_agg_is_expression() {
    let keys = vec![("k".to_string(), MetalDtype::I64)];
    let aggs = vec![
        AggSpec::Simple { input_col: "v".into(), op: AggOp::Sum, output_alias: "v_sum".into() },
        AggSpec::Expression {
            expr: AggExpr::Binary {
                op: BinaryOp::Mul,
                lhs: Box::new(AggExpr::Column("a".into())),
                rhs: Box::new(AggExpr::Column("b".into())),
            },
            op: AggOp::Sum,
            output_alias: "sum_ab".into(),
        },
    ];
    let decision = decide_groupby_with_keys(1_000_000, &keys, &aggs);
    match decision {
        NodeDecision::Fallback(reason) => {
            assert!(reason.contains("Expression"), "expected Expression reason, got: {reason}");
        }
        other => panic!("expected Fallback, got: {other:?}"),
    }
}

#[test]
fn router_passes_when_only_simple_and_length_aggs() {
    let keys = vec![("k".to_string(), MetalDtype::I64)];
    let aggs = vec![
        AggSpec::Simple { input_col: "v".into(), op: AggOp::Sum, output_alias: "v_sum".into() },
        AggSpec::Length { output_alias: "n".into() },
    ];
    let decision = decide_groupby_with_keys(1_000_000, &keys, &aggs);
    assert!(matches!(decision, NodeDecision::GpuLift));
}
```

Match the existing path-to-`NodeDecision` import style in the file. The exact path to `decide_groupby_with_keys` (`router::cost::...` vs. re-exported) follows whatever the existing tests in that file do.

- [ ] **Step 3: Run unit + integration tests**

```bash
cargo test -p polars-metal-core --test test_router_cost -- --test-threads=1
make wheel
pytest tests/python_integration/test_walker_expression_unfolding.py -v
```

Expected: router tests pass (new gate-firing test asserts the Fallback reason); integration tests pass — `sum(a*b)` shapes still produce correct results via CPU fallback (`assert_frame_equal` matches CPU).

- [ ] **Step 4: Commit**

```bash
git add crates/polars-metal-core/src/router/cost.rs crates/polars-metal-core/tests/test_router_cost.rs
git commit -m "Router: Expression aggs fall back until fused-kernel consumer lands

Capability G — temporary gate. Walker emits Expression specs in Tasks 8-9;
the router's decide_groupby_with_keys rejects them with a clear reason.
Phase 3 removes this gate when the fused kernel can emit the inline
arithmetic. Test asserts the Fallback fires.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 3 — Fused multi-aggregation kernel template (capability B)

The most novel piece of M3. M2 dispatches N kernels for N aggregations. M3 emits **one MSL kernel per query signature**, code-generated at plan time. The kernel reads each row once, loads each value column once, and updates all per-group accumulators in one pass. For Q1's 8 aggregations over 4 value columns: M2 dispatches 8 kernels with 16 value-column reads total; M3 dispatches 1 kernel with 4 reads.

This phase also unblocks capability G (expression unfolding), since the fused kernel is what consumes `AggSpec::Expression` inline.

### Task 11: `AggSignature` — hashable cache key for compiled fused kernels

**Files:**
- Create: `crates/polars-metal-kernels/src/aggregate_fused/mod.rs`
- Create: `crates/polars-metal-kernels/src/aggregate_fused/signature.rs`
- Create: `crates/polars-metal-kernels/tests/test_aggregate_fused_signature.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-kernels/tests/test_aggregate_fused_signature.rs
use polars_metal_core::plan::{AggExpr, AggOp, AggSpec, BinaryOp, MetalDtype};
use polars_metal_kernels::aggregate_fused::signature::AggSignature;

fn simple(col: &str, op: AggOp, alias: &str, dt: MetalDtype) -> AggSpec {
    AggSpec::Simple {
        input_column: col.into(),
        op,
        output_alias: alias.into(),
        output_dtype: dt,
    }
}

#[test]
fn signature_same_for_isomorphic_specs() {
    let a = AggSignature::from_specs(&[
        simple("v", AggOp::Sum, "sum_v", MetalDtype::F64),
        simple("v", AggOp::Mean, "mean_v", MetalDtype::F64),
    ]);
    // Same shape, different aliases — aliases must NOT affect signature.
    let b = AggSignature::from_specs(&[
        simple("v", AggOp::Sum, "anything_else", MetalDtype::F64),
        simple("v", AggOp::Mean, "doesnt_matter", MetalDtype::F64),
    ]);
    assert_eq!(a, b);
}

#[test]
fn signature_differs_when_op_set_differs() {
    let a = AggSignature::from_specs(&[simple("v", AggOp::Sum, "s", MetalDtype::F64)]);
    let b = AggSignature::from_specs(&[simple("v", AggOp::Mean, "m", MetalDtype::F64)]);
    assert_ne!(a, b);
}

#[test]
fn signature_differs_when_dtype_differs() {
    let a = AggSignature::from_specs(&[simple("v", AggOp::Sum, "s", MetalDtype::F32)]);
    let b = AggSignature::from_specs(&[simple("v", AggOp::Sum, "s", MetalDtype::F64)]);
    assert_ne!(a, b);
}

#[test]
fn signature_differs_when_column_count_differs() {
    let a = AggSignature::from_specs(&[
        simple("a", AggOp::Sum, "s", MetalDtype::F64),
        simple("b", AggOp::Sum, "t", MetalDtype::F64),
    ]);
    let b = AggSignature::from_specs(&[
        simple("a", AggOp::Sum, "s", MetalDtype::F64),
    ]);
    assert_ne!(a, b);
}

#[test]
fn signature_collapses_aliases_but_not_column_distinction() {
    // Two aggs over the *same* column should produce a signature that
    // shares the load; two aggs over different columns must differ.
    let same_col = AggSignature::from_specs(&[
        simple("a", AggOp::Sum, "s", MetalDtype::F64),
        simple("a", AggOp::Mean, "m", MetalDtype::F64),
    ]);
    let diff_col = AggSignature::from_specs(&[
        simple("a", AggOp::Sum, "s", MetalDtype::F64),
        simple("b", AggOp::Mean, "m", MetalDtype::F64),
    ]);
    assert_ne!(same_col, diff_col);
}

#[test]
fn signature_for_expression_includes_expr_shape() {
    let s_inline = AggSignature::from_specs(&[AggSpec::Expression {
        expr: AggExpr::Binary {
            op: BinaryOp::Mul,
            lhs: Box::new(AggExpr::Column("a".into())),
            rhs: Box::new(AggExpr::Column("b".into())),
        },
        op: AggOp::Sum,
        output_alias: "sum_ab".into(),
        output_dtype: MetalDtype::F64,
    }]);
    let s_simple = AggSignature::from_specs(&[
        simple("a", AggOp::Sum, "s", MetalDtype::F64),
    ]);
    assert_ne!(s_inline, s_simple);
}
```

- [ ] **Step 2: Implement `AggSignature`**

```rust
// crates/polars-metal-kernels/src/aggregate_fused/signature.rs
//! Hashable cache key for fused-aggregation kernel sources.
//!
//! Two query plans with isomorphic agg shapes (same per-column op set,
//! same dtypes, same expression structure — aliases ignored) share one
//! compiled MSL library. The key is canonicalized: column names are
//! replaced with indices in first-seen order to maximize cache hits.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

use polars_metal_core::plan::{AggExpr, AggOp, AggSpec, BinaryOp, MetalDtype};

/// Canonical signature of a fused-agg query. Identical shape ⇒ identical
/// signature even across different column names.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AggSignature {
    /// Per-column-slot dtype, in first-seen order from the agg specs.
    column_dtypes: Vec<MetalDtype>,
    /// Per-agg shape, with column references rewritten as slot indices.
    aggs: Vec<CanonicalAgg>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CanonicalAgg {
    Simple { col_slot: u16, op: AggOp, dtype: MetalDtype },
    Expression { expr: CanonicalExpr, op: AggOp, dtype: MetalDtype },
    Length,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CanonicalExpr {
    Column(u16),
    LiteralF64Bits(u64), // bits, not f64, for Hash/Eq
    LiteralI64(i64),
    Binary { op: BinaryOp, lhs: Box<CanonicalExpr>, rhs: Box<CanonicalExpr> },
}

impl AggSignature {
    pub fn from_specs(specs: &[AggSpec]) -> Self {
        let mut column_slots: BTreeMap<String, u16> = BTreeMap::new();
        let mut column_dtypes: Vec<MetalDtype> = Vec::new();
        let mut aggs = Vec::with_capacity(specs.len());

        fn intern(
            col: &str,
            dtype: MetalDtype,
            slots: &mut BTreeMap<String, u16>,
            dtypes: &mut Vec<MetalDtype>,
        ) -> u16 {
            if let Some(&s) = slots.get(col) { return s; }
            let s = dtypes.len() as u16;
            slots.insert(col.into(), s);
            dtypes.push(dtype);
            s
        }

        fn canon_expr(
            e: &AggExpr,
            input_dtype: MetalDtype,
            slots: &mut BTreeMap<String, u16>,
            dtypes: &mut Vec<MetalDtype>,
        ) -> CanonicalExpr {
            match e {
                AggExpr::Column(name) => {
                    CanonicalExpr::Column(intern(name, input_dtype, slots, dtypes))
                }
                AggExpr::LiteralF64(v) => CanonicalExpr::LiteralF64Bits(v.to_bits()),
                AggExpr::LiteralI64(v) => CanonicalExpr::LiteralI64(*v),
                AggExpr::Binary { op, lhs, rhs } => CanonicalExpr::Binary {
                    op: *op,
                    lhs: Box::new(canon_expr(lhs, input_dtype, slots, dtypes)),
                    rhs: Box::new(canon_expr(rhs, input_dtype, slots, dtypes)),
                },
            }
        }

        for spec in specs {
            let canonical = match spec {
                AggSpec::Simple { input_column, op, output_dtype, .. } => {
                    let s = intern(input_column, *output_dtype, &mut column_slots, &mut column_dtypes);
                    CanonicalAgg::Simple { col_slot: s, op: *op, dtype: *output_dtype }
                }
                AggSpec::Expression { expr, op, output_dtype, .. } => {
                    let ce = canon_expr(expr, *output_dtype, &mut column_slots, &mut column_dtypes);
                    CanonicalAgg::Expression { expr: ce, op: *op, dtype: *output_dtype }
                }
                AggSpec::Length { .. } => CanonicalAgg::Length,
            };
            aggs.push(canonical);
        }
        Self { column_dtypes, aggs }
    }

    /// Stable 64-bit hash for use as the library-cache key.
    pub fn hash64(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }

    pub fn column_count(&self) -> usize { self.column_dtypes.len() }
    pub fn agg_count(&self) -> usize { self.aggs.len() }
}
```

- [ ] **Step 3: Wire `aggregate_fused` into the kernels crate**

```rust
// crates/polars-metal-kernels/src/aggregate_fused/mod.rs
//! Fused multi-aggregation kernel: MSL template engine + library cache.
//!
//! See spec § B (Fused multi-aggregation kernel template).
//!
//! Lifecycle per query:
//!   1. AggSignature::from_specs(aggs) → cache key
//!   2. emit_msl_for(signature) → MSL source (if not in cache)
//!   3. compile via MTLDevice::newLibraryWithSource → cached
//!   4. dispatch with bound buffers per signature's column order

pub mod signature;
pub mod emitter;
pub mod cache;
```

Add the corresponding `pub mod aggregate_fused;` line to `crates/polars-metal-kernels/src/lib.rs`.

- [ ] **Step 4: Run test**

```bash
cargo test -p polars-metal-kernels --test test_aggregate_fused_signature -- --test-threads=1
```

Expected: 6 passes.

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-kernels/src/aggregate_fused/ crates/polars-metal-kernels/src/lib.rs crates/polars-metal-kernels/tests/test_aggregate_fused_signature.rs
git commit -m "Kernel: AggSignature canonical cache key for fused kernels

Capability B. Hashable signature collapses aliases and column names
to slot indices; preserves shape, ops, dtypes, expression structure.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 12: MSL template emitter — Simple aggs (no Expression yet)

**Files:**
- Create: `crates/polars-metal-kernels/src/aggregate_fused/emitter.rs`
- Create: `crates/polars-metal-kernels/tests/test_aggregate_fused_emitter.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-kernels/tests/test_aggregate_fused_emitter.rs
use polars_metal_core::plan::{AggOp, AggSpec, MetalDtype};
use polars_metal_kernels::aggregate_fused::emitter::emit_msl;
use polars_metal_kernels::aggregate_fused::signature::AggSignature;

fn simple(col: &str, op: AggOp, dt: MetalDtype) -> AggSpec {
    AggSpec::Simple {
        input_column: col.into(),
        op,
        output_alias: format!("{op:?}_{col}"),
        output_dtype: dt,
    }
}

#[test]
fn emitted_kernel_compiles() {
    let specs = vec![
        simple("a", AggOp::Sum,  MetalDtype::F32),
        simple("a", AggOp::Mean, MetalDtype::F32),
    ];
    let sig = AggSignature::from_specs(&specs);
    let src = emit_msl(&sig, &specs);

    // Smoke check: produces a kernel entry point and binds the expected
    // number of buffers.
    assert!(src.contains("kernel void aggregate_fused"));
    assert!(src.contains("device const uint*  row_to_group"));
    // 1 value column + (sum out, count out for mean) = 3 buffers besides keys
    let buffer_count = src.matches("[[buffer(").count();
    assert!(buffer_count >= 4, "got {buffer_count} buffer bindings:\n{src}");

    // Compile via Metal to verify well-formedness.
    polars_metal_buffer::test_helpers::compile_msl(&src)
        .expect("emitted MSL must compile");
}

#[test]
fn fused_sum_only_emits_one_atomic_per_row() {
    let specs = vec![simple("v", AggOp::Sum, MetalDtype::F32)];
    let sig = AggSignature::from_specs(&specs);
    let src = emit_msl(&sig, &specs);
    let atomics = src.matches("atomic_fetch_add_explicit").count();
    assert_eq!(atomics, 1, "expected 1 atomic, got {atomics}:\n{src}");
}

#[test]
fn fused_sum_mean_count_shares_load_once() {
    let specs = vec![
        simple("v", AggOp::Sum,   MetalDtype::F32),
        simple("v", AggOp::Mean,  MetalDtype::F32),
        simple("v", AggOp::Count, MetalDtype::I32),
    ];
    let sig = AggSignature::from_specs(&specs);
    let src = emit_msl(&sig, &specs);
    // The value load for "v" must appear exactly once (shared across
    // three aggs).
    let loads = src.matches("value_0[gid]").count();
    assert_eq!(loads, 1, "expected 1 shared load, got {loads}:\n{src}");
}
```

(`polars_metal_buffer::test_helpers::compile_msl` is a small helper that exists today — check `crates/polars-metal-buffer/src/test_helpers.rs`. If it doesn't, add it: it wraps `MTLDevice::newLibraryWithSource_options_error_` and returns `Result<(), String>`.)

- [ ] **Step 2: Implement the emitter**

```rust
// crates/polars-metal-kernels/src/aggregate_fused/emitter.rs
//! MSL template emitter. Given an AggSignature + the original specs,
//! produces an MSL source string with one fused kernel.
//!
//! Per spec § B: each value column loaded once; each agg over that
//! column updates its output via 32-bit atomics. 64-bit accumulators
//! finalize on CPU (M2's pattern preserved).

use polars_metal_core::plan::{AggOp, AggSpec, MetalDtype};
use super::signature::AggSignature;

pub fn emit_msl(sig: &AggSignature, specs: &[AggSpec]) -> String {
    let mut s = String::new();
    s.push_str("#include <metal_stdlib>\n");
    s.push_str("#include <metal_atomic>\n");
    s.push_str("using namespace metal;\n\n");

    // Compute buffer slot indices.
    // Slots: 0 = row_to_group, 1 = n_rows
    // Then 2..2+C = value columns (32-bit each)
    // Then 2+C..2+2C = validity bitmaps per column
    // Then ...= output buffers, one per agg.
    let n_cols = sig.column_count();
    let n_aggs = sig.agg_count();

    let mut slot = 0usize;
    let row_to_group_slot = slot; slot += 1;
    let n_rows_slot = slot; slot += 1;

    let mut value_slots: Vec<usize> = Vec::with_capacity(n_cols);
    for _ in 0..n_cols {
        value_slots.push(slot);
        slot += 1;
    }
    let mut validity_slots: Vec<usize> = Vec::with_capacity(n_cols);
    for _ in 0..n_cols {
        validity_slots.push(slot);
        slot += 1;
    }

    // Output buffers and their slots (one per agg; mean needs 2: sum + count).
    let mut output_slots: Vec<Vec<usize>> = Vec::with_capacity(n_aggs);
    for spec in specs {
        let mut outs = Vec::new();
        match spec {
            AggSpec::Simple { op, .. } | AggSpec::Expression { op, .. } => {
                outs.push(slot); slot += 1;
                if *op == AggOp::Mean {
                    outs.push(slot); slot += 1;  // count buffer for mean
                }
            }
            AggSpec::Length { .. } => {
                outs.push(slot); slot += 1;
            }
        }
        output_slots.push(outs);
    }

    // Emit kernel signature.
    s.push_str("kernel void aggregate_fused(\n");
    s.push_str(&format!("  device const uint*  row_to_group [[buffer({})]],\n", row_to_group_slot));
    s.push_str(&format!("  device const uint*  n_rows       [[buffer({})]],\n", n_rows_slot));
    for (i, sl) in value_slots.iter().enumerate() {
        let ty = msl_type_for_value_load(&specs, i);
        s.push_str(&format!("  device const {ty}*  value_{i}    [[buffer({sl})]],\n"));
    }
    for (i, sl) in validity_slots.iter().enumerate() {
        s.push_str(&format!("  device const uchar* validity_{i} [[buffer({sl})]],\n"));
    }
    for (a, outs) in output_slots.iter().enumerate() {
        for (j, sl) in outs.iter().enumerate() {
            let ty = msl_atomic_type_for_agg(&specs[a], j);
            s.push_str(&format!("  device {ty}*  out_{a}_{j}     [[buffer({sl})]],\n"));
        }
    }
    s.push_str("  uint gid [[thread_position_in_grid]])\n{\n");

    // Bounds check.
    s.push_str("  if (gid >= n_rows[0]) return;\n");
    s.push_str("  uint g = row_to_group[gid];\n\n");

    // Per-column load + per-agg update.
    for (col_idx, _) in value_slots.iter().enumerate() {
        // Find which aggs reference this column. For Simple, that's the input column.
        // (Expression handled in Task 13.)
        s.push_str(&format!("  // --- value_{col_idx} ---\n"));
        s.push_str(&format!("  uchar val_{col_idx}_valid = (validity_{col_idx}[gid >> 3] >> (gid & 7)) & 1u;\n"));
        s.push_str(&format!("  auto val_{col_idx} = value_{col_idx}[gid];\n"));
        for (a, spec) in specs.iter().enumerate() {
            if column_index_for(spec, sig).map_or(false, |c| c == col_idx) {
                s.push_str(&emit_agg_update(spec, a, col_idx, &output_slots[a]));
            }
        }
    }

    // Length aggs (no value column).
    for (a, spec) in specs.iter().enumerate() {
        if matches!(spec, AggSpec::Length { .. }) {
            s.push_str(&format!(
                "  atomic_fetch_add_explicit((device atomic_uint*)&out_{a}_0[g], 1u, memory_order_relaxed);\n"
            ));
        }
    }

    s.push_str("}\n");
    s
}

fn msl_type_for_value_load(specs: &[AggSpec], col_idx: usize) -> &'static str {
    // Determined by the dtype of the first agg referencing this column.
    // M3 hot path is 32-bit; 64-bit accumulators use 32-bit value loads
    // with CPU finalize. Default to float for safety; integer-only ops
    // can override.
    for spec in specs {
        if let AggSpec::Simple { output_dtype, .. } = spec {
            return match output_dtype {
                MetalDtype::I32 | MetalDtype::I64 => "int",
                MetalDtype::U32 | MetalDtype::U64 => "uint",
                _ => "float",
            };
        }
    }
    "float"
}

fn msl_atomic_type_for_agg(spec: &AggSpec, j: usize) -> &'static str {
    // M3: 32-bit atomics only on Apple Silicon.
    // out_a_0 = primary (sum/min/max/count); out_a_1 (mean's count) = uint.
    match spec {
        AggSpec::Simple { op, output_dtype, .. }
        | AggSpec::Expression { op, output_dtype, .. } => match (op, j) {
            (AggOp::Mean, 1) => "atomic_uint",  // count of non-null
            (AggOp::Count, _) => "atomic_uint",
            _ => match output_dtype {
                MetalDtype::I32 | MetalDtype::I64 => "atomic_int",
                _ => "atomic_uint",  // float sums via uint reinterpretation
                                      // (use Metal's atomic_float when available;
                                      // here we accumulate to atomic_uint via
                                      // float-bit-pattern compare-exchange)
            },
        },
        AggSpec::Length { .. } => "atomic_uint",
    }
}

fn column_index_for(spec: &AggSpec, sig: &AggSignature) -> Option<usize> {
    // For Simple: returns the slot index from the signature canonicalization.
    // For Expression: returns None (Task 13 handles).
    // For Length: returns None.
    match spec {
        AggSpec::Simple { input_column, .. } => {
            // Walk sig.column_dtypes order; the first column wins slot 0, etc.
            // We need access to the original column-name ordering; the cleanest
            // solution is to expose this on AggSignature. For now, iterate specs
            // in-order and dedup the column names: slot index = position of
            // first occurrence.
            let mut seen: Vec<&str> = Vec::new();
            for s2 in [/* original specs */].iter() {
                // Stub — see implementation note below.
                let _ = (s2, &mut seen);
            }
            // For Task 12 simplicity: store column_order on the signature.
            // Implemented in the actual code; tests verify behavior.
            None
        }
        _ => None,
    }
}

fn emit_agg_update(spec: &AggSpec, agg_idx: usize, col_idx: usize, out_slots: &[usize]) -> String {
    let mut s = String::new();
    match spec {
        AggSpec::Simple { op, .. } => match op {
            AggOp::Sum => {
                s.push_str(&format!(
                    "  if (val_{col_idx}_valid) {{\n\
                     \    atomic_fetch_add_explicit(\n\
                     \      (device atomic_float*)&out_{agg_idx}_0[g],\n\
                     \      (float)val_{col_idx},\n\
                     \      memory_order_relaxed);\n\
                     \  }}\n"
                ));
            }
            AggOp::Count => {
                s.push_str(&format!(
                    "  if (val_{col_idx}_valid) {{\n\
                     \    atomic_fetch_add_explicit(\n\
                     \      (device atomic_uint*)&out_{agg_idx}_0[g],\n\
                     \      1u, memory_order_relaxed);\n\
                     \  }}\n"
                ));
            }
            AggOp::Mean => {
                // mean = sum + count; CPU finalizes the division.
                s.push_str(&format!(
                    "  if (val_{col_idx}_valid) {{\n\
                     \    atomic_fetch_add_explicit(\n\
                     \      (device atomic_float*)&out_{agg_idx}_0[g],\n\
                     \      (float)val_{col_idx}, memory_order_relaxed);\n\
                     \    atomic_fetch_add_explicit(\n\
                     \      (device atomic_uint*)&out_{agg_idx}_1[g],\n\
                     \      1u, memory_order_relaxed);\n\
                     \  }}\n"
                ));
            }
            AggOp::Min => {
                // 32-bit float min via atomic compare-exchange loop.
                s.push_str(&format!(
                    "  if (val_{col_idx}_valid) {{\n\
                     \    uint cur, desired;\n\
                     \    float curf, valf = (float)val_{col_idx};\n\
                     \    do {{\n\
                     \      cur = atomic_load_explicit(\n\
                     \        (device atomic_uint*)&out_{agg_idx}_0[g],\n\
                     \        memory_order_relaxed);\n\
                     \      curf = as_type<float>(cur);\n\
                     \      if (curf <= valf) break;\n\
                     \      desired = as_type<uint>(valf);\n\
                     \    }} while (!atomic_compare_exchange_weak_explicit(\n\
                     \      (device atomic_uint*)&out_{agg_idx}_0[g],\n\
                     \      &cur, desired,\n\
                     \      memory_order_relaxed, memory_order_relaxed));\n\
                     \  }}\n"
                ));
            }
            AggOp::Max => {
                // Mirror of Min, with reversed comparison.
                s.push_str(&format!(
                    "  if (val_{col_idx}_valid) {{\n\
                     \    uint cur, desired;\n\
                     \    float curf, valf = (float)val_{col_idx};\n\
                     \    do {{\n\
                     \      cur = atomic_load_explicit(\n\
                     \        (device atomic_uint*)&out_{agg_idx}_0[g],\n\
                     \        memory_order_relaxed);\n\
                     \      curf = as_type<float>(cur);\n\
                     \      if (curf >= valf) break;\n\
                     \      desired = as_type<uint>(valf);\n\
                     \    }} while (!atomic_compare_exchange_weak_explicit(\n\
                     \      (device atomic_uint*)&out_{agg_idx}_0[g],\n\
                     \      &cur, desired,\n\
                     \      memory_order_relaxed, memory_order_relaxed));\n\
                     \  }}\n"
                ));
            }
        },
        AggSpec::Expression { .. } => {
            // Task 13.
        }
        AggSpec::Length { .. } => {}
    }
    s
}
```

**Implementation note:** The stub `column_index_for` should be replaced by exposing the column ordering on `AggSignature`. Add a method `pub fn column_order(&self) -> &[&str]` that returns first-seen column names. The emitter passes the original specs alongside the signature to keep alias info available.

- [ ] **Step 3: Build + run**

```bash
cargo test -p polars-metal-kernels --test test_aggregate_fused_emitter -- --test-threads=1
```

Expected: 3 passes. If MSL compile fails, inspect the emitted source by adding `eprintln!("{src}")` temporarily and iterate.

- [ ] **Step 4: Commit**

```bash
git add crates/polars-metal-kernels/src/aggregate_fused/emitter.rs crates/polars-metal-kernels/tests/test_aggregate_fused_emitter.rs
git commit -m "Kernel: fused-agg MSL emitter for Simple aggs (sum/mean/count/min/max)

Capability B (Simple cases). One kernel per signature; one value-column
load shared across all aggs over that column. 32-bit atomics throughout;
mean returns (sum, count) for CPU-side division.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 13: MSL emitter — Expression aggs (capability G integration)

**Files:**
- Modify: `crates/polars-metal-kernels/src/aggregate_fused/emitter.rs`
- Create: `crates/polars-metal-kernels/tests/test_aggregate_fused_expression.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-kernels/tests/test_aggregate_fused_expression.rs
use polars_metal_core::plan::{AggExpr, AggOp, AggSpec, BinaryOp, MetalDtype};
use polars_metal_kernels::aggregate_fused::emitter::emit_msl;
use polars_metal_kernels::aggregate_fused::signature::AggSignature;

#[test]
fn sum_a_mul_b_emits_one_kernel_with_both_loads() {
    let specs = vec![AggSpec::Expression {
        expr: AggExpr::Binary {
            op: BinaryOp::Mul,
            lhs: Box::new(AggExpr::Column("a".into())),
            rhs: Box::new(AggExpr::Column("b".into())),
        },
        op: AggOp::Sum,
        output_alias: "sum_ab".into(),
        output_dtype: MetalDtype::F32,
    }];
    let sig = AggSignature::from_specs(&specs);
    let src = emit_msl(&sig, &specs);

    // Both column loads must appear.
    assert!(src.contains("value_0[gid]"), "missing value_0 load:\n{src}");
    assert!(src.contains("value_1[gid]"), "missing value_1 load:\n{src}");
    // The multiplication itself.
    assert!(src.contains("* (float)val_1") || src.contains("* val_1") || src.contains("val_0 * val_1"));

    polars_metal_buffer::test_helpers::compile_msl(&src)
        .expect("emitted MSL with expression must compile");
}

#[test]
fn sum_a_mul_one_minus_b_emits_literal_subtraction() {
    let specs = vec![AggSpec::Expression {
        expr: AggExpr::Binary {
            op: BinaryOp::Mul,
            lhs: Box::new(AggExpr::Column("a".into())),
            rhs: Box::new(AggExpr::Binary {
                op: BinaryOp::Sub,
                lhs: Box::new(AggExpr::LiteralF64(1.0)),
                rhs: Box::new(AggExpr::Column("b".into())),
            }),
        },
        op: AggOp::Sum,
        output_alias: "disc".into(),
        output_dtype: MetalDtype::F32,
    }];
    let sig = AggSignature::from_specs(&specs);
    let src = emit_msl(&sig, &specs);

    assert!(src.contains("1.0f"), "missing literal:\n{src}");
    polars_metal_buffer::test_helpers::compile_msl(&src)
        .expect("Q1 disc_price expression must compile");
}
```

- [ ] **Step 2: Implement expression-to-MSL conversion**

Extend `emit_agg_update` to recognize `AggSpec::Expression { expr, op, .. }`:

```rust
fn emit_expr_msl(expr: &AggExpr, col_order: &[String]) -> String {
    match expr {
        AggExpr::Column(name) => {
            let idx = col_order.iter().position(|c| c == name).expect("column registered");
            format!("(float)val_{idx}")
        }
        AggExpr::LiteralF64(v) => format!("{v}f"),
        AggExpr::LiteralI64(v) => format!("(float){v}"),
        AggExpr::Binary { op, lhs, rhs } => {
            let l = emit_expr_msl(lhs, col_order);
            let r = emit_expr_msl(rhs, col_order);
            let op_str = match op {
                BinaryOp::Add => "+",
                BinaryOp::Sub => "-",
                BinaryOp::Mul => "*",
                BinaryOp::Div => "/",
            };
            format!("({l} {op_str} {r})")
        }
    }
}

fn emit_expr_validity_check(expr: &AggExpr, col_order: &[String]) -> String {
    // AND of all referenced columns' validity bits.
    let cols = expr.referenced_columns();
    if cols.is_empty() { return "1u".into(); }
    let mut parts = Vec::new();
    for c in cols {
        let idx = col_order.iter().position(|x| x == &c).expect("registered");
        parts.push(format!("val_{idx}_valid"));
    }
    parts.join(" & ")
}
```

Then in `emit_agg_update`, when the spec is `AggSpec::Expression`, emit a load of each referenced column (deduplicated; the per-column load section already handles this), evaluate the expression inline, and apply the agg op exactly like the Simple case.

Place this *outside* the per-column loop in the kernel — Expression aggs are emitted in their own section after all column loads have been issued (since they may reference multiple columns).

- [ ] **Step 3: Update Task 10's router gate**

Remove the temporary "Expression falls back" gate:

```rust
// crates/polars-metal-core/src/router/cost.rs
// Delete the for-loop that fell back on AggSpec::Expression.
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p polars-metal-kernels --test test_aggregate_fused_expression -- --test-threads=1
```

Expected: 2 passes.

- [ ] **Step 5: Re-run Phase 2's walker test on GPU path**

```bash
make wheel
pytest tests/python_integration/test_walker_expression_unfolding.py -v
```

Expected: same 4 pass, but now the GPU path runs (verify via `MetalEngine(debug=True)`).

- [ ] **Step 6: Commit**

```bash
git add crates/polars-metal-kernels/src/aggregate_fused/emitter.rs crates/polars-metal-core/src/router/cost.rs crates/polars-metal-kernels/tests/test_aggregate_fused_expression.rs
git commit -m "Kernel + Router: AggSpec::Expression compiles to inline MSL math

Capability G (full). Binary arithmetic emits per-row evaluation; routes
through validity-AND across referenced columns. Phase 2's temporary
router fallback removed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 14: Library cache — compile fused kernels lazily, key by signature hash

**Files:**
- Create: `crates/polars-metal-kernels/src/aggregate_fused/cache.rs`
- Create: `crates/polars-metal-kernels/tests/test_fused_library_cache.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-kernels/tests/test_fused_library_cache.rs
use polars_metal_core::plan::{AggOp, AggSpec, MetalDtype};
use polars_metal_kernels::aggregate_fused::cache::FusedLibraryCache;
use polars_metal_kernels::aggregate_fused::signature::AggSignature;
use polars_metal_buffer::MetalDevice;

fn simple(col: &str, op: AggOp, dt: MetalDtype) -> AggSpec {
    AggSpec::Simple {
        input_column: col.into(), op,
        output_alias: "x".into(), output_dtype: dt,
    }
}

#[test]
fn cache_returns_same_pipeline_for_isomorphic_signatures() {
    let device = MetalDevice::system_default().unwrap();
    let cache = FusedLibraryCache::new(&device);

    let specs1 = vec![simple("a", AggOp::Sum, MetalDtype::F32)];
    let specs2 = vec![simple("b", AggOp::Sum, MetalDtype::F32)];
    let sig1 = AggSignature::from_specs(&specs1);
    let sig2 = AggSignature::from_specs(&specs2);
    assert_eq!(sig1, sig2);

    let p1 = cache.get_or_compile(&sig1, &specs1).expect("compile 1");
    let p2 = cache.get_or_compile(&sig2, &specs2).expect("compile 2 (cache hit)");
    // ComputePipelineState handles compare via pointer identity in our wrapper.
    assert!(std::ptr::eq(p1.as_ref(), p2.as_ref()), "cache should reuse compiled library");
}

#[test]
fn cache_compiles_distinct_for_different_signatures() {
    let device = MetalDevice::system_default().unwrap();
    let cache = FusedLibraryCache::new(&device);

    let specs1 = vec![simple("v", AggOp::Sum,  MetalDtype::F32)];
    let specs2 = vec![simple("v", AggOp::Mean, MetalDtype::F32)];
    let sig1 = AggSignature::from_specs(&specs1);
    let sig2 = AggSignature::from_specs(&specs2);
    assert_ne!(sig1, sig2);

    let p1 = cache.get_or_compile(&sig1, &specs1).expect("compile 1");
    let p2 = cache.get_or_compile(&sig2, &specs2).expect("compile 2");
    assert!(!std::ptr::eq(p1.as_ref(), p2.as_ref()));
}
```

- [ ] **Step 2: Implement the cache**

```rust
// crates/polars-metal-kernels/src/aggregate_fused/cache.rs
//! Library cache: compile MSL once per AggSignature, reuse across queries.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use polars_metal_buffer::{MetalDevice, ComputePipeline};
use polars_metal_core::plan::AggSpec;

use super::emitter::emit_msl;
use super::signature::AggSignature;

pub struct FusedLibraryCache {
    device: MetalDevice,
    by_hash: Mutex<HashMap<u64, Arc<ComputePipeline>>>,
}

impl FusedLibraryCache {
    pub fn new(device: &MetalDevice) -> Self {
        Self {
            device: device.clone(),
            by_hash: Mutex::new(HashMap::new()),
        }
    }

    pub fn get_or_compile(
        &self,
        sig: &AggSignature,
        specs: &[AggSpec],
    ) -> Result<Arc<ComputePipeline>, String> {
        let h = sig.hash64();
        if let Some(p) = self.by_hash.lock().unwrap().get(&h) {
            return Ok(p.clone());
        }
        let src = emit_msl(sig, specs);
        let lib = self.device.new_library_from_source(&src)
            .map_err(|e| format!("MSL compile failed: {e}\n--- source ---\n{src}"))?;
        let pso = self.device.pipeline_for_function(&lib, "aggregate_fused")
            .map_err(|e| format!("pipeline creation failed: {e}"))?;
        let arc = Arc::new(pso);
        self.by_hash.lock().unwrap().insert(h, arc.clone());
        Ok(arc)
    }

    pub fn warmup(&self, signatures: &[(AggSignature, Vec<AggSpec>)]) {
        for (sig, specs) in signatures {
            let _ = self.get_or_compile(sig, specs);
        }
    }
}
```

(`new_library_from_source` and `pipeline_for_function` likely already exist on M2's `MetalDevice` from `crates/polars-metal-buffer/`; if not, add wrappers around `objc2-metal`'s `MTLDevice::newLibraryWithSource_options_error_` and `MTLDevice::newComputePipelineStateWithFunction_error_`.)

- [ ] **Step 3: Run test**

```bash
cargo test -p polars-metal-kernels --test test_fused_library_cache -- --test-threads=1
```

Expected: 2 passes.

- [ ] **Step 4: Commit**

```bash
git add crates/polars-metal-kernels/src/aggregate_fused/cache.rs crates/polars-metal-kernels/tests/test_fused_library_cache.rs
git commit -m "Kernel: FusedLibraryCache compiles MSL lazily, keyed by signature hash

Capability B. Same signature reuses one compiled MTLComputePipelineState
across queries. Includes warmup() for pre-compilation at module import.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 15: Wire fused kernel through groupby pipeline (replaces M2's per-agg)

**Files:**
- Modify: `crates/polars-metal-kernels/src/groupby.rs`
- Modify: `crates/polars-metal-core/src/udf.rs`

- [ ] **Step 1: Add fused dispatch to `groupby.rs`**

Find M2's `execute_groupby_aggregation` (the function that loops over `aggs` and dispatches per-agg kernels). Replace the loop with a single fused dispatch:

```rust
// crates/polars-metal-kernels/src/groupby.rs (excerpt)
use crate::aggregate_fused::{
    cache::FusedLibraryCache, signature::AggSignature,
};

pub fn execute_groupby_aggregation_fused(
    device: &MetalDevice,
    cache: &FusedLibraryCache,
    queue: &CommandQueue,
    row_to_group: &MetalBuffer<u32>,
    n_groups: u32,
    value_columns: &[(MetalBuffer<u8>, MetalBuffer<u8>)],  // (data, validity)
    aggs: &[AggSpec],
) -> Result<Vec<AggOutput>, EngineError> {
    let sig = AggSignature::from_specs(aggs);
    let pso = cache.get_or_compile(&sig, aggs)
        .map_err(|e| EngineError::Compute(e))?;

    // Allocate output buffers (one per agg; mean gets 2: sum + count).
    let mut output_buffers: Vec<MetalBuffer<u8>> = Vec::new();
    for spec in aggs {
        match spec {
            AggSpec::Simple { op, output_dtype, .. }
            | AggSpec::Expression { op, output_dtype, .. } => {
                output_buffers.push(allocate_output(device, *op, *output_dtype, n_groups));
                if *op == AggOp::Mean {
                    output_buffers.push(allocate_count_buffer(device, n_groups));
                }
            }
            AggSpec::Length { .. } => {
                output_buffers.push(allocate_count_buffer(device, n_groups));
            }
        }
    }
    initialize_outputs_for_ops(&mut output_buffers, aggs);

    // Bind buffers per emitter's slot layout.
    let mut binds: Vec<&MetalBuffer<u8>> = vec![row_to_group.as_u8(), /* n_rows scalar */];
    for (data, _) in value_columns { binds.push(data); }
    for (_, validity) in value_columns { binds.push(validity); }
    for ob in &output_buffers { binds.push(ob); }

    let n_rows = row_to_group.len() as u32;
    queue.dispatch_1d(&pso, &binds, n_rows)?;

    // CPU finalize for mean / 64-bit cases.
    finalize_outputs(&output_buffers, aggs)
}
```

- [ ] **Step 2: Switch the UDF entry to fused**

In `crates/polars-metal-core/src/udf.rs` (the `execute_groupby` entry), change the call from M2's per-agg loop to `execute_groupby_aggregation_fused`. Keep the per-agg path behind a feature flag temporarily so we can compare in benchmarks.

```rust
#[cfg(feature = "fused-agg")]
fn aggregation_pass(...) -> Result<..., EngineError> {
    execute_groupby_aggregation_fused(...)
}

#[cfg(not(feature = "fused-agg"))]
fn aggregation_pass(...) -> Result<..., EngineError> {
    execute_groupby_aggregation_m2(...)  // M2's loop, renamed
}
```

Default feature in `Cargo.toml`: `fused-agg = []` (on by default for M3).

- [ ] **Step 3: Run M2's existing integration tests under the new path**

```bash
make wheel
pytest tests/python_integration/ -k groupby -v
```

Expected: all M2 groupby tests still pass (correctness preserved). If any fail, the emitter has a bug — debug by dumping the emitted MSL.

- [ ] **Step 4: Commit**

```bash
git add crates/polars-metal-kernels/src/groupby.rs crates/polars-metal-core/src/udf.rs crates/polars-metal-kernels/Cargo.toml
git commit -m "Kernel: groupby pipeline uses fused-agg kernel via FusedLibraryCache

Capability B (full wiring). M2's per-agg loop becomes the no-feature
fallback path; default build uses one fused kernel per signature.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 16: Proptest — fused output byte-equal to M2 per-agg kernels

**Files:**
- Create: `crates/polars-metal-kernels/tests/test_fused_vs_per_agg.rs`

- [ ] **Step 1: Write the test**

```rust
// crates/polars-metal-kernels/tests/test_fused_vs_per_agg.rs
//! Verify the fused kernel produces byte-equal results to M2's per-agg
//! kernels across (agg signature × null density × group cardinality).

use polars_metal_core::plan::{AggOp, AggSpec, MetalDtype};
use polars_metal_kernels::groupby::{
    execute_groupby_aggregation_fused,
    execute_groupby_aggregation_m2,
    // ...
};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn fused_eq_per_agg_sum_only(
        values in proptest::collection::vec(any::<f32>(), 1..10_000),
        groups in 2u32..256,
        null_density in 0.0f32..1.0,
    ) {
        let aggs = vec![AggSpec::Simple {
            input_column: "v".into(),
            op: AggOp::Sum,
            output_alias: "s".into(),
            output_dtype: MetalDtype::F32,
        }];
        let result_fused = run_fused(&aggs, &values, groups, null_density);
        let result_per_agg = run_per_agg(&aggs, &values, groups, null_density);
        prop_assert_eq!(result_fused, result_per_agg);
    }

    #[test]
    fn fused_eq_per_agg_full_q1_shape(
        seed in any::<u64>(),
    ) {
        // Q1's 8 aggs over 4 value columns.
        let aggs = q1_aggs();
        let inputs = q1_inputs(seed, /*n_rows=*/ 100_000);
        let result_fused = run_fused(&aggs, &inputs, /*groups=*/ 4, /*null_density=*/ 0.05);
        let result_per_agg = run_per_agg(&aggs, &inputs, 4, 0.05);
        prop_assert_eq!(result_fused, result_per_agg);
    }
}
```

The helpers `run_fused`, `run_per_agg`, `q1_aggs`, `q1_inputs` are local utility functions that synthesize inputs and call the respective entry points. Place them in the same test file or in `crates/polars-metal-kernels/tests/common/mod.rs`.

- [ ] **Step 2: Run**

```bash
cargo test -p polars-metal-kernels --test test_fused_vs_per_agg -- --test-threads=1
```

Expected: 2 properties at 256 cases each, all pass. If byte-mismatches appear, they're correctness bugs in the emitter — debug per-case.

- [ ] **Step 3: Commit**

```bash
git add crates/polars-metal-kernels/tests/test_fused_vs_per_agg.rs
git commit -m "Test: fused agg byte-equal to per-agg across Q1-shape proptest

Capability B correctness proof. Two properties at 256 cases each
covering sum-only and full Q1-shape (8 aggs / 4 value columns).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 17: Criterion bench — fused vs per-agg dispatch count + wall-clock

**Files:**
- Create: `benches/aggregate_fused.rs`

- [ ] **Step 1: Write the bench**

```rust
// benches/aggregate_fused.rs
//! Compares fused-kernel dispatch (one kernel, multi-agg) against M2's
//! per-agg loop (N kernels). The metric of interest is wall-clock time
//! at fixed input sizes — dispatch count is observable but secondary.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use polars_metal_kernels::groupby::{
    execute_groupby_aggregation_fused,
    execute_groupby_aggregation_m2,
};

fn bench_fused_vs_m2(c: &mut Criterion) {
    let mut group = c.benchmark_group("aggregate_fused_vs_per_agg");
    for size in [100_000usize, 1_000_000, 10_000_000] {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("fused_q1_8aggs", size), &size, |b, &n| {
            let inputs = setup_q1_inputs(n);
            b.iter(|| execute_groupby_aggregation_fused(&inputs));
        });
        group.bench_with_input(BenchmarkId::new("per_agg_q1_8aggs", size), &size, |b, &n| {
            let inputs = setup_q1_inputs(n);
            b.iter(|| execute_groupby_aggregation_m2(&inputs));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_fused_vs_m2);
criterion_main!(benches);
```

Add the bench to `crates/polars-metal-kernels/Cargo.toml`:

```toml
[[bench]]
name = "aggregate_fused"
harness = false
```

- [ ] **Step 2: Run + record**

```bash
cargo bench -p polars-metal-kernels --bench aggregate_fused
```

Expected: fused beats per-agg meaningfully at 1M+ rows (target ≥ 3× at 10M per spec). At 100K the gap is smaller (kernel-launch overhead amortizes less).

- [ ] **Step 3: Commit**

```bash
git add benches/aggregate_fused.rs crates/polars-metal-kernels/Cargo.toml
git commit -m "Bench: fused vs per-agg Q1-shape across 100K/1M/10M rows

Capability B perf evidence. Tracked alongside criterion baselines.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 18: Pre-compile common signatures at module import

**Files:**
- Modify: `crates/polars-metal-kernels/src/aggregate_fused/cache.rs`
- Modify: `python/polars_metal/__init__.py`

- [ ] **Step 1: Define the warmup signature list**

```rust
// crates/polars-metal-kernels/src/aggregate_fused/cache.rs (additions)
pub fn common_signatures() -> Vec<(AggSignature, Vec<AggSpec>)> {
    use polars_metal_core::plan::{AggOp, AggSpec, MetalDtype};
    let f32_sum = vec![AggSpec::Simple {
        input_column: "x".into(), op: AggOp::Sum,
        output_alias: "s".into(), output_dtype: MetalDtype::F32,
    }];
    // ... build Q1's 8-agg signature, single-count, single-mean, etc.
    vec![
        (AggSignature::from_specs(&f32_sum), f32_sum),
        // (q1_signature, q1_specs),
    ]
}
```

- [ ] **Step 2: Wire warmup into Python import**

```python
# python/polars_metal/__init__.py (excerpt)
def _warmup_kernels():
    """Pre-compile common fused-agg signatures at import time.

    Cost: ~100-500 ms one-time per process. Benefit: first user query
    of common shapes (sum-only, Q1, count, mean) doesn't pay MSL compile.
    """
    from polars_metal import _native
    _native.warmup_common_fused_signatures()

# Triggered after the engine is registered.
_warmup_kernels()
```

Expose `warmup_common_fused_signatures` from Rust via `polars-metal-core/src/native.rs` (PyO3 module).

- [ ] **Step 3: Verify import-time warmup runs**

```bash
make wheel
python -c "
import time
t0 = time.perf_counter()
import polars_metal
import_dt = time.perf_counter() - t0
print(f'import + warmup: {import_dt*1000:.1f} ms')

t0 = time.perf_counter()
import polars as pl
df = pl.DataFrame({'k': [0,0,1,1], 'v': [1.0,2.0,3.0,4.0]})
df.lazy().group_by('k').agg(pl.col('v').sum()).collect(engine=polars_metal.MetalEngine())
first_query = time.perf_counter() - t0
print(f'first query (cached): {first_query*1000:.1f} ms')
"
```

Expected: import is 200-800 ms (warmup happens here); first query is fast (no MSL compile). Without warmup, the first query would pay ~100-300 ms compile.

- [ ] **Step 4: Commit**

```bash
git add crates/polars-metal-kernels/src/aggregate_fused/cache.rs python/polars_metal/__init__.py crates/polars-metal-core/src/native.rs
git commit -m "Bootstrap: pre-compile common fused-agg signatures at import

Capability B perf polish. First query of common shapes (sum-only, Q1,
count, mean) skips MSL compile.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 4 — Partitioned hash build phase (capability A1)

M2's build phase runs CPU HashMap (4 unique groups for Q1 → trivially fast). At higher cardinality, the CPU HashMap becomes the bottleneck. M3 ships a partitioned-hash GPU build: split keys into P partitions by hash bits, each threadgroup builds its partition's hash table in TGSM (no global atomics), then a CPU exclusive-scan assigns global group IDs. Works to ~65K unique groups (when partitions stay within TGSM); above that, A2 takes over (Phase 5).

### Task 19: CPU-side reference implementation + helpers

**Files:**
- Create: `crates/polars-metal-kernels/src/groupby_build_partitioned/mod.rs`
- Create: `crates/polars-metal-kernels/src/groupby_build_partitioned/reference.rs`
- Create: `crates/polars-metal-kernels/tests/test_groupby_build_partitioned_reference.rs`

- [ ] **Step 1: Write the failing test for CPU reference**

```rust
// crates/polars-metal-kernels/tests/test_groupby_build_partitioned_reference.rs
//! The CPU reference is the ground truth for proptest comparisons.
//! It implements the *same algorithm* the GPU runs — not a high-level
//! HashMap — so that any algorithmic bug surfaces equally on both sides.

use polars_metal_kernels::groupby_build_partitioned::reference::cpu_partitioned_hash;

#[test]
fn empty_input_yields_zero_groups() {
    let out = cpu_partitioned_hash(&[], /*n_partitions=*/ 4);
    assert_eq!(out.n_groups, 0);
    assert!(out.row_to_group.is_empty());
}

#[test]
fn all_same_key_yields_one_group() {
    let keys: Vec<u128> = vec![0xdeadbeef_cafebabe; 100];
    let out = cpu_partitioned_hash(&keys, 4);
    assert_eq!(out.n_groups, 1);
    assert!(out.row_to_group.iter().all(|&g| g == 0));
}

#[test]
fn all_distinct_keys_yields_n_groups() {
    let keys: Vec<u128> = (0u128..256).collect();
    let out = cpu_partitioned_hash(&keys, 4);
    assert_eq!(out.n_groups, 256);
    // Each key gets a unique group_id.
    let mut seen = std::collections::HashSet::new();
    for &g in &out.row_to_group { assert!(seen.insert(g)); }
}

#[test]
fn round_trip_first_row_per_group_indexes_original_rows() {
    let keys: Vec<u128> = vec![10, 20, 10, 30, 20, 10];
    let out = cpu_partitioned_hash(&keys, 4);
    for g in 0..out.n_groups {
        let fr = out.first_row_per_group[g as usize] as usize;
        // The first_row's key should match every row in that group.
        for (r, &group_of_r) in out.row_to_group.iter().enumerate() {
            if group_of_r == g {
                assert_eq!(keys[r], keys[fr]);
            }
        }
    }
}
```

- [ ] **Step 2: Implement the CPU reference**

```rust
// crates/polars-metal-kernels/src/groupby_build_partitioned/mod.rs
//! Partitioned-hash build phase (capability A1).
//!
//! Algorithm — per spec § "Algorithm details / A1":
//!   1. Per-row partition_id = (hash(key) >> log2(TGSM_slots)) & (P-1)
//!   2. Scatter rows into partition lanes.
//!   3. Per partition (one threadgroup), build hash table in TGSM with
//!      open addressing + linear probe. Emit (row, local_group_id).
//!   4. CPU: exclusive scan over n_groups_per_partition; offset local
//!      group_ids to produce global row_to_group.
//!   5. CPU: derive first_row_per_group for result reconstruction.
//!
//! See `references/cudf/cpp/src/groupby/hash/groupby.cu` for the source
//! algorithm. Our adaptation: 32-bit atomics only (Apple Silicon
//! constraint); per-threadgroup hash tables (no global atomic-CAS).

pub mod reference;
pub mod gpu;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildOutput {
    /// Per-row group_id (global).
    pub row_to_group: Vec<u32>,
    /// For each group, the index of its first occurrence in the input.
    pub first_row_per_group: Vec<u32>,
    /// Total number of unique groups.
    pub n_groups: u32,
}
```

```rust
// crates/polars-metal-kernels/src/groupby_build_partitioned/reference.rs
use super::BuildOutput;

/// xxhash-style mixing function. Same constants as the MSL implementation.
fn hash_u128(key: u128) -> u64 {
    let mut h = 0x9E3779B97F4A7C15u64;
    h ^= (key as u64).wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= ((key >> 64) as u64).wrapping_mul(0x94D049BB133111EB);
    h ^= h >> 31;
    h.wrapping_mul(0x9E3779B97F4A7C15)
}

/// Tuneable. The MSL kernel uses 1024 slots per threadgroup;
/// the reference matches that.
const TGSM_SLOTS_PER_PARTITION: u32 = 1024;

pub fn cpu_partitioned_hash(keys: &[u128], n_partitions: u32) -> BuildOutput {
    if keys.is_empty() {
        return BuildOutput { row_to_group: vec![], first_row_per_group: vec![], n_groups: 0 };
    }
    assert!(n_partitions.is_power_of_two() && n_partitions > 0);

    // Phase 1: partition scatter (rows by partition_id).
    let mut rows_by_partition: Vec<Vec<u32>> = vec![Vec::new(); n_partitions as usize];
    for (r, &k) in keys.iter().enumerate() {
        let h = hash_u128(k);
        let part = ((h >> (TGSM_SLOTS_PER_PARTITION.trailing_zeros())) & (n_partitions as u64 - 1)) as usize;
        rows_by_partition[part].push(r as u32);
    }

    // Phase 2: per-partition build.
    let mut per_partition_groups: Vec<Vec<(u128, u32)>> = vec![Vec::new(); n_partitions as usize];
    let mut row_local_group: Vec<u32> = vec![0; keys.len()];
    for (p, rows) in rows_by_partition.iter().enumerate() {
        let table = &mut per_partition_groups[p];
        // Open-addressing hash table; capacity = 2 × expected unique keys.
        // Reference uses Vec<(u128, u32)> with sentinel.
        let cap = (rows.len() * 2).next_power_of_two().max(8);
        let mut slots: Vec<Option<(u128, u32)>> = vec![None; cap];
        let mut local_next = 0u32;
        for &r in rows {
            let k = keys[r as usize];
            let h = hash_u128(k) as usize;
            let mut idx = h & (cap - 1);
            loop {
                match slots[idx] {
                    None => {
                        slots[idx] = Some((k, local_next));
                        table.push((k, local_next));
                        row_local_group[r as usize] = local_next;
                        local_next += 1;
                        break;
                    }
                    Some((existing_k, gid)) if existing_k == k => {
                        row_local_group[r as usize] = gid;
                        break;
                    }
                    Some(_) => {
                        idx = (idx + 1) & (cap - 1);
                    }
                }
            }
        }
    }

    // Phase 3: global group_id offsetting.
    let mut partition_offset = vec![0u32; n_partitions as usize + 1];
    for (p, table) in per_partition_groups.iter().enumerate() {
        partition_offset[p + 1] = partition_offset[p] + table.len() as u32;
    }
    let n_groups = *partition_offset.last().unwrap();
    let mut row_to_group = vec![0u32; keys.len()];
    let mut first_row_per_group = vec![u32::MAX; n_groups as usize];
    for (r, &k) in keys.iter().enumerate() {
        let h = hash_u128(k);
        let part = ((h >> TGSM_SLOTS_PER_PARTITION.trailing_zeros()) & (n_partitions as u64 - 1)) as usize;
        let local = row_local_group[r];
        let global = partition_offset[part] + local;
        row_to_group[r] = global;
        if first_row_per_group[global as usize] == u32::MAX {
            first_row_per_group[global as usize] = r as u32;
        }
    }

    BuildOutput { row_to_group, first_row_per_group, n_groups }
}
```

- [ ] **Step 3: Run**

```bash
cargo test -p polars-metal-kernels --test test_groupby_build_partitioned_reference -- --test-threads=1
```

Expected: 4 passes.

- [ ] **Step 4: Commit**

```bash
git add crates/polars-metal-kernels/src/groupby_build_partitioned/ crates/polars-metal-kernels/tests/test_groupby_build_partitioned_reference.rs
git commit -m "Kernel: CPU reference for partitioned-hash build phase

Capability A1. The reference is the proptest oracle for the GPU
implementation. Uses 32-bit-atomic semantics throughout to match
hardware constraints.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 20: MSL kernel — partition scatter

**Files:**
- Create: `shaders/groupby_build_partitioned_scatter.metal`
- Create: `crates/polars-metal-kernels/tests/test_groupby_build_partitioned_scatter.rs`

- [ ] **Step 1: Write the MSL kernel**

```metal
// shaders/groupby_build_partitioned_scatter.metal
//
// Capability A1, phase 1: per-row, compute partition_id from a hash of
// the encoded composite key, then scatter row indices into per-partition
// lanes. The scatter uses a two-pass approach:
//   Pass A: count per-partition row counts (atomic).
//   (CPU between A and B: exclusive scan → partition_offsets[].)
//   Pass B: scatter row_idx into per-partition slot using atomic-add
//           on a write cursor seeded by partition_offsets.
//
// Both passes are 32-bit atomics — well within Apple Silicon's set.

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant uint TGSM_SLOTS_PER_PARTITION = 1024;

// xxhash-style mixer matching the CPU reference.
uint64_t hash_u128(uint64_t key_lo, uint64_t key_hi) {
    uint64_t h = 0x9E3779B97F4A7C15ull;
    h ^= key_lo * 0xBF58476D1CE4E5B9ull;
    h ^= key_hi * 0x94D049BB133111EBull;
    h ^= h >> 31;
    return h * 0x9E3779B97F4A7C15ull;
}

uint partition_id(uint64_t key_lo, uint64_t key_hi, uint n_partitions, uint log2_tgsm_slots) {
    uint64_t h = hash_u128(key_lo, key_hi);
    return (uint)((h >> log2_tgsm_slots) & (uint64_t)(n_partitions - 1u));
}

kernel void partition_count(
    device const uint64_t* keys_lo  [[buffer(0)]],
    device const uint64_t* keys_hi  [[buffer(1)]],
    device atomic_uint*    counts   [[buffer(2)]],  // [n_partitions]
    constant uint&         n_rows         [[buffer(3)]],
    constant uint&         n_partitions   [[buffer(4)]],
    constant uint&         log2_tgsm      [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    uint p = partition_id(keys_lo[gid], keys_hi[gid], n_partitions, log2_tgsm);
    atomic_fetch_add_explicit(&counts[p], 1u, memory_order_relaxed);
}

kernel void partition_scatter(
    device const uint64_t* keys_lo            [[buffer(0)]],
    device const uint64_t* keys_hi            [[buffer(1)]],
    device const uint*     partition_offsets  [[buffer(2)]],  // [n_partitions+1]
    device atomic_uint*    write_cursors      [[buffer(3)]],  // [n_partitions], init=0
    device uint*           row_indices_out    [[buffer(4)]],  // [n_rows]
    constant uint&         n_rows             [[buffer(5)]],
    constant uint&         n_partitions       [[buffer(6)]],
    constant uint&         log2_tgsm          [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    uint p = partition_id(keys_lo[gid], keys_hi[gid], n_partitions, log2_tgsm);
    uint slot = atomic_fetch_add_explicit(&write_cursors[p], 1u, memory_order_relaxed);
    row_indices_out[partition_offsets[p] + slot] = gid;
}
```

- [ ] **Step 2: Write the Rust dispatch + test**

```rust
// crates/polars-metal-kernels/tests/test_groupby_build_partitioned_scatter.rs
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::groupby_build_partitioned::gpu::partition_and_scatter;

#[test]
fn scatter_produces_partition_layout_matching_cpu_reference() {
    let device = MetalDevice::system_default().unwrap();
    // Build a small case: 8 keys, 4 partitions.
    let keys: Vec<u128> = vec![10, 20, 30, 10, 20, 30, 10, 50];
    let n_partitions = 4;
    let (row_indices, partition_offsets) = partition_and_scatter(&device, &keys, n_partitions).unwrap();

    // Each row index appears exactly once in `row_indices`.
    let mut seen = vec![false; keys.len()];
    for &r in &row_indices { seen[r as usize] = true; }
    assert!(seen.iter().all(|&b| b));

    // partition_offsets is non-decreasing and ends at keys.len().
    for w in partition_offsets.windows(2) { assert!(w[0] <= w[1]); }
    assert_eq!(*partition_offsets.last().unwrap(), keys.len() as u32);
}

#[test]
fn scatter_proptest_matches_cpu_reference() {
    use proptest::test_runner::TestRunner;
    use proptest::strategy::Strategy;
    let device = MetalDevice::system_default().unwrap();
    let mut runner = TestRunner::default();
    let strat = proptest::collection::vec(any::<u128>(), 1..1024usize)
        .prop_map(|v| v);
    runner.run(&strat, |keys| {
        for n_part in [2u32, 4, 8, 16] {
            let (gpu_idx, gpu_off) = partition_and_scatter(&device, &keys, n_part).unwrap();
            let cpu = polars_metal_kernels::groupby_build_partitioned::reference::cpu_partition_layout(&keys, n_part);
            assert_eq!(gpu_idx, cpu.row_indices);
            assert_eq!(gpu_off, cpu.partition_offsets);
        }
        Ok(())
    }).unwrap();
}
```

(Add `cpu_partition_layout` to the reference module as a public helper that produces the same scatter layout the GPU produces — same hash, same partition_id, just on CPU. The two should be byte-equal.)

- [ ] **Step 3: Add the Rust dispatch**

```rust
// crates/polars-metal-kernels/src/groupby_build_partitioned/gpu.rs
use polars_metal_buffer::{MetalDevice, MetalBuffer};

pub fn partition_and_scatter(
    device: &MetalDevice,
    keys: &[u128],
    n_partitions: u32,
) -> Result<(Vec<u32>, Vec<u32>), String> {
    // Split keys into lo/hi u64 buffers (Metal doesn't have a u128 primitive).
    let keys_lo: Vec<u64> = keys.iter().map(|k| *k as u64).collect();
    let keys_hi: Vec<u64> = keys.iter().map(|k| (*k >> 64) as u64).collect();
    let n_rows = keys.len() as u32;
    let log2_tgsm = 10u32;  // 1024 slots

    let lib = device.load_shader_library("groupby_build_partitioned_scatter")?;
    let pso_count = device.pipeline_for_function(&lib, "partition_count")?;
    let pso_scatter = device.pipeline_for_function(&lib, "partition_scatter")?;

    let buf_keys_lo = device.new_buffer_from_slice(&keys_lo);
    let buf_keys_hi = device.new_buffer_from_slice(&keys_hi);
    let buf_counts = device.new_buffer_zeroed::<u32>(n_partitions as usize);
    let buf_n_rows = device.new_buffer_from_slice(&[n_rows]);
    let buf_n_part = device.new_buffer_from_slice(&[n_partitions]);
    let buf_log2 = device.new_buffer_from_slice(&[log2_tgsm]);

    let queue = device.new_command_queue();
    queue.dispatch_1d(&pso_count, &[
        &buf_keys_lo, &buf_keys_hi, &buf_counts,
        &buf_n_rows, &buf_n_part, &buf_log2,
    ], n_rows)?;
    let counts: Vec<u32> = buf_counts.to_vec();
    let mut partition_offsets = vec![0u32; n_partitions as usize + 1];
    for i in 0..n_partitions as usize {
        partition_offsets[i + 1] = partition_offsets[i] + counts[i];
    }

    let buf_offsets = device.new_buffer_from_slice(&partition_offsets);
    let buf_cursors = device.new_buffer_zeroed::<u32>(n_partitions as usize);
    let buf_row_idx = device.new_buffer_zeroed::<u32>(n_rows as usize);

    queue.dispatch_1d(&pso_scatter, &[
        &buf_keys_lo, &buf_keys_hi, &buf_offsets, &buf_cursors, &buf_row_idx,
        &buf_n_rows, &buf_n_part, &buf_log2,
    ], n_rows)?;
    let row_indices: Vec<u32> = buf_row_idx.to_vec();
    Ok((row_indices, partition_offsets))
}
```

- [ ] **Step 4: Run**

```bash
cargo test -p polars-metal-kernels --test test_groupby_build_partitioned_scatter -- --test-threads=1
```

Expected: 2 passes (including proptest 256 cases).

- [ ] **Step 5: Commit**

```bash
git add shaders/groupby_build_partitioned_scatter.metal crates/polars-metal-kernels/src/groupby_build_partitioned/gpu.rs crates/polars-metal-kernels/tests/test_groupby_build_partitioned_scatter.rs
git commit -m "Kernel: A1 partition_count + partition_scatter MSL kernels

Capability A1, phase 1. Two-pass scatter: count per-partition rows
then write into reserved lanes. 32-bit atomics throughout.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 21: MSL kernel — per-partition hash build (TGSM hash table)

**Files:**
- Create: `shaders/groupby_build_partitioned_build.metal`
- Modify: `crates/polars-metal-kernels/src/groupby_build_partitioned/gpu.rs`
- Create: `crates/polars-metal-kernels/tests/test_groupby_build_partitioned_build.rs`

- [ ] **Step 1: Write the MSL kernel**

```metal
// shaders/groupby_build_partitioned_build.metal
//
// Capability A1, phase 2: per-partition hash-table build in TGSM.
// One threadgroup per partition. Each threadgroup:
//   - Loads its row_indices_in_partition slice.
//   - Builds a local hash table in TGSM (open addressing, linear probe).
//   - Assigns per-partition local group_ids 0..k_p.
//   - Writes row_to_local_group + n_groups_in_partition out to global.
//
// TGSM_SLOTS_PER_PARTITION = 1024 slots × (16 bytes key + 4 bytes
// group_id + 4 bytes occupied flag) = 24 KB; fits in the 32 KB
// threadgroup memory. Load factor capped at 75% (768 unique keys).
//
// Overflow detection: if any slot insertion's probe distance > 64
// (heuristic), set the overflow flag for this partition.

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant uint TGSM_SLOTS = 1024;
constant uint TGSM_PROBE_LIMIT = 64;

uint64_t hash_u128_again(uint64_t lo, uint64_t hi) {
    // Same function as scatter; duplicated to keep the kernel
    // self-contained. (Inlined by the compiler.)
    uint64_t h = 0x9E3779B97F4A7C15ull;
    h ^= lo * 0xBF58476D1CE4E5B9ull;
    h ^= hi * 0x94D049BB133111EBull;
    h ^= h >> 31;
    return h * 0x9E3779B97F4A7C15ull;
}

kernel void partition_build(
    device const uint64_t* keys_lo            [[buffer(0)]],
    device const uint64_t* keys_hi            [[buffer(1)]],
    device const uint*     row_indices        [[buffer(2)]],  // by partition
    device const uint*     partition_offsets  [[buffer(3)]],  // [n_part+1]
    device uint*           row_to_local_group [[buffer(4)]],  // [n_rows], by original row
    device atomic_uint*    n_groups_per_part  [[buffer(5)]],  // [n_partitions]
    device atomic_uint*    overflow_flag      [[buffer(6)]],  // [1]
    constant uint&         n_rows             [[buffer(7)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup uint64_t  slot_key_lo[TGSM_SLOTS];
    threadgroup uint64_t  slot_key_hi[TGSM_SLOTS];
    threadgroup atomic_uint slot_state[TGSM_SLOTS];  // 0 = empty, !=0 = group_id+1
    threadgroup atomic_uint next_local_id;

    // Initialize TGSM.
    for (uint i = tid; i < TGSM_SLOTS; i += tg_size) {
        slot_key_lo[i] = 0;
        slot_key_hi[i] = 0;
        atomic_store_explicit(&slot_state[i], 0u, memory_order_relaxed);
    }
    if (tid == 0) atomic_store_explicit(&next_local_id, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint start = partition_offsets[tg_id];
    uint end   = partition_offsets[tg_id + 1];
    uint count = end - start;

    for (uint i = tid; i < count; i += tg_size) {
        uint r = row_indices[start + i];
        uint64_t klo = keys_lo[r];
        uint64_t khi = keys_hi[r];
        uint64_t h = hash_u128_again(klo, khi);
        uint slot = (uint)(h & (TGSM_SLOTS - 1u));
        uint probe = 0;
        uint group_id = UINT_MAX;
        while (probe < TGSM_PROBE_LIMIT) {
            uint state = atomic_load_explicit(&slot_state[slot], memory_order_relaxed);
            if (state == 0u) {
                // Try to claim.
                uint expected = 0u;
                // Atomic CAS: claim by writing UINT_MAX as "claiming" sentinel,
                // then fill key, then publish group_id.
                if (atomic_compare_exchange_weak_explicit(
                        &slot_state[slot], &expected, UINT_MAX,
                        memory_order_relaxed, memory_order_relaxed)) {
                    slot_key_lo[slot] = klo;
                    slot_key_hi[slot] = khi;
                    uint gid = atomic_fetch_add_explicit(&next_local_id, 1u, memory_order_relaxed);
                    atomic_store_explicit(&slot_state[slot], gid + 1u, memory_order_relaxed);
                    group_id = gid;
                    break;
                } else {
                    // Lost the race; re-read this slot.
                    state = atomic_load_explicit(&slot_state[slot], memory_order_relaxed);
                }
            }
            // Spin while another thread is in the claiming phase
            // (state == UINT_MAX). This is safe because the claimer is
            // in a *different* SIMD-group (we hash to spread keys), and
            // the claim phase completes in a few cycles.
            //
            // ⚠ NOTE: per M2 retrospective, SIMD-lockstep means same-warp
            // spin-wait can deadlock. The mitigation: hash spreads keys
            // sparsely; same-key contention falls into the != path below.
            while (state == UINT_MAX) {
                state = atomic_load_explicit(&slot_state[slot], memory_order_relaxed);
            }
            // State is now a real group_id+1.
            if (slot_key_lo[slot] == klo && slot_key_hi[slot] == khi) {
                group_id = state - 1u;
                break;
            }
            slot = (slot + 1u) & (TGSM_SLOTS - 1u);
            probe += 1;
        }
        if (group_id == UINT_MAX) {
            atomic_store_explicit(overflow_flag, 1u, memory_order_relaxed);
            // Fallback: assign sentinel; CPU will re-dispatch as A2.
            row_to_local_group[r] = UINT_MAX;
        } else {
            row_to_local_group[r] = group_id;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        atomic_store_explicit(
            &n_groups_per_part[tg_id],
            atomic_load_explicit(&next_local_id, memory_order_relaxed),
            memory_order_relaxed
        );
    }
}
```

**Important deviation from spec § A1.2.** The TGSM hash-table writes use atomic CAS *combined with a UINT_MAX "claiming" sentinel + spin-wait on same-slot races*. Per M2's retrospective (T19, commit `2d60b23`), SIMD-lockstep can deadlock spin-wait designs when threads in the same warp contend. The mitigation here: the partition scatter (Task 20) used the hash's *high* bits to assign rows to partitions; the partition_build uses the *low* bits to assign slots within the partition. The two-stage hashing makes same-warp collisions on the same slot rare.

**If proptest surfaces deadlocks at scale**, the fallback is to drop the spin-wait: claim a slot atomically, but if another thread claimed it first, advance to the next slot (no waiting). This is the design M2 ultimately abandoned for the *global* build because contention was too high; here at the per-threadgroup level with hash-spread keys, the design is salvageable. The proptest at Task 23 must include adversarial cases (constructed hash collisions) to verify.

- [ ] **Step 2: Write proptest comparing GPU to CPU reference**

```rust
// crates/polars-metal-kernels/tests/test_groupby_build_partitioned_build.rs
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::groupby_build_partitioned::gpu::partition_and_build;
use polars_metal_kernels::groupby_build_partitioned::reference::cpu_partitioned_hash;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn build_matches_cpu_reference(
        keys in proptest::collection::vec(any::<u128>(), 1..2048usize),
        n_partitions in proptest::sample::select(vec![4u32, 8, 16, 32]),
    ) {
        let device = MetalDevice::system_default().unwrap();
        let gpu_out = partition_and_build(&device, &keys, n_partitions)?;
        let cpu_out = cpu_partitioned_hash(&keys, n_partitions);

        // n_groups must match.
        prop_assert_eq!(gpu_out.n_groups, cpu_out.n_groups);
        // Group assignment must be a permutation of each other (group IDs
        // may differ in numbering but the *grouping* must be identical).
        // Verify by: for any pair of rows (a, b), gpu[a] == gpu[b] iff cpu[a] == cpu[b].
        for a in 0..keys.len() {
            for b in 0..keys.len() {
                let gpu_same = gpu_out.row_to_group[a] == gpu_out.row_to_group[b];
                let cpu_same = cpu_out.row_to_group[a] == cpu_out.row_to_group[b];
                prop_assert_eq!(gpu_same, cpu_same);
            }
        }
    }

    #[test]
    fn build_handles_extreme_collision_case(
        // Force keys whose low bits collide deterministically.
        seed in any::<u64>(),
    ) {
        let device = MetalDevice::system_default().unwrap();
        // Construct 100 keys all hashing to the same partition.
        let keys: Vec<u128> = (0..100u128)
            .map(|i| (i << 10) | (seed as u128))
            .collect();
        let gpu_out = partition_and_build(&device, &keys, 4)?;
        let cpu_out = cpu_partitioned_hash(&keys, 4);
        prop_assert_eq!(gpu_out.n_groups, cpu_out.n_groups);
    }
}
```

- [ ] **Step 3: Add `partition_and_build` Rust dispatch**

In `crates/polars-metal-kernels/src/groupby_build_partitioned/gpu.rs`, add the higher-level entry that chains scatter + build + offset assignment. Returns `BuildOutput`.

- [ ] **Step 4: Run**

```bash
cargo test -p polars-metal-kernels --test test_groupby_build_partitioned_build -- --test-threads=1
```

Expected: 2 properties pass at 64 cases each. (Reduced from 256 because each case allocates Metal buffers; this stays under ~30s wall-clock.)

- [ ] **Step 5: Commit**

```bash
git add shaders/groupby_build_partitioned_build.metal crates/polars-metal-kernels/src/groupby_build_partitioned/gpu.rs crates/polars-metal-kernels/tests/test_groupby_build_partitioned_build.rs
git commit -m "Kernel: A1 per-partition hash-table build in TGSM

Capability A1, phase 2. One threadgroup per partition; 1024 slots in
TGSM with open addressing + linear probe. CAS-claim with sentinel to
allow same-key matching across SIMD lanes. Overflow flag flips on
probe-limit hit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 22: TGSM overflow → A2 fallback

**Files:**
- Modify: `crates/polars-metal-kernels/src/groupby_build_partitioned/gpu.rs`

- [ ] **Step 1: Add overflow detection + fallback signal**

```rust
// In partition_and_build (after dispatch):
let overflow = buf_overflow_flag.to_vec();
if overflow[0] != 0 {
    return Err("A1 TGSM overflow; fallback to A2".into());
}
```

The caller (groupby pipeline orchestrator, Phase 6 Task 32) handles this error by re-dispatching as A2.

- [ ] **Step 2: Test the overflow path**

```rust
#[test]
fn build_signals_overflow_at_extreme_cardinality() {
    let device = MetalDevice::system_default().unwrap();
    // 10K unique keys into 4 partitions = 2.5K per partition, exceeds
    // the 1024 TGSM slot cap → overflow expected.
    let keys: Vec<u128> = (0u128..10_000).collect();
    let result = partition_and_build(&device, &keys, 4);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("overflow"));
}
```

- [ ] **Step 3: Commit**

```bash
git add crates/polars-metal-kernels/src/groupby_build_partitioned/gpu.rs crates/polars-metal-kernels/tests/test_groupby_build_partitioned_build.rs
git commit -m "Kernel: A1 overflow flag triggers Err; orchestrator routes to A2

Capability A1. Detection happens after dispatch via shared flag buffer.
Orchestrator handles re-dispatch (Phase 6).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 23: Criterion bench — A1 vs M2 CPU HashMap

**Files:**
- Create: `benches/groupby_build_partitioned.rs`

- [ ] **Step 1: Write the bench**

```rust
// benches/groupby_build_partitioned.rs
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::groupby_build_partitioned::gpu::partition_and_build;

fn bench_a1(c: &mut Criterion) {
    let device = MetalDevice::system_default().unwrap();
    let mut group = c.benchmark_group("groupby_build_partitioned");
    for &n_rows in &[100_000usize, 1_000_000, 10_000_000] {
        for &n_groups in &[4u32, 1024, 16_384] {
            let keys: Vec<u128> = (0..n_rows).map(|i| (i % n_groups as usize) as u128).collect();
            group.bench_with_input(
                BenchmarkId::new(format!("rows{n_rows}_groups{n_groups}"), n_rows),
                &keys,
                |b, keys| b.iter(|| partition_and_build(&device, keys, 16)),
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_a1);
criterion_main!(benches);
```

Add to `crates/polars-metal-kernels/Cargo.toml`:
```toml
[[bench]]
name = "groupby_build_partitioned"
harness = false
```

- [ ] **Step 2: Run**

```bash
cargo bench -p polars-metal-kernels --bench groupby_build_partitioned
```

Expected: A1 at 10M rows × 1K groups completes in ~10-30 ms (per spec). At 16K groups, runs close to TGSM overflow; should still complete (the GPU build is faster than CPU here).

- [ ] **Step 3: Commit**

```bash
git add benches/groupby_build_partitioned.rs crates/polars-metal-kernels/Cargo.toml
git commit -m "Bench: A1 partitioned-hash across 100K/1M/10M × 4/1K/16K groups

Capability A1 perf evidence.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 5 — Sort-then-segment-reduce build phase (capability A2)

For high-cardinality cases (>~65K unique groups) A1's TGSM tables overflow. A2 sorts the encoded composite keys via GPU radix sort, then a single scan finds segment boundaries between distinct keys. This is the algorithm cuDF uses for high-cardinality groupby. The radix sort kernel here is **a restricted-scope internal helper for groupby only** — not a Polars Sort IR-node handler. M4's full sort milestone may generalize or replace it.

### Task 24: CPU reference for sort+segment

**Files:**
- Create: `crates/polars-metal-kernels/src/groupby_build_sort/mod.rs`
- Create: `crates/polars-metal-kernels/src/groupby_build_sort/reference.rs`
- Create: `crates/polars-metal-kernels/tests/test_groupby_build_sort_reference.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-kernels/tests/test_groupby_build_sort_reference.rs
use polars_metal_kernels::groupby_build_sort::reference::cpu_sort_segment;

#[test]
fn empty_input_yields_zero_groups() {
    let out = cpu_sort_segment(&[]);
    assert_eq!(out.n_groups, 0);
}

#[test]
fn all_unique_yields_n_groups() {
    let keys: Vec<u128> = (0u128..1000).collect();
    let out = cpu_sort_segment(&keys);
    assert_eq!(out.n_groups, 1000);
}

#[test]
fn all_same_yields_one_group() {
    let keys: Vec<u128> = vec![42u128; 1000];
    let out = cpu_sort_segment(&keys);
    assert_eq!(out.n_groups, 1);
    assert!(out.row_to_group.iter().all(|&g| g == 0));
}

#[test]
fn duplicates_collapsed_in_arbitrary_order() {
    let keys: Vec<u128> = vec![10, 20, 10, 30, 20, 10];
    let out = cpu_sort_segment(&keys);
    assert_eq!(out.n_groups, 3);
    // All rows with key=10 get the same group_id; same for 20 and 30.
    let g_for = |original_idx: usize| out.row_to_group[original_idx];
    assert_eq!(g_for(0), g_for(2));
    assert_eq!(g_for(0), g_for(5));
    assert_eq!(g_for(1), g_for(4));
}
```

- [ ] **Step 2: Implement the CPU reference**

```rust
// crates/polars-metal-kernels/src/groupby_build_sort/reference.rs
use crate::groupby_build_partitioned::BuildOutput;

pub fn cpu_sort_segment(keys: &[u128]) -> BuildOutput {
    if keys.is_empty() {
        return BuildOutput { row_to_group: vec![], first_row_per_group: vec![], n_groups: 0 };
    }
    let mut pairs: Vec<(u128, u32)> = keys.iter().enumerate()
        .map(|(i, &k)| (k, i as u32))
        .collect();
    pairs.sort_unstable_by_key(|(k, _)| *k);

    // Segment scan: each new key starts a new group.
    let mut row_to_group = vec![0u32; keys.len()];
    let mut first_row_per_group: Vec<u32> = Vec::new();
    let mut cur_group: u32 = 0;
    first_row_per_group.push(pairs[0].1);
    row_to_group[pairs[0].1 as usize] = 0;
    for i in 1..pairs.len() {
        if pairs[i].0 != pairs[i - 1].0 {
            cur_group += 1;
            first_row_per_group.push(pairs[i].1);
        }
        row_to_group[pairs[i].1 as usize] = cur_group;
    }
    BuildOutput {
        row_to_group,
        first_row_per_group,
        n_groups: cur_group + 1,
    }
}
```

```rust
// crates/polars-metal-kernels/src/groupby_build_sort/mod.rs
pub mod reference;
pub mod gpu;
```

- [ ] **Step 3: Run + commit**

```bash
cargo test -p polars-metal-kernels --test test_groupby_build_sort_reference -- --test-threads=1
```

Expected: 4 passes.

```bash
git add crates/polars-metal-kernels/src/groupby_build_sort/ crates/polars-metal-kernels/tests/test_groupby_build_sort_reference.rs
git commit -m "Kernel: CPU reference for sort-then-segment groupby build

Capability A2. Sort-stable + adjacent-difference scan; oracle for GPU
implementation's proptest comparison.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 25: GPU radix sort u128 — single-lane count + scatter

The full u128 radix sort needs 16 passes of 8-bit-radix sort. Task 25 implements one pass (one 8-bit lane); Task 26 chains them.

**Files:**
- Create: `shaders/groupby_sort_u128_lane.metal`
- Create: `crates/polars-metal-kernels/src/groupby_build_sort/gpu.rs` (stub for lane dispatch)
- Create: `crates/polars-metal-kernels/tests/test_groupby_sort_lane.rs`

- [ ] **Step 1: Write the MSL kernel — one 8-bit lane**

```metal
// shaders/groupby_sort_u128_lane.metal
//
// Per-lane (8-bit) radix sort pass for u128 keys paired with u32 row
// indices. Three kernels per lane:
//   1. histogram: count occurrences of each 0..255 digit globally.
//   2. (CPU exclusive scan turns counts → offsets)
//   3. scatter:   write (key, row_idx) into target positions.
//
// 8-bit lanes give 256 histogram bins, which fit in TGSM comfortably.

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

uint8_t extract_digit(uint64_t key_lo, uint64_t key_hi, uint lane_idx) {
    // lane_idx 0..7  → byte of key_lo
    // lane_idx 8..15 → byte of key_hi
    if (lane_idx < 8) return (uint8_t)((key_lo >> (lane_idx * 8)) & 0xFFu);
    return (uint8_t)((key_hi >> ((lane_idx - 8) * 8)) & 0xFFu);
}

kernel void lane_histogram(
    device const uint64_t* keys_lo  [[buffer(0)]],
    device const uint64_t* keys_hi  [[buffer(1)]],
    device atomic_uint*    bins     [[buffer(2)]],   // [256], pre-zeroed
    constant uint&         n_rows   [[buffer(3)]],
    constant uint&         lane_idx [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    uint8_t d = extract_digit(keys_lo[gid], keys_hi[gid], lane_idx);
    atomic_fetch_add_explicit(&bins[d], 1u, memory_order_relaxed);
}

kernel void lane_scatter(
    device const uint64_t* keys_lo_in  [[buffer(0)]],
    device const uint64_t* keys_hi_in  [[buffer(1)]],
    device const uint*     row_idx_in  [[buffer(2)]],
    device uint64_t*       keys_lo_out [[buffer(3)]],
    device uint64_t*       keys_hi_out [[buffer(4)]],
    device uint*           row_idx_out [[buffer(5)]],
    device atomic_uint*    cursors     [[buffer(6)]],  // seeded with offsets
    constant uint&         n_rows      [[buffer(7)]],
    constant uint&         lane_idx    [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    uint8_t d = extract_digit(keys_lo_in[gid], keys_hi_in[gid], lane_idx);
    uint write_pos = atomic_fetch_add_explicit(&cursors[d], 1u, memory_order_relaxed);
    keys_lo_out[write_pos] = keys_lo_in[gid];
    keys_hi_out[write_pos] = keys_hi_in[gid];
    row_idx_out[write_pos] = row_idx_in[gid];
}
```

- [ ] **Step 2: Write the lane test**

```rust
// crates/polars-metal-kernels/tests/test_groupby_sort_lane.rs
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::groupby_build_sort::gpu::run_radix_lane;

#[test]
fn lane_0_sorts_by_low_byte() {
    let device = MetalDevice::system_default().unwrap();
    let keys: Vec<u128> = vec![0x300, 0x100, 0x200, 0x400, 0x500];
    let idx:  Vec<u32>  = vec![0, 1, 2, 3, 4];
    let (sorted_keys, sorted_idx) = run_radix_lane(&device, &keys, &idx, /*lane=*/ 0).unwrap();
    // Sorting on byte 0 of each: the low byte is 0 for all of them — so
    // the sort is stable on the input order... but byte 1 differs.
    // Switch test: use the low byte directly:
    let keys: Vec<u128> = vec![0x3, 0x1, 0x2, 0x4, 0x5];
    let (sk, si) = run_radix_lane(&device, &keys, &[0,1,2,3,4], 0).unwrap();
    assert_eq!(sk, vec![0x1, 0x2, 0x3, 0x4, 0x5]);
    assert_eq!(si, vec![1, 2, 0, 3, 4]);
}
```

- [ ] **Step 3: Implement `run_radix_lane`**

```rust
// crates/polars-metal-kernels/src/groupby_build_sort/gpu.rs
pub fn run_radix_lane(
    device: &MetalDevice,
    keys: &[u128],
    row_idx_in: &[u32],
    lane: u32,
) -> Result<(Vec<u128>, Vec<u32>), String> {
    let keys_lo: Vec<u64> = keys.iter().map(|k| *k as u64).collect();
    let keys_hi: Vec<u64> = keys.iter().map(|k| (*k >> 64) as u64).collect();
    let n_rows = keys.len() as u32;

    let lib = device.load_shader_library("groupby_sort_u128_lane")?;
    let pso_hist = device.pipeline_for_function(&lib, "lane_histogram")?;
    let pso_scat = device.pipeline_for_function(&lib, "lane_scatter")?;

    let buf_lo  = device.new_buffer_from_slice(&keys_lo);
    let buf_hi  = device.new_buffer_from_slice(&keys_hi);
    let buf_idx = device.new_buffer_from_slice(row_idx_in);
    let buf_bins = device.new_buffer_zeroed::<u32>(256);
    let buf_n = device.new_buffer_from_slice(&[n_rows]);
    let buf_lane = device.new_buffer_from_slice(&[lane]);
    let queue = device.new_command_queue();

    queue.dispatch_1d(&pso_hist, &[&buf_lo, &buf_hi, &buf_bins, &buf_n, &buf_lane], n_rows)?;
    let bins: Vec<u32> = buf_bins.to_vec();

    // CPU exclusive scan.
    let mut offsets = vec![0u32; 256];
    for i in 1..256 { offsets[i] = offsets[i-1] + bins[i-1]; }
    let buf_offsets = device.new_buffer_from_slice(&offsets);
    let buf_cursors = device.new_buffer_from_slice(&offsets);  // seed cursors with offsets

    let buf_lo_out  = device.new_buffer_zeroed::<u64>(keys.len());
    let buf_hi_out  = device.new_buffer_zeroed::<u64>(keys.len());
    let buf_idx_out = device.new_buffer_zeroed::<u32>(keys.len());

    queue.dispatch_1d(
        &pso_scat,
        &[&buf_lo, &buf_hi, &buf_idx, &buf_lo_out, &buf_hi_out, &buf_idx_out, &buf_cursors, &buf_n, &buf_lane],
        n_rows,
    )?;

    let sorted_lo: Vec<u64> = buf_lo_out.to_vec();
    let sorted_hi: Vec<u64> = buf_hi_out.to_vec();
    let sorted_idx: Vec<u32> = buf_idx_out.to_vec();
    let sorted_keys: Vec<u128> = sorted_lo.iter().zip(sorted_hi.iter())
        .map(|(&lo, &hi)| (hi as u128) << 64 | lo as u128)
        .collect();
    Ok((sorted_keys, sorted_idx))
}
```

- [ ] **Step 4: Run + commit**

```bash
cargo test -p polars-metal-kernels --test test_groupby_sort_lane -- --test-threads=1
```

```bash
git add shaders/groupby_sort_u128_lane.metal crates/polars-metal-kernels/src/groupby_build_sort/gpu.rs crates/polars-metal-kernels/tests/test_groupby_sort_lane.rs
git commit -m "Kernel: single-lane radix-sort pass (histogram + scatter) for u128

Capability A2 building block. One 8-bit lane sorts by that byte stable;
chained across 16 lanes for full u128 sort (Task 26).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 26: Chain 16 lanes for full u128 sort

**Files:**
- Modify: `crates/polars-metal-kernels/src/groupby_build_sort/gpu.rs`
- Create: `crates/polars-metal-kernels/tests/test_groupby_sort_full.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-kernels/tests/test_groupby_sort_full.rs
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::groupby_build_sort::gpu::sort_u128;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn full_sort_matches_cpu_sort(
        keys in proptest::collection::vec(any::<u128>(), 1..4096usize),
    ) {
        let device = MetalDevice::system_default().unwrap();
        let (sorted_keys, sorted_idx) = sort_u128(&device, &keys).unwrap();

        // Sorted in ascending u128 order.
        for w in sorted_keys.windows(2) {
            prop_assert!(w[0] <= w[1]);
        }
        // sorted_idx is a permutation of 0..n.
        let mut perm = sorted_idx.clone();
        perm.sort();
        prop_assert_eq!(perm, (0u32..keys.len() as u32).collect::<Vec<_>>());
        // Each sorted_keys[i] equals keys[sorted_idx[i]].
        for i in 0..keys.len() {
            prop_assert_eq!(sorted_keys[i], keys[sorted_idx[i] as usize]);
        }
    }
}
```

- [ ] **Step 2: Implement `sort_u128` chaining 16 lanes**

```rust
// In crates/polars-metal-kernels/src/groupby_build_sort/gpu.rs
pub fn sort_u128(device: &MetalDevice, keys: &[u128]) -> Result<(Vec<u128>, Vec<u32>), String> {
    let mut current_keys: Vec<u128> = keys.to_vec();
    let mut current_idx:  Vec<u32>  = (0u32..keys.len() as u32).collect();
    for lane in 0u32..16 {
        let (next_keys, next_idx) = run_radix_lane(device, &current_keys, &current_idx, lane)?;
        current_keys = next_keys;
        current_idx = next_idx;
    }
    Ok((current_keys, current_idx))
}
```

- [ ] **Step 3: Run + commit**

```bash
cargo test -p polars-metal-kernels --test test_groupby_sort_full -- --test-threads=1
```

Expected: 1 property pass at 64 cases. (Each case allocates 16 sort passes × 256-bin histograms + scatters; budget ~2 min wall-clock.)

```bash
git add crates/polars-metal-kernels/src/groupby_build_sort/gpu.rs crates/polars-metal-kernels/tests/test_groupby_sort_full.rs
git commit -m "Kernel: full u128 radix sort by chaining 16 8-bit-lane passes

Capability A2. Restricted-scope helper for groupby; not a Polars
Sort op handler. M4 may generalize.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 27: Segment-boundary kernel + row_to_group derivation

**Files:**
- Create: `shaders/groupby_segments.metal`
- Modify: `crates/polars-metal-kernels/src/groupby_build_sort/gpu.rs` (add `sort_and_segment`)
- Create: `crates/polars-metal-kernels/tests/test_groupby_segment.rs`

- [ ] **Step 1: Write the MSL kernel**

```metal
// shaders/groupby_segments.metal
//
// Given sorted (key_lo, key_hi) arrays, find segment-start indices —
// rows where the key changes vs. the previous row. Output: a boolean
// mask of segment starts. CPU then scans this mask to assign group_ids.

#include <metal_stdlib>
using namespace metal;

kernel void segment_starts(
    device const uint64_t* sorted_lo [[buffer(0)]],
    device const uint64_t* sorted_hi [[buffer(1)]],
    device uchar*          starts    [[buffer(2)]],  // bit-packed 0/1; size = ceil(n_rows/8)
    constant uint&         n_rows    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    uchar bit = 0;
    if (gid == 0) {
        bit = 1;
    } else if (sorted_lo[gid] != sorted_lo[gid - 1] || sorted_hi[gid] != sorted_hi[gid - 1]) {
        bit = 1;
    }
    // Pack into byte; thread-safe write via atomic OR.
    uint byte_idx = gid >> 3;
    uint bit_idx = gid & 7;
    if (bit) {
        device atomic_uint* word = (device atomic_uint*)&starts[byte_idx & ~3u];
        uint mask = 1u << ((gid & 31));
        atomic_fetch_or_explicit(word, mask, memory_order_relaxed);
    }
}
```

- [ ] **Step 2: Write the high-level test**

```rust
// crates/polars-metal-kernels/tests/test_groupby_segment.rs
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::groupby_build_sort::gpu::sort_and_segment;
use polars_metal_kernels::groupby_build_sort::reference::cpu_sort_segment;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn sort_and_segment_matches_cpu_reference(
        keys in proptest::collection::vec(any::<u128>(), 1..2048usize),
    ) {
        let device = MetalDevice::system_default().unwrap();
        let gpu_out = sort_and_segment(&device, &keys).unwrap();
        let cpu_out = cpu_sort_segment(&keys);

        prop_assert_eq!(gpu_out.n_groups, cpu_out.n_groups);
        // Group assignment must be a permutation: same partitioning of rows.
        for a in 0..keys.len() {
            for b in 0..keys.len() {
                prop_assert_eq!(
                    gpu_out.row_to_group[a] == gpu_out.row_to_group[b],
                    cpu_out.row_to_group[a] == cpu_out.row_to_group[b]
                );
            }
        }
    }
}
```

- [ ] **Step 3: Implement `sort_and_segment`**

```rust
// In gpu.rs
pub fn sort_and_segment(device: &MetalDevice, keys: &[u128]) -> Result<BuildOutput, String> {
    let (sorted_keys, sorted_idx) = sort_u128(device, keys)?;
    let n_rows = keys.len() as u32;

    // Segment boundaries via GPU kernel.
    let buf_lo: Vec<u64> = sorted_keys.iter().map(|k| *k as u64).collect();
    let buf_hi: Vec<u64> = sorted_keys.iter().map(|k| (*k >> 64) as u64).collect();
    let lib = device.load_shader_library("groupby_segments")?;
    let pso = device.pipeline_for_function(&lib, "segment_starts")?;
    let starts_size = ((keys.len() + 7) >> 3) + 4;  // pad for atomic OR
    let buf_starts = device.new_buffer_zeroed::<u8>(starts_size);
    let buf_lo_d = device.new_buffer_from_slice(&buf_lo);
    let buf_hi_d = device.new_buffer_from_slice(&buf_hi);
    let buf_n = device.new_buffer_from_slice(&[n_rows]);
    let queue = device.new_command_queue();
    queue.dispatch_1d(&pso, &[&buf_lo_d, &buf_hi_d, &buf_starts, &buf_n], n_rows)?;
    let starts: Vec<u8> = buf_starts.to_vec();

    // CPU: scan segment bits to derive group_ids in sorted order, then
    // permute back to original order via sorted_idx.
    let mut row_to_group = vec![0u32; keys.len()];
    let mut first_row_per_group: Vec<u32> = Vec::new();
    let mut cur_group: u32 = 0;
    for i in 0..keys.len() {
        let bit = (starts[i >> 3] >> (i & 7)) & 1u8;
        if i > 0 && bit == 1 {
            cur_group += 1;
        }
        if i == 0 || bit == 1 {
            first_row_per_group.push(sorted_idx[i]);
        }
        row_to_group[sorted_idx[i] as usize] = cur_group;
    }
    Ok(BuildOutput { row_to_group, first_row_per_group, n_groups: cur_group + 1 })
}
```

- [ ] **Step 4: Run + commit**

```bash
cargo test -p polars-metal-kernels --test test_groupby_segment -- --test-threads=1
```

```bash
git add shaders/groupby_segments.metal crates/polars-metal-kernels/src/groupby_build_sort/gpu.rs crates/polars-metal-kernels/tests/test_groupby_segment.rs
git commit -m "Kernel: A2 sort+segment derives row_to_group via GPU sort + CPU scan

Capability A2 complete. Includes segment_starts MSL kernel; final
row_to_group derivation in CPU (microseconds at any practical size).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 28: Criterion bench — A2 vs A1 across cardinality

**Files:**
- Create: `benches/groupby_build_sort.rs`

- [ ] **Step 1: Write the bench**

```rust
// benches/groupby_build_sort.rs
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::groupby_build_partitioned::gpu::partition_and_build;
use polars_metal_kernels::groupby_build_sort::gpu::sort_and_segment;

fn bench_build_modes(c: &mut Criterion) {
    let device = MetalDevice::system_default().unwrap();
    let mut group = c.benchmark_group("groupby_build_a1_vs_a2");
    for &n_rows in &[1_000_000usize, 10_000_000] {
        for &n_groups in &[1024u32, 65_536, 1_048_576] {
            let keys: Vec<u128> = (0..n_rows).map(|i| (i % n_groups as usize) as u128).collect();
            group.bench_with_input(
                BenchmarkId::new(format!("a1_rows{n_rows}_groups{n_groups}"), n_rows),
                &keys,
                |b, keys| b.iter(|| partition_and_build(&device, keys, 16)),
            );
            group.bench_with_input(
                BenchmarkId::new(format!("a2_rows{n_rows}_groups{n_groups}"), n_rows),
                &keys,
                |b, keys| b.iter(|| sort_and_segment(&device, keys)),
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_build_modes);
criterion_main!(benches);
```

Add `[[bench]] name = "groupby_build_sort"` to `Cargo.toml`.

- [ ] **Step 2: Run + record + commit**

```bash
cargo bench -p polars-metal-kernels --bench groupby_build_sort
```

Expected: A1 wins at 1K groups (~10 ms vs A2's 80 ms). A2 wins at 1M groups (A1 overflows; A2 ~85 ms). Crossover around 32-64K.

```bash
git add benches/groupby_build_sort.rs crates/polars-metal-kernels/Cargo.toml
git commit -m "Bench: A1 vs A2 across 1024/65K/1M groups × 1M/10M rows

Capability A2 perf evidence; crossover tuning input for Phase 6 router.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 6 — A1/A2 routing + cardinality estimation

### Task 29: HyperLogLog++ cardinality estimator

**Files:**
- Create: `crates/polars-metal-kernels/src/groupby_cardinality.rs`
- Create: `crates/polars-metal-kernels/tests/test_groupby_cardinality.rs`
- Create: `shaders/groupby_cardinality_hll.metal`

- [ ] **Step 1: Implement CPU HLL++ sample-and-extrapolate**

```rust
// crates/polars-metal-kernels/src/groupby_cardinality.rs
//! HyperLogLog++ cardinality estimator over a sample of rows.
//! Used by the router to choose A1 vs A2 build phase.

const PRECISION: u32 = 14;  // 2^14 = 16K registers; ~1% error
const M: usize = 1 << PRECISION;

pub struct HllSketch {
    registers: Vec<u8>,
}

impl HllSketch {
    pub fn new() -> Self { Self { registers: vec![0u8; M] } }

    pub fn add_u128(&mut self, key: u128) {
        let hash = fxhash::hash64(&key.to_le_bytes());
        let bucket = (hash >> (64 - PRECISION)) as usize;
        let leading = ((hash << PRECISION) | (1u64 << (PRECISION - 1))).leading_zeros() as u8 + 1;
        if leading > self.registers[bucket] {
            self.registers[bucket] = leading;
        }
    }

    pub fn estimate(&self) -> u64 {
        // Standard HLL with the HLL++ small-cardinality bias correction.
        let alpha = 0.7213 / (1.0 + 1.079 / M as f64);
        let mut sum = 0.0f64;
        let mut zeros = 0;
        for &r in &self.registers {
            sum += 2.0f64.powi(-(r as i32));
            if r == 0 { zeros += 1; }
        }
        let raw = alpha * (M as f64).powi(2) / sum;
        if raw <= 2.5 * M as f64 && zeros != 0 {
            // Linear counting for small cardinality.
            (M as f64 * (M as f64 / zeros as f64).ln()) as u64
        } else {
            raw as u64
        }
    }
}

/// Convenience: estimate distinct keys in a sample, then extrapolate
/// to the full input size assuming uniform distribution. Returns an
/// upper-bound-friendly estimate.
pub fn estimate_cardinality(keys_sample: &[u128], total_rows: usize) -> u64 {
    if keys_sample.is_empty() { return 0; }
    let mut sketch = HllSketch::new();
    for &k in keys_sample { sketch.add_u128(k); }
    let sample_est = sketch.estimate();
    // Conservative extrapolation: assume cardinality scales linearly with
    // sample fraction up to total rows. (Real Zipf distributions converge
    // sub-linearly; this is the upper bound, which is what we want for
    // A1/A2 routing — better to over-estimate and run A2 than miss a
    // TGSM-overflow case.)
    let sample_fraction = keys_sample.len() as f64 / total_rows as f64;
    if sample_fraction >= 1.0 { sample_est } else { (sample_est as f64 / sample_fraction).min(total_rows as f64) as u64 }
}
```

Add `fxhash = "0.2"` to `crates/polars-metal-kernels/Cargo.toml` if not present.

- [ ] **Step 2: Test**

```rust
// crates/polars-metal-kernels/tests/test_groupby_cardinality.rs
use polars_metal_kernels::groupby_cardinality::estimate_cardinality;

#[test]
fn estimate_within_5_percent_for_uniform_input() {
    let keys: Vec<u128> = (0..100_000u128).collect();
    let est = estimate_cardinality(&keys, 100_000);
    let actual = 100_000;
    let err = ((est as f64 - actual as f64).abs() / actual as f64);
    assert!(err < 0.05, "got estimate {est}, actual {actual}, err {err:.3}");
}

#[test]
fn estimate_low_card_returns_small_value() {
    let keys: Vec<u128> = (0..10_000).map(|i| (i % 4) as u128).collect();
    let est = estimate_cardinality(&keys, 10_000);
    assert!(est < 10, "got estimate {est}, expected ~4");
}

#[test]
fn extrapolation_from_sample_in_range() {
    // 1K row sample suggesting 100 distinct keys → 10M-row total ⇒
    // a Zipf-ish workload might still have only ~hundred-thousand;
    // the conservative upper bound is min(linear_extrap, total_rows).
    let keys: Vec<u128> = (0..1000u128).map(|i| i % 100).collect();
    let est = estimate_cardinality(&keys, 10_000_000);
    assert!(est <= 10_000_000);
    assert!(est >= 100);  // at least the sample's cardinality
}
```

- [ ] **Step 3: Run + commit**

```bash
cargo test -p polars-metal-kernels --test test_groupby_cardinality -- --test-threads=1
```

```bash
git add crates/polars-metal-kernels/src/groupby_cardinality.rs crates/polars-metal-kernels/tests/test_groupby_cardinality.rs crates/polars-metal-kernels/Cargo.toml
git commit -m "Kernel: HLL++ cardinality estimator (CPU-side, sample-extrapolated)

Capability A routing input. Conservative (upper-bound) extrapolation
biases the router toward A2 when uncertain.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 30: Router cost-model — A1/A2 selection

**Files:**
- Modify: `crates/polars-metal-core/src/router/cost.rs`
- Modify: `crates/polars-metal-core/src/plan/mod.rs`
- Create: `crates/polars-metal-core/tests/test_router_a1_a2_selection.rs`

- [ ] **Step 1: Add `BuildPhaseMode` to plan IR**

```rust
// In crates/polars-metal-core/src/plan/mod.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildPhaseMode { Partitioned, Sort }
```

- [ ] **Step 2: Add the cost-model decision**

```rust
// In crates/polars-metal-core/src/router/cost.rs
pub fn decide_build_mode(estimated_cardinality: u64) -> BuildPhaseMode {
    // Crossover from Task 28 benches: A1 wins below ~64K groups; A2 wins above.
    // Use 32K as the conservative threshold (bias toward A2 when uncertain).
    if estimated_cardinality <= 32_768 {
        BuildPhaseMode::Partitioned
    } else {
        BuildPhaseMode::Sort
    }
}
```

- [ ] **Step 3: Test**

```rust
// crates/polars-metal-core/tests/test_router_a1_a2_selection.rs
use polars_metal_core::plan::BuildPhaseMode;
use polars_metal_core::router::cost::decide_build_mode;

#[test]
fn low_cardinality_picks_partitioned() {
    assert_eq!(decide_build_mode(4), BuildPhaseMode::Partitioned);
    assert_eq!(decide_build_mode(1024), BuildPhaseMode::Partitioned);
    assert_eq!(decide_build_mode(32_000), BuildPhaseMode::Partitioned);
}

#[test]
fn high_cardinality_picks_sort() {
    assert_eq!(decide_build_mode(65_536), BuildPhaseMode::Sort);
    assert_eq!(decide_build_mode(1_000_000), BuildPhaseMode::Sort);
}

#[test]
fn boundary_at_threshold() {
    assert_eq!(decide_build_mode(32_768), BuildPhaseMode::Partitioned);
    assert_eq!(decide_build_mode(32_769), BuildPhaseMode::Sort);
}
```

- [ ] **Step 4: Wire estimator into the UDF dispatch**

In `crates/polars-metal-core/src/udf.rs`'s `execute_groupby` entry, before invoking the build phase:

```rust
// Sample 16K rows or all (if fewer). Encode the sampled keys to u128.
let sample_size = 16_384.min(n_rows);
let sample_keys = encode_keys_sample(&key_columns, sample_size)?;
let est_card = estimate_cardinality(&sample_keys, n_rows);
let mode = decide_build_mode(est_card);
let build_out = match mode {
    BuildPhaseMode::Partitioned => {
        match partition_and_build(&device, &encoded_keys, /*n_partitions=*/ 16) {
            Ok(out) => out,
            Err(e) if e.contains("overflow") => {
                eprintln!("polars-metal: A1 TGSM overflow, falling back to A2");
                sort_and_segment(&device, &encoded_keys)?
            }
            Err(e) => return Err(EngineError::Compute(e)),
        }
    }
    BuildPhaseMode::Sort => sort_and_segment(&device, &encoded_keys)?,
};
```

- [ ] **Step 5: Run + commit**

```bash
cargo test -p polars-metal-core --test test_router_a1_a2_selection -- --test-threads=1
make wheel
pytest tests/python_integration/ -k groupby -v
```

```bash
git add crates/polars-metal-core/src/router/cost.rs crates/polars-metal-core/src/plan/mod.rs crates/polars-metal-core/src/udf.rs crates/polars-metal-core/tests/test_router_a1_a2_selection.rs
git commit -m "Router: A1/A2 selection by HLL++ cardinality estimate

Capability A complete. Threshold 32K (conservative, biased toward A2);
overflow fallback re-dispatches as A2 transparently.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 31: Python integration test — A1/A2 routing visible in debug log

**Files:**
- Modify: `tests/python_integration/test_routing.py`

- [ ] **Step 1: Add the test**

```python
# tests/python_integration/test_routing.py (additions)
import polars as pl
import polars_metal as pm


def test_low_cardinality_routes_a1():
    df = pl.DataFrame({"k": [(i % 4) for i in range(100_000)],
                       "v": [i * 1.0 for i in range(100_000)]})
    q = df.lazy().group_by("k").agg(pl.col("v").sum())
    logs = []
    eng = pm.MetalEngine(debug=True, debug_sink=logs.append)
    _ = q.collect(engine=eng)
    assert any("build_mode=Partitioned" in line for line in logs), logs


def test_high_cardinality_routes_a2():
    n_groups = 100_000
    df = pl.DataFrame({"k": [(i % n_groups) for i in range(1_000_000)],
                       "v": [i * 1.0 for i in range(1_000_000)]})
    q = df.lazy().group_by("k").agg(pl.col("v").sum())
    logs = []
    eng = pm.MetalEngine(debug=True, debug_sink=logs.append)
    _ = q.collect(engine=eng)
    assert any("build_mode=Sort" in line for line in logs), logs


def test_a1_overflow_falls_back_to_a2():
    """If A1 reports TGSM overflow, the orchestrator transparently retries A2."""
    # Construct adversarial input: 50K unique keys collapsed into 4 partitions
    # → ~12K per partition, exceeds 1024 TGSM slots × 75% load factor.
    n_groups = 50_000
    df = pl.DataFrame({"k": [(i % n_groups) for i in range(500_000)],
                       "v": [i * 1.0 for i in range(500_000)]})
    q = df.lazy().group_by("k").agg(pl.col("v").sum())
    logs = []
    eng = pm.MetalEngine(debug=True, debug_sink=logs.append)
    cpu = q.collect(engine="cpu").sort("k")
    metal = q.collect(engine=eng).sort("k")
    pl.testing.assert_frame_equal(cpu, metal)
    # Log should mention either Sort (cardinality estimator picked it directly)
    # or a fallback (estimator picked A1, runtime overflowed).
    assert any("Sort" in line or "fallback" in line for line in logs), logs
```

Logging integration assumes M2 has a `debug_sink` parameter; if not, add it in `python/polars_metal/_engine.py` as a list-callable that captures lines emitted from Rust via `tracing` or `println!`. M2's `debug=True` likely prints to stderr; redirect via a hook for testability.

- [ ] **Step 2: Build + run**

```bash
make wheel
pytest tests/python_integration/test_routing.py -v
```

Expected: all 3 routing tests pass (plus M2 pre-existing).

- [ ] **Step 3: Commit**

```bash
git add tests/python_integration/test_routing.py python/polars_metal/_engine.py
git commit -m "Tests: A1/A2 routing decisions visible in MetalEngine debug log

Capability A. Verifies the cost model picks Partitioned at low
cardinality, Sort at high, and that overflow fallback runs cleanly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 7 — String-key groupby, dictionary path (capability D Phase 1)

For low-to-medium cardinality string keys, the cheapest path is CPU dictionary encoding: build a `Vec<&str>` of unique strings, replace each row's string with its `u32` index, feed the indices to the existing composite-key encoder. Phase 8 adds a GPU hash kernel for the high-cardinality case.

### Task 32: Dictionary encoder in `polars-metal-buffer`

**Files:**
- Create: `crates/polars-metal-buffer/src/dict.rs`
- Create: `crates/polars-metal-buffer/tests/test_dict.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-buffer/tests/test_dict.rs
use polars_metal_buffer::dict::{build_dict, decode_dict};

#[test]
fn dict_roundtrip_simple() {
    let strings = vec!["apple", "banana", "apple", "cherry", "banana"];
    let (dict, codes) = build_dict(&strings);
    assert_eq!(dict.len(), 3);
    assert!(dict.contains(&"apple".to_string()));
    let decoded: Vec<String> = codes.iter().map(|&c| dict[c as usize].clone()).collect();
    assert_eq!(decoded, vec!["apple", "banana", "apple", "cherry", "banana"]);
}

#[test]
fn dict_handles_empty_strings_and_nulls() {
    let strings: Vec<Option<&str>> = vec![Some("a"), None, Some(""), Some("a"), None];
    let (dict, codes, nulls) = build_dict_nullable(&strings);
    assert_eq!(dict.len(), 2); // "a" and ""
    assert_eq!(nulls, vec![true, false, true, true, false]);
}
```

- [ ] **Step 2: Implement**

```rust
// crates/polars-metal-buffer/src/dict.rs
use std::collections::HashMap;

pub fn build_dict(strings: &[&str]) -> (Vec<String>, Vec<u32>) {
    let mut dict: Vec<String> = Vec::new();
    let mut seen: HashMap<&str, u32> = HashMap::new();
    let mut codes = Vec::with_capacity(strings.len());
    for &s in strings {
        let code = *seen.entry(s).or_insert_with(|| {
            let idx = dict.len() as u32;
            dict.push(s.to_string());
            idx
        });
        codes.push(code);
    }
    (dict, codes)
}

pub fn build_dict_nullable(strings: &[Option<&str>]) -> (Vec<String>, Vec<u32>, Vec<bool>) {
    let mut dict: Vec<String> = Vec::new();
    let mut seen: HashMap<String, u32> = HashMap::new();
    let mut codes = Vec::with_capacity(strings.len());
    let mut valid = Vec::with_capacity(strings.len());
    for opt in strings {
        match opt {
            Some(s) => {
                valid.push(true);
                let code = *seen.entry(s.to_string()).or_insert_with(|| {
                    let idx = dict.len() as u32;
                    dict.push(s.to_string());
                    idx
                });
                codes.push(code);
            }
            None => {
                valid.push(false);
                codes.push(0); // sentinel; ignored when valid=false
            }
        }
    }
    (dict, codes, valid)
}

pub fn decode_dict(dict: &[String], codes: &[u32]) -> Vec<String> {
    codes.iter().map(|&c| dict[c as usize].clone()).collect()
}
```

- [ ] **Step 3: Run + commit**

```bash
cargo test -p polars-metal-buffer --test test_dict -- --test-threads=1
```

```bash
git add crates/polars-metal-buffer/src/dict.rs crates/polars-metal-buffer/tests/test_dict.rs crates/polars-metal-buffer/src/lib.rs
git commit -m "Buffer: dictionary encoder for Utf8 columns (CPU-side)

Capability D Phase 1 helper. Build dict + u32 codes; null-aware variant
returns validity vector. Feeds the composite-key encoder.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 33: Extend `KeyDtype` with `Utf8` variant

**Files:**
- Modify: `crates/polars-metal-core/src/plan/mod.rs`
- Modify: `crates/polars-metal-kernels/src/groupby.rs`

- [ ] **Step 1: Add variant**

```rust
// In plan/mod.rs KeyDtype enum:
KeyDtype::Utf8,
// width_bits arm: KeyDtype::Utf8 => 32  // dictionary-encoded as u32
```

The composite-key encoder treats a `KeyDtype::Utf8` column exactly like a `KeyDtype::U32` column whose bytes come from the dictionary-encoded codes. The result reconstruction needs to remember the dictionary alongside the schema so decode can map codes → strings.

- [ ] **Step 2: Extend `KeySchema` to optionally carry per-column dictionaries**

```rust
pub struct KeySchema {
    pub columns: Vec<KeyColSchema>,
}

pub struct KeyColSchema {
    pub dtype: KeyDtype,
    pub width_bits: u32,
    pub offset_bits: u32,
    pub name: String,
    pub dict: Option<Vec<String>>,  // only for Utf8
}
```

- [ ] **Step 3: Encoder/decoder dispatch**

```rust
// In encode_keys: when KeyColumn::Utf8 { strings } is encountered,
// build dict, encode codes as u32, attach dict to KeyColSchema.
//
// In decode_keys: when KeyColSchema::dtype == Utf8, read u32 code,
// look up dict[code], emit String.
```

- [ ] **Step 4: Add test**

```rust
// crates/polars-metal-kernels/tests/test_key_encoding_utf8.rs
use polars_metal_kernels::groupby::{encode_keys, decode_keys, KeyColumn};

#[test]
fn utf8_roundtrip() {
    let col = KeyColumn::from_utf8(&["a", "b", "a", "c"]);
    let (encoded, schema) = encode_keys(&[col]);
    let decoded = decode_keys(&encoded, &schema);
    assert_eq!(decoded[0].as_utf8_slice(), vec!["a", "b", "a", "c"]);
}

#[test]
fn utf8_combined_with_int_key() {
    let col_s = KeyColumn::from_utf8(&["x", "y", "x", "z"]);
    let col_i = KeyColumn::from_i32(&[1, 2, 1, 3]);
    let (encoded, schema) = encode_keys(&[col_s, col_i]);
    let decoded = decode_keys(&encoded, &schema);
    assert_eq!(decoded[0].as_utf8_slice(), vec!["x", "y", "x", "z"]);
    assert_eq!(decoded[1].as_i32_slice(), &[1, 2, 1, 3]);
}
```

- [ ] **Step 5: Run + commit**

```bash
cargo test -p polars-metal-kernels --test test_key_encoding_utf8 -- --test-threads=1
```

```bash
git add crates/polars-metal-core/src/plan/mod.rs crates/polars-metal-kernels/src/groupby.rs crates/polars-metal-kernels/tests/test_key_encoding_utf8.rs
git commit -m "Plan + Kernel: KeyDtype::Utf8 via dictionary encoding

Capability D Phase 1. Utf8 column → dict + u32 codes → composite-key
encoder treats as 32-bit slot. Schema carries dict for decode.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 34: Walker — recognize Polars Utf8 dtype as key

**Files:**
- Modify: `python/polars_metal/_walker.py`
- Create: `tests/python_integration/test_string_groupby.py`

- [ ] **Step 1: Extend dtype map**

```python
_POLARS_DTYPE_TO_KEY_DTYPE[pl.Utf8] = "Utf8"
# (in some Polars versions this is pl.String — handle both)
if hasattr(pl, "String"):
    _POLARS_DTYPE_TO_KEY_DTYPE[pl.String] = "Utf8"
```

- [ ] **Step 2: Walker handles Utf8 key columns**

The walker passes the raw string column to the Rust side; the Rust side builds the dictionary. The wire format gains a `key_dtype: "Utf8"` field; the Rust side builds the dictionary upon receipt.

- [ ] **Step 3: Integration test**

```python
# tests/python_integration/test_string_groupby.py
import polars as pl
from polars.testing import assert_frame_equal

import polars_metal as pm


def test_groupby_single_string_key():
    df = pl.DataFrame({
        "k": ["A", "N", "A", "N", "A"] * 2000,
        "v": [1.0, 2.0, 3.0, 4.0, 5.0] * 2000,
    })
    q = df.lazy().group_by("k").agg(pl.col("v").sum(), pl.len())
    assert_frame_equal(
        q.collect(engine="cpu").sort("k"),
        q.collect(engine=pm.MetalEngine()).sort("k"),
    )


def test_groupby_two_string_keys():
    """Q1 shape: two string keys, multiple aggregations."""
    df = pl.DataFrame({
        "rf": ["A", "N", "R", "A", "N", "R"] * 2000,
        "ls": ["F", "F", "O", "O", "O", "F"] * 2000,
        "v":  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 2000,
    })
    q = df.lazy().group_by("rf", "ls").agg(pl.col("v").sum(), pl.col("v").mean())
    assert_frame_equal(
        q.collect(engine="cpu").sort(["rf", "ls"]),
        q.collect(engine=pm.MetalEngine()).sort(["rf", "ls"]),
    )


def test_groupby_null_strings():
    df = pl.DataFrame({
        "k": ["A", None, "A", None, "B"] * 1000,
        "v": [1.0, 2.0, 3.0, 4.0, 5.0] * 1000,
    })
    q = df.lazy().group_by("k").agg(pl.col("v").sum())
    assert_frame_equal(
        q.collect(engine="cpu").sort("k"),
        q.collect(engine=pm.MetalEngine()).sort("k"),
    )


def test_groupby_empty_strings():
    df = pl.DataFrame({
        "k": ["", "x", "", "x"] * 500,
        "v": [1.0, 2.0, 3.0, 4.0] * 500,
    })
    q = df.lazy().group_by("k").agg(pl.col("v").sum())
    assert_frame_equal(
        q.collect(engine="cpu").sort("k"),
        q.collect(engine=pm.MetalEngine()).sort("k"),
    )


def test_groupby_unicode_strings():
    df = pl.DataFrame({
        "k": ["α", "β", "α", "β", "γ"] * 1000,
        "v": [1.0, 2.0, 3.0, 4.0, 5.0] * 1000,
    })
    q = df.lazy().group_by("k").agg(pl.col("v").sum())
    assert_frame_equal(
        q.collect(engine="cpu").sort("k"),
        q.collect(engine=pm.MetalEngine()).sort("k"),
    )
```

- [ ] **Step 4: Build + run + commit**

```bash
make wheel
pytest tests/python_integration/test_string_groupby.py -v
```

```bash
git add python/polars_metal/_walker.py tests/python_integration/test_string_groupby.py
git commit -m "Walker + Tests: Utf8 keys via dictionary path

Capability D Phase 1 user-visible. Single key, multi-key, nulls, empty
strings, unicode — all match CPU byte-exact.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 8 — String-key groupby, dedicated hash kernel (capability D Phase 2)

For high-cardinality string keys (>~32K unique), dictionary build cost on CPU dominates. Phase 8 adds a GPU MSL kernel that hashes Utf8 columns directly (FNV-1a) and feeds 32-bit hashes into the composite-key encoder. Collisions are resolved by full-string compare in A1's inner loop.

### Task 35: String-hash MSL kernel

**Files:**
- Create: `shaders/groupby_string_hash.metal`
- Create: `crates/polars-metal-kernels/src/string_hash.rs`
- Create: `crates/polars-metal-kernels/tests/test_string_hash.rs`

- [ ] **Step 1: Write the MSL kernel**

```metal
// shaders/groupby_string_hash.metal
//
// FNV-1a per-row hash over Polars' Utf8 layout (offsets + data).
// 32-bit output; downstream A1/A2 build phase reads these as a
// fixed-width u32 key column.

#include <metal_stdlib>
using namespace metal;

kernel void hash_strings_fnv1a(
    device const uint*   offsets [[buffer(0)]],  // [n_rows+1]
    device const uchar*  data    [[buffer(1)]],
    device uint*         hashes  [[buffer(2)]],  // [n_rows]
    constant uint&       n_rows  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    uint start = offsets[gid];
    uint end   = offsets[gid + 1];
    uint h = 2166136261u;  // FNV-1a seed
    for (uint i = start; i < end; ++i) {
        h ^= (uint)data[i];
        h *= 16777619u;
    }
    hashes[gid] = h;
}
```

- [ ] **Step 2: Write the dispatch + test**

```rust
// crates/polars-metal-kernels/src/string_hash.rs
use polars_metal_buffer::{MetalDevice, MetalBuffer};

pub fn hash_strings_fnv1a(
    device: &MetalDevice,
    offsets: &[u32],
    data: &[u8],
) -> Result<Vec<u32>, String> {
    let n_rows = (offsets.len() - 1) as u32;
    let lib = device.load_shader_library("groupby_string_hash")?;
    let pso = device.pipeline_for_function(&lib, "hash_strings_fnv1a")?;
    let buf_offsets = device.new_buffer_from_slice(offsets);
    let buf_data = device.new_buffer_from_slice(data);
    let buf_hashes = device.new_buffer_zeroed::<u32>(n_rows as usize);
    let buf_n = device.new_buffer_from_slice(&[n_rows]);
    let queue = device.new_command_queue();
    queue.dispatch_1d(&pso, &[&buf_offsets, &buf_data, &buf_hashes, &buf_n], n_rows)?;
    Ok(buf_hashes.to_vec())
}

/// CPU reference for proptest comparison.
pub fn cpu_hash_strings_fnv1a(strings: &[&str]) -> Vec<u32> {
    strings.iter().map(|s| {
        let mut h = 2166136261u32;
        for b in s.bytes() {
            h ^= b as u32;
            h = h.wrapping_mul(16777619);
        }
        h
    }).collect()
}
```

```rust
// crates/polars-metal-kernels/tests/test_string_hash.rs
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::string_hash::{cpu_hash_strings_fnv1a, hash_strings_fnv1a};
use proptest::prelude::*;

fn strings_to_arrow(strs: &[&str]) -> (Vec<u32>, Vec<u8>) {
    let mut offsets = vec![0u32];
    let mut data = Vec::new();
    for s in strs {
        data.extend_from_slice(s.as_bytes());
        offsets.push(data.len() as u32);
    }
    (offsets, data)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn fnv1a_gpu_matches_cpu(
        strings in proptest::collection::vec("\\PC*", 1..256usize),
    ) {
        let device = MetalDevice::system_default().unwrap();
        let strs: Vec<&str> = strings.iter().map(|s| s.as_str()).collect();
        let (offsets, data) = strings_to_arrow(&strs);
        let gpu_hashes = hash_strings_fnv1a(&device, &offsets, &data).unwrap();
        let cpu_hashes = cpu_hash_strings_fnv1a(&strs);
        prop_assert_eq!(gpu_hashes, cpu_hashes);
    }
}
```

- [ ] **Step 3: Run + commit**

```bash
cargo test -p polars-metal-kernels --test test_string_hash -- --test-threads=1
```

```bash
git add shaders/groupby_string_hash.metal crates/polars-metal-kernels/src/string_hash.rs crates/polars-metal-kernels/tests/test_string_hash.rs
git commit -m "Kernel: FNV-1a string hash MSL kernel + CPU reference

Capability D Phase 2. Per-row hash; downstream A1/A2 consume as u32 key.
Collision resolution deferred to A1's inner loop (full-string compare
slot-by-slot).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 36: Router decision — dict vs hash by sample cardinality

**Files:**
- Modify: `crates/polars-metal-core/src/router/cost.rs`
- Modify: `crates/polars-metal-core/src/udf.rs`

- [ ] **Step 1: Add the decision**

```rust
// In cost.rs
pub enum StringKeyMode { Dictionary, GpuHash }

pub fn decide_string_key_mode(estimated_unique: u64) -> StringKeyMode {
    if estimated_unique <= 32_768 {
        StringKeyMode::Dictionary
    } else {
        StringKeyMode::GpuHash
    }
}
```

- [ ] **Step 2: Wire into UDF**

In `udf.rs` `execute_groupby`, when a key column is Utf8:

```rust
let unique_est = estimate_string_cardinality(&utf8_column_sample);
let mode = decide_string_key_mode(unique_est);
let key_buffer: MetalBuffer<u32> = match mode {
    StringKeyMode::Dictionary => {
        let (dict, codes) = build_dict_nullable(&utf8_column);
        // ... encode codes as u32 key column
    }
    StringKeyMode::GpuHash => {
        let (offsets, data) = polars_utf8_to_arrow(&utf8_column);
        hash_strings_fnv1a(&device, &offsets, &data)?
        // ... use hashes as u32 key column; collisions resolved in A1
    }
};
```

For the GpuHash case, collisions mean two different strings hash to the same u32. The downstream build phase (A1 or A2) needs to **compare the actual strings** when slot keys equal — otherwise two distinct strings could merge into one group. The simplest implementation:

- For A1 (partitioned hash): the per-row hash is *only* the u32; on slot-match in TGSM, the kernel reads the *original* string column and compares. This is a 1-2 cache line read per probe.
- For A2 (sort+segment): sort by `(hash, string)` lexicographically; the segment-boundary kernel compares strings directly when hashes equal.

Both adaptations are non-trivial — add a `FullKeyCompare` callback that the build kernels invoke on tie.

**Practical scope cut for M3:** if collision handling proves complex, dictionary-encode anyway as the safe path and document the high-card string-hash kernel as a Phase-8 follow-on. The CPU dict build at 1M unique strings is ~50-150 ms — slow but correct. M3's exit criterion 6 says "high-cardinality verified" which we can meet by *also* benchmarking the dict path at high cardinality; the dedicated hash kernel becomes an *optional* perf path.

- [ ] **Step 3: Test (collision behavior under FNV-1a)**

```rust
// crates/polars-metal-kernels/tests/test_string_hash_collisions.rs
#[test]
fn fnv1a_collisions_are_rare_in_practice() {
    // Sanity: 100K random English-language strings.
    let strs: Vec<String> = (0..100_000)
        .map(|i| format!("user_{}", i))
        .collect();
    let strs_ref: Vec<&str> = strs.iter().map(|s| s.as_str()).collect();
    let hashes = polars_metal_kernels::string_hash::cpu_hash_strings_fnv1a(&strs_ref);
    let mut seen = std::collections::HashMap::new();
    let mut collisions = 0;
    for (i, &h) in hashes.iter().enumerate() {
        if let Some(&prev) = seen.get(&h) {
            if strs_ref[i] != strs_ref[prev] {
                collisions += 1;
            }
        } else {
            seen.insert(h, i);
        }
    }
    // Expect a small handful (< 5 expected at 100K via birthday bound),
    // but the test asserts the bound to catch a broken hash.
    assert!(collisions < 50, "got {collisions} collisions, want < 50");
}
```

- [ ] **Step 4: Commit**

```bash
git add crates/polars-metal-core/src/router/cost.rs crates/polars-metal-core/src/udf.rs crates/polars-metal-kernels/tests/test_string_hash_collisions.rs
git commit -m "Router: choose dict vs gpu-hash for Utf8 keys by cardinality

Capability D Phase 2 routing. 32K threshold. Collision handling: see
spec § D Phase 2 risk note; M3 may ship dict-only for safety, with
hash-kernel as a profile-then-optimize follow-on.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 9 — Multi-chunk Series support (capability E)

M2 falls back when any input Series has more than one Arrow chunk. Real LazyFrames often arrive chunked (parquet row-group reads, concatenated DataFrames). M3 lands two regimes:

1. **Rechunk-on-import** when the total bytes of a column is < 256 MB. CPU is fast at `series.rechunk()`; the resulting single-chunk view feeds existing kernels unchanged.
2. **Chunked dispatch** when total bytes ≥ 256 MB. Kernels accept multi-chunk inputs; iterate chunks per pass, accumulating into one output buffer.

For M3's queries, all four queries fit comfortably under 256 MB per column at 10M rows (largest column: 80 MB). The 100M-row Q1 has 800 MB per column → exercises chunked dispatch.

### Task 37: Chunked buffer view

**Files:**
- Create: `crates/polars-metal-buffer/src/chunked.rs`
- Create: `crates/polars-metal-buffer/tests/test_chunked.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-buffer/tests/test_chunked.rs
use polars_metal_buffer::chunked::{ChunkedView, RechunkPolicy};

#[test]
fn small_series_takes_rechunk_path() {
    let chunks: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let view = ChunkedView::from_chunks(&chunks, RechunkPolicy::Auto);
    assert!(view.is_single_chunk());
    assert_eq!(view.total_len(), 4);
    assert_eq!(view.as_single_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn large_series_keeps_chunks() {
    let chunks: Vec<Vec<f32>> = vec![vec![0.0; 50_000_000], vec![0.0; 50_000_000]];
    let view = ChunkedView::from_chunks(&chunks, RechunkPolicy::Auto);
    assert!(!view.is_single_chunk());
    assert_eq!(view.total_len(), 100_000_000);
    assert_eq!(view.chunk_count(), 2);
}

#[test]
fn iteration_yields_chunk_slices() {
    let chunks: Vec<Vec<f32>> = vec![vec![1.0, 2.0], vec![3.0, 4.0, 5.0]];
    let view = ChunkedView::from_chunks(&chunks, RechunkPolicy::AlwaysChunked);
    let mut collected = Vec::new();
    for chunk in view.iter_chunks() {
        collected.extend_from_slice(chunk);
    }
    assert_eq!(collected, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
}
```

- [ ] **Step 2: Implement**

```rust
// crates/polars-metal-buffer/src/chunked.rs
pub enum RechunkPolicy {
    /// < 256 MB total → rechunk; ≥ 256 MB → keep chunks.
    Auto,
    /// Always rechunk into one buffer.
    AlwaysSingle,
    /// Always keep chunks, even small ones (test convenience).
    AlwaysChunked,
}

const AUTO_RECHUNK_THRESHOLD_BYTES: usize = 256 * 1024 * 1024;

pub struct ChunkedView<T: Copy> {
    chunks: Vec<Vec<T>>,
    single: Option<Vec<T>>,
}

impl<T: Copy> ChunkedView<T> {
    pub fn from_chunks(chunks: &[Vec<T>], policy: RechunkPolicy) -> Self {
        let total_bytes = chunks.iter().map(|c| c.len() * std::mem::size_of::<T>()).sum::<usize>();
        let should_rechunk = match policy {
            RechunkPolicy::AlwaysSingle => true,
            RechunkPolicy::AlwaysChunked => false,
            RechunkPolicy::Auto => total_bytes < AUTO_RECHUNK_THRESHOLD_BYTES,
        };
        if should_rechunk {
            let mut single = Vec::with_capacity(chunks.iter().map(|c| c.len()).sum());
            for c in chunks { single.extend_from_slice(c); }
            Self { chunks: vec![], single: Some(single) }
        } else {
            Self { chunks: chunks.iter().map(|c| c.clone()).collect(), single: None }
        }
    }

    pub fn is_single_chunk(&self) -> bool { self.single.is_some() }
    pub fn total_len(&self) -> usize {
        self.single.as_ref().map(|s| s.len())
            .unwrap_or_else(|| self.chunks.iter().map(|c| c.len()).sum())
    }
    pub fn chunk_count(&self) -> usize {
        if self.single.is_some() { 1 } else { self.chunks.len() }
    }
    pub fn as_single_slice(&self) -> &[T] {
        self.single.as_ref().expect("not single-chunk")
    }
    pub fn iter_chunks(&self) -> impl Iterator<Item = &[T]> + '_ {
        if let Some(s) = &self.single {
            Box::new(std::iter::once(s.as_slice())) as Box<dyn Iterator<Item = _>>
        } else {
            Box::new(self.chunks.iter().map(|c| c.as_slice()))
        }
    }
}
```

- [ ] **Step 3: Run + commit**

```bash
cargo test -p polars-metal-buffer --test test_chunked -- --test-threads=1
```

```bash
git add crates/polars-metal-buffer/src/chunked.rs crates/polars-metal-buffer/tests/test_chunked.rs crates/polars-metal-buffer/src/lib.rs
git commit -m "Buffer: ChunkedView with auto/always policies

Capability E foundation. 256 MB threshold for auto-rechunk; tests
verify both regimes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 38: Walker — remove M2's multi-chunk fallback

**Files:**
- Modify: `python/polars_metal/_walker.py`
- Modify: `crates/polars-metal-core/src/udf.rs` (use `ChunkedView` to receive multi-chunk inputs)
- Create: `tests/python_integration/test_multichunk.py`

- [ ] **Step 1: Remove the fallback**

In M2's walker, find the line that falls back when `series.n_chunks() > 1`. Replace it with: pass all chunks to Rust via the PyO3 boundary. Rust constructs `ChunkedView` with `RechunkPolicy::Auto`.

For M3's groupby pipeline: when `ChunkedView::is_single_chunk()` (most cases under 256 MB), the existing kernel dispatch path runs unchanged. When chunked, the kernel orchestrator iterates chunks and accumulates into single output buffers (kernels are already row-oriented, so chunking by row range is straightforward).

- [ ] **Step 2: Integration test**

```python
# tests/python_integration/test_multichunk.py
import polars as pl
from polars.testing import assert_frame_equal

import polars_metal as pm


def test_concat_two_dataframes_groupby():
    """LazyFrame from concat has multi-chunk Series."""
    a = pl.DataFrame({"k": [0, 0, 1, 1], "v": [1.0, 2.0, 3.0, 4.0]})
    b = pl.DataFrame({"k": [0, 1, 0, 1], "v": [5.0, 6.0, 7.0, 8.0]})
    combined = pl.concat([a, b], rechunk=False)
    assert combined["k"].n_chunks() > 1
    q = combined.lazy().group_by("k").agg(pl.col("v").sum())
    assert_frame_equal(
        q.collect(engine="cpu").sort("k"),
        q.collect(engine=pm.MetalEngine()).sort("k"),
    )


def test_many_small_chunks_groupby():
    parts = [pl.DataFrame({"k": [i % 4], "v": [float(i)]}) for i in range(1000)]
    combined = pl.concat(parts, rechunk=False)
    q = combined.lazy().group_by("k").agg(pl.col("v").sum())
    assert_frame_equal(
        q.collect(engine="cpu").sort("k"),
        q.collect(engine=pm.MetalEngine()).sort("k"),
    )
```

- [ ] **Step 3: Build + run + commit**

```bash
make wheel
pytest tests/python_integration/test_multichunk.py -v
```

```bash
git add python/polars_metal/_walker.py crates/polars-metal-core/src/udf.rs tests/python_integration/test_multichunk.py
git commit -m "Walker: chunked Series no longer falls back

Capability E. ChunkedView with Auto policy handles both rechunk and
chunked-dispatch transparently. Multi-chunk integration tests pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 10 — Filter routing into GPU when downstream is GPU (capability C)

M2 routes filter unconditionally to CPU. For `filter → groupby` pipelines where the groupby runs GPU, the filter routes GPU too — eliminating the intermediate Polars DataFrame materialization. The filter kernel from M1 is reused unchanged; what changes is *when* the router calls it.

### Task 39: Subtree-id marking in the router

**Files:**
- Modify: `crates/polars-metal-core/src/router/affinity.rs`
- Create: `crates/polars-metal-core/tests/test_affinity_filter_groupby.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-core/tests/test_affinity_filter_groupby.rs
use polars_metal_core::plan::{MetalPlanNode, AggSpec, AggOp, MetalDtype};
use polars_metal_core::router::{compute_lifting_plan, NodeDecision};

#[test]
fn filter_above_gpu_groupby_lifts_to_gpu() {
    let plan = MetalPlanNode::Filter {
        predicate: /* synthetic predicate */,
        input: Box::new(MetalPlanNode::GroupBy {
            keys: /* 2 fixed-width keys */,
            aggs: vec![AggSpec::Simple {
                input_column: "v".into(),
                op: AggOp::Sum,
                output_alias: "s".into(),
                output_dtype: MetalDtype::F32,
            }],
            input: Box::new(MetalPlanNode::DataFrameScan { n_rows: 1_000_000, .. }),
        }),
    };
    // Wait — Filter is *above* GroupBy in IR, not below. Polars IR has
    // Filter as an op that takes a child input. The "filter feeds groupby"
    // shape is GroupBy { input: Filter { input: Scan } }. Let's rewrite:

    let plan = MetalPlanNode::GroupBy {
        keys: /* ... */,
        aggs: vec![/* ... */],
        input: Box::new(MetalPlanNode::Filter {
            predicate: /* ... */,
            input: Box::new(MetalPlanNode::DataFrameScan { n_rows: 1_000_000, .. }),
        }),
    };

    let lifting = compute_lifting_plan(&plan);
    let filter_id = /* ... */;
    let groupby_id = /* ... */;
    assert_eq!(lifting.get(&filter_id), Some(&NodeDecision::GpuLift));
    assert_eq!(lifting.get(&groupby_id), Some(&NodeDecision::GpuLift));
}

#[test]
fn filter_above_non_gpu_consumer_stays_cpu() {
    let plan = MetalPlanNode::Filter {
        predicate: /* ... */,
        input: Box::new(MetalPlanNode::DataFrameScan { n_rows: 100, .. }),
        // n_rows below GroupBy threshold → not lifted; consumer is implicit Sink.
    };
    let lifting = compute_lifting_plan(&plan);
    let filter_id = /* ... */;
    // Filter alone with no GPU downstream → CPU.
    assert_eq!(lifting.get(&filter_id), Some(&NodeDecision::CpuLeave));
}
```

- [ ] **Step 2: Implement affinity smoothing for filter+groupby**

In `router/affinity.rs`, after the cost-model pass that decides each node, walk the plan:

```rust
pub fn smooth_filter_into_gpu_consumer(plan: &MetalPlanNode, decisions: &mut LiftingPlan) {
    // Post-order traverse. For each Filter node whose immediate parent (consumer)
    // is GpuLift'd, promote the Filter from CpuLeave → GpuLift.
    walk_with_parent(plan, None, &mut |node, parent| {
        if let MetalPlanNode::Filter { .. } = node {
            let id = id_of(node);
            if matches!(decisions.get(&id), Some(NodeDecision::CpuLeave)) {
                if let Some(p) = parent {
                    let parent_id = id_of(p);
                    if matches!(decisions.get(&parent_id), Some(NodeDecision::GpuLift)) {
                        decisions.set(id, NodeDecision::GpuLift);
                    }
                }
            }
        }
    });
}
```

Add a call to `smooth_filter_into_gpu_consumer` in the router's main entry point, after the cost-model pass and before returning the plan.

- [ ] **Step 3: Run + commit**

```bash
cargo test -p polars-metal-core --test test_affinity_filter_groupby -- --test-threads=1
```

```bash
git add crates/polars-metal-core/src/router/affinity.rs crates/polars-metal-core/tests/test_affinity_filter_groupby.rs
git commit -m "Router: filter promoted to GPU when consumer is GpuLift

Capability C. Affinity smoothing pass; runs after cost-model decisions.
Filter alone stays CPU (memory-bound); filter feeding GPU groupby lifts.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 40: Walker — fused filter+groupby dispatch

**Files:**
- Modify: `python/polars_metal/_walker.py`
- Modify: `crates/polars-metal-core/src/udf.rs`
- Create: `tests/python_integration/test_filter_groupby_fusion.py`

- [ ] **Step 1: Walker recognizes the fused subtree**

When both Filter and its parent GroupBy are GpuLift'd, the walker invokes a single UDF for the GroupBy node that takes the *raw* input (before filter) plus the filter predicate. The Rust side runs filter+groupby as one pipeline.

```python
# python/polars_metal/_walker.py (excerpt)
def _apply_lifting_plan(nt, plan, decisions):
    # Post-order; when we see a GroupBy that's GpuLift AND its child Filter is
    # also GpuLift, install a single UDF on the GroupBy that takes the Filter's
    # input as the "raw" input and the Filter's predicate as extra payload.
    ...
```

- [ ] **Step 2: Rust side — `execute_filter_groupby` entry**

```rust
// In crates/polars-metal-core/src/udf.rs
pub fn execute_filter_groupby(
    py_df: &PyAny,
    filter_predicate: &PyAny,
    plan_dict: &PyDict,
) -> PyResult<PyObject> {
    // 1. Materialize input columns as MetalBuffers (single or chunked view).
    // 2. Evaluate filter predicate → keep mask (M1's existing filter kernel).
    // 3. Build phase + agg phase on the filtered indices (no DataFrame materialization).
    // 4. Return result Polars DataFrame.
}
```

Crucially, step 3 reads value columns *by index* (via the filter's surviving-row indices), not by reading a compacted DataFrame. This is the "skip materialize" win.

- [ ] **Step 3: Integration test**

```python
# tests/python_integration/test_filter_groupby_fusion.py
import polars as pl
from polars.testing import assert_frame_equal

import polars_metal as pm


def test_filter_groupby_fused_subtree():
    df = pl.DataFrame({
        "k":  [(i % 4) for i in range(1_000_000)],
        "v":  [float(i) for i in range(1_000_000)],
        "shp": [(i % 100) for i in range(1_000_000)],
    })
    q = (df.lazy()
           .filter(pl.col("shp") <= 50)
           .group_by("k")
           .agg(pl.col("v").sum()))
    logs = []
    eng = pm.MetalEngine(debug=True, debug_sink=logs.append)
    metal = q.collect(engine=eng).sort("k")
    cpu = q.collect(engine="cpu").sort("k")
    assert_frame_equal(cpu, metal)
    assert any("fused_filter_groupby" in line for line in logs), logs


def test_filter_alone_stays_cpu():
    df = pl.DataFrame({"k": [0, 1, 2, 3] * 1000, "v": [1.0, 2.0, 3.0, 4.0] * 1000})
    q = df.lazy().filter(pl.col("k") <= 1)
    logs = []
    eng = pm.MetalEngine(debug=True, debug_sink=logs.append)
    _ = q.collect(engine=eng)
    # Filter alone (no GPU consumer) → CPU.
    assert not any("Filter.*GpuLift" in line for line in logs), logs
```

- [ ] **Step 4: Build + run + commit**

```bash
make wheel
pytest tests/python_integration/test_filter_groupby_fusion.py -v
```

```bash
git add python/polars_metal/_walker.py crates/polars-metal-core/src/udf.rs tests/python_integration/test_filter_groupby_fusion.py
git commit -m "Walker + UDF: fused filter+groupby dispatch

Capability C complete. Skips intermediate DataFrame materialization.
Test verifies fusion via debug log; M2's tpch_q1_modified ratio
should improve from this change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 11 — Canonical TPC-H Q1 benchmark

The headline M3 query: real Q1, Utf8 keys, inline `sum(a*b)` expressions, no pre-projection. Target ratio `< 0.7` on M2 Ultra at 10M rows.

### Task 41: Canonical Q1 fixture (Utf8 keys + raw expressions)

**Files:**
- Create: `tests/bench/_canonical_q1_fixture.py`
- Modify: `tests/bench/_lineitem_fixture.py` (or extend with Utf8 key option)

- [ ] **Step 1: Implement deterministic fixture**

```python
# tests/bench/_canonical_q1_fixture.py
"""Canonical TPC-H Q1 input fixture: lineitem with Utf8 l_returnflag
and l_linestatus, no encoding shortcuts. Numeric columns as in
_lineitem_fixture.py."""
from __future__ import annotations

import numpy as np
import polars as pl

_RETURN_FLAGS = ["A", "N", "R"]
_LINE_STATUSES = ["F", "O"]


def make_canonical_q1_fixture(n_rows: int = 10_000_000, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rf_idx = rng.integers(0, 3, size=n_rows)
    ls_idx = rng.integers(0, 2, size=n_rows)
    return pl.DataFrame({
        "l_returnflag":    [_RETURN_FLAGS[i] for i in rf_idx],
        "l_linestatus":    [_LINE_STATUSES[i] for i in ls_idx],
        "l_quantity":      rng.integers(1, 51, size=n_rows).astype(np.float64),
        "l_extendedprice": (rng.uniform(900.0, 105_000.0, size=n_rows)).astype(np.float64),
        "l_discount":      rng.uniform(0.0, 0.11, size=n_rows).astype(np.float64),
        "l_tax":           rng.uniform(0.0, 0.09, size=n_rows).astype(np.float64),
        "l_shipdate":      pl.date_range(
            pl.date(1992, 1, 1), pl.date(1998, 12, 31),
            n_rows, closed="left", eager=True
        ).to_list()[:n_rows],
    })
```

(Adjust `pl.date_range` API to match `py-1.40.1`'s signature; M2's `_lineitem_fixture.py` already has the correct pattern.)

- [ ] **Step 2: Commit**

```bash
git add tests/bench/_canonical_q1_fixture.py
git commit -m "Bench: deterministic canonical TPC-H Q1 fixture with Utf8 keys

Capability D + G's headline workload. 10M rows, seed=42, no encoding
shortcuts.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 42: Canonical Q1 bench + baseline entry

**Files:**
- Create: `tests/bench/test_tpch_canonical_q1.py`
- Modify: `tests/bench/baseline.json`

- [ ] **Step 1: Write the bench**

```python
# tests/bench/test_tpch_canonical_q1.py
"""Canonical TPC-H Q1 bench: CPU vs Metal, recorded in baseline.json."""
import json
from datetime import date
from pathlib import Path

import polars as pl
import polars_metal as pm
import pytest

from tests.bench._canonical_q1_fixture import make_canonical_q1_fixture


_THRESHOLD = date(1998, 9, 2)
_BASELINE_PATH = Path(__file__).parent / "baseline.json"


def _query(df: pl.DataFrame, engine):
    return (df.lazy()
        .filter(pl.col("l_shipdate") <= _THRESHOLD)
        .group_by("l_returnflag", "l_linestatus")
        .agg(
            pl.col("l_quantity").sum().alias("sum_qty"),
            pl.col("l_extendedprice").sum().alias("sum_base_price"),
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).sum().alias("sum_disc_price"),
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount")) * (1 + pl.col("l_tax"))).sum().alias("sum_charge"),
            pl.col("l_quantity").mean().alias("avg_qty"),
            pl.col("l_extendedprice").mean().alias("avg_price"),
            pl.col("l_discount").mean().alias("avg_disc"),
            pl.len().alias("count_order"),
        )
        .sort("l_returnflag", "l_linestatus")
        .collect(engine=engine))


@pytest.fixture(scope="module")
def df():
    return make_canonical_q1_fixture()


def test_correctness_cpu_eq_metal(df):
    cpu = _query(df, "cpu")
    metal = _query(df, pm.MetalEngine())
    from polars.testing import assert_frame_equal
    assert_frame_equal(cpu, metal)


def test_bench_cpu(benchmark, df):
    result = benchmark(lambda: _query(df, "cpu"))
    assert result.shape[0] >= 1


def test_bench_metal(benchmark, df):
    result = benchmark(lambda: _query(df, pm.MetalEngine()))
    assert result.shape[0] >= 1


def test_record_baseline_ratio(df):
    """Compute median wall-clock for both, write into baseline.json."""
    import time
    def median_time(fn, n=5):
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        times.sort()
        return times[n // 2]

    cpu_ms = median_time(lambda: _query(df, "cpu")) * 1000
    metal_ms = median_time(lambda: _query(df, pm.MetalEngine())) * 1000
    ratio = metal_ms / cpu_ms

    baseline = json.loads(_BASELINE_PATH.read_text())
    baseline["tpch_q1_canonical"] = {
        "cpu_ms": cpu_ms,
        "metal_ms": metal_ms,
        "ratio_metal_over_cpu": ratio,
        "n_rows": 10_000_000,
        "_gate": {"ratio_lt": 0.7},
    }
    _BASELINE_PATH.write_text(json.dumps(baseline, indent=2))
    assert ratio < 0.7, f"M3 perf gate: canonical Q1 ratio {ratio:.3f} not < 0.7"
```

- [ ] **Step 2: Build + run**

```bash
make wheel
pytest tests/bench/test_tpch_canonical_q1.py -v
```

Expected: all 4 tests pass on M2 Ultra. If `test_record_baseline_ratio` fails (ratio ≥ 0.7), the M3 perf gate is unmet — investigate fused-agg kernel performance, dictionary build cost, expression-emission MSL inefficiencies. Profiling via `Instruments → Metal System Trace` is the next step.

- [ ] **Step 3: Commit (only if gate passes)**

```bash
git add tests/bench/test_tpch_canonical_q1.py tests/bench/baseline.json
git commit -m "Bench: canonical TPC-H Q1 (Utf8 keys + inline expressions)

M3 headline. baseline.json::tpch_q1_canonical with _gate ratio_lt 0.7.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 12 — High-cardinality Q1 (64K and 1M groups)

Two benchmark entries; one exercises A1, the other A2.

### Task 43: High-card fixtures (synthetic customer_key)

**Files:**
- Create: `tests/bench/_highcard_q1_fixture.py`

- [ ] **Step 1: Implement**

```python
# tests/bench/_highcard_q1_fixture.py
"""Q1-shape fixture with a synthetic high-cardinality customer_key column.
The groupby is over (l_returnflag, l_linestatus, customer_key); aggregations
unchanged from canonical Q1.

n_groups parameter controls the customer_key range (4 × 2 × n_customers
distinct groups expected)."""
import numpy as np
import polars as pl

from tests.bench._canonical_q1_fixture import make_canonical_q1_fixture


def make_highcard_q1_fixture(n_rows: int = 10_000_000, n_customers: int = 64_000, seed: int = 42):
    df = make_canonical_q1_fixture(n_rows, seed)
    rng = np.random.default_rng(seed + 1)
    customer_key = rng.integers(0, n_customers, size=n_rows, dtype=np.int32)
    return df.with_columns(pl.Series("customer_key", customer_key, dtype=pl.Int32))
```

- [ ] **Step 2: Commit**

```bash
git add tests/bench/_highcard_q1_fixture.py
git commit -m "Bench: high-cardinality Q1 fixture (synthetic customer_key)

Capability A workload. n_customers parameter controls A1/A2 routing
exercise: 64K stresses A1's TGSM cap; 1M exercises A2's sort.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 44: High-card bench entries

**Files:**
- Create: `tests/bench/test_tpch_q1_highcard.py`

- [ ] **Step 1: Write the bench**

```python
# tests/bench/test_tpch_q1_highcard.py
"""Two-variant high-card bench: 64K groups (exercises A1) and 1M (exercises A2)."""
import json
import time
from pathlib import Path

import polars as pl
import polars_metal as pm
import pytest

from tests.bench._highcard_q1_fixture import make_highcard_q1_fixture


_BASELINE_PATH = Path(__file__).parent / "baseline.json"


def _q_highcard(df, engine):
    return (df.lazy()
        .filter(pl.col("l_shipdate") <= pl.date(1998, 9, 2))
        .group_by("l_returnflag", "l_linestatus", "customer_key")
        .agg(
            pl.col("l_quantity").sum(),
            pl.col("l_extendedprice").sum(),
            pl.col("l_quantity").mean(),
            pl.col("l_extendedprice").mean(),
            pl.len().alias("count_order"),
        )
        .sort("l_returnflag", "l_linestatus", "customer_key")
        .collect(engine=engine))


def _median_time(fn, n=5):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[n // 2]


@pytest.mark.parametrize("n_customers, key, gate", [
    (64_000, "tpch_q1_highcard_64k", 0.7),
    (1_000_000, "tpch_q1_highcard_1m", 1.0),
])
def test_record_baseline(n_customers, key, gate):
    df = make_highcard_q1_fixture(n_rows=10_000_000, n_customers=n_customers)
    cpu_ms = _median_time(lambda: _q_highcard(df, "cpu")) * 1000
    metal_ms = _median_time(lambda: _q_highcard(df, pm.MetalEngine())) * 1000
    ratio = metal_ms / cpu_ms

    baseline = json.loads(_BASELINE_PATH.read_text())
    baseline[key] = {
        "cpu_ms": cpu_ms, "metal_ms": metal_ms, "ratio_metal_over_cpu": ratio,
        "n_rows": 10_000_000, "n_customers": n_customers,
        "_gate": {"ratio_lt": gate},
    }
    _BASELINE_PATH.write_text(json.dumps(baseline, indent=2))
    assert ratio < gate, f"{key}: ratio {ratio:.3f} not < {gate}"
```

- [ ] **Step 2: Run + commit**

```bash
make wheel
pytest tests/bench/test_tpch_q1_highcard.py -v
```

```bash
git add tests/bench/test_tpch_q1_highcard.py tests/bench/baseline.json
git commit -m "Bench: high-card Q1 (64K + 1M groups) baseline + gates

Capability A perf evidence. 64K gate < 0.7 (A1 path); 1M gate < 1.0
(A2 path).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 13 — TPC-H Q6 (filter-heavy single-group reduction)

### Task 45: Q6 fixture + Polars IR shape investigation

**Files:**
- Create: `tests/bench/_q6_fixture.py`
- Create: `docs/q6-ir-investigation.md` (scratch notes; can be deleted before final commit)

- [ ] **Step 1: Diagnostic — print Q6's IR shape**

```bash
python -c "
import polars as pl
df = pl.DataFrame({'p': [1.0]*100, 'd': [0.1]*100, 'q': [1]*100, 's': [pl.date(1995,1,1)]*100})
q = (df.lazy()
     .filter((pl.col('s') >= pl.date(1995,1,1)) & (pl.col('s') < pl.date(1996,1,1))
             & (pl.col('d') >= 0.05) & (pl.col('d') <= 0.07)
             & (pl.col('q') < 24))
     .select((pl.col('p') * pl.col('d')).sum().alias('revenue')))
print(q.explain())
"
```

Record the output. The IR node shape (Select-with-Agg / Reduce / GroupBy-empty-keys) determines walker dispatch in Task 46.

- [ ] **Step 2: Implement fixture**

```python
# tests/bench/_q6_fixture.py
import numpy as np
import polars as pl
from datetime import date


def make_q6_fixture(n_rows: int = 10_000_000, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "l_extendedprice": rng.uniform(900, 105_000, n_rows).astype(np.float64),
        "l_discount":      rng.uniform(0.0, 0.11, n_rows).astype(np.float64),
        "l_quantity":      rng.integers(1, 51, n_rows).astype(np.int32),
        "l_shipdate":      pl.date_range(
            pl.date(1992, 1, 1), pl.date(1998, 12, 31),
            n_rows, closed="left", eager=True
        ).to_list()[:n_rows],
    })
```

- [ ] **Step 3: Commit**

```bash
git add tests/bench/_q6_fixture.py
git commit -m "Bench: TPC-H Q6 fixture (4-column lineitem subset)

Capability C workload. Tests filter→reduction routing.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 46: Walker — single-group aggregation as `MetalPlanNode::GroupBy { keys: [], aggs: [...] }`

**Files:**
- Modify: `python/polars_metal/_walker.py`
- Modify: `crates/polars-metal-kernels/src/groupby.rs` (handle `keys.is_empty()`)

- [ ] **Step 1: Recognize Q6's IR shape**

Based on Task 45 Step 1's output. Likely either:

- `Select(Filter, [Agg(BinaryExpr(...))])` — Polars represents `filter().select(sum(a*b))` as a Select with an Agg expression.
- A distinct `Reduce` node — if Polars folded the aggregation into its own node type.
- `GroupBy { keys: [], aggs: [...] }` — if the optimizer rewrote into an empty-key groupby.

Pattern-match the discovered shape in the walker and emit:

```python
MetalPlanNode::GroupBy {
    keys: [],
    aggs: [AggSpec::Expression { expr: ..., op: Sum, ... }],
    input: <child>,
}
```

- [ ] **Step 2: Build phase handles empty keys**

In `crates/polars-metal-kernels/src/groupby.rs`:

```rust
fn execute_groupby(..., keys: &[KeyColumn], aggs: &[AggSpec], ...) -> Result<...> {
    if keys.is_empty() {
        // Single-group case: row_to_group = [0; n_rows], n_groups = 1.
        // Skip the build phase entirely; go straight to aggregation.
        let row_to_group = MetalBuffer::filled(device, 0u32, n_rows as usize);
        execute_groupby_aggregation_fused(..., &row_to_group, 1, ...)
    } else {
        // M3 dual-mode build — full body specified in Phase 6 Task 30 Step 4
        // (estimate_cardinality → decide_build_mode → A1 with overflow→A2).
        execute_groupby_dual_mode(device, &encoded_keys, n_rows, aggs)?
    }
}
```

- [ ] **Step 3: Q6 bench entry**

```python
# tests/bench/test_tpch_q6.py
import json, time
from datetime import date
from pathlib import Path

import polars as pl
import polars_metal as pm

from tests.bench._q6_fixture import make_q6_fixture


_BASELINE_PATH = Path(__file__).parent / "baseline.json"


def _q6(df, engine):
    return (df.lazy()
        .filter(
            (pl.col("l_shipdate") >= date(1994, 1, 1))
            & (pl.col("l_shipdate") < date(1995, 1, 1))
            & (pl.col("l_discount") >= 0.05) & (pl.col("l_discount") <= 0.07)
            & (pl.col("l_quantity") < 24)
        )
        .select((pl.col("l_extendedprice") * pl.col("l_discount")).sum().alias("revenue"))
        .collect(engine=engine))


def test_record_baseline_q6():
    df = make_q6_fixture()
    def t(eng):
        return min(
            time.perf_counter() - t0 if (t0 := time.perf_counter()) else 0
            for _ in range(5)
            for _ in [_q6(df, eng)]
        )
    # ^ replace with the standard median_time helper from Task 42
    from tests.bench.test_tpch_canonical_q1 import _median_time
    cpu_ms = _median_time(lambda: _q6(df, "cpu")) * 1000
    metal_ms = _median_time(lambda: _q6(df, pm.MetalEngine())) * 1000
    ratio = metal_ms / cpu_ms
    baseline = json.loads(_BASELINE_PATH.read_text())
    baseline["tpch_q6"] = {
        "cpu_ms": cpu_ms, "metal_ms": metal_ms, "ratio_metal_over_cpu": ratio,
        "n_rows": 10_000_000, "_gate": {"ratio_lt": 0.7},
    }
    _BASELINE_PATH.write_text(json.dumps(baseline, indent=2))
    assert ratio < 0.7
```

- [ ] **Step 4: Run + commit**

```bash
make wheel
pytest tests/bench/test_tpch_q6.py -v
```

```bash
git add python/polars_metal/_walker.py crates/polars-metal-kernels/src/groupby.rs tests/bench/test_tpch_q6.py tests/bench/baseline.json
git commit -m "Walker + Kernel + Bench: TPC-H Q6 routes filter→reduction GPU

Capability C end-to-end. Empty-key groupby = single-group reduction.
baseline.json::tpch_q6 with _gate ratio_lt 0.7.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 14 — 100M-row Q1 (scale validation)

### Task 47: 100M Q1 bench

**Files:**
- Create: `tests/bench/test_tpch_q1_100m.py`

- [ ] **Step 1: Write the bench**

```python
# tests/bench/test_tpch_q1_100m.py
"""100M-row scale of M2's modified Q1. Verifies overhead-vs-compute
crossover widens Metal's lead. M2 Ultra only — too large for 16/8 GB
machines (~9 GB DRAM)."""
import json, time
from pathlib import Path

import polars as pl
import polars_metal as pm
import pytest

from tests.bench._lineitem_fixture import make_lineitem_fixture
from tests.bench.test_tpch_canonical_q1 import _median_time


_BASELINE_PATH = Path(__file__).parent / "baseline.json"


@pytest.mark.slow
def test_100m_q1_ratio():
    if pl.thread_pool_size() < 8:
        pytest.skip("100M Q1 requires M2 Ultra–class hardware")
    df = make_lineitem_fixture(n_rows=100_000_000)
    # Same modified Q1 query as M2 (int-encoded keys).
    from tests.bench.test_tpch_q1_modified import _query as q_m2  # M2's helper
    cpu_ms = _median_time(lambda: q_m2(df, "cpu"), n=3) * 1000
    metal_ms = _median_time(lambda: q_m2(df, pm.MetalEngine()), n=3) * 1000
    ratio = metal_ms / cpu_ms

    baseline = json.loads(_BASELINE_PATH.read_text())
    baseline["tpch_q1_100m"] = {
        "cpu_ms": cpu_ms, "metal_ms": metal_ms, "ratio_metal_over_cpu": ratio,
        "n_rows": 100_000_000, "_gate": {"ratio_lt": 0.5},
        "_notes": "M2 Ultra only — ~9 GB DRAM. Other machines skip via thread_pool_size guard.",
    }
    _BASELINE_PATH.write_text(json.dumps(baseline, indent=2))
    assert ratio < 0.5
```

- [ ] **Step 2: Run on M2 Ultra**

```bash
make wheel
pytest tests/bench/test_tpch_q1_100m.py -v
```

Expected: ratio < 0.5. If it fails, fixed-overhead is still bigger than expected — profile the per-query setup.

- [ ] **Step 3: Commit**

```bash
git add tests/bench/test_tpch_q1_100m.py tests/bench/baseline.json
git commit -m "Bench: 100M-row Q1 — scale validation, M2 Ultra only

Capability H — _gate ratio_lt 0.5. Skipped on smaller machines via
thread_pool_size guard.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 15 — Conformance: wire upstream Polars paths

### Task 48: Verify Polars py-1.40.1 upstream path names

**Files:** none (investigation).

- [ ] **Step 1: List candidate paths**

```bash
ls references/polars/py-polars/tests/unit/datatypes/test_string.py 2>&1 || echo MISSING
ls references/polars/py-polars/tests/unit/operations/test_filter.py 2>&1 || echo MISSING
ls references/polars/py-polars/tests/unit/operations/test_group_by.py 2>&1 || echo MISSING
ls references/polars/py-polars/tests/unit/operations/aggregation/ 2>&1 || echo MISSING
find references/polars/py-polars/tests -name "*chunk*" 2>&1
```

Record actual filenames; some may differ from spec § Conformance ("verified at implementation time").

- [ ] **Step 2: Record findings** in `tests/conformance/SUITE_PATHS_M3.md`:

```markdown
# Polars py-1.40.1 paths wired for M3 conformance

Discovered via Task 48 Step 1.

- `tests/unit/datatypes/test_string.py` (D)
- `tests/unit/operations/test_filter.py` (C)
- `tests/unit/operations/test_group_by.py` (D Utf8 cases)
- `tests/unit/operations/aggregation/test_agg.py` (G expression cases)
- `tests/unit/operations/test_chunked.py` if present (E)
```

### Task 49: Wire paths into the conformance harness

**Files:**
- Modify: `tests/conformance/test_polars_suite.py`

- [ ] **Step 1: Add paths to `SUITE_PATHS`**

In M2's harness, find the constant `SUITE_PATHS = [...]` and append the M3 paths from Task 48. Format follows M2's existing pattern (relative to `references/polars/py-polars/`).

- [ ] **Step 2: Establish per-path baselines under `engine="cpu"`**

```bash
pytest tests/conformance/test_polars_suite.py --collect-only > /tmp/m3_collect.txt
pytest tests/conformance/test_polars_suite.py --polars-engine=cpu --tb=no -q > /tmp/m3_baseline_cpu.txt
```

Record any tests that fail CPU baseline; they're not M3's bugs but must be in the skip/xfail registry to prevent false alarms.

- [ ] **Step 3: Run Metal engine + compare**

```bash
pytest tests/conformance/test_polars_suite.py --polars-engine=metal --tb=no -q > /tmp/m3_metal.txt
diff /tmp/m3_baseline_cpu.txt /tmp/m3_metal.txt
```

Expected diff: no new failures (Metal-side count of failures equals or beats CPU baseline count). If new failures appear, debug per-test — these are M3 bugs.

- [ ] **Step 4: Commit**

```bash
git add tests/conformance/test_polars_suite.py tests/conformance/SUITE_PATHS_M3.md
git commit -m "Conformance: wire M3 upstream paths (string, filter, group_by, agg, chunked)

Capabilities C/D/E/G. SUITE_PATHS extended; baseline captured; per-path
xfails recorded for pre-existing CPU failures.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 50: Run `make gate` end-to-end

**Files:** none (verification).

- [ ] **Step 1: Run the full gate**

```bash
make gate
```

Expected: all phases pass. Wall-clock ~8-15 min on M2 Ultra (M3 adds more tests and a larger conformance set).

- [ ] **Step 2: If anything fails, fix root cause**

Common failure modes:
- Metal command-queue contention from missing `--test-threads=1` somewhere → check Cargo test config.
- MSL compile failures on adversarial inputs → add the case to the emitter test suite and fix.
- Conformance regression → identify the failing test, decide if it's a bug or an out-of-scope case (add to xfail) — never silently broaden.

- [ ] **Step 3: Run `make bench` and verify all gates pass**

```bash
pytest tests/bench/ -v
```

Expected: all 5 new entries (tpch_q1_canonical, tpch_q1_highcard_64k, tpch_q1_highcard_1m, tpch_q6, tpch_q1_100m) meet their `_gate` thresholds. M2's `tpch_q1_modified` does not regress.

- [ ] **Step 4: Commit baseline.json if anything was updated**

```bash
git add tests/bench/baseline.json
git status  # verify no other dirty state
git diff --cached tests/bench/baseline.json
git commit -m "Bench: M3 baseline snapshot at M3 ship

All 5 new entries meet _gate thresholds. M2 entries unchanged
(or within ±5% noise).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 16 — Documentation + retrospective

### Task 51: `docs/architecture.md` — M3 additions

**Files:**
- Modify: `docs/architecture.md`

- [ ] **Step 1: Add an "M3: real-workload quality" section**

Below M2's section, write a section covering: dual-mode build phase (A1/A2 + cardinality routing), MSL template engine (B + cache + warmup), filter+groupby fusion (C), string-key paths (D Phase 1 dict + Phase 2 hash kernel), multi-chunk bridge (E), expression unfolding (G), perf gate change (H).

Each subsection ~150-300 words, with a small ASCII diagram for non-obvious dataflow. Cross-reference the relevant kernel files in `crates/` and `shaders/`.

- [ ] **Step 2: Commit**

```bash
git add docs/architecture.md
git commit -m "Docs: architecture.md extended with M3 sections

A/B/C/D/E/G/H captured. References point at landed code.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 52: `docs/kernel-authoring.md` — M3 idioms

**Files:**
- Modify: `docs/kernel-authoring.md`

- [ ] **Step 1: Add idioms**

- **Partitioned-hash idiom (TGSM hash table).** Per-threadgroup tables avoid global atomics; spin-wait on slot-claim sentinel is safe across SIMD groups when hash spreads keys; overflow detection via global flag → CPU re-dispatch.
- **Restricted-scope GPU sort.** Per-lane radix sort can be kept narrow when sorting an internal key type (u128 here); generalization to user-visible Sort op deferred to M4.
- **MSL template-engine pattern.** Rust code generation per query signature; signature hash → cache; pre-compilation at module import for common shapes.
- **FNV-1a string hash kernel.** Per-row byte iteration; cheap but collision-prone; downstream must resolve collisions on slot match.
- **Chunked dispatch.** Kernels accept `&[MetalBuffer<T>]` slices; iterate chunks per pass; accumulators are single output buffers.

- [ ] **Step 2: Commit**

```bash
git add docs/kernel-authoring.md
git commit -m "Docs: kernel-authoring.md adds M3 idioms

Partitioned hash, restricted-scope sort, MSL template engine, FNV-1a
strings, chunked dispatch.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 53: `docs/open-questions.md` — M3 resolved + new

**Files:**
- Modify: `docs/open-questions.md`

- [ ] **Step 1: Strike resolved M2 items**

For each M2-listed open question that M3 resolves, wrap the heading in `~~strikethrough~~` and add a one-line note pointing to the resolution.

Items M3 resolves: "Apple Silicon Metal 64-bit atomics gap" (partial — M3 doesn't change this, but documents the trade-off more sharply); "GroupBy build phase on CPU" (resolved by A1/A2); "Composite key 128-bit limit" (resolved by F + Utf8 via dict).

- [ ] **Step 2: Add M3-surfaced open questions**

- **MSL template-engine compile latency.** Per-signature compile cost; warmup helps but doesn't eliminate.
- **A1 vs A2 cardinality threshold tuning.** 32K hard-coded; revisit with more workloads.
- **Hash collision policy for string keys above 32K cardinality.** Dict path is correct but slow; GPU hash kernel needs slot-match string compare.
- **Multi-chunk threshold (256 MB).** Heuristic; tune with real workloads.
- **Filter+groupby fusion benefit on unified memory.** Spec § C calls out the win as "skip materialize"; measurable but quantified yet to be re-measured at M3 close.
- **Expression unfolding scope.** M3 supports binary arithmetic on column/literal; conditionals, function calls, ternaries deferred.

- [ ] **Step 3: Commit**

```bash
git add docs/open-questions.md
git commit -m "Docs: open-questions M3 update — resolved + newly surfaced

Strike-through M2 items A/D-128bit-limit resolve in M3. Six new items
land for M4+ consideration.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 54: M3 retrospective in the design spec

**Files:**
- Modify: `docs/superpowers/specs/2026-05-22-m3-design.md`

- [ ] **Step 1: Write the retrospective**

Append to the "M3 retrospective (to be written)" section. Follow M2's retrospective structure:

- **Outcome.** Per-exit-criterion pass/fail (criteria 1–30).
- **Surprises during execution.** Plan-vs-reality deltas, especially around algorithmic decisions (build-phase routing, MSL emission, collision handling).
- **Resolved in PR follow-up commits.** Items that landed between in-phase work and retro.
- **Still to revisit at M4.** Items deferred but tracked; pointers into `docs/open-questions.md`.
- **Portability gate results.** M2 Ultra ✓; M2 16 GB ✓ or ⏳; M1 8 GB ✓ or ⏳ (100M Q1 excluded).

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-05-22-m3-design.md
git commit -m "Docs: M3 retrospective — outcome, surprises, hand-off

Per spec § "M3 retrospective". Closes the spec document.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 55: Portability gate — manual checks on smaller machines

**Files:**
- Modify: `docs/superpowers/specs/2026-05-22-m3-design.md` (record results)

- [ ] **Step 1: M2 16 GB**

```bash
# On the M2 16 GB machine:
git checkout m3-realworkload
make gate
pytest tests/bench/ -v --ignore=tests/bench/test_tpch_q1_100m.py
```

Record pass/fail and git SHA in the retrospective's "Portability gate results" section.

- [ ] **Step 2: M1 8 GB**

Same steps as Step 1, on the M1 8 GB machine.

- [ ] **Step 3: Commit final retrospective update**

```bash
git add docs/superpowers/specs/2026-05-22-m3-design.md
git commit -m "Docs: portability gate results recorded in M3 retro

M2 16 GB / M1 8 GB results captured. 100M Q1 excluded from these
machines per design § Layer 5.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 56: Open PR to main

**Files:** none (PR creation).

- [ ] **Step 1: Push the branch**

```bash
git push -u origin m3-realworkload
```

- [ ] **Step 2: Create the PR**

```bash
gh pr create --title "M3: real-workload quality — TPC-H slice + dual-mode GPU build + strings" --body "$(cat <<'EOF'
## Summary

- Ships canonical TPC-H Q1 (Utf8 keys + inline `sum(a*b)`), high-card Q1 (64K + 1M groups), Q6, and 100M Q1 — all faster than CPU Polars on M2 Ultra.
- Lands 8 capabilities: dual-mode GPU build (A1 partitioned hash + A2 sort-segment-reduce), multi-agg kernel fusion (B), filter→GPU when downstream is GPU (C), string keys via dictionary + hash kernel (D), multi-chunk Series (E), smaller-integer keys (F), binary-expression unfolding (G), per-entry perf gates (H).
- Spec: `docs/superpowers/specs/2026-05-22-m3-design.md`.

## Test plan

- [ ] `make gate` green on M2 Ultra
- [ ] `pytest tests/bench/` — all 5 new entries meet `_gate` thresholds; M2's `tpch_q1_modified` no regression
- [ ] Portability gate on M2 16 GB (excluding 100M Q1)
- [ ] Portability gate on M1 8 GB (excluding 100M Q1)
- [ ] Conformance no new failures vs CPU baseline

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Capture PR URL for the spec**

Add the PR URL to the retrospective's first paragraph: "M3 shipped via PR #N (link)."

```bash
git add docs/superpowers/specs/2026-05-22-m3-design.md
git commit -m "Docs: link PR #N to M3 retrospective

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
git push
```

---

## Post-merge checklist

After PR merges to main:

- [ ] Update `docs/superpowers/specs/2026-05-19-master-plan-design.md` — strike M3, advance roadmap pointer to M4 (radix sort + joins).
- [ ] Refresh `docs/architecture.md` cross-references if anything moved.
- [ ] Tag the release: `git tag m3-v0.3.0 && git push --tags`.

---

**End of M3 plan.**

This plan is intentionally large because M3 is wide. Recommended execution mode: **subagent-driven** (one subagent per task, review between tasks). The phases form natural break points; the user can pause execution after any phase boundary without leaving the engine in a broken state — each phase produces a complete, tested capability.

If a task hits unexpected friction (e.g. SIMD-lockstep deadlock in Phase 4's TGSM spin-wait, or a Polars IR shape that doesn't match what the walker expects in Q6), **stop and reassess** rather than ploughing through. Per CLAUDE.md: "Don't introduce a new dependency without a written justification." The same applies to algorithmic pivots — if the partitioned-hash design doesn't work on this hardware, document why and route through A2 universally; don't quietly weaken the spec.






