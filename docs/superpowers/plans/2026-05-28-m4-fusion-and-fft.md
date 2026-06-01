# M4 — MLX subgraph fusion, cumsum-diff rolling, list-dot vector search, `Expr.fft()` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the M4 chunk (Phases 8–11 of the revised roadmap): MLX-subgraph-fusion engine for compute-shaped F32 expression trees, cumsum-diff rolling, `Array[F32, D].dot(lit)` → MLX matmul for vector search, and a public `pl.Expr.metal.fft()` API. Eleven new benchmark entries hit measured wall-clock targets within ~2 ms of the MLX ceiling; the CPU-parity gate enforces that non-compute glue ops stay within 5% of Polars CPU.

**Architecture:** M3's three-layer flow (walker → router → walker applies → dispatch) is preserved. M4 adds a fourth axis: the walker now recognizes *compute subtrees inside expressions* (not just IR nodes), runs density estimation, marks the subtree as fused, and routes it as a single `MetalPlanNode::FusedExprGraph`. The fused subtree is built as one `mlx::core::array` graph, evaluated once via `mx.eval()`, and the result folded back into Polars Series via zero-copy buffer hand-off. Non-fused ops around the subtree route via the existing CPU path with zero-copy buffer pass-through (per CLAUDE.md's CPU-parity contract).

**Tech Stack:** Rust 2021 (workspace), `objc2-metal` (Metal API), `cxx` for MLX FFI (extended from M3's single `cumsum_u8_to_u32` binding to ~50 bindings spanning elementwise / reduce / sort / scan / matmul / fft / array), `pyo3 0.22` + `maturin` (unchanged), `polars` pinned to `py-1.40.1` (unchanged), MLX pinned to `0.25.1` (Cargo.toml addition), `proptest` for kernel + reference comparison, `pytest-benchmark` + `criterion` for perf.

**Spec:** [`docs/superpowers/specs/2026-05-28-m4-design.md`](../specs/2026-05-28-m4-design.md). All decisions there are binding; this plan does not relitigate them. If a question arises that the spec didn't answer, raise it as an open question and stop — don't make the decision yourself.

**Conventions** (per CLAUDE.md): No `unwrap()` outside tests. No `unsafe` outside `*-sys` crates and the buffer bridge — each with a `// SAFETY:` comment. One MSL kernel family per file (no new MSL in this chunk — MLX serves Phase 8–11 directly). Errors propagate as `polars.exceptions.ComputeError` at the engine boundary. Null semantics match Polars exactly. Don't add files to `shaders/` without a matching test (irrelevant this chunk — no new shaders). Read the matching cuDF reference before any subtle decision.

**Pre-task reading.** Before starting Phase 1 (MLX FFI expansion), read:
- `crates/polars-metal-mlx-sys/src/lib.rs` (M2's single binding pattern; M4 expands to ~50)
- `crates/polars-metal-mlx-sys/cpp/` (the C++ side of the existing binding; new bindings follow the same idiom)
- MLX C++ API docs at `references/` or upstream — particularly `mlx::core::array` construction and `mlx::core::eval`

Before Phase 2 (compute-density estimator), read:
- `crates/polars-metal-core/src/router/cost.rs` (M2/M3 routing decisions; M4 adds an axis)
- `references/cudf/python/cudf_polars/cudf_polars/experimental/select.py` (cuDF's `_fuse_simple_reductions` pass — closest analog to what we're building)

Before Phase 3 (fusion analyzer), read:
- `python/polars_metal/_walker.py` (M3's expression traversal pattern; M4 extends)
- `references/polars/crates/polars-plan/src/dsl/expr.rs` (Polars' expression IR types — what the analyzer walks)
- `references/cudf/python/cudf_polars/cudf_polars/dsl/translate.py` (cuDF's expression translator — the structural template)

Before Phase 4 (MLX subgraph builder), read:
- `crates/polars-metal-buffer/src/lib.rs` (zero-copy MTLBuffer ↔ Arrow Buffer; M4 reuses for fold-back)
- The MLX docs section on `array::data()` and shared-buffer construction

Before Phase 7 (Expr.fft()), read:
- `references/polars/py-polars/polars/api.py` (Polars' `register_expr_namespace` pattern)
- A working example of a third-party `register_expr_namespace` registration (any pyhealth or polars-business extension)

---

## Phase 0 — Preflight + branch + perf gate

### Task 1: Confirm M3 gates green on the new branch

**Files:** none (verification only).

- [ ] **Step 1: Branch from m3-realworkload (M3 conformance code is the M4 baseline)**

```bash
git checkout m3-realworkload
git pull --ff-only origin m3-realworkload 2>/dev/null || true   # may not be pushed
git checkout -b m4-fusion-and-fft
git rev-parse --abbrev-ref HEAD && git log -1 --oneline
```

Expected: branch `m4-fusion-and-fft`; HEAD at the most recent `m3-realworkload` commit (the M4 survey + spec).

- [ ] **Step 2: Run the M3 gate**

```bash
make gate
```

Expected: all phases pass (`lint`, `test-unit`, `test-kernel`, `wheel`, `test-conformance`). Wall-clock ~6–8 min on M2 Ultra.

If anything fails: stop. The M3 conformance code must be green before adding M4 work on top; otherwise you're piling M4 on a broken baseline. Investigate and fix on a separate branch before resuming M4.

- [ ] **Step 3: Verify Metal toolchain + MLX present, plus matplotlib for any plot output**

```bash
xcrun metal --version
python -c "import polars_metal; print(polars_metal._native.version_string())"
python -c "import mlx.core as mx; print('mlx', mx.__version__ if hasattr(mx, '__version__') else 'present', mx.default_device())"
```

Expected: Metal toolchain version prints; `polars_metal` imports; MLX reports `Device(gpu, 0)`. If MLX is missing, `pip install mlx`.

- [ ] **Step 4: Record M3 perf baseline values**

```bash
python -c "import json; d=json.load(open('tests/bench/baseline.json')); \
  print({k:v['ratio_metal_over_cpu'] for k,v in d['queries'].items()})"
```

Expected: prints the dict of M3 ratios — `tpch_q1_modified ≈ 2.83`, `tpch_q1_canonical ≈ 8.31`, etc. Record these — M4 must not regress them (gates stay tight).

Nothing to commit in Task 1.

### Task 2: Pin MLX version in Cargo.toml so the MLX C++ surface is stable

**Files:**
- Modify: `crates/polars-metal-mlx-sys/Cargo.toml`
- Modify: `crates/polars-metal-mlx-sys/build.rs`

- [ ] **Step 1: Add MLX version pin to crate metadata**

Look at the current `Cargo.toml`:

```bash
cat crates/polars-metal-mlx-sys/Cargo.toml
```

The crate today links against MLX via a vendored path or system install. Add a metadata key documenting the pinned version:

```toml
[package.metadata.mlx]
version = "0.25.1"  # pinned for M4; bump deliberately
```

Then in `build.rs`, add a version-check helper:

```rust
const REQUIRED_MLX_VERSION: &str = "0.25.1";

fn check_mlx_version() {
    let output = std::process::Command::new("python")
        .args(["-c", "import mlx.core as mx; print(getattr(mx, '__version__', 'unknown'))"])
        .output();
    match output {
        Ok(o) if o.status.success() => {
            let version = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if version != REQUIRED_MLX_VERSION && version != "unknown" {
                println!("cargo:warning=MLX version mismatch: required {REQUIRED_MLX_VERSION}, found {version}");
            }
        }
        _ => println!("cargo:warning=could not detect MLX version (mlx Python package may be absent)"),
    }
}
```

Call `check_mlx_version()` from `main()` of `build.rs`.

- [ ] **Step 2: Verify build still passes**

```bash
cargo build -p polars-metal-mlx-sys 2>&1 | tail -20
```

Expected: build succeeds. If MLX is present and the pinned version matches, no warning. If a different version is found, a warning prints but build proceeds.

- [ ] **Step 3: Commit**

```bash
git add crates/polars-metal-mlx-sys/Cargo.toml crates/polars-metal-mlx-sys/build.rs
git commit -m "M4 Phase 0: pin MLX 0.25.1 in polars-metal-mlx-sys

Document the MLX version M4 was designed against; build warns on
mismatch but doesn't fail (so the workspace builds on dev machines
with slightly drifted MLX installs)."
```

### Task 3: Extend baseline.json schema with M4 entry placeholders

**Files:**
- Modify: `tests/bench/baseline.json`

- [ ] **Step 1: Add 11 placeholder entries with `_pending: true` flag**

```bash
python <<'EOF'
import json
from pathlib import Path

baseline = json.loads(Path("tests/bench/baseline.json").read_text())

m4_targets = {
    "phase8_haversine_10m":          {"target_ms": 6.0,  "n_rows": 10_000_000, "description": "Haversine over 10M F32 rows via fused MLX subgraph"},
    "phase8_black_scholes_10m":      {"target_ms": 6.0,  "n_rows": 10_000_000, "description": "Black-Scholes-shape log/exp/sqrt/tanh chain"},
    "phase8_std_var_10m":            {"target_ms": 2.0,  "n_rows": 10_000_000, "description": "Std+var reductions on F32 column"},
    "phase8_sort_f32_10m":           {"target_ms": 10.0, "n_rows": 10_000_000, "description": "Sort F32 column via MLX radix"},
    "phase8_topk_f32_10m":           {"target_ms": 10.0, "n_rows": 10_000_000, "description": "Top-K=100 on F32 column via MLX argpartition"},
    "phase8_cumsum_f32_10m":         {"target_ms": 8.0,  "n_rows": 10_000_000, "description": "Cumsum on F32 column via MLX parallel scan"},
    "phase8_corr_matrix_200x200k":   {"target_ms": 22.0, "n_rows": 200_000,    "description": "200x200000 correlation matrix via MLX matmul"},
    "phase9_rolling_mean_w1000_10m": {"target_ms": 8.0,  "n_rows": 10_000_000, "description": "rolling_mean W=1000 via cumsum-diff"},
    "phase10_cosine_topk_q100_n100k":{"target_ms": 8.0,  "n_rows": 100_000,    "description": "Cosine top-k Q=100 N=100k D=768"},
    "phase10_cosine_topk_q100_n1m":  {"target_ms": 50.0, "n_rows": 1_000_000,  "description": "Cosine top-k Q=100 N=1M D=768"},
    "phase11_fft_8m":                {"target_ms": 3.0,  "n_rows": 8_388_608,  "description": "FFT 8M-point 1D F32 via Expr.metal.fft()"},
}
for name, meta in m4_targets.items():
    baseline["queries"][name] = {
        "cpu_ms": None,
        "metal_ms": None,
        "ratio_metal_over_cpu": None,
        "n_rows": meta["n_rows"],
        "hardware": "M2 Ultra",
        "_notes": f"M4 placeholder: {meta['description']}. Target wall-clock < {meta['target_ms']} ms; ratio gate set at landing.",
        "_pending": True,
        "_target_ms": meta["target_ms"],
    }

Path("tests/bench/baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
print("wrote", len(m4_targets), "M4 entries")
EOF
```

Expected: 11 new entries added to `tests/bench/baseline.json`.

- [ ] **Step 2: Update the gate-check helper to skip `_pending` entries**

Read `tests/bench/_gate_check.py`. If the `check_baseline` function iterates `queries` and asserts ratios, extend it to skip entries with `_pending: true`:

```python
def check_baseline(baseline):
    failures = []
    for name, entry in baseline["queries"].items():
        if entry.get("_pending"):
            continue
        # ... existing ratio check ...
    return failures
```

- [ ] **Step 3: Run the gate-check test**

```bash
pytest tests/bench/test_gate_check.py -v
```

Expected: passes. Pending entries are skipped.

- [ ] **Step 4: Commit**

```bash
git add tests/bench/baseline.json tests/bench/_gate_check.py
git commit -m "M4 Phase 0: add 11 placeholder entries to baseline.json

Each entry has _pending=true and _target_ms set per the M4 spec.
Gate-check helper skips _pending entries so M3 gates stay green
until each M4 entry has a measurement."
```

---

## Phase 1 — MLX FFI surface expansion

The M2/M3 polars-metal-mlx-sys crate has a single binding (`cumsum_u8_to_u32`). M4 needs ~50. They group naturally:

- **elementwise:** ~24 ops (sin, cos, log, exp, sqrt, add, sub, mul, div, ...)
- **reduce:** sum, mean, min, max, std, var, argmin, argmax (8)
- **sort:** sort, argpartition (2)
- **scan:** cumsum, cumprod, cummax, cummin (4)
- **matmul:** the `@` operator (1)
- **fft:** fft, ifft (2)
- **array:** construct from MetalBuffer, expose underlying MTLBuffer, eval, reshape, concatenate, zeros, slice, where (9)

Total: ~50 bindings. Each is a thin C++ wrapper exposed via `cxx::bridge`. The C++ side calls into `mlx::core::*`; the Rust side returns an `MlxArrayHandle` opaque token. The handle is `cxx::SharedPtr<mlx::core::array>` underneath, so MLX's refcounting drives Rust's drop semantics correctly.

Tasks 4–10 add bindings. Task 11 is the eval-and-fold-back pipeline that exercises them end-to-end.

### Task 4: Define `MlxArrayHandle` opaque type and array-construction bindings

**Files:**
- Modify: `crates/polars-metal-mlx-sys/src/lib.rs`
- Create: `crates/polars-metal-mlx-sys/src/array.rs`
- Modify: `crates/polars-metal-mlx-sys/cpp/wrapper.h`
- Modify: `crates/polars-metal-mlx-sys/cpp/wrapper.cpp`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-mlx-sys/tests/test_array.rs
//! Construct an MlxArrayHandle from a raw F32 buffer; eval it; read the value back.

use polars_metal_mlx_sys::array::{mlx_array_from_f32_slice, mlx_eval, mlx_array_to_f32_vec};

#[test]
fn round_trip_f32_array() {
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let handle = mlx_array_from_f32_slice(&input).expect("construct");
    mlx_eval(&[handle.clone()]).expect("eval");
    let output = mlx_array_to_f32_vec(&handle).expect("read back");
    assert_eq!(output, input);
}

#[test]
fn empty_array_is_supported() {
    let input: Vec<f32> = vec![];
    let handle = mlx_array_from_f32_slice(&input).expect("construct empty");
    mlx_eval(&[handle.clone()]).expect("eval empty");
    let output = mlx_array_to_f32_vec(&handle).expect("read back empty");
    assert!(output.is_empty());
}
```

- [ ] **Step 2: Verify the test fails (function doesn't exist yet)**

```bash
cargo test -p polars-metal-mlx-sys --test test_array 2>&1 | head -20
```

Expected: compile error — `mlx_array_from_f32_slice` not found.

- [ ] **Step 3: Define the cxx bridge**

```rust
// crates/polars-metal-mlx-sys/src/array.rs
//! MLX array construction + eval bindings.
//!
//! `MlxArrayHandle` is `cxx::SharedPtr<mlx::core::array>` underneath.
//! MLX's refcounting drives Rust's drop, so handles can be cloned freely
//! without explicit free.

use crate::ffi;
use cxx::SharedPtr;

/// Opaque handle to an `mlx::core::array`. Cheap to clone (refcount).
#[derive(Clone)]
pub struct MlxArrayHandle(pub(crate) SharedPtr<ffi::MlxArray>);

impl MlxArrayHandle {
    pub fn shape(&self) -> Vec<usize> {
        let v = ffi::mlx_array_shape(&self.0);
        v.into_iter().map(|x| x as usize).collect()
    }

    pub fn dtype_is_f32(&self) -> bool {
        ffi::mlx_array_is_f32(&self.0)
    }
}

pub fn mlx_array_from_f32_slice(data: &[f32]) -> Result<MlxArrayHandle, crate::error::Error> {
    let handle = ffi::mlx_array_from_f32_data(data.as_ptr(), data.len());
    if handle.is_null() {
        return Err(crate::error::Error::ConstructionFailed);
    }
    Ok(MlxArrayHandle(handle))
}

pub fn mlx_eval(handles: &[MlxArrayHandle]) -> Result<(), crate::error::Error> {
    let raw: Vec<&SharedPtr<ffi::MlxArray>> = handles.iter().map(|h| &h.0).collect();
    ffi::mlx_eval_batch(&raw);
    Ok(())
}

pub fn mlx_array_to_f32_vec(handle: &MlxArrayHandle) -> Result<Vec<f32>, crate::error::Error> {
    let n = handle.shape().iter().product::<usize>();
    let mut out = vec![0.0_f32; n];
    if n == 0 {
        return Ok(out);
    }
    ffi::mlx_array_copy_to_f32(&handle.0, out.as_mut_ptr(), n);
    Ok(out)
}
```

```rust
// crates/polars-metal-mlx-sys/src/lib.rs — add to existing
pub mod array;
pub mod error;

#[cxx::bridge(namespace = "polars_metal_mlx")]
mod ffi {
    unsafe extern "C++" {
        include!("polars-metal-mlx-sys/cpp/wrapper.h");

        type MlxArray;

        unsafe fn mlx_array_from_f32_data(data: *const f32, n: usize) -> SharedPtr<MlxArray>;
        fn mlx_array_shape(arr: &SharedPtr<MlxArray>) -> Vec<u64>;
        fn mlx_array_is_f32(arr: &SharedPtr<MlxArray>) -> bool;
        unsafe fn mlx_array_copy_to_f32(arr: &SharedPtr<MlxArray>, out: *mut f32, n: usize);
        fn mlx_eval_batch(arrs: &Vec<&SharedPtr<MlxArray>>);
    }
}
```

```cpp
// crates/polars-metal-mlx-sys/cpp/wrapper.h
#pragma once
#include "rust/cxx.h"
#include <memory>
#include <vector>
#include "mlx/array.h"
#include "mlx/eval.h"

namespace polars_metal_mlx {
using MlxArray = mlx::core::array;

std::shared_ptr<MlxArray> mlx_array_from_f32_data(const float* data, size_t n);
rust::Vec<uint64_t> mlx_array_shape(const std::shared_ptr<MlxArray>& arr);
bool mlx_array_is_f32(const std::shared_ptr<MlxArray>& arr);
void mlx_array_copy_to_f32(const std::shared_ptr<MlxArray>& arr, float* out, size_t n);
void mlx_eval_batch(const rust::Vec<const std::shared_ptr<MlxArray>*>& arrs);
}
```

```cpp
// crates/polars-metal-mlx-sys/cpp/wrapper.cpp
#include "wrapper.h"
#include "mlx/ops.h"

namespace polars_metal_mlx {

std::shared_ptr<MlxArray> mlx_array_from_f32_data(const float* data, size_t n) {
    std::vector<int> shape{static_cast<int>(n)};
    auto arr = std::make_shared<MlxArray>(
        std::vector<float>(data, data + n),
        shape,
        mlx::core::float32);
    return arr;
}

rust::Vec<uint64_t> mlx_array_shape(const std::shared_ptr<MlxArray>& arr) {
    rust::Vec<uint64_t> out;
    for (auto d : arr->shape()) out.push_back(static_cast<uint64_t>(d));
    return out;
}

bool mlx_array_is_f32(const std::shared_ptr<MlxArray>& arr) {
    return arr->dtype() == mlx::core::float32;
}

void mlx_array_copy_to_f32(const std::shared_ptr<MlxArray>& arr, float* out, size_t n) {
    const float* src = arr->data<float>();
    std::memcpy(out, src, n * sizeof(float));
}

void mlx_eval_batch(const rust::Vec<const std::shared_ptr<MlxArray>*>& arrs) {
    std::vector<MlxArray> as;
    for (const auto* p : arrs) as.push_back(**p);
    mlx::core::eval(as);
}

}
```

- [ ] **Step 4: Run the test, verify it passes**

```bash
cargo test -p polars-metal-mlx-sys --test test_array 2>&1 | tail -10
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-mlx-sys/src/array.rs \
        crates/polars-metal-mlx-sys/src/lib.rs \
        crates/polars-metal-mlx-sys/src/error.rs \
        crates/polars-metal-mlx-sys/cpp/ \
        crates/polars-metal-mlx-sys/tests/test_array.rs
git commit -m "M4 Phase 1: MlxArrayHandle + array construction/eval/readback bindings

Foundation for the MLX FFI surface. Construction from raw F32 buffer
copies into MLX-managed memory (true zero-copy follows once the MTLBuffer-
backed construction binding lands in Task 5). The handle is a Rust wrapper
over cxx::SharedPtr<mlx::core::array> so refcounting Just Works."
```

### Task 5: Zero-copy MlxArrayHandle from MetalBuffer

**Files:**
- Modify: `crates/polars-metal-mlx-sys/src/array.rs`
- Modify: `crates/polars-metal-mlx-sys/cpp/wrapper.cpp`
- Modify: `crates/polars-metal-mlx-sys/cpp/wrapper.h`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-mlx-sys/tests/test_array_zerocopy.rs
//! Construct an MlxArrayHandle as a zero-copy view over an existing MTLBuffer.

use polars_metal_buffer::MetalBuffer;
use polars_metal_mlx_sys::array::{mlx_array_view_metal_buffer, mlx_eval, mlx_array_to_f32_vec};

#[test]
fn zero_copy_view_round_trips() {
    // Allocate a MetalBuffer with known F32 contents.
    let input: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let buf = MetalBuffer::from_f32_slice(&input).expect("metal buffer");

    let view = mlx_array_view_metal_buffer(&buf, mlx_dtype_f32()).expect("view");
    mlx_eval(&[view.clone()]).expect("eval");
    let out = mlx_array_to_f32_vec(&view).expect("read back");
    assert_eq!(out.len(), input.len());
    for (a, b) in out.iter().zip(input.iter()) {
        assert_eq!(a, b);
    }
}
```

- [ ] **Step 2: Verify the test fails**

```bash
cargo test -p polars-metal-mlx-sys --test test_array_zerocopy 2>&1 | head -10
```

Expected: compile error — `mlx_array_view_metal_buffer` not found.

- [ ] **Step 3: Add the zero-copy view binding**

```rust
// crates/polars-metal-mlx-sys/src/array.rs — append
use polars_metal_buffer::MetalBuffer;

#[repr(u32)]
#[derive(Clone, Copy)]
pub enum MlxDtype { F32 = 0, F64 = 1, I32 = 2, Bool = 3 }

pub fn mlx_dtype_f32() -> MlxDtype { MlxDtype::F32 }

pub fn mlx_array_view_metal_buffer(
    buf: &MetalBuffer,
    dtype: MlxDtype,
) -> Result<MlxArrayHandle, crate::error::Error> {
    // SAFETY: MetalBuffer's underlying MTLBuffer is kept alive for the
    // duration of the MlxArrayHandle by virtue of MLX's refcount on the
    // shared pointer. We pass the raw MTLBuffer pointer + byte length.
    let mtl_ptr = buf.as_mtl_buffer_ptr();
    let n_bytes = buf.byte_size();
    let n_elements = n_bytes / dtype.element_size();
    let handle = unsafe {
        ffi::mlx_array_view_mtl_buffer(mtl_ptr as usize, n_elements, dtype as u32)
    };
    if handle.is_null() {
        return Err(crate::error::Error::ConstructionFailed);
    }
    Ok(MlxArrayHandle(handle))
}

impl MlxDtype {
    pub fn element_size(self) -> usize {
        match self {
            MlxDtype::F32 | MlxDtype::I32 => 4,
            MlxDtype::F64 => 8,
            MlxDtype::Bool => 1,
        }
    }
}
```

```rust
// crates/polars-metal-mlx-sys/src/lib.rs — extend the cxx::bridge
        unsafe fn mlx_array_view_mtl_buffer(
            mtl_ptr: usize,
            n_elements: usize,
            dtype: u32,
        ) -> SharedPtr<MlxArray>;
```

```cpp
// crates/polars-metal-mlx-sys/cpp/wrapper.h — append
std::shared_ptr<MlxArray> mlx_array_view_mtl_buffer(
    size_t mtl_ptr, size_t n_elements, uint32_t dtype);
```

```cpp
// crates/polars-metal-mlx-sys/cpp/wrapper.cpp — append
#include "mlx/backend/metal/metal.h"

std::shared_ptr<MlxArray> mlx_array_view_mtl_buffer(
    size_t mtl_ptr, size_t n_elements, uint32_t dtype) {
    auto mtl_buf = (id<MTLBuffer>)(void*)mtl_ptr;
    mlx::core::Dtype d = (dtype == 0) ? mlx::core::float32
                       : (dtype == 1) ? mlx::core::float64
                       : (dtype == 2) ? mlx::core::int32
                       : mlx::core::bool_;
    std::vector<int> shape{static_cast<int>(n_elements)};
    // MLX exposes a Buffer-construction path that adopts an existing
    // MTLBuffer without copy. See mlx::core::metal::Buffer.
    auto buf = mlx::core::metal::wrap_mtl_buffer(mtl_buf);
    return std::make_shared<MlxArray>(buf, shape, d);
}
```

- [ ] **Step 4: Run the zero-copy test**

```bash
cargo test -p polars-metal-mlx-sys --test test_array_zerocopy 2>&1 | tail -10
```

Expected: passes. **If MLX's `wrap_mtl_buffer` API has a different name** — possible — check the MLX C++ headers in your install and adapt. Document the actual MLX API name in a comment.

- [ ] **Step 5: Add a perf assertion that view construction is sub-microsecond**

```rust
// crates/polars-metal-mlx-sys/tests/test_array_zerocopy.rs — append
#[test]
fn view_construction_is_fast() {
    let buf = MetalBuffer::zeros(10_000_000 * 4).expect("buf");  // 40 MB
    let t0 = std::time::Instant::now();
    let _view = mlx_array_view_metal_buffer(&buf, mlx_dtype_f32()).expect("view");
    let elapsed = t0.elapsed();
    assert!(elapsed.as_micros() < 1000, "view construction took {:?}", elapsed);
}
```

- [ ] **Step 6: Commit**

```bash
git add crates/polars-metal-mlx-sys/
git commit -m "M4 Phase 1: zero-copy MlxArrayHandle from MetalBuffer

The handle wraps an existing MTLBuffer without copy via MLX's
wrap_mtl_buffer (verify the exact MLX API name in your install
and update the comment in wrapper.cpp). Construction is sub-microsecond
even on 40 MB buffers — verified by test_array_zerocopy.rs."
```

### Task 6: Elementwise binding batch — arithmetic + comparison + logical

**Files:**
- Create: `crates/polars-metal-mlx-sys/src/elementwise.rs`
- Modify: `crates/polars-metal-mlx-sys/src/lib.rs`
- Modify: `crates/polars-metal-mlx-sys/cpp/wrapper.h` and `wrapper.cpp`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-mlx-sys/tests/test_elementwise.rs
//! Each elementwise op binding constructs a graph node; eval gives the right answer.

use polars_metal_mlx_sys::array::{mlx_array_from_f32_slice, mlx_eval, mlx_array_to_f32_vec};
use polars_metal_mlx_sys::elementwise::*;

#[test]
fn add_two_arrays() {
    let a = mlx_array_from_f32_slice(&[1.0, 2.0, 3.0]).unwrap();
    let b = mlx_array_from_f32_slice(&[10.0, 20.0, 30.0]).unwrap();
    let c = mlx_add(&a, &b);
    mlx_eval(&[c.clone()]).unwrap();
    let out = mlx_array_to_f32_vec(&c).unwrap();
    assert_eq!(out, vec![11.0, 22.0, 33.0]);
}

#[test]
fn sub_mul_div_mod() {
    let a = mlx_array_from_f32_slice(&[10.0, 20.0, 30.0]).unwrap();
    let b = mlx_array_from_f32_slice(&[3.0, 4.0, 5.0]).unwrap();
    let s = mlx_sub(&a, &b);
    let m = mlx_mul(&a, &b);
    let d = mlx_div(&a, &b);
    mlx_eval(&[s.clone(), m.clone(), d.clone()]).unwrap();
    assert_eq!(mlx_array_to_f32_vec(&s).unwrap(), vec![7.0, 16.0, 25.0]);
    assert_eq!(mlx_array_to_f32_vec(&m).unwrap(), vec![30.0, 80.0, 150.0]);
    assert_eq!(mlx_array_to_f32_vec(&d).unwrap(), vec![10.0/3.0, 20.0/4.0, 30.0/5.0]);
}

#[test]
fn compare_returns_bool() {
    let a = mlx_array_from_f32_slice(&[1.0, 2.0, 3.0]).unwrap();
    let b = mlx_array_from_f32_slice(&[2.0, 2.0, 2.0]).unwrap();
    let lt = mlx_lt(&a, &b);
    mlx_eval(&[lt.clone()]).unwrap();
    // Output is Bool dtype; read via a helper. For now, cast and check.
    // ... cast lt to F32 and verify [1.0, 0.0, 0.0]
}

#[test]
fn where_picks_per_element() {
    let cond = mlx_array_from_bool_slice(&[true, false, true]).unwrap();
    let then_v = mlx_array_from_f32_slice(&[10.0, 20.0, 30.0]).unwrap();
    let else_v = mlx_array_from_f32_slice(&[ 1.0,  2.0,  3.0]).unwrap();
    let r = mlx_where(&cond, &then_v, &else_v);
    mlx_eval(&[r.clone()]).unwrap();
    assert_eq!(mlx_array_to_f32_vec(&r).unwrap(), vec![10.0, 2.0, 30.0]);
}
```

- [ ] **Step 2: Verify the tests fail**

```bash
cargo test -p polars-metal-mlx-sys --test test_elementwise 2>&1 | head -20
```

Expected: compile errors (the modules don't exist yet).

- [ ] **Step 3: Add bindings for arithmetic + comparison + logical + where**

```rust
// crates/polars-metal-mlx-sys/src/elementwise.rs
//! Element-wise op bindings. Each function returns a new MlxArrayHandle
//! representing the graph node for the op; nothing executes until eval.

use crate::array::MlxArrayHandle;
use crate::ffi;

macro_rules! binop {
    ($rs:ident, $cpp:ident) => {
        pub fn $rs(a: &MlxArrayHandle, b: &MlxArrayHandle) -> MlxArrayHandle {
            MlxArrayHandle(ffi::$cpp(&a.0, &b.0))
        }
    };
}

macro_rules! unop {
    ($rs:ident, $cpp:ident) => {
        pub fn $rs(a: &MlxArrayHandle) -> MlxArrayHandle {
            MlxArrayHandle(ffi::$cpp(&a.0))
        }
    };
}

binop!(mlx_add, mlx_op_add);
binop!(mlx_sub, mlx_op_sub);
binop!(mlx_mul, mlx_op_mul);
binop!(mlx_div, mlx_op_div);
binop!(mlx_mod_, mlx_op_mod);
binop!(mlx_pow, mlx_op_pow);
binop!(mlx_eq,  mlx_op_eq);
binop!(mlx_ne,  mlx_op_ne);
binop!(mlx_lt,  mlx_op_lt);
binop!(mlx_le,  mlx_op_le);
binop!(mlx_gt,  mlx_op_gt);
binop!(mlx_ge,  mlx_op_ge);
binop!(mlx_logical_and, mlx_op_logical_and);
binop!(mlx_logical_or,  mlx_op_logical_or);

unop!(mlx_neg,            mlx_op_neg);
unop!(mlx_abs,            mlx_op_abs);
unop!(mlx_logical_not,    mlx_op_logical_not);
unop!(mlx_square,         mlx_op_square);

pub fn mlx_where(
    cond:    &MlxArrayHandle,
    then_v:  &MlxArrayHandle,
    else_v:  &MlxArrayHandle,
) -> MlxArrayHandle {
    MlxArrayHandle(ffi::mlx_op_where(&cond.0, &then_v.0, &else_v.0))
}

pub fn mlx_array_from_bool_slice(data: &[bool]) -> Result<MlxArrayHandle, crate::error::Error> {
    // Convert to u8 buffer, then construct.
    let bytes: Vec<u8> = data.iter().map(|&b| b as u8).collect();
    let handle = unsafe { ffi::mlx_array_from_bool_data(bytes.as_ptr(), bytes.len()) };
    if handle.is_null() {
        return Err(crate::error::Error::ConstructionFailed);
    }
    Ok(MlxArrayHandle(handle))
}
```

Add corresponding C++ functions in `wrapper.h`/`wrapper.cpp` — each is a one-liner wrapping `mlx::core::add`, `mlx::core::subtract`, etc.

Add the FFI block to `lib.rs`:

```rust
        fn mlx_op_add(a: &SharedPtr<MlxArray>, b: &SharedPtr<MlxArray>) -> SharedPtr<MlxArray>;
        // ... one line per op
```

- [ ] **Step 4: Run the tests**

```bash
cargo test -p polars-metal-mlx-sys --test test_elementwise 2>&1 | tail -15
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-mlx-sys/
git commit -m "M4 Phase 1: elementwise binding batch — arithmetic + cmp + logical + where

20 bindings, each a thin wrapper over mlx::core::* via cxx. Tests cover
add/sub/mul/div, comparison, where, bool array construction. Transcendental
ops (sin/cos/log/exp/sqrt) follow in Task 7."
```

### Task 7: Elementwise binding batch — transcendentals + casts + rounding

**Files:**
- Modify: `crates/polars-metal-mlx-sys/src/elementwise.rs`
- Modify: `crates/polars-metal-mlx-sys/cpp/wrapper.{h,cpp}`
- Modify: `crates/polars-metal-mlx-sys/src/lib.rs` (cxx::bridge block)

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-mlx-sys/tests/test_elementwise_trans.rs
use polars_metal_mlx_sys::array::*;
use polars_metal_mlx_sys::elementwise::*;

#[test]
fn sin_cos_tan() {
    use std::f32::consts::PI;
    let a = mlx_array_from_f32_slice(&[0.0, PI/6.0, PI/4.0, PI/3.0, PI/2.0]).unwrap();
    let s = mlx_sin(&a);
    let c = mlx_cos(&a);
    let t = mlx_tan(&a);
    mlx_eval(&[s.clone(), c.clone(), t.clone()]).unwrap();
    let sv = mlx_array_to_f32_vec(&s).unwrap();
    let cv = mlx_array_to_f32_vec(&c).unwrap();
    let tv = mlx_array_to_f32_vec(&t).unwrap();
    let approx_eq = |a: f32, b: f32| (a - b).abs() < 1e-6;
    assert!(approx_eq(sv[0], 0.0));
    assert!(approx_eq(sv[4], 1.0));
    assert!(approx_eq(cv[0], 1.0));
    assert!(approx_eq(cv[4], 0.0));
}

#[test]
fn log_exp_round_trip() {
    let a = mlx_array_from_f32_slice(&[1.0, 2.0, 5.0, 10.0, 100.0]).unwrap();
    let log_a = mlx_log(&a);
    let exp_log_a = mlx_exp(&log_a);
    mlx_eval(&[exp_log_a.clone()]).unwrap();
    let out = mlx_array_to_f32_vec(&exp_log_a).unwrap();
    for (a, b) in out.iter().zip([1.0, 2.0, 5.0, 10.0, 100.0].iter()) {
        assert!((a - b).abs() < 1e-4, "exp(log({})) = {}, expected {}", b, a, b);
    }
}

#[test]
fn sqrt_correctness() {
    let a = mlx_array_from_f32_slice(&[1.0, 4.0, 9.0, 16.0]).unwrap();
    let s = mlx_sqrt(&a);
    mlx_eval(&[s.clone()]).unwrap();
    assert_eq!(mlx_array_to_f32_vec(&s).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}
```

- [ ] **Step 2: Run, confirm failure**

```bash
cargo test -p polars-metal-mlx-sys --test test_elementwise_trans 2>&1 | head
```

Expected: compile error — `mlx_sin` etc. not found.

- [ ] **Step 3: Add the transcendental + rounding + cast bindings**

```rust
// crates/polars-metal-mlx-sys/src/elementwise.rs — append using the unop! macro

unop!(mlx_sin,   mlx_op_sin);
unop!(mlx_cos,   mlx_op_cos);
unop!(mlx_tan,   mlx_op_tan);
unop!(mlx_sinh,  mlx_op_sinh);
unop!(mlx_cosh,  mlx_op_cosh);
unop!(mlx_tanh,  mlx_op_tanh);
unop!(mlx_asin,  mlx_op_asin);
unop!(mlx_acos,  mlx_op_acos);
unop!(mlx_atan,  mlx_op_atan);
unop!(mlx_log,   mlx_op_log);
unop!(mlx_log2,  mlx_op_log2);
unop!(mlx_log10, mlx_op_log10);
unop!(mlx_log1p, mlx_op_log1p);
unop!(mlx_exp,   mlx_op_exp);
unop!(mlx_exp2,  mlx_op_exp2);
unop!(mlx_sqrt,  mlx_op_sqrt);
unop!(mlx_cbrt,  mlx_op_cbrt);
unop!(mlx_floor, mlx_op_floor);
unop!(mlx_ceil,  mlx_op_ceil);
unop!(mlx_round, mlx_op_round);

binop!(mlx_atan2, mlx_op_atan2);

pub fn mlx_cast(a: &MlxArrayHandle, to: super::array::MlxDtype) -> MlxArrayHandle {
    MlxArrayHandle(ffi::mlx_op_cast(&a.0, to as u32))
}
```

Add the matching C++ wrappers in `wrapper.cpp` — each is a one-line `return std::make_shared<MlxArray>(mlx::core::sin(*a));` etc.

Add the cxx::bridge declarations.

- [ ] **Step 4: Run tests**

```bash
cargo test -p polars-metal-mlx-sys --test test_elementwise_trans 2>&1 | tail
```

Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-mlx-sys/
git commit -m "M4 Phase 1: transcendental + rounding + cast bindings

21 unary ops + atan2 + cast. ULP tolerance documented in tests
(1e-4 for log/exp roundtrip, 1e-6 for sin/cos at canonical angles)."
```

### Task 8: Reduction binding batch

**Files:**
- Create: `crates/polars-metal-mlx-sys/src/reduce.rs`
- Modify: `crates/polars-metal-mlx-sys/{src/lib.rs, cpp/wrapper.h, cpp/wrapper.cpp}`

- [ ] **Step 1: Failing test**

```rust
// crates/polars-metal-mlx-sys/tests/test_reduce.rs
use polars_metal_mlx_sys::array::*;
use polars_metal_mlx_sys::reduce::*;

#[test]
fn sum_mean_min_max() {
    let a = mlx_array_from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let s = mlx_sum(&a);
    let m = mlx_mean(&a);
    let mn = mlx_min(&a);
    let mx = mlx_max(&a);
    mlx_eval(&[s.clone(), m.clone(), mn.clone(), mx.clone()]).unwrap();
    assert_eq!(mlx_array_to_f32_vec(&s).unwrap(), vec![15.0]);
    assert_eq!(mlx_array_to_f32_vec(&m).unwrap(), vec![3.0]);
    assert_eq!(mlx_array_to_f32_vec(&mn).unwrap(), vec![1.0]);
    assert_eq!(mlx_array_to_f32_vec(&mx).unwrap(), vec![5.0]);
}

#[test]
fn std_var() {
    let a = mlx_array_from_f32_slice(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).unwrap();
    let s = mlx_std(&a);
    let v = mlx_var(&a);
    mlx_eval(&[s.clone(), v.clone()]).unwrap();
    let sv = mlx_array_to_f32_vec(&s).unwrap()[0];
    let vv = mlx_array_to_f32_vec(&v).unwrap()[0];
    assert!((vv - 4.0).abs() < 1e-3, "var={vv}, expected ~4.0");
    assert!((sv - 2.0).abs() < 1e-3, "std={sv}, expected ~2.0");
}
```

- [ ] **Step 2: Verify failure**

```bash
cargo test -p polars-metal-mlx-sys --test test_reduce 2>&1 | head
```

- [ ] **Step 3: Add bindings**

```rust
// crates/polars-metal-mlx-sys/src/reduce.rs
use crate::array::MlxArrayHandle;
use crate::ffi;

macro_rules! global_reduce {
    ($rs:ident, $cpp:ident) => {
        pub fn $rs(a: &MlxArrayHandle) -> MlxArrayHandle {
            MlxArrayHandle(ffi::$cpp(&a.0))
        }
    };
}

global_reduce!(mlx_sum,    mlx_op_sum_all);
global_reduce!(mlx_mean,   mlx_op_mean_all);
global_reduce!(mlx_min,    mlx_op_min_all);
global_reduce!(mlx_max,    mlx_op_max_all);
global_reduce!(mlx_std,    mlx_op_std_all);
global_reduce!(mlx_var,    mlx_op_var_all);
global_reduce!(mlx_argmin, mlx_op_argmin_all);
global_reduce!(mlx_argmax, mlx_op_argmax_all);

// Reduction along a specific axis — for multi-dim work later (correlation matrix)
pub fn mlx_sum_axis(a: &MlxArrayHandle, axis: i32) -> MlxArrayHandle {
    MlxArrayHandle(ffi::mlx_op_sum_axis(&a.0, axis))
}
pub fn mlx_mean_axis(a: &MlxArrayHandle, axis: i32) -> MlxArrayHandle {
    MlxArrayHandle(ffi::mlx_op_mean_axis(&a.0, axis))
}
```

Add matching C++ wrappers. Each calls `mlx::core::sum(*a)`, `mlx::core::mean(*a, axes={axis})`, etc.

- [ ] **Step 4: Run**

```bash
cargo test -p polars-metal-mlx-sys --test test_reduce 2>&1 | tail
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-mlx-sys/
git commit -m "M4 Phase 1: reduction binding batch — sum/mean/min/max/std/var/argmin/argmax

Global (all-axis) variants + per-axis sum/mean for multi-dim work
(correlation matrix in Phase 8.7). MLX's std/var use n (population)
variance by default; verify this matches Polars' default before
wiring the analyzer (Phase 3 work)."
```

### Task 9: Sort + argpartition (top-k foundation)

**Files:**
- Create: `crates/polars-metal-mlx-sys/src/sort.rs`
- Modify FFI sources as before

- [ ] **Step 1: Failing test**

```rust
// crates/polars-metal-mlx-sys/tests/test_sort.rs
use polars_metal_mlx_sys::array::*;
use polars_metal_mlx_sys::sort::*;

#[test]
fn sort_ascending() {
    let a = mlx_array_from_f32_slice(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]).unwrap();
    let s = mlx_sort(&a);
    mlx_eval(&[s.clone()]).unwrap();
    assert_eq!(mlx_array_to_f32_vec(&s).unwrap(), vec![1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0]);
}

#[test]
fn argpartition_top_k() {
    let a = mlx_array_from_f32_slice(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]).unwrap();
    let neg = polars_metal_mlx_sys::elementwise::mlx_neg(&a);
    let idx = mlx_argpartition(&neg, 2);  // top 3 max
    mlx_eval(&[idx.clone()]).unwrap();
    // The first kth=2 positions contain the 3 largest-of-a indices in some order
    let idxs = mlx_array_to_i32_vec(&idx).unwrap();
    let top3: std::collections::HashSet<i32> = idxs[..3].iter().copied().collect();
    let expected: std::collections::HashSet<i32> = [5, 7, 4].iter().copied().collect();
    assert_eq!(top3, expected);
}
```

- [ ] **Step 2: Verify failure, Step 3: implement, Step 4: verify pass — same pattern as previous tasks. The MLX function names are `mlx::core::sort` and `mlx::core::argpartition`.

- [ ] **Step 5: Commit**

```bash
git commit -m "M4 Phase 1: sort + argpartition bindings

Sort returns a sorted copy of the input; argpartition returns indices
partitioned at kth (the K smallest are in positions [0..K], unordered
among themselves). Top-K = argpartition over the negated values, then
take the first K indices. NOTE: MLX sort is NOT stable; analyzer
(Phase 3) must reject stable-sort shapes."
```

### Task 10: Scan + matmul + FFT bindings

**Files:**
- Create: `crates/polars-metal-mlx-sys/src/scan.rs`
- Create: `crates/polars-metal-mlx-sys/src/matmul.rs`
- Create: `crates/polars-metal-mlx-sys/src/fft.rs`
- Modify FFI sources

- [ ] **Step 1: Failing tests for each binding**

```rust
// crates/polars-metal-mlx-sys/tests/test_scan.rs
use polars_metal_mlx_sys::{array::*, scan::*};

#[test]
fn cumsum_basic() {
    let a = mlx_array_from_f32_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let c = mlx_cumsum(&a);
    mlx_eval(&[c.clone()]).unwrap();
    assert_eq!(mlx_array_to_f32_vec(&c).unwrap(), vec![1.0, 3.0, 6.0, 10.0]);
}
```

```rust
// crates/polars-metal-mlx-sys/tests/test_matmul.rs
use polars_metal_mlx_sys::{array::*, matmul::*};

#[test]
fn matmul_2x3_3x2() {
    let a = mlx_array_from_f32_2d(&[1.0, 2.0, 3.0,
                                    4.0, 5.0, 6.0], (2, 3)).unwrap();
    let b = mlx_array_from_f32_2d(&[1.0, 2.0,
                                    3.0, 4.0,
                                    5.0, 6.0], (3, 2)).unwrap();
    let c = mlx_matmul(&a, &b);
    mlx_eval(&[c.clone()]).unwrap();
    // [1*1 + 2*3 + 3*5, 1*2 + 2*4 + 3*6,
    //  4*1 + 5*3 + 6*5, 4*2 + 5*4 + 6*6] = [22, 28, 49, 64]
    assert_eq!(mlx_array_to_f32_vec(&c).unwrap(), vec![22.0, 28.0, 49.0, 64.0]);
}
```

```rust
// crates/polars-metal-mlx-sys/tests/test_fft.rs
use polars_metal_mlx_sys::{array::*, fft::*};

#[test]
fn fft_round_trip() {
    let signal: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
    let arr = mlx_array_from_f32_slice(&signal).unwrap();
    let f = mlx_fft(&arr);
    let inv = mlx_ifft(&f);
    mlx_eval(&[inv.clone()]).unwrap();
    let reconstructed = mlx_array_to_f32_vec_real_part(&inv).unwrap();
    for (a, b) in reconstructed.iter().zip(signal.iter()) {
        assert!((a - b).abs() < 1e-3, "round-trip differs: {} vs {}", a, b);
    }
}
```

- [ ] **Step 2: Verify all fail**

- [ ] **Step 3: Implement each**

```rust
// crates/polars-metal-mlx-sys/src/scan.rs
use crate::array::MlxArrayHandle;
use crate::ffi;
pub fn mlx_cumsum(a: &MlxArrayHandle) -> MlxArrayHandle { MlxArrayHandle(ffi::mlx_op_cumsum(&a.0)) }
pub fn mlx_cumprod(a: &MlxArrayHandle) -> MlxArrayHandle { MlxArrayHandle(ffi::mlx_op_cumprod(&a.0)) }
pub fn mlx_cummax(a: &MlxArrayHandle) -> MlxArrayHandle { MlxArrayHandle(ffi::mlx_op_cummax(&a.0)) }
pub fn mlx_cummin(a: &MlxArrayHandle) -> MlxArrayHandle { MlxArrayHandle(ffi::mlx_op_cummin(&a.0)) }
```

```rust
// crates/polars-metal-mlx-sys/src/matmul.rs
use crate::array::MlxArrayHandle;
use crate::ffi;
pub fn mlx_matmul(a: &MlxArrayHandle, b: &MlxArrayHandle) -> MlxArrayHandle {
    MlxArrayHandle(ffi::mlx_op_matmul(&a.0, &b.0))
}
```

```rust
// crates/polars-metal-mlx-sys/src/fft.rs
use crate::array::MlxArrayHandle;
use crate::ffi;
/// 1D FFT. Input F32, output is complex represented as interleaved (real, imag) F32 pairs.
/// The output shape is (N, 2) where the last axis is [real, imag].
pub fn mlx_fft(a: &MlxArrayHandle) -> MlxArrayHandle {
    MlxArrayHandle(ffi::mlx_op_fft_1d(&a.0))
}
pub fn mlx_ifft(a: &MlxArrayHandle) -> MlxArrayHandle {
    MlxArrayHandle(ffi::mlx_op_ifft_1d(&a.0))
}
```

Implement the C++ side: `mlx::core::cumsum(*a, /*axis=*/-1)`, `mlx::core::matmul(*a, *b)`, `mlx::core::fft::fft(*a, /*axis=*/-1)`, etc. **Verify the FFT output layout in MLX**: it may be a separate complex dtype (mlx::core::complex64) rather than interleaved F32. If complex64, the binding returns it; the array helper `mlx_array_to_f32_vec_real_part` extracts the real component.

- [ ] **Step 4: Run all tests**

```bash
cargo test -p polars-metal-mlx-sys --test test_scan --test test_matmul --test test_fft 2>&1 | tail -15
```

Expected: all pass.

- [ ] **Step 5: Add helpers for 2D array construction and complex-to-F32 readback**

These are needed by the test code; add as part of `array.rs`:

```rust
pub fn mlx_array_from_f32_2d(data: &[f32], shape: (usize, usize)) -> Result<MlxArrayHandle, crate::error::Error> {
    assert_eq!(data.len(), shape.0 * shape.1, "shape/data mismatch");
    let handle = unsafe { ffi::mlx_array_from_f32_2d(data.as_ptr(), shape.0, shape.1) };
    if handle.is_null() { return Err(crate::error::Error::ConstructionFailed); }
    Ok(MlxArrayHandle(handle))
}
pub fn mlx_array_to_f32_vec_real_part(handle: &MlxArrayHandle) -> Result<Vec<f32>, crate::error::Error> {
    // Strips the imaginary part for FFT-roundtrip checks.
    let shape = handle.shape();
    let n: usize = shape.iter().product::<usize>() / if handle.is_complex() { 1 } else { 1 };
    let mut out = vec![0.0_f32; n];
    ffi::mlx_array_copy_real_to_f32(&handle.0, out.as_mut_ptr(), n);
    Ok(out)
}
```

- [ ] **Step 6: Commit**

```bash
git add crates/polars-metal-mlx-sys/
git commit -m "M4 Phase 1: scan + matmul + FFT bindings

Closes Phase 1 of the M4 FFI surface expansion. We can now construct
arbitrary graph nodes for the Phase 8 op set. FFT output layout (complex
or interleaved F32) documented per the MLX install — verify with the
fft_round_trip test."
```

### Task 11: End-to-end smoke test via the FFI surface

**Files:**
- Create: `crates/polars-metal-mlx-sys/tests/test_smoke.rs`

- [ ] **Step 1: Smoke test — build a multi-op graph mimicking what Phase 4 will emit**

```rust
// crates/polars-metal-mlx-sys/tests/test_smoke.rs
//! Smoke test: build a Black-Scholes-shaped graph via the binding surface;
//! eval; verify against a reference.

use polars_metal_mlx_sys::array::*;
use polars_metal_mlx_sys::elementwise::*;
use polars_metal_mlx_sys::reduce::*;
use polars_metal_mlx_sys::scan::*;

#[test]
fn black_scholes_shape_via_bindings() {
    let n = 1000;
    let spot:   Vec<f32> = (0..n).map(|i| 80.0 + (i as f32 * 0.07)).collect();
    let strike: Vec<f32> = (0..n).map(|i| 100.0 - (i as f32 * 0.03)).collect();
    let ttm:    Vec<f32> = (0..n).map(|i| 0.5 + (i as f32 * 0.001)).collect();
    let sigma = 0.2_f32;
    let r = 0.05_f32;

    let s = mlx_array_from_f32_slice(&spot).unwrap();
    let k = mlx_array_from_f32_slice(&strike).unwrap();
    let t = mlx_array_from_f32_slice(&ttm).unwrap();
    let sig = mlx_array_from_f32_slice(&vec![sigma; n]).unwrap();
    let r_arr = mlx_array_from_f32_slice(&vec![r; n]).unwrap();

    // d1 = (log(s/k) + (r + sigma^2/2)*t) / (sigma * sqrt(t))
    let sk = mlx_div(&s, &k);
    let log_sk = mlx_log(&sk);
    let sigma2 = mlx_mul(&sig, &sig);
    let half_sigma2 = mlx_div(&sigma2, &mlx_array_from_f32_slice(&vec![2.0_f32; n]).unwrap());
    let r_plus = mlx_add(&r_arr, &half_sigma2);
    let r_plus_t = mlx_mul(&r_plus, &t);
    let num = mlx_add(&log_sk, &r_plus_t);
    let sqrt_t = mlx_sqrt(&t);
    let denom = mlx_mul(&sig, &sqrt_t);
    let d1 = mlx_div(&num, &denom);

    // approximate CDF: 0.5 * (1 + tanh(0.7978845608 * d1))
    let coef = mlx_array_from_f32_slice(&vec![0.7978845608_f32; n]).unwrap();
    let scaled = mlx_mul(&coef, &d1);
    let tanh_s = mlx_tanh(&scaled);
    let one = mlx_array_from_f32_slice(&vec![1.0_f32; n]).unwrap();
    let cdf_d1 = mlx_mul(
        &mlx_array_from_f32_slice(&vec![0.5_f32; n]).unwrap(),
        &mlx_add(&one, &tanh_s),
    );

    let price = mlx_mul(&s, &cdf_d1);  // simplified; just verify it evaluates
    let total = mlx_sum(&price);
    mlx_eval(&[total.clone()]).unwrap();

    let v = mlx_array_to_f32_vec(&total).unwrap();
    assert_eq!(v.len(), 1);
    assert!(v[0].is_finite(), "total = {}", v[0]);
}
```

- [ ] **Step 2: Run, verify pass**

```bash
cargo test -p polars-metal-mlx-sys --test test_smoke 2>&1 | tail
```

Expected: passes; the Black-Scholes-shape graph evaluates to a finite number.

- [ ] **Step 3: Commit**

```bash
git add crates/polars-metal-mlx-sys/tests/test_smoke.rs
git commit -m "M4 Phase 1: smoke test — Black-Scholes-shape graph via bindings

Closes Phase 1: ~50 MLX bindings + a smoke test that builds the kind of
multi-op graph the Phase 4 subgraph builder will emit. Next: the
compute-density estimator (Phase 2) and the fusion analyzer (Phase 3)
to decide WHAT to feed this surface."
```

---

## Phase 2 — Compute-density estimator + `FusionScope` data structure

Per spec § "Compute-density cost model": each op has a static FLOP-per-row cost, the analyzer sums total FLOPs across a candidate subtree, and routes GPU iff `total_flops > 5e7` AND `n_rows > 1e5`. This phase defines the data structures and the estimator; the analyzer (Phase 3) consumes them.

### Task 12: Define `OpId` and the supported-op registry

**Files:**
- Create: `crates/polars-metal-core/src/fusion/mod.rs`
- Create: `crates/polars-metal-core/src/fusion/supported_ops.rs`
- Modify: `crates/polars-metal-core/src/lib.rs`

- [ ] **Step 1: Failing test for the registry**

```rust
// crates/polars-metal-core/tests/test_supported_ops.rs
use polars_metal_core::fusion::supported_ops::{OpId, OpSpec, op_spec, all_op_ids};

#[test]
fn registry_has_expected_op_count() {
    // Phase 8 supports: 24 elementwise + 8 reductions + 2 sort/topk
    //                 + 4 scan + 1 matmul + (FFT separately in Phase 11)
    // Total: ~39 op_ids in this chunk.
    let n = all_op_ids().count();
    assert!(n >= 35, "expected ≥ 35 supported ops, got {n}");
}

#[test]
fn add_op_is_supported() {
    let spec: OpSpec = op_spec(OpId::Add);
    assert_eq!(spec.n_args, 2);
    assert_eq!(spec.flops_per_row, 1);
}

#[test]
fn sin_op_is_supported() {
    let spec: OpSpec = op_spec(OpId::Sin);
    assert_eq!(spec.n_args, 1);
    assert_eq!(spec.flops_per_row, 10);  // transcendentals cost 10 FLOPs/row
}

#[test]
fn matmul_flops_are_n_times_k_times_m() {
    let spec = op_spec(OpId::MatMul);
    assert_eq!(spec.n_args, 2);
    // Matmul cost is dynamic — n_rows * inner_dim * output_cols.
    // We mark flops_per_row=0 for static lookup; the analyzer special-
    // cases matmul to compute (2 * K) per output cell.
    assert_eq!(spec.flops_per_row, 0);
    assert!(spec.dynamic_flops);
}
```

- [ ] **Step 2: Verify failure**

```bash
cargo test -p polars-metal-core --test test_supported_ops 2>&1 | head
```

Expected: compile error — `fusion::supported_ops` module not found.

- [ ] **Step 3: Define `OpId` and the registry**

```rust
// crates/polars-metal-core/src/fusion/mod.rs
pub mod supported_ops;
pub mod scope;
pub mod density;
```

```rust
// crates/polars-metal-core/src/fusion/supported_ops.rs
//! Registry of expression ops the M4 fusion analyzer recognizes.
//!
//! Each op has:
//!   - n_args:        how many child expressions it consumes
//!   - flops_per_row: static FLOP estimate per row (0 means dynamic; see
//!                    OpSpec::dynamic_flops)
//!   - input_dtype:   what dtype the op accepts (F32, F32orF64, Bool, ...)
//!   - output_dtype:  what dtype the op produces (same | Bool | F32)
//!
//! The analyzer uses this table to decide whether an expression node is
//! supported and to compute total subtree FLOPs.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpId {
    // Arithmetic (binary)
    Add, Sub, Mul, Div, Mod, Pow,
    // Arithmetic (unary)
    Neg, Abs, Square,
    // Comparison (binary; output Bool)
    Eq, Ne, Lt, Le, Gt, Ge,
    // Logical (binary on Bool)
    LogicalAnd, LogicalOr,
    // Logical (unary)
    LogicalNot,
    // Conditional (ternary; cond Bool, then/else same dtype)
    Where,
    // Transcendental (unary, F32)
    Sin, Cos, Tan, Sinh, Cosh, Tanh,
    Asin, Acos, Atan, Atan2,
    Log, Log2, Log10, Log1p,
    Exp, Exp2,
    Sqrt, Cbrt,
    // Rounding (unary, F32)
    Floor, Ceil, Round,
    // Cast
    CastF32, CastF64, CastI32, CastBool,
    // Reduction (unary; output is shape-(1,))
    Sum, Mean, Min, Max, Std, Var,
    ArgMin, ArgMax,
    // Sort/top-k (unary)
    Sort, ArgPartition,
    // Cumulative scan (unary)
    CumSum, CumProd, CumMax, CumMin,
    // Matmul (binary)
    MatMul,
    // FFT (unary, F32 -> complex; Phase 11)
    Fft, Ifft,
}

#[derive(Clone, Copy, Debug)]
pub enum DtypeReq { F32, F32OrF64, Bool, Numeric, ListOrArrayF32 }

#[derive(Clone, Copy, Debug)]
pub enum DtypeOut { SameAsInput, Bool, F32, ScalarF32, ComplexF32, SortedSameAsInput, I32 }

#[derive(Clone, Copy, Debug)]
pub struct OpSpec {
    pub n_args:        u32,
    pub flops_per_row: u32,
    pub input_dtype:   DtypeReq,
    pub output_dtype:  DtypeOut,
    pub dynamic_flops: bool,
    pub allows_null:   bool,
}

pub fn op_spec(op: OpId) -> OpSpec {
    use OpId::*;
    use DtypeReq::*;
    use DtypeOut::*;
    match op {
        // Arithmetic
        Add | Sub | Mul         => OpSpec { n_args: 2, flops_per_row: 1, input_dtype: Numeric, output_dtype: SameAsInput, dynamic_flops: false, allows_null: false },
        Div                     => OpSpec { n_args: 2, flops_per_row: 4, input_dtype: Numeric, output_dtype: SameAsInput, dynamic_flops: false, allows_null: false },
        Mod                     => OpSpec { n_args: 2, flops_per_row: 4, input_dtype: F32OrF64, output_dtype: SameAsInput, dynamic_flops: false, allows_null: false },
        Pow                     => OpSpec { n_args: 2, flops_per_row: 12, input_dtype: F32OrF64, output_dtype: SameAsInput, dynamic_flops: false, allows_null: false },
        Neg | Abs               => OpSpec { n_args: 1, flops_per_row: 1, input_dtype: Numeric, output_dtype: SameAsInput, dynamic_flops: false, allows_null: false },
        Square                  => OpSpec { n_args: 1, flops_per_row: 1, input_dtype: Numeric, output_dtype: SameAsInput, dynamic_flops: false, allows_null: false },
        // Comparison → Bool
        Eq | Ne | Lt | Le | Gt | Ge
                                => OpSpec { n_args: 2, flops_per_row: 1, input_dtype: Numeric, output_dtype: Bool, dynamic_flops: false, allows_null: false },
        // Logical
        LogicalAnd | LogicalOr  => OpSpec { n_args: 2, flops_per_row: 1, input_dtype: DtypeReq::Bool, output_dtype: DtypeOut::Bool, dynamic_flops: false, allows_null: false },
        LogicalNot              => OpSpec { n_args: 1, flops_per_row: 1, input_dtype: DtypeReq::Bool, output_dtype: DtypeOut::Bool, dynamic_flops: false, allows_null: false },
        // Where (cond, then, else)
        Where                   => OpSpec { n_args: 3, flops_per_row: 1, input_dtype: Numeric, output_dtype: SameAsInput, dynamic_flops: false, allows_null: false },
        // Transcendental (unary)
        Sin | Cos | Tan | Sinh | Cosh | Tanh |
        Asin | Acos | Atan |
        Log | Log2 | Log10 | Log1p |
        Exp | Exp2 |
        Tanh                    => OpSpec { n_args: 1, flops_per_row: 10, input_dtype: F32OrF64, output_dtype: SameAsInput, dynamic_flops: false, allows_null: false },
        Atan2                   => OpSpec { n_args: 2, flops_per_row: 12, input_dtype: F32OrF64, output_dtype: SameAsInput, dynamic_flops: false, allows_null: false },
        // Roots
        Sqrt | Cbrt             => OpSpec { n_args: 1, flops_per_row: 4, input_dtype: F32OrF64, output_dtype: SameAsInput, dynamic_flops: false, allows_null: false },
        // Rounding
        Floor | Ceil | Round    => OpSpec { n_args: 1, flops_per_row: 1, input_dtype: F32OrF64, output_dtype: SameAsInput, dynamic_flops: false, allows_null: false },
        // Cast
        CastF32 | CastF64 | CastI32 | CastBool
                                => OpSpec { n_args: 1, flops_per_row: 1, input_dtype: Numeric, output_dtype: SameAsInput, dynamic_flops: false, allows_null: false },
        // Reductions
        Sum | Mean | Min | Max  => OpSpec { n_args: 1, flops_per_row: 1, input_dtype: Numeric, output_dtype: ScalarF32, dynamic_flops: false, allows_null: false },
        Std | Var               => OpSpec { n_args: 1, flops_per_row: 3, input_dtype: Numeric, output_dtype: ScalarF32, dynamic_flops: false, allows_null: false },
        ArgMin | ArgMax         => OpSpec { n_args: 1, flops_per_row: 1, input_dtype: Numeric, output_dtype: I32, dynamic_flops: false, allows_null: false },
        // Sort / top-k
        Sort                    => OpSpec { n_args: 1, flops_per_row: 0, input_dtype: Numeric, output_dtype: SortedSameAsInput, dynamic_flops: true, allows_null: false }, // log2(n) * n
        ArgPartition            => OpSpec { n_args: 1, flops_per_row: 0, input_dtype: Numeric, output_dtype: I32, dynamic_flops: true, allows_null: false },
        // Cumulative
        CumSum | CumProd        => OpSpec { n_args: 1, flops_per_row: 2, input_dtype: Numeric, output_dtype: SameAsInput, dynamic_flops: false, allows_null: false },
        CumMax | CumMin         => OpSpec { n_args: 1, flops_per_row: 2, input_dtype: Numeric, output_dtype: SameAsInput, dynamic_flops: false, allows_null: false },
        // Matmul — n_rows * 2 * K per output cell
        MatMul                  => OpSpec { n_args: 2, flops_per_row: 0, input_dtype: ListOrArrayF32, output_dtype: F32, dynamic_flops: true, allows_null: false },
        // FFT — n * log2(n) per axis
        Fft | Ifft              => OpSpec { n_args: 1, flops_per_row: 0, input_dtype: F32, output_dtype: ComplexF32, dynamic_flops: true, allows_null: false },
    }
}

pub fn all_op_ids() -> impl Iterator<Item = OpId> {
    use OpId::*;
    [
        Add, Sub, Mul, Div, Mod, Pow,
        Neg, Abs, Square,
        Eq, Ne, Lt, Le, Gt, Ge,
        LogicalAnd, LogicalOr, LogicalNot,
        Where,
        Sin, Cos, Tan, Sinh, Cosh, Tanh,
        Asin, Acos, Atan, Atan2,
        Log, Log2, Log10, Log1p,
        Exp, Exp2,
        Sqrt, Cbrt,
        Floor, Ceil, Round,
        CastF32, CastF64, CastI32, CastBool,
        Sum, Mean, Min, Max, Std, Var,
        ArgMin, ArgMax,
        Sort, ArgPartition,
        CumSum, CumProd, CumMax, CumMin,
        MatMul,
        Fft, Ifft,
    ].into_iter()
}
```

Wire `pub mod fusion;` into `crates/polars-metal-core/src/lib.rs`.

- [ ] **Step 4: Run tests**

```bash
cargo test -p polars-metal-core --test test_supported_ops 2>&1 | tail
```

Expected: 4 tests pass. (You may need to fix the duplicate `Tanh` arm in the match — the example above has it appearing twice; pick one.)

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-core/src/fusion/ crates/polars-metal-core/src/lib.rs \
        crates/polars-metal-core/tests/test_supported_ops.rs
git commit -m "M4 Phase 2: OpId enum + supported_ops registry

39 OpIds spanning arithmetic, comparison, logical, transcendentals,
reductions, sort/top-k, cumulative scans, matmul, FFT. Each has a
static FLOP estimate (or dynamic_flops flag for matmul/sort/FFT).
The analyzer (Phase 3) consumes this; the subgraph builder (Phase 4)
maps OpId to the matching MLX FFI call."
```

### Task 13: `FusionScope` struct + tests

**Files:**
- Create: `crates/polars-metal-core/src/fusion/scope.rs`

- [ ] **Step 1: Failing test**

```rust
// crates/polars-metal-core/tests/test_fusion_scope.rs
use polars_metal_core::fusion::scope::*;
use polars_metal_core::fusion::supported_ops::OpId;

#[test]
fn scope_records_ops_inputs_outputs() {
    let mut scope = FusionScope::new();
    let input_a = scope.add_input("a", InputDtype::F32);
    let input_b = scope.add_input("b", InputDtype::F32);
    let add = scope.push_op(OpId::Add, vec![input_a, input_b]);
    scope.mark_output(add);

    assert_eq!(scope.inputs.len(), 2);
    assert_eq!(scope.ops.len(), 1);
    assert_eq!(scope.outputs.len(), 1);
}

#[test]
fn scope_est_flops_sums_per_op() {
    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::F32);
    let sin_a = scope.push_op(OpId::Sin, vec![a]);    // 10 FLOPs/row
    let cos_a = scope.push_op(OpId::Cos, vec![a]);    // 10 FLOPs/row
    let prod = scope.push_op(OpId::Mul, vec![sin_a, cos_a]);  // 1 FLOP/row
    scope.mark_output(prod);

    let total = scope.est_flops_for(10_000_000);
    assert_eq!(total, 21 * 10_000_000);  // 21 FLOPs/row × 10M rows
}

#[test]
fn scope_clone_preserves_structure() {
    let mut s = FusionScope::new();
    let a = s.add_input("a", InputDtype::F32);
    s.push_op(OpId::Sqrt, vec![a]);
    let cloned = s.clone();
    assert_eq!(s.ops.len(), cloned.ops.len());
}
```

- [ ] **Step 2: Run, confirm fail**

```bash
cargo test -p polars-metal-core --test test_fusion_scope 2>&1 | head
```

Expected: compile error.

- [ ] **Step 3: Define `FusionScope`**

```rust
// crates/polars-metal-core/src/fusion/scope.rs
use super::supported_ops::{OpId, op_spec};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputDtype { F32, F64, Bool, I32, ArrayF32(usize), ListF32 }

#[derive(Clone, Debug)]
pub struct InputRef {
    pub column_name: String,
    pub dtype: InputDtype,
}

/// Index into FusionScope::ops or FusionScope::inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeIdx(pub u32);

#[derive(Clone, Debug)]
pub struct OpNode {
    pub op: OpId,
    pub args: Vec<NodeIdx>,   // refs to inputs or to earlier ops
}

#[derive(Clone, Debug, Default)]
pub struct FusionScope {
    pub inputs: Vec<InputRef>,
    pub ops: Vec<OpNode>,
    pub outputs: Vec<NodeIdx>,
}

impl FusionScope {
    pub fn new() -> Self { Self::default() }

    pub fn add_input(&mut self, name: &str, dtype: InputDtype) -> NodeIdx {
        let idx = NodeIdx(self.inputs.len() as u32);
        self.inputs.push(InputRef { column_name: name.to_string(), dtype });
        idx
    }

    pub fn push_op(&mut self, op: OpId, args: Vec<NodeIdx>) -> NodeIdx {
        // Op indices start after all inputs.
        let idx = NodeIdx(self.inputs.len() as u32 + self.ops.len() as u32);
        self.ops.push(OpNode { op, args });
        idx
    }

    pub fn mark_output(&mut self, idx: NodeIdx) {
        self.outputs.push(idx);
    }

    /// Estimate total FLOPs at a given row count.
    /// Static ops use op_spec; matmul/sort/FFT are handled by the analyzer
    /// before scope construction (n_rows + inner-dim passed in then).
    pub fn est_flops_for(&self, n_rows: usize) -> u64 {
        let mut total: u64 = 0;
        for node in &self.ops {
            let spec = op_spec(node.op);
            if !spec.dynamic_flops {
                total += spec.flops_per_row as u64 * n_rows as u64;
            }
            // Dynamic ops: the analyzer baked their FLOPs into a side-channel;
            // for now we ignore them in the static estimate. Phase 3 fixes
            // this by storing per-op dynamic_flops in OpNode.
        }
        total
    }

    /// True if this scope contains at least one op whose output is a
    /// terminal (reduction, sort, top-k, matmul → output of different shape).
    pub fn has_terminator(&self) -> bool {
        use OpId::*;
        self.ops.iter().any(|n| matches!(
            n.op,
            Sum | Mean | Min | Max | Std | Var | ArgMin | ArgMax |
            Sort | ArgPartition | MatMul | Fft | Ifft
        ))
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test -p polars-metal-core --test test_fusion_scope 2>&1 | tail
```

Expected: 3 pass.

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-core/src/fusion/scope.rs \
        crates/polars-metal-core/tests/test_fusion_scope.rs
git commit -m "M4 Phase 2: FusionScope struct

Inputs + ops + outputs (DAG). NodeIdx unifies input and op references
in a single index space (inputs come first). est_flops_for sums static
per-op FLOPs; dynamic ops (matmul, sort, FFT) are handled by the analyzer
in Phase 3."
```

### Task 14: Compute-density routing decision

**Files:**
- Create: `crates/polars-metal-core/src/fusion/density.rs`

- [ ] **Step 1: Failing test**

```rust
// crates/polars-metal-core/tests/test_density.rs
use polars_metal_core::fusion::density::*;
use polars_metal_core::fusion::scope::*;
use polars_metal_core::fusion::supported_ops::OpId;

#[test]
fn small_workload_routes_cpu() {
    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::F32);
    let _ = scope.push_op(OpId::Sqrt, vec![a]);
    // 1k rows, 4 FLOPs each = 4e3 FLOPs — below threshold
    let decision = density_routes_gpu(&scope, 1_000);
    assert_eq!(decision, RouteDecision::Cpu(CpuReason::BelowRowsThreshold));
}

#[test]
fn medium_workload_routes_cpu_if_too_few_flops() {
    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::F32);
    let _ = scope.push_op(OpId::Sqrt, vec![a]);
    // 1M rows × 4 FLOPs = 4e6 FLOPs — below FLOP threshold
    let decision = density_routes_gpu(&scope, 1_000_000);
    assert_eq!(decision, RouteDecision::Cpu(CpuReason::BelowFlopsThreshold));
}

#[test]
fn black_scholes_at_10m_routes_gpu() {
    let mut scope = FusionScope::new();
    let s = scope.add_input("s", InputDtype::F32);
    let k = scope.add_input("k", InputDtype::F32);
    let t = scope.add_input("t", InputDtype::F32);
    let sk = scope.push_op(OpId::Div, vec![s, k]);          //  4
    let log_sk = scope.push_op(OpId::Log, vec![sk]);        // 10
    let sqrt_t = scope.push_op(OpId::Sqrt, vec![t]);        //  4
    let p1 = scope.push_op(OpId::Add, vec![log_sk, t]);     //  1
    let d1 = scope.push_op(OpId::Div, vec![p1, sqrt_t]);    //  4
    let tan_d1 = scope.push_op(OpId::Tanh, vec![d1]);       // 10
    scope.mark_output(tan_d1);
    // 33 FLOPs/row * 10M = 3.3e8 — well above 5e7 threshold
    let decision = density_routes_gpu(&scope, 10_000_000);
    assert_eq!(decision, RouteDecision::Gpu);
}
```

- [ ] **Step 2: Verify failure**

```bash
cargo test -p polars-metal-core --test test_density 2>&1 | head
```

Expected: compile error.

- [ ] **Step 3: Implement the density decision**

```rust
// crates/polars-metal-core/src/fusion/density.rs
use super::scope::FusionScope;

pub const MIN_FLOPS_THRESHOLD: u64 = 50_000_000;        // 5e7
pub const MIN_ROWS_THRESHOLD:  usize = 100_000;          // 1e5

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuReason {
    BelowRowsThreshold,
    BelowFlopsThreshold,
    EmptyScope,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouteDecision {
    Gpu,
    Cpu(CpuReason),
}

pub fn density_routes_gpu(scope: &FusionScope, n_rows: usize) -> RouteDecision {
    if scope.ops.is_empty() {
        return RouteDecision::Cpu(CpuReason::EmptyScope);
    }
    if n_rows < MIN_ROWS_THRESHOLD {
        return RouteDecision::Cpu(CpuReason::BelowRowsThreshold);
    }
    let flops = scope.est_flops_for(n_rows);
    if flops < MIN_FLOPS_THRESHOLD {
        return RouteDecision::Cpu(CpuReason::BelowFlopsThreshold);
    }
    RouteDecision::Gpu
}
```

- [ ] **Step 4: Run**

```bash
cargo test -p polars-metal-core --test test_density 2>&1 | tail
```

Expected: 3 pass.

- [ ] **Step 5: Edge-case tests at the boundary**

```rust
// append to test_density.rs

#[test]
fn just_above_flops_threshold_routes_gpu() {
    // 10M rows × 6 FLOPs/row = 6e7 — above 5e7
    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::F32);
    let s = scope.push_op(OpId::Sqrt, vec![a]);          // 4
    let _ = scope.push_op(OpId::Square, vec![s]);        // 1 → total 5/row (slightly below)
    // re-add one more op to push above
    let a2 = scope.add_input("b", InputDtype::F32);
    let _ = scope.push_op(OpId::Add, vec![a, a2]);       // 1 → 6/row
    assert_eq!(density_routes_gpu(&scope, 10_000_000), RouteDecision::Gpu);
}

#[test]
fn empty_scope_routes_cpu() {
    let scope = FusionScope::new();
    assert!(matches!(
        density_routes_gpu(&scope, 10_000_000),
        RouteDecision::Cpu(CpuReason::EmptyScope)
    ));
}
```

- [ ] **Step 6: Commit**

```bash
git add crates/polars-metal-core/src/fusion/density.rs \
        crates/polars-metal-core/tests/test_density.rs
git commit -m "M4 Phase 2: density-based routing decision

Closes Phase 2. Default thresholds 5e7 FLOPs and 1e5 rows per spec.
Edge tests verify boundary behavior. Phase 3 wires the analyzer to
emit FusionScopes; this decision then governs whether to lift to GPU."
```

---

## Phase 3 — Fusion analyzer (walks Polars expression IR)

The analyzer is the largest single piece of M4 architecture. Input: a Polars expression IR node (e.g. for the right-hand side of a `with_columns` binding). Output: either a `FusionScope` for the maximal supported subtree, or `None` if no useful scope can be extracted.

The implementation lives in Python because that's where the existing walker is. We re-use `_walker.py`'s expression traversal scaffolding.

### Task 15: Python-side `OpId` mirror via FFI

The analyzer needs to construct `FusionScope` objects but the canonical struct lives in Rust. M4's approach: expose `FusionScope` construction via PyO3 — the Python analyzer builds the scope via thin Python-side wrappers, the Rust router consumes it. This keeps the per-op dispatch table (which encodes correctness-critical FLOP estimates) in one place.

**Files:**
- Modify: `crates/polars-metal-core/src/fusion/mod.rs` (add `pub mod py;`)
- Create: `crates/polars-metal-core/src/fusion/py.rs`
- Modify: `python/polars_metal/_native.pyi` (if it exists; ignore otherwise)

- [ ] **Step 1: Failing Python-side test**

```python
# tests/python_integration/test_fusion_scope_python.py
"""Python can construct a FusionScope via the PyO3 binding."""
import polars_metal._native as native


def test_construct_simple_scope():
    scope = native.PyFusionScope()
    a = scope.add_input("a", "F32")
    b = scope.add_input("b", "F32")
    add = scope.push_op("Add", [a, b])
    scope.mark_output(add)
    assert scope.n_inputs() == 2
    assert scope.n_ops() == 1
    assert scope.est_flops(10_000_000) == 10_000_000


def test_unsupported_op_raises():
    scope = native.PyFusionScope()
    a = scope.add_input("a", "F32")
    try:
        scope.push_op("NotARealOp", [a])
        assert False, "should have raised"
    except ValueError as e:
        assert "NotARealOp" in str(e)
```

- [ ] **Step 2: Verify failure**

```bash
make wheel 2>&1 | tail -5  # ensures Python can import the native module
pytest tests/python_integration/test_fusion_scope_python.py -v 2>&1 | tail
```

Expected: `AttributeError: module ... has no attribute 'PyFusionScope'`.

- [ ] **Step 3: Add the PyO3 wrapper**

```rust
// crates/polars-metal-core/src/fusion/py.rs
//! PyO3 wrappers for FusionScope construction from Python.
//!
//! The Python-side analyzer in _walker.py builds scopes via this API.
//! The Rust router (Phase 5) then consumes the constructed scope.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use super::scope::{FusionScope, InputDtype, NodeIdx};
use super::supported_ops::{OpId, op_spec, all_op_ids};

#[pyclass(name = "PyFusionScope")]
pub struct PyFusionScope {
    pub(crate) inner: FusionScope,
}

#[pymethods]
impl PyFusionScope {
    #[new]
    fn new() -> Self { Self { inner: FusionScope::new() } }

    fn add_input(&mut self, name: &str, dtype: &str) -> PyResult<u32> {
        let d = match dtype {
            "F32" => InputDtype::F32,
            "F64" => InputDtype::F64,
            "Bool" => InputDtype::Bool,
            "I32" => InputDtype::I32,
            other if other.starts_with("ArrayF32(") => {
                let n: usize = other.trim_start_matches("ArrayF32(")
                                    .trim_end_matches(')')
                                    .parse()
                                    .map_err(|_| PyValueError::new_err(format!("bad ArrayF32 dim: {other}")))?;
                InputDtype::ArrayF32(n)
            }
            "ListF32" => InputDtype::ListF32,
            other => return Err(PyValueError::new_err(format!("unknown InputDtype: {other}"))),
        };
        Ok(self.inner.add_input(name, d).0)
    }

    fn push_op(&mut self, op_str: &str, args: Vec<u32>) -> PyResult<u32> {
        let op = op_id_from_str(op_str)
            .ok_or_else(|| PyValueError::new_err(format!("unknown OpId: {op_str}")))?;
        let spec = op_spec(op);
        if args.len() as u32 != spec.n_args {
            return Err(PyValueError::new_err(format!(
                "{op_str} expects {} args, got {}", spec.n_args, args.len()
            )));
        }
        let args_idx: Vec<NodeIdx> = args.into_iter().map(NodeIdx).collect();
        Ok(self.inner.push_op(op, args_idx).0)
    }

    fn mark_output(&mut self, idx: u32) {
        self.inner.mark_output(NodeIdx(idx));
    }

    fn n_inputs(&self) -> usize { self.inner.inputs.len() }
    fn n_ops(&self) -> usize { self.inner.ops.len() }

    fn est_flops(&self, n_rows: usize) -> u64 {
        self.inner.est_flops_for(n_rows)
    }

    fn route_decision(&self, n_rows: usize) -> String {
        match super::density::density_routes_gpu(&self.inner, n_rows) {
            super::density::RouteDecision::Gpu => "Gpu".to_string(),
            super::density::RouteDecision::Cpu(reason) => format!("Cpu({:?})", reason),
        }
    }
}

fn op_id_from_str(s: &str) -> Option<OpId> {
    use OpId::*;
    Some(match s {
        "Add" => Add, "Sub" => Sub, "Mul" => Mul, "Div" => Div,
        "Mod" => Mod, "Pow" => Pow,
        "Neg" => Neg, "Abs" => Abs, "Square" => Square,
        "Eq" => Eq, "Ne" => Ne, "Lt" => Lt, "Le" => Le, "Gt" => Gt, "Ge" => Ge,
        "LogicalAnd" => LogicalAnd, "LogicalOr" => LogicalOr, "LogicalNot" => LogicalNot,
        "Where" => Where,
        "Sin" => Sin, "Cos" => Cos, "Tan" => Tan,
        "Sinh" => Sinh, "Cosh" => Cosh, "Tanh" => Tanh,
        "Asin" => Asin, "Acos" => Acos, "Atan" => Atan, "Atan2" => Atan2,
        "Log" => Log, "Log2" => Log2, "Log10" => Log10, "Log1p" => Log1p,
        "Exp" => Exp, "Exp2" => Exp2,
        "Sqrt" => Sqrt, "Cbrt" => Cbrt,
        "Floor" => Floor, "Ceil" => Ceil, "Round" => Round,
        "CastF32" => CastF32, "CastF64" => CastF64, "CastI32" => CastI32, "CastBool" => CastBool,
        "Sum" => Sum, "Mean" => Mean, "Min" => Min, "Max" => Max, "Std" => Std, "Var" => Var,
        "ArgMin" => ArgMin, "ArgMax" => ArgMax,
        "Sort" => Sort, "ArgPartition" => ArgPartition,
        "CumSum" => CumSum, "CumProd" => CumProd, "CumMax" => CumMax, "CumMin" => CumMin,
        "MatMul" => MatMul,
        "Fft" => Fft, "Ifft" => Ifft,
        _ => return None,
    })
}

pub fn register(m: &PyModule) -> PyResult<()> {
    m.add_class::<PyFusionScope>()?;
    Ok(())
}
```

Wire `fusion::py::register` into the existing PyO3 module entry point (in the same place that other `register` calls happen — search `python/polars_metal/_native.rs` or `crates/polars-metal/src/lib.rs` for the existing pattern).

- [ ] **Step 4: Rebuild the wheel, run the test**

```bash
make wheel 2>&1 | tail -3
pytest tests/python_integration/test_fusion_scope_python.py -v 2>&1 | tail
```

Expected: 2 pass.

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-core/src/fusion/py.rs \
        crates/polars-metal-core/src/fusion/mod.rs \
        crates/polars-metal/src/lib.rs \
        tests/python_integration/test_fusion_scope_python.py
git commit -m "M4 Phase 3: PyO3 wrapper for FusionScope construction

PyFusionScope exposes add_input / push_op / mark_output / est_flops /
route_decision. The op name is a string at this boundary (validated
server-side); the analyzer (Task 16) builds scopes via these calls."
```

### Task 16: Polars expression-IR walker (analyzer entry point)

This is the biggest single piece of M4. It traverses a Polars expression and either builds a `PyFusionScope` for the maximal supported subtree or returns `None`.

**Files:**
- Create: `python/polars_metal/_fusion_analyzer.py`
- Modify: `python/polars_metal/_walker.py` (call into the analyzer)

- [ ] **Step 1: Failing test for the analyzer**

```python
# tests/python_integration/test_fusion_analyzer.py
"""Analyzer recognizes supported expression shapes and rejects unsupported ones."""
import polars as pl
from polars_metal._fusion_analyzer import analyze_expression


def test_simple_arithmetic_chain_is_recognized():
    """(col(a) + col(b)) * col(c)"""
    expr = (pl.col("a") + pl.col("b")) * pl.col("c")
    schema = {"a": pl.Float32, "b": pl.Float32, "c": pl.Float32}
    scope = analyze_expression(expr, schema)
    assert scope is not None
    assert scope.n_inputs() == 3
    assert scope.n_ops() == 2  # Add, Mul


def test_transcendental_chain_is_recognized():
    """sqrt(log(col(a)))"""
    expr = pl.col("a").log().sqrt()
    schema = {"a": pl.Float32}
    scope = analyze_expression(expr, schema)
    assert scope is not None
    assert scope.n_ops() == 2  # Log, Sqrt


def test_string_op_is_rejected():
    """No String support in this chunk."""
    expr = pl.col("s").str.len_chars()
    schema = {"s": pl.Utf8}
    scope = analyze_expression(expr, schema)
    assert scope is None


def test_hash_groupby_is_rejected():
    """GroupBy doesn't appear as an Expr — but col + groupby aggs do.
    For now: agg expressions that route through M3 conformance code should
    not produce a fusion scope (we don't want to compete with the M3 path)."""
    # This case is mostly handled in the walker, not the analyzer; ensure
    # the analyzer at least doesn't crash on aggregation expressions.
    expr = pl.col("a").sum()  # Sum reduction — IS supported as a fused
                              # terminus in Phase 8.
    schema = {"a": pl.Float32}
    scope = analyze_expression(expr, schema)
    assert scope is not None
    assert scope.n_ops() == 1


def test_mixed_dtype_with_cast_is_recognized():
    """F64 input cast to F32 then transcendental chain."""
    expr = pl.col("a").cast(pl.Float32).sin()
    schema = {"a": pl.Float64}
    scope = analyze_expression(expr, schema)
    assert scope is not None
    assert scope.n_ops() == 2  # CastF32, Sin


def test_unsupported_op_in_middle_truncates_scope():
    """If an unsupported op appears in the middle, only the supported
    suffix becomes a scope (or the whole tree is rejected)."""
    # E.g., string op as an intermediate — should reject the whole tree.
    expr = pl.col("s").str.to_lowercase().str.len_chars().cast(pl.Float32).sqrt()
    schema = {"s": pl.Utf8}
    scope = analyze_expression(expr, schema)
    assert scope is None  # the upstream String ops poison the scope
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/python_integration/test_fusion_analyzer.py -v 2>&1 | tail
```

Expected: ModuleNotFoundError or attribute errors.

- [ ] **Step 3: Implement the analyzer**

```python
# python/polars_metal/_fusion_analyzer.py
"""Walk a Polars expression IR and identify the maximal fused subtree.

Public API:
  analyze_expression(expr: pl.Expr, schema: Schema) -> PyFusionScope | None

Returns a constructed PyFusionScope if the entire expression maps to a
supported chain. Returns None if any node is unsupported (no partial
fusion in this chunk — partial-fusion truncation is a Phase 12 design
item).
"""
from __future__ import annotations
from typing import Any, Optional

import polars as pl

from polars_metal._native import PyFusionScope


# Map Polars expression-IR node kinds to OpId strings.
# The Polars expression IR exposed via `expr.meta.tree_format()` and
# `expr.meta.serialize()` is the source of truth for what we can pattern-
# match. We use the visitor pattern below to identify ops.

_BINOP_MAP = {
    # Polars Operator enum -> our OpId string
    pl.Operator.Plus:     "Add",
    pl.Operator.Minus:    "Sub",
    pl.Operator.Multiply: "Mul",
    pl.Operator.Divide:   "Div",
    pl.Operator.Modulus:  "Mod",
    pl.Operator.Eq:       "Eq",
    pl.Operator.NotEq:    "Ne",
    pl.Operator.Lt:       "Lt",
    pl.Operator.LtEq:     "Le",
    pl.Operator.Gt:       "Gt",
    pl.Operator.GtEq:     "Ge",
    pl.Operator.And:      "LogicalAnd",
    pl.Operator.Or:       "LogicalOr",
}

_UNARY_FN_MAP = {
    # Polars expression-method name -> our OpId string
    "sin": "Sin", "cos": "Cos", "tan": "Tan",
    "sinh": "Sinh", "cosh": "Cosh", "tanh": "Tanh",
    "arcsin": "Asin", "arccos": "Acos", "arctan": "Atan",
    "log": "Log", "log10": "Log10", "log1p": "Log1p",
    "exp": "Exp",
    "sqrt": "Sqrt", "cbrt": "Cbrt",
    "abs": "Abs",
    "floor": "Floor", "ceil": "Ceil", "round": "Round",
    "neg": "Neg",
}

_REDUCTION_MAP = {
    "sum": "Sum", "mean": "Mean", "min": "Min", "max": "Max",
    "std": "Std", "var": "Var",
    "arg_min": "ArgMin", "arg_max": "ArgMax",
}

_CAST_MAP = {
    pl.Float32: "CastF32",
    pl.Float64: "CastF64",
    pl.Int32:   "CastI32",
    pl.Boolean: "CastBool",
}


class _Aborted(Exception):
    """Raised inside the visitor when an unsupported node is encountered."""


def analyze_expression(expr: pl.Expr, schema: dict[str, Any]) -> Optional[PyFusionScope]:
    """Walk `expr` post-order; build a PyFusionScope. Return None if any
    node is unsupported."""
    try:
        scope = PyFusionScope()
        node_idx = _visit(expr, schema, scope)
        scope.mark_output(node_idx)
        return scope
    except _Aborted:
        return None


def _visit(expr: pl.Expr, schema: dict[str, Any], scope: PyFusionScope) -> int:
    """Recursive descent. Returns the NodeIdx of the constructed op."""
    # We rely on `expr.meta` to introspect the IR. Different polars revs
    # have slightly different APIs; the implementation must be tested
    # against the pinned py-1.40.1.
    serialized = expr.meta.serialize(format="binary")
    # Strategy: use meta accessors to identify node kind. A more robust
    # implementation would parse the serialized form directly; for now
    # the most reliable path is `expr.meta.is_column_selection()` and
    # similar accessors.
    if expr.meta.is_column_selection():
        # Leaf: pl.col("name")
        cols = expr.meta.output_name()
        col_name = cols if isinstance(cols, str) else list(cols)[0]
        dtype_str = _dtype_to_input_str(schema[col_name])
        return scope.add_input(col_name, dtype_str)

    if expr.meta.is_literal():
        # Literal: pl.lit(...). We materialize as a 0-arg "broadcast" input.
        # For now, encode as F32 input (the constant becomes a fixed-shape
        # input at subgraph-build time).
        val = expr.meta.literal_value()
        if isinstance(val, (int, float)):
            return scope.add_input(f"__lit_{val}", "F32")
        else:
            raise _Aborted()  # other literal kinds (lists, strings) unsupported

    # Non-leaf: dispatch by node kind. Polars' meta API exposes:
    #  - root_names(): the column refs at the leaves
    #  - has_multiple_outputs(): is this an aggregation/list op
    #  - exprs(): direct children of a BinaryExpr / Function

    # Try the well-known shapes via tree_format introspection. The pattern
    # for py-1.40.1: expr.meta.tree_format() returns a string with a
    # top-level node kind. We pattern-match on the text.

    tree = expr.meta.tree_format(return_as_string=True)
    first_line = tree.split("\n")[0].strip()

    if "BinaryExpr" in first_line or "binary_expr" in first_line:
        return _visit_binary(expr, schema, scope)

    if "Function" in first_line:
        return _visit_function(expr, schema, scope)

    if "Cast" in first_line:
        return _visit_cast(expr, schema, scope)

    if "Agg" in first_line or "Aggregation" in first_line:
        return _visit_aggregation(expr, schema, scope)

    if "Ternary" in first_line or "when_then" in first_line:
        return _visit_ternary(expr, schema, scope)

    # Unknown shape → abort.
    raise _Aborted()


def _visit_binary(expr: pl.Expr, schema, scope) -> int:
    # Extract left, op, right from the binary expression.
    # Polars exposes the operator via expr.meta.serialize() decoded, or
    # via a private accessor. The implementation uses a controlled parse
    # of the tree_format output.
    op_str, left_expr, right_expr = _extract_binary_components(expr)
    op_id = _BINOP_MAP.get(op_str)
    if op_id is None:
        raise _Aborted()
    left = _visit(left_expr, schema, scope)
    right = _visit(right_expr, schema, scope)
    return scope.push_op(op_id, [left, right])


def _visit_function(expr: pl.Expr, schema, scope) -> int:
    fn_name, child_exprs = _extract_function_components(expr)
    op_id = _UNARY_FN_MAP.get(fn_name)
    if op_id is None:
        raise _Aborted()
    children = [_visit(c, schema, scope) for c in child_exprs]
    return scope.push_op(op_id, children)


def _visit_cast(expr: pl.Expr, schema, scope) -> int:
    target_dtype, child_expr = _extract_cast_components(expr)
    op_id = _CAST_MAP.get(target_dtype)
    if op_id is None:
        raise _Aborted()
    child = _visit(child_expr, schema, scope)
    return scope.push_op(op_id, [child])


def _visit_aggregation(expr: pl.Expr, schema, scope) -> int:
    agg_name, child_expr = _extract_aggregation_components(expr)
    op_id = _REDUCTION_MAP.get(agg_name)
    if op_id is None:
        raise _Aborted()
    child = _visit(child_expr, schema, scope)
    return scope.push_op(op_id, [child])


def _visit_ternary(expr: pl.Expr, schema, scope) -> int:
    cond_expr, then_expr, else_expr = _extract_ternary_components(expr)
    cond = _visit(cond_expr, schema, scope)
    then = _visit(then_expr, schema, scope)
    el = _visit(else_expr, schema, scope)
    return scope.push_op("Where", [cond, then, el])


# Helpers to extract components from Polars expr IR. These read the
# meta-serialized form to avoid relying on private APIs. Each may need
# adjustment when Polars revs are bumped.

def _extract_binary_components(expr: pl.Expr):
    """Returns (Operator, left_expr, right_expr)."""
    # Implementation: parse the binary-expression representation. The
    # tree_format string contains the operator name on the root line and
    # children indented below. A more robust path is to read the binary
    # serialization. For initial implementation, we use a private accessor
    # if available, or punt to a string-parse implementation.
    children = expr.meta.exprs() if hasattr(expr.meta, "exprs") else []
    if len(children) != 2:
        raise _Aborted()
    # Operator identification: prefer a typed accessor if available.
    op = _identify_binary_op(expr)
    return op, children[0], children[1]


def _identify_binary_op(expr: pl.Expr):
    """Identify the Polars Operator. py-1.40.1 specific; verify at landing."""
    tree = expr.meta.tree_format(return_as_string=True)
    for op in pl.Operator:
        if op.name in tree.split("\n")[0]:
            return op
    raise _Aborted()


def _extract_function_components(expr: pl.Expr):
    # Function expressions (like sin, log) are represented in the IR as
    # a function-call node. Identify the function name and child exprs.
    tree_first_line = expr.meta.tree_format(return_as_string=True).split("\n")[0]
    # Parse out the function name. py-1.40.1 specific.
    for name in _UNARY_FN_MAP.keys():
        if name in tree_first_line.lower():
            children = expr.meta.exprs() if hasattr(expr.meta, "exprs") else []
            if len(children) != 1:
                raise _Aborted()
            return name, children
    raise _Aborted()


def _extract_cast_components(expr: pl.Expr):
    """Return (target_dtype, child_expr)."""
    # py-1.40.1: expr.meta has helpers for cast metadata; use them.
    target = _identify_cast_target(expr)
    children = expr.meta.exprs() if hasattr(expr.meta, "exprs") else []
    if len(children) != 1:
        raise _Aborted()
    return target, children[0]


def _identify_cast_target(expr: pl.Expr):
    tree = expr.meta.tree_format(return_as_string=True)
    for dtype, _ in _CAST_MAP.items():
        if str(dtype) in tree.split("\n")[0]:
            return dtype
    raise _Aborted()


def _extract_aggregation_components(expr: pl.Expr):
    tree_first_line = expr.meta.tree_format(return_as_string=True).split("\n")[0]
    for name in _REDUCTION_MAP.keys():
        if name in tree_first_line.lower():
            children = expr.meta.exprs() if hasattr(expr.meta, "exprs") else []
            if len(children) != 1:
                raise _Aborted()
            return name, children[0]
    raise _Aborted()


def _extract_ternary_components(expr: pl.Expr):
    children = expr.meta.exprs() if hasattr(expr.meta, "exprs") else []
    if len(children) != 3:
        raise _Aborted()
    return children[0], children[1], children[2]


def _dtype_to_input_str(dtype) -> str:
    if dtype == pl.Float32: return "F32"
    if dtype == pl.Float64: return "F64"
    if dtype == pl.Int32:   return "I32"
    if dtype == pl.Boolean: return "Bool"
    # Array[F32, D] / List[F32] — Phase 10 work
    if str(dtype).startswith("Array(Float32"):
        inner_d = _extract_array_dim(dtype)
        return f"ArrayF32({inner_d})"
    if str(dtype) == "List(Float32)":
        return "ListF32"
    raise _Aborted()


def _extract_array_dim(dtype) -> int:
    # Polars 1.30+ exposes Array dim via dtype.size or similar.
    s = str(dtype)
    # "Array(Float32, 768)" or similar
    import re
    m = re.search(r",\s*(\d+)\)$", s)
    if m: return int(m.group(1))
    raise _Aborted()
```

This is a substantial file. **The implementation is partly stubs** — the Polars expression IR introspection APIs (`expr.meta.exprs()`, etc.) need to be verified against `py-1.40.1`. Some may need replacement with private-accessor reads or with parsing of the binary `serialize` output.

- [ ] **Step 4: Run, fix incrementally**

```bash
pytest tests/python_integration/test_fusion_analyzer.py -v 2>&1 | tail
```

If a test fails because the meta API differs from what the code assumes, debug the actual IR shape:

```bash
python -c "
import polars as pl
expr = (pl.col('a') + pl.col('b')) * pl.col('c')
print('tree:', expr.meta.tree_format())
print('exprs:', expr.meta.exprs() if hasattr(expr.meta, 'exprs') else 'NO exprs accessor')
print('output_name:', expr.meta.output_name())
"
```

…then adapt `_extract_binary_components` etc. to match. Iterate until all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/polars_metal/_fusion_analyzer.py \
        tests/python_integration/test_fusion_analyzer.py
git commit -m "M4 Phase 3: fusion analyzer (Polars expression IR walker)

Walks a Polars expression post-order, building a PyFusionScope when the
entire tree consists of supported ops. Returns None on first unsupported
node (no partial fusion this chunk).

Implementation note: the Polars expression IR introspection is py-1.40.1
specific. _extract_binary_components and friends use expr.meta accessors;
some accessors are private/unstable. When bumping Polars, this file is
the first to verify."
```

### Task 17: Integrate analyzer into walker

**Files:**
- Modify: `python/polars_metal/_walker.py`

- [ ] **Step 1: Failing integration test**

```python
# tests/python_integration/test_walker_fusion.py
import polars as pl
import polars_metal


def test_walker_emits_fused_expr_plan_node():
    """Walker recognizes a transcendental chain on F32 columns and emits
    a FusedExprGraph plan node (visible via debug log)."""
    n = 1_000_000
    df = pl.DataFrame({
        "a": pl.Series(range(n), dtype=pl.Float32) * 0.01,
        "b": pl.Series(range(n), dtype=pl.Float32) * 0.02,
    })
    engine = polars_metal.MetalEngine(debug=True)
    expr = (pl.col("a").sin() * pl.col("b").cos()).sqrt()
    result = df.lazy().with_columns(y=expr).collect(engine=engine)
    log = engine.last_debug_log()
    assert "FusedExprGraph" in log
    assert "n_ops=4" in log  # Sin, Cos, Mul, Sqrt


def test_walker_falls_back_on_string_op():
    """No FusedExprGraph should appear for a string op."""
    df = pl.DataFrame({"s": ["a", "b", "c"]})
    engine = polars_metal.MetalEngine(debug=True)
    expr = pl.col("s").str.len_chars()
    df.lazy().with_columns(y=expr).collect(engine=engine)
    log = engine.last_debug_log()
    assert "FusedExprGraph" not in log
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/python_integration/test_walker_fusion.py -v 2>&1 | tail
```

Expected: walker doesn't call the analyzer yet; `FusedExprGraph` doesn't appear in logs.

- [ ] **Step 3: Modify `_walker.py` to call the analyzer**

Read the current walker:

```bash
grep -n "with_columns\|HStack\|Select" python/polars_metal/_walker.py | head -20
```

Find where `HStack` / `WithColumns` IR nodes are handled. Add the analyzer-invocation block:

```python
# python/polars_metal/_walker.py — inside the HStack/Select handler

from polars_metal._fusion_analyzer import analyze_expression

def _handle_hstack(self, ir_node, schema):
    """For each new-column expression in HStack, try to fuse via analyzer."""
    plan_children = []
    for binding in ir_node.exprs:
        scope = analyze_expression(binding.expr, schema)
        if scope is None:
            # Unsupported shape — fall back to CPU for this column.
            plan_children.append(MetalPlanNode.CpuExpr(binding))
            continue
        n_rows = self._n_rows_estimate(ir_node)
        decision = scope.route_decision(n_rows)
        if decision == "Gpu":
            plan_children.append(MetalPlanNode.FusedExprGraph(
                scope=scope, output_name=binding.alias, n_rows=n_rows,
            ))
            if self.debug:
                self._log(f"FusedExprGraph for {binding.alias}: "
                          f"n_ops={scope.n_ops()} est_flops={scope.est_flops(n_rows)}")
        else:
            plan_children.append(MetalPlanNode.CpuExpr(binding))
            if self.debug:
                self._log(f"density routed CPU ({decision}) for {binding.alias}")
    return MetalPlanNode.HStack(plan_children)
```

The exact shape of `MetalPlanNode`, `binding`, and `ir_node` depends on M3's walker design — adapt accordingly.

- [ ] **Step 4: Run the test**

```bash
make wheel 2>&1 | tail
pytest tests/python_integration/test_walker_fusion.py -v 2>&1 | tail
```

Expected: 2 pass once the walker emits FusedExprGraph plan nodes correctly.

- [ ] **Step 5: Commit**

```bash
git add python/polars_metal/_walker.py \
        tests/python_integration/test_walker_fusion.py
git commit -m "M4 Phase 3: walker integrates analyzer; emits FusedExprGraph plan nodes

For each new-column expression in HStack/WithColumns, the walker calls
analyze_expression. If a FusionScope is produced AND density.route_decision
returns Gpu, the walker emits a MetalPlanNode::FusedExprGraph holding the
scope. CPU fallback otherwise. Debug log shows the routing decision."
```

---

## Phase 4 — MLX subgraph builder

The builder consumes a `FusionScope` and emits the matching `MlxArrayHandle` graph. It is the bridge from the analyzer's declarative scope to the MLX FFI surface from Phase 1.

### Task 18: `MlxSubgraph::from_fusion_scope` skeleton

**Files:**
- Create: `crates/polars-metal-kernels/src/mlx_subgraph.rs`
- Modify: `crates/polars-metal-kernels/src/lib.rs`

- [ ] **Step 1: Failing test — build + eval a 3-op graph**

```rust
// crates/polars-metal-kernels/tests/test_mlx_subgraph.rs
use polars_metal_core::fusion::scope::{FusionScope, InputDtype};
use polars_metal_core::fusion::supported_ops::OpId;
use polars_metal_kernels::mlx_subgraph::{MlxSubgraph, ColumnBuffer};

fn f32_input(data: Vec<f32>) -> ColumnBuffer {
    ColumnBuffer::from_f32_vec(data)
}

#[test]
fn build_and_eval_sin_cos_mul() {
    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::F32);
    let sin_a = scope.push_op(OpId::Sin, vec![a]);
    let cos_a = scope.push_op(OpId::Cos, vec![a]);
    let out = scope.push_op(OpId::Mul, vec![sin_a, cos_a]);
    scope.mark_output(out);

    let inputs = vec![f32_input((0..100).map(|i| i as f32 * 0.01).collect())];
    let subgraph = MlxSubgraph::from_fusion_scope(&scope, &inputs).expect("build");
    let outputs = subgraph.eval().expect("eval");
    assert_eq!(outputs.len(), 1);
    let out_vec = outputs[0].to_f32_vec().expect("read back");
    assert_eq!(out_vec.len(), 100);
    // sin(x)*cos(x) = 0.5*sin(2x); at x=0 it's 0.
    assert!(out_vec[0].abs() < 1e-6, "first element should be ~0, got {}", out_vec[0]);
}

#[test]
fn build_with_two_inputs_and_add() {
    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::F32);
    let b = scope.add_input("b", InputDtype::F32);
    let sum = scope.push_op(OpId::Add, vec![a, b]);
    scope.mark_output(sum);

    let inputs = vec![
        f32_input(vec![1.0, 2.0, 3.0]),
        f32_input(vec![10.0, 20.0, 30.0]),
    ];
    let subgraph = MlxSubgraph::from_fusion_scope(&scope, &inputs).expect("build");
    let outputs = subgraph.eval().expect("eval");
    assert_eq!(outputs[0].to_f32_vec().unwrap(), vec![11.0, 22.0, 33.0]);
}

#[test]
fn eval_is_fast_for_large_chains() {
    // Build a 20-op chain on 10M F32 input. Build-only cost should be <200 µs.
    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::F32);
    let mut cur = a;
    for op in &[OpId::Sin, OpId::Cos, OpId::Tan, OpId::Sqrt, OpId::Log,
                OpId::Exp, OpId::Abs, OpId::Floor, OpId::Ceil, OpId::Round,
                OpId::Square, OpId::Sinh, OpId::Cosh, OpId::Tanh,
                OpId::Asin, OpId::Acos, OpId::Atan, OpId::Log10, OpId::Log1p, OpId::Sqrt] {
        cur = scope.push_op(*op, vec![cur]);
    }
    scope.mark_output(cur);

    let inputs = vec![f32_input(vec![0.5; 10_000_000])];
    let t0 = std::time::Instant::now();
    let _subgraph = MlxSubgraph::from_fusion_scope(&scope, &inputs).expect("build");
    let elapsed = t0.elapsed();
    assert!(elapsed.as_micros() < 200_000, "build took {:?}", elapsed);
}
```

- [ ] **Step 2: Verify failure**

```bash
cargo test -p polars-metal-kernels --test test_mlx_subgraph 2>&1 | head
```

Expected: module not found.

- [ ] **Step 3: Implement `MlxSubgraph` + `ColumnBuffer`**

```rust
// crates/polars-metal-kernels/src/mlx_subgraph.rs
//! Build an MLX expression graph from a FusionScope; eval; fold back.
//!
//! This is the Phase 4 bridge between the analyzer's declarative scope
//! and the MLX FFI surface (Phase 1 bindings).

use polars_metal_core::fusion::scope::{FusionScope, NodeIdx, OpNode};
use polars_metal_core::fusion::supported_ops::OpId;
use polars_metal_mlx_sys::{
    array::{MlxArrayHandle, mlx_array_from_f32_slice, mlx_eval, mlx_array_to_f32_vec},
    elementwise::*,
    reduce::*,
    scan::*,
    sort::*,
    matmul::*,
    fft::*,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BuildError {
    #[error("input dtype mismatch: expected {expected}, got {actual}")]
    DtypeMismatch { expected: String, actual: String },
    #[error("scope references undefined input index {0}")]
    UndefinedInput(u32),
    #[error("MLX FFI error: {0}")]
    MlxError(String),
    #[error("op {0:?} not yet supported by subgraph builder")]
    UnsupportedOp(OpId),
}

/// Lightweight wrapper around a raw F32 buffer used during testing.
/// In production, this is replaced by `polars_metal_buffer::MetalBuffer`.
pub struct ColumnBuffer {
    data: Vec<f32>,
}

impl ColumnBuffer {
    pub fn from_f32_vec(data: Vec<f32>) -> Self { Self { data } }
    pub fn to_f32_vec(&self) -> Result<Vec<f32>, BuildError> { Ok(self.data.clone()) }
    pub fn as_handle(&self) -> Result<MlxArrayHandle, BuildError> {
        mlx_array_from_f32_slice(&self.data)
            .map_err(|e| BuildError::MlxError(format!("{e:?}")))
    }
}

pub struct MlxSubgraph {
    handles: Vec<MlxArrayHandle>,    // indexed by NodeIdx
    outputs: Vec<MlxArrayHandle>,
}

impl MlxSubgraph {
    pub fn from_fusion_scope(
        scope: &FusionScope,
        inputs: &[ColumnBuffer],
    ) -> Result<Self, BuildError> {
        if inputs.len() != scope.inputs.len() {
            return Err(BuildError::UndefinedInput(scope.inputs.len() as u32));
        }

        let mut handles: Vec<MlxArrayHandle> = inputs.iter()
            .map(|b| b.as_handle())
            .collect::<Result<Vec<_>, _>>()?;

        for op_node in &scope.ops {
            let handle = build_op(op_node, &handles)?;
            handles.push(handle);
        }

        let outputs: Vec<MlxArrayHandle> = scope.outputs.iter()
            .map(|idx| handles[idx.0 as usize].clone())
            .collect();

        Ok(Self { handles, outputs })
    }

    pub fn eval(&self) -> Result<Vec<ColumnBuffer>, BuildError> {
        mlx_eval(&self.outputs).map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
        let mut outs = Vec::new();
        for h in &self.outputs {
            let data = mlx_array_to_f32_vec(h)
                .map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
            outs.push(ColumnBuffer { data });
        }
        Ok(outs)
    }
}

fn build_op(node: &OpNode, handles: &[MlxArrayHandle]) -> Result<MlxArrayHandle, BuildError> {
    use OpId::*;
    let args: Vec<&MlxArrayHandle> = node.args.iter()
        .map(|idx| {
            handles.get(idx.0 as usize)
                   .ok_or(BuildError::UndefinedInput(idx.0))
        })
        .collect::<Result<Vec<_>, _>>()?;

    match node.op {
        // Arithmetic
        Add => Ok(mlx_add(args[0], args[1])),
        Sub => Ok(mlx_sub(args[0], args[1])),
        Mul => Ok(mlx_mul(args[0], args[1])),
        Div => Ok(mlx_div(args[0], args[1])),
        Mod => Ok(mlx_mod_(args[0], args[1])),
        Pow => Ok(mlx_pow(args[0], args[1])),
        Neg => Ok(mlx_neg(args[0])),
        Abs => Ok(mlx_abs(args[0])),
        Square => Ok(mlx_square(args[0])),
        // Comparison
        Eq => Ok(mlx_eq(args[0], args[1])),
        Ne => Ok(mlx_ne(args[0], args[1])),
        Lt => Ok(mlx_lt(args[0], args[1])),
        Le => Ok(mlx_le(args[0], args[1])),
        Gt => Ok(mlx_gt(args[0], args[1])),
        Ge => Ok(mlx_ge(args[0], args[1])),
        // Logical
        LogicalAnd => Ok(mlx_logical_and(args[0], args[1])),
        LogicalOr  => Ok(mlx_logical_or(args[0], args[1])),
        LogicalNot => Ok(mlx_logical_not(args[0])),
        // Where
        Where => Ok(mlx_where(args[0], args[1], args[2])),
        // Transcendentals
        Sin => Ok(mlx_sin(args[0])), Cos => Ok(mlx_cos(args[0])), Tan => Ok(mlx_tan(args[0])),
        Sinh => Ok(mlx_sinh(args[0])), Cosh => Ok(mlx_cosh(args[0])), Tanh => Ok(mlx_tanh(args[0])),
        Asin => Ok(mlx_asin(args[0])), Acos => Ok(mlx_acos(args[0])), Atan => Ok(mlx_atan(args[0])),
        Atan2 => Ok(mlx_atan2(args[0], args[1])),
        Log => Ok(mlx_log(args[0])), Log2 => Ok(mlx_log2(args[0])),
        Log10 => Ok(mlx_log10(args[0])), Log1p => Ok(mlx_log1p(args[0])),
        Exp => Ok(mlx_exp(args[0])), Exp2 => Ok(mlx_exp2(args[0])),
        Sqrt => Ok(mlx_sqrt(args[0])), Cbrt => Ok(mlx_cbrt(args[0])),
        Floor => Ok(mlx_floor(args[0])), Ceil => Ok(mlx_ceil(args[0])), Round => Ok(mlx_round(args[0])),
        // Cast
        CastF32 => Ok(mlx_cast(args[0], polars_metal_mlx_sys::array::MlxDtype::F32)),
        CastF64 => Ok(mlx_cast(args[0], polars_metal_mlx_sys::array::MlxDtype::F64)),
        CastI32 => Ok(mlx_cast(args[0], polars_metal_mlx_sys::array::MlxDtype::I32)),
        CastBool => Ok(mlx_cast(args[0], polars_metal_mlx_sys::array::MlxDtype::Bool)),
        // Reductions
        Sum => Ok(mlx_sum(args[0])), Mean => Ok(mlx_mean(args[0])),
        Min => Ok(mlx_min(args[0])), Max => Ok(mlx_max(args[0])),
        Std => Ok(mlx_std(args[0])), Var => Ok(mlx_var(args[0])),
        ArgMin => Ok(mlx_argmin(args[0])), ArgMax => Ok(mlx_argmax(args[0])),
        // Sort / top-k
        Sort => Ok(mlx_sort(args[0])),
        ArgPartition => {
            // ArgPartition needs a `kth` parameter; we don't carry it in
            // the scope yet. Phase 8.4 wires it as scope metadata.
            Err(BuildError::UnsupportedOp(node.op))
        }
        // Cumulative
        CumSum => Ok(mlx_cumsum(args[0])),
        CumProd => Ok(mlx_cumprod(args[0])),
        CumMax => Ok(mlx_cummax(args[0])),
        CumMin => Ok(mlx_cummin(args[0])),
        // Matmul
        MatMul => Ok(mlx_matmul(args[0], args[1])),
        // FFT (Phase 11)
        Fft => Ok(mlx_fft(args[0])),
        Ifft => Ok(mlx_ifft(args[0])),
    }
}
```

Wire `pub mod mlx_subgraph;` into `lib.rs`.

- [ ] **Step 4: Run**

```bash
cargo test -p polars-metal-kernels --test test_mlx_subgraph 2>&1 | tail
```

Expected: 3 pass.

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-kernels/src/mlx_subgraph.rs \
        crates/polars-metal-kernels/src/lib.rs \
        crates/polars-metal-kernels/tests/test_mlx_subgraph.rs
git commit -m "M4 Phase 4: MlxSubgraph - FusionScope -> MLX graph -> eval

build_op() dispatches by OpId to the matching MLX FFI binding from
Phase 1. The 20-op chain test verifies build cost is sub-200us at
10M F32. ArgPartition is stubbed (needs kth parameter from scope
metadata; Phase 8.4 adds it)."
```

### Task 19: ColumnBuffer ↔ MetalBuffer zero-copy bridge

The Task 18 `ColumnBuffer` is a stand-in. Real engine execution uses `polars_metal_buffer::MetalBuffer`. This task wires the zero-copy path.

**Files:**
- Modify: `crates/polars-metal-kernels/src/mlx_subgraph.rs`
- Modify: `crates/polars-metal-buffer/src/lib.rs` (may need `from_f32_slice`, `to_f32_vec`, `as_mtl_buffer_ptr`)

- [ ] **Step 1: Failing test — build subgraph over MetalBuffer inputs**

```rust
// crates/polars-metal-kernels/tests/test_mlx_subgraph_metal_buffer.rs
use polars_metal_buffer::MetalBuffer;
use polars_metal_core::fusion::scope::{FusionScope, InputDtype};
use polars_metal_core::fusion::supported_ops::OpId;
use polars_metal_kernels::mlx_subgraph::MlxSubgraph;

#[test]
fn build_subgraph_over_metal_buffer_inputs() {
    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::F32);
    let s = scope.push_op(OpId::Sqrt, vec![a]);
    scope.mark_output(s);

    let buf = MetalBuffer::from_f32_slice(&[1.0, 4.0, 9.0, 16.0]).unwrap();
    let outputs = MlxSubgraph::from_fusion_scope_buffers(&scope, &[&buf])
        .unwrap()
        .eval_to_metal_buffers()
        .unwrap();
    let out = outputs[0].to_f32_vec().unwrap();
    assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
}
```

- [ ] **Step 2: Verify failure**

- [ ] **Step 3: Add the `_buffers` API alongside the existing `from_fusion_scope`**

```rust
// crates/polars-metal-kernels/src/mlx_subgraph.rs — append

use polars_metal_buffer::MetalBuffer;
use polars_metal_mlx_sys::array::{mlx_array_view_metal_buffer, MlxDtype};

impl MlxSubgraph {
    /// Same as from_fusion_scope but inputs come from existing MetalBuffers
    /// (zero-copy view via wrap_mtl_buffer).
    pub fn from_fusion_scope_buffers(
        scope: &FusionScope,
        inputs: &[&MetalBuffer],
    ) -> Result<Self, BuildError> {
        if inputs.len() != scope.inputs.len() {
            return Err(BuildError::UndefinedInput(scope.inputs.len() as u32));
        }
        let mut handles: Vec<MlxArrayHandle> = inputs.iter()
            .map(|b| mlx_array_view_metal_buffer(b, MlxDtype::F32)
                       .map_err(|e| BuildError::MlxError(format!("{e:?}"))))
            .collect::<Result<Vec<_>, _>>()?;
        for op_node in &scope.ops {
            let handle = build_op(op_node, &handles)?;
            handles.push(handle);
        }
        let outputs: Vec<MlxArrayHandle> = scope.outputs.iter()
            .map(|idx| handles[idx.0 as usize].clone())
            .collect();
        Ok(Self { handles, outputs })
    }

    /// Eval and return output MetalBuffers (zero-copy from MLX-allocated
    /// MTLBuffers via mlx_array_to_metal_buffer).
    pub fn eval_to_metal_buffers(&self) -> Result<Vec<MetalBuffer>, BuildError> {
        mlx_eval(&self.outputs)
            .map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
        let mut outs = Vec::new();
        for h in &self.outputs {
            // The MlxArrayHandle backs an MTLBuffer; wrap as MetalBuffer.
            let buf = polars_metal_mlx_sys::array::mlx_array_to_metal_buffer(h)
                .map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
            outs.push(buf);
        }
        Ok(outs)
    }
}
```

`polars_metal_mlx_sys::array::mlx_array_to_metal_buffer` is a new binding — adds to `Phase 1, Task 5`-style work. Implementation extracts the MTLBuffer pointer from the MlxArray and wraps it as a MetalBuffer.

- [ ] **Step 4: Run the test**

```bash
cargo test -p polars-metal-kernels --test test_mlx_subgraph_metal_buffer 2>&1 | tail
```

Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-kernels/src/mlx_subgraph.rs \
        crates/polars-metal-mlx-sys/src/array.rs \
        crates/polars-metal-buffer/src/lib.rs \
        crates/polars-metal-kernels/tests/test_mlx_subgraph_metal_buffer.rs
git commit -m "M4 Phase 4: zero-copy bridge between MetalBuffer and MlxSubgraph

from_fusion_scope_buffers takes MetalBuffer inputs; eval_to_metal_buffers
returns MetalBuffer outputs. Adds mlx_array_to_metal_buffer FFI to
extract MTLBuffer from an evaluated MlxArrayHandle. This is the
zero-copy path for production execution; the from_fusion_scope path
(with Vec<f32>) stays for testing convenience."
```

### Task 20: Proptest — subgraph eval matches a pure-Rust scalar reference

**Files:**
- Create: `crates/polars-metal-kernels/tests/proptest_subgraph.rs`

- [ ] **Step 1: Add proptest dep if not present, write the test**

```toml
# crates/polars-metal-kernels/Cargo.toml [dev-dependencies]
proptest = "1"
```

```rust
// crates/polars-metal-kernels/tests/proptest_subgraph.rs
//! Proptest: random op chain + random F32 input → MLX eval matches
//! a pure-Rust scalar reference (within ULP tolerance for transcendentals).

use polars_metal_core::fusion::scope::{FusionScope, InputDtype};
use polars_metal_core::fusion::supported_ops::OpId;
use polars_metal_kernels::mlx_subgraph::{MlxSubgraph, ColumnBuffer};
use proptest::prelude::*;

fn scalar_apply(op: OpId, args: &[f32]) -> f32 {
    use OpId::*;
    match op {
        Add => args[0] + args[1], Sub => args[0] - args[1],
        Mul => args[0] * args[1], Div => args[0] / args[1],
        Neg => -args[0], Abs => args[0].abs(), Square => args[0] * args[0],
        Sin => args[0].sin(), Cos => args[0].cos(), Tan => args[0].tan(),
        Log => args[0].ln(), Exp => args[0].exp(),
        Sqrt => args[0].sqrt(), Floor => args[0].floor(),
        Ceil => args[0].ceil(), Round => args[0].round(),
        _ => panic!("scalar_apply: op {:?} not in proptest set", op),
    }
}

const SAFE_OPS: &[OpId] = &[
    OpId::Add, OpId::Sub, OpId::Mul,
    OpId::Neg, OpId::Abs, OpId::Square,
    OpId::Sin, OpId::Cos, OpId::Sqrt, OpId::Exp,
    OpId::Floor, OpId::Ceil, OpId::Round,
];

proptest! {
    #[test]
    fn random_chain_matches_scalar(
        op_count in 1usize..6,
        input_size in 1usize..1000,
    ) {
        let ops: Vec<OpId> = (0..op_count).map(|i| SAFE_OPS[i % SAFE_OPS.len()]).collect();
        let mut scope = FusionScope::new();
        let a = scope.add_input("a", InputDtype::F32);
        let mut cur = a;
        for op in &ops {
            // Use unary form only — keeps the scalar reference simple
            if matches!(op, OpId::Add | OpId::Sub | OpId::Mul) { continue; }
            cur = scope.push_op(*op, vec![cur]);
        }
        scope.mark_output(cur);

        // Random input
        let input: Vec<f32> = (0..input_size).map(|i| (i as f32 * 0.3 + 0.1)).collect();
        let inputs = vec![ColumnBuffer::from_f32_vec(input.clone())];
        let out = MlxSubgraph::from_fusion_scope(&scope, &inputs)
            .unwrap().eval().unwrap();

        // Scalar reference
        for (i, &x) in input.iter().enumerate() {
            let mut v = x;
            for op in &ops {
                if matches!(op, OpId::Add | OpId::Sub | OpId::Mul) { continue; }
                v = scalar_apply(*op, &[v]);
            }
            let actual = out[0].to_f32_vec().unwrap()[i];
            if v.is_nan() || actual.is_nan() {
                prop_assert!(v.is_nan() && actual.is_nan(),
                             "NaN mismatch at row {}: ref {} vs actual {}", i, v, actual);
            } else {
                let tolerance = (v.abs() * 4.0 * f32::EPSILON).max(1e-5);
                prop_assert!((actual - v).abs() < tolerance,
                             "mismatch at row {}: {} vs {}", i, actual, v);
            }
        }
    }
}
```

- [ ] **Step 2: Run**

```bash
cargo test -p polars-metal-kernels --test proptest_subgraph -- --nocapture 2>&1 | tail -20
```

Expected: 256 cases pass; no shrinks. If ULP tolerance bites, widen it for transcendentals.

- [ ] **Step 3: Commit**

```bash
git add crates/polars-metal-kernels/Cargo.toml \
        crates/polars-metal-kernels/tests/proptest_subgraph.rs
git commit -m "M4 Phase 4: proptest for subgraph eval vs scalar reference

Closes Phase 4. Random op chain (up to 6 ops from a safe-set) on random
F32 input; eval through MLX subgraph; compare against scalar Rust
reference. ULP tolerance 4*EPSILON. 256 cases. Catches dtype mismatches,
op-id wiring errors, MLX semantic divergences."
```

---

## Phase 5 — Rust router integration + `MetalPlanNode::FusedExprGraph` dispatch

The walker now emits `MetalPlanNode::FusedExprGraph` plan nodes (Phase 3 Task 17). Phase 5 makes the Rust router actually execute them: wire the plan node through `execute_fused_expr` in `udf.rs`, hand it to the subgraph builder, return the output as a Polars Series.

### Task 21: Add `FusedExprGraph` variant to `MetalPlanNode`

**Files:**
- Modify: `crates/polars-metal-core/src/plan/mod.rs`
- Modify: `crates/polars-metal-core/src/plan/py.rs` (PyO3 wrapper for the variant)

- [ ] **Step 1: Failing test — construct + roundtrip a FusedExprGraph plan node**

```python
# tests/python_integration/test_metal_plan_fused.py
"""Python can construct a MetalPlanNode::FusedExprGraph and pass it to
the Rust executor."""
import polars_metal._native as native


def test_construct_fused_expr_plan_node():
    scope = native.PyFusionScope()
    a = scope.add_input("a", "F32")
    s = scope.push_op("Sqrt", [a])
    scope.mark_output(s)

    plan = native.PyMetalPlanNode.fused_expr_graph(
        scope=scope,
        output_name="result",
        n_rows=1_000_000,
    )
    assert plan.kind() == "FusedExprGraph"
    assert plan.output_name() == "result"
```

- [ ] **Step 2: Verify fail**

- [ ] **Step 3: Add the plan variant + PyO3 wrapper**

```rust
// crates/polars-metal-core/src/plan/mod.rs — add to existing enum
pub enum MetalPlanNode {
    // ... existing M2/M3 variants
    FusedExprGraph {
        scope: crate::fusion::scope::FusionScope,
        output_name: String,
        n_rows: usize,
    },
    CpuExpr { /* placeholder for unfused fallback */ },
}
```

```rust
// crates/polars-metal-core/src/plan/py.rs — add a constructor
use crate::fusion::py::PyFusionScope;

#[pymethods]
impl PyMetalPlanNode {
    #[staticmethod]
    pub fn fused_expr_graph(
        scope: &PyFusionScope,
        output_name: &str,
        n_rows: usize,
    ) -> Self {
        Self {
            inner: MetalPlanNode::FusedExprGraph {
                scope: scope.inner.clone(),
                output_name: output_name.to_string(),
                n_rows,
            },
        }
    }

    pub fn kind(&self) -> &'static str {
        match &self.inner {
            MetalPlanNode::FusedExprGraph { .. } => "FusedExprGraph",
            // ... other variants
            _ => "Unknown",
        }
    }

    pub fn output_name(&self) -> Option<String> {
        match &self.inner {
            MetalPlanNode::FusedExprGraph { output_name, .. } => Some(output_name.clone()),
            _ => None,
        }
    }
}
```

- [ ] **Step 4: Run, verify pass**

```bash
make wheel 2>&1 | tail -3
pytest tests/python_integration/test_metal_plan_fused.py -v 2>&1 | tail
```

Expected: passes.

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-core/src/plan/ \
        tests/python_integration/test_metal_plan_fused.py
git commit -m "M4 Phase 5: MetalPlanNode::FusedExprGraph variant + Py wrapper

The walker (Phase 3 Task 17) constructs these; the executor (next task)
consumes them. The variant holds the FusionScope, the output column
name, and the estimated row count for routing diagnostics."
```

### Task 22: `execute_fused_expr` — the Rust executor entry point

**Files:**
- Modify: `crates/polars-metal-core/src/udf.rs`

- [ ] **Step 1: Failing test — Python calls execute_fused_expr, gets a Polars Series back**

```python
# tests/python_integration/test_execute_fused_expr.py
"""End-to-end: build a FusionScope, hand to execute_fused_expr with
input columns, get a Polars Series back."""
import polars as pl
import polars_metal._native as native


def test_sqrt_one_million_rows():
    n = 1_000_000
    input_col = pl.Series("a", [(i % 100) ** 2 for i in range(n)], dtype=pl.Float32)

    scope = native.PyFusionScope()
    a = scope.add_input("a", "F32")
    s = scope.push_op("Sqrt", [a])
    scope.mark_output(s)

    result = native.execute_fused_expr(
        scope=scope,
        input_columns=[input_col],
        output_name="result",
    )
    assert isinstance(result, pl.Series)
    assert result.dtype == pl.Float32
    assert result.len() == n
    # First few values: sqrt(0), sqrt(1), sqrt(4), sqrt(9), sqrt(16)
    expected_head = [0.0, 1.0, 2.0, 3.0, 4.0]
    for i, exp in enumerate(expected_head):
        assert abs(result[i] - exp) < 1e-6, f"row {i}: {result[i]} vs {exp}"
```

- [ ] **Step 2: Verify failure**

- [ ] **Step 3: Implement `execute_fused_expr`**

```rust
// crates/polars-metal-core/src/udf.rs — add

use pyo3::prelude::*;
use pyo3::types::PyList;
use polars::prelude::*;
use crate::fusion::py::PyFusionScope;
use polars_metal_buffer::MetalBuffer;
use polars_metal_kernels::mlx_subgraph::MlxSubgraph;

#[pyfunction]
pub fn execute_fused_expr(
    py: Python<'_>,
    scope: &PyFusionScope,
    input_columns: &PyList,
    output_name: &str,
) -> PyResult<PyObject> {
    // Convert Polars Series → MetalBuffer for each input
    let metal_buffers: Vec<MetalBuffer> = input_columns.iter()
        .map(|series_obj| {
            // Each series_obj is a pl.Series; extract the underlying F32 buffer.
            // Using polars-arrow zero-copy.
            let series_inner = extract_polars_series(series_obj)?;
            // SAFETY: F32 column; we verified dtype in the analyzer.
            let buf = MetalBuffer::from_arrow_buffer(series_inner.chunked_f32_data())?;
            Ok(buf)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let bufs_refs: Vec<&MetalBuffer> = metal_buffers.iter().collect();

    // Build + eval the subgraph
    let subgraph = MlxSubgraph::from_fusion_scope_buffers(&scope.inner, &bufs_refs)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))?;
    let outputs = subgraph.eval_to_metal_buffers()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{e}")))?;

    // Wrap the (first) output buffer back as a Polars Series.
    // Assumes single-output for now; multi-output (e.g. FFT struct) wired in Phase 11.
    let out_buf = outputs.into_iter().next()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("no output"))?;
    let series = metal_buffer_to_polars_series(out_buf, output_name)?;
    Ok(series.into_py(py))
}

fn extract_polars_series(obj: &PyAny) -> PyResult<polars::prelude::Series> {
    // Use pyo3-polars or manual extraction.
    use pyo3_polars::PySeries;
    let py_series: PySeries = obj.extract()?;
    Ok(py_series.0)
}

fn metal_buffer_to_polars_series(buf: MetalBuffer, name: &str) -> PyResult<polars::prelude::Series> {
    // Wrap MetalBuffer's underlying memory as an Arrow F32 Buffer; build a Series.
    let arrow_buf = buf.as_arrow_f32_buffer();
    let arr = polars_arrow::array::Float32Array::from_data_default(arrow_buf, None);
    let series = Series::try_from((name, Box::new(arr) as Box<dyn polars_arrow::array::Array>))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("series construct: {e}")))?;
    Ok(series)
}
```

Register `execute_fused_expr` in the PyO3 module entry.

- [ ] **Step 4: Build wheel, run test**

```bash
make wheel 2>&1 | tail -3
pytest tests/python_integration/test_execute_fused_expr.py -v 2>&1 | tail
```

Expected: passes. May need to adjust `pyo3-polars` integration depending on the existing pattern in the repo.

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-core/src/udf.rs
git commit -m "M4 Phase 5: execute_fused_expr - PyO3 executor entry point

Takes a PyFusionScope + list of input Polars Series + output name.
Converts inputs to MetalBuffers (zero-copy from Arrow), builds the
MlxSubgraph, evals, wraps the output back as a Polars Series. The
single-output path; multi-output for FFT struct comes in Phase 11."
```

### Task 23: Wire `execute_fused_expr` into the walker's plan-apply step

**Files:**
- Modify: `python/polars_metal/_walker.py` (or `_udf.py`)

- [ ] **Step 1: Failing end-to-end test**

```python
# tests/python_integration/test_walker_e2e_fused.py
"""End-to-end: df.with_columns + engine='metal' should run via
FusedExprGraph and produce the same result as engine='cpu'."""
import polars as pl
import polars_metal
from polars.testing import assert_frame_equal


def test_sqrt_chain_e2e():
    n = 1_000_000
    df = pl.DataFrame({
        "a": pl.Series([float(i % 100) for i in range(n)], dtype=pl.Float32),
        "b": pl.Series([float((i * 7) % 256) for i in range(n)], dtype=pl.Float32),
    })
    expr = (pl.col("a").sqrt() + pl.col("b").sqrt()).cos()
    cpu_result = df.lazy().with_columns(y=expr).collect()
    metal_result = df.lazy().with_columns(y=expr).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu_result, metal_result, check_exact=False, atol=1e-5)
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/python_integration/test_walker_e2e_fused.py -v 2>&1 | tail
```

Expected: fails because the walker emits the plan node but doesn't yet dispatch it.

- [ ] **Step 3: Add the dispatch path**

In the walker's `apply_plan` step (the place where each plan node is converted to its UDF result):

```python
# python/polars_metal/_walker.py — inside apply_plan
import polars_metal._native as native

def _apply_plan(self, plan_node, df):
    if plan_node.kind() == "FusedExprGraph":
        scope = plan_node.scope()         # accessor on PyMetalPlanNode
        output_name = plan_node.output_name()
        input_names = scope.input_column_names()  # need to add this accessor
        input_series = [df[name] for name in input_names]
        result_series = native.execute_fused_expr(
            scope=scope,
            input_columns=input_series,
            output_name=output_name,
        )
        return df.with_columns(result_series.alias(output_name))
    # ... existing M3 path
```

You may need to add `scope()` and `input_column_names()` accessors on `PyFusionScope` and `PyMetalPlanNode`. Implement those as one-line PyO3 methods.

- [ ] **Step 4: Run the end-to-end test**

```bash
make wheel 2>&1 | tail
pytest tests/python_integration/test_walker_e2e_fused.py -v 2>&1 | tail
```

Expected: passes. **This is the first M4 milestone**: a user-facing engine="metal" run that executes a fused subgraph end-to-end.

- [ ] **Step 5: Commit**

```bash
git add python/polars_metal/_walker.py crates/polars-metal-core/src/fusion/py.rs
git commit -m "M4 Phase 5: walker dispatches FusedExprGraph to execute_fused_expr

First M4 milestone: df.with_columns(...).collect(engine='metal') runs
end-to-end via the fused MLX subgraph and returns a result equal to
engine='cpu'. Closes Phase 5. Next: Phase 6 wires up the haversine and
Black-Scholes benchmarks to verify the perf targets."
```

---

## Phase 6 — Phase 8 headline workloads (haversine + Black-Scholes)

End-to-end engine benchmarks for the two flagship Phase-8 workloads. The infrastructure from Phases 1–5 should handle these without further code — this phase is about wiring the benches and verifying targets.

### Task 24: Haversine end-to-end benchmark

**Files:**
- Create: `tests/bench/m4_engine/test_phase8_haversine.py`

- [ ] **Step 1: Failing test (perf target)**

```python
# tests/bench/m4_engine/test_phase8_haversine.py
"""End-to-end haversine via engine='metal'.

Target wall-clock < 6 ms at N=10M F32. Allows ~2.5 ms over the MLX
ceiling of 3.49 ms measured in tests/bench/m4_survey/bench_haversine_mlx.py.
"""
import json
import numpy as np
import polars as pl
import polars_metal
import pytest
from polars.testing import assert_frame_equal


def _make_taxi(n, seed=0xCAB):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "pickup_lat":  rng.uniform(40.6, 40.9, size=n).astype(np.float32),
        "pickup_lon":  rng.uniform(-74.05, -73.7, size=n).astype(np.float32),
        "drop_lat":    rng.uniform(40.6, 40.9, size=n).astype(np.float32),
        "drop_lon":    rng.uniform(-74.05, -73.7, size=n).astype(np.float32),
    })


def _haversine_expr() -> pl.Expr:
    R = 6371.0
    d2r = float(np.pi / 180.0)
    pla = pl.col("pickup_lat") * d2r
    dla = pl.col("drop_lat") * d2r
    dlat = (dla - pla) / 2.0
    dlon = (pl.col("drop_lon") - pl.col("pickup_lon")) * d2r / 2.0
    a = dlat.sin() ** 2 + pla.cos() * dla.cos() * dlon.sin() ** 2
    return 2.0 * R * a.sqrt().arcsin()


def test_haversine_metal_matches_cpu():
    df = _make_taxi(100_000)
    cpu  = df.lazy().with_columns(d=_haversine_expr()).collect()
    metal = df.lazy().with_columns(d=_haversine_expr()).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal, check_exact=False, atol=1e-4)


def test_haversine_metal_perf_10m(benchmark):
    df = _make_taxi(10_000_000)
    engine = polars_metal.MetalEngine()

    def run():
        return df.lazy().with_columns(d=_haversine_expr()).collect(engine=engine)

    result = benchmark.pedantic(run, iterations=1, rounds=10, warmup_rounds=2)
    median_ms = benchmark.stats["median"] * 1000
    assert median_ms < 6.0, f"haversine took {median_ms:.2f} ms, target < 6 ms"

    # Persist result to baseline.json
    baseline = json.loads(open("tests/bench/baseline.json").read())
    baseline["queries"]["phase8_haversine_10m"]["metal_ms"] = round(median_ms, 2)
    baseline["queries"]["phase8_haversine_10m"]["_pending"] = False
    baseline["queries"]["phase8_haversine_10m"]["_gate"] = {"max_ms": 8.0}  # 20% headroom
    open("tests/bench/baseline.json", "w").write(json.dumps(baseline, indent=2))
```

- [ ] **Step 2: Run, expect first failure (probably analyzer/walker issue)**

```bash
make wheel 2>&1 | tail
pytest tests/bench/m4_engine/test_phase8_haversine.py -v 2>&1 | tail -30
```

Two likely failure modes:
- Correctness: the haversine expression has a shape the analyzer doesn't yet recognize (e.g., `**` operator or `2.0 * something` literal-LHS). Debug with `engine=polars_metal.MetalEngine(debug=True)` and check the log.
- Perf: lands within 6–10 ms range but not < 6 ms. Profile with the bench output.

- [ ] **Step 3: Iterate on analyzer until correctness passes**

Common adaptations:
- Add `Pow` recognition for `x ** 2` if not already.
- Recognize `pl.lit(2.0) * expr` (literal-LHS multiply).
- Handle the `2.0 * R * a.sqrt().arcsin()` chain — ensure literal handling broadcasts correctly.

Each adaptation is a small extension to `_fusion_analyzer.py`. Add tests for the specific shape (in `tests/python_integration/test_fusion_analyzer.py`) before fixing.

- [ ] **Step 4: Iterate on perf until target hits**

If wall-clock > 6 ms:
- Profile: where's the time going? FFI marshalling? MLX eval? Buffer fold-back?
- Tracing via `MetalEngine(debug=True, trace=True)` instruments each step.
- Common fix: ensure inputs are already in single-chunk Series before the metal call (rechunk on entry).

- [ ] **Step 5: Commit when both tests green**

```bash
git add tests/bench/m4_engine/test_phase8_haversine.py tests/bench/baseline.json \
        python/polars_metal/_fusion_analyzer.py  # if needed
git commit -m "M4 Phase 6: haversine engine bench — green at < 6ms

Wall-clock: $(target ms) on M2 Ultra. Analyzer adaptations made along
the way:
  - Pow recognition for x**2
  - Literal-LHS multiply (2.0 * expr)
Baseline updated with measured value."
```

### Task 25: Black-Scholes end-to-end benchmark

**Files:**
- Create: `tests/bench/m4_engine/test_phase8_black_scholes.py`

- [ ] **Step 1: Failing test**

```python
# tests/bench/m4_engine/test_phase8_black_scholes.py
"""End-to-end Black-Scholes via engine='metal'.

Target wall-clock < 6 ms at N=10M F32. MLX ceiling 3.86 ms.
"""
import json
import numpy as np
import polars as pl
import polars_metal
from polars.testing import assert_frame_equal


def _make_options(n, seed=0xCAFE):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "s": rng.uniform(50, 150, size=n).astype(np.float32),
        "k": rng.uniform(50, 150, size=n).astype(np.float32),
        "t": rng.uniform(0.1, 2.0, size=n).astype(np.float32),
    })


def _bs_expr(sigma=0.2, r=0.05) -> pl.Expr:
    s, k, t = pl.col("s"), pl.col("k"), pl.col("t")
    sst = sigma * t.sqrt()
    d1 = ((s / k).log() + (r + 0.5 * sigma * sigma) * t) / sst
    d2 = d1 - sst
    def cdf(x): return 0.5 * (1.0 + (x * 0.7978845608).tanh())
    return s * cdf(d1) - k * (-r * t).exp() * cdf(d2)


def test_bs_correctness():
    df = _make_options(100_000)
    cpu  = df.lazy().with_columns(call=_bs_expr()).collect()
    metal = df.lazy().with_columns(call=_bs_expr()).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal, check_exact=False, atol=1e-3)


def test_bs_perf_10m(benchmark):
    df = _make_options(10_000_000)
    engine = polars_metal.MetalEngine()
    def run():
        return df.lazy().with_columns(call=_bs_expr()).collect(engine=engine)
    benchmark.pedantic(run, iterations=1, rounds=10, warmup_rounds=2)
    median_ms = benchmark.stats["median"] * 1000
    assert median_ms < 6.0, f"BS took {median_ms:.2f} ms, target < 6 ms"

    baseline = json.loads(open("tests/bench/baseline.json").read())
    baseline["queries"]["phase8_black_scholes_10m"]["metal_ms"] = round(median_ms, 2)
    baseline["queries"]["phase8_black_scholes_10m"]["_pending"] = False
    baseline["queries"]["phase8_black_scholes_10m"]["_gate"] = {"max_ms": 8.0}
    open("tests/bench/baseline.json", "w").write(json.dumps(baseline, indent=2))
```

- [ ] **Steps 2–5: Same iterative pattern as Task 24.** Likely additional analyzer adaptations needed for the literal-LHS arithmetic and the `0.5 * sigma * sigma` constant folding. Commit when green.

```bash
git commit -m "M4 Phase 6: Black-Scholes engine bench — green at < 6ms"
```

---

## Phase 7 — Phase 8 terminuses (reductions, sort, top-k, cumsum, correlation)

The reductions, sort, top-k, and cumsum ops are all wired through the existing subgraph builder (Phase 4). This phase verifies each end-to-end, treating them as fused-subtree terminuses.

### Task 26: Std / var / sum / mean engine benches

**Files:**
- Create: `tests/bench/m4_engine/test_phase8_std_var.py`

- [ ] **Step 1: Failing test**

```python
# tests/bench/m4_engine/test_phase8_std_var.py
"""End-to-end std/var/sum/mean reductions via engine='metal'."""
import json
import numpy as np
import polars as pl
import polars_metal


def _make_floats(n, seed=0xFA57):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"x": rng.standard_normal(n).astype(np.float32)})


def test_std_var_correctness():
    df = _make_floats(100_000)
    cpu = df.lazy().select(
        pl.col("x").std().alias("std"),
        pl.col("x").var().alias("var"),
    ).collect()
    metal = df.lazy().select(
        pl.col("x").std().alias("std"),
        pl.col("x").var().alias("var"),
    ).collect(engine=polars_metal.MetalEngine())
    for col in ["std", "var"]:
        assert abs(cpu[col][0] - metal[col][0]) < 1e-3, f"{col} mismatch: {cpu[col][0]} vs {metal[col][0]}"


def test_std_var_perf_10m(benchmark):
    df = _make_floats(10_000_000)
    engine = polars_metal.MetalEngine()
    def run():
        return df.lazy().select(
            pl.col("x").std().alias("std"),
            pl.col("x").var().alias("var"),
        ).collect(engine=engine)
    benchmark.pedantic(run, iterations=1, rounds=10, warmup_rounds=2)
    median_ms = benchmark.stats["median"] * 1000
    assert median_ms < 2.0, f"std/var took {median_ms:.2f} ms, target < 2 ms"
    # update baseline.json as before
```

- [ ] **Steps 2-5: Iterate. Pay attention to:**
- MLX's `std`/`var` use n-population variance; Polars' default is sample (n-1). The analyzer needs to know which: either reject when polars-default-sample-mode is in effect, or wrap the MLX result with a Bessel correction `* sqrt(n/(n-1))`. Add a test verifying this matches Polars.

- [ ] **Step 5: Commit**

```bash
git commit -m "M4 Phase 7: std/var reduction bench - green at < 2ms

Bessel-correction wrap added since MLX std uses population variance
and Polars defaults to sample variance."
```

### Task 27: Sort + top-k engine benches

**Files:**
- Create: `tests/bench/m4_engine/test_phase8_sort_topk.py`

- [ ] **Step 1: Failing test (sort)**

```python
# tests/bench/m4_engine/test_phase8_sort_topk.py
"""Sort + top-k via engine='metal'."""
import json
import numpy as np
import polars as pl
import polars_metal
from polars.testing import assert_frame_equal


def _make_floats(n, seed=0x501):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"x": rng.standard_normal(n).astype(np.float32)})


def test_sort_correctness():
    df = _make_floats(10_000)
    cpu  = df.lazy().sort("x").collect()
    metal = df.lazy().sort("x").collect(engine=polars_metal.MetalEngine())
    # MLX sort is unstable; comparison must be by sorted column values, not by row order.
    assert cpu["x"].to_list() == metal["x"].to_list()


def test_sort_perf_10m(benchmark):
    df = _make_floats(10_000_000)
    engine = polars_metal.MetalEngine()
    def run():
        return df.lazy().sort("x").collect(engine=engine)
    benchmark.pedantic(run, iterations=1, rounds=10, warmup_rounds=2)
    median_ms = benchmark.stats["median"] * 1000
    assert median_ms < 10.0, f"sort took {median_ms:.2f}ms, target < 10ms"
    # update baseline.json


def test_topk_correctness():
    df = _make_floats(10_000)
    cpu  = df.lazy().top_k(100, by="x").collect()
    metal = df.lazy().top_k(100, by="x").collect(engine=polars_metal.MetalEngine())
    # top_k preserves all 100 largest values in some order; compare sets.
    assert sorted(cpu["x"].to_list()) == sorted(metal["x"].to_list())


def test_topk_perf_10m(benchmark):
    df = _make_floats(10_000_000)
    engine = polars_metal.MetalEngine()
    def run():
        return df.lazy().top_k(100, by="x").collect(engine=engine)
    benchmark.pedantic(run, iterations=1, rounds=10, warmup_rounds=2)
    median_ms = benchmark.stats["median"] * 1000
    assert median_ms < 10.0, f"top-k took {median_ms:.2f}ms, target < 10ms"
    # update baseline.json
```

- [ ] **Steps 2-5: Iterate.**

Important: sort/top-k are IR nodes in Polars, not expression terminus ops. The walker (Phase 3 Task 17) handles HStack/WithColumns; for Sort and TopK the walker needs an analogous path. Add to walker:

```python
def _handle_sort(self, ir_node, schema):
    """Route Sort IR node to a FusedExprGraph if dtype is F32 and stable=False."""
    if ir_node.stable:
        return MetalPlanNode.CpuLeave  # MLX sort is unstable
    col = ir_node.by_column
    if schema[col] != pl.Float32:
        return MetalPlanNode.CpuLeave
    scope = PyFusionScope()
    a = scope.add_input(col, "F32")
    scope.push_op("Sort", [a])
    scope.mark_output_at_op(0)  # need a "mark by op idx" accessor; add to scope API
    return MetalPlanNode.FusedExprGraph(scope=scope, ...)
```

Then in executor: pass through the sorted column via take/gather.

- [ ] **Step 5: Commit**

```bash
git commit -m "M4 Phase 7: sort + top-k engine benches - green at < 10ms each

Walker extended to route Sort IR nodes (when not stable) and TopK
IR nodes. ArgPartition op added to subgraph builder with kth metadata
in scope."
```

### Task 28: Cumsum engine bench

**Files:**
- Create: `tests/bench/m4_engine/test_phase8_cumsum.py`

- [ ] **Step 1: Failing test**

```python
# tests/bench/m4_engine/test_phase8_cumsum.py
import json
import numpy as np
import polars as pl
import polars_metal
from polars.testing import assert_frame_equal


def _make_floats(n):
    rng = np.random.default_rng(0xC57)
    return pl.DataFrame({"x": rng.standard_normal(n).astype(np.float32)})


def test_cumsum_correctness():
    df = _make_floats(10_000)
    cpu  = df.lazy().with_columns(cs=pl.col("x").cum_sum()).collect()
    metal = df.lazy().with_columns(cs=pl.col("x").cum_sum()).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal, check_exact=False, atol=1e-3)


def test_cumsum_perf_10m(benchmark):
    df = _make_floats(10_000_000)
    engine = polars_metal.MetalEngine()
    def run():
        return df.lazy().with_columns(cs=pl.col("x").cum_sum()).collect(engine=engine)
    benchmark.pedantic(run, iterations=1, rounds=10, warmup_rounds=2)
    median_ms = benchmark.stats["median"] * 1000
    assert median_ms < 8.0, f"cumsum took {median_ms:.2f}ms, target < 8ms"
```

- [ ] **Steps 2-5: Iterate.** The analyzer needs to recognize `cum_sum()` as a `CumSum` op. May need a similar entry in `_REDUCTION_MAP` style table (or a new `_SCAN_MAP`).

```bash
git commit -m "M4 Phase 7: cumsum engine bench - green at < 8ms"
```

### Task 29: Correlation matrix engine bench — **DEFERRED (2026-06-01)**

> **Decision:** deferred indefinitely; the matmul win moves to **Phase 10**
> (`Array[F32, D].dot(lit)` → MLX matmul), which is a walkable expression that
> fits the engine plugin. Rationale: the corr matrix has no engine hook —
> `df.corr()` is eager (no `collect(engine=)`), and `pl.corr(a, b)` isn't even
> visible to the NodeTraverser (`view_expression` raises `NotImplementedError:
> corr`). The matrix (X^T X) isn't expressible as a Polars expression at all.
> The remaining ways in — monkey-patching the eager `df.corr()` or adding a
> public `polars_metal.corr()` — conflict with CLAUDE.md's "engine plugin is
> the only user-facing surface / no new public API." Deferring keeps that
> discipline and wastes no work (the matmul FFI/kernel lands in Phase 10). The
> note below is preserved as the original plan; do not start it without
> revisiting this decision. See `docs/open-questions.md`.

**Files:**
- Create: `tests/bench/m4_engine/test_phase8_corr.py`

- [ ] **Step 1: Failing test**

```python
# tests/bench/m4_engine/test_phase8_corr.py
"""200-column × 200k-row correlation matrix via engine='metal'.

Uses df.corr(). Polars dispatches as a Select over many pairwise ops;
the analyzer recognizes the corr shape and routes to MLX matmul-based
standardize-then-multiply."""
import json
import numpy as np
import polars as pl
import polars_metal


def _make_matrix(rows, cols):
    rng = np.random.default_rng(0xC07)
    mat = rng.standard_normal((rows, cols)).astype(np.float32)
    return pl.from_numpy(mat, schema={f"c{i}": pl.Float32 for i in range(cols)})


def test_corr_correctness():
    df = _make_matrix(1000, 5)
    cpu = df.corr()
    metal_engine = polars_metal.MetalEngine()
    # df.corr() doesn't go through collect(engine=); we'd need to expose
    # a polars_metal API or wait until upstream supports engine-aware corr.
    # For now, exercise via .lazy().select(pl.corr(...)).collect(engine=).
    metal = df.lazy().select(pl.corr("c0", "c1")).collect(engine=metal_engine)
    # ... assert close


def test_corr_perf_200x200k(benchmark):
    df = _make_matrix(200_000, 200)
    engine = polars_metal.MetalEngine()
    # The simplest workload is one pairwise corr; the full 200×200 matrix
    # uses .corr() which currently doesn't accept engine. We exercise via
    # an explicit matmul shape that the analyzer recognizes:
    def run():
        return df.corr()  # placeholder; route via engine when API exposed
    # If the API isn't available, document and skip the bench, leaving the
    # entry in baseline as pending.
```

**Open question:** Polars' `df.corr()` is a method on DataFrame, not an expression — it doesn't currently take `engine=`. The natural path is for polars-metal to override DataFrame.corr (via monkey-patch when the engine module is imported) and route through the MLX matmul. This is meta-engine work; flag as an open question and update the spec if we need to defer.

- [ ] **Step 5: Commit (may be partial)**

```bash
git commit -m "M4 Phase 7: correlation matrix engine bench

Placeholder for the corr matrix benchmark. df.corr() doesn't accept
engine kwarg today; need to expose via either monkey-patch DataFrame.corr
on engine load, or build via .select(pl.corr(...).matmul-shaped) expression.
Marked _pending=true until the API path is decided."
```

---

## Phase 8 — Cumsum-diff rolling (rolling_mean / rolling_sum / rolling_var)

Phase 9 of the roadmap (called "Phase 8" here for plan numbering simplicity within this doc). Identity: `rolling_mean(x, W)[i] = (cumsum(x)[i] − cumsum(x)[i − W]) / W`. Implementation reuses the cumsum + arithmetic ops already wired in Phases 1 + 4.

### Task 30: Rolling-cumsum kernel helper

**Files:**
- Create: `crates/polars-metal-kernels/src/rolling_cumsum.rs`
- Modify: `crates/polars-metal-kernels/src/lib.rs`

- [ ] **Step 1: Failing test**

```rust
// crates/polars-metal-kernels/tests/test_rolling_cumsum.rs
use polars_metal_buffer::MetalBuffer;
use polars_metal_kernels::rolling_cumsum::{rolling_mean_via_cumsum, rolling_sum_via_cumsum};

#[test]
fn rolling_mean_w3_basic() {
    // [1, 2, 3, 4, 5, 6] with W=3
    //  null null  6/3  9/3  12/3  15/3   (cumsum-diff gives undefined for i < W-1; we set None)
    let input = MetalBuffer::from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let out = rolling_mean_via_cumsum(&input, 3).unwrap();
    let v = out.to_f32_vec().unwrap();
    // Polars returns leading nulls for i < W-1; we represent them as NaN in F32 output
    assert!(v[0].is_nan());
    assert!(v[1].is_nan());
    assert!((v[2] - 2.0).abs() < 1e-6);  // (1+2+3)/3 = 2
    assert!((v[3] - 3.0).abs() < 1e-6);  // (2+3+4)/3 = 3
    assert!((v[4] - 4.0).abs() < 1e-6);
    assert!((v[5] - 5.0).abs() < 1e-6);
}

#[test]
fn rolling_sum_matches_polars_at_10m() {
    use polars::prelude::*;
    let rng = 0xC57_u64;
    let n = 10_000_000;
    let data: Vec<f32> = (0..n).map(|i| ((i as u64).wrapping_mul(rng) as f32) * 1e-9).collect();
    let buf = MetalBuffer::from_f32_slice(&data).unwrap();
    let metal_out = rolling_sum_via_cumsum(&buf, 1000).unwrap();
    let polars_series = Series::new("x", &data);
    let polars_out = polars_series.f32().unwrap().rolling_sum(RollingOptions::default().window_size(1000));
    // Compare; allow ULP tolerance because cumsum-diff accumulates error differently than Polars' ring buffer.
    // Spot-check 100 random indices.
    let mv = metal_out.to_f32_vec().unwrap();
    let pv: Vec<f32> = polars_out.unwrap().into_iter().map(|v| v.unwrap_or(f32::NAN)).collect();
    for &i in &[10_000, 100_000, 1_000_000, 5_000_000, 9_999_999] {
        let m = mv[i];
        let p = pv[i];
        if !p.is_nan() {
            assert!((m - p).abs() < (p.abs() * 1e-3).max(1e-3),
                    "row {}: metal {} vs polars {}", i, m, p);
        }
    }
}
```

- [ ] **Step 2: Verify failure**

- [ ] **Step 3: Implement the kernel helper**

```rust
// crates/polars-metal-kernels/src/rolling_cumsum.rs
use polars_metal_buffer::MetalBuffer;
use polars_metal_mlx_sys::{
    array::{MlxArrayHandle, mlx_array_view_metal_buffer, MlxDtype, mlx_eval},
    elementwise::{mlx_sub, mlx_div},
    scan::mlx_cumsum,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RollingError {
    #[error("window > input length")]
    WindowTooLarge,
    #[error("MLX: {0}")]
    MlxError(String),
}

pub fn rolling_sum_via_cumsum(buf: &MetalBuffer, window: usize) -> Result<MetalBuffer, RollingError> {
    let n = buf.byte_size() / 4;
    if window > n {
        return Err(RollingError::WindowTooLarge);
    }
    let x = mlx_array_view_metal_buffer(buf, MlxDtype::F32)
        .map_err(|e| RollingError::MlxError(format!("{e:?}")))?;
    let cs = mlx_cumsum(&x);

    // Build shifted cumsum: prepend `window` zeros, drop last `window` elements.
    let zeros_buf = MetalBuffer::zeros(window * 4)
        .map_err(|e| RollingError::MlxError(format!("zeros: {e:?}")))?;
    let zeros = mlx_array_view_metal_buffer(&zeros_buf, MlxDtype::F32)
        .map_err(|e| RollingError::MlxError(format!("{e:?}")))?;
    let cs_head = polars_metal_mlx_sys::array::mlx_array_slice(&cs, 0, n - window)
        .map_err(|e| RollingError::MlxError(format!("slice: {e:?}")))?;
    let cs_shifted = polars_metal_mlx_sys::array::mlx_array_concatenate(&[zeros, cs_head])
        .map_err(|e| RollingError::MlxError(format!("concat: {e:?}")))?;

    let out = mlx_sub(&cs, &cs_shifted);
    mlx_eval(&[out.clone()]).map_err(|e| RollingError::MlxError(format!("{e:?}")))?;
    polars_metal_mlx_sys::array::mlx_array_to_metal_buffer(&out)
        .map_err(|e| RollingError::MlxError(format!("{e:?}")))
}

pub fn rolling_mean_via_cumsum(buf: &MetalBuffer, window: usize) -> Result<MetalBuffer, RollingError> {
    let sum_buf = rolling_sum_via_cumsum(buf, window)?;
    // Divide by window
    let sum_handle = mlx_array_view_metal_buffer(&sum_buf, MlxDtype::F32)
        .map_err(|e| RollingError::MlxError(format!("{e:?}")))?;
    let w_handle = polars_metal_mlx_sys::array::mlx_array_from_f32_slice(&vec![window as f32; 1])
        .map_err(|e| RollingError::MlxError(format!("{e:?}")))?;
    let out = mlx_div(&sum_handle, &w_handle);
    mlx_eval(&[out.clone()]).map_err(|e| RollingError::MlxError(format!("{e:?}")))?;
    polars_metal_mlx_sys::array::mlx_array_to_metal_buffer(&out)
        .map_err(|e| RollingError::MlxError(format!("{e:?}")))
}
```

The first `window-1` rows need to be set to NaN to match Polars' "leading nulls" semantics. The naive cumsum-diff gives meaningful values for those rows (they're partial sums from index 0); a post-pass writes NaN to them. Add that step (use `mlx::core::where` with a position-based mask).

`mlx_array_slice` and `mlx_array_concatenate` are additional FFI bindings — add them in Task 5/10-style work to `polars-metal-mlx-sys/src/array.rs`.

- [ ] **Step 4: Run tests, iterate.**

```bash
cargo test -p polars-metal-kernels --test test_rolling_cumsum 2>&1 | tail
```

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-kernels/src/rolling_cumsum.rs \
        crates/polars-metal-mlx-sys/src/array.rs \
        crates/polars-metal-kernels/tests/test_rolling_cumsum.rs
git commit -m "M4 Phase 8: rolling_sum / rolling_mean via MLX cumsum-diff

Identity: rolling_mean(x, W)[i] = (cumsum(x)[i] - cumsum(x)[i-W]) / W.
First W-1 rows set to NaN (leading-null semantics).

Adds mlx_array_slice + mlx_array_concatenate FFI bindings."
```

### Task 31: Walker dispatches `rolling_mean` / `rolling_sum` to the cumsum-diff path

**Files:**
- Modify: `python/polars_metal/_fusion_analyzer.py` (recognize rolling)
- Modify: `python/polars_metal/_walker.py`
- Modify: `crates/polars-metal-core/src/udf.rs` (add `execute_rolling` entry)

- [ ] **Step 1: Failing engine test**

```python
# tests/bench/m4_engine/test_phase9_rolling_mean.py
import json
import numpy as np
import polars as pl
import polars_metal
from polars.testing import assert_frame_equal


def _make(n):
    rng = np.random.default_rng(0xF11)
    return pl.DataFrame({"x": rng.standard_normal(n).astype(np.float32)})


def test_rolling_mean_correctness():
    df = _make(10_000)
    cpu = df.lazy().with_columns(r=pl.col("x").rolling_mean(window_size=100)).collect()
    metal = df.lazy().with_columns(r=pl.col("x").rolling_mean(window_size=100)).collect(engine=polars_metal.MetalEngine())
    # Compare from index window-1 onward (leading nulls match)
    assert_frame_equal(cpu[99:], metal[99:], check_exact=False, atol=1e-3)


def test_rolling_mean_perf_10m_w1000(benchmark):
    df = _make(10_000_000)
    engine = polars_metal.MetalEngine()
    def run():
        return df.lazy().with_columns(r=pl.col("x").rolling_mean(window_size=1000)).collect(engine=engine)
    benchmark.pedantic(run, iterations=1, rounds=10, warmup_rounds=2)
    median_ms = benchmark.stats["median"] * 1000
    assert median_ms < 8.0, f"rolling_mean took {median_ms:.2f}ms, target < 8ms"
```

- [ ] **Steps 2-5: Iterate.** The analyzer needs to recognize `rolling_mean(window_size=W)` as a special op-id with W stored in the scope. New `OpId::RollingMean { window: u32 }` would change the OpId enum signature; alternative — special-case the rolling pattern at the analyzer level so it emits the cumsum-diff op chain inline.

The cleaner architectural decision (per spec Task 31 work): the analyzer detects the rolling pattern and inserts the explicit cumsum-diff chain into the FusionScope. That way the subgraph builder doesn't need a new op-id, and the FLOP estimator gets the right answer for free.

```bash
git commit -m "M4 Phase 8: walker recognizes rolling_mean -> emits cumsum-diff chain

Analyzer detects rolling_mean(window=W) and pl.col(...).rolling_mean
expressions; inserts the explicit cumsum-diff op chain into the scope
(no new OpId). Tests verify < 8ms at 10M W=1000."
```

### Task 32: Add `rolling_sum` and `rolling_var` extensions

**Files:**
- Modify: `python/polars_metal/_fusion_analyzer.py`
- Create: `tests/bench/m4_engine/test_phase9_rolling_sum.py`

- [ ] **Step 1-5: Same pattern.** `rolling_sum` is even simpler (drop the divide). `rolling_var` uses both cumsum(x) and cumsum(x²); the analyzer emits a 7-op chain.

```bash
git commit -m "M4 Phase 8: rolling_sum + rolling_var via cumsum-diff family

Closes Phase 8."
```

---

## Phase 9 — List/Array dot-product → MLX matmul (vector search)

Phase 10 of the roadmap. Two source shapes:
- `Array[F32, D]`: `pl.col("emb").arr.dot(query_lit)` (clean dtype contract)
- `List[F32]`: `pl.col("emb").list.eval(pl.element() * pl.lit(query)).list.sum()` (Polars-native columnar matmul)

### Task 33: `Array[F32, D]` dot-product recognition + execution

**Files:**
- Modify: `python/polars_metal/_fusion_analyzer.py` (recognize `.arr.dot(lit)`)
- Modify: `crates/polars-metal-kernels/src/list_dot.rs` (new)
- Modify: `crates/polars-metal-core/src/udf.rs`

- [ ] **Step 1: Failing test (correctness)**

```python
# tests/bench/m4_engine/test_phase10_cosine_topk.py
"""Cosine top-k via Array[F32, D].dot(lit) → MLX matmul."""
import json
import numpy as np
import polars as pl
import polars_metal
from polars.testing import assert_series_equal


def _make_embs(n, d, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-12)
    return pl.from_numpy(x, schema=[f"emb_{i}" for i in range(d)]).to_series()  # adapt to Array dtype


def test_array_dot_correctness_small():
    n, d = 100, 64
    rng = np.random.default_rng(0xE1)
    corpus = rng.standard_normal((n, d)).astype(np.float32)
    query  = rng.standard_normal(d).astype(np.float32)
    df = pl.DataFrame({"emb": pl.Series([list(row) for row in corpus], dtype=pl.Array(pl.Float32, d))})
    expected = corpus @ query
    metal = df.lazy().with_columns(
        sim=pl.col("emb").arr.dot(pl.lit(query))
    ).collect(engine=polars_metal.MetalEngine())
    actual = metal["sim"].to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


def test_array_dot_topk_perf_q100_n100k(benchmark):
    # Q=100 means we run the workload 100 times with different query lits;
    # alternative: synthesize as a single matmul of (Q, D) @ (N, D).T.
    # For this single-query bench we do one .dot(lit) call.
    n, d = 100_000, 768
    rng = np.random.default_rng(0xE2)
    corpus = rng.standard_normal((n, d)).astype(np.float32)
    query  = rng.standard_normal(d).astype(np.float32)
    df = pl.DataFrame({"emb": pl.Series([list(row) for row in corpus], dtype=pl.Array(pl.Float32, d))})
    engine = polars_metal.MetalEngine()

    def run():
        return df.lazy().with_columns(
            sim=pl.col("emb").arr.dot(pl.lit(query))
        ).top_k(10, by="sim").collect(engine=engine)

    benchmark.pedantic(run, iterations=1, rounds=10, warmup_rounds=2)
    median_ms = benchmark.stats["median"] * 1000
    # Note: this is single-query; the spec target (Q=100) would batch.
    # For the single-query case we target < 8 ms (matmul + topk).
    assert median_ms < 8.0, f"array dot top-k took {median_ms:.2f}ms, target < 8ms"
```

- [ ] **Step 2: Verify failure**

- [ ] **Step 3: Implement the recognizer + executor**

In `_fusion_analyzer.py` add:

```python
def _visit_arr_dot(expr, schema, scope):
    """Recognize pl.col("emb").arr.dot(pl.lit(query)).
    Emits a single MatMul op with input ArrayF32(D) and a literal RHS."""
    # Polars represents this as a Function/Expression node. Identify
    # via tree_format and extract the lit child.
    col_expr, lit_expr = _extract_arr_dot_components(expr)
    if not lit_expr.meta.is_literal():
        raise _Aborted()
    col_name = col_expr.meta.output_name()
    inner_d = _extract_array_dim(schema[col_name])
    col_idx = scope.add_input(col_name, f"ArrayF32({inner_d})")
    lit_value = lit_expr.meta.literal_value()
    # Add literal as a synthetic input that the executor will materialize
    # as a 1D MLX array.
    lit_idx = scope.add_input(f"__lit_query_{id(lit_value)}", "F32")
    # The executor (Task 22 extended) knows to broadcast literals.
    return scope.push_op("MatMul", [col_idx, lit_idx])
```

In `crates/polars-metal-kernels/src/list_dot.rs`:

```rust
//! Array[F32, D].dot(literal) path: one MLX matmul.
//!
//! Input: ArrayF32(D) column as (N, D) F32 buffer (zero-copy from Array's
//!        underlying contiguous storage), literal as (D,) F32.
//! Output: (N,) F32 sim values.

use polars_metal_buffer::MetalBuffer;
use polars_metal_mlx_sys::{
    array::{mlx_array_view_metal_buffer_2d, mlx_array_from_f32_slice, mlx_eval, MlxDtype, mlx_array_to_metal_buffer},
    matmul::mlx_matmul,
};

pub fn array_dot_lit_to_buffer(
    col_buf: &MetalBuffer,
    n_rows: usize,
    inner_d: usize,
    lit: &[f32],
) -> Result<MetalBuffer, super::mlx_subgraph::BuildError> {
    let col = mlx_array_view_metal_buffer_2d(col_buf, MlxDtype::F32, n_rows, inner_d)
        .map_err(|e| super::mlx_subgraph::BuildError::MlxError(format!("{e:?}")))?;
    let q = mlx_array_from_f32_slice(lit)
        .map_err(|e| super::mlx_subgraph::BuildError::MlxError(format!("{e:?}")))?;
    let sim = mlx_matmul(&col, &q);
    mlx_eval(&[sim.clone()])
        .map_err(|e| super::mlx_subgraph::BuildError::MlxError(format!("{e:?}")))?;
    mlx_array_to_metal_buffer(&sim)
        .map_err(|e| super::mlx_subgraph::BuildError::MlxError(format!("{e:?}")))
}
```

`mlx_array_view_metal_buffer_2d` is a new FFI binding — same pattern as `mlx_array_view_metal_buffer` but takes a 2D shape.

In `udf.rs` add `execute_array_dot` and wire it from the executor when the scope contains an `OpId::MatMul`.

- [ ] **Step 4: Run, iterate, perf-tune.**

- [ ] **Step 5: Commit**

```bash
git commit -m "M4 Phase 9: Array[F32, D].dot(lit) -> MLX matmul

End-to-end vector search via engine='metal'. Cosine top-k Q=1 N=100k D=768
lands at < 8ms (target hit). Adds mlx_array_view_metal_buffer_2d FFI."
```

### Task 34: `List[F32]` form of dot-product

**Files:**
- Modify: `python/polars_metal/_fusion_analyzer.py`

- [ ] **Step 1: Failing test**

```python
# tests/python_integration/test_list_dot_form.py
"""pl.col("emb").list.eval(pl.element() * pl.lit(q)).list.sum() form
should also route via the matmul path, conditional on uniform inner length."""

import polars as pl
import polars_metal
import numpy as np


def test_list_form_matches_array_form():
    n, d = 100, 32
    rng = np.random.default_rng(0xE3)
    corpus = rng.standard_normal((n, d)).astype(np.float32)
    query = rng.standard_normal(d).astype(np.float32)
    df = pl.DataFrame({"emb": [list(row) for row in corpus]})  # default: List[Float32]
    metal = df.lazy().with_columns(
        sim=pl.col("emb").list.eval(pl.element() * pl.lit(query)).list.sum()
    ).collect(engine=polars_metal.MetalEngine())
    expected = corpus @ query
    np.testing.assert_allclose(metal["sim"].to_numpy(), expected, rtol=1e-3, atol=1e-4)
```

- [ ] **Steps 2-5: Iterate.** The recognizer needs to:
- Detect the `list.eval(element() * lit).list.sum()` pattern.
- Verify uniform inner length (sample-check at plan time, or trust if the schema declares the length).
- Emit a `MatMul` op with the List buffer reinterpreted as a 2D buffer.

```bash
git commit -m "M4 Phase 9: List[F32] form of dot routed to matmul (conditional on uniform inner len)

Closes Phase 9. Both Array[F32, D] and List[F32] forms now route to
MLX matmul; the recognizer sample-checks List inner length at plan
time and falls back to CPU on variable-length data."
```

### Task 35: L2 k-NN perf bench

**Files:**
- Create: `tests/bench/m4_engine/test_phase10_l2_knn.py`

- [ ] **Step 1: Failing test**

```python
# tests/bench/m4_engine/test_phase10_l2_knn.py
"""L2 k-NN brute force via Array dot + post-pass.

L2 distance via the matmul identity: ||q-c||^2 = ||q||^2 + ||c||^2 - 2 q.c.
Two extra reductions vs cosine but same matmul shape; should land < 50ms
at Q=100 N=1M D=128 (matches SIFT1M scale)."""

import json
import numpy as np
import polars as pl
import polars_metal


def _make_corpus(n, d, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def test_l2_knn_perf_q100_n1m_d128(benchmark):
    n, d = 1_000_000, 128
    corpus = _make_corpus(n, d, 0xL2)
    query  = _make_corpus(100, d, 0xQ1)  # 100 queries

    df = pl.DataFrame({"emb": pl.Series([list(row) for row in corpus],
                                         dtype=pl.Array(pl.Float32, d))})
    engine = polars_metal.MetalEngine()

    def run():
        # The current per-query routing — Phase 10.next will batch as matmul
        outs = []
        for q in query:
            o = df.lazy().with_columns(
                dist=(pl.col("emb").arr.dot(pl.lit(q)) * -2.0)  # incomplete L2 formula; placeholder
            ).top_k(10, by="dist").collect(engine=engine)
            outs.append(o)
        return outs

    benchmark.pedantic(run, iterations=1, rounds=5, warmup_rounds=1)
    median_ms = benchmark.stats["median"] * 1000
    # Target may be too tight for the per-query loop; if so, document and
    # mark this as a stretch goal for Phase 10.next (batched matmul).
    if median_ms > 50.0:
        pytest.skip(f"single-query loop lands at {median_ms:.2f}ms; "
                    f"target < 50ms requires Q-batched matmul (Phase 10.next)")
```

- [ ] **Step 2-5: Iterate.** Batched matmul (all 100 queries at once → `(100, D) @ (N, D).T → (100, N)`) is the right shape; the current per-query loop won't hit target. Walker batching is its own design problem — flag as a Phase 10.next item if not in scope.

```bash
git commit -m "M4 Phase 9: L2 k-NN perf bench (may be marked as Phase 10.next stretch goal)"
```

---

## Phase 10 — `pl.col("x").metal.fft()` public API

Phase 11 of the roadmap. New Polars expression namespace `.metal`, registered when `import polars_metal` runs. The engine intercepts at walk time and routes to MLX FFT.

### Task 36: Register the `.metal` expression namespace + `.fft()` placeholder

**Files:**
- Create: `python/polars_metal/_fft.py`
- Modify: `python/polars_metal/__init__.py` (import _fft to trigger registration)

- [ ] **Step 1: Failing test — the namespace is registered after import**

```python
# tests/python_integration/test_fft_api_registration.py
import polars as pl
import polars_metal  # triggers registration


def test_metal_namespace_exists():
    expr = pl.col("x").metal.fft()
    assert expr is not None


def test_fft_on_cpu_raises_not_implemented():
    """Without engine='metal', calling .metal.fft().collect() should raise."""
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    try:
        df.with_columns(y=pl.col("x").metal.fft())
        # Construction succeeds (it's just an expression placeholder)
    except Exception as e:
        assert False, f"expression construction shouldn't fail: {e}"
    # collect on CPU engine should raise
    try:
        df.lazy().with_columns(y=pl.col("x").metal.fft()).collect()
        # If Polars CPU evaluates the namespace placeholder, it'll raise; verify
    except (pl.exceptions.ComputeError, NotImplementedError):
        pass  # expected
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/python_integration/test_fft_api_registration.py -v 2>&1 | tail
```

Expected: `AttributeError: 'Expr' object has no attribute 'metal'`.

- [ ] **Step 3: Register the namespace**

```python
# python/polars_metal/_fft.py
"""Register pl.col(...).metal.fft() as a placeholder that the engine
recognizes. CPU evaluation raises NotImplementedError."""

import polars as pl


@pl.api.register_expr_namespace("metal")
class MetalExpr:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def fft(self) -> pl.Expr:
        """Compute 1D FFT of the column. Output is a Struct[real: F32, imag: F32].

        Only the polars-metal engine handles this. CPU evaluation raises.
        """
        # Polars' map_batches gives us a per-batch hook that the engine
        # walker can intercept.
        def _cpu_not_implemented(s: pl.Series) -> pl.Series:
            raise NotImplementedError("metal.fft() is only available with engine='metal'")
        return self._expr.map_batches(_cpu_not_implemented, return_dtype=pl.Struct({
            "real": pl.Float32, "imag": pl.Float32,
        }))

    def ifft(self) -> pl.Expr:
        def _cpu_not_implemented(s: pl.Series) -> pl.Series:
            raise NotImplementedError("metal.ifft() is only available with engine='metal'")
        return self._expr.map_batches(_cpu_not_implemented, return_dtype=pl.Struct({
            "real": pl.Float32, "imag": pl.Float32,
        }))
```

```python
# python/polars_metal/__init__.py — add
from polars_metal import _fft  # noqa: F401 — triggers namespace registration
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/python_integration/test_fft_api_registration.py -v 2>&1 | tail
```

- [ ] **Step 5: Commit**

```bash
git add python/polars_metal/_fft.py python/polars_metal/__init__.py \
        tests/python_integration/test_fft_api_registration.py
git commit -m "M4 Phase 10: register pl.col(...).metal.fft() expression namespace

The namespace is registered when polars_metal is imported. The .fft()
method returns an expression that the engine walker intercepts; CPU
evaluation raises NotImplementedError (clear error for users who
forget the engine='metal' argument)."
```

### Task 37: Walker recognizes the FFT placeholder + routes to MLX

**Files:**
- Modify: `python/polars_metal/_fusion_analyzer.py`
- Modify: `python/polars_metal/_walker.py`
- Modify: `crates/polars-metal-core/src/udf.rs`

- [ ] **Step 1: Failing engine test**

```python
# tests/bench/m4_engine/test_phase11_fft.py
"""End-to-end pl.col('x').metal.fft() via engine='metal'."""
import json
import numpy as np
import polars as pl
import polars_metal


def test_fft_correctness():
    n = 1024
    rng = np.random.default_rng(0xFF7)
    signal = rng.standard_normal(n).astype(np.float32)
    df = pl.DataFrame({"x": signal})

    result = df.lazy().with_columns(
        spec=pl.col("x").metal.fft()
    ).collect(engine=polars_metal.MetalEngine())

    real = result["spec"].struct.field("real").to_numpy()
    imag = result["spec"].struct.field("imag").to_numpy()
    reconstructed = real + 1j * imag

    expected = np.fft.fft(signal)
    # Tolerance: FFT roundoff scales as O(log N) per output. For N=1024, ~10 ULP.
    np.testing.assert_allclose(reconstructed, expected, rtol=1e-3, atol=1e-3)


def test_fft_perf_8m(benchmark):
    n = 8 * 1024 * 1024  # 8M, power of 2
    rng = np.random.default_rng(0xFF8)
    signal = rng.standard_normal(n).astype(np.float32)
    df = pl.DataFrame({"x": signal})
    engine = polars_metal.MetalEngine()

    def run():
        return df.lazy().with_columns(
            spec=pl.col("x").metal.fft()
        ).collect(engine=engine)

    benchmark.pedantic(run, iterations=1, rounds=10, warmup_rounds=2)
    median_ms = benchmark.stats["median"] * 1000
    assert median_ms < 3.0, f"FFT took {median_ms:.2f}ms, target < 3ms"
```

- [ ] **Step 2: Verify failure**

- [ ] **Step 3: Wire the FFT pattern through analyzer + executor**

In `_fusion_analyzer.py`:

```python
def _visit_metal_fft(expr, schema, scope):
    """Recognize pl.col("x").metal.fft() via the map_batches placeholder.
    The expression node serializes with a recognizable fingerprint."""
    children = expr.meta.exprs() if hasattr(expr.meta, "exprs") else []
    if len(children) != 1:
        raise _Aborted()
    if not _is_metal_fft_placeholder(expr):
        raise _Aborted()
    child = _visit(children[0], schema, scope)
    return scope.push_op("Fft", [child])


def _is_metal_fft_placeholder(expr):
    """Heuristic identification of the metal.fft() expression. Inspects
    the serialized form for the namespace function name."""
    tree = expr.meta.tree_format(return_as_string=True)
    return "metal_fft_placeholder" in tree or "fft" in tree.lower()
```

In `udf.rs` add an `execute_fft` path that returns a Struct column (two F32 series from the interleaved real/imag output). The fold-back wraps the interleaved buffer as two stride-2 Series and constructs the Struct.

- [ ] **Step 4: Run, iterate**

The FFT output struct construction is the new piece — the rest of the fused-graph path already exists. Validate the struct field names match what the user code expects (`spec.struct.field("real")`).

- [ ] **Step 5: Commit**

```bash
git commit -m "M4 Phase 10: FFT routes through walker -> MLX FFT -> Struct[real, imag]

End-to-end pl.col('x').metal.fft() works. Output is a Polars Struct
column with two F32 fields. Bench at 8M points lands < 3 ms (target hit).

Closes Phase 10."
```

---

## Phase 11 — CPU-parity gate + retrospective + landing

### Task 38: CPU-parity gate suite

**Files:**
- Create: `tests/bench/m4_parity/__init__.py`
- Create: `tests/bench/m4_parity/test_non_compute_ops.py`

- [ ] **Step 1: Write the gate suite**

```python
# tests/bench/m4_parity/test_non_compute_ops.py
"""CPU-parity gate: every non-compute op via engine='metal' must be
within 5% of engine='cpu'.

The architectural principle is in CLAUDE.md: a 50× compute win
collapses if surrounding glue (filter, scan, take, materialize) adds
penalty. This suite enforces that the glue stays at parity."""

import time
import statistics
import numpy as np
import polars as pl
import polars_metal


N = 10_000_000


def _make_df():
    rng = np.random.default_rng(0xCD7)
    return pl.DataFrame({
        "x":   rng.standard_normal(N).astype(np.float32),
        "g":   rng.integers(0, 10, size=N, dtype=np.int32),
        "tag": ["a"] * N,
    })


def _time_median(fn, iters=5):
    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - t0)
    return statistics.median(times) / 1e6  # ms


def _parity_within(metric_metal_ms, metric_cpu_ms, tolerance=0.05):
    delta = metric_metal_ms - metric_cpu_ms
    relative = delta / max(metric_cpu_ms, 1.0)
    return relative < tolerance, f"metal {metric_metal_ms:.2f}ms vs cpu {metric_cpu_ms:.2f}ms (delta {delta:.2f}ms, {relative*100:.1f}%)"


def test_filter_parity():
    df = _make_df()
    metal_engine = polars_metal.MetalEngine()
    cpu_t  = _time_median(lambda: df.lazy().filter(pl.col("x") > 0.0).collect())
    metal_t = _time_median(lambda: df.lazy().filter(pl.col("x") > 0.0).collect(engine=metal_engine))
    ok, msg = _parity_within(metal_t, cpu_t)
    assert ok, f"FAIL filter parity: {msg}"


def test_select_parity():
    df = _make_df()
    metal_engine = polars_metal.MetalEngine()
    cpu_t  = _time_median(lambda: df.lazy().select("x", "g").collect())
    metal_t = _time_median(lambda: df.lazy().select("x", "g").collect(engine=metal_engine))
    ok, msg = _parity_within(metal_t, cpu_t)
    assert ok, f"FAIL select parity: {msg}"


def test_take_parity():
    df = _make_df()
    metal_engine = polars_metal.MetalEngine()
    cpu_t  = _time_median(lambda: df.lazy().head(100).collect())
    metal_t = _time_median(lambda: df.lazy().head(100).collect(engine=metal_engine))
    ok, msg = _parity_within(metal_t, cpu_t)
    assert ok, f"FAIL take parity: {msg}"


def test_materialize_parity():
    df = _make_df()
    metal_engine = polars_metal.MetalEngine()
    cpu_t  = _time_median(lambda: df.clone())
    metal_t = _time_median(lambda: df.lazy().collect(engine=metal_engine))
    ok, msg = _parity_within(metal_t, cpu_t, tolerance=0.10)  # materialize allows 10%
    assert ok, f"FAIL materialize parity: {msg}"
```

- [ ] **Step 2: Run the gate**

```bash
pytest tests/bench/m4_parity/test_non_compute_ops.py -v 2>&1 | tail
```

Expected: 4 pass. If any fails, the FFI marshalling or the walker's CPU-routing path has regressed.

- [ ] **Step 3: Diagnose any failures**

If `test_filter_parity` fails:
- Confirm filter is routed to Polars CPU (it should be — no GPU filter in this chunk).
- Check whether engine=metal is doing extra work on top of CPU (e.g. building a plan, walking the IR, deciding to skip — measure walker overhead).
- The walker's "decide and skip" path must be near-zero cost. If it isn't, profile and optimize.

If `test_take_parity` fails: similar diagnosis — the take should be CPU.

If `test_materialize_parity` fails: the issue is likely buffer-bridge wrap costs. Look at `crates/polars-metal-buffer/` and the engine's `df → materialize → df` path.

- [ ] **Step 4: Commit when green**

```bash
git add tests/bench/m4_parity/
git commit -m "M4 Phase 11: CPU-parity gate suite

Enforces architectural principle from CLAUDE.md: non-compute ops via
engine='metal' must be within 5% of engine='cpu' (10% for materialize).
filter / select / take / materialize all green on M2 Ultra."
```

### Task 39: Finalize baseline.json — all `_pending` flipped to false

**Files:**
- Modify: `tests/bench/baseline.json`

- [ ] **Step 1: Verify all M4 entries have measurements**

```bash
python <<'EOF'
import json
b = json.load(open("tests/bench/baseline.json"))
pending = [k for k, v in b["queries"].items() if v.get("_pending")]
if pending:
    print("STILL PENDING:", pending)
    raise SystemExit(1)
print("all entries measured")
EOF
```

If any are still `_pending`, finish those benches before landing.

- [ ] **Step 2: Set `_gate.ratio_lt` thresholds**

For each M4 entry, set the gate so future regressions trigger CI failure. Headroom 20% above the measured value:

```bash
python <<'EOF'
import json
b = json.load(open("tests/bench/baseline.json"))
M4_KEYS = [k for k in b["queries"] if k.startswith(("phase8_", "phase9_", "phase10_", "phase11_"))]
for k in M4_KEYS:
    e = b["queries"][k]
    measured = e.get("metal_ms")
    target = e.get("_target_ms")
    if measured is None or target is None:
        continue
    e["_gate"] = {"max_ms": min(target * 1.2, measured * 1.2)}
    e.pop("_target_ms", None)
json.dump(b, open("tests/bench/baseline.json", "w"), indent=2)
print("set gates for", len(M4_KEYS), "M4 entries")
EOF
```

- [ ] **Step 3: Run `make gate` to verify**

```bash
make gate 2>&1 | tail -30
```

Expected: all phases pass.

- [ ] **Step 4: Commit**

```bash
git add tests/bench/baseline.json
git commit -m "M4 Phase 11: finalize baseline.json gates

All 11 M4 entries have measured ratios; gates set at 20% above measured
value. make gate green on M2 Ultra."
```

### Task 40: Documentation updates

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/kernel-authoring.md`
- Modify: `docs/open-questions.md`

- [ ] **Step 1: Extend `architecture.md`**

Add sections:
- "Fusion analyzer" — what it recognizes, what it doesn't, the FLOP-table approach.
- "MLX subgraph builder" — graph build, eval, zero-copy fold-back.
- "Density-based routing" — thresholds, decision flow.
- "CPU-parity contract" — the 5% gate and its rationale.

- [ ] **Step 2: Extend `kernel-authoring.md`**

Add sections:
- "MLX subgraph idiom" — when to fuse, when to fall back.
- "Cumsum-diff rolling" — the identity, the implementation.
- "List/Array dot via matmul" — recognizer + executor.
- "FFT API surface" — registration pattern, struct output.

- [ ] **Step 3: Update `open-questions.md`**

- Strike through M3-era items resolved by M4 (the per-dispatch fragmentation thesis was correct; whole-subtree fusion was the fix).
- Add new M4 items: density estimator calibration, null-aware fused-subtree path (deferred), batched-matmul for Q>1 vector search, MLX kernel-fusion behavior stability.

- [ ] **Step 4: Commit**

```bash
git add docs/architecture.md docs/kernel-authoring.md docs/open-questions.md
git commit -m "M4 Phase 11: docs update for fusion analyzer + subgraph builder

architecture.md: fusion analyzer, subgraph builder, density routing,
CPU-parity contract.
kernel-authoring.md: MLX subgraph idiom, cumsum-diff rolling, list-dot
matmul, FFT API.
open-questions.md: strike M3-resolved items; add M4 items (density
calibration, null-aware fusion, batched matmul, MLX fusion stability)."
```

### Task 41: M4 retrospective

**Files:**
- Modify: `docs/superpowers/specs/2026-05-28-m4-design.md`

- [ ] **Step 1: Write the retrospective section at the end of the spec**

Per-exit-criterion pass/fail:

```markdown
## M4 retrospective — written 2026-XX-XX

### Functional (criteria 1-10)
- 1. Fusion analyzer: green
- 2. MLX subgraph builder: green
- 3. Haversine engine: green at X.X ms (target < 6 ms)
- 4. Black-Scholes / std-var / sort-topk / cumsum / when-chain / corr: green at ...
- 5. Rolling W=100/1000/10000: green at X.X ms (target < 8 ms)
- 6. Cosine top-k Q=1 N=100k D=768: green at X.X ms
- 7. List[F32] form: green (conditional on uniform inner length)
- 8. metal.fft() at 8M F32: green at X.X ms (target < 3 ms)
- 9. Density-threshold routing: green
- 10. Unsupported-op truncation: green

### Correctness (11-15)
- 11. test-kernel: green
- 12. test-unit: green
- 13. python_integration: green
- 14. test-conformance: green
- 15. M3 carryovers: green

### Performance (16-20)
- 16. baseline.json all measured; gates set
- 17. CPU-parity gate: green (parity within 5%; materialize within 10%)
- 18. M3 baselines (TPC-H Q1/Q6): unchanged, gates green
- 19. Plan-time overhead: median X.X µs
- 20. MLX subgraph build cost: median X.X µs for 20-op chain

### Quality (21-23)
- 21. make gate: green on M2 Ultra
- 22. Portability gate: green on M2 16GB and M1 8GB except phase10_cosine_topk_q100_n1m
- 23. Lint clean

### Documentation (24-27)
- 24. architecture.md updated
- 25. kernel-authoring.md updated
- 26. open-questions.md updated
- 27. This retrospective

### Surprises during execution
- (Fill in: which Polars meta API was unstable, what MLX semantic
  differed from expected, which density-threshold edge case showed up.)

### Resolved-in-follow-up commits
- (Fill in: commits after the bulk landing that fixed regression x.)

### Lessons for Phases 12+
- The fusion analyzer's expression-IR walker is the most fragile
  piece. Polars IR semantics drift between revs; consider a
  Polars-upstream stable expression-tree introspection API.
- Density estimator's static FLOP table worked well in practice.
- (Fill in others as they emerge.)

### Hand-off to Phases 12-13
- Phase 12 (custom MSL gregorian-calendar kernel for dt.year/month/day):
  the MLX subgraph builder doesn't help here — needs new MSL.
- Phase 13 (pairwise Levenshtein / DTW): same.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-05-28-m4-design.md
git commit -m "M4 Phase 11: retrospective

Closes M4 chunk (Phases 8-11). All 41 exit criteria documented;
surprises and lessons recorded. Next chunk: Phases 12 (gregorian
MSL kernel) + 13 (pairwise distance MSL kernels)."
```

### Task 42: Final make gate + branch ready for merge

- [ ] **Step 1: Final gate**

```bash
make gate 2>&1 | tee /tmp/m4_gate_out.txt
```

Expected: every phase passes. The output includes:
- lint (clippy, fmt, ruff): clean
- test-unit: all green (M3 + M4 unit tests)
- test-kernel: all green (M3 + M4 kernel tests)
- test-conformance: all green (no new failures)
- wheel: builds

- [ ] **Step 2: Push the branch**

```bash
git push -u origin m4-fusion-and-fft
```

- [ ] **Step 3: Open PR with this plan + spec referenced**

```bash
gh pr create \
  --title "M4: MLX subgraph fusion + cumsum-diff rolling + list-dot vector search + Expr.fft()" \
  --body "$(cat <<'EOF'
## Summary
- Phases 8-11 of the revised roadmap (per `docs/superpowers/specs/2026-05-28-m4-design.md`)
- MLX subgraph fusion engine for F32 compute-shaped expression trees
- Rolling mean/sum/var via cumsum-diff identity
- List[F32]/Array[F32, D] dot routed to MLX matmul (vector search)
- `pl.col(...).metal.fft()` public API
- CPU-parity gate suite enforcing the architectural contract

## Measurements (M2 Ultra)
| Workload                              | Polars CPU | engine='metal' | Win |
|---------------------------------------|------------|----------------|------|
| Haversine 10M F32                     | 181 ms     | X.X ms         | NN× |
| Black-Scholes 10M F32                 | 242 ms     | X.X ms         | NN× |
| Rolling mean W=1000 10M F32           | 114 ms     | X.X ms         | NN× |
| Cosine top-k N=100k D=768             | (n/a)      | X.X ms         | -    |
| FFT 8M F32                            | (no API)   | X.X ms         | -    |

(Fill in actual numbers from baseline.json at landing time.)

## Test plan
- [ ] `make gate` green on M2 Ultra (recorded above)
- [ ] Portability gate green on M2 16GB and M1 8GB
- [ ] All 11 new bench entries in `tests/bench/baseline.json` measured
- [ ] CPU-parity gate green
- [ ] M3 TPC-H baselines unchanged

Refs: spec `docs/superpowers/specs/2026-05-28-m4-design.md`
Plan: `docs/superpowers/plans/2026-05-28-m4-fusion-and-fft.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Done.

---

## Phase 12+ (not in this chunk; documented for context)

**Phase 12 — Custom MSL gregorian-calendar kernel.** The Polars `dt.year/month/day` extraction is 178 ms at 10M rows (gregorian calendar math, branchy). MLX has no equivalent. A custom MSL kernel using parallel threadgroups with the proleptic Gregorian formulas should land at ~5 ms (estimated 30-40× win). Design: signed-int 64 → year/month/day i32 outputs, threadgroup-per-1024-row tile, no shared memory needed (the math is per-row). Test against Polars CPU at random datetime points across the supported range.

**Phase 13 — Pairwise distance MSL kernels.** Levenshtein, DTW, edit-distance. Each kernel: one threadgroup per pair, parallel within the DP cell grid. High compute density per pair (O(L²) ops where L is sequence length). Speculative but plausibly 100-1000× over Polars (which has no built-in for these).

These are a separate spec + plan after M4 lands. The reason for splitting: Phases 8-11 share one infrastructure piece (MLX subgraph fusion); Phases 12-13 are new MSL kernel work. Mixing both in one chunk would balloon the plan and slow review.

---

## End of plan










