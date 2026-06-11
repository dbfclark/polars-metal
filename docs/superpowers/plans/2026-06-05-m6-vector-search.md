# M6 Vector Search (`.metal.cosine_topk` / `.metal.knn`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship GPU-accelerated batched vector search as an expression namespace —
`pl.col("emb").metal.cosine_topk(corpus, k)` and `.metal.knn(corpus, k)` — under
`engine="metal"`, returning a `Struct{indices: List[UInt32], scores: List[Float32]}` per query row.

**Architecture:** MLX composition. Stage query `(Q,D)` and corpus `(N,D)` F32 buffers as 2-D MLX
arrays, normalize, GEMM → `(Q,N)` similarities, `argpartition`+`slice`+`take_along_axis` → `(Q,k)`
indices/scores, read back to host, sort each row, build a Polars struct column. Recognition reuses
the **M5 collect-and-stitch template** (serialize-detect a sentinel expression + corpus captured
by handle-id; drop output cols, CPU-collect the rest, stitch GPU columns).

**Tech Stack:** Rust + `cxx` FFI to MLX (C++), PyO3, Polars py-1.40.1, numpy (oracle/marshalling).

**Spec:** `docs/superpowers/specs/2026-06-04-m6-metal-namespace-design.md` (§A2).

---

## File Structure

**New files:**
- `crates/polars-metal-mlx-sys/src/shape.rs` — Rust wrappers: `mlx_transpose`, `mlx_reshape`, `mlx_slice`, `mlx_take_along_axis`.
- `crates/polars-metal-mlx-sys/tests/test_vector_ffi.rs` — FFI building-block + characterization tests.
- `crates/polars-metal-core/src/vector_search.rs` — the `execute_vector_search` dispatcher + PyO3 binding.
- `python/polars_metal/_vector_namespace.py` — `register_expr_namespace("metal")`, corpus capture, sentinel builder.
- `python/polars_metal/_vector_detect.py` — serialize-detect sentinel bindings from a LazyFrame.
- `python/polars_metal/_vector_dispatch.py` — collect-and-stitch dispatch (build the struct column).
- `tests/python_integration/test_vector_search.py` — differential vs numpy oracle + raise tests.
- `tests/bench/m4_survey/bench_cosine_topk.py` — perf bench + `ratio_lt` gate.

**Modified files:**
- `crates/polars-metal-mlx-sys/cxx/mlx_bridge.h` / `mlx_bridge.cc` — C++ wrappers for transpose/reshape/slice/take_along_axis/i32-readback.
- `crates/polars-metal-mlx-sys/src/lib.rs` — `extern "C++"` declarations for the new ops; `pub mod shape;`.
- `crates/polars-metal-mlx-sys/src/array.rs` — `mlx_array_to_i32_vec` (I32 readback).
- `crates/polars-metal-core/src/lib.rs` — register `vector_search::execute_vector_search`.
- `python/polars_metal/__init__.py` — import the namespace module (registers it) + wire detection/dispatch into `collect_wrapper`.

---

## Phase 0 — MLX FFI building blocks

### Task 0: Characterize `argpartition` 2-D / axis / dtype semantics (de-risk first)

**Files:**
- Test: `crates/polars-metal-mlx-sys/tests/test_vector_ffi.rs` (create)

The whole top-k design rests on `mlx_op_argpartition` partitioning along the **last axis** of a 2-D
array and on the index dtype. Pin both with a test before building on them.

- [ ] **Step 1: Write the characterization test**

```rust
//! M6 vector-search FFI building blocks + characterization.
use polars_metal_mlx_sys::array::{
    mlx_array_eval, mlx_array_to_f32_vec, mlx_array_view_metal_buffer, MlxArrayHandle, MlxDtype,
};
use polars_metal_mlx_sys::sort::mlx_argpartition;
use polars_metal_mlx_sys::elementwise::mlx_cast;
use polars_metal_buffer::MetalBuffer;
use std::sync::Arc;

/// Build a 2-D (rows, cols) F32 MLX array from a row-major host slice.
fn arr2d(data: &[f32], rows: i64, cols: i64) -> MlxArrayHandle {
    let device = polars_metal_buffer::default_device();
    // SAFETY: data outlives the borrowed buffer within this call.
    let buf = unsafe {
        MetalBuffer::from_borrowed_f32(&device, data.as_ptr(), data.len())
    }
    .map(Arc::new)
    .expect("metal buffer");
    mlx_array_view_metal_buffer(buf, &[rows, cols], MlxDtype::F32).expect("2d view")
}

#[test]
fn argpartition_2d_is_last_axis() {
    // Two rows; smallest value per row sits at a different column.
    // Row 0 min at col 2; row 1 min at col 0.
    let data = [3.0f32, 5.0, 1.0,   2.0, 8.0, 9.0];
    let a = arr2d(&data, 2, 3);
    // kth=0 → position 0 of each row holds that row's smallest index.
    let idx = mlx_argpartition(&a, 0).expect("argpartition");
    let idx_f = mlx_cast(&idx, MlxDtype::F32).expect("cast");
    mlx_array_eval(&[idx_f.clone()]).expect("eval");
    let v = mlx_array_to_f32_vec(&idx_f).expect("readback");
    // Expect a (2,3) index array; column 0 of each row = argmin of that row.
    assert_eq!(v.len(), 6);
    assert_eq!(v[0] as i32, 2, "row 0 argmin should be col 2");
    assert_eq!(v[3] as i32, 0, "row 1 argmin should be col 0");
}
```

- [ ] **Step 2: Run and observe**

Run: `cargo test -p polars-metal-mlx-sys --test test_vector_ffi argpartition_2d_is_last_axis -- --test-threads=1 --nocapture`
Expected: PASS. If it FAILS (wrong axis), the dispatcher (Task 6) must `reshape`/`transpose` so the
top-k axis is last, or pass an explicit axis — record the observed behavior in a comment and adjust.
(`mlx_array_to_i32_vec`, `MlxDtype`, `default_device` may not exist yet — if the test doesn't
compile for those reasons, stub via the F32-cast path shown above; this task only needs argmin
positions, which survive the F32 cast.)

- [ ] **Step 3: Commit**

```bash
git add crates/polars-metal-mlx-sys/tests/test_vector_ffi.rs
git commit -m "test(mlx): characterize argpartition 2-D last-axis semantics (M6 vector search)"
```

---

### Task 1: `transpose` FFI wrapper

**Files:**
- Modify: `crates/polars-metal-mlx-sys/cxx/mlx_bridge.h`, `crates/polars-metal-mlx-sys/cxx/mlx_bridge.cc`
- Modify: `crates/polars-metal-mlx-sys/src/lib.rs`
- Create: `crates/polars-metal-mlx-sys/src/shape.rs`
- Modify: `crates/polars-metal-mlx-sys/src/lib.rs` (`pub mod shape;`)
- Test: `crates/polars-metal-mlx-sys/tests/test_vector_ffi.rs`

- [ ] **Step 1: Write the failing test**

```rust
use polars_metal_mlx_sys::shape::mlx_transpose;

#[test]
fn transpose_2x3_to_3x2() {
    let data = [1.0f32, 2.0, 3.0,   4.0, 5.0, 6.0]; // (2,3) row-major
    let a = arr2d(&data, 2, 3);
    let t = mlx_transpose(&a, &[1, 0]).expect("transpose");
    mlx_array_eval(&[t.clone()]).expect("eval");
    assert_eq!(t.shape(), vec![3, 2]);
    // (3,2) row-major = columns of original = [1,4, 2,5, 3,6]
    assert_eq!(mlx_array_to_f32_vec(&t).unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p polars-metal-mlx-sys --test test_vector_ffi transpose_2x3_to_3x2 -- --test-threads=1`
Expected: FAIL — `mlx_transpose` not found.

- [ ] **Step 3: Add the C++ wrapper**

In `mlx_bridge.h`, near the matmul declaration (around line 245), add:

```cpp
// ── M6 vector search: shape ops ──────────────────────────────────────────────
std::shared_ptr<MlxArray> mlx_op_transpose(
    const std::shared_ptr<MlxArray>& a,
    rust::Slice<const int32_t> axes);
```

In `mlx_bridge.cc`, near the matmul implementation (around line 441), add:

```cpp
std::shared_ptr<MlxArray> mlx_op_transpose(
    const std::shared_ptr<MlxArray>& a,
    rust::Slice<const int32_t> axes) {
    std::vector<int> ax(axes.begin(), axes.end());
    auto base = std::make_shared<mlx::core::array>(mlx::core::transpose(*a, ax));
    return std::shared_ptr<MlxArray>(base, static_cast<MlxArray*>(base.get()));
}
```

- [ ] **Step 4: Declare it in the cxx bridge**

In `crates/polars-metal-mlx-sys/src/lib.rs`, inside the `extern "C++"` block near `mlx_op_matmul`
(line ~255), add:

```rust
        fn mlx_op_transpose(
            a: &SharedPtr<MlxArray>,
            axes: &[i32],
        ) -> Result<SharedPtr<MlxArray>>;
```

- [ ] **Step 5: Create the Rust wrapper module**

Create `crates/polars-metal-mlx-sys/src/shape.rs`:

```rust
//! M6 vector search: shape-manipulation wrappers (transpose/reshape/slice/take_along_axis).
use crate::array::MlxArrayHandle;
use crate::ffi;
use crate::FfiError;

/// Transpose `a` according to `axes` (a permutation of `0..ndim`).
pub fn mlx_transpose(a: &MlxArrayHandle, axes: &[i32]) -> Result<MlxArrayHandle, FfiError> {
    let ptr = ffi::mlx_op_transpose(&a.ptr, axes).map_err(FfiError::from)?;
    Ok(MlxArrayHandle { ptr, _input_refs: a._input_refs.clone() })
}
```

In `crates/polars-metal-mlx-sys/src/lib.rs`, add near the other `pub mod` lines (~line 18):

```rust
pub mod shape;
```

(If `MlxArrayHandle`'s fields aren't visible from `shape.rs`, mirror how `matmul.rs` constructs its
result — copy that exact pattern, including `_input_refs` propagation.)

- [ ] **Step 6: Run to verify it passes**

Run: `cargo test -p polars-metal-mlx-sys --test test_vector_ffi transpose_2x3_to_3x2 -- --test-threads=1`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/polars-metal-mlx-sys/
git commit -m "feat(mlx): transpose FFI wrapper (M6 vector search)"
```

---

### Task 2: `reshape` FFI wrapper

**Files:** same set as Task 1.

- [ ] **Step 1: Write the failing test**

```rust
use polars_metal_mlx_sys::shape::mlx_reshape;

#[test]
fn reshape_6_to_3x2_and_keepdim() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = arr2d(&data, 1, 6);
    let r = mlx_reshape(&a, &[3, 2]).expect("reshape");
    mlx_array_eval(&[r.clone()]).expect("eval");
    assert_eq!(r.shape(), vec![3, 2]);
    // (N,) -> (N,1) keepdim case used by norm broadcasting:
    let n = arr2d(&[7.0, 8.0, 9.0], 1, 3);
    let col = mlx_reshape(&n, &[3, 1]).expect("reshape col");
    mlx_array_eval(&[col.clone()]).expect("eval");
    assert_eq!(col.shape(), vec![3, 1]);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p polars-metal-mlx-sys --test test_vector_ffi reshape_6_to_3x2_and_keepdim -- --test-threads=1`
Expected: FAIL — `mlx_reshape` not found.

- [ ] **Step 3: Add the C++ wrapper**

`mlx_bridge.h`:

```cpp
std::shared_ptr<MlxArray> mlx_op_reshape(
    const std::shared_ptr<MlxArray>& a,
    rust::Slice<const int32_t> shape);
```

`mlx_bridge.cc`:

```cpp
std::shared_ptr<MlxArray> mlx_op_reshape(
    const std::shared_ptr<MlxArray>& a,
    rust::Slice<const int32_t> shape) {
    std::vector<int> sh(shape.begin(), shape.end());
    auto base = std::make_shared<mlx::core::array>(mlx::core::reshape(*a, sh));
    return std::shared_ptr<MlxArray>(base, static_cast<MlxArray*>(base.get()));
}
```

- [ ] **Step 4: Declare + wrap**

`lib.rs` extern block:

```rust
        fn mlx_op_reshape(
            a: &SharedPtr<MlxArray>,
            shape: &[i32],
        ) -> Result<SharedPtr<MlxArray>>;
```

`shape.rs`:

```rust
/// Reshape `a` to `shape` (total element count must match).
pub fn mlx_reshape(a: &MlxArrayHandle, shape: &[i32]) -> Result<MlxArrayHandle, FfiError> {
    let ptr = ffi::mlx_op_reshape(&a.ptr, shape).map_err(FfiError::from)?;
    Ok(MlxArrayHandle { ptr, _input_refs: a._input_refs.clone() })
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cargo test -p polars-metal-mlx-sys --test test_vector_ffi reshape_6_to_3x2_and_keepdim -- --test-threads=1`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/polars-metal-mlx-sys/
git commit -m "feat(mlx): reshape FFI wrapper (M6 vector search)"
```

---

### Task 3: `slice` FFI wrapper (first-k along an axis)

**Files:** same set.

- [ ] **Step 1: Write the failing test**

```rust
use polars_metal_mlx_sys::shape::mlx_slice;

#[test]
fn slice_first_2_cols_of_2x3() {
    let data = [1.0f32, 2.0, 3.0,   4.0, 5.0, 6.0]; // (2,3)
    let a = arr2d(&data, 2, 3);
    // start=[0,0], stop=[2,2], strides=[1,1] -> (2,2) first two columns.
    let s = mlx_slice(&a, &[0, 0], &[2, 2], &[1, 1]).expect("slice");
    mlx_array_eval(&[s.clone()]).expect("eval");
    assert_eq!(s.shape(), vec![2, 2]);
    assert_eq!(mlx_array_to_f32_vec(&s).unwrap(), vec![1.0, 2.0, 4.0, 5.0]);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p polars-metal-mlx-sys --test test_vector_ffi slice_first_2_cols_of_2x3 -- --test-threads=1`
Expected: FAIL — `mlx_slice` not found.

- [ ] **Step 3: Add the C++ wrapper**

`mlx_bridge.h`:

```cpp
std::shared_ptr<MlxArray> mlx_op_slice(
    const std::shared_ptr<MlxArray>& a,
    rust::Slice<const int32_t> start,
    rust::Slice<const int32_t> stop,
    rust::Slice<const int32_t> strides);
```

`mlx_bridge.cc`:

```cpp
std::shared_ptr<MlxArray> mlx_op_slice(
    const std::shared_ptr<MlxArray>& a,
    rust::Slice<const int32_t> start,
    rust::Slice<const int32_t> stop,
    rust::Slice<const int32_t> strides) {
    std::vector<int> lo(start.begin(), start.end());
    std::vector<int> hi(stop.begin(), stop.end());
    std::vector<int> st(strides.begin(), strides.end());
    auto base = std::make_shared<mlx::core::array>(mlx::core::slice(*a, lo, hi, st));
    return std::shared_ptr<MlxArray>(base, static_cast<MlxArray*>(base.get()));
}
```

- [ ] **Step 4: Declare + wrap**

`lib.rs`:

```rust
        fn mlx_op_slice(
            a: &SharedPtr<MlxArray>,
            start: &[i32],
            stop: &[i32],
            strides: &[i32],
        ) -> Result<SharedPtr<MlxArray>>;
```

`shape.rs`:

```rust
/// Slice `a` with per-axis `start`/`stop`/`strides` (NumPy-style half-open).
pub fn mlx_slice(
    a: &MlxArrayHandle,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
) -> Result<MlxArrayHandle, FfiError> {
    let ptr = ffi::mlx_op_slice(&a.ptr, start, stop, strides).map_err(FfiError::from)?;
    Ok(MlxArrayHandle { ptr, _input_refs: a._input_refs.clone() })
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cargo test -p polars-metal-mlx-sys --test test_vector_ffi slice_first_2_cols_of_2x3 -- --test-threads=1`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/polars-metal-mlx-sys/
git commit -m "feat(mlx): slice FFI wrapper (M6 vector search)"
```

---

### Task 4: `take_along_axis` FFI wrapper (gather scores at partitioned indices)

**Files:** same set.

- [ ] **Step 1: Write the failing test**

```rust
use polars_metal_mlx_sys::shape::mlx_take_along_axis;
use polars_metal_mlx_sys::elementwise::mlx_cast;

#[test]
fn take_along_axis_gathers_per_row() {
    // values (2,3); gather columns [2,0] from row 0 and [0,1] from row 1.
    let vals = arr2d(&[10.0f32, 11.0, 12.0,   20.0, 21.0, 22.0], 2, 3);
    // index array (2,2) as F32 then cast to I32 (FFI takes an MLX integer array).
    let idx_f = arr2d(&[2.0f32, 0.0,   0.0, 1.0], 2, 2);
    let idx = mlx_cast(&idx_f, MlxDtype::I32).expect("cast idx");
    let g = mlx_take_along_axis(&vals, &idx, 1).expect("gather");
    mlx_array_eval(&[g.clone()]).expect("eval");
    assert_eq!(g.shape(), vec![2, 2]);
    assert_eq!(mlx_array_to_f32_vec(&g).unwrap(), vec![12.0, 10.0, 20.0, 21.0]);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p polars-metal-mlx-sys --test test_vector_ffi take_along_axis_gathers_per_row -- --test-threads=1`
Expected: FAIL — `mlx_take_along_axis` not found.

- [ ] **Step 3: Add the C++ wrapper**

`mlx_bridge.h`:

```cpp
std::shared_ptr<MlxArray> mlx_op_take_along_axis(
    const std::shared_ptr<MlxArray>& a,
    const std::shared_ptr<MlxArray>& indices,
    int32_t axis);
```

`mlx_bridge.cc`:

```cpp
std::shared_ptr<MlxArray> mlx_op_take_along_axis(
    const std::shared_ptr<MlxArray>& a,
    const std::shared_ptr<MlxArray>& indices,
    int32_t axis) {
    auto base = std::make_shared<mlx::core::array>(
        mlx::core::take_along_axis(*a, *indices, axis));
    return std::shared_ptr<MlxArray>(base, static_cast<MlxArray*>(base.get()));
}
```

- [ ] **Step 4: Declare + wrap**

`lib.rs`:

```rust
        fn mlx_op_take_along_axis(
            a: &SharedPtr<MlxArray>,
            indices: &SharedPtr<MlxArray>,
            axis: i32,
        ) -> Result<SharedPtr<MlxArray>>;
```

`shape.rs`:

```rust
/// Gather along `axis`: `out[i,j] = a[i, indices[i,j]]` for `axis=1`.
pub fn mlx_take_along_axis(
    a: &MlxArrayHandle,
    indices: &MlxArrayHandle,
    axis: i32,
) -> Result<MlxArrayHandle, FfiError> {
    let ptr = ffi::mlx_op_take_along_axis(&a.ptr, &indices.ptr, axis).map_err(FfiError::from)?;
    let mut refs = a._input_refs.clone();
    refs.extend(indices._input_refs.iter().cloned());
    Ok(MlxArrayHandle { ptr, _input_refs: refs })
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cargo test -p polars-metal-mlx-sys --test test_vector_ffi take_along_axis_gathers_per_row -- --test-threads=1`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/polars-metal-mlx-sys/
git commit -m "feat(mlx): take_along_axis FFI wrapper (M6 vector search)"
```

---

### Task 5: I32 readback (`mlx_array_to_i32_vec`)

**Files:**
- Modify: `crates/polars-metal-mlx-sys/cxx/mlx_bridge.h`, `mlx_bridge.cc`, `src/lib.rs`, `src/array.rs`
- Test: `crates/polars-metal-mlx-sys/tests/test_vector_ffi.rs`

`argpartition` indices must reach the host as integers. There is no I32 readback today; add one.
The dispatcher will `mlx_cast(idx, I32)` before calling this, so the readback always sees I32.

- [ ] **Step 1: Write the failing test**

```rust
use polars_metal_mlx_sys::array::mlx_array_to_i32_vec;
use polars_metal_mlx_sys::elementwise::mlx_cast;

#[test]
fn i32_readback_roundtrip() {
    let f = arr2d(&[2.0f32, 0.0, 5.0, 9.0], 1, 4);
    let i = mlx_cast(&f, MlxDtype::I32).expect("cast i32");
    mlx_array_eval(&[i.clone()]).expect("eval");
    assert_eq!(mlx_array_to_i32_vec(&i).unwrap(), vec![2, 0, 5, 9]);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p polars-metal-mlx-sys --test test_vector_ffi i32_readback_roundtrip -- --test-threads=1`
Expected: FAIL — `mlx_array_to_i32_vec` not found.

- [ ] **Step 3: Add the C++ wrapper**

`mlx_bridge.h` (near `mlx_array_copy_to_f32`, ~line 79):

```cpp
void mlx_array_copy_to_i32(const std::shared_ptr<MlxArray>& arr, int32_t* out, size_t n);
```

`mlx_bridge.cc` (near `mlx_array_copy_to_f32`, ~line 122):

```cpp
void mlx_array_copy_to_i32(const std::shared_ptr<MlxArray>& arr, int32_t* out, size_t n) {
    const int32_t* src = arr->data<int32_t>();
    std::memcpy(out, src, n * sizeof(int32_t));
}
```

- [ ] **Step 4: Declare + wrap**

`lib.rs` extern block (near `mlx_array_copy_to_f32`, ~line 74):

```rust
        unsafe fn mlx_array_copy_to_i32(arr: &SharedPtr<MlxArray>, out: *mut i32, n: usize);
```

`array.rs` (mirror `mlx_array_to_f32_vec`, ~line 181):

```rust
/// Read a materialized I32 array back to a host `Vec<i32>`. Call after `mlx_array_eval`.
pub fn mlx_array_to_i32_vec(handle: &MlxArrayHandle) -> Result<Vec<i32>, FfiError> {
    let n: usize = handle.shape().iter().product();
    let mut out = vec![0i32; n];
    // SAFETY: `out` has exactly `n` i32 slots; matches the array element count.
    unsafe { ffi::mlx_array_copy_to_i32(&handle.ptr, out.as_mut_ptr(), n) };
    Ok(out)
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cargo test -p polars-metal-mlx-sys --test test_vector_ffi i32_readback_roundtrip -- --test-threads=1`
Expected: PASS.

- [ ] **Step 6: Run the whole FFI test file**

Run: `cargo test -p polars-metal-mlx-sys --test test_vector_ffi -- --test-threads=1`
Expected: all Phase-0 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/polars-metal-mlx-sys/
git commit -m "feat(mlx): i32 readback (M6 vector search)"
```

---

### Task 5b: axis-aware `argpartition` wrapper (REQUIRED — Task 0 finding)

**Files:** same set as Task 1 (`mlx_bridge.h/.cc`, `src/lib.rs`, `src/sort.rs`, `tests/test_vector_ffi.rs`).

Task 0 proved the existing `mlx_op_argpartition` **flattens to 1-D** (calls `mlx::core::argpartition(a, kth)`,
ops.h:704) — it does NOT partition per-row. Per-query top-k needs the axis-aware overload
(`argpartition(a, kth, axis)`, ops.h:710). Add it without touching the existing flattening wrapper.

- [ ] **Step 1: Write the failing test** (append to `tests/test_vector_ffi.rs`)

```rust
use polars_metal_mlx_sys::sort::mlx_argpartition_axis;

#[test]
fn argpartition_axis_is_per_row() {
    // (2,3): row 0 min at col 2, row 1 min at col 0. axis=-1 → per-row partition.
    let a = arr2d(&[3.0f32, 5.0, 1.0,   2.0, 8.0, 9.0], 2, 3);
    let idx = mlx_argpartition_axis(&a, 0, -1).expect("argpartition_axis");
    let idx_f = mlx_cast(&idx, MlxDtype::F32).expect("cast");
    mlx_array_eval(&[idx_f.clone()]).expect("eval");
    assert_eq!(idx_f.shape(), vec![2, 3], "axis-aware keeps 2-D shape");
    let v = mlx_array_to_f32_vec(&idx_f).unwrap();
    assert_eq!(v[0] as i32, 2, "row 0 col0 = argmin = col 2");
    assert_eq!(v[3] as i32, 0, "row 1 col0 = argmin = col 0");
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p polars-metal-mlx-sys --test test_vector_ffi argpartition_axis_is_per_row -- --test-threads=1`
Expected: FAIL — `mlx_argpartition_axis` not found.

- [ ] **Step 3: Add the C++ wrapper**

`mlx_bridge.h` (near the existing `mlx_op_argpartition`, ~line 207):

```cpp
std::shared_ptr<MlxArray> mlx_op_argpartition_axis(
    const std::shared_ptr<MlxArray>& a, int32_t kth, int32_t axis);
```

`mlx_bridge.cc` (next to the existing `mlx_op_argpartition`, ~line 376):

```cpp
std::shared_ptr<MlxArray> mlx_op_argpartition_axis(
    const std::shared_ptr<MlxArray>& a, int32_t kth, int32_t axis) {
    auto base = std::make_shared<mlx::core::array>(
        mlx::core::argpartition(*a, kth, axis));
    return std::shared_ptr<MlxArray>(base, static_cast<MlxArray*>(base.get()));
}
```

- [ ] **Step 4: Declare + wrap**

`src/lib.rs` extern block (near `mlx_op_argpartition`, ~line 230):

```rust
        fn mlx_op_argpartition_axis(
            a: &SharedPtr<MlxArray>,
            kth: i32,
            axis: i32,
        ) -> Result<SharedPtr<MlxArray>>;
```

`src/sort.rs` (next to `mlx_argpartition`, ~line 38):

```rust
/// Argpartition along `axis` (use `-1` for the last axis). Returns integer indices,
/// same shape as `a`, with the `0..=kth` positions along `axis` holding the kth-smallest.
pub fn mlx_argpartition_axis(
    a: &MlxArrayHandle,
    kth: i32,
    axis: i32,
) -> Result<MlxArrayHandle, FfiError> {
    let ptr = ffi::mlx_op_argpartition_axis(&a.ptr, kth, axis).map_err(FfiError::from)?;
    Ok(MlxArrayHandle { ptr, _input_refs: a._input_refs.clone() })
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cargo test -p polars-metal-mlx-sys --test test_vector_ffi argpartition_axis_is_per_row -- --test-threads=1`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/polars-metal-mlx-sys/
git commit -m "feat(mlx): axis-aware argpartition wrapper (M6 vector search)"
```

---

## Phase 1 — Rust dispatcher + PyO3

### Task 6: `vector_search_topk` core (cosine path)

> **Corrections from Task 0:** use `MetalDevice::system_default()` (the plan's `default_device()`
> does not exist) and `mlx_argpartition_axis(&x, k-1, -1)` (the bare `mlx_argpartition` flattens to
> 1-D). `MetalBuffer::from_borrowed_f32` / `from_f32_slice` and `MlxDtype` are confirmed real.

**Files:**
- Create: `crates/polars-metal-core/src/vector_search.rs`
- Modify: `crates/polars-metal-core/src/lib.rs` (`mod vector_search;`)
- Test: `crates/polars-metal-core/src/vector_search.rs` (`#[cfg(test)]`)

Implements the pure-Rust core (no PyO3 yet): given query `(Q,D)` and corpus `(N,D)` host F32
slices, op tag, and k, return **unordered** `(Q,k)` indices (`Vec<i32>`) and gathered metric values
(`Vec<f32>`). cosine = op 0.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_topk_small() {
        // D=2. corpus rows: e0=[1,0], e1=[0,1], e2=[1,1]. query=[1,0].
        // cosine(query,·): e0=1.0, e1=0.0, e2=0.707. top-2 = {e0, e2}.
        let q = [1.0f32, 0.0];                 // (1,2)
        let c = [1.0f32, 0.0,  0.0, 1.0,  1.0, 1.0]; // (3,2)
        let (idx, score) = vector_search_topk(&q, 1, &c, 3, 2, /*k=*/2, OP_COSINE).unwrap();
        assert_eq!(idx.len(), 2);
        assert_eq!(score.len(), 2);
        // Sort the (unordered) result by score desc for a stable assertion.
        let mut pairs: Vec<(i32, f32)> = idx.iter().copied().zip(score.iter().copied()).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        assert_eq!(pairs[0].0, 0);
        assert!((pairs[0].1 - 1.0).abs() < 1e-5);
        assert_eq!(pairs[1].0, 2);
        assert!((pairs[1].1 - 0.70710677).abs() < 1e-5);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p polars-metal-core vector_search::tests::cosine_topk_small -- --test-threads=1`
Expected: FAIL — `vector_search_topk` not found.

- [ ] **Step 3: Implement the core**

Create `crates/polars-metal-core/src/vector_search.rs`:

```rust
//! M6 vector search: MLX-composition top-k over a query×corpus GEMM.
use std::sync::Arc;

use polars_metal_buffer::{MetalBuffer, MetalDevice};
use polars_metal_mlx_sys::array::{
    mlx_array_eval, mlx_array_to_f32_vec, mlx_array_to_i32_vec, mlx_array_view_metal_buffer,
    MlxArrayHandle, MlxDtype,
};
use polars_metal_mlx_sys::elementwise::{mlx_add, mlx_cast, mlx_div, mlx_mul, mlx_neg, mlx_sub};
use polars_metal_mlx_sys::matmul::mlx_matmul;
use polars_metal_mlx_sys::reduce::mlx_sum_axis;
use polars_metal_mlx_sys::shape::{mlx_reshape, mlx_slice, mlx_take_along_axis, mlx_transpose};
use polars_metal_mlx_sys::sort::mlx_argpartition_axis;
use polars_metal_mlx_sys::FfiError;

pub const OP_COSINE: u32 = 0;
pub const OP_KNN_L2: u32 = 1;

/// View a row-major host F32 slice as a 2-D `(rows, cols)` MLX array.
fn view2d(data: &[f32], rows: i64, cols: i64) -> Result<MlxArrayHandle, FfiError> {
    let device = MetalDevice::system_default();
    // SAFETY: `data` outlives every use of the returned handle within this fn's callers,
    // which eval and read back before returning. MetalBuffer borrows, does not own.
    let buf = unsafe { MetalBuffer::from_borrowed_f32(&device, data.as_ptr(), data.len()) }
        .map(Arc::new)?;
    mlx_array_view_metal_buffer(buf, &[rows, cols], MlxDtype::F32)
}

/// L2-normalize rows of `(rows, d)`: `x / sqrt(sum(x^2, axis=1))` (keepdim broadcast).
fn l2_normalize_rows(x: &MlxArrayHandle, rows: i32, d: i32) -> Result<MlxArrayHandle, FfiError> {
    let sq = mlx_mul(x, x)?;
    let ss = mlx_sum_axis(&sq, 1)?;                 // (rows,)
    let ss = mlx_reshape(&ss, &[rows, 1])?;         // (rows,1)
    let norm = polars_metal_mlx_sys::elementwise::mlx_sqrt_fn(&ss)?; // see note below
    mlx_div(x, &norm)
}

/// Compute unordered top-k. Returns `(indices (Q*k, i32), values (Q*k, f32))` row-major.
pub fn vector_search_topk(
    query: &[f32],
    q_rows: i64,
    corpus: &[f32],
    n_rows: i64,
    d: i64,
    k: i64,
    op: u32,
) -> Result<(Vec<i32>, Vec<f32>), FfiError> {
    let q = view2d(query, q_rows, d)?;
    let c = view2d(corpus, n_rows, d)?;

    // metric: (Q,N) similarity (cosine) or squared distance (knn).
    let (metric, partition_on) = match op {
        OP_COSINE => {
            let qn = l2_normalize_rows(&q, q_rows as i32, d as i32)?;
            let cn = l2_normalize_rows(&c, n_rows as i32, d as i32)?;
            let ct = mlx_transpose(&cn, &[1, 0])?; // (D,N)
            let sims = mlx_matmul(&qn, &ct)?;      // (Q,N)
            let neg = mlx_neg(&sims)?;             // argpartition picks SMALLEST → largest sims
            (sims, neg)
        }
        OP_KNN_L2 => {
            // d2 = q2 + c2 - 2 q·cᵀ  (broadcast (Q,1)+(1,N))
            let q2 = mlx_reshape(&mlx_sum_axis(&mlx_mul(&q, &q)?, 1)?, &[q_rows as i32, 1])?;
            let c2 = mlx_reshape(&mlx_sum_axis(&mlx_mul(&c, &c)?, 1)?, &[1, n_rows as i32])?;
            let ct = mlx_transpose(&c, &[1, 0])?;  // (D,N)
            let cross = mlx_matmul(&q, &ct)?;      // (Q,N)
            let two_cross = mlx_add(&cross, &cross)?;
            let d2 = mlx_sub(&mlx_add(&q2, &c2)?, &two_cross)?;
            (d2.clone(), d2) // knn partitions on the distance directly (smallest)
        }
        _ => return Err(FfiError::from_msg("unknown vector-search op")),
    };

    // argpartition along LAST axis (axis=-1) → (Q,N) indices; take first k columns.
    // NOTE: must use the axis-aware wrapper; the bare mlx_argpartition flattens to 1-D (Task 0).
    let part = mlx_argpartition_axis(&partition_on, (k - 1) as i32, -1)?;
    let idx_k = mlx_slice(&part, &[0, 0], &[q_rows as i32, k as i32], &[1, 1])?; // (Q,k)
    let idx_k_i = mlx_cast(&idx_k, MlxDtype::I32)?;
    // gather the metric values at those indices.
    let val_k = mlx_take_along_axis(&metric, &idx_k_i, 1)?; // (Q,k)

    mlx_array_eval(&[idx_k_i.clone(), val_k.clone()])?;
    let indices = mlx_array_to_i32_vec(&idx_k_i)?;
    let values = mlx_array_to_f32_vec(&val_k)?;
    Ok((indices, values))
}
```

Notes for the implementer:
- `mlx_sqrt_fn` stands for the existing sqrt wrapper. Find its real name in
  `crates/polars-metal-mlx-sys/src/elementwise.rs` (it wraps `mlx_op_sqrt`, lib.rs:200) and use
  that exact name. If only the FFI `ffi::mlx_op_sqrt` exists with no Rust wrapper, add a one-line
  `pub fn mlx_sqrt(a: &MlxArrayHandle) -> Result<MlxArrayHandle, FfiError>` to `elementwise.rs`
  mirroring `mlx_neg`.
- Confirm `mlx_add`/`mlx_sub`/`mlx_mul`/`mlx_div`/`mlx_neg`/`mlx_cast` wrapper names against
  `elementwise.rs` (Task-0 exploration listed the FFI ops; the Rust wrappers use a macro — names
  may be e.g. `mlx_add`). Adjust imports to the real names.
- `FfiError::from_msg` may not exist; use the crate's actual error constructor (check `lib.rs`),
  or return the first available variant.
- In `lib.rs` add `mod vector_search;` (near the other `mod` declarations).

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p polars-metal-core vector_search::tests::cosine_topk_small -- --test-threads=1`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-core/src/vector_search.rs crates/polars-metal-core/src/lib.rs
git commit -m "feat(core): vector_search_topk cosine path via MLX composition (M6)"
```

---

### Task 7: knn (L2) path test

**Files:**
- Test: `crates/polars-metal-core/src/vector_search.rs` (`#[cfg(test)]`)

The knn branch is already implemented in Task 6; lock it with a test (TDD-after for the branch).

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn knn_l2_small() {
        // D=2. corpus e0=[0,0], e1=[3,4], e2=[1,0]. query=[0,0].
        // squared dists: e0=0, e1=25, e2=1. top-2 nearest = {e0, e2}.
        let q = [0.0f32, 0.0];
        let c = [0.0f32, 0.0,  3.0, 4.0,  1.0, 0.0];
        let (idx, d2) = vector_search_topk(&q, 1, &c, 3, 2, 2, OP_KNN_L2).unwrap();
        let mut pairs: Vec<(i32, f32)> = idx.iter().copied().zip(d2.iter().copied()).collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap()); // ascending (nearest)
        assert_eq!(pairs[0].0, 0);
        assert!(pairs[0].1.abs() < 1e-4);          // squared distance 0
        assert_eq!(pairs[1].0, 2);
        assert!((pairs[1].1 - 1.0).abs() < 1e-4);  // squared distance 1 (sqrt applied later, host)
    }
```

- [ ] **Step 2: Run to verify it passes (branch already implemented)**

Run: `cargo test -p polars-metal-core vector_search::tests::knn_l2_small -- --test-threads=1`
Expected: PASS. If it fails on the broadcast shapes, verify `mlx_add` broadcasts `(Q,1)+(1,N)`;
if not, `mlx_reshape`/expand explicitly. (knn core returns **squared** distance; host applies `sqrt`.)

- [ ] **Step 3: Commit**

```bash
git add crates/polars-metal-core/src/vector_search.rs
git commit -m "test(core): knn L2 path for vector_search_topk (M6)"
```

---

### Task 8: Tiling over N (large corpus)

**Files:**
- Modify: `crates/polars-metal-core/src/vector_search.rs`
- Test: `crates/polars-metal-core/src/vector_search.rs`

Wrap `vector_search_topk` in `vector_search_topk_tiled` that splits the corpus into row-tiles when
`Q*N*4` bytes exceeds a threshold, runs each tile, and merges per-query top-k on host (with tile
index offset). For correctness we test that tiling produces the same result as non-tiled.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn tiling_matches_untiled() {
        // 6 corpus rows, force a tiny tile size so multiple tiles run.
        let q = [1.0f32, 0.0];
        let c = [1.0f32,0.0, 0.0,1.0, 1.0,1.0, 0.9,0.1, 0.2,0.2, 0.95,0.0]; // (6,2)
        let (i_ref, s_ref) = vector_search_topk(&q, 1, &c, 6, 2, 3, OP_COSINE).unwrap();
        let (i_t, s_t) = vector_search_topk_tiled(&q, 1, &c, 6, 2, 3, OP_COSINE, /*tile_rows=*/2).unwrap();
        // Compare as score-sorted sets.
        let norm = |idx: &[i32], sc: &[f32]| {
            let mut p: Vec<(i32,i32)> = idx.iter().zip(sc).map(|(i,s)| (*i, (s*1e4) as i32)).collect();
            p.sort();
            p
        };
        assert_eq!(norm(&i_ref, &s_ref), norm(&i_t, &s_t));
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p polars-metal-core vector_search::tests::tiling_matches_untiled -- --test-threads=1`
Expected: FAIL — `vector_search_topk_tiled` not found.

- [ ] **Step 3: Implement tiling**

Append to `vector_search.rs`:

```rust
/// Default tile threshold: 256 MiB of (Q,N) F32 similarity matrix.
pub const TILE_BYTES: usize = 256 * 1024 * 1024;

/// Top-k with corpus row-tiling. `tile_rows` caps corpus rows per GPU pass.
/// Merges per-query partial top-k on host, correcting indices by tile offset.
pub fn vector_search_topk_tiled(
    query: &[f32],
    q_rows: i64,
    corpus: &[f32],
    n_rows: i64,
    d: i64,
    k: i64,
    op: u32,
    tile_rows: i64,
) -> Result<(Vec<i32>, Vec<f32>), FfiError> {
    if tile_rows >= n_rows {
        return vector_search_topk(query, q_rows, corpus, n_rows, d, k, op);
    }
    let kk = k.min(n_rows) as usize;
    // Per-query running top-k as (index, value); kept short (≤ k).
    let mut best: Vec<Vec<(i32, f32)>> = vec![Vec::new(); q_rows as usize];
    let mut offset: i64 = 0;
    while offset < n_rows {
        let rows = (n_rows - offset).min(tile_rows);
        let start = (offset * d) as usize;
        let end = ((offset + rows) * d) as usize;
        let (idx, val) = vector_search_topk(query, q_rows, &corpus[start..end], rows, d,
                                            kk.min(rows as usize) as i64, op)?;
        let per = kk.min(rows as usize);
        for qi in 0..q_rows as usize {
            for j in 0..per {
                let global_idx = idx[qi * per + j] + offset as i32;
                best[qi].push((global_idx, val[qi * per + j]));
            }
            // Keep only top-k by op order; cosine=desc, knn(squared)=asc.
            if op == OP_COSINE {
                best[qi].sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            } else {
                best[qi].sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(a.0.cmp(&b.0)));
            }
            best[qi].truncate(kk);
        }
        offset += rows;
    }
    let mut out_idx = Vec::with_capacity(q_rows as usize * kk);
    let mut out_val = Vec::with_capacity(q_rows as usize * kk);
    for qi in 0..q_rows as usize {
        for (i, v) in &best[qi] {
            out_idx.push(*i);
            out_val.push(*v);
        }
    }
    Ok((out_idx, out_val))
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cargo test -p polars-metal-core vector_search::tests::tiling_matches_untiled -- --test-threads=1`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-core/src/vector_search.rs
git commit -m "feat(core): corpus row-tiling with host top-k merge (M6)"
```

---

### Task 9: PyO3 binding `execute_vector_search`

**Files:**
- Modify: `crates/polars-metal-core/src/vector_search.rs` (add `#[pyfunction]`)
- Modify: `crates/polars-metal-core/src/lib.rs` (register)
- Test: deferred to Python (Task 15) — this task ends at a successful `maturin develop`.

Expose the tiled core to Python with the established `(ptr, len)` buffer convention. Returns two
flat numpy-friendly `Vec`s via PyO3 (indices as `Vec<u32>`, values as `Vec<f32>`), shape `(Q*k)`.

- [ ] **Step 1: Add the pyfunction**

Append to `vector_search.rs`:

```rust
use pyo3::prelude::*;

/// PyO3 entry: `_native.execute_vector_search(query, q_rows, corpus, n_rows, d, k, op, tile_rows)`.
/// `query`/`corpus` are `(ptr, len)` of contiguous row-major F32. Returns `(indices, values)`
/// each length `q_rows*min(k,n_rows)`, row-major. `op`: 0=cosine, 1=knn(L2²). Values are raw
/// metric (cosine sim / squared L2); the Python layer applies `sqrt` for knn and sorts each row.
#[pyfunction]
#[pyo3(signature = (query, q_rows, corpus, n_rows, d, k, op, tile_rows))]
#[allow(clippy::too_many_arguments)]
pub fn execute_vector_search(
    query: (usize, usize),
    q_rows: i64,
    corpus: (usize, usize),
    n_rows: i64,
    d: i64,
    k: i64,
    op: u32,
    tile_rows: i64,
) -> PyResult<(Vec<u32>, Vec<f32>)> {
    let (qptr, qlen) = query;
    let (cptr, clen) = corpus;
    // SAFETY: Python guarantees these point to contiguous F32 arrays of the given lengths,
    // kept alive (via numpy arrays / rechunked Series) for the duration of the call.
    let qslice = unsafe { std::slice::from_raw_parts(qptr as *const f32, qlen) };
    let cslice = unsafe { std::slice::from_raw_parts(cptr as *const f32, clen) };
    let (idx, val) = vector_search_topk_tiled(qslice, q_rows, cslice, n_rows, d, k, op, tile_rows)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("vector search: {e}")))?;
    let idx_u32: Vec<u32> = idx.into_iter().map(|i| i as u32).collect();
    Ok((idx_u32, val))
}
```

In `crates/polars-metal-core/src/lib.rs`, register near the other functions (~line 72):

```rust
    m.add_function(wrap_pyfunction!(vector_search::execute_vector_search, m)?)?;
```

- [ ] **Step 2: Build the wheel**

Run: `make wheel`
Expected: `maturin develop --release` succeeds; no compile errors.

- [ ] **Step 3: Smoke-test the binding from Python**

Run:
```bash
python -c "
import numpy as np, polars_metal._native as n
q = np.array([1,0, 0,1], dtype=np.float32)      # (2,2)
c = np.array([1,0, 0,1, 1,1], dtype=np.float32)  # (3,2)
idx, val = n.execute_vector_search((q.ctypes.data, q.size), 2, (c.ctypes.data, c.size), 3, 2, 2, 0, 1<<30)
print(idx, val)
"
```
Expected: prints 4 indices and 4 values (2 queries × k=2), e.g. query0 top-2 includes index 0.

- [ ] **Step 4: Commit**

```bash
git add crates/polars-metal-core/
git commit -m "feat(core): execute_vector_search PyO3 binding (M6)"
```

---

## Phase 2 — Python expression namespace + capture

### Task 10: Corpus capture + sentinel builder

**Files:**
- Create: `python/polars_metal/_vector_namespace.py`
- Test: `tests/python_integration/test_vector_search.py` (create; capture-only tests here)

Build the module-global handle-id capture and the sentinel expression. The sentinel must (a) be
serialize-detectable, (b) carry the query column + handle-id, (c) **raise on plain CPU**.

- [ ] **Step 1: Write the failing test**

```python
import polars as pl
import pytest
from polars_metal import _vector_namespace as vns


def test_capture_assigns_unique_handles_and_stores_corpus():
    corpus = pl.DataFrame({"emb": [[1.0, 0.0], [0.0, 1.0]]},
                          schema={"emb": pl.Array(pl.Float32, 2)}).lazy()
    h1 = vns._capture_corpus(corpus, "emb", k=5, metric="cosine")
    h2 = vns._capture_corpus(corpus, "emb", k=5, metric="cosine")
    assert h1 != h2
    spec = vns._peek_capture(h1)            # non-destructive peek for the test
    assert spec.corpus_col == "emb"
    assert spec.k == 5 and spec.metric == "cosine"


def test_sentinel_raises_on_plain_cpu():
    df = pl.DataFrame({"emb": [[1.0, 0.0]]}, schema={"emb": pl.Array(pl.Float32, 2)})
    corpus = df.lazy()
    expr = pl.col("emb").metal.cosine_topk(corpus, k=1)  # registered in Task 11
    with pytest.raises(Exception):
        df.lazy().with_columns(expr.alias("hits")).collect()   # no engine="metal"
```

(The second test depends on Task 11's registration; it will be skipped/xfail until then — keep it
in this file and it goes green after Task 11.)

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/python_integration/test_vector_search.py::test_capture_assigns_unique_handles_and_stores_corpus -v`
Expected: FAIL — module/functions not found.

- [ ] **Step 3: Implement the capture + sentinel**

Create `python/polars_metal/_vector_namespace.py`:

```python
"""M6 vector search: `.metal` expression namespace, corpus capture, sentinel builder.

User surface:
    pl.col("emb").metal.cosine_topk(corpus_lf, k, corpus_col="emb")
    pl.col("emb").metal.knn(corpus_lf, k, corpus_col="emb")

These return a *sentinel* expression that:
  - is serialize-detectable (carries the query column + an integer handle-id),
  - raises on plain-CPU collect (an opaque map_batches marker),
  - is recognized + dispatched to the GPU by collect(engine="metal").

The corpus (a LazyFrame / DataFrame / numpy array) is held by-reference in a module-global
dict keyed by the handle-id, with pop-on-consume eviction at dispatch (mirrors M5 rolling).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import polars as pl

_HANDLE_COUNTER = itertools.count(1)


@dataclass(frozen=True)
class CorpusSpec:
    corpus: Any        # LazyFrame | DataFrame | numpy ndarray (by-reference)
    corpus_col: str
    k: int
    metric: str        # "cosine" | "knn"
    query_col: str


_CORPUS_CACHE: dict[int, CorpusSpec] = {}

# Magic prefix embedded in the sentinel's output alias-independent literal so the
# serialize detector can find our bindings unambiguously.
SENTINEL_TAG = "__pm_vsearch__"


def _capture_corpus(corpus: Any, corpus_col: str, k: int, metric: str,
                    query_col: str = "") -> int:
    handle = next(_HANDLE_COUNTER)
    _CORPUS_CACHE[handle] = CorpusSpec(corpus, corpus_col, k, metric, query_col)
    return handle


def _peek_capture(handle: int) -> CorpusSpec:
    return _CORPUS_CACHE[handle]


def pop_capture(handle: int) -> CorpusSpec | None:
    return _CORPUS_CACHE.pop(handle, None)


def _raise_cpu(_s: pl.Series) -> pl.Series:
    raise RuntimeError(
        "polars_metal: .metal.cosine_topk/.knn require collect(engine='metal'); "
        "they have no CPU implementation."
    )


def build_sentinel(query_col_expr: pl.Expr, query_col_name: str, handle: int) -> pl.Expr:
    """Build the recognizable, CPU-raising sentinel struct expression.

    Shape (serialized): a struct with three fields:
      - field 0: the query column (so the detector reads the input column name),
      - field 1: a literal i64 handle-id tagged with SENTINEL_TAG via its alias,
      - field 2: an opaque map_batches(_raise) over the query column → raises on CPU.
    Under engine="metal", dispatch DROPS this output column before the CPU collect, so the
    map_batches never executes; on plain CPU it executes and raises.
    """
    return pl.struct(
        [
            query_col_expr.alias("__pm_vs_query"),
            pl.lit(handle, dtype=pl.Int64).alias(f"{SENTINEL_TAG}{query_col_name}"),
            query_col_expr.map_batches(_raise_cpu, return_dtype=pl.Float32).alias("__pm_vs_raise"),
        ]
    )
```

- [ ] **Step 4: Run to verify the capture test passes**

Run: `pytest tests/python_integration/test_vector_search.py::test_capture_assigns_unique_handles_and_stores_corpus -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/polars_metal/_vector_namespace.py tests/python_integration/test_vector_search.py
git commit -m "feat(py): vector-search corpus capture + CPU-raising sentinel (M6)"
```

---

### Task 11: Register the `metal` expression namespace

**Files:**
- Modify: `python/polars_metal/_vector_namespace.py`
- Modify: `python/polars_metal/__init__.py` (import to trigger registration)
- Test: `tests/python_integration/test_vector_search.py`

- [ ] **Step 1: Write the failing test**

```python
def test_namespace_methods_build_sentinel_structs():
    corpus = pl.DataFrame({"emb": [[1.0, 0.0]]},
                          schema={"emb": pl.Array(pl.Float32, 2)}).lazy()
    e1 = pl.col("emb").metal.cosine_topk(corpus, k=3)
    e2 = pl.col("emb").metal.knn(corpus, k=3)
    # Both serialize to a struct ("as_struct") carrying our tag.
    assert vns.SENTINEL_TAG in e1.meta.serialize(format="json")
    assert vns.SENTINEL_TAG in e2.meta.serialize(format="json")
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/python_integration/test_vector_search.py::test_namespace_methods_build_sentinel_structs -v`
Expected: FAIL — `metal` namespace not registered (`AttributeError`).

- [ ] **Step 3: Register the namespace**

Append to `_vector_namespace.py`:

```python
@pl.api.register_expr_namespace("metal")
class MetalExprNamespace:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def _query_col_name(self) -> str:
        # Best-effort: the root column name drives detection. meta.root_names() returns the
        # input column(s); we require exactly one.
        roots = self._expr.meta.root_names()
        if len(roots) != 1:
            raise ValueError(
                "polars_metal: .metal.cosine_topk/.knn must be applied to a single column "
                f"(got roots {roots})."
            )
        return roots[0]

    def cosine_topk(self, corpus: Any, k: int, corpus_col: str = "emb") -> pl.Expr:
        if k < 1:
            raise ValueError("k must be >= 1")
        qcol = self._query_col_name()
        handle = _capture_corpus(corpus, corpus_col, k, "cosine", qcol)
        return build_sentinel(self._expr, qcol, handle)

    def knn(self, corpus: Any, k: int, corpus_col: str = "emb") -> pl.Expr:
        if k < 1:
            raise ValueError("k must be >= 1")
        qcol = self._query_col_name()
        handle = _capture_corpus(corpus, corpus_col, k, "knn", qcol)
        return build_sentinel(self._expr, qcol, handle)
```

In `python/polars_metal/__init__.py`, near the other module imports (~line 13), add:

```python
from polars_metal import _vector_namespace as _vector_namespace_module  # noqa: F401  (registers .metal)
```

- [ ] **Step 4: Run to verify it passes (and the earlier CPU-raise test)**

Run: `pytest tests/python_integration/test_vector_search.py::test_namespace_methods_build_sentinel_structs tests/python_integration/test_vector_search.py::test_sentinel_raises_on_plain_cpu -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add python/polars_metal/_vector_namespace.py python/polars_metal/__init__.py tests/python_integration/test_vector_search.py
git commit -m "feat(py): register .metal expr namespace (cosine_topk / knn) (M6)"
```

---

## Phase 3 — Detection + collect-and-stitch dispatch

### Task 12: Serialize-detect sentinel bindings

**Files:**
- Create: `python/polars_metal/_vector_detect.py`
- Test: `tests/python_integration/test_vector_search.py`

Mirror `_rolling_detect`: capture `with_columns` exprs (fast path) / serialize-parse (fallback) and
return `VectorBinding(out_name, query_col, handle)` for each sentinel in the outermost layer.

- [ ] **Step 1: Write the failing test**

```python
from polars_metal import _vector_detect as vdet


def test_detect_finds_sentinel_binding():
    df = pl.DataFrame({"id": [0, 1], "emb": [[1.0, 0.0], [0.0, 1.0]]},
                      schema={"id": pl.Int64, "emb": pl.Array(pl.Float32, 2)})
    corpus = df.lazy()
    lf = df.lazy().with_columns(pl.col("emb").metal.cosine_topk(corpus, k=1).alias("hits"))
    bindings = vdet.find_vector_bindings(lf)
    assert len(bindings) == 1
    b = bindings[0]
    assert b.out_name == "hits"
    assert b.query_col == "emb"
    assert b.handle in vns._CORPUS_CACHE  # not yet popped
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/python_integration/test_vector_search.py::test_detect_finds_sentinel_binding -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement detection**

Create `python/polars_metal/_vector_detect.py`:

```python
"""M6 vector search: detect sentinel bindings from a LazyFrame's outermost with_columns layer.

Reuses the M5 detection strategy:
  - Fast path: a `with_columns` monkey-patch records the Python Expr objects keyed by result
    LazyFrame id(); we serialize each expr individually (tiny) and look for our SENTINEL_TAG.
  - Slow fallback: lf.explain() pre-filter, then a bounded parse for the tag.

We never json.loads the full plan (it embeds the DataFrame at scale — the M5 gotcha).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import polars as pl
import polars.lazyframe.frame as _plf

from polars_metal._vector_namespace import SENTINEL_TAG

_lf_exprs_cache: dict[int, list[pl.Expr]] = {}
_PATCH_ATTR = "_polars_metal_vs_original_with_columns"

if not hasattr(_plf.LazyFrame, _PATCH_ATTR):
    _orig_wc = _plf.LazyFrame.with_columns
    setattr(_plf.LazyFrame, _PATCH_ATTR, _orig_wc)

    def _patched_wc(self, *exprs, **named):  # type: ignore[no-untyped-def]
        result = _orig_wc(self, *exprs, **named)
        try:
            flat: list[pl.Expr] = [e for e in exprs if isinstance(e, pl.Expr)]
            flat += [e.alias(n) for n, e in named.items() if isinstance(e, pl.Expr)]
            if flat:
                _lf_exprs_cache[id(result)] = flat
        except Exception:
            pass
        return result

    _plf.LazyFrame.with_columns = _patched_wc  # type: ignore[method-assign]


@dataclass(frozen=True)
class VectorBinding:
    out_name: str
    query_col: str
    handle: int


def _binding_from_expr_json(expr_json: dict, out_name: str) -> VectorBinding | None:
    """Find the SENTINEL_TAG literal + query column inside a serialized struct expr."""
    try:
        s = json.dumps(expr_json)
        if SENTINEL_TAG not in s:
            return None
        # The tag is the alias of the Int64 literal field: f"{SENTINEL_TAG}{query_col}".
        # Walk the as_struct field aliases to recover query_col + the literal handle value.
        fields = _struct_fields(expr_json)
        query_col = None
        handle = None
        for fld in fields:
            alias_name = _alias_name(fld)
            if alias_name and alias_name.startswith(SENTINEL_TAG):
                query_col = alias_name[len(SENTINEL_TAG):]
                handle = _literal_int(fld)
        if query_col is None or handle is None:
            return None
        return VectorBinding(out_name=out_name, query_col=query_col, handle=handle)
    except Exception:
        return None


def _struct_fields(expr_json: dict) -> list:
    """Return the list of field-expr nodes of an as_struct Function, else []."""
    fn = expr_json.get("Function")
    if isinstance(fn, dict):
        inp = fn.get("input")
        if isinstance(inp, list):
            return inp
    return []


def _alias_name(node) -> str | None:
    if isinstance(node, dict):
        a = node.get("Alias")
        if isinstance(a, list) and len(a) == 2 and isinstance(a[1], str):
            return a[1]
    return None


def _literal_int(node) -> int | None:
    """Extract the Int64 handle from an Alias([Literal, name]) node.

    CONFIRMED at py-1.40.1 (Phase 2): the shape is
        {"Literal": {"Scalar": {"Int64": <value>}}}
    i.e. value at node["Alias"][0]["Literal"]["Scalar"]["Int64"]. We match that
    primarily, with a couple of legacy/fallback shapes for resilience.
    """
    if isinstance(node, dict):
        a = node.get("Alias")
        if isinstance(a, list) and len(a) == 2 and isinstance(a[0], dict):
            lit = a[0].get("Literal")
            if isinstance(lit, dict):
                # Primary (py-1.40.1): {"Scalar": {"Int64": N}}
                scalar = lit.get("Scalar")
                if isinstance(scalar, dict):
                    for key in ("Int64", "Int32", "Int"):
                        v = scalar.get(key)
                        if isinstance(v, int):
                            return v
                # Fallbacks for other Polars revs.
                for key in ("Int64", "Int32", "Int"):
                    v = lit.get(key)
                    if isinstance(v, int):
                        return v
                    if isinstance(v, dict) and isinstance(v.get("Int"), int):
                        return v["Int"]
            if isinstance(lit, int):
                return lit
    return None


def find_vector_bindings(lf: pl.LazyFrame) -> list[VectorBinding]:
    """Return VectorBinding for each sentinel alias in the outermost with_columns layer."""
    try:
        cached = _lf_exprs_cache.pop(id(lf), None)
        if cached is not None:
            out: list[VectorBinding] = []
            for expr in cached:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    j = json.loads(expr.meta.serialize(format="json"))
                name = _alias_name(j)
                inner = j["Alias"][0] if name else j
                b = _binding_from_expr_json(inner, name or "")
                if b is not None and b.out_name:
                    out.append(b)
            return out

        # Slow fallback: pre-filter then bounded scan.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            if SENTINEL_TAG not in lf.explain():
                return []
            plan = lf.serialize(format="json")
        # Bounded parse of the exprs fragment (same rfind trick as _rolling_detect).
        key = '"exprs":['
        i = plan.rfind(key)
        if i == -1:
            return []
        start = i + len(key) - 1
        j = plan.rfind(',"options":', start)
        frag = plan[start:j] if j != -1 else plan[start:]
        nodes = json.loads(frag)
        out = []
        for node in nodes if isinstance(nodes, list) else []:
            name = _alias_name(node)
            inner = node["Alias"][0] if name else node
            b = _binding_from_expr_json(inner, name or "")
            if b is not None and b.out_name:
                out.append(b)
        return out
    except Exception:
        return []
```

Note: the exact `Literal`/`Int` JSON shape at py-1.40.1 must be confirmed empirically. Before
relying on `_literal_int`, run `pl.lit(7, dtype=pl.Int64).alias("x").meta.serialize(format="json")`
and adjust the key probing to match. (This is the one place the serialized shape is version-fragile.)

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/python_integration/test_vector_search.py::test_detect_finds_sentinel_binding -v`
Expected: PASS. If `_literal_int` returns None, fix the JSON key probing per the note above, then re-run.

- [ ] **Step 5: Commit**

```bash
git add python/polars_metal/_vector_detect.py tests/python_integration/test_vector_search.py
git commit -m "feat(py): serialize-detect vector-search sentinel bindings (M6)"
```

---

### Task 13: Collect-and-stitch dispatch (build the struct column)

**Files:**
- Create: `python/polars_metal/_vector_dispatch.py`
- Test: `tests/python_integration/test_vector_search.py`

Given bindings + the collected query frame, run the GPU op and build the
`Struct{indices: List[UInt32], scores: List[Float32]}` column, then stitch into the result frame
(mirrors `_rolling_dispatch.apply_rolling`).

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from polars_metal import _vector_dispatch as vdisp


def test_dispatch_builds_struct_column_cosine():
    corpus = pl.DataFrame(
        {"emb": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
        schema={"emb": pl.Array(pl.Float32, 2)},
    ).lazy()
    qframe = pl.DataFrame(
        {"id": [0], "emb": [[1.0, 0.0]]},
        schema={"id": pl.Int64, "emb": pl.Array(pl.Float32, 2)},
    )
    lf = qframe.lazy().with_columns(
        pl.col("emb").metal.cosine_topk(corpus, k=2).alias("hits")
    )
    bindings = vdet.find_vector_bindings(lf)
    df = vdisp.apply_vector_search(lf, bindings, collect_fn=lambda rest: rest.collect())
    assert df.columns == ["id", "emb", "hits"]
    hits = df.get_column("hits")
    assert hits.dtype == pl.Struct(
        {"indices": pl.List(pl.UInt32), "scores": pl.List(pl.Float32)}
    )
    row = hits[0]
    assert list(row["indices"]) == [0, 2]   # cosine: e0=1.0 then e2=0.707, desc
    assert abs(row["scores"][0] - 1.0) < 1e-5
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/python_integration/test_vector_search.py::test_dispatch_builds_struct_column_cosine -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement dispatch**

Create `python/polars_metal/_vector_dispatch.py`:

```python
"""M6 vector search: execute detected bindings on the GPU and stitch struct columns in.

Collect-and-stitch over whole, materialized columns (chunk-safe), mirroring _rolling_dispatch:
  1. drop the sentinel output columns → CPU-collect the rest (projection pushdown elides them),
  2. for each binding: materialize the query column + collect the corpus (pushdown), run the GPU
     op, sort each query's k by metric order, build the Struct column,
  3. reassemble in original schema order.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from polars_metal import _native
from polars_metal._vector_detect import VectorBinding
from polars_metal._vector_namespace import pop_capture

_OP_CODE = {"cosine": 0, "knn": 1}
_TILE_ROWS_DEFAULT = 1 << 30  # effectively no tiling unless the corpus is enormous


def _corpus_matrix(spec_corpus, corpus_col: str) -> tuple[np.ndarray, int, int]:
    """Return (contiguous (N*D) f32, N, D) for the corpus embedding column."""
    if isinstance(spec_corpus, np.ndarray):
        m = np.ascontiguousarray(spec_corpus, dtype=np.float32)
        if m.ndim != 2:
            raise ValueError("numpy corpus must be 2-D (N, D)")
        return m.reshape(-1), m.shape[0], m.shape[1]
    corpus_df = spec_corpus.collect() if isinstance(spec_corpus, pl.LazyFrame) else spec_corpus
    s = corpus_df.get_column(corpus_col)
    return _array_col_to_matrix(s)


def _array_col_to_matrix(s: pl.Series) -> tuple[np.ndarray, int, int]:
    if not isinstance(s.dtype, pl.Array) or s.dtype.inner != pl.Float32:
        raise ValueError(
            f"polars_metal vector search requires Array(Float32, D); got {s.dtype}"
        )
    d = s.dtype.size
    n = s.len()
    flat = s.to_numpy()  # Array(F32, D) → (N, D) ndarray
    m = np.ascontiguousarray(flat, dtype=np.float32).reshape(-1)
    return m, n, d


def _build_struct(indices: np.ndarray, scores: np.ndarray, q_rows: int, k: int,
                  metric: str) -> pl.Series:
    """Sort each query's k by metric order and build Struct{indices, scores}."""
    idx_lists: list[list[int]] = []
    score_lists: list[list[float]] = []
    for qi in range(q_rows):
        ii = indices[qi * k:(qi + 1) * k]
        ss = scores[qi * k:(qi + 1) * k]
        if metric == "knn":
            ss = np.sqrt(np.maximum(ss, 0.0))           # squared → true L2
            order = np.lexsort((ii, ss))                # asc dist, then index asc
        else:
            order = np.lexsort((ii, -ss))               # desc sim, then index asc
        idx_lists.append([int(x) for x in ii[order]])
        score_lists.append([float(x) for x in ss[order]])
    return pl.Series(
        "",
        [{"indices": il, "scores": sl} for il, sl in zip(idx_lists, score_lists)],
        dtype=pl.Struct({"indices": pl.List(pl.UInt32), "scores": pl.List(pl.Float32)}),
    )


def _run_binding(qframe: pl.DataFrame, b: VectorBinding) -> pl.Series:
    spec = pop_capture(b.handle)
    if spec is None:
        raise RuntimeError("polars_metal: vector-search corpus handle missing (already consumed?)")
    qmat, q_rows, qd = _array_col_to_matrix(qframe.get_column(b.query_col).rechunk())
    cmat, n_rows, cd = _corpus_matrix(spec.corpus, spec.corpus_col)
    if qd != cd:
        raise ValueError(f"query D={qd} != corpus D={cd}")
    k = min(spec.k, n_rows)
    idx, val = _native.execute_vector_search(
        (qmat.ctypes.data, qmat.size), q_rows,
        (cmat.ctypes.data, cmat.size), n_rows, qd, k,
        _OP_CODE[spec.metric], _TILE_ROWS_DEFAULT,
    )
    idx = np.asarray(idx, dtype=np.uint32)
    val = np.asarray(val, dtype=np.float32)
    return _build_struct(idx, val, q_rows, k, spec.metric).rename(b.out_name)


def apply_vector_search(lf: pl.LazyFrame, bindings: list[VectorBinding], collect_fn) -> pl.DataFrame:
    out_names = [b.out_name for b in bindings]
    order = lf.collect_schema().names()
    rest_lf = lf.drop(out_names)
    df = collect_fn(rest_lf)
    cols: dict[str, pl.Series] = {c: df.get_column(c) for c in df.columns}
    for b in bindings:
        cols[b.out_name] = _run_binding(df, b)
    return pl.DataFrame([cols[c] for c in order])
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/python_integration/test_vector_search.py::test_dispatch_builds_struct_column_cosine -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/polars_metal/_vector_dispatch.py tests/python_integration/test_vector_search.py
git commit -m "feat(py): vector-search collect-and-stitch dispatch (M6)"
```

---

### Task 14: Wire detection + dispatch into `collect_wrapper`

**Files:**
- Modify: `python/polars_metal/__init__.py` (`collect_wrapper`, ~lines 184-200)
- Test: `tests/python_integration/test_vector_search.py`

- [ ] **Step 1: Write the failing end-to-end test**

```python
def test_end_to_end_cosine_topk_via_engine():
    import polars_metal
    from polars_metal import MetalEngine

    corpus = pl.DataFrame(
        {"emb": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.9, 0.1]]},
        schema={"emb": pl.Array(pl.Float32, 2)},
    ).lazy()
    qframe = pl.DataFrame(
        {"id": [10, 20], "emb": [[1.0, 0.0], [0.0, 1.0]]},
        schema={"id": pl.Int64, "emb": pl.Array(pl.Float32, 2)},
    )
    out = qframe.lazy().with_columns(
        pl.col("emb").metal.cosine_topk(corpus, k=2).alias("hits")
    ).collect(engine=MetalEngine())

    assert out.columns == ["id", "emb", "hits"]
    assert out.get_column("hits").dtype == pl.Struct(
        {"indices": pl.List(pl.UInt32), "scores": pl.List(pl.Float32)}
    )
    # query 0 = [1,0]: nearest is corpus[0]; query 1 = [0,1]: nearest is corpus[1].
    assert out["hits"][0]["indices"][0] == 0
    assert out["hits"][1]["indices"][0] == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/python_integration/test_vector_search.py::test_end_to_end_cosine_topk_via_engine -v`
Expected: FAIL — `hits` is still the raw sentinel struct (not dispatched), or the map_batches raises.

- [ ] **Step 3: Wire it in**

In `python/polars_metal/__init__.py`, inside `collect_wrapper`, alongside the rolling block
(after the rolling dispatch, ~line 195), add vector-search detection **before** the generic
`original_collect(..., post_opt_callback=cb, ...)` return:

```python
            # M6 vector search: serialize-detected .metal.cosine_topk/.knn sentinels run on GPU.
            from polars_metal import _vector_detect, _vector_dispatch

            vector_bindings = [] if streaming else _vector_detect.find_vector_bindings(self)
            if vector_bindings:
                def _collect_rest_vs(rest_lf: Any) -> Any:
                    return original_collect(rest_lf, engine="cpu", post_opt_callback=cb, **kwargs)

                return _vector_dispatch.apply_vector_search(self, vector_bindings, _collect_rest_vs)
```

(Place this immediately after the `if rolling_bindings: ... return ...` block so both detectors get
a chance; rolling and vector search don't co-occur in the same outermost layer in practice, and if
they did, rolling consumes first — acceptable for the MVP. `streaming` is already computed above.)

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/python_integration/test_vector_search.py::test_end_to_end_cosine_topk_via_engine -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/polars_metal/__init__.py tests/python_integration/test_vector_search.py
git commit -m "feat(py): route .metal vector search through collect(engine=metal) (M6)"
```

---

## Phase 4 — Correctness, raises, benchmark

### Task 15: Differential tests vs numpy oracle

**Files:**
- Test: `tests/python_integration/test_vector_search.py`

- [ ] **Step 1: Write the oracle + randomized tests**

```python
def _oracle_cosine_topk(q: np.ndarray, c: np.ndarray, k: int):
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    cn = c / np.linalg.norm(c, axis=1, keepdims=True)
    sims = qn @ cn.T                       # (Q,N)
    out_idx, out_sc = [], []
    for row in sims:
        order = sorted(range(len(row)), key=lambda j: (-row[j], j))[:k]
        out_idx.append(order)
        out_sc.append([row[j] for j in order])
    return out_idx, out_sc


def _oracle_knn(q: np.ndarray, c: np.ndarray, k: int):
    d = np.sqrt(((q[:, None, :] - c[None, :, :]) ** 2).sum(-1))  # (Q,N) true L2
    out_idx, out_sc = [], []
    for row in d:
        order = sorted(range(len(row)), key=lambda j: (row[j], j))[:k]
        out_idx.append(order)
        out_sc.append([row[j] for j in order])
    return out_idx, out_sc


@pytest.mark.parametrize("metric", ["cosine", "knn"])
@pytest.mark.parametrize("Q,N,D,k", [(1, 5, 3, 2), (4, 50, 8, 5), (3, 17, 4, 17)])
def test_matches_numpy_oracle(metric, Q, N, D, k):
    from polars_metal import MetalEngine
    rng = np.random.default_rng(0)
    qv = rng.standard_normal((Q, D)).astype(np.float32) + 0.1  # avoid zero-norm
    cv = rng.standard_normal((N, D)).astype(np.float32) + 0.1
    corpus = pl.DataFrame({"emb": list(cv)},
                          schema={"emb": pl.Array(pl.Float32, D)}).lazy()
    qframe = pl.DataFrame({"emb": list(qv)}, schema={"emb": pl.Array(pl.Float32, D)})
    verb = "cosine_topk" if metric == "cosine" else "knn"
    out = qframe.lazy().with_columns(
        getattr(pl.col("emb").metal, verb)(corpus, k=k).alias("hits")
    ).collect(engine=MetalEngine())

    oi, osc = (_oracle_cosine_topk if metric == "cosine" else _oracle_knn)(qv, cv, min(k, N))
    for qi in range(Q):
        got_idx = list(out["hits"][qi]["indices"])
        got_sc = list(out["hits"][qi]["scores"])
        assert got_idx == oi[qi], f"q{qi} idx {got_idx} != {oi[qi]}"
        np.testing.assert_allclose(got_sc, osc[qi], rtol=1e-4, atol=1e-4)
```

- [ ] **Step 2: Run**

Run: `pytest tests/python_integration/test_vector_search.py::test_matches_numpy_oracle -v`
Expected: all parametrizations PASS. If an index mismatch appears only on near-ties, widen the
oracle tie-break note and assert score-set equality there; document any tolerance in a comment.

- [ ] **Step 3: Commit**

```bash
git add tests/python_integration/test_vector_search.py
git commit -m "test(py): vector search differential vs numpy oracle (M6)"
```

---

### Task 16: Mismatch / raise tests

**Files:**
- Test: `tests/python_integration/test_vector_search.py`

- [ ] **Step 1: Write the raise tests**

```python
@pytest.mark.parametrize("bad", ["dtype", "dmismatch", "ragged"])
def test_raises_on_bad_inputs(bad):
    from polars_metal import MetalEngine
    if bad == "dtype":
        corpus = pl.DataFrame({"emb": [[1.0, 0.0]]},
                              schema={"emb": pl.Array(pl.Float64, 2)}).lazy()
        qframe = pl.DataFrame({"emb": [[1.0, 0.0]]},
                              schema={"emb": pl.Array(pl.Float64, 2)})
    elif bad == "dmismatch":
        corpus = pl.DataFrame({"emb": [[1.0, 0.0, 0.0]]},
                              schema={"emb": pl.Array(pl.Float32, 3)}).lazy()
        qframe = pl.DataFrame({"emb": [[1.0, 0.0]]},
                              schema={"emb": pl.Array(pl.Float32, 2)})
    else:  # ragged List, not Array
        corpus = pl.DataFrame({"emb": [[1.0, 0.0], [1.0]]}).lazy()
        qframe = pl.DataFrame({"emb": [[1.0, 0.0]]})
    with pytest.raises(Exception):
        qframe.lazy().with_columns(
            pl.col("emb").metal.cosine_topk(corpus, k=1).alias("hits")
        ).collect(engine=MetalEngine())


def test_k_greater_than_n_clamps():
    from polars_metal import MetalEngine
    corpus = pl.DataFrame({"emb": [[1.0, 0.0], [0.0, 1.0]]},
                          schema={"emb": pl.Array(pl.Float32, 2)}).lazy()
    qframe = pl.DataFrame({"emb": [[1.0, 0.0]]}, schema={"emb": pl.Array(pl.Float32, 2)})
    out = qframe.lazy().with_columns(
        pl.col("emb").metal.cosine_topk(corpus, k=10).alias("hits")
    ).collect(engine=MetalEngine())
    assert len(out["hits"][0]["indices"]) == 2  # clamped to N
```

- [ ] **Step 2: Run**

Run: `pytest tests/python_integration/test_vector_search.py -k "raises or clamps" -v`
Expected: PASS. (If a bad-input case currently produces wrong output instead of raising, add the
explicit validation in `_vector_dispatch._array_col_to_matrix` / `_run_binding` — the dtype, D, and
Array-vs-List checks are already there; ensure they fire before the FFI call.)

- [ ] **Step 3: Commit**

```bash
git add tests/python_integration/test_vector_search.py
git commit -m "test(py): vector search raises on bad inputs; k>N clamps (M6)"
```

---

### Task 17: Benchmark + `ratio_lt` gate

**Files:**
- Create: `tests/bench/m4_survey/bench_cosine_topk.py`

- [ ] **Step 1: Write the benchmark**

```python
"""M6 cosine top-k benchmark: Q queries × N corpus × D dims, k neighbours.

Records metal-vs-CPU(numpy) ratio. Gate: metal must be faster than the numpy brute force.
Headline survey number is ~29× at Q=100, N=1M, D=768; this bench uses smaller defaults for CI.
"""

import time

import numpy as np
import polars as pl
import pytest

from polars_metal import MetalEngine


def _make(Q, N, D, seed=0):
    rng = np.random.default_rng(seed)
    qv = rng.standard_normal((Q, D)).astype(np.float32)
    cv = rng.standard_normal((N, D)).astype(np.float32)
    return qv, cv


def _numpy_cosine_topk(qv, cv, k):
    qn = qv / np.linalg.norm(qv, axis=1, keepdims=True)
    cn = cv / np.linalg.norm(cv, axis=1, keepdims=True)
    sims = qn @ cn.T
    return np.argsort(-sims, axis=1)[:, :k]


@pytest.mark.benchmark
def test_bench_cosine_topk(_gate):
    Q, N, D, k = 100, 200_000, 256, 10
    qv, cv = _make(Q, N, D)
    corpus = pl.DataFrame({"emb": list(cv)}, schema={"emb": pl.Array(pl.Float32, D)}).lazy()
    qframe = pl.DataFrame({"emb": list(qv)}, schema={"emb": pl.Array(pl.Float32, D)})

    # warmup (kernel/pipeline build)
    qframe.lazy().with_columns(
        pl.col("emb").metal.cosine_topk(corpus, k=k).alias("h")
    ).collect(engine=MetalEngine())

    t0 = time.perf_counter()
    qframe.lazy().with_columns(
        pl.col("emb").metal.cosine_topk(corpus, k=k).alias("h")
    ).collect(engine=MetalEngine())
    metal_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    _numpy_cosine_topk(qv, cv, k)
    cpu_s = time.perf_counter() - t0

    print(f"cosine_topk Q={Q} N={N} D={D} k={k}: metal {metal_s*1e3:.1f}ms cpu {cpu_s*1e3:.1f}ms "
          f"ratio {cpu_s/metal_s:.1f}x")
    _gate.ratio_lt("phase10_cosine_topk_q100_n200k_d256", metal_s / cpu_s, 1.0)
```

- [ ] **Step 2: Run the benchmark**

Run: `pytest tests/bench/m4_survey/bench_cosine_topk.py -v -s`
Expected: prints a ratio; gate `metal/cpu < 1.0` (metal faster) PASSES. If the `_gate` fixture name
differs, copy the exact fixture import/usage from `tests/bench/m4_survey/bench_haversine_mlx.py`.

- [ ] **Step 3: Record baseline + commit**

```bash
git add tests/bench/m4_survey/bench_cosine_topk.py
git commit -m "bench: cosine top-k metal-vs-numpy + ratio_lt gate (M6)"
```

---

## Final verification

- [ ] **Run the full gate**

Run: `make gate`
Expected: lint clean; `make test-unit`, `make test-kernel`, conformance at the M3 baseline
(only the known `lazyframe` + `operations_group_by` deferrals fail); new vector-search tests green.

- [ ] **Update docs + memory**

- Add a one-paragraph "vector search delivered" note to `docs/open-questions.md` and the M6 spec's
  consolidation section.
- Update `[[m6-scope-and-api-direction]]` memory: A2 shipped, with the measured ratio.

- [ ] **Open the PR**

```bash
git push -u origin m6-vector-search
gh pr create --title "M6: GPU vector search (.metal.cosine_topk / .metal.knn)" \
  --body "Implements §A2 of the M6 .metal-namespace spec. MLX-composition top-k (5 new reusable FFI wrappers), expr namespace + M5-style collect-and-stitch dispatch. F32-only; raises on mismatch."
```

---

## Notes for the implementer

- **Build after every Rust change:** `make wheel` before running Python tests that touch `_native`.
- **Threading:** Rust kernel/FFI tests run with `--test-threads=1` (Metal command-queue contention).
- **Version-fragile spot:** the `Literal`/`Int` JSON shape in `_vector_detect._literal_int` — verify
  empirically at py-1.40.1 (Task 12 note). Everything else rides stable serialize shapes.
- **Don't extend scope:** corpus reuse cache, `df.corr()`, FFT, dt are separate sub-projects. This
  plan ends at a working, tested `cosine_topk`/`knn`.
- **If MLX `argpartition` 2-D semantics differ** from Task 0's finding, fix in `vector_search_topk`
  (reshape so the top-k axis is last) — do not work around it in Python.
