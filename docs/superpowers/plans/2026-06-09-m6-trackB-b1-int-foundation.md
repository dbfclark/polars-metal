# B1 — Integer Buffer/FFI/Subgraph Dtype-Awareness Implementation Plan

**REQUIRED SUB-SKILL: superpowers:subagent-driven-development** — execute this plan task-by-task via dispatched implementer subagents; each task is a self-contained TDD loop (failing test → minimal impl → green → commit), and the orchestrator reviews each task's diff before dispatching the next.

**Goal:** Make the Metal execution engine carry a column's declared integer dtype end-to-end through the fused-expression path — for **all eight** integer dtypes (`Int8/16/32/64`, `UInt8/16/32/64`) — instead of hard-coding `F32`. The runtime already *declares* dtypes (`InputDtype`, `MlxDtype`) but never *consults* them: the subgraph builder pins `MlxDtype::F32`, the Python dispatch force-casts every column to `np.float32`, and the FFI returns an undtyped f32 byte stream. B1 removes those three hard-codes and threads the dtype through the buffer crate, the MLX FFI, the subgraph builder, and the Python fold-back.

**Exit bar (the only acceptance criterion for B1):** an `Int32` **and** an `Int64` column — plus at least one narrow (`Int8`) and one unsigned (`UInt64`) — round-trip through a trivial fused chain `pl.col("x") + 1` under `engine="metal"`, **byte-exact** vs Polars CPU, **nulls preserved**.

**Out of scope (explicit — do NOT plan these):**
- B2 — fused-walker op semantics, reduction upcast (`sum(Int32)→Int64`), int↔int / int↔f32 cast op coverage, mixed-dtype chains beyond the `col + 1` literal case.
- B3 — `dt` gregorian MSL kernel.
- B4 — bare-reduction routing thresholds, re-baselined benchmarks.
- Any dtype-aware change to the `mlx_op_cast` switch beyond the view-path switch (cast is a B2 op).
- The reduction-path F32-only gate in `analyze_ir_reduction` (line ~575) — that is a *reduction* terminator (B2). B1 only opens the **elementwise HStack** path that the exit-bar `col + 1` shape uses.

**Architecture.** The fused HStack path for `col + 1` flows:
`_fusion_analyzer.analyze_ir_with_columns` (Python) builds a `PyFusionScope` (one `InputDtype` per leaf) + an ordered descriptor list → `_udf._dispatch_hstack_fused` (Python) stages each input column/literal as native bytes and calls `_native.execute_fused_expr` → `udf::execute_fused_expr` (Rust/PyO3) stages each `(ptr, n)` into a `MetalBuffer` and builds `MlxSubgraph::from_fusion_scope_buffers` → `subgraph.rs` wraps each buffer as an `MlxArrayHandle` via `mlx_array_view_metal_buffer(buf, shape, MlxDtype)` → eval → reads the output back via dtype-dispatched readback → Python wraps the result `pl.Series` with the correct dtype + null mask. B1 makes every hard-coded `F32` / `/ 4` / `np.float32` in that chain consult the declared dtype.

The **load-bearing seam** is the output-dtype return: today `execute_fused_expr` returns `usize` (count) and writes into a Python-pre-allocated `f32` buffer (output-zero-copy). B1 cannot keep output-zero-copy *and* be dtype-polymorphic, because Python does not know the output dtype before eval. The chosen design (Task 6) **infers the output dtype statically in the analyzer** (the exit-bar chain is monomorphic in its single column's dtype) and threads it as a per-binding `_fused_out_dtype` string, so Python pre-allocates the correct-width output buffer *and* the Rust side validates the eval'd output dtype matches via the new `mlx_array_dtype` query. This keeps output-zero-copy intact and decouples B1 from B2's general output-dtype tracking.

**Tech Stack.** Rust 2021 (engine, FFI), cxx (`#[cxx::bridge]` four-side: Rust extern decl + `.h` decl + `.cc` impl + Rust wrapper), MLX 0.25.1 (C++, vendored — never patched), PyO3 + maturin (wheel via `make wheel`), Python 3 + numpy + pyarrow (fold-back), proptest (Rust round-trip), pytest (Python integration). Polars CPU is the differential oracle.

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `crates/polars-metal-core/src/fusion/scope.rs` | Modify | Add 7 int variants to `InputDtype`; helper `InputDtype::element_size()`. |
| `crates/polars-metal-mlx-sys/src/array.rs` | Modify | Add 7 int variants + tags to `MlxDtype`; extend `element_size()`; add `MlxArrayHandle::dtype()`; per-width `mlx_array_to_<t>_vec` readback wrappers; `InputDtype`→`MlxDtype` is mapped in subgraph.rs (no scope dep here). |
| `crates/polars-metal-mlx-sys/src/lib.rs` | Modify | cxx `extern "C++"` decls for the 7 readback fns + `mlx_array_dtype`. |
| `crates/polars-metal-mlx-sys/cxx/mlx_bridge.h` | Modify | C++ decls (four-side consistency) for the 7 readback fns + `mlx_array_dtype` + shared `mlx_dtype_from_tag`. |
| `crates/polars-metal-mlx-sys/cxx/mlx_bridge.cc` | Modify | Extend the view-path dtype switch (factored into `mlx_dtype_from_tag`); per-width readback impls (macro); `mlx_array_dtype` impl. |
| `crates/polars-metal-buffer/src/bridge.rs` | Modify | Per-width `from_<t>_slice` / `to_<t>_vec` / `from_borrowed_<t>` (macro-generated); proptest round-trip. |
| `crates/polars-metal-core/src/fusion/subgraph.rs` | Modify | `InputDtype`→`MlxDtype` mapping; dtype-aware input handle construction (element_size, not `/4`); output readback dispatch on `mlx_array_dtype`; dtype-aware `MetalBuffer` construction in `eval_to_metal_buffers`. |
| `crates/polars-metal-core/src/fusion/py.rs` | Modify | `add_input` string-match arms for the 7 new int dtypes. |
| `crates/polars-metal-core/src/udf.rs` | Modify | `execute_fused_expr`: accept per-input dtype tags + output dtype tag; stage inputs by width; validate eval'd output dtype; write back width-aware. |
| `python/polars_metal/_fusion_analyzer.py` | Modify | `_dtype_to_input_str` maps all 8 int dtypes; infer + stamp `_fused_out_dtype`; literal dtype carried in descriptors. |
| `python/polars_metal/_udf.py` | Modify | Stage each column/literal as its native dtype; pre-allocate output by `_fused_out_dtype`; pass dtype tags through the FFI; wrap result Series with the right dtype. |
| `crates/polars-metal-mlx-sys/tests/test_int_readback.rs` | Create | Rust integration test: build int arrays, eval, read back per width; `mlx_array_dtype` returns correct tag. |
| `tests/python_integration/test_int_foundation.py` | Create | Exit-bar differential test: `col + 1` for Int8/Int32/Int64/UInt64, with nulls, byte-exact vs CPU. |

---

## Task 1 — `InputDtype` + `MlxDtype` variants + `element_size`

Add the 7 missing integer variants to both enums with stable u32 tags, and a width helper on each. Rust-only; no MLX calls, so this task is pure unit-test.

**Files**
- Modify: `crates/polars-metal-core/src/fusion/scope.rs`
- Modify: `crates/polars-metal-mlx-sys/src/array.rs`
- Test (modify): inline `#[cfg(test)]` in `array.rs`; inline `#[cfg(test)]` in `scope.rs`

Chosen tag assignment (must stay stable — the C++ switch keys off these): `F32=0, F64=1, I32=2, Bool=3, I8=4, I16=5, I64=6, U8=7, U16=8, U32=9, U64=10`.

**Steps**

- [ ] Add a failing unit test in `crates/polars-metal-mlx-sys/src/array.rs` (append to the existing `#[cfg(test)] mod tests`, or create one) asserting tags + widths:
  ```rust
  #[cfg(test)]
  mod dtype_tests {
      use super::MlxDtype;

      #[test]
      fn int_dtype_tags_are_stable() {
          assert_eq!(MlxDtype::I8 as u32, 4);
          assert_eq!(MlxDtype::I16 as u32, 5);
          assert_eq!(MlxDtype::I64 as u32, 6);
          assert_eq!(MlxDtype::U8 as u32, 7);
          assert_eq!(MlxDtype::U16 as u32, 8);
          assert_eq!(MlxDtype::U32 as u32, 9);
          assert_eq!(MlxDtype::U64 as u32, 10);
      }

      #[test]
      fn int_element_sizes() {
          assert_eq!(MlxDtype::I8.element_size(), 1);
          assert_eq!(MlxDtype::I16.element_size(), 2);
          assert_eq!(MlxDtype::I64.element_size(), 8);
          assert_eq!(MlxDtype::U8.element_size(), 1);
          assert_eq!(MlxDtype::U16.element_size(), 2);
          assert_eq!(MlxDtype::U32.element_size(), 4);
          assert_eq!(MlxDtype::U64.element_size(), 8);
      }
  }
  ```
- [ ] Run it, expect FAIL (variants don't exist → compile error):
  `cargo test -p polars-metal-mlx-sys dtype_tests -- --test-threads=1`
  Expected: compile error `no variant named I8` etc.
- [ ] Implement in `crates/polars-metal-mlx-sys/src/array.rs` — extend the enum and `element_size`:
  ```rust
  #[derive(Debug, Clone, Copy, PartialEq, Eq)]
  #[repr(u32)]
  pub enum MlxDtype {
      F32 = 0,
      /// Not supported in MLX (no float64 on Metal); returns `Err` if passed
      /// to `mlx_array_view_metal_buffer`. Kept for forward compatibility.
      F64 = 1,
      I32 = 2,
      Bool = 3,
      I8 = 4,
      I16 = 5,
      I64 = 6,
      U8 = 7,
      U16 = 8,
      U32 = 9,
      U64 = 10,
  }

  impl MlxDtype {
      /// Size of one element in bytes.
      pub fn element_size(self) -> usize {
          match self {
              MlxDtype::Bool | MlxDtype::I8 | MlxDtype::U8 => 1,
              MlxDtype::I16 | MlxDtype::U16 => 2,
              MlxDtype::F32 | MlxDtype::I32 | MlxDtype::U32 => 4,
              MlxDtype::F64 | MlxDtype::I64 | MlxDtype::U64 => 8,
          }
      }
  }
  ```
- [ ] Add a failing unit test in `crates/polars-metal-core/src/fusion/scope.rs` (append a `#[cfg(test)] mod scope_dtype_tests`):
  ```rust
  #[cfg(test)]
  mod scope_dtype_tests {
      use super::InputDtype;

      #[test]
      fn int_input_dtype_element_sizes() {
          assert_eq!(InputDtype::I8.element_size(), 1);
          assert_eq!(InputDtype::I16.element_size(), 2);
          assert_eq!(InputDtype::I32.element_size(), 4);
          assert_eq!(InputDtype::I64.element_size(), 8);
          assert_eq!(InputDtype::U8.element_size(), 1);
          assert_eq!(InputDtype::U16.element_size(), 2);
          assert_eq!(InputDtype::U32.element_size(), 4);
          assert_eq!(InputDtype::U64.element_size(), 8);
          assert_eq!(InputDtype::F32.element_size(), 4);
      }
  }
  ```
- [ ] Run it, expect FAIL:
  `cargo test -p polars-metal-core scope_dtype_tests -- --test-threads=1`
  Expected: compile error `no variant named I8` and `no method element_size`.
- [ ] Implement in `crates/polars-metal-core/src/fusion/scope.rs` — extend `InputDtype` and add `element_size`:
  ```rust
  #[derive(Clone, Copy, Debug, PartialEq, Eq)]
  pub enum InputDtype {
      F32,
      F64,
      Bool,
      I32,
      I8,
      I16,
      I64,
      U8,
      U16,
      U32,
      U64,
      ArrayF32(usize),
      ListF32,
  }

  impl InputDtype {
      /// Element width in bytes for the scalar numeric dtypes. `ArrayF32(d)`
      /// returns `d * 4`; `ListF32` is variable-length and returns `4` (the
      /// per-element width — callers that need a row stride must not use this).
      pub fn element_size(self) -> usize {
          match self {
              InputDtype::Bool | InputDtype::I8 | InputDtype::U8 => 1,
              InputDtype::I16 | InputDtype::U16 => 2,
              InputDtype::F32 | InputDtype::I32 | InputDtype::U32 | InputDtype::ListF32 => 4,
              InputDtype::F64 | InputDtype::I64 | InputDtype::U64 => 8,
              InputDtype::ArrayF32(d) => d * 4,
          }
      }
  }
  ```
- [ ] Run both, expect PASS:
  `cargo test -p polars-metal-mlx-sys dtype_tests -- --test-threads=1` and
  `cargo test -p polars-metal-core scope_dtype_tests -- --test-threads=1`
  Expected: both green.
- [ ] Lint:
  `cargo fmt && cargo clippy -p polars-metal-mlx-sys -p polars-metal-core`
  Expected: no warnings.
- [ ] Commit:
  `git add crates/polars-metal-mlx-sys/src/array.rs crates/polars-metal-core/src/fusion/scope.rs && git commit -m "B1 T1: add 8 integer variants + element_size to MlxDtype and InputDtype"`

---

## Task 2 — C++ bridge: view-switch + per-width readback + `mlx_array_dtype` (four-side cxx)

Extend the `mlx_array_view_mtl_buffer` dtype switch to the 7 new int tags, factor the tag→`mlx::core::Dtype` mapping into a shared `mlx_dtype_from_tag` helper, add per-width readback C++ functions (macro), and add a `mlx_array_dtype` query that returns the MlxDtype tag of an eval'd array. cxx requires all four sides change together: Rust extern decl (`lib.rs`), `.h` decl, `.cc` impl, plus the Rust wrapper (Task 3 covers the safe wrapper; here we wire the raw extern decls + C++).

**Files**
- Modify: `crates/polars-metal-mlx-sys/cxx/mlx_bridge.h`
- Modify: `crates/polars-metal-mlx-sys/cxx/mlx_bridge.cc`
- Modify: `crates/polars-metal-mlx-sys/src/lib.rs` (the `#[cxx::bridge]` block)
- Test (create): `crates/polars-metal-mlx-sys/tests/test_int_readback.rs`

**Steps**

- [ ] Write a failing integration test `crates/polars-metal-mlx-sys/tests/test_int_readback.rs` exercising the raw FFI through the safe wrappers added in Task 3 — but for *this* task scope the test to what Task 2 alone delivers: that the C++ links and `mlx_array_dtype` returns the right tag for an f32 array we already know how to build. (The per-width readback round-trip lands as the Task 3 test; here we only prove the C++ compiles + `mlx_array_dtype` works on the existing f32 path.)
  ```rust
  // crates/polars-metal-mlx-sys/tests/test_int_readback.rs
  use polars_metal_mlx_sys::array::{mlx_array_from_f32_slice, mlx_array_eval, MlxDtype};

  #[test]
  fn dtype_query_reports_f32() {
      let a = mlx_array_from_f32_slice(&[1.0, 2.0, 3.0]).expect("build f32");
      mlx_array_eval(&[a.clone()]).expect("eval");
      assert_eq!(a.dtype(), MlxDtype::F32);
  }
  ```
  (`a.dtype()` is the Task-3 wrapper; this test is written here but only goes green after Task 3. Sequence: this task lands the C++ + extern decls and leaves the test red on the missing wrapper; Task 3 adds the wrapper and turns it green. If you prefer a strictly-green-per-task cadence, defer running this test's assertion until Task 3 and in Task 2 only run `cargo build -p polars-metal-mlx-sys` to prove the C++ links.)
- [ ] Run the build to confirm RED (wrapper not yet present → the test file won't compile; the C++ itself must build):
  `cargo build -p polars-metal-mlx-sys`
  Expected: library builds (C++ compiles); `cargo test -p polars-metal-mlx-sys --test test_int_readback -- --test-threads=1` FAILs to compile on `a.dtype()` / `mlx_array_eval` import until Task 3.
- [ ] In `crates/polars-metal-mlx-sys/cxx/mlx_bridge.h`, add the shared helper decl + readback decls + dtype-query decl (place after the existing `mlx_array_copy_to_i32` decl, before the view-buffer block):
  ```cpp
  // Shared tag → mlx::core::Dtype mapping (tags match MlxDtype in array.rs:
  // 0=f32, 1=f64[unsupported], 2=i32, 3=bool, 4=i8, 5=i16, 6=i64,
  // 7=u8, 8=u16, 9=u32, 10=u64). Throws std::invalid_argument on f64/unknown.
  mlx::core::Dtype mlx_dtype_from_tag(uint32_t tag);

  // Return the MlxDtype tag (the inverse of mlx_dtype_from_tag) of arr's dtype.
  // Throws std::invalid_argument if arr has a dtype we do not map.
  uint32_t mlx_array_dtype(const std::shared_ptr<MlxArray>& arr);

  // Per-width integer readback. Each copies `n` elements of the matching
  // width from the materialized (eval'd) array into the caller's buffer.
  // The array must have the matching dtype (caller contract). Raw memcpy in
  // storage order (row-major-contiguous assumption, same as copy_to_i32).
  void mlx_array_copy_to_i8(const std::shared_ptr<MlxArray>& arr, int8_t* out, size_t n);
  void mlx_array_copy_to_i16(const std::shared_ptr<MlxArray>& arr, int16_t* out, size_t n);
  void mlx_array_copy_to_i64(const std::shared_ptr<MlxArray>& arr, int64_t* out, size_t n);
  void mlx_array_copy_to_u8(const std::shared_ptr<MlxArray>& arr, uint8_t* out, size_t n);
  void mlx_array_copy_to_u16(const std::shared_ptr<MlxArray>& arr, uint16_t* out, size_t n);
  void mlx_array_copy_to_u32(const std::shared_ptr<MlxArray>& arr, uint32_t* out, size_t n);
  void mlx_array_copy_to_u64(const std::shared_ptr<MlxArray>& arr, uint64_t* out, size_t n);
  ```
- [ ] In `crates/polars-metal-mlx-sys/cxx/mlx_bridge.cc`, add the shared helper near the top of the MlxArray section (after `mlx_array_is_f32`, before `mlx_array_copy_to_f32`), and refactor `mlx_array_view_mtl_buffer`'s switch to call it:
  ```cpp
  mlx::core::Dtype mlx_dtype_from_tag(uint32_t tag) {
      switch (tag) {
          case 0:  return mlx::core::float32;
          case 1:
              throw std::invalid_argument(
                  "mlx_dtype_from_tag: float64 is not supported on Metal");
          case 2:  return mlx::core::int32;
          case 3:  return mlx::core::bool_;
          case 4:  return mlx::core::int8;
          case 5:  return mlx::core::int16;
          case 6:  return mlx::core::int64;
          case 7:  return mlx::core::uint8;
          case 8:  return mlx::core::uint16;
          case 9:  return mlx::core::uint32;
          case 10: return mlx::core::uint64;
          default:
              throw std::invalid_argument("mlx_dtype_from_tag: unknown dtype tag");
      }
  }

  uint32_t mlx_array_dtype(const std::shared_ptr<MlxArray>& arr) {
      mlx::core::Dtype dt = arr->dtype();
      if (dt == mlx::core::float32) return 0;
      if (dt == mlx::core::int32)   return 2;
      if (dt == mlx::core::bool_)   return 3;
      if (dt == mlx::core::int8)    return 4;
      if (dt == mlx::core::int16)   return 5;
      if (dt == mlx::core::int64)   return 6;
      if (dt == mlx::core::uint8)   return 7;
      if (dt == mlx::core::uint16)  return 8;
      if (dt == mlx::core::uint32)  return 9;
      if (dt == mlx::core::uint64)  return 10;
      throw std::invalid_argument("mlx_array_dtype: unmapped dtype");
  }
  ```
- [ ] In the same `.cc`, replace the body of the `switch (dtype)` inside `mlx_array_view_mtl_buffer` with a call to the helper:
  ```cpp
      // Map the dtype tag to mlx::core::Dtype (shared with mlx_op_cast et al.).
      mlx::core::Dtype dt = mlx_dtype_from_tag(dtype);
  ```
  (Delete the old inline `switch` and the `mlx::core::Dtype dt = mlx::core::float32; // initialise` line — `mlx_dtype_from_tag` either returns or throws.)
- [ ] In the same `.cc`, add a readback macro + the 7 instantiations (place after `mlx_array_copy_to_i32`):
  ```cpp
  // Per-width integer readback. Same raw-memcpy contract as copy_to_i32:
  // caller guarantees the array is eval'd, has the matching dtype, and `out`
  // holds at least n elements.
  #define MLX_WRAP_READBACK(fn_name, ctype, mlx_ctype)                       \
  void fn_name(const std::shared_ptr<MlxArray>& arr, ctype* out, size_t n) { \
      const mlx_ctype* src = arr->data<mlx_ctype>();                         \
      std::memcpy(out, src, n * sizeof(ctype));                              \
  }
  MLX_WRAP_READBACK(mlx_array_copy_to_i8,  int8_t,   int8_t)
  MLX_WRAP_READBACK(mlx_array_copy_to_i16, int16_t,  int16_t)
  MLX_WRAP_READBACK(mlx_array_copy_to_i64, int64_t,  int64_t)
  MLX_WRAP_READBACK(mlx_array_copy_to_u8,  uint8_t,  uint8_t)
  MLX_WRAP_READBACK(mlx_array_copy_to_u16, uint16_t, uint16_t)
  MLX_WRAP_READBACK(mlx_array_copy_to_u32, uint32_t, uint32_t)
  MLX_WRAP_READBACK(mlx_array_copy_to_u64, uint64_t, uint64_t)
  #undef MLX_WRAP_READBACK
  ```
- [ ] In `crates/polars-metal-mlx-sys/src/lib.rs`, add the cxx `extern "C++"` decls inside the `unsafe extern "C++"` block (place after the existing `mlx_array_copy_to_i32` decl). Note `mlx_dtype_from_tag` is C++-internal and is **not** declared to cxx (it has no Rust caller):
  ```rust
        // M6 Track B (B1): integer dtype query + per-width readback.
        //
        // Return the MlxDtype tag of `arr`'s dtype (0=f32, 2=i32, 3=bool,
        // 4=i8, 5=i16, 6=i64, 7=u8, 8=u16, 9=u32, 10=u64). Throws on an
        // unmapped dtype (e.g. float64), which cxx surfaces as Err.
        fn mlx_array_dtype(arr: &SharedPtr<MlxArray>) -> Result<u32>;

        // Per-width integer readback. Each copies `n` values of the matching
        // width into the caller buffer. Array must be eval'd and have the
        // matching dtype (caller contract).
        // SAFETY: `out` must point to a buffer of at least `n` elements.
        unsafe fn mlx_array_copy_to_i8(arr: &SharedPtr<MlxArray>, out: *mut i8, n: usize);
        unsafe fn mlx_array_copy_to_i16(arr: &SharedPtr<MlxArray>, out: *mut i16, n: usize);
        unsafe fn mlx_array_copy_to_i64(arr: &SharedPtr<MlxArray>, out: *mut i64, n: usize);
        unsafe fn mlx_array_copy_to_u8(arr: &SharedPtr<MlxArray>, out: *mut u8, n: usize);
        unsafe fn mlx_array_copy_to_u16(arr: &SharedPtr<MlxArray>, out: *mut u16, n: usize);
        unsafe fn mlx_array_copy_to_u32(arr: &SharedPtr<MlxArray>, out: *mut u32, n: usize);
        unsafe fn mlx_array_copy_to_u64(arr: &SharedPtr<MlxArray>, out: *mut u64, n: usize);
  ```
  Note: `mlx_array_dtype` returns `Result<u32>` because the C++ throws on an unmapped dtype; cxx requires a `Result`-typed return for any C++ fn declared to throw. The `.h`/`.cc` signatures return `uint32_t` and throw — cxx's generated glue catches the C++ exception and maps it to the Rust `Err` automatically.
- [ ] Run the build to confirm C++ compiles and links:
  `cargo build -p polars-metal-mlx-sys`
  Expected: clean build. (The Task-3 wrapper test still red on `a.dtype()` import — that's expected; it goes green in Task 3.)
- [ ] Lint:
  `cargo fmt && cargo clippy -p polars-metal-mlx-sys`
  Expected: no warnings.
- [ ] Commit:
  `git add crates/polars-metal-mlx-sys/cxx/mlx_bridge.h crates/polars-metal-mlx-sys/cxx/mlx_bridge.cc crates/polars-metal-mlx-sys/src/lib.rs crates/polars-metal-mlx-sys/tests/test_int_readback.rs && git commit -m "B1 T2: C++ bridge — shared dtype helper, per-width readback, mlx_array_dtype query"`

---

## Task 3 — `array.rs`: `MlxArrayHandle::dtype()` + per-width readback wrappers

Add the safe Rust wrappers: `MlxArrayHandle::dtype() -> Result<MlxDtype>` (over `ffi::mlx_array_dtype`) and `mlx_array_to_<t>_vec` per width (mirroring `mlx_array_to_i32_vec`). This turns the Task-2 test green and gives the subgraph builder its output-dtype dispatch primitives.

**Files**
- Modify: `crates/polars-metal-mlx-sys/src/array.rs`
- Test (modify): `crates/polars-metal-mlx-sys/tests/test_int_readback.rs`

**Steps**

- [ ] Extend `crates/polars-metal-mlx-sys/tests/test_int_readback.rs` with a per-width round-trip via the zero-copy view path (build a `MetalBuffer` of int bytes, view it as the matching `MlxDtype`, eval, read back). Add an `i64` and a `u64` case (the widths the f32-only path could never carry):
  ```rust
  use std::sync::Arc;
  use polars_metal_buffer::{MetalBuffer, MetalDevice};
  use polars_metal_mlx_sys::array::{
      mlx_array_eval, mlx_array_to_i64_vec, mlx_array_to_u64_vec,
      mlx_array_view_metal_buffer, MlxDtype,
  };

  #[test]
  fn i64_view_round_trips() {
      let device = MetalDevice::system_default().expect("metal");
      let vals: Vec<i64> = vec![-3, 0, 5, 3_000_000_000, -2_000_000_000];
      let buf = Arc::new(MetalBuffer::from_i64_slice(&device, &vals).expect("stage"));
      let h = mlx_array_view_metal_buffer(buf, &[vals.len() as i64], MlxDtype::I64).expect("view");
      mlx_array_eval(&[h.clone()]).expect("eval");
      assert_eq!(h.dtype().expect("dtype"), MlxDtype::I64);
      assert_eq!(mlx_array_to_i64_vec(&h).expect("readback"), vals);
  }

  #[test]
  fn u64_view_round_trips_beyond_i64_range() {
      let device = MetalDevice::system_default().expect("metal");
      let vals: Vec<u64> = vec![0, 1, u64::MAX, 10_000_000_000_000_000_000];
      let buf = Arc::new(MetalBuffer::from_u64_slice(&device, &vals).expect("stage"));
      let h = mlx_array_view_metal_buffer(buf, &[vals.len() as i64], MlxDtype::U64).expect("view");
      mlx_array_eval(&[h.clone()]).expect("eval");
      assert_eq!(h.dtype().expect("dtype"), MlxDtype::U64);
      assert_eq!(mlx_array_to_u64_vec(&h).expect("readback"), vals);
  }
  ```
  (These call `MetalBuffer::from_i64_slice`/`from_u64_slice` from Task 4. Run order note: this test fully greens only after Task 4. For a strictly-green Task 3, run just `dtype_query_reports_f32` here and add the view round-trips at the end of Task 4. The plan keeps them together for readability; the implementer should run `dtype_query_reports_f32` to gate Task 3.)
- [ ] Run the f32 dtype-query test, expect FAIL (wrapper missing):
  `cargo test -p polars-metal-mlx-sys --test test_int_readback dtype_query_reports_f32 -- --test-threads=1`
  Expected: compile error on `a.dtype()` / missing import.
- [ ] Implement in `crates/polars-metal-mlx-sys/src/array.rs` — add `dtype()` to `impl MlxArrayHandle` (next to `dtype_is_f32`):
  ```rust
      /// Return the array's dtype as an [`MlxDtype`].
      ///
      /// # Errors
      /// `FfiError::Runtime` if the array's dtype is one we do not map
      /// (e.g. float64), surfaced from the C++ `mlx_array_dtype` throw.
      pub fn dtype(&self) -> Result<MlxDtype, FfiError> {
          let tag = ffi::mlx_array_dtype(&self.ptr).map_err(FfiError::from)?;
          MlxDtype::from_tag(tag)
      }
  ```
- [ ] In the same file, add a `from_tag` constructor to `impl MlxDtype` (the inverse of `as u32`, used to decode the FFI return):
  ```rust
      /// Decode a `u32` dtype tag (as returned by `mlx_array_dtype`) into an
      /// `MlxDtype`. `FfiError::Runtime` on an unknown tag.
      pub fn from_tag(tag: u32) -> Result<Self, crate::error::FfiError> {
          Ok(match tag {
              0 => MlxDtype::F32,
              1 => MlxDtype::F64,
              2 => MlxDtype::I32,
              3 => MlxDtype::Bool,
              4 => MlxDtype::I8,
              5 => MlxDtype::I16,
              6 => MlxDtype::I64,
              7 => MlxDtype::U8,
              8 => MlxDtype::U16,
              9 => MlxDtype::U32,
              10 => MlxDtype::U64,
              other => {
                  return Err(crate::error::FfiError::Runtime(format!(
                      "unknown MlxDtype tag {other}"
                  )))
              }
          })
      }
  ```
- [ ] In the same file, add the per-width readback wrappers (macro, mirroring `mlx_array_to_i32_vec`). Place after `mlx_array_to_i32_vec`:
  ```rust
  /// Generate a `mlx_array_to_<t>_vec` readback wrapper mirroring
  /// `mlx_array_to_i32_vec`: row-major-contiguous memcpy after eval. Callers
  /// are responsible for ensuring the handle's dtype matches `$t` (the
  /// subgraph builder checks via `MlxArrayHandle::dtype()` before dispatch).
  macro_rules! impl_to_vec {
      ($fn_name:ident, $t:ty, $ffi:path) => {
          #[doc = concat!("Read a materialized array back to a host `Vec<", stringify!($t), ">`. Call after `mlx_array_eval`.")]
          pub fn $fn_name(handle: &MlxArrayHandle) -> Result<Vec<$t>, FfiError> {
              let n: usize = handle.shape().iter().product();
              if n == 0 {
                  return Ok(Vec::new());
              }
              let mut out = vec![0 as $t; n];
              // SAFETY: `out` has exactly `n` slots; the array is eval'd and of
              // the matching dtype (caller contract). `arr->data<T>()` is valid
              // for `n` elements.
              unsafe { $ffi(&handle.ptr, out.as_mut_ptr(), n) };
              Ok(out)
          }
      };
  }
  impl_to_vec!(mlx_array_to_i8_vec, i8, ffi::mlx_array_copy_to_i8);
  impl_to_vec!(mlx_array_to_i16_vec, i16, ffi::mlx_array_copy_to_i16);
  impl_to_vec!(mlx_array_to_i64_vec, i64, ffi::mlx_array_copy_to_i64);
  impl_to_vec!(mlx_array_to_u8_vec, u8, ffi::mlx_array_copy_to_u8);
  impl_to_vec!(mlx_array_to_u16_vec, u16, ffi::mlx_array_copy_to_u16);
  impl_to_vec!(mlx_array_to_u32_vec, u32, ffi::mlx_array_copy_to_u32);
  impl_to_vec!(mlx_array_to_u64_vec, u64, ffi::mlx_array_copy_to_u64);
  ```
- [ ] Run the dtype-query test, expect PASS:
  `cargo test -p polars-metal-mlx-sys --test test_int_readback dtype_query_reports_f32 -- --test-threads=1`
  Expected: green. (The view round-trip tests stay red until Task 4's `from_i64_slice` lands.)
- [ ] Lint:
  `cargo fmt && cargo clippy -p polars-metal-mlx-sys`
  Expected: no warnings.
- [ ] Commit:
  `git add crates/polars-metal-mlx-sys/src/array.rs crates/polars-metal-mlx-sys/tests/test_int_readback.rs && git commit -m "B1 T3: array.rs — MlxArrayHandle::dtype(), MlxDtype::from_tag, per-width to_<t>_vec readback"`

---

## Task 4 — Buffer bridge: per-width `from_<t>_slice` / `to_<t>_vec` / `from_borrowed_<t>`

The Arrow↔MTL bridge is byte-agnostic, so each typed accessor is a thin wrapper over the existing byte path. Add them via a macro (no new deps — `bytemuck` is **not** a dependency of `polars-metal-buffer`, confirmed; hand-roll with the existing `unsafe`+`// SAFETY:` slice-reinterpret pattern). This turns the Task-3 view round-trip tests green.

**Files**
- Modify: `crates/polars-metal-buffer/src/bridge.rs`
- Test (modify): the `#[cfg(test)] mod tests` in `bridge.rs` (proptest round-trip)

**Steps**

- [ ] Add a failing proptest round-trip to the `proptest! { ... }` block in `crates/polars-metal-buffer/src/bridge.rs`:
  ```rust
      #[test]
      fn i64_slice_round_trip(vals in proptest::collection::vec(any::<i64>(), 1..512)) {
          let device = device();
          let metal = MetalBuffer::from_i64_slice(&device, &vals).expect("stage i64");
          prop_assert_eq!(metal.to_i64_vec(), vals);
      }

      #[test]
      fn u64_slice_round_trip(vals in proptest::collection::vec(any::<u64>(), 1..512)) {
          let device = device();
          let metal = MetalBuffer::from_u64_slice(&device, &vals).expect("stage u64");
          prop_assert_eq!(metal.to_u64_vec(), vals);
      }

      #[test]
      fn i8_slice_round_trip(vals in proptest::collection::vec(any::<i8>(), 1..512)) {
          let device = device();
          let metal = MetalBuffer::from_i8_slice(&device, &vals).expect("stage i8");
          prop_assert_eq!(metal.to_i8_vec(), vals);
      }
  ```
- [ ] Run, expect FAIL (methods missing → compile error):
  `cargo test -p polars-metal-buffer i64_slice_round_trip -- --test-threads=1`
  Expected: `no function or associated item named from_i64_slice`.
- [ ] Implement in `crates/polars-metal-buffer/src/bridge.rs` — add a macro that generates `from_<t>_slice`, `to_<t>_vec`, and `unsafe from_borrowed_<t>` for a given Pod scalar, inside `impl MetalBuffer` (place after `from_f32_slice` / `to_f32_vec` / `from_borrowed_f32`):
  ```rust
      // ── Per-width typed accessors (M6 Track B / B1) ──────────────────────
      //
      // The Arrow↔MTL bridge is byte-agnostic; each typed accessor reinterprets
      // a contiguous `&[T]` as `&[u8]` (and back) over the existing byte path.
      // Generated by macro to avoid eight near-identical copies. `bytemuck` is
      // deliberately not a dependency — we hand-roll the reinterpret with an
      // explicit `// SAFETY:` per the crate convention.
  ```
  Then, *outside* the existing `impl MetalBuffer` is awkward for a macro that emits methods, so define the macro and invoke it in a fresh `impl MetalBuffer` block at the end of the non-test region of the file:
  ```rust
  macro_rules! impl_typed_accessors {
      ($t:ty, $from_slice:ident, $to_vec:ident, $from_borrowed:ident) => {
          impl MetalBuffer {
              #[doc = concat!("Construct a `MetalBuffer` by copying a `&[", stringify!($t), "]` into a new Metal allocation. Errors on empty input (Metal rejects zero-byte allocations).")]
              pub fn $from_slice(device: &MetalDevice, data: &[$t]) -> Result<Self, BufferError> {
                  // SAFETY: `$t` is a Pod integer with no padding; reinterpreting
                  // `&[$t]` as `&[u8]` is valid (align of [u8] is 1 ≤ align of
                  // [$t]; byte length is exact). Read-only view bounded by `data`.
                  let bytes = unsafe {
                      std::slice::from_raw_parts(
                          data.as_ptr() as *const u8,
                          std::mem::size_of_val(data),
                      )
                  };
                  device.new_buffer_from_bytes(bytes)
              }

              #[doc = concat!("Copy the buffer's contents out as a `Vec<", stringify!($t), ">` (assumes the buffer holds `", stringify!($t), "` values; length must be a multiple of the element size).")]
              pub fn $to_vec(&self) -> Vec<$t> {
                  let bytes = self.as_slice();
                  let n = bytes.len() / std::mem::size_of::<$t>();
                  let mut out: Vec<$t> = Vec::with_capacity(n);
                  // SAFETY: `bytes` is valid for `n * size_of::<$t>()` bytes; we
                  // copy `n` `$t` values out. `copy_nonoverlapping` tolerates a
                  // misaligned source because it copies bytewise.
                  unsafe {
                      let src = bytes.as_ptr().cast::<$t>();
                      out.set_len(n);
                      std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), n);
                  }
                  out
              }

              #[doc = concat!("Wrap a borrowed `", stringify!($t), "` region (input-zero-copy fused path), mirroring `from_borrowed_f32`. Zero-copy via `newBufferWithBytesNoCopy` when `ptr` is page-aligned, else copies.")]
              ///
              /// # Safety
              /// `ptr` must be non-null and valid for `n_elements * size_of::<T>()`
              /// bytes for the entire lifetime of the returned `MetalBuffer`. On
              /// the zero-copy path the buffer holds no keep-alive — the caller
              /// owns the memory and must outlive the buffer. On the copy path the
              /// data is read once during construction.
              pub unsafe fn $from_borrowed(
                  device: &MetalDevice,
                  ptr: *const $t,
                  n_elements: usize,
              ) -> Result<Self, BufferError> {
                  // SAFETY: caller guarantees `ptr` is valid for
                  // `n_elements * size_of::<$t>()` bytes; reinterpret as bytes.
                  let byte_ptr = ptr as *const u8;
                  // SAFETY (delegated): from_borrowed_f32 is f32-specific only in
                  // its element size; we reuse the byte-level borrow by computing
                  // the byte count here and calling the shared byte path.
                  unsafe {
                      Self::from_borrowed_bytes(
                          device,
                          byte_ptr,
                          n_elements * std::mem::size_of::<$t>(),
                      )
                  }
              }
          }
      };
  }
  impl_typed_accessors!(i8, from_i8_slice, to_i8_vec, from_borrowed_i8);
  impl_typed_accessors!(i16, from_i16_slice, to_i16_vec, from_borrowed_i16);
  impl_typed_accessors!(i32, from_i32_slice, to_i32_vec, from_borrowed_i32);
  impl_typed_accessors!(i64, from_i64_slice, to_i64_vec, from_borrowed_i64);
  impl_typed_accessors!(u8, from_u8_slice, to_u8_vec, from_borrowed_u8);
  impl_typed_accessors!(u16, from_u16_slice, to_u16_vec, from_borrowed_u16);
  impl_typed_accessors!(u32, from_u32_slice, to_u32_vec, from_borrowed_u32);
  impl_typed_accessors!(u64, from_u64_slice, to_u64_vec, from_borrowed_u64);
  ```
- [ ] Add the shared byte-level borrow helper the macro calls, extracted from the existing `from_borrowed_f32` body (so the page-aligned / copy split lives in one place). In `impl MetalBuffer`, add:
  ```rust
      /// Byte-level borrowed-buffer ingest shared by every `from_borrowed_<t>`.
      /// Zero-copy `newBufferWithBytesNoCopy` when `ptr` is page-aligned, else a
      /// single copy. `len` is the total byte length.
      ///
      /// # Safety
      /// `ptr` must be non-null and valid for `len` bytes for the buffer's
      /// lifetime (zero-copy path holds no keep-alive); the caller owns the
      /// memory. On the copy path the borrow ends when this returns.
      pub(crate) unsafe fn from_borrowed_bytes(
          device: &MetalDevice,
          ptr: *const u8,
          len: usize,
      ) -> Result<Self, BufferError> {
          if len == 0 {
              return Err(BufferError::AllocationFailed { bytes: 0 });
          }
          let nn = NonNull::new(ptr as *mut std::ffi::c_void)
              .ok_or(BufferError::AllocationFailed { bytes: len })?;
          if is_ptr_page_aligned(ptr as usize) {
              let deallocator: RcBlock<dyn Fn(NonNull<std::ffi::c_void>, usize)> =
                  RcBlock::new(move |_ptr, _len| {});
              // SAFETY: `ptr` valid for `len` bytes for the buffer's lifetime
              // (caller-guaranteed); page-aligned pointer satisfies bytesNoCopy.
              let inner = device
                  .raw()
                  .newBufferWithBytesNoCopy_length_options_deallocator(
                      nn,
                      len,
                      MTLResourceOptions::MTLResourceStorageModeShared,
                      Some(&deallocator),
                  )
                  .ok_or(BufferError::AllocationFailed { bytes: len })?;
              Ok(Self { inner, _owner: None, _view_parent: None })
          } else {
              // SAFETY: `ptr` valid for `len` bytes; Metal copies them in now.
              let inner = device
                  .raw()
                  .newBufferWithBytes_length_options(
                      nn,
                      len,
                      MTLResourceOptions::MTLResourceStorageModeShared,
                  )
                  .ok_or(BufferError::AllocationFailed { bytes: len })?;
              Ok(Self { inner, _owner: None, _view_parent: None })
          }
      }
  ```
  Then refactor `from_borrowed_f32` to delegate (keeps its existing public signature + doc; the two existing `from_borrowed_f32` tests still exercise the same code path):
  ```rust
      pub unsafe fn from_borrowed_f32(
          device: &MetalDevice,
          ptr: *const f32,
          n_elements: usize,
      ) -> Result<Self, BufferError> {
          // SAFETY: caller guarantees `ptr` valid for `n_elements * 4` bytes for
          // the buffer's lifetime; delegate to the byte-level shared path.
          unsafe {
              Self::from_borrowed_bytes(device, ptr as *const u8, n_elements * std::mem::size_of::<f32>())
          }
      }
  ```
- [ ] Run, expect PASS (all new round-trips + the unchanged `from_borrowed_f32` tests):
  `cargo test -p polars-metal-buffer -- --test-threads=1`
  Expected: green, including `from_borrowed_f32_zero_copy_when_page_aligned` / `from_borrowed_f32_copies_when_unaligned` (regression on the refactor).
- [ ] Run the Task-3 view round-trip tests now that `from_i64_slice`/`from_u64_slice` exist, expect PASS:
  `cargo test -p polars-metal-mlx-sys --test test_int_readback -- --test-threads=1`
  Expected: `i64_view_round_trips` + `u64_view_round_trips_beyond_i64_range` green.
- [ ] Lint:
  `cargo fmt && cargo clippy -p polars-metal-buffer -p polars-metal-mlx-sys`
  Expected: no warnings.
- [ ] Commit:
  `git add crates/polars-metal-buffer/src/bridge.rs crates/polars-metal-mlx-sys/tests/test_int_readback.rs && git commit -m "B1 T4: buffer bridge — per-width from/to/borrowed accessors; shared from_borrowed_bytes; int view round-trips"`

---

## Task 5 — `subgraph.rs`: dtype-aware input wrapping + output readback dispatch

Remove the two F32 hard-codes in the buffer-path subgraph: (1) `from_fusion_scope_buffers` must map each input's `InputDtype` → `MlxDtype` and derive the element count from `dtype.element_size()` (not `/ 4`); (2) `eval_to_metal_buffers` must read each output back by its eval'd dtype (`mlx_array_dtype`) and build the `MetalBuffer` width-aware. `eval_into` (the output-zero-copy path used by `execute_fused_expr`) is handled in Task 6 because its `&mut [f32]` signature is the FFI boundary.

**Files**
- Modify: `crates/polars-metal-core/src/fusion/subgraph.rs`
- Test (create): `crates/polars-metal-core/tests/test_subgraph_int.rs`

**Steps**

- [ ] Write a failing test `crates/polars-metal-core/tests/test_subgraph_int.rs` that builds an I32 scope over an int buffer and evals to a `MetalBuffer`, asserting the I32 round-trip (proves both hard-codes are gone):
  ```rust
  use std::sync::Arc;
  use polars_metal_buffer::{MetalBuffer, MetalDevice};
  use polars_metal_core::fusion::scope::{FusionScope, InputDtype};
  use polars_metal_core::fusion::subgraph::MlxSubgraph;

  #[test]
  fn i32_identity_subgraph_round_trips() {
      let device = MetalDevice::system_default().expect("metal");
      let vals: Vec<i32> = vec![-7, 0, 1, 100, 2_000_000_000];
      let buf = Arc::new(MetalBuffer::from_i32_slice(&device, &vals).expect("stage"));

      let mut scope = FusionScope::new();
      let a = scope.add_input("a", InputDtype::I32);
      scope.mark_output(a);

      let sg = MlxSubgraph::from_fusion_scope_buffers(&scope, &[buf]).expect("build");
      let outs = sg.eval_to_metal_buffers(&device).expect("eval");
      assert_eq!(outs.len(), 1);
      assert_eq!(outs[0].to_i32_vec(), vals);
  }
  ```
  (Confirm the test-visible module path: `polars_metal_core::fusion::scope` / `::subgraph` must be `pub`. If the existing tests reach them via `pub use`, mirror that import. Check `crates/polars-metal-core/tests/test_mlx_subgraph_metal_buffer.rs` for the exact path and copy it.)
- [ ] Run, expect FAIL — current code wraps the buffer as `MlxDtype::F32` (so a 20-byte i32 buffer → 5 f32 elems, garbage values; output read as f32 then `to_i32_vec` mismatches):
  `cargo test -p polars-metal-core --test test_subgraph_int -- --test-threads=1`
  Expected: assertion failure (values wrong / dtype mismatch).
- [ ] Implement in `crates/polars-metal-core/src/fusion/subgraph.rs` — add an `InputDtype`→`MlxDtype` mapper (free fn near the top of the file, after the imports):
  ```rust
  /// Map a fused-scope `InputDtype` to the `MlxDtype` tag used to wrap its
  /// buffer. Returns `BuildError::UnsupportedInputDtype` for the composite /
  /// unsupported dtypes (`ArrayF32`/`ListF32`/`F64`), which the buffer path
  /// does not ingest as a flat 1-D numeric column.
  fn input_dtype_to_mlx(dtype: InputDtype) -> Result<MlxDtype, BuildError> {
      Ok(match dtype {
          InputDtype::F32 => MlxDtype::F32,
          InputDtype::I32 => MlxDtype::I32,
          InputDtype::Bool => MlxDtype::Bool,
          InputDtype::I8 => MlxDtype::I8,
          InputDtype::I16 => MlxDtype::I16,
          InputDtype::I64 => MlxDtype::I64,
          InputDtype::U8 => MlxDtype::U8,
          InputDtype::U16 => MlxDtype::U16,
          InputDtype::U32 => MlxDtype::U32,
          InputDtype::U64 => MlxDtype::U64,
          InputDtype::F64 | InputDtype::ArrayF32(_) | InputDtype::ListF32 => {
              return Err(BuildError::UnsupportedInputDtype(format!("{dtype:?}")))
          }
      })
  }
  ```
  Add the error variant to the `BuildError` enum (find its `#[derive(thiserror::Error)] enum BuildError`):
  ```rust
      #[error("unsupported input dtype for buffer path: {0}")]
      UnsupportedInputDtype(String),
  ```
- [ ] In the same file, fix `from_fusion_scope_buffers` to zip inputs with their declared dtype and use `element_size`:
  ```rust
          let mut handles: Vec<MlxArrayHandle> = inputs
              .iter()
              .zip(scope.inputs.iter())
              .map(|(buf, input_ref)| {
                  let mlx_dtype = input_dtype_to_mlx(input_ref.dtype)?;
                  let n_elements = (buf.len() / mlx_dtype.element_size()) as i64;
                  let shape = [n_elements];
                  mlx_array_view_metal_buffer(buf.clone(), &shape, mlx_dtype)
                      .map_err(|e| BuildError::MlxError(format!("{e:?}")))
              })
              .collect::<Result<Vec<_>, _>>()?;
  ```
- [ ] In the same file, fix `eval_to_metal_buffers` to dispatch readback on the output dtype:
  ```rust
      pub fn eval_to_metal_buffers(
          &self,
          device: &MetalDevice,
      ) -> Result<Vec<MetalBuffer>, BuildError> {
          mlx_array_eval(&self.outputs).map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
          let mut outs = Vec::with_capacity(self.outputs.len());
          for h in &self.outputs {
              let dtype = h.dtype().map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
              let buf = match dtype {
                  MlxDtype::F32 => {
                      let data = mlx_array_to_f32_vec(h).map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
                      MetalBuffer::from_f32_slice(device, &data)
                  }
                  MlxDtype::I32 => {
                      let data = mlx_array_to_i32_vec(h).map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
                      MetalBuffer::from_i32_slice(device, &data)
                  }
                  MlxDtype::I8 => {
                      let data = mlx_array_to_i8_vec(h).map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
                      MetalBuffer::from_i8_slice(device, &data)
                  }
                  MlxDtype::I16 => {
                      let data = mlx_array_to_i16_vec(h).map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
                      MetalBuffer::from_i16_slice(device, &data)
                  }
                  MlxDtype::I64 => {
                      let data = mlx_array_to_i64_vec(h).map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
                      MetalBuffer::from_i64_slice(device, &data)
                  }
                  MlxDtype::U8 => {
                      let data = mlx_array_to_u8_vec(h).map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
                      MetalBuffer::from_u8_slice(device, &data)
                  }
                  MlxDtype::U16 => {
                      let data = mlx_array_to_u16_vec(h).map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
                      MetalBuffer::from_u16_slice(device, &data)
                  }
                  MlxDtype::U32 => {
                      let data = mlx_array_to_u32_vec(h).map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
                      MetalBuffer::from_u32_slice(device, &data)
                  }
                  MlxDtype::U64 => {
                      let data = mlx_array_to_u64_vec(h).map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
                      MetalBuffer::from_u64_slice(device, &data)
                  }
                  MlxDtype::Bool | MlxDtype::F64 => {
                      return Err(BuildError::UnsupportedInputDtype(format!("output {dtype:?}")))
                  }
              }
              .map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
              outs.push(buf);
          }
          Ok(outs)
      }
  ```
  Update the `use polars_metal_mlx_sys::array::{...}` import line to add the new readback fns (`mlx_array_to_i8_vec`, `mlx_array_to_i16_vec`, `mlx_array_to_i32_vec`, `mlx_array_to_i64_vec`, `mlx_array_to_u8_vec`, `mlx_array_to_u16_vec`, `mlx_array_to_u32_vec`, `mlx_array_to_u64_vec`) and `MlxDtype`.
- [ ] Run, expect PASS:
  `cargo test -p polars-metal-core --test test_subgraph_int -- --test-threads=1`
  Expected: green.
- [ ] Run the existing F32 subgraph tests to confirm no regression (the F32 path now flows through `element_size()==4` and the F32 readback arm):
  `cargo test -p polars-metal-core --test test_mlx_subgraph_metal_buffer --test test_mlx_subgraph --test proptest_subgraph -- --test-threads=1`
  Expected: green.
- [ ] Lint:
  `cargo fmt && cargo clippy -p polars-metal-core`
  Expected: no warnings.
- [ ] Commit:
  `git add crates/polars-metal-core/src/fusion/subgraph.rs crates/polars-metal-core/tests/test_subgraph_int.rs && git commit -m "B1 T5: subgraph.rs — dtype-aware input wrapping + output readback dispatch on mlx_array_dtype"`

---

## Task 6 — End-to-end dtype threading: `py.rs`, `execute_fused_expr`, Python analyzer + dispatch

Wire the dtype through the full PyO3 + Python boundary so `pl.col("x") + 1` on an int column round-trips. Five coupled edits, one task because they cross the FFI together and have no independent green state:

1. `py.rs::PyFusionScope::add_input` — accept the 7 new dtype strings.
2. `_fusion_analyzer._dtype_to_input_str` — map all 8 int dtypes; **infer + stamp the output dtype** (`_fused_out_dtype`) on the binding; carry the **literal dtype** so `col + 1` stages `1` as the column's int type (else MLX promotes the add to f32).
3. `execute_fused_expr` (Rust/PyO3) — accept per-input dtype tags and an output dtype tag; stage inputs by width via `from_borrowed_<t>`; validate the eval'd output dtype matches the declared tag; write back width-aware into the caller buffer.
4. `_udf._dispatch_hstack_fused` — stage each column/literal as native bytes; pre-allocate the output by `_fused_out_dtype`; wrap the result `pl.Series` with the right dtype + null mask.

**The output-dtype-return design (the seam the spec left open).** The current FFI keeps **output-zero-copy**: Python pre-allocates the result array and Rust writes into it via `eval_into(&mut [&mut [f32]])`, returning only the element count. To stay zero-copy *and* be dtype-polymorphic, Python must allocate the **right-width** buffer before it can know the eval'd dtype. Rather than make the FFI two-pass (eval, return dtype, re-call to fill), B1 **infers the output dtype statically in the analyzer**: the exit-bar chain `col + 1` is monomorphic — its output dtype equals the single input column's dtype (B2 will generalize, e.g. `sum→Int64` upcast). The analyzer stamps `_fused_out_dtype` (a dtype-tag string) on the binding; `_udf` pre-allocates `np.empty(n_rows, dtype=<that>)`; the FFI receives the declared output tag, and after eval the Rust side **asserts** the eval'd `mlx_array_dtype` equals the declared tag (a hard error on mismatch — catches an analyzer mis-inference before it corrupts bytes), then writes width-aware. This keeps one synchronous call, no re-eval, output-zero-copy intact, and a self-checking contract. For B1 the inference rule is exactly "output dtype = the (single, common) input column dtype"; the analyzer aborts (CPU fallback) if the chain mixes column dtypes (B2 handles mixed-dtype promotion).

**Files**
- Modify: `crates/polars-metal-core/src/fusion/py.rs`
- Modify: `crates/polars-metal-core/src/udf.rs`
- Modify: `python/polars_metal/_fusion_analyzer.py`
- Modify: `python/polars_metal/_udf.py`
- Test (create): `tests/python_integration/test_int_foundation.py` (final exit-bar test lands in Task 7; here add a minimal smoke assertion to drive the impl)

**Steps**

- [ ] Add a failing PyO3-level unit test for `add_input` string parsing in `crates/polars-metal-core/src/fusion/py.rs` (append to its `#[cfg(test)] mod tests`, or add one). It exercises the parser directly:
  ```rust
  #[cfg(test)]
  mod add_input_tests {
      use super::PyFusionScope;

      #[test]
      fn accepts_all_int_dtype_strings() {
          for s in ["I8", "I16", "I32", "I64", "U8", "U16", "U32", "U64"] {
              let mut scope = PyFusionScope::new();
              scope.add_input("c", s).unwrap_or_else(|_| panic!("dtype {s} should parse"));
          }
      }
  }
  ```
- [ ] Run, expect FAIL (only `I32` parses today):
  `cargo test -p polars-metal-core add_input_tests -- --test-threads=1`
  Expected: panic on `I8` ("unknown InputDtype: I8").
- [ ] Implement in `crates/polars-metal-core/src/fusion/py.rs` — extend the `match dtype` arms in `add_input`:
  ```rust
              "F32" => InputDtype::F32,
              "F64" => InputDtype::F64,
              "Bool" => InputDtype::Bool,
              "I32" => InputDtype::I32,
              "I8" => InputDtype::I8,
              "I16" => InputDtype::I16,
              "I64" => InputDtype::I64,
              "U8" => InputDtype::U8,
              "U16" => InputDtype::U16,
              "U32" => InputDtype::U32,
              "U64" => InputDtype::U64,
  ```
  (Leave the `ArrayF32(` / `ListF32` arms and the `other =>` error arm unchanged.)
- [ ] Run, expect PASS:
  `cargo test -p polars-metal-core add_input_tests -- --test-threads=1`
  Expected: green.
- [ ] Change the `execute_fused_expr` signature in `crates/polars-metal-core/src/udf.rs` to thread dtype tags. New signature + body (replaces the current f32-only version):
  ```rust
  #[pyfunction]
  pub fn execute_fused_expr(
      scope: &crate::fusion::py::PyFusionScope,
      // Each input: (ptr, n_elements, dtype_tag) — dtype_tag is the MlxDtype u32.
      inputs: Vec<(usize, usize, u32)>,
      // Output: (ptr, capacity_elements, dtype_tag) — caller pre-allocated to the
      // analyzer-inferred output dtype; we validate the eval'd dtype matches.
      out: (usize, usize, u32),
  ) -> PyResult<usize> {
      use std::sync::Arc;
      use polars_metal_mlx_sys::array::MlxDtype;

      let device = MetalDevice::system_default().map_err(|e| {
          pyo3::exceptions::PyRuntimeError::new_err(format!(
              "polars_metal: metal device unavailable: {e}"
          ))
      })?;

      // Stage each input buffer by its declared width via the matching
      // from_borrowed_<t>. Caller keeps the source arrays alive for the call.
      let metal_buffers: Vec<Arc<polars_metal_buffer::MetalBuffer>> = inputs
          .iter()
          .map(|&(ptr, n, tag)| {
              let dtype = MlxDtype::from_tag(tag).map_err(|e| {
                  pyo3::exceptions::PyValueError::new_err(format!(
                      "polars_metal: bad input dtype tag: {e:?}"
                  ))
              })?;
              // SAFETY: `ptr` addresses `n` live, contiguous elements of `dtype`
              // (caller contract) for the whole call; from_borrowed_bytes picks
              // zero-copy vs copy by alignment.
              let buf = unsafe {
                  polars_metal_buffer::MetalBuffer::from_borrowed_bytes(
                      &device,
                      ptr as *const u8,
                      n * dtype.element_size(),
                  )
              }
              .map_err(|e| {
                  pyo3::exceptions::PyRuntimeError::new_err(format!(
                      "polars_metal: input buffer staging failed: {e}"
                  ))
              })?;
              Ok(Arc::new(buf))
          })
          .collect::<PyResult<Vec<_>>>()?;

      let subgraph = crate::fusion::subgraph::MlxSubgraph::from_fusion_scope_buffers(
          &scope.inner,
          &metal_buffers,
      )
      .map_err(|e| {
          pyo3::exceptions::PyValueError::new_err(format!("polars_metal: subgraph build: {e}"))
      })?;

      let (out_ptr, out_cap, out_tag) = out;
      let out_dtype = MlxDtype::from_tag(out_tag).map_err(|e| {
          pyo3::exceptions::PyValueError::new_err(format!(
              "polars_metal: bad output dtype tag: {e:?}"
          ))
      })?;

      subgraph.eval_into_typed(&device, out_ptr, out_cap, out_dtype).map_err(|e| {
          pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: subgraph eval: {e}"))
      })
  }
  ```
  This delegates to a new `eval_into_typed` on `MlxSubgraph` (added next) so the unsafe width-aware write lives in one place. `from_borrowed_bytes` was made `pub(crate)` in Task 4 — promote it to `pub` if `udf.rs` is in a different crate than the buffer crate (it is: `polars-metal-core` calls `polars_metal_buffer::MetalBuffer::from_borrowed_bytes`). Change Task 4's `pub(crate) unsafe fn from_borrowed_bytes` to `pub unsafe fn from_borrowed_bytes` and add a `# Safety` doc (it's a cross-crate public unsafe API). If you prefer to keep it crate-private, instead route through the typed `from_borrowed_<t>` by matching on `dtype` here — but that reintroduces an 8-arm match; the `pub from_borrowed_bytes` is cleaner. **Decision: make `from_borrowed_bytes` `pub`.**
- [ ] Add `eval_into_typed` to `crates/polars-metal-core/src/fusion/subgraph.rs` (single output, validates dtype, writes width-aware into the caller's raw buffer):
  ```rust
      /// Eval the (single-output) subgraph and write the output directly into
      /// the caller's raw buffer at `out_ptr` (output-zero-copy), interpreting
      /// it as `out_dtype`. Returns the element count written. Errors if the
      /// eval'd output dtype != `out_dtype` (analyzer mis-inference guard) or if
      /// the output does not fit in `out_cap` elements.
      ///
      /// # Safety contract (caller): `out_ptr` addresses `out_cap` writable,
      /// contiguous elements of `out_dtype`, kept alive for the whole call.
      pub fn eval_into_typed(
          &self,
          _device: &MetalDevice,
          out_ptr: usize,
          out_cap: usize,
          out_dtype: MlxDtype,
      ) -> Result<usize, BuildError> {
          if self.outputs.len() != 1 {
              return Err(BuildError::InputCountMismatch {
                  expected: 1,
                  actual: self.outputs.len(),
              });
          }
          mlx_array_eval(&self.outputs).map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
          let h = &self.outputs[0];
          let got = h.dtype().map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
          if got != out_dtype {
              return Err(BuildError::MlxError(format!(
                  "output dtype mismatch: declared {out_dtype:?}, eval'd {got:?}"
              )));
          }
          let n: usize = h.shape().iter().product();
          if n > out_cap {
              return Err(BuildError::MlxError(format!(
                  "output too large: {n} > capacity {out_cap}"
              )));
          }
          if n == 0 {
              return Ok(0);
          }
          // Width-aware copy into the caller buffer. SAFETY: out_ptr addresses
          // out_cap >= n elements of out_dtype (caller contract); we copy n.
          macro_rules! write_back {
              ($to_vec:path, $t:ty) => {{
                  let data = $to_vec(h).map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
                  let dst = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut $t, n) };
                  dst.copy_from_slice(&data);
              }};
          }
          match out_dtype {
              MlxDtype::F32 => write_back!(mlx_array_to_f32_vec, f32),
              MlxDtype::I32 => write_back!(mlx_array_to_i32_vec, i32),
              MlxDtype::I8 => write_back!(mlx_array_to_i8_vec, i8),
              MlxDtype::I16 => write_back!(mlx_array_to_i16_vec, i16),
              MlxDtype::I64 => write_back!(mlx_array_to_i64_vec, i64),
              MlxDtype::U8 => write_back!(mlx_array_to_u8_vec, u8),
              MlxDtype::U16 => write_back!(mlx_array_to_u16_vec, u16),
              MlxDtype::U32 => write_back!(mlx_array_to_u32_vec, u32),
              MlxDtype::U64 => write_back!(mlx_array_to_u64_vec, u64),
              MlxDtype::Bool | MlxDtype::F64 => {
                  return Err(BuildError::UnsupportedInputDtype(format!("output {out_dtype:?}")))
              }
          }
          Ok(n)
      }
  ```
  Keep the existing `eval_into(&mut [&mut [f32]])` for any other caller (it stays f32-only; the FFI no longer uses it). If nothing else calls it, you may delete it — grep `eval_into(` across the crate first; if only `execute_fused_expr` used it, remove it to avoid dead code (clippy will flag it otherwise).
- [ ] Rebuild the wheel so Python sees the new signature:
  `make wheel`
  Expected: maturin builds the `.so`.
- [ ] In `python/polars_metal/_fusion_analyzer.py`, extend `_dtype_to_input_str` to map all 8 int dtypes:
  ```python
  def _dtype_to_input_str(dtype: Any) -> str:
      """Map a Polars dtype to the input-dtype string the PyFusionScope accepts."""
      if dtype == pl.Float32:
          return "F32"
      if dtype == pl.Float64:
          return "F64"
      if dtype == pl.Boolean:
          return "Bool"
      if dtype == pl.Int8:
          return "I8"
      if dtype == pl.Int16:
          return "I16"
      if dtype == pl.Int32:
          return "I32"
      if dtype == pl.Int64:
          return "I64"
      if dtype == pl.UInt8:
          return "U8"
      if dtype == pl.UInt16:
          return "U16"
      if dtype == pl.UInt32:
          return "U32"
      if dtype == pl.UInt64:
          return "U64"
      s = str(dtype)
      if s.startswith("Array(Float32"):
          inner_d = _extract_array_dim(s)
          return f"ArrayF32({inner_d})"
      if s == "List(Float32)":
          return "ListF32"
      raise _Aborted
  ```
- [ ] Add a dtype-tag helper + literal-dtype carry in `_fusion_analyzer.py`. First a tag map (place near `_dtype_to_input_str`):
  ```python
  # MlxDtype u32 tags — must match MlxDtype in mlx-sys/array.rs and the C++
  # switch. Used to thread the output dtype + per-input dtype through the FFI.
  _MLX_TAG: dict[str, int] = {
      "F32": 0, "F64": 1, "I32": 2, "Bool": 3,
      "I8": 4, "I16": 5, "I64": 6, "U8": 7, "U16": 8, "U32": 9, "U64": 10,
  }


  def _input_str_to_tag(s: str) -> int:
      return _MLX_TAG[s]
  ```
- [ ] In `_fusion_analyzer.py`, make the literal leaf carry the chain's column dtype so `col + 1` stays integer. In `_gather_leaves_ir`, the `Literal` branch currently always stages `"F32"`. For B1 the simplest correct rule that satisfies the exit bar: stage a literal with the **dtype of the sibling column(s)** in the binding. Implement by having `analyze_ir_with_columns` resolve a single "binding dtype" from the column leaves after pass 1, then re-tag literal inputs and stamp the output dtype. Concretely, after `_gather_leaves_ir` + `_visit_ir_ops` in `analyze_ir_with_columns`:
  ```python
      try:
          scope = PyFusionScope()
          descriptors: list[tuple[str, str | float]] = []
          leaf_idx: dict[int, int] = {}
          col_dedup: dict[str, int] = {}
          lit_dedup: dict[float, int] = {}
          _gather_leaves_ir(nt, node_id, schema, scope, descriptors, leaf_idx, col_dedup, lit_dedup)
          # Resolve the binding's common column dtype (B1: monomorphic chains
          # only — every column leaf must share one dtype, else fall back so B2
          # can handle promotion). Output dtype == that column dtype for the
          # exit-bar `col + 1` shape.
          col_dtypes = {
              _dtype_to_input_str(schema[name])
              for kind, name in descriptors
              if kind == "col"
          }
          if len(col_dtypes) != 1:
              # No columns, or mixed dtypes → not a B1 monomorphic chain.
              raise _Aborted
          out_dtype_str = next(iter(col_dtypes))
          idx = _visit_ir_ops(nt, node_id, schema, scope, leaf_idx)
          scope.mark_output(idx)
          return scope, descriptors, out_dtype_str
      except _Aborted:
          return None
  ```
  Change `analyze_ir_with_columns`'s return type/signature to `tuple[PyFusionScope, list[...], str] | None` (the third element is the output dtype string). Update its docstring. **Important:** the literal leaf staging itself happens in `_udf.py` (next step) — the analyzer only needs to (a) confirm the chain is monomorphic and (b) report `out_dtype_str`; the literal *value* is carried in `descriptors` as today, and `_udf` casts it to the binding dtype using `out_dtype_str`. So the literal's scope `add_input` string in `_gather_leaves_ir` can stay `"F32"` *only if* MLX would re-promote — but it would promote the whole add to f32. **Fix the literal input dtype too:** thread `out_dtype_str` into `_gather_leaves_ir` is awkward (it runs before dtype resolution). Cleaner: in `_gather_leaves_ir`, when adding a `Literal`, look at the schema-derived dtype of the binding by passing the resolved column dtype in. Since pass 1 visits columns and literals in DFS order, resolve the column dtype in a pre-scan. **Decision (documented):** do a tiny pre-scan for the binding's column dtype *before* `_gather_leaves_ir`, and pass `lit_dtype_str` into it so the `Literal` branch stages `scope.add_input(f"__lit_{val_f}", lit_dtype_str)`:
  ```python
      # Pre-scan column leaves for the binding dtype so literal inputs stage at
      # the same width (else MLX promotes `int + f32_literal` to f32).
      lit_dtype_str = _scan_binding_col_dtype(nt, node_id, schema)  # "I64" etc. or None
      if lit_dtype_str is None:
          raise _Aborted
  ```
  with helper:
  ```python
  def _scan_binding_col_dtype(nt: Any, node_id: int, schema: dict[str, Any]) -> str | None:
      """Return the single InputDtype string shared by all Column leaves under
      `node_id`, or None if there are zero columns or mixed dtypes (B1 handles
      only monomorphic chains; B2 handles promotion)."""
      found: set[str] = set()

      def walk(nid: int) -> None:
          try:
              node = nt.view_expression(nid)
          except Exception as e:
              raise _Aborted from e
          cls = type(node).__name__
          if cls == "Column":
              name = str(getattr(node, "name", ""))
              dt = schema.get(name)
              if dt is None:
                  raise _Aborted
              found.add(_dtype_to_input_str(dt))
              return
          if cls == "Literal":
              return
          if cls == "BinaryExpr":
              walk(node.left)
              walk(node.right)
              return
          if cls == "Cast":
              walk(node.expr)
              return
          if cls == "Function":
              for cid in list(getattr(node, "input", [])):
                  walk(cid)
              return
          if cls == "Ternary":
              walk(node.predicate)
              walk(node.truthy)
              walk(node.falsy)
              return
          raise _Aborted

      try:
          walk(node_id)
      except _Aborted:
          return None
      if len(found) != 1:
          return None
      return next(iter(found))
  ```
  and modify `_gather_leaves_ir`'s signature to take `lit_dtype_str: str` and use it in the `Literal` branch: `idx = scope.add_input(f"__lit_{val_f}", lit_dtype_str)`. Update the single call site in `analyze_ir_with_columns` to pass it. (The other callers — `analyze_ir_reduction`, the validity path — are B2/out-of-scope; give them the existing `"F32"` default by passing `lit_dtype_str="F32"` so their behavior is unchanged.) **Note:** this means `_gather_leaves_ir` grows one parameter; update all call sites to pass `"F32"` except the `analyze_ir_with_columns` one.
- [ ] Update the **callers** of `analyze_ir_with_columns` in the walker/wire-plan builder to carry `out_dtype_str` onto the binding as `_fused_out_dtype`. Grep for `analyze_ir_with_columns(` to find them (the HStack walker that stamps `_fused_scope` / `_fused_columns`). At each call site, capture the new third return element and add to the binding dict: `binding["_fused_out_dtype"] = out_dtype_str`.
- [ ] In `python/polars_metal/_udf.py::_dispatch_hstack_fused`, stage inputs natively and allocate the output by `_fused_out_dtype`. Add a dtype-tag/numpy map at module top:
  ```python
  def _np_dtype_and_tag(dtype_str: str):
      """InputDtype string -> (numpy dtype, MlxDtype u32 tag). Mirrors _MLX_TAG
      in _fusion_analyzer and MlxDtype in mlx-sys/array.rs."""
      import numpy as np
      table = {
          "F32": (np.float32, 0), "I32": (np.int32, 2),
          "I8": (np.int8, 4), "I16": (np.int16, 5), "I64": (np.int64, 6),
          "U8": (np.uint8, 7), "U16": (np.uint16, 8), "U32": (np.uint32, 9),
          "U64": (np.uint64, 10),
      }
      return table[dtype_str]
  ```
  Then rewrite the input-staging + output-alloc + FFI-call portion of `_dispatch_hstack_fused`:
  ```python
      out_dtype_str = binding.get("_fused_out_dtype", "F32")
      out_np_dtype, out_tag = _np_dtype_and_tag(out_dtype_str)

      input_arrays: list[np.ndarray] = []
      input_meta: list[int] = []  # parallel dtype tags
      for kind, payload in descriptors:
          if kind == "col":
              series = upstream.get_column(payload)
              col_str = _series_input_dtype_str(series)  # "I64" etc.
              col_np, col_tag = _np_dtype_and_tag(col_str)
              # Native-dtype contiguous bytes (no force-cast to f32).
              input_arrays.append(np.ascontiguousarray(series.to_numpy(), dtype=col_np))
              input_meta.append(col_tag)
          elif kind == "lit":
              # Stage the literal at the binding's output dtype so MLX does not
              # promote `int_col + f32_lit` to f32. For B1 the binding is
              # monomorphic, so out_dtype_str is the common column dtype.
              input_arrays.append(np.asarray([payload], dtype=out_np_dtype))
              input_meta.append(out_tag)
          else:
              raise RuntimeError(f"polars_metal: unknown input descriptor {kind!r}")

      out_arr = np.empty(n_rows, dtype=out_np_dtype)
      inputs = [
          (int(a.__array_interface__["data"][0]), int(a.size), tag)
          for a, tag in zip(input_arrays, input_meta)
      ]
      written = _native.execute_fused_expr(
          scope=scope,
          inputs=inputs,
          out=(int(out_arr.__array_interface__["data"][0]), int(out_arr.size), out_tag),
      )
  ```
  Add the small helper:
  ```python
  def _series_input_dtype_str(series: "pl.Series") -> str:
      dt = series.dtype
      table = {
          pl.Float32: "F32", pl.Int8: "I8", pl.Int16: "I16", pl.Int32: "I32",
          pl.Int64: "I64", pl.UInt8: "U8", pl.UInt16: "U16", pl.UInt32: "U32",
          pl.UInt64: "U64",
      }
      s = table.get(dt)
      if s is None:
          raise RuntimeError(f"polars_metal: unsupported fused input dtype {dt}")
      return s
  ```
  The `written == 1 and n_rows != 1` literal-broadcast block and the null-mask block stay as-is (they operate on `out_arr` regardless of dtype; `pl.Series(name, out_arr)` / `pa.array(out_arr, mask=null_mask)` carry the numpy dtype through to Polars).
- [ ] Rebuild the wheel:
  `make wheel`
  Expected: builds.
- [ ] Add a minimal smoke test to `tests/python_integration/test_int_foundation.py` to drive the impl green (the full exit-bar matrix is Task 7):
  ```python
  import polars as pl
  from polars_metal import MetalEngine


  def test_int64_add_one_roundtrips():
      df = pl.DataFrame({"x": pl.Series([1, 2, 3, 1_000_000_000_000], dtype=pl.Int64)})
      got = df.lazy().with_columns((pl.col("x") + 1).alias("y")).collect(engine=MetalEngine())
      want = df.lazy().with_columns((pl.col("x") + 1).alias("y")).collect()
      assert got.equals(want)
      assert got["y"].dtype == pl.Int64
  ```
  (Confirm the engine entry-point name from `python/polars_metal/__init__.py` — it may be `MetalEngine` or a factory; copy the import other `tests/python_integration/test_*` files use.)
- [ ] Run, expect PASS:
  `pytest tests/python_integration/test_int_foundation.py -v`
  Expected: green. If it fails on the literal still promoting to f32, confirm the `lit` descriptor staged at `out_tag` (not f32) and that `_gather_leaves_ir` added the literal input as `lit_dtype_str`.
- [ ] Lint:
  `cargo fmt && cargo clippy -p polars-metal-core && ruff check python/polars_metal/_fusion_analyzer.py python/polars_metal/_udf.py`
  Expected: no warnings.
- [ ] Commit:
  `git add crates/polars-metal-core/src/fusion/py.rs crates/polars-metal-core/src/fusion/subgraph.rs crates/polars-metal-core/src/udf.rs python/polars_metal/_fusion_analyzer.py python/polars_metal/_udf.py tests/python_integration/test_int_foundation.py && git commit -m "B1 T6: thread dtype end-to-end — py.rs strings, execute_fused_expr typed I/O, analyzer out-dtype inference, native-dtype dispatch"`

---

## Task 7 — Exit-bar differential test: Int8/Int32/Int64/UInt64 `col + 1` with nulls

Lock the B1 exit bar with a parametrized differential test: `pl.col("x") + 1` over Int8, Int32, Int64, UInt64 columns (narrow + signed + wide + unsigned), each with and without nulls, byte-exact vs Polars CPU, output dtype preserved.

**Files**
- Modify: `tests/python_integration/test_int_foundation.py`

**Steps**

- [ ] Replace/extend `tests/python_integration/test_int_foundation.py` with the full matrix:
  ```python
  import polars as pl
  import pytest

  from polars_metal import MetalEngine


  _CASES = [
      (pl.Int8, [1, 2, 3, 100, -5, 126]),          # +1 stays in range (127 max)
      (pl.Int32, [-7, 0, 1, 100, 2_000_000_000]),
      (pl.Int64, [1, 2, 3, 3_000_000_000, -2_000_000_000]),
      (pl.UInt64, [0, 1, 5, 10_000_000_000_000_000_000]),  # beyond i64 range
  ]


  def _run(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
      lf = df.lazy().with_columns((pl.col("x") + 1).alias("y"))
      return lf.collect(engine=MetalEngine()), lf.collect()


  @pytest.mark.parametrize("dtype,values", _CASES)
  def test_add_one_roundtrip_no_nulls(dtype, values):
      df = pl.DataFrame({"x": pl.Series(values, dtype=dtype)})
      got, want = _run(df)
      assert got.equals(want), f"{dtype}: {got} != {want}"
      assert got["y"].dtype == dtype


  @pytest.mark.parametrize("dtype,values", _CASES)
  def test_add_one_roundtrip_with_nulls(dtype, values):
      values_with_null = values[:1] + [None] + values[1:]
      df = pl.DataFrame({"x": pl.Series(values_with_null, dtype=dtype)})
      got, want = _run(df)
      assert got.equals(want), f"{dtype} (nulls): {got} != {want}"
      assert got["y"].dtype == dtype
      # Null position preserved exactly.
      assert got["y"].is_null().to_list() == want["y"].is_null().to_list()
  ```
  (If `Int8 + 1` where a value is `127` would overflow, Polars wraps mod 256 on CPU — the chosen values avoid the boundary so the test asserts plain equality. Overflow-domain testing is B2; do not add it here.)
- [ ] Run, expect PASS:
  `pytest tests/python_integration/test_int_foundation.py -v`
  Expected: all 8 parametrizations green (4 dtypes × {no-null, null}).
- [ ] Run the broader Python integration suite to confirm no F32 regression (the f32 `col + 1` / haversine / black-scholes paths must still pass through the now-dtype-aware machinery):
  `pytest tests/python_integration/test_execute_fused_expr.py -v`
  Expected: green.
- [ ] Lint:
  `ruff check tests/python_integration/test_int_foundation.py`
  Expected: clean.
- [ ] Run the full gate to confirm nothing regressed across crates + conformance:
  `make gate`
  Expected: green (or only the documented pre-existing baseline divergences — see MEMORY `M3 conformance deferrals` / `M6 conformance fixes`; no NEW failures).
- [ ] Commit:
  `git add tests/python_integration/test_int_foundation.py && git commit -m "B1 T7: exit-bar differential test — Int8/Int32/Int64/UInt64 col+1 with nulls, byte-exact vs CPU"`

---

## Self-review: B1 scope coverage

Mapping each spec-defined B1 deliverable to a task:

| Spec B1 bullet (lines 97–117) | Covered by |
|---|---|
| `InputDtype` + `MlxDtype` variants for all int widths, each with `element_size` | Task 1 |
| Buffer crate `from_<t>_slice` / `to_<t>_vec` / `from_borrowed_<t>` per width | Task 4 (macro-generated; shared `from_borrowed_bytes`) |
| MLX FFI `mlx_array_to_<t>_vec` readback per width | Tasks 2 (C++) + 3 (Rust wrapper) |
| `mlx_array_view_metal_buffer` already takes `MlxDtype` → view path ready | Confirmed; switch extended to all 8 int tags (Task 2 `mlx_dtype_from_tag`) |
| `subgraph.rs` :170/:184/:227 — map `InputDtype→MlxDtype`, `element_size` count, readback dispatch on output dtype | Task 5 (`from_fusion_scope_buffers`, `eval_to_metal_buffers`) + Task 6 (`eval_into_typed`, the :227 zero-copy FFI path) |
| Python `_udf.py` :170/:181 — native-dtype input, output by inferred dtype | Task 6 |
| Analyzer `_fusion_analyzer.py` — map every Polars int dtype to `InputDtype` | Task 6 (`_dtype_to_input_str`) |
| **Exit bar:** Int32 + Int64 `col + 1` byte-exact, nulls preserved | Task 7 (extended to Int8 + UInt64 per the prompt's narrow+unsigned requirement) |

Cross-cutting requirements honored:
- **cxx four-side consistency** — every new FFI fn (`mlx_array_dtype`, 7 readbacks) lands as Rust extern decl (`lib.rs`) + `.h` decl + `.cc` impl + safe Rust wrapper (`array.rs`) **in the same task** (Tasks 2+3).
- **No `unwrap`/`expect`/`panic` in non-test Rust** — all new paths return `Result` with `BuildError`/`FfiError`/PyO3 exceptions; `MlxDtype::from_tag` and `input_dtype_to_mlx` return `Result`; `add_input` returns `PyResult`.
- **`// SAFETY:` on every new `unsafe`** — `from_borrowed_bytes`, the typed-accessor macro bodies, `eval_into_typed`'s width-aware write, and the `execute_fused_expr` staging all carry SAFETY comments.
- **Never patch `vendor/mlx`** — all MLX integer dtypes used (`int8`…`uint64`) are existing MLX 0.25.1 dtypes (spike-verified per the spec); B1 only calls them, never modifies the vendored tree.
- **`--test-threads=1`** on every `cargo test` (Metal command-queue contention).
- **`make wheel`** after the Rust signature change (Task 6) so Python sees the new `execute_fused_expr` arity.
- **Per-task lint** (`cargo fmt` + `clippy` + `ruff`) to prevent the drift that otherwise only surfaces at the final `make gate`.

Explicitly **not** touched (B2+ territory, would be scope creep): `mlx_op_cast` dtype switch (cast is a B2 op), `analyze_ir_reduction`'s F32-only gate (reduction terminator), reduction upcast `sum(Int32)→Int64`, mixed int↔f32 promotion, overflow-domain tests, the `dt` kernel, reduction routing thresholds. The analyzer deliberately **aborts to CPU fallback** on mixed-dtype chains (`_scan_binding_col_dtype` returns `None`), so B1 never produces a wrong answer on a shape B2 owns — it just declines it.
