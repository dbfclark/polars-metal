# M5 Rolling Custom-Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accelerate native `pl.col(x).rolling_{sum,mean,var,std}(window)` under `engine="metal"` via a numerically-stable custom Metal kernel, matching Polars CPU within a tight tolerance, with CPU fallback for everything unsupported.

**Architecture:** A custom `shaders/rolling.metal` family computes windowed statistics from a per-threadgroup input *tile* (magnitudes stay window/tile-scale → no F32 cancellation). A Rust dispatcher in `polars-metal-kernels` drives it; a PyO3 `execute_rolling` binding exposes it. Pre-optimization, the `collect` wrapper detects handleable `rolling_*` bindings (`lf.serialize` parse), splits them out, collects the rest via the existing in-memory metal path, runs the kernel over the fully-materialized source column, applies the first-`w-1` structural null mask, and stitches the result back. No MLX graph, no `int_range`, no streaming.

**Tech Stack:** MSL (Metal Shading Language), Rust 2021 (`polars-metal-kernels` dispatcher, `polars-metal-core` PyO3 + buffer bridge), PyO3, Python (`polars_metal` detection + collect wrapper), `pytest`.

**Spec:** [`docs/superpowers/specs/2026-06-02-m5-rolling-kernel-design.md`](../specs/2026-06-02-m5-rolling-kernel-design.md). Binding.

**Reading before starting:**
- `references/cudf/cpp/src/rolling/` — the cuDF rolling algorithm (port, don't re-derive).
- `docs/kernel-authoring.md` — MSL + dispatcher conventions.
- `shaders/filter_scatter.metal` — an existing kernel with `device const T*` in / `device T*` out / `constant uint& n [[buffer(k)]]` scalar params (the param-passing convention).
- `crates/polars-metal-kernels/src/filter.rs` — dispatcher shape: `FilterError` enum (`#[from] ShaderError/DispatchError/BufferError`), `shared_library(device)?.pipeline("name")?`, buffer creation, `CommandQueue::dispatch_1d_with_tg`.
- `crates/polars-metal-kernels/src/command.rs:171` — `dispatch_1d_with_tg(pso, buffers: &[&MetalBuffer], n_threads, threadgroup_width)`.
- `crates/polars-metal-kernels/src/shader_lib.rs` — `shared_library(device)`, `ShaderLibrary::pipeline(name)`.
- `crates/polars-metal-core/src/udf.rs::execute_fused_expr` — the pointer-tuple PyO3 convention (`inputs: Vec<(usize,usize)>`, `out: (usize,usize)`, `MetalBuffer::from_borrowed_f32`) to mirror.
- `crates/polars-metal-core/src/lib.rs` `#[pymodule] _native` — where to register `execute_rolling`.
- `python/polars_metal/_udf.py::_fused_null_mask` — how a Python-side null mask is built and applied.
- `python/polars_metal/__init__.py` `collect_wrapper` (~line 157) — the pre-opt hook site; `_opt_flags_without_cse`.

**Conventions:** No `unwrap()` outside tests; no `unsafe` outside `*-sys`/buffer with `// SAFETY:`; errors → `PolarsError::ComputeError` / `PyRuntimeError` at boundaries; null semantics match Polars exactly; one MSL family per file with threadgroup/grid assumptions documented at top; no `shaders/` file without a kernels-crate test; `make lint` before declaring a task done; `make test-conformance` after the collect-wrapper/detection change (baseline = `lazyframe` + `operations_group_by` only).

---

## File structure

- **Create** `shaders/rolling.metal` — `rolling_sum_f32` + `rolling_var_f32` entry points; tile-blocked, F32-stable; `constant uint&` scalar params; static `threadgroup` tile.
- **Create** `crates/polars-metal-kernels/src/rolling.rs` — `RollingError` + `dispatch_rolling_sum_f32` / `dispatch_rolling_var_f32`; register `mod rolling;` in `lib.rs`.
- **Create** `crates/polars-metal-kernels/tests/test_rolling.rs` — kernel correctness vs F64 reference.
- **Modify** `crates/polars-metal-core/src/udf.rs` — add `execute_rolling` pyfunction.
- **Modify** `crates/polars-metal-core/src/lib.rs` — register `udf::execute_rolling` in the `_native` pymodule.
- **Create** `python/polars_metal/_rolling_detect.py` — `RollingBinding` dataclass + `find_rolling_bindings(lf)`.
- **Create** `python/polars_metal/_rolling_dispatch.py` — `apply_rolling(lf, bindings, **collect_kwargs) -> pl.DataFrame` (split / collect-rest / kernel / mask / stitch).
- **Modify** `python/polars_metal/__init__.py` — wire detection + streaming guard into `collect_wrapper`.
- **Create** `tests/python_integration/test_rolling_*.py` — e2e, fallback, property.
- **Modify** `tests/bench/m4_survey/bench_rolling_mlx.py` + `tests/bench/baseline.json` — bench + gate.

**Kernel constants (used across tasks):** `TG_SIZE = 256` (outputs per threadgroup; ≤1024 thread cap), `MAX_W = 4096` (max window; tile is `TG_SIZE + MAX_W` floats = `(256+4096)*4 ≈ 17 KB` < 32 KB threadgroup limit). Op codes: `is_mean: u32` (0=sum, 1=mean) for the sum kernel; `is_std: u32` (0=var, 1=std) + `ddof: u32` for the var kernel.

---

## Phase 1 — `rolling_sum_f32` kernel (sum + mean)

### Task 1: Tile-blocked windowed sum kernel + dispatcher + kernel test

**Files:**
- Create: `shaders/rolling.metal`
- Create: `crates/polars-metal-kernels/src/rolling.rs`; Modify `crates/polars-metal-kernels/src/lib.rs`
- Create: `crates/polars-metal-kernels/tests/test_rolling.rs`

This task lands a **correct, stable** windowed sum: each output sums its `w` inputs from a threadgroup-cached tile (O(w)/output; magnitudes ~`w·mean`). The O(N) prefix-scan optimization is Task 9 (the correctness test here guards that conversion).

- [ ] **Step 1: Write the failing dispatcher test**

```rust
// crates/polars-metal-kernels/tests/test_rolling.rs
#![allow(clippy::unwrap_used)]
use polars_metal_buffer::{MetalDevice, MetalBuffer};
use polars_metal_kernels::rolling::dispatch_rolling_sum_f32;

/// Exact f64 reference rolling sum (no windows < w handled: caller compares from w-1).
fn ref_rolling_sum(x: &[f32], w: usize) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            if i + 1 < w { f64::NAN }
            else { (i + 1 - w..=i).map(|j| x[j] as f64).sum() }
        })
        .collect()
}

fn run_sum(x: &[f32], w: usize, is_mean: bool) -> Vec<f32> {
    let device = MetalDevice::system_default().unwrap();
    let n = x.len();
    let mut out = vec![0.0f32; n];
    // SAFETY: x and out are live, contiguous, n f32 for the call.
    let inb = unsafe { MetalBuffer::from_borrowed_f32(&device, x.as_ptr(), n) }.unwrap();
    let outb = unsafe { MetalBuffer::from_borrowed_f32(&device, out.as_mut_ptr(), n) }.unwrap();
    dispatch_rolling_sum_f32(&device, &inb, &outb, n as u32, w as u32, is_mean).unwrap();
    out
}

#[test]
fn rolling_sum_multi_tile_and_boundary() {
    // n spans multiple TG_SIZE=256 tiles; window straddles tile boundaries.
    let n = 1000usize;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 100.0).collect();
    for w in [1usize, 2, 7, 256, 257, 300] {
        let got = run_sum(&x, w, false);
        let want = ref_rolling_sum(&x, w);
        for i in (w - 1)..n {
            assert!((got[i] as f64 - want[i]).abs() < 1e-3, "w={w} i={i} got={} want={}", got[i], want[i]);
        }
    }
}

#[test]
fn rolling_mean_divides_by_w() {
    let x: Vec<f32> = (1..=10).map(|i| i as f32).collect();
    let got = run_sum(&x, 3, true);
    // window [1,2,3]=6/3=2 at i=2; [8,9,10]=27/3=9 at i=9
    assert!((got[2] - 2.0).abs() < 1e-5);
    assert!((got[9] - 9.0).abs() < 1e-5);
}
```

- [ ] **Step 2: Verify it fails** — `cargo test -p polars-metal-kernels --test test_rolling 2>&1 | head`. Expected: unresolved import `rolling` / `dispatch_rolling_sum_f32`.

- [ ] **Step 3: Write the kernel** `shaders/rolling.metal`

```metal
#include <metal_stdlib>
using namespace metal;

// Rolling windowed statistics over a 1-D F32 column, tile-blocked for F32
// numerical stability (per-threadgroup accumulation magnitudes stay
// ~window/tile-scale, never ~N — avoids the catastrophic cancellation of a
// global cumsum-diff). One thread per output; each threadgroup owns TG_SIZE
// consecutive outputs and loads its inputs + left halo into threadgroup
// memory once. The first w-1 outputs underflow (zero-filled) and are
// structurally null — the host masks them; their value here is don't-care.
//
// Grid: dispatch n threads, threadgroup width = TG_SIZE.

constant constexpr uint TG_SIZE = 256;   // outputs per threadgroup (<=1024)
constant constexpr uint MAX_W   = 4096;  // host guarantees w <= MAX_W (else CPU)

kernel void rolling_sum_f32(
    device const float* input   [[buffer(0)]],
    device       float* output  [[buffer(1)]],
    constant     uint&  n       [[buffer(2)]],
    constant     uint&  w       [[buffer(3)]],
    constant     uint&  is_mean [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup float tile[TG_SIZE + MAX_W];
    uint halo = w - 1;
    uint base = tgid * TG_SIZE;          // first output index of this group
    uint load_count = TG_SIZE + halo;    // inputs needed: [base-halo, base+TG_SIZE)
    for (uint j = lid; j < load_count; j += TG_SIZE) {
        long src = (long)base - (long)halo + (long)j;
        tile[j] = (src >= 0 && src < (long)n) ? input[src] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint i = gid;
    if (i >= n) return;
    uint t0 = lid;                       // window in tile = [t0, t0+w)
    float s = 0.0f;
    for (uint k = 0; k < w; ++k) s += tile[t0 + k];
    output[i] = (is_mean != 0u) ? (s / (float)w) : s;
}
```

- [ ] **Step 4: Write the dispatcher** `crates/polars-metal-kernels/src/rolling.rs` (mirror `filter.rs` for the error enum + buffer/param-buffer creation + `dispatch_1d_with_tg`). Param scalars are passed as 1-element `MetalBuffer`s (the `constant uint& [[buffer(k)]]` convention).

```rust
//! Rolling windowed-statistics kernel dispatchers (M5).
//!
//! Tile-blocked rolling sum/mean/var/std over F32 columns. Scalar params
//! (n, w, op flags) are passed as 1-element MetalBuffers, matching the
//! `constant uint& x [[buffer(k)]]` convention used across `shaders/`.

use crate::command::{CommandQueue, DispatchError};
use crate::shader_lib::{shared_library, ShaderError};
use polars_metal_buffer::{BufferError, MetalBuffer, MetalDevice};

/// Outputs per threadgroup — kept in sync with `TG_SIZE` in shaders/rolling.metal.
pub const TG_SIZE: usize = 256;
/// Max supported window — kept in sync with `MAX_W` in shaders/rolling.metal.
pub const MAX_W: usize = 4096;

#[derive(Debug, thiserror::Error)]
pub enum RollingError {
    #[error("shader library: {0}")]
    Shader(#[from] ShaderError),
    #[error("dispatch: {0}")]
    Dispatch(#[from] DispatchError),
    #[error("buffer: {0}")]
    Buffer(#[from] BufferError),
    #[error("window {w} out of range 1..={MAX_W}")]
    WindowOutOfRange { w: usize },
}

fn u32_buffer(device: &MetalDevice, v: u32) -> Result<MetalBuffer, BufferError> {
    // 1-element u32 buffer for a `constant uint&` kernel arg.
    let bytes = v.to_ne_bytes();
    MetalBuffer::from_bytes(device, &bytes)
}

/// Dispatch `rolling_sum_f32` (is_mean=false) / mean (is_mean=true).
/// `input` and `output` are length-`n` F32 MetalBuffers (output written in place).
pub fn dispatch_rolling_sum_f32(
    device: &MetalDevice,
    input: &MetalBuffer,
    output: &MetalBuffer,
    n: u32,
    w: u32,
    is_mean: bool,
) -> Result<(), RollingError> {
    if w < 1 || w as usize > MAX_W {
        return Err(RollingError::WindowOutOfRange { w: w as usize });
    }
    let lib = shared_library(device)?;
    let pso = lib.pipeline("rolling_sum_f32")?;
    let nb = u32_buffer(device, n)?;
    let wb = u32_buffer(device, w)?;
    let mb = u32_buffer(device, u32::from(is_mean))?;
    let mut q = CommandQueue::new(device)?;
    q.dispatch_1d_with_tg(&pso, &[input, output, &nb, &wb, &mb], n as usize, TG_SIZE)?;
    q.wait_until_complete()?;
    Ok(())
}
```
Add `pub mod rolling;` to `crates/polars-metal-kernels/src/lib.rs`. **Note:** verify `MetalBuffer::from_bytes` exists for a small host slice; if the bridge exposes a different small-buffer constructor (grep `pub fn` in `crates/polars-metal-buffer/src/bridge.rs`), use that and adjust `u32_buffer`. If none exists, add a `from_bytes(device, &[u8]) -> Result<MetalBuffer, BufferError>` helper to the bridge in this task.

- [ ] **Step 5: Verify it passes** — `cargo test -p polars-metal-kernels --test test_rolling 2>&1 | tail`. Expected: 2 passed. Fix MSL/dispatcher until green. (The build compiles `shaders/rolling.metal` into the metallib via the kernels-crate build; confirm the build picks up the new shader — check `crates/polars-metal-kernels/build.rs` or `Cargo.toml` shader list and add `rolling` if shaders are enumerated.)

- [ ] **Step 6: Commit** — `git add shaders/rolling.metal crates/polars-metal-kernels/ && git commit -m "M5 rolling: rolling_sum_f32 tile-blocked kernel + dispatcher"` (end body with the Co-Authored-By line).

---

## Phase 2 — `rolling_var_f32` kernel (var + std)

### Task 2: Centered two-pass variance kernel + dispatcher + test

**Files:** Modify `shaders/rolling.metal`, `crates/polars-metal-kernels/src/rolling.rs`, `tests/test_rolling.rs`

Variance via centered two-pass per window (subtract the window mean before squaring → no mean-dominated cancellation). ddof=1 (Polars default). `w <= ddof` → caller falls back / masks (var of a length-1 window is undefined).

- [ ] **Step 1: Failing test** (append to `test_rolling.rs`)

```rust
use polars_metal_kernels::rolling::dispatch_rolling_var_f32;

fn ref_rolling_var(x: &[f32], w: usize, ddof: usize) -> Vec<f64> {
    (0..x.len()).map(|i| {
        if i + 1 < w { return f64::NAN; }
        let win: Vec<f64> = (i + 1 - w..=i).map(|j| x[j] as f64).collect();
        let mu = win.iter().sum::<f64>() / w as f64;
        win.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / (w - ddof) as f64
    }).collect()
}

fn run_var(x: &[f32], w: usize, is_std: bool) -> Vec<f32> {
    let device = MetalDevice::system_default().unwrap();
    let n = x.len();
    let mut out = vec![0.0f32; n];
    let inb = unsafe { MetalBuffer::from_borrowed_f32(&device, x.as_ptr(), n) }.unwrap();
    let outb = unsafe { MetalBuffer::from_borrowed_f32(&device, out.as_mut_ptr(), n) }.unwrap();
    dispatch_rolling_var_f32(&device, &inb, &outb, n as u32, w as u32, 1, is_std).unwrap();
    out
}

#[test]
fn rolling_var_std_match_reference() {
    let n = 600usize;
    let x: Vec<f32> = (0..n).map(|i| ((i * 7 % 13) as f32) + 1000.0).collect(); // large offset stresses cancellation
    for w in [2usize, 5, 256, 300] {
        let var = run_var(&x, w, false);
        let std = run_var(&x, w, true);
        let rv = ref_rolling_var(&x, w, 1);
        for i in (w - 1)..n {
            assert!((var[i] as f64 - rv[i]).abs() < 1e-2, "var w={w} i={i} got={} want={}", var[i], rv[i]);
            assert!((std[i] as f64 - rv[i].sqrt()).abs() < 1e-2, "std w={w} i={i}");
        }
    }
}
```

- [ ] **Step 2: Verify it fails** — `cargo test -p polars-metal-kernels --test test_rolling rolling_var 2>&1 | head`. Expected: `dispatch_rolling_var_f32` unresolved.

- [ ] **Step 3: Add the var kernel** to `shaders/rolling.metal` (shares the tile-load preamble):

```metal
kernel void rolling_var_f32(
    device const float* input   [[buffer(0)]],
    device       float* output  [[buffer(1)]],
    constant     uint&  n       [[buffer(2)]],
    constant     uint&  w       [[buffer(3)]],
    constant     uint&  ddof    [[buffer(4)]],
    constant     uint&  is_std  [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    threadgroup float tile[TG_SIZE + MAX_W];
    uint halo = w - 1;
    uint base = tgid * TG_SIZE;
    uint load_count = TG_SIZE + halo;
    for (uint j = lid; j < load_count; j += TG_SIZE) {
        long src = (long)base - (long)halo + (long)j;
        tile[j] = (src >= 0 && src < (long)n) ? input[src] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint i = gid;
    if (i >= n) return;
    uint t0 = lid;
    // pass 1: window mean
    float s = 0.0f;
    for (uint k = 0; k < w; ++k) s += tile[t0 + k];
    float mu = s / (float)w;
    // pass 2: centered sum of squares (cancellation-free)
    float ss = 0.0f;
    for (uint k = 0; k < w; ++k) { float d = tile[t0 + k] - mu; ss += d * d; }
    float denom = (float)(w - ddof);            // host guarantees w > ddof
    float var = ss / denom;
    output[i] = (is_std != 0u) ? sqrt(var) : var;
}
```

- [ ] **Step 4: Add the dispatcher** to `rolling.rs`:

```rust
/// Dispatch `rolling_var_f32` (is_std=false) / std (is_std=true). `ddof` is
/// typically 1 (Polars sample default). Caller must ensure `w > ddof`.
pub fn dispatch_rolling_var_f32(
    device: &MetalDevice,
    input: &MetalBuffer,
    output: &MetalBuffer,
    n: u32,
    w: u32,
    ddof: u32,
    is_std: bool,
) -> Result<(), RollingError> {
    if w < 1 || w as usize > MAX_W {
        return Err(RollingError::WindowOutOfRange { w: w as usize });
    }
    let lib = shared_library(device)?;
    let pso = lib.pipeline("rolling_var_f32")?;
    let nb = u32_buffer(device, n)?;
    let wb = u32_buffer(device, w)?;
    let db = u32_buffer(device, ddof)?;
    let sb = u32_buffer(device, u32::from(is_std))?;
    let mut q = CommandQueue::new(device)?;
    q.dispatch_1d_with_tg(&pso, &[input, output, &nb, &wb, &db, &sb], n as usize, TG_SIZE)?;
    q.wait_until_complete()?;
    Ok(())
}
```

- [ ] **Step 5: Verify it passes** — `cargo test -p polars-metal-kernels --test test_rolling 2>&1 | tail`. Expected: all passed.

- [ ] **Step 6: Commit** — `git commit -am "M5 rolling: rolling_var_f32 centered two-pass kernel + dispatcher"`.

---

## Phase 3 — PyO3 binding

### Task 3: `execute_rolling` pyfunction

**Files:** Modify `crates/polars-metal-core/src/udf.rs`, `crates/polars-metal-core/src/lib.rs`; Test: `tests/python_integration/test_rolling_binding.py`

- [ ] **Step 1: Failing test**

```python
# tests/python_integration/test_rolling_binding.py
import numpy as np
from polars_metal import _native

def test_execute_rolling_sum_matches_numpy():
    x = np.arange(1, 11, dtype=np.float32)
    out = np.zeros_like(x)
    # op codes: 0=sum,1=mean,2=var,3=std ; ddof for var/std
    _native.execute_rolling(
        inp=(x.ctypes.data, x.size),
        out=(out.ctypes.data, out.size),
        w=3, op=0, ddof=1,
    )
    # window [1,2,3]=6 at idx2, [8,9,10]=27 at idx9
    assert abs(out[2] - 6.0) < 1e-5
    assert abs(out[9] - 27.0) < 1e-5
```

- [ ] **Step 2: Verify it fails** — `make wheel && pytest tests/python_integration/test_rolling_binding.py -q`. Expected: `_native` has no `execute_rolling`.

- [ ] **Step 3: Implement** in `crates/polars-metal-core/src/udf.rs` (mirror `execute_fused_expr`'s pointer-tuple staging; output-zero-copy into the caller's `out`). `op`: 0=sum,1=mean,2=var,3=std.

```rust
#[pyfunction]
#[pyo3(signature = (inp, out, w, op, ddof=1))]
pub fn execute_rolling(
    inp: (usize, usize),
    out: (usize, usize),
    w: u32,
    op: u32,
    ddof: u32,
) -> PyResult<()> {
    use polars_metal_kernels::rolling::{dispatch_rolling_sum_f32, dispatch_rolling_var_f32};
    let device = MetalDevice::system_default().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: metal device unavailable: {e}"))
    })?;
    let (in_ptr, in_n) = inp;
    let (out_ptr, out_n) = out;
    if in_n != out_n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "polars_metal: rolling input/output length mismatch",
        ));
    }
    // SAFETY: caller (Python dispatch) guarantees both pointers address `in_n`
    // live, contiguous f32 for the whole call (the source Series is rechunked
    // and held; `out` is a freshly-allocated contiguous f32 array).
    let inb = unsafe {
        polars_metal_buffer::MetalBuffer::from_borrowed_f32(&device, in_ptr as *const f32, in_n)
    }
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: rolling input staging: {e}")))?;
    let outb = unsafe {
        polars_metal_buffer::MetalBuffer::from_borrowed_f32(&device, out_ptr as *const f32, out_n)
    }
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: rolling output staging: {e}")))?;
    let n = in_n as u32;
    let res = match op {
        0 => dispatch_rolling_sum_f32(&device, &inb, &outb, n, w, false),
        1 => dispatch_rolling_sum_f32(&device, &inb, &outb, n, w, true),
        2 => dispatch_rolling_var_f32(&device, &inb, &outb, n, w, ddof, false),
        3 => dispatch_rolling_var_f32(&device, &inb, &outb, n, w, ddof, true),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "polars_metal: unknown rolling op {other}"
            )))
        }
    };
    res.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: rolling dispatch: {e}")))
}
```
Register in `crates/polars-metal-core/src/lib.rs` `_native` pymodule: `m.add_function(wrap_pyfunction!(udf::execute_rolling, m)?)?;`. **Note:** `MetalBuffer::from_borrowed_f32` on the `out` pointer must yield a buffer Metal can *write* (unified memory; the existing `eval_into` path writes into a borrowed-out buffer the same way — confirm `from_borrowed_f32` is usable as an output target, else allocate an output `MetalBuffer` and copy back into the `out` slice after `wait_until_complete`).

- [ ] **Step 4: Verify it passes** — `make wheel && pytest tests/python_integration/test_rolling_binding.py -q`. Expected: PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "M5 rolling: execute_rolling PyO3 binding"`.

---

## Phase 4 — Detection

### Task 4: `find_rolling_bindings` (serialize parse)

**Files:** Create `python/polars_metal/_rolling_detect.py`; Test `tests/python_integration/test_rolling_detect.py`

- [ ] **Step 1: Pin the JSON shape empirically (do FIRST).** Run a throwaway script that prints `pl.col("x").rolling_mean(3).meta.serialize(format="json")` (and for `rolling_sum/var/std`, and a variant with `center=True`, `min_samples=1`, `weights=[...]`). Record the exact node name (`Function`? `Window`? `RollingExpr`?), the function-data field carrying the window/options, and how the column argument appears. Pin the discovered shape in a module docstring comment. (Earlier work confirmed `cum_sum`→`{"CumSum":{"reverse":false}}` and `shift`→`"Shift"`; rolling's tag must be verified the same way — do not guess.)

- [ ] **Step 2: Failing test**

```python
# tests/python_integration/test_rolling_detect.py
import polars as pl
from polars_metal._rolling_detect import find_rolling_bindings

def test_detects_rolling_mean_f32_default():
    lf = pl.DataFrame({"x": pl.Series([1.0,2,3], dtype=pl.Float32)}).lazy().with_columns(
        r=pl.col("x").rolling_mean(3))
    found = find_rolling_bindings(lf)
    assert len(found) == 1
    b = found[0]
    assert (b.op, b.column, b.window, b.out_name) == ("mean", "x", 3, "r")

def test_rejects_non_f32_and_options():
    base = pl.DataFrame({"x": pl.Series([1.0,2,3], dtype=pl.Float64)}).lazy()
    assert find_rolling_bindings(base.with_columns(r=pl.col("x").rolling_mean(3))) == []  # F64
    f32 = pl.DataFrame({"x": pl.Series([1.0,2,3], dtype=pl.Float32)}).lazy()
    assert find_rolling_bindings(f32.with_columns(r=pl.col("x").rolling_mean(3, center=True))) == []
    assert find_rolling_bindings(f32.with_columns(r=pl.col("x").rolling_mean(3, min_samples=1))) == []
```

- [ ] **Step 3: Verify it fails** — `pytest tests/python_integration/test_rolling_detect.py -q`. Expected: module missing.

- [ ] **Step 4: Implement** `python/polars_metal/_rolling_detect.py`. Parse `lf.serialize(format="json")` (suppress the deprecation warning), walk the top `HStack`/`with_columns` expressions, and return a `RollingBinding` only for handleable shapes. Reject (omit) when: op ∉ {sum,mean,var,std}; argument isn't a bare `Column`; `schema[col] != Float32`; non-default options (`weights`, `center`, `min_samples`/`min_periods` ≠ window, `by`); `window < 1 or window > MAX_W` (MAX_W=4096, import-mirror the kernel constant). Use the exact field names pinned in Step 1.

```python
"""Detect handleable native rolling_* bindings from a LazyFrame's
pre-optimization serialized plan, for the M5 custom-kernel path.

Serialized JSON shape (py-1.40.1, pinned empirically in Step 1):
  <FILL FROM STEP 1 — e.g. {"Function": {"input": [...], "function": {"RollingExpr"|...: {...}}}}>
Anything not matching the handleable shape is omitted → native Polars/CPU.
"""
from __future__ import annotations
import json
import warnings
from dataclasses import dataclass
import polars as pl

MAX_W = 4096  # keep in sync with shaders/rolling.metal / rolling.rs

_OP_NAMES = {"rolling_mean": "mean", "rolling_sum": "sum", "rolling_var": "var", "rolling_std": "std"}

@dataclass(frozen=True)
class RollingBinding:
    op: str          # "mean" | "sum" | "var" | "std"
    column: str
    window: int
    out_name: str
    ddof: int = 1

def find_rolling_bindings(lf: pl.LazyFrame) -> list[RollingBinding]:
    schema = lf.collect_schema()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = json.loads(lf.serialize(format="json"))
    bindings: list[RollingBinding] = []
    # Walk to the outermost HStack's named expressions. <Use the Step-1 shape.>
    # For each (out_name, expr_json): parse a handleable rolling node or skip.
    # ... implementation per pinned shape; on ANY parse uncertainty, skip (omit). ...
    return bindings
```
(The walk is concrete once Step 1 pins the shape; implement it against the recorded JSON. Guard every field access — a `KeyError`/unexpected shape means "not handleable" → skip, never raise.)

- [ ] **Step 5: Verify it passes** — `pytest tests/python_integration/test_rolling_detect.py -q`. Expected: PASS.

- [ ] **Step 6: Commit** — `git add -A && git commit -m "M5 rolling: detect handleable rolling_* from serialized plan"`.

---

## Phase 5 — Dispatch, fold-back, collect-wrapper wiring

### Task 5: `apply_rolling` + collect-wrapper integration (mean/sum e2e)

**Files:** Create `python/polars_metal/_rolling_dispatch.py`; Modify `python/polars_metal/__init__.py`; Test `tests/python_integration/test_rolling_e2e.py`

- [ ] **Step 1: Failing e2e test**

```python
# tests/python_integration/test_rolling_e2e.py
import numpy as np, polars as pl
from polars.testing import assert_frame_equal
import polars_metal

def test_rolling_mean_sum_e2e_match_cpu():
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"x": rng.standard_normal(4096).astype(np.float32)})
    eng = polars_metal.MetalEngine()
    for op, w in [("mean", 64), ("sum", 50)]:
        expr = getattr(pl.col("x"), f"rolling_{op}")(w)
        lf = df.lazy().with_columns(r=expr)
        assert_frame_equal(lf.collect(engine=eng), lf.collect(), check_exact=False, rtol=1e-4, atol=1e-4)
        # first w-1 structurally null
        assert lf.collect(engine=eng)["r"][:w-1].null_count() == w-1
```

- [ ] **Step 2: Verify it fails** — `pytest tests/python_integration/test_rolling_e2e.py -q`. Expected: values/nulls differ (not yet wired).

- [ ] **Step 3: Implement** `python/polars_metal/_rolling_dispatch.py`:

```python
"""Execute detected rolling bindings via the custom Metal kernel and stitch
results onto the collected frame. Collect-and-stitch over whole, materialized
columns (chunk-safe): no map_batches, no streaming."""
from __future__ import annotations
import numpy as np, polars as pl
from polars_metal import _native
from polars_metal._rolling_detect import RollingBinding

_OP_CODE = {"sum": 0, "mean": 1, "var": 2, "std": 3}

def _rolling_series(src: pl.Series, b: RollingBinding) -> pl.Series:
    s = src.rechunk()                                  # contiguous F32 buffer
    x = s.to_numpy()                                   # F32, no copy when contiguous
    out = np.empty(x.shape[0], dtype=np.float32)
    _native.execute_rolling(
        inp=(x.ctypes.data, x.size),
        out=(out.ctypes.data, out.size),
        w=b.window, op=_OP_CODE[b.op], ddof=b.ddof,
    )
    res = pl.Series(b.out_name, out, dtype=pl.Float32)
    # First w-1 rows are structurally null (insufficient window).
    if b.window > 1:
        mask = pl.Series(np.arange(x.shape[0]) >= (b.window - 1))
        res = pl.select(pl.when(mask).then(res).otherwise(None).alias(b.out_name)).to_series()
    return res

def apply_rolling(lf: pl.LazyFrame, bindings: list[RollingBinding], collect_fn) -> pl.DataFrame:
    """`collect_fn(lf_rest) -> DataFrame` runs the existing in-memory metal collect.
    Splits the rolling output columns out, collects the rest, computes each
    rolling column on the GPU, and stitches in the original column order."""
    out_names = [b.out_name for b in bindings]
    # lf_rest = same frame minus the rolling output columns.
    rest = lf.drop(out_names)
    df = collect_fn(rest)
    cols = {c: df.get_column(c) for c in df.columns}
    for b in bindings:
        cols[b.out_name] = _rolling_series(df.get_column(b.column), b)
    # original output order = rest columns then rolling, unless rolling replaced
    # an existing col — preserve lf's declared output schema order.
    order = lf.collect_schema().names()
    return pl.DataFrame([cols[c] for c in order])
```
Then in `python/polars_metal/__init__.py` `collect_wrapper`, before the `return original_collect(...)` for the MetalEngine branch:

```python
            # M5 rolling: serialize-detected rolling_* run on a custom Metal kernel.
            # Skip under streaming (adapter is in-memory only) and when nothing matches.
            from polars_metal import _rolling_detect, _rolling_dispatch
            streaming = bool(kwargs.get("streaming") or kwargs.get("new_streaming"))
            rolling_bindings = [] if streaming else _rolling_detect.find_rolling_bindings(self)
            if rolling_bindings:
                def _collect_rest(rest_lf):
                    return original_collect(rest_lf, engine="cpu", post_opt_callback=cb, **kwargs)
                return _rolling_dispatch.apply_rolling(self, rolling_bindings, _collect_rest)
```
(Place this after `cb`/`kwargs` are finalized. Keep the existing `return original_collect(...)` as the no-rolling path.) **Verify** `lf.drop(out_names)` cleanly removes only the rolling outputs for the bounded shape (a `with_columns` adding new columns); if the rolling expr *replaces* a source column or shares a `with_columns` with other exprs that can't be split, `find_rolling_bindings` must already have rejected it — add that guard in Task 4 if Step 1's shape allows detecting it.

- [ ] **Step 4: Verify it passes** — `pytest tests/python_integration/test_rolling_e2e.py -q`. Expected: PASS (values within tol; first w-1 null).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "M5 rolling: collect-wrapper dispatch + stitch (mean/sum e2e); streaming guard"`.

---

## Phase 6 — var/std e2e, fallbacks, properties, conformance, perf, docs

### Task 6: var/std end-to-end

**Files:** Test `tests/python_integration/test_rolling_var_std.py`

- [ ] **Step 1: Failing test**

```python
import numpy as np, polars as pl
from polars.testing import assert_frame_equal
import polars_metal

def test_rolling_var_std_e2e_match_cpu():
    df = pl.DataFrame({"x": np.random.default_rng(1).standard_normal(2048).astype(np.float32)})
    eng = polars_metal.MetalEngine()
    for op in ("var", "std"):
        lf = df.lazy().with_columns(r=getattr(pl.col("x"), f"rolling_{op}")(32))
        assert_frame_equal(lf.collect(engine=eng), lf.collect(), check_exact=False, rtol=1e-3, atol=1e-4)
```

- [ ] **Step 2: Verify** — `pytest tests/python_integration/test_rolling_var_std.py -q`. If it fails only on the `w<=ddof` edge or numerical tol, adjust the detector guard (`window > ddof`) / tolerance; the kernel (Task 2) already handles var/std, so this should pass once detection routes var/std (confirm Task 4 maps `rolling_var`/`rolling_std`).

- [ ] **Step 3: Commit** — `git commit -am "M5 rolling: rolling_var/std end-to-end"`.

### Task 7: Fallback guards

**Files:** Test `tests/python_integration/test_rolling_fallback.py`

- [ ] **Step 1: Failing test** — reuse a dispatch counter (monkeypatch `_native.execute_rolling`); assert 0 dispatches AND equality with CPU for: `center=True`, `min_samples=1`, `weights=[...]`, F64 column, null-bearing F32 column, `window > MAX_W`, and a streaming collect.

```python
import numpy as np, polars as pl
from polars.testing import assert_frame_equal
import polars_metal
from polars_metal import _native

def _count(lf, eng):
    n = {"c": 0}; orig = _native.execute_rolling
    def cnt(**kw): n["c"] += 1; return orig(**kw)
    _native.execute_rolling = cnt
    try: out = lf.collect(engine=eng)
    finally: _native.execute_rolling = orig
    return n["c"], out

def test_fallbacks_route_zero_and_match_cpu():
    eng = polars_metal.MetalEngine()
    f32 = pl.DataFrame({"x": np.arange(10, dtype=np.float32)}).lazy()
    cases = [
        f32.with_columns(r=pl.col("x").rolling_mean(3, center=True)),
        f32.with_columns(r=pl.col("x").rolling_mean(3, min_samples=1)),
        pl.DataFrame({"x": np.arange(10, dtype=np.float64)}).lazy().with_columns(r=pl.col("x").rolling_mean(3)),
        pl.DataFrame({"x": pl.Series([1.0,None,3,4,5], dtype=pl.Float32)}).lazy().with_columns(r=pl.col("x").rolling_mean(2)),
    ]
    for lf in cases:
        c, out = _count(lf, eng)
        assert c == 0
        assert_frame_equal(out, lf.collect())
```

- [ ] **Step 2: Verify it passes** — `pytest tests/python_integration/test_rolling_fallback.py -q`. Fix detector guards (Task 4) until all route 0 and match CPU.

- [ ] **Step 3: Commit** — `git commit -am "M5 rolling: fallback guards (options/F64/null/large-w/streaming) → CPU"`.

### Task 8: Differential property test

**Files:** Create `tests/python_integration/test_rolling_property.py`

- [ ] **Step 1: Write** a randomized differential test (50 iters, random `n∈[1,5000]`, `w∈[1,min(n,512)]`, random F32 values) over {mean,sum,var,std}, `engine="metal"` == CPU within `rtol=1e-3, atol=1e-4`.

```python
import numpy as np, polars as pl
from polars.testing import assert_frame_equal
import polars_metal

def test_rolling_matches_cpu_random():
    eng = polars_metal.MetalEngine(); rng = np.random.default_rng(7)
    for _ in range(50):
        n = int(rng.integers(1, 5000)); w = int(rng.integers(1, max(2, min(n, 512))))
        x = rng.standard_normal(n).astype(np.float32)
        df = pl.DataFrame({"x": x})
        for op in ("mean", "sum", "var", "std"):
            lf = df.lazy().with_columns(r=getattr(pl.col("x"), f"rolling_{op}")(w))
            assert_frame_equal(lf.collect(engine=eng), lf.collect(), check_exact=False, rtol=1e-3, atol=1e-4)
```

- [ ] **Step 2: Run** — `pytest tests/python_integration/test_rolling_property.py -q`. Fix any boundary/tolerance mismatch (this is the correctness gate). Expected: PASS.

- [ ] **Step 3: Commit** — `git commit -m "M5 rolling: differential property test vs Polars CPU"`.

### Task 9: O(N) prefix-scan optimization for `rolling_sum_f32`

**Files:** Modify `shaders/rolling.metal` (sum path), `tests/test_rolling.rs` (add a large-`w` case)

The Task 1 kernel is O(N·w). Convert the sum path to a tile-local **inclusive prefix sum** so window sum = `pref[t0+w-1] - (t0>0 ? pref[t0-1] : 0)`, making it O(N) regardless of `w`. The Task 1/8 correctness tests guard the conversion.

- [ ] **Step 1: Add a large-window kernel test** (e.g. `w=1000`, `n=20000`) to `test_rolling.rs` asserting equality to the f64 reference (tight tol). Run — passes with the O(N·w) kernel (baseline), then must still pass after the rewrite.
- [ ] **Step 2: Rewrite** the `rolling_sum_f32` body to compute a cooperative tile-local inclusive prefix sum over `tile[0..TG_SIZE+halo)` (Hillis–Steele in threadgroup memory, with `threadgroup_barrier` between steps), then `sum = pref[t0 + w - 1] - (t0 == 0 ? 0.0f : pref[t0 - 1])`. Keep magnitudes tile-scale (prefix over ≤ `TG_SIZE+MAX_W` values). Leave `rolling_var_f32` as the two-pass kernel (var isn't the bench target).
- [ ] **Step 3: Verify** — `cargo test -p polars-metal-kernels --test test_rolling 2>&1 | tail` all pass (incl. large-w). Re-run `pytest tests/python_integration/test_rolling_property.py -q`.
- [ ] **Step 4: Commit** — `git commit -am "M5 rolling: O(N) prefix-scan rolling_sum (perf), correctness preserved"`.

### Task 10: Conformance stays at baseline

- [ ] **Step 1: Run** `make test-conformance 2>&1 | tail -6`. Expected: only `lazyframe` + `operations_group_by` fail; any Polars rolling tests now route + pass or fall back + pass. Investigate/fix any new regression before proceeding.
- [ ] **Step 2: Commit** any conformance-driven fixes — `git commit -am "M5 rolling: conformance fixes"` (skip if none).

### Task 11: Bench + baseline gate

**Files:** `tests/bench/m4_survey/bench_rolling_mlx.py`, `tests/bench/baseline.json`

- [ ] **Step 1: Measure** `df.lazy().with_columns(pl.col("x").rolling_mean(1000)).collect(engine="metal")` vs CPU at 10M F32; record median ms (extend the existing bench for the engine path).
- [ ] **Step 2: Flip** `phase9_rolling_mean_w1000_10m` in `baseline.json` from `_pending` to measured with `_gate.ratio_lt` (~0.1 for a 10× target; set from the measured ratio with headroom). Run `pytest tests/bench/test_baseline_gate.py -q`.
- [ ] **Step 3: Commit** — `git commit -m "M5 rolling: record rolling_mean 10M baseline + ratio_lt gate"`.

### Task 12: Final gate + docs

- [ ] **Step 1:** `make lint` — clean.
- [ ] **Step 2:** `make test-unit && make test-kernel` — green (incl. `test_rolling.rs`).
- [ ] **Step 3:** `python -m pytest tests/python_integration -q` — only the pre-existing groupby Mean-dtype deferral fails.
- [ ] **Step 4:** Update `CLAUDE.md` M5 roadmap entry (rolling delivered via custom kernel) and `docs/open-questions.md` (note the deferred opacity subproject for dt/list/fft/corr; F64/int/streaming rolling out of scope).
- [ ] **Step 5: Commit** — `git commit -m "M5 rolling: docs + roadmap update (rolling delivered via custom kernel)"`.

---

## Test plan
- [ ] Kernel correctness (Rust): sum/mean/var/std vs F64 reference; multi-tile; tile-boundary windows; large-w (Task 9).
- [ ] PyO3 binding smoke (Task 3).
- [ ] Detection: handleable shapes recognized, options/F64 rejected (Task 4).
- [ ] e2e mean/sum/var/std == CPU incl. head nulls (Tasks 5, 6).
- [ ] Fallback: options/F64/null/large-w/streaming → CPU, exact (Task 7).
- [ ] Differential property (random n/w/values × 4 ops) — the correctness gate (Task 8).
- [ ] Conformance at baseline (Task 10); bench gated (Task 11); final gate (Task 12).

## Open items (resolve during implementation, don't guess)
- Exact `rolling_*` JSON node shape at the pinned Polars rev (Task 4 Step 1) — verify empirically, pin in a comment; add a version-probe guard so a Polars bump degrades to CPU rather than mis-parsing.
- `MetalBuffer` small-scalar constructor (`from_bytes`) and write-target borrow for `out` (Tasks 1, 3) — confirm against the buffer bridge; add a helper if absent.
- Whether `lf.drop(out_names)` / column-order restoration covers all bounded rolling `with_columns` shapes (Task 5) — reject un-splittable shapes in detection.
- Shifted-data prefix variance (a future optimization over the Task 2 two-pass var) — only if a var bench demands it.
