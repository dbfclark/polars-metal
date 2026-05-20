# M1 — Scan / Project / Filter on GPU — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the first real GPU execution path. `df.collect(engine=MetalEngine())` runs scan / project / filter on GPU for i64 / f64 / bool columns with the closed set of predicate shapes defined in the M1 spec, byte-identical to CPU Polars on null-heavy inputs, no slower than CPU by more than 5% on M2 Ultra.

**Architecture:** A bottom-up IR walker in `polars-metal-core` produces `Handled(MetalPlanNode)` or `FallBack` per node; only fully-handled subtrees get `nt.set_udf(...)`. Five MSL kernels (`cmp_i64`, `cmp_f64`, `logical_bool`, `filter_predicate`, `filter_scatter_*`) compiled into a single metallib at build time, dispatched via `polars-metal-kernels` wrappers. MLX `cumsum` provides the prefix-sum for stream compaction. Per-query `ScratchArena` owns all intermediate and output `MetalBuffer`s; deallocators tied into Polars' output Arrow buffers keep the arena alive across the UDF boundary.

**Tech Stack:** Rust 2021 (workspace), `objc2-metal` for `MTLDevice` / `MTLBuffer` / `MTLCommandBuffer`, `cxx` for MLX FFI (M0 commitment, M2 reconsidered), `proptest` for kernel correctness, `pyo3 0.22` + `maturin` for the Python extension, `polars` pinned to `py-1.40.1` (M0 commitment), `hypothesis` for Python differential, `pytest-benchmark` + `criterion` for perf, `xcrun metal` / `xcrun metallib` invoked from `build.rs` for shader compilation.

**Spec:** [`docs/superpowers/specs/2026-05-20-m1-design.md`](../specs/2026-05-20-m1-design.md). All decisions there are binding; this plan does not relitigate them.

**Conventions** (per CLAUDE.md): No `unwrap()` outside tests. No `unsafe` outside `*-sys` crates and the buffer bridge — each with a `// SAFETY:` comment. One MSL kernel per file. Errors propagate as `polars.exceptions.ComputeError` at the engine boundary. Null semantics match Polars exactly. Don't add files to `shaders/` without a matching test in the kernel crate. Read the matching cuDF kernel before writing MSL.

**Pre-task reading.** Before Phase 4 (walker), read:
- `references/cudf/python/cudf_polars/dsl/__init__.py` — how cuDF-polars dispatches IR nodes; we mirror its shape.
- `references/cudf/python/cudf_polars/dsl/ir.py` — its per-node handlers.
- `references/polars/crates/polars-plan/src/plans/ir/mod.rs` — the canonical IR enum we're matching against.

Before any MSL kernel task (Phase 5+), read:
- `references/cudf/cpp/src/copying/copy_if.cu` — cuDF's stream compaction pattern.
- `references/cudf/cpp/src/copying/concatenate.cu` — atomic validity-bit writes.

---

## Phase 0 — Preflight

### Task 1: Verify M0 gates still green; confirm dev env

**Files:** none (verification only).

- [ ] **Step 1: Check we're on a fresh branch off main**

Run: `git rev-parse --abbrev-ref HEAD && git status --porcelain`
Expected: `m1-scan-project-filter` and empty status.

- [ ] **Step 2: Run the M0 gate**

Run: `make gate`
Expected: passes. Total wall-clock ~6s on M2 Ultra per M0 retro.
If anything fails: stop and fix before starting M1; do not pile new work on top of a broken baseline.

- [ ] **Step 3: Verify Metal toolchain present**

Run: `xcrun metal --version && xcrun metallib --version`
Expected: both report a version (e.g. `Metal toolchain ...`).
If missing: `sudo xcodebuild -runFirstLaunch && xcodebuild -downloadComponent MetalToolchain` (per M0's resolution of the same issue).

- [ ] **Step 4: Verify reference clones pin matches the spec**

Run: `(cd references/polars && git rev-parse HEAD) && (cd references/cudf && git rev-parse HEAD)`
Expected: Polars at the `py-1.40.1` tag SHA, cuDF at the SHA from M0.
If drift: `bash scripts/refresh-references.sh`.

Nothing to commit in Task 1.

---

## Phase 1 — Shader build infrastructure

### Task 2: Add the `shaders/` directory and a hello-world kernel that validates the build pipeline

The metallib compile/load path must be working before any kernel work begins. We prove it end-to-end with a kernel that writes a constant.

**Files:**
- Create: `shaders/_hello.metal`
- Create: `crates/polars-metal-kernels/build.rs`
- Modify: `crates/polars-metal-kernels/Cargo.toml`
- Create: `crates/polars-metal-kernels/src/shader_lib.rs`
- Modify: `crates/polars-metal-kernels/src/lib.rs`
- Create: `crates/polars-metal-kernels/tests/test_shader_lib.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-kernels/tests/test_shader_lib.rs
use polars_metal_kernels::shader_lib::ShaderLibrary;
use polars_metal_buffer::MetalDevice;

#[test]
fn loads_metallib_and_finds_hello_kernel() {
    let device = MetalDevice::system_default().expect("Metal-capable hardware");
    let lib = ShaderLibrary::load(&device).expect("metallib must load");
    let pso = lib.pipeline("hello_write_constant").expect("entry point must exist");
    assert!(pso.max_total_threads_per_threadgroup() > 0);
}
```

- [ ] **Step 2: Write the hello-world MSL kernel**

```msl
// shaders/_hello.metal
#include <metal_stdlib>
using namespace metal;

kernel void hello_write_constant(
    device uint32_t* out [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
    out[gid] = 42u;
}
```

- [ ] **Step 3: Write `build.rs` that compiles every `.metal` in `shaders/` to a metallib**

```rust
// crates/polars-metal-kernels/build.rs
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let shaders_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../shaders");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR set by cargo"));
    let metallib_path = out_dir.join("polars_metal.metallib");

    println!("cargo:rerun-if-changed={}", shaders_dir.display());

    let mut air_files = Vec::new();
    for entry in std::fs::read_dir(&shaders_dir).expect("shaders dir exists") {
        let path = entry.expect("readable").path();
        if path.extension().and_then(|s| s.to_str()) != Some("metal") {
            continue;
        }
        // Skip files starting with '_' if they're meant as headers (none in M0; convention starts in M1)
        // For now, compile every .metal — headers are handled via `#include`.
        let stem = path.file_stem().expect("has stem").to_string_lossy().to_string();
        let air_path = out_dir.join(format!("{stem}.air"));
        let status = Command::new("xcrun")
            .args(["metal", "-c", "-frecord-sources", "-o"])
            .arg(&air_path)
            .arg(&path)
            .status()
            .expect("xcrun metal runs");
        assert!(status.success(), "metal compile failed for {}", path.display());
        air_files.push(air_path);
    }

    let mut cmd = Command::new("xcrun");
    cmd.args(["metallib", "-o"]).arg(&metallib_path);
    for f in &air_files {
        cmd.arg(f);
    }
    let status = cmd.status().expect("xcrun metallib runs");
    assert!(status.success(), "metallib link failed");

    println!("cargo:rustc-env=POLARS_METAL_METALLIB={}", metallib_path.display());
}
```

- [ ] **Step 4: Write `shader_lib.rs`**

```rust
// crates/polars-metal-kernels/src/shader_lib.rs
//
// Loads the embedded metallib (compiled by build.rs) at most once per
// process and caches MTLComputePipelineState by entry-point name.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLComputePipelineState, MTLDevice as _, MTLLibrary as _};
use polars_metal_buffer::MetalDevice;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const METALLIB_BYTES: &[u8] = include_bytes!(env!("POLARS_METAL_METALLIB"));

pub struct ShaderLibrary {
    library: Retained<ProtocolObject<dyn objc2_metal::MTLLibrary>>,
    psos: Mutex<HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
}

#[derive(Debug, thiserror::Error)]
pub enum ShaderError {
    #[error("metallib failed to load")]
    LibraryLoad,
    #[error("unknown kernel entry point: {0}")]
    UnknownEntryPoint(String),
    #[error("pipeline state object creation failed: {0}")]
    PipelineStateFailed(String),
}

impl ShaderLibrary {
    pub fn load(device: &MetalDevice) -> Result<Self, ShaderError> {
        // ... use device.raw().newLibraryWithData_error_(...) wrapping
        // METALLIB_BYTES as a dispatch_data_t. See objc2-metal docs.
        // SAFETY: METALLIB_BYTES has static lifetime; we never mutate.
        todo!("implement: wrap METALLIB_BYTES as dispatch_data, call newLibraryWithData")
    }

    pub fn pipeline(
        &self,
        entry_point: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, ShaderError> {
        let mut psos = self.psos.lock().expect("not poisoned");
        if let Some(pso) = psos.get(entry_point) {
            return Ok(pso.clone());
        }
        // ... look up function in library, build PSO, insert into cache.
        todo!("implement PSO lookup + cache")
    }
}

pub fn shared_library(device: &MetalDevice) -> Result<&'static ShaderLibrary, ShaderError> {
    static INSTANCE: OnceLock<Result<ShaderLibrary, ShaderError>> = OnceLock::new();
    INSTANCE
        .get_or_init(|| ShaderLibrary::load(device))
        .as_ref()
        .map_err(|e| match e {
            ShaderError::LibraryLoad => ShaderError::LibraryLoad,
            ShaderError::UnknownEntryPoint(s) => ShaderError::UnknownEntryPoint(s.clone()),
            ShaderError::PipelineStateFailed(s) => ShaderError::PipelineStateFailed(s.clone()),
        })
}
```

- [ ] **Step 5: Modify `lib.rs` to expose `shader_lib`**

```rust
// crates/polars-metal-kernels/src/lib.rs
pub mod shader_lib;
```

- [ ] **Step 6: Add `objc2-metal`, `polars-metal-buffer`, `thiserror` to the kernels crate**

```toml
# crates/polars-metal-kernels/Cargo.toml — additions only
[dependencies]
objc2 = "0.5"
objc2-metal = "0.2"
polars-metal-buffer = { path = "../polars-metal-buffer" }
thiserror = "1"
```

- [ ] **Step 7: Run the test**

Run: `cargo test -p polars-metal-kernels --test test_shader_lib`
Expected first run (with todo!()s): PANIC at todo!() — fix the todos until the test PASSES.

- [ ] **Step 8: Commit**

```bash
git add shaders/_hello.metal crates/polars-metal-kernels/{build.rs,Cargo.toml,src/shader_lib.rs,src/lib.rs,tests/test_shader_lib.rs}
git commit -m "Compile .metal files to metallib at build time; expose ShaderLibrary"
```

### Task 3: Dispatch the hello-world kernel end-to-end (proves command-queue + buffer + PSO works)

**Files:**
- Create: `crates/polars-metal-kernels/src/command.rs`
- Modify: `crates/polars-metal-kernels/src/lib.rs`
- Create: `crates/polars-metal-kernels/tests/test_dispatch.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-kernels/tests/test_dispatch.rs
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::shader_lib;
use polars_metal_buffer::MetalDevice;

#[test]
fn hello_kernel_writes_42_into_every_slot() {
    let device = MetalDevice::system_default().expect("Metal-capable hardware");
    let lib = shader_lib::shared_library(&device).expect("library loads");
    let pso = lib.pipeline("hello_write_constant").expect("entry point exists");

    let n: usize = 1024;
    // Allocate a u32-aligned scratch buffer
    let mut queue = CommandQueue::new(&device).expect("queue creation");
    let buf = device
        .new_buffer_zeroed(n * std::mem::size_of::<u32>())
        .expect("alloc");

    queue
        .dispatch_1d(&pso, &[&buf], n)
        .expect("dispatch succeeds");
    queue.wait_until_complete().expect("no GPU errors");

    let slice: &[u32] = unsafe { std::slice::from_raw_parts(buf.as_slice().as_ptr() as *const u32, n) };
    for v in slice {
        assert_eq!(*v, 42);
    }
}
```

- [ ] **Step 2: Add `new_buffer_zeroed` to `polars-metal-buffer::MetalDevice`**

Look up the API on `MTLDevice::newBufferWithLength_options_` (objc2-metal). Allocate StorageModeShared, zero the contents (shared storage means CPU can write before submission). Return a `MetalBuffer` with `_owner: None`. Add a unit test in the buffer crate.

- [ ] **Step 3: Implement `command::CommandQueue`**

```rust
// crates/polars-metal-kernels/src/command.rs
//
// A thin wrapper over MTLCommandQueue + per-dispatch MTLCommandBuffer.
// Owns the queue for the duration of a query.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue as _, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLDevice as _, MTLSize,
};
use polars_metal_buffer::{MetalBuffer, MetalDevice};

pub struct CommandQueue {
    queue: Retained<ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
    in_flight: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}

#[derive(Debug, thiserror::Error)]
pub enum DispatchError {
    #[error("command queue creation failed")]
    QueueCreation,
    #[error("command buffer creation failed")]
    CommandBufferCreation,
    #[error("compute encoder creation failed")]
    EncoderCreation,
    #[error("GPU error: {0}")]
    GpuError(String),
}

impl CommandQueue {
    pub fn new(device: &MetalDevice) -> Result<Self, DispatchError> {
        let queue = device.raw().newCommandQueue().ok_or(DispatchError::QueueCreation)?;
        Ok(Self { queue, in_flight: None })
    }

    /// Dispatch a 1D grid of `n_threads` threads against the given PSO with
    /// the given buffers bound to slots 0..buffers.len().
    pub fn dispatch_1d(
        &mut self,
        pso: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        buffers: &[&MetalBuffer],
        n_threads: usize,
    ) -> Result<(), DispatchError> {
        // Threadgroup size: min(maxThreadsPerThreadgroup, pso.maxTotalThreadsPerThreadgroup, 256 sane default).
        // Per CLAUDE.md gotcha: query at runtime, do not hardcode.
        // ... implement encoder, bind buffers, dispatch, end encoding, commit.
        todo!("implement dispatch")
    }

    pub fn wait_until_complete(&mut self) -> Result<(), DispatchError> {
        if let Some(buf) = self.in_flight.take() {
            // SAFETY: MTLCommandBuffer's waitUntilCompleted blocks until GPU is done.
            unsafe { buf.waitUntilCompleted() };
            // Check buf.error() for GPU errors.
            // ... return Err(GpuError) if non-nil.
        }
        Ok(())
    }
}
```

- [ ] **Step 4: Modify `lib.rs` to expose `command`**

```rust
// crates/polars-metal-kernels/src/lib.rs
pub mod command;
pub mod shader_lib;
```

- [ ] **Step 5: Implement the todos**

Iterate until the test passes. Refer to `references/cudf/cpp/` for any threadgroup-sizing patterns (cuDF launches with explicit block size; we query MTLDevice). Concretely: threadgroup width = min(`pso.max_total_threads_per_threadgroup()`, 256). Grid = ceil(n_threads / threadgroup width).

- [ ] **Step 6: Run the test**

Run: `cargo test -p polars-metal-kernels --test test_dispatch`
Expected: PASS. Every slot reads `42`.

- [ ] **Step 7: Commit**

```bash
git add crates/polars-metal-kernels/src/command.rs crates/polars-metal-kernels/src/lib.rs crates/polars-metal-kernels/tests/test_dispatch.rs crates/polars-metal-buffer/src/device.rs
git commit -m "Dispatch infrastructure: CommandQueue + MetalDevice::new_buffer_zeroed"
```

---

## Phase 2 — Real ScratchArena

### Task 4: Replace `StubArena` with a real bump arena over shared-storage MTLBuffers

**Files:**
- Modify: `crates/polars-metal-core/src/arena.rs`
- Create: `crates/polars-metal-core/tests/test_arena.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-core/tests/test_arena.rs
use polars_metal_core::arena::{ScratchArena, BumpArena};
use polars_metal_buffer::MetalDevice;

#[test]
fn allocs_two_buffers_with_distinct_pointers() {
    let device = MetalDevice::system_default().expect("Metal hardware");
    let mut arena = BumpArena::with_capacity(&device, 1024 * 1024).expect("alloc backing");
    let a = arena.alloc(64).expect("first alloc");
    let b = arena.alloc(64).expect("second alloc");
    assert_ne!(a.as_slice().as_ptr(), b.as_slice().as_ptr());
}

#[test]
fn alignment_at_least_16_bytes() {
    let device = MetalDevice::system_default().expect("Metal hardware");
    let mut arena = BumpArena::with_capacity(&device, 1024).expect("alloc backing");
    let a = arena.alloc(1).expect("first alloc");
    let b = arena.alloc(1).expect("second alloc");
    let pa = a.as_slice().as_ptr() as usize;
    let pb = b.as_slice().as_ptr() as usize;
    assert_eq!(pa % 16, 0);
    assert_eq!(pb % 16, 0);
}

#[test]
fn exhaustion_returns_error_not_panic() {
    let device = MetalDevice::system_default().expect("Metal hardware");
    let mut arena = BumpArena::with_capacity(&device, 256).expect("alloc backing");
    let _ = arena.alloc(200).expect("fits");
    let result = arena.alloc(200);
    assert!(matches!(result, Err(_)), "second 200B alloc must fail");
}
```

- [ ] **Step 2: Implement `BumpArena`**

```rust
// crates/polars-metal-core/src/arena.rs — add BumpArena alongside the existing trait

use polars_metal_buffer::{MetalBuffer, MetalDevice, BufferError};
use std::sync::Arc;

pub struct BumpArena {
    backing: Arc<MetalBuffer>,
    cursor: usize,
}

impl BumpArena {
    /// Pre-allocate one large MTLBuffer; serve allocations as offsets into it.
    /// Capacity is in bytes. Returns Err if the backing alloc fails.
    pub fn with_capacity(device: &MetalDevice, bytes: usize) -> Result<Self, BufferError> {
        let backing = Arc::new(device.new_buffer_zeroed(bytes)?);
        Ok(Self { backing, cursor: 0 })
    }

    /// Hand out a slice of `bytes` bytes, 16-byte-aligned.
    /// All allocations share the same MTLBuffer; the returned MetalBuffer
    /// is a no-op-deallocator view onto the parent's bytes.
    pub fn alloc(&mut self, bytes: usize) -> Result<MetalBuffer, BufferError> {
        let aligned_start = (self.cursor + 15) & !15;
        let aligned_end = aligned_start + bytes;
        if aligned_end > self.backing.len() {
            return Err(BufferError::AllocationFailed { bytes });
        }
        self.cursor = aligned_end;
        // Construct a view MetalBuffer over self.backing[aligned_start..aligned_end].
        // The view keeps an Arc clone of self.backing so the parent stays alive.
        // SAFETY: bounds checked above; the parent's MTLBuffer is StorageModeShared.
        unsafe { MetalBuffer::view_into(&self.backing, aligned_start, bytes) }
    }

    pub fn shared(&self) -> Arc<MetalBuffer> {
        self.backing.clone()
    }
}
```

- [ ] **Step 3: Add `MetalBuffer::view_into`**

In `crates/polars-metal-buffer/src/bridge.rs`, add an unsafe constructor that takes a parent `Arc<MetalBuffer>`, an offset, and a length. The returned buffer holds a clone of the parent's `inner` Retained and a clone of the Arc as `_owner` to keep the parent alive. The view's `as_slice` returns `parent.as_slice()[offset..offset+len]`. Document the SAFETY contract.

(Note: `MetalBuffer` does not currently allow sub-views. This is the M1 addition. Refactor accordingly.)

- [ ] **Step 4: Run the tests**

Run: `cargo test -p polars-metal-core --test test_arena`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-core/src/arena.rs crates/polars-metal-core/tests/test_arena.rs crates/polars-metal-buffer/src/bridge.rs
git commit -m "Real BumpArena + MetalBuffer sub-views for arena allocation"
```

---

## Phase 3 — MLX cumsum binding

### Task 5: Add `cumsum_u8_to_u32` to `polars-metal-mlx-sys`

**Files:**
- Modify: `crates/polars-metal-mlx-sys/src/lib.rs`
- Modify: `crates/polars-metal-mlx-sys/build.rs` (or wherever the cxx bridge is declared)
- Create: `crates/polars-metal-mlx-sys/src/cumsum.cc` (or extend the existing C++ side)
- Create: `crates/polars-metal-mlx-sys/tests/test_cumsum.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-mlx-sys/tests/test_cumsum.rs
use polars_metal_mlx_sys::cumsum_u8_to_u32;

#[test]
fn cumsum_basic_inclusive() {
    let input: Vec<u8> = vec![1, 0, 1, 1, 0, 1];
    let mut output = vec![0u32; input.len()];
    cumsum_u8_to_u32(&input, &mut output).expect("dispatch succeeds");
    assert_eq!(output, vec![1u32, 1, 2, 3, 3, 4]);
}

#[test]
fn cumsum_all_zeros() {
    let input = vec![0u8; 1024];
    let mut output = vec![0u32; 1024];
    cumsum_u8_to_u32(&input, &mut output).unwrap();
    for v in &output {
        assert_eq!(*v, 0);
    }
}

#[test]
fn cumsum_all_ones_large() {
    let input = vec![1u8; 10_000];
    let mut output = vec![0u32; 10_000];
    cumsum_u8_to_u32(&input, &mut output).unwrap();
    for (i, v) in output.iter().enumerate() {
        assert_eq!(*v, (i as u32) + 1);
    }
}
```

- [ ] **Step 2: Extend the cxx C++ side**

```cpp
// crates/polars-metal-mlx-sys/src/cumsum.cc
#include <mlx/mlx.h>
#include <mlx/transforms.h>
#include <vector>
#include <memory>

namespace polars_metal_mlx {

std::unique_ptr<std::vector<uint32_t>>
cumsum_u8_to_u32(const std::vector<uint8_t>& input) {
    mlx::core::StreamContext ctx(mlx::core::Device::gpu);
    mlx::core::Shape shape{static_cast<int32_t>(input.size())};
    auto in = mlx::core::array(input.data(), shape, mlx::core::uint8);
    auto in_u32 = mlx::core::astype(in, mlx::core::uint32);
    auto scanned = mlx::core::cumsum(in_u32, 0, /*reverse=*/false, /*inclusive=*/true);
    mlx::core::eval(scanned);
    auto out = std::make_unique<std::vector<uint32_t>>(input.size());
    std::memcpy(out->data(), scanned.data<uint32_t>(), input.size() * sizeof(uint32_t));
    return out;
}

} // namespace
```

- [ ] **Step 3: Declare in the cxx bridge**

```rust
// crates/polars-metal-mlx-sys/src/lib.rs — add to the existing cxx::bridge module
#[cxx::bridge(namespace = "polars_metal_mlx")]
mod ffi {
    unsafe extern "C++" {
        // ... existing add_f32, etc.
        fn cumsum_u8_to_u32(input: &CxxVector<u8>) -> UniquePtr<CxxVector<u32>>;
    }
}

/// Inclusive cumsum over u8 input, output cast to u32 (so n_rows up to ~4B is safe).
pub fn cumsum_u8_to_u32(input: &[u8], output: &mut [u32]) -> Result<(), MlxError> {
    assert_eq!(input.len(), output.len(), "input and output must have same length");
    let mut cxx_in = cxx::CxxVector::<u8>::new();
    for &b in input {
        cxx_in.pin_mut().push(b);
    }
    let cxx_out = ffi::cumsum_u8_to_u32(&cxx_in);
    if cxx_out.is_null() {
        return Err(MlxError::DispatchFailed);
    }
    for (i, v) in cxx_out.iter().enumerate() {
        output[i] = *v;
    }
    Ok(())
}
```

- [ ] **Step 4: Run the test**

Run: `cargo test -p polars-metal-mlx-sys --test test_cumsum`
Expected: PASS on all three cases.

- [ ] **Step 5: Note for follow-up**

The copy-in / copy-out via `CxxVector` is a perf cost. Acceptable for M1 — the eventual replacement is wiring MLX's `array` directly over a Metal buffer pointer (which is what cuDF does on CUDA). Tracked in `docs/open-questions.md` as part of the M2 MLX-FFI revisit.

- [ ] **Step 6: Commit**

```bash
git add crates/polars-metal-mlx-sys/src/{lib.rs,cumsum.cc} crates/polars-metal-mlx-sys/tests/test_cumsum.rs
git commit -m "MLX cumsum_u8_to_u32 binding for filter compaction"
```

---

## Phase 4 — Walker + Scan + Project end-to-end

Before this phase, read:
- `references/cudf/python/cudf_polars/dsl/__init__.py`
- `references/cudf/python/cudf_polars/dsl/ir.py`
- `references/polars/crates/polars-plan/src/plans/ir/mod.rs`

The walker's shape is borrowed from cuDF-polars. Where we diverge, document why in `docs/architecture.md` at the end of M1.

### Task 6: `MetalPlanNode` intermediate IR

**Files:**
- Create: `crates/polars-metal-core/src/plan/mod.rs`
- Modify: `crates/polars-metal-core/src/lib.rs`
- Create: `crates/polars-metal-core/tests/test_plan_ir.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-core/tests/test_plan_ir.rs
use polars_metal_core::plan::{MetalPlanNode, MetalDtype, PredicateAst, CompareOp};

#[test]
fn constructs_and_inspects_scan_node() {
    let scan = MetalPlanNode::Scan {
        n_rows: 100,
        columns: vec![("a".into(), MetalDtype::I64), ("b".into(), MetalDtype::F64)],
    };
    match scan {
        MetalPlanNode::Scan { n_rows, columns } => {
            assert_eq!(n_rows, 100);
            assert_eq!(columns.len(), 2);
        }
        _ => panic!("expected Scan variant"),
    }
}

#[test]
fn constructs_filter_with_compound_predicate() {
    let pred = PredicateAst::And(
        Box::new(PredicateAst::Compare {
            op: CompareOp::Gt,
            lhs: Box::new(PredicateAst::Column { name: "a".into(), dtype: MetalDtype::I64 }),
            rhs: Box::new(PredicateAst::LiteralI64(0)),
        }),
        Box::new(PredicateAst::Compare {
            op: CompareOp::Lt,
            lhs: Box::new(PredicateAst::Column { name: "b".into(), dtype: MetalDtype::I64 }),
            rhs: Box::new(PredicateAst::Column { name: "c".into(), dtype: MetalDtype::I64 }),
        }),
    );
    // Just construct it; semantic checks happen at validation time.
    let _ = MetalPlanNode::Filter {
        input: Box::new(MetalPlanNode::Scan { n_rows: 100, columns: vec![] }),
        predicate: pred,
    };
}
```

- [ ] **Step 2: Implement the plan IR**

```rust
// crates/polars-metal-core/src/plan/mod.rs

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalDtype {
    I64,
    F64,
    Bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Eq, Ne, Lt, Le, Gt, Ge,
}

#[derive(Debug, Clone)]
pub enum PredicateAst {
    Column { name: String, dtype: MetalDtype },
    LiteralI64(i64),
    LiteralF64(f64),
    LiteralBool(bool),
    Compare { op: CompareOp, lhs: Box<PredicateAst>, rhs: Box<PredicateAst> },
    And(Box<PredicateAst>, Box<PredicateAst>),
    Or(Box<PredicateAst>, Box<PredicateAst>),
}

#[derive(Debug, Clone)]
pub enum MetalPlanNode {
    Scan {
        n_rows: usize,
        columns: Vec<(String, MetalDtype)>,
    },
    Project {
        input: Box<MetalPlanNode>,
        columns: Vec<String>,
    },
    Filter {
        input: Box<MetalPlanNode>,
        predicate: PredicateAst,
    },
}
```

- [ ] **Step 3: Expose from `lib.rs`**

```rust
// crates/polars-metal-core/src/lib.rs — add
pub mod plan;
```

- [ ] **Step 4: Run the test**

Run: `cargo test -p polars-metal-core --test test_plan_ir`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/polars-metal-core/src/{plan/mod.rs,lib.rs} crates/polars-metal-core/tests/test_plan_ir.rs
git commit -m "MetalPlanNode + PredicateAst intermediate IR"
```

### Task 7: Bottom-up walker on the Python side, recognizing only DataFrameScan and SimpleProjection (filter still falls back)

This task lands the walker structural change, sufficient to take a Polars IR through `set_udf` for `df.select(...)`-style queries. Filter still falls back; the kernel layer is not yet wired in.

**Files:**
- Modify: `python/polars_metal/_callback.py`
- Create: `python/polars_metal/_walker.py`
- Create: `python/polars_metal/_udf.py`
- Create: `tests/python_integration/test_walker_select.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/python_integration/test_walker_select.py
import polars as pl
import polars_metal
from polars.testing import assert_frame_equal


def test_select_only_query_goes_through_walker():
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
    cpu = df.lazy().select(["b", "a"]).collect()
    metal = df.lazy().select(["b", "a"]).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_select_with_unsupported_dtype_falls_back_cleanly():
    df = pl.DataFrame({"a": [1, 2, 3], "s": ["x", "y", "z"]})  # string dtype = fallback
    cpu = df.lazy().select(["s", "a"]).collect()
    metal = df.lazy().select(["s", "a"]).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)
```

- [ ] **Step 2: Implement the walker**

```python
# python/polars_metal/_walker.py
"""Bottom-up walk of the Polars IR via NodeTraverser.

For each node, returns either:
- `Handled(MetalPlanNode)` — the subtree can run on GPU; carries the plan.
- `FallBack` — at least one descendant or the node itself is not supported.

The walker's policy: any FallBack poisons its parent. We never lift partial
subtrees to GPU in M1 (see spec, Section 4 "Partial dispatch policy").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Handled:
    plan: Any  # opaque MetalPlanNode-shaped dict; serialized to Rust at UDF dispatch


@dataclass
class FallBack:
    reason: str  # for debug logging


WalkResult = Handled | FallBack  # Python 3.10+ union syntax


def walk(nt: Any) -> WalkResult | FallBack:
    """Recursively walk nt's IR. Returns Handled if the whole tree fits M1's
    closed set; FallBack otherwise (with the first reason found).
    """
    node = nt.view_current_node()
    # Pattern-match by Python type name — Polars exposes IR nodes as wrapped
    # classes; cf. references/polars/py-polars/src/lazyframe/visitor/.
    cls = type(node).__name__

    if cls == "PythonScan":
        # In-memory DataFrame source (our M0 fallback also receives these).
        return _walk_scan(nt, node)
    if cls in ("SimpleProjection", "Select"):
        return _walk_project(nt, node)
    if cls == "Filter":
        # M1 Phase 7: enable. For Tasks 7..., return FallBack.
        return FallBack(reason="Filter not yet implemented in this phase")
    return FallBack(reason=f"unsupported IR node: {cls}")


def _walk_scan(nt, node) -> WalkResult | FallBack:
    # Get the materialized columns: dtype check.
    # `node.schema` returns dict of name -> Polars dtype.
    schema = dict(node.schema().items()) if hasattr(node, "schema") else {}
    columns: list[tuple[str, str]] = []
    for name, dtype in schema.items():
        m1 = _map_dtype(dtype)
        if m1 is None:
            return FallBack(reason=f"unsupported dtype {dtype} on column {name}")
        columns.append((name, m1))
    return Handled(plan={"kind": "Scan", "n_rows": _n_rows(node), "columns": columns})


def _walk_project(nt, node) -> WalkResult | FallBack:
    # Children: dispatch to the only child input.
    nt.set_node(nt.get_inputs()[0])
    input_result = walk(nt)
    if isinstance(input_result, FallBack):
        return input_result
    # node.columns or node.expressions — the exact attribute depends on the
    # Polars rev; SimpleProjection has `.columns: list[str]`.
    cols: list[str] = list(getattr(node, "columns", []))
    return Handled(plan={"kind": "Project", "input": input_result.plan, "columns": cols})


def _map_dtype(dt: Any) -> Optional[str]:
    s = str(dt)
    if s == "Int64":
        return "I64"
    if s == "Float64":
        return "F64"
    if s == "Boolean":
        return "Bool"
    return None


def _n_rows(node) -> int:
    """Best-effort row count. Returns 0 if the IR doesn't expose it pre-execution."""
    return getattr(node, "n_rows", 0)
```

- [ ] **Step 3: Implement `_udf.py` trampoline**

```python
# python/polars_metal/_udf.py
"""Polars UDF entry point. Receives a DataFrame at execution time (after
Polars has run the parts of the plan we couldn't lift), dispatches GPU
kernels per the MetalPlanNode, returns a Polars DataFrame."""
from __future__ import annotations

import polars as pl
from polars_metal import _native


def build_udf(plan: dict):
    """Build a callable suitable for `nt.set_udf(...)`. The plan is the
    serialized MetalPlanNode tree produced by _walker.walk()."""
    def udf(df: pl.DataFrame) -> pl.DataFrame:
        # M1 Phase 4: only Scan + Project. Just slice the DataFrame.
        # Real kernel dispatch arrives in Phase 5+.
        return _execute(plan, df)
    return udf


def _execute(plan: dict, df: pl.DataFrame) -> pl.DataFrame:
    kind = plan["kind"]
    if kind == "Scan":
        # No-op: df IS the scan result.
        return df
    if kind == "Project":
        upstream = _execute(plan["input"], df)
        return upstream.select(plan["columns"])
    if kind == "Filter":
        # M1 Phase 5+: dispatch GPU kernels via _native.
        raise NotImplementedError("Filter dispatch lands in Phase 5+")
    raise ValueError(f"unknown plan kind: {kind}")
```

- [ ] **Step 4: Modify `_callback.py` to call the walker and set the UDF when fully handled**

```python
# python/polars_metal/_callback.py — replace the M0 stub with:
from polars_metal._walker import walk, Handled, FallBack
from polars_metal._udf import build_udf


def execute_with_metal(nt, duration_since_start, *, config):
    if config.debug:
        log.debug("polars_metal: execute_with_metal invoked")
    result = walk(nt)
    if isinstance(result, FallBack):
        if config.debug:
            log.debug("polars_metal: falling back: %s", result.reason)
        return  # don't set_udf; Polars runs CPU
    # Handled: install our UDF.
    nt.set_udf(build_udf(result.plan))
    if config.debug:
        log.debug("polars_metal: installed UDF for plan %s", result.plan["kind"])
    return
```

- [ ] **Step 5: Rebuild the wheel**

Run: `make wheel`
Expected: builds without errors.

- [ ] **Step 6: Run the test**

Run: `pytest tests/python_integration/test_walker_select.py -v`
Expected: PASS on both tests.

- [ ] **Step 7: Run the conformance gate to confirm zero new failures**

Run: `make test-conformance`
Expected: still 721 pass / 0 fail / 1 skip (or matching M0 baseline).

- [ ] **Step 8: Commit**

```bash
git add python/polars_metal/_callback.py python/polars_metal/_walker.py python/polars_metal/_udf.py tests/python_integration/test_walker_select.py
git commit -m "Bottom-up IR walker; dispatch DataFrameScan + Project via UDF"
```

### Task 8: Surface `MetalPlanNode` to Rust via PyO3 and wire scan/project through `_native`

The walker currently builds a Python dict and the UDF re-implements the logic in Python. To exercise the buffer bridge and arena, we need the UDF to call into Rust. This task lands the Python-dict → Rust-`MetalPlanNode` serialization plus a no-op Rust executor.

**Files:**
- Modify: `crates/polars-metal-core/src/lib.rs`
- Create: `crates/polars-metal-core/src/udf.rs`
- Modify: `python/polars_metal/_udf.py`
- Create: `tests/python_integration/test_udf_dispatch.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/python_integration/test_udf_dispatch.py
import polars as pl
import polars_metal
from polars_metal import _native
from polars.testing import assert_frame_equal


def test_native_execute_plan_round_trips_scan():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]})
    plan = {"kind": "Scan", "n_rows": 3, "columns": [["a", "I64"], ["b", "F64"]]}
    # `execute_plan` accepts the Arrow record batch of df + the plan,
    # returns an Arrow record batch.
    result = _native.execute_plan(df, plan)
    assert_frame_equal(df, result)


def test_native_execute_plan_round_trips_scan_then_project():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]})
    plan = {
        "kind": "Project",
        "columns": ["b"],
        "input": {"kind": "Scan", "n_rows": 3, "columns": [["a", "I64"], ["b", "F64"]]},
    }
    result = _native.execute_plan(df, plan)
    assert_frame_equal(df.select("b"), result)
```

- [ ] **Step 2: Implement `udf::execute_plan` in Rust**

```rust
// crates/polars-metal-core/src/udf.rs
//
// PyO3 entry point: receives a Python DataFrame + a Python dict
// (serialized MetalPlanNode), returns a Python DataFrame.
//
// M1 Phase 4: handles Scan + Project only (filter & kernels arrive in Phase 5+).

use crate::plan::{MetalDtype, MetalPlanNode};
use polars_metal_buffer::MetalDevice;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[pyfunction]
pub fn execute_plan<'py>(
    py: Python<'py>,
    df_in: Bound<'py, PyAny>,  // pl.DataFrame
    plan_dict: Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let plan = deserialize_plan(&plan_dict)?;
    // M1 Phase 4: walk MetalPlanNode and produce the result by adapting the
    // input DataFrame. No GPU buffers materialized yet; scan/project are pure
    // metadata. Phase 5+ replaces this with real GPU dispatch.
    let result = execute_node(py, df_in, &plan)?;
    Ok(result)
}

fn execute_node<'py>(
    py: Python<'py>,
    df: Bound<'py, PyAny>,
    node: &MetalPlanNode,
) -> PyResult<Bound<'py, PyAny>> {
    match node {
        MetalPlanNode::Scan { .. } => Ok(df),
        MetalPlanNode::Project { input, columns } => {
            let upstream = execute_node(py, df, input)?;
            let py_cols: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
            upstream.call_method1("select", (py_cols,))
        }
        MetalPlanNode::Filter { .. } => Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Filter dispatch lands in M1 Phase 5+",
        )),
    }
}

fn deserialize_plan(dict: &Bound<PyDict>) -> PyResult<MetalPlanNode> {
    let kind: String = dict.get_item("kind")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("missing 'kind'"))?
        .extract()?;
    match kind.as_str() {
        "Scan" => {
            let n_rows: usize = dict.get_item("n_rows")?.unwrap().extract()?;
            let cols_obj = dict.get_item("columns")?.unwrap();
            let cols_list: Bound<PyList> = cols_obj.downcast_into()?;
            let mut columns = Vec::new();
            for entry in cols_list.iter() {
                let pair: (String, String) = entry.extract()?;
                let dtype = match pair.1.as_str() {
                    "I64" => MetalDtype::I64,
                    "F64" => MetalDtype::F64,
                    "Bool" => MetalDtype::Bool,
                    other => return Err(pyo3::exceptions::PyValueError::new_err(format!("bad dtype {other}"))),
                };
                columns.push((pair.0, dtype));
            }
            Ok(MetalPlanNode::Scan { n_rows, columns })
        }
        "Project" => {
            let input_dict: Bound<PyDict> = dict.get_item("input")?.unwrap().downcast_into()?;
            let input = Box::new(deserialize_plan(&input_dict)?);
            let cols: Vec<String> = dict.get_item("columns")?.unwrap().extract()?;
            Ok(MetalPlanNode::Project { input, columns: cols })
        }
        "Filter" => {
            // Stub: deserialize but error in execute_node until Phase 5+.
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Filter deserialization lands in Phase 7",
            ))
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!("unknown plan kind: {other}"))),
    }
}
```

- [ ] **Step 3: Register `execute_plan` in the pymodule**

```rust
// crates/polars-metal-core/src/lib.rs — add to polars_metal_native
#[pymodule]
fn polars_metal_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ... existing
    m.add_function(wrap_pyfunction!(udf::execute_plan, m)?)?;
    Ok(())
}

mod udf;
```

- [ ] **Step 4: Wire `_udf.py` to call into `_native.execute_plan`**

```python
# python/polars_metal/_udf.py — replace the pure-Python _execute with:
def build_udf(plan: dict):
    def udf(df: pl.DataFrame) -> pl.DataFrame:
        return _native.execute_plan(df, plan)
    return udf
```

- [ ] **Step 5: Rebuild wheel and run**

Run: `make wheel && pytest tests/python_integration/test_udf_dispatch.py -v`
Expected: PASS.

- [ ] **Step 6: Re-run conformance + select walker tests**

Run: `pytest tests/python_integration/test_walker_select.py tests/conformance -v`
Expected: PASS. Conformance count matches M0 baseline.

- [ ] **Step 7: Commit**

```bash
git add crates/polars-metal-core/src/{lib.rs,udf.rs} python/polars_metal/_udf.py tests/python_integration/test_udf_dispatch.py
git commit -m "Wire walker plan through Rust _native.execute_plan; scan/project pass through"
```

---

## Phase 5 — Compaction kernels (filter on a precomputed bool column)

This phase makes `df.filter(pl.col("mask"))` where `mask` is a precomputed Boolean column work end-to-end on GPU. Comparison and AND/OR are NOT yet supported as predicate shapes — that's Phases 6 and 7.

### Task 9: `_validity.metal` — shared MSL helpers for bit-packed validity

This is included via `#include` by every subsequent .metal file. It is NOT a standalone kernel; the build.rs already compiles every .metal file in shaders/, but headers (no kernel functions) emit empty air files which the metallib step accepts.

To distinguish from kernel files: convention is leading underscore = header. We update build.rs to skip leading-underscore files for the .air compile path, but `#include` them via -I.

**Files:**
- Create: `shaders/_validity.metal`
- Modify: `crates/polars-metal-kernels/build.rs`

- [ ] **Step 1: Write the MSL helpers**

```msl
// shaders/_validity.metal
//
// Bit-packed Arrow validity helpers (little-endian).
// Each row's validity bit lives at byte (row / 8), bit (row % 8).
// Round-up byte count for n_rows: (n_rows + 7) / 8.

#pragma once
#include <metal_stdlib>
using namespace metal;

inline bool get_valid(device const uint8_t* bitmap, uint row) {
    return (bitmap[row >> 3] >> (row & 7)) & 1u;
}

inline void set_valid_nonatomic(device uint8_t* bitmap, uint row, bool v) {
    uint byte = row >> 3;
    uint bit  = row & 7;
    if (v) {
        bitmap[byte] |= (1u << bit);
    } else {
        bitmap[byte] &= ~(1u << bit);
    }
}

// Atomically OR a single bit into a validity byte. Used in scatter where
// 8 output rows may be written by different threads to the same byte.
// The validity buffer must be 4-byte-aligned and zero-initialized before
// the dispatch.
inline void set_valid_atomic_or(device atomic_uint* atomic_words, uint row) {
    uint byte_idx = row >> 3;
    uint word_idx = byte_idx >> 2;            // 4 bytes per word
    uint bit_in_word = ((byte_idx & 3) << 3) | (row & 7);
    atomic_fetch_or_explicit(&atomic_words[word_idx], 1u << bit_in_word, memory_order_relaxed);
}
```

- [ ] **Step 2: Update `build.rs` to skip leading-underscore files at compile but pass `-I shaders` for includes**

```rust
// crates/polars-metal-kernels/build.rs — modify the compile loop:
for entry in std::fs::read_dir(&shaders_dir).expect("shaders dir exists") {
    let path = entry.expect("readable").path();
    let stem = path.file_stem().expect("has stem").to_string_lossy().to_string();
    if stem.starts_with('_') {
        continue; // header-only file; included by the kernels themselves
    }
    if path.extension().and_then(|s| s.to_str()) != Some("metal") {
        continue;
    }
    let air_path = out_dir.join(format!("{stem}.air"));
    let status = Command::new("xcrun")
        .args(["metal", "-c", "-frecord-sources", "-I"])
        .arg(&shaders_dir)
        .arg("-o")
        .arg(&air_path)
        .arg(&path)
        .status()
        .expect("xcrun metal runs");
    assert!(status.success(), "metal compile failed for {}", path.display());
    air_files.push(air_path);
}
```

- [ ] **Step 3: Run the build to confirm `_validity.metal` doesn't break the metallib**

Run: `cargo build -p polars-metal-kernels`
Expected: builds. `_validity` is skipped at compile time, but its `#include` will be available to all other kernels.

- [ ] **Step 4: Commit**

```bash
git add shaders/_validity.metal crates/polars-metal-kernels/build.rs
git commit -m "MSL validity-bitmap helpers (shaders/_validity.metal)"
```

### Task 10: `filter_predicate.metal` — bool-bit-packed + validity → dense u8

**Files:**
- Create: `shaders/filter_predicate.metal`
- Create: `crates/polars-metal-kernels/src/filter.rs`
- Modify: `crates/polars-metal-kernels/src/lib.rs`
- Create: `crates/polars-metal-kernels/tests/test_filter_predicate.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-kernels/tests/test_filter_predicate.rs
use polars_metal_kernels::filter::dispatch_predicate_to_u8;
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use proptest::prelude::*;

#[test]
fn all_true_no_nulls_outputs_all_ones() {
    let device = MetalDevice::system_default().unwrap();
    let mut queue = CommandQueue::new(&device).unwrap();
    // 16 rows, all bits set
    let data = vec![0xFFu8, 0xFF];
    let valid = vec![0xFFu8, 0xFF];
    let mut out = vec![0u8; 16];
    dispatch_predicate_to_u8(&device, &mut queue, &data, &valid, 16, &mut out).unwrap();
    assert_eq!(out, vec![1u8; 16]);
}

#[test]
fn null_rows_mask_to_zero() {
    let device = MetalDevice::system_default().unwrap();
    let mut queue = CommandQueue::new(&device).unwrap();
    let data = vec![0xFFu8];   // all 8 "true"
    let valid = vec![0b00001111u8]; // only rows 0..3 valid
    let mut out = vec![0u8; 8];
    dispatch_predicate_to_u8(&device, &mut queue, &data, &valid, 8, &mut out).unwrap();
    assert_eq!(out, vec![1, 1, 1, 1, 0, 0, 0, 0]);
}

proptest! {
    #[test]
    fn matches_reference(
        n in 1usize..1024,
        data_seed in any::<u64>(),
        valid_seed in any::<u64>(),
    ) {
        let bytes = (n + 7) / 8;
        let mut data = vec![0u8; bytes];
        let mut valid = vec![0u8; bytes];
        for r in 0..n {
            if (data_seed.rotate_left(r as u32 & 63) & 1) == 1 {
                data[r >> 3] |= 1 << (r & 7);
            }
            if (valid_seed.rotate_left(r as u32 & 63) & 1) == 1 {
                valid[r >> 3] |= 1 << (r & 7);
            }
        }
        let device = MetalDevice::system_default().unwrap();
        let mut queue = CommandQueue::new(&device).unwrap();
        let mut got = vec![0u8; n];
        dispatch_predicate_to_u8(&device, &mut queue, &data, &valid, n, &mut got).unwrap();
        for r in 0..n {
            let d_bit = (data[r >> 3] >> (r & 7)) & 1;
            let v_bit = (valid[r >> 3] >> (r & 7)) & 1;
            let expected = d_bit & v_bit;
            prop_assert_eq!(got[r], expected);
        }
    }
}
```

- [ ] **Step 2: Write the MSL kernel**

```msl
// shaders/filter_predicate.metal
#include "_validity.metal"

kernel void filter_predicate_to_u8(
    device const uint8_t* pred_data   [[buffer(0)]],
    device const uint8_t* pred_valid  [[buffer(1)]],
    device       uint8_t* keep_flags  [[buffer(2)]],
    constant     uint32_t& n_rows     [[buffer(3)]],
    uint                  gid         [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    bool d = get_valid(pred_data, gid);   // reuse bitmap helper — same shape
    bool v = get_valid(pred_valid, gid);
    keep_flags[gid] = (d && v) ? 1u : 0u;
}
```

- [ ] **Step 3: Implement the Rust dispatcher**

```rust
// crates/polars-metal-kernels/src/filter.rs
use crate::command::CommandQueue;
use crate::shader_lib::shared_library;
use polars_metal_buffer::{MetalBuffer, MetalDevice};

#[derive(Debug, thiserror::Error)]
pub enum FilterError {
    #[error("shader library: {0}")]
    Shader(#[from] crate::shader_lib::ShaderError),
    #[error("dispatch: {0}")]
    Dispatch(#[from] crate::command::DispatchError),
    #[error("buffer: {0}")]
    Buffer(#[from] polars_metal_buffer::BufferError),
}

pub fn dispatch_predicate_to_u8(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    pred_data: &[u8],
    pred_valid: &[u8],
    n_rows: usize,
    out: &mut [u8],
) -> Result<(), FilterError> {
    let lib = shared_library(device)?;
    let pso = lib.pipeline("filter_predicate_to_u8")?;
    // Copy input slices into Metal buffers (TODO Phase later: take MetalBuffers directly).
    let in_data = device.new_buffer_from_bytes(pred_data)?;
    let in_valid = device.new_buffer_from_bytes(pred_valid)?;
    let out_buf = device.new_buffer_zeroed(n_rows)?;
    let n: u32 = n_rows as u32;
    let n_buf = device.new_buffer_from_bytes(&n.to_le_bytes())?;
    queue.dispatch_1d(&pso, &[&in_data, &in_valid, &out_buf, &n_buf], n_rows)?;
    queue.wait_until_complete()?;
    out.copy_from_slice(&out_buf.as_slice()[..n_rows]);
    Ok(())
}
```

- [ ] **Step 4: Implement `MetalDevice::new_buffer_from_bytes`**

Add to `crates/polars-metal-buffer/src/device.rs`. Copy-path only (uses `newBufferWithBytes:length:options:`).

- [ ] **Step 5: Run the test**

Run: `cargo test -p polars-metal-kernels --test test_filter_predicate`
Expected: PASS, all three (two explicit + one proptest with 256 cases).

- [ ] **Step 6: Commit**

```bash
git add shaders/filter_predicate.metal crates/polars-metal-kernels/src/{filter.rs,lib.rs} crates/polars-metal-kernels/tests/test_filter_predicate.rs crates/polars-metal-buffer/src/device.rs
git commit -m "Kernel: filter_predicate_to_u8 (bit-packed bool + validity → dense u8)"
```

### Task 11: `filter_scatter_i64.metal` — atomic-validity scatter

**Files:**
- Create: `shaders/filter_scatter.metal` (one file, multiple entry points for the data-type variants per the spec; convention "one MSL kernel family per file")
- Modify: `crates/polars-metal-kernels/src/filter.rs`
- Create: `crates/polars-metal-kernels/tests/test_filter_scatter.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-kernels/tests/test_filter_scatter.rs
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::filter::dispatch_scatter_i64;
use polars_metal_buffer::MetalDevice;
use proptest::prelude::*;

fn cpu_compact_i64(src: &[i64], src_valid: &[u8], keep: &[u8]) -> (Vec<i64>, Vec<u8>) {
    let mut data = Vec::new();
    let mut valid_bits = Vec::new();
    for (i, &k) in keep.iter().enumerate() {
        if k == 1 {
            data.push(src[i]);
            valid_bits.push(((src_valid[i >> 3] >> (i & 7)) & 1) == 1);
        }
    }
    let n_out = data.len();
    let mut valid = vec![0u8; (n_out + 7) / 8];
    for (i, b) in valid_bits.iter().enumerate() {
        if *b {
            valid[i >> 3] |= 1 << (i & 7);
        }
    }
    (data, valid)
}

#[test]
fn alternating_keep_compacts_correctly() {
    let device = MetalDevice::system_default().unwrap();
    let mut queue = CommandQueue::new(&device).unwrap();
    let src: Vec<i64> = (0..16).collect();
    let src_valid = vec![0xFFu8, 0xFFu8];
    let keep = vec![1u8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
    // Compute prefix sum (inclusive) on CPU for the test:
    let mut prefix = vec![0u32; 16];
    let mut acc = 0u32;
    for (i, &k) in keep.iter().enumerate() {
        acc += k as u32;
        prefix[i] = acc;
    }
    let n_out = acc as usize;
    let mut dst = vec![0i64; n_out + 1]; // +1 for sentinel
    let mut dst_valid = vec![0u8; (n_out + 7) / 8 + 4]; // round up + atomic alignment
    dispatch_scatter_i64(&device, &mut queue, &src, &src_valid, &keep, &prefix, n_out, &mut dst, &mut dst_valid).unwrap();
    let (exp_data, exp_valid) = cpu_compact_i64(&src, &src_valid, &keep);
    assert_eq!(&dst[..n_out], &exp_data[..]);
    for r in 0..n_out {
        let got = (dst_valid[r >> 3] >> (r & 7)) & 1;
        let exp = (exp_valid[r >> 3] >> (r & 7)) & 1;
        assert_eq!(got, exp, "row {r}");
    }
}

proptest! {
    #[test]
    fn matches_cpu_reference(n in 8usize..256, seed in any::<u64>()) {
        let src: Vec<i64> = (0..n as i64).collect();
        let mut src_valid = vec![0u8; (n + 7) / 8];
        let mut keep = vec![0u8; n];
        for r in 0..n {
            if (seed.rotate_left(r as u32 & 63) & 1) == 1 {
                src_valid[r >> 3] |= 1 << (r & 7);
            }
            if (seed.rotate_left((r as u32 * 7) & 63) & 1) == 1 {
                keep[r] = 1;
            }
        }
        let mut prefix = vec![0u32; n];
        let mut acc = 0u32;
        for (i, &k) in keep.iter().enumerate() {
            acc += k as u32;
            prefix[i] = acc;
        }
        let n_out = acc as usize;
        if n_out == 0 { return Ok(()); }
        let mut dst = vec![0i64; n_out + 1];
        let mut dst_valid = vec![0u8; ((n_out + 7) / 8 + 4) & !3];
        let device = MetalDevice::system_default().unwrap();
        let mut queue = CommandQueue::new(&device).unwrap();
        dispatch_scatter_i64(&device, &mut queue, &src, &src_valid, &keep, &prefix, n_out, &mut dst, &mut dst_valid).unwrap();
        let (exp_data, exp_valid) = cpu_compact_i64(&src, &src_valid, &keep);
        prop_assert_eq!(&dst[..n_out], &exp_data[..]);
        for r in 0..n_out {
            let got = (dst_valid[r >> 3] >> (r & 7)) & 1;
            let exp = (exp_valid[r >> 3] >> (r & 7)) & 1;
            prop_assert_eq!(got, exp);
        }
    }
}
```

- [ ] **Step 2: Write the MSL kernel**

```msl
// shaders/filter_scatter.metal
#include "_validity.metal"

kernel void filter_scatter_i64(
    device const int64_t*  src_data       [[buffer(0)]],
    device const uint8_t*  src_valid      [[buffer(1)]],
    device const uint8_t*  keep           [[buffer(2)]],
    device const uint32_t* prefix_sum     [[buffer(3)]],
    device       int64_t*  dst_data       [[buffer(4)]],
    device       atomic_uint* dst_valid_atomic [[buffer(5)]],
    constant     uint32_t& n_rows         [[buffer(6)]],
    constant     uint32_t& n_out          [[buffer(7)]],
    uint                   gid            [[thread_position_in_grid]])
{
    if (gid >= n_rows || keep[gid] == 0) return;
    uint32_t out_idx = prefix_sum[gid] - 1u;
    // Sentinel-overrun check (release-safety, see spec §"Scatter overrun sentinel"):
    if (out_idx >= n_out) {
        dst_data[n_out] = (int64_t)0xDEADBEEFCAFEBABEll;  // recognizable sentinel
        return;
    }
    dst_data[out_idx] = src_data[gid];
    if (get_valid(src_valid, gid)) {
        set_valid_atomic_or(dst_valid_atomic, out_idx);
    }
}
```

- [ ] **Step 3: Implement `dispatch_scatter_i64`**

Analogous to `dispatch_predicate_to_u8` from Task 10. Bind buffers 0..7, dispatch n_rows threads. After `wait_until_complete`, check `dst_data[n_out]` for the sentinel value — if present, return `FilterError::ScatterOverrun`.

- [ ] **Step 4: Run the test**

Run: `cargo test -p polars-metal-kernels --test test_filter_scatter`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add shaders/filter_scatter.metal crates/polars-metal-kernels/src/filter.rs crates/polars-metal-kernels/tests/test_filter_scatter.rs
git commit -m "Kernel: filter_scatter_i64 with atomic validity-bit writes + overrun sentinel"
```

### Task 12: `filter_scatter_f64` — mirror of i64

**Files:**
- Modify: `shaders/filter_scatter.metal` (add `filter_scatter_f64` entry point)
- Modify: `crates/polars-metal-kernels/src/filter.rs` (add `dispatch_scatter_f64`)
- Create: `crates/polars-metal-kernels/tests/test_filter_scatter_f64.rs`

- [ ] **Step 1: Write the failing test**

Mirror `test_filter_scatter.rs` from Task 11. Replace `i64` with `f64`. Include a case with NaN in the source data — NaN should round-trip as bit-identical NaN (per `f64::to_bits` / `from_bits`), with its validity bit preserved per source.

- [ ] **Step 2: Write the MSL kernel**

Same body as `filter_scatter_i64`, with `int64_t` → `float`, and the sentinel changed to `__builtin_nanf("0xCAFE")` (or just `1e308` — a recognizable non-NaN sentinel value).

Wait — sentinel value choice matters. If the source data legitimately contains the sentinel, we false-trigger. Use a distinct `NaN` payload: `NaN` with mantissa pattern `0x7FF_DEAD_BEEF_CAFE`. CPU-side check: `dst_data[n_out].to_bits() == that pattern`.

- [ ] **Step 3: Implement `dispatch_scatter_f64`**

Analogous to i64.

- [ ] **Step 4: Run the test**

Run: `cargo test -p polars-metal-kernels --test test_filter_scatter_f64`
Expected: PASS, including the NaN case.

- [ ] **Step 5: Commit**

```bash
git add shaders/filter_scatter.metal crates/polars-metal-kernels/src/filter.rs crates/polars-metal-kernels/tests/test_filter_scatter_f64.rs
git commit -m "Kernel: filter_scatter_f64 with NaN-preserving payload + distinct sentinel"
```

### Task 13: `filter_scatter_bool` — atomics for BOTH data and validity

This is the kernel where the data output is also bit-packed, so the data write needs the same atomic OR pattern as validity.

**Files:**
- Modify: `shaders/filter_scatter.metal` (add `filter_scatter_bool` entry point)
- Modify: `crates/polars-metal-kernels/src/filter.rs` (add `dispatch_scatter_bool`)
- Create: `crates/polars-metal-kernels/tests/test_filter_scatter_bool.rs`

- [ ] **Step 1: Write the failing test, specifically engineered to expose multi-thread same-byte races**

```rust
#[test]
fn keep_pattern_forces_atomic_data_writes_into_same_byte() {
    // 8 source rows, all kept, all bool=1. All 8 outputs land in dst_data[0..1] (one byte).
    let device = MetalDevice::system_default().unwrap();
    let mut queue = CommandQueue::new(&device).unwrap();
    let src_data = vec![0xFFu8];           // 8 rows of true
    let src_valid = vec![0xFFu8];
    let keep = vec![1u8; 8];
    let prefix: Vec<u32> = (1..=8).collect();
    let mut dst_data = vec![0u8; 1 + 4];   // 1 output byte + 4 sentinel alignment
    let mut dst_valid = vec![0u8; 4];      // 4-byte aligned for atomic ops
    dispatch_scatter_bool(&device, &mut queue, &src_data, &src_valid, &keep, &prefix, 8, &mut dst_data, &mut dst_valid).unwrap();
    // All 8 bits set.
    assert_eq!(dst_data[0], 0xFFu8);
    assert_eq!(dst_valid[0], 0xFFu8);
}
```

Plus a proptest with the same `cpu_compact_bool` reference style as i64/f64.

- [ ] **Step 2: Write the MSL kernel**

```msl
kernel void filter_scatter_bool(
    device const uint8_t*  src_data           [[buffer(0)]],  // bit-packed
    device const uint8_t*  src_valid          [[buffer(1)]],  // bit-packed
    device const uint8_t*  keep               [[buffer(2)]],
    device const uint32_t* prefix_sum         [[buffer(3)]],
    device       atomic_uint* dst_data_atomic [[buffer(4)]],
    device       atomic_uint* dst_valid_atomic [[buffer(5)]],
    constant     uint32_t& n_rows             [[buffer(6)]],
    constant     uint32_t& n_out              [[buffer(7)]],
    uint                   gid                [[thread_position_in_grid]])
{
    if (gid >= n_rows || keep[gid] == 0) return;
    uint32_t out_idx = prefix_sum[gid] - 1u;
    if (out_idx >= n_out) return;  // overrun: silently drop; sentinel for bool uses dst_valid[n_out] byte
    if (get_valid(src_data, gid)) {
        set_valid_atomic_or(dst_data_atomic, out_idx);
    }
    if (get_valid(src_valid, gid)) {
        set_valid_atomic_or(dst_valid_atomic, out_idx);
    }
}
```

The "sentinel" pattern is harder for bit-packed bool. For M1: bool scatter does NOT use a sentinel slot (the data buffer has no spare value, since every bit is 0 or 1 and both are legal). The overrun check is the prefix-sum invariant — `prefix_sum[n_rows - 1] == n_out` — verified CPU-side post-cumsum before the scatter. If `out_idx >= n_out` is reached, that's a kernel-logic bug, not a runtime corruption; in debug builds we panic, in release we early-return as above.

- [ ] **Step 3: Implement `dispatch_scatter_bool`**

Analogous to i64/f64. Skip the post-dispatch sentinel check (per above).

- [ ] **Step 4: Run the test**

Run: `cargo test -p polars-metal-kernels --test test_filter_scatter_bool`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add shaders/filter_scatter.metal crates/polars-metal-kernels/src/filter.rs crates/polars-metal-kernels/tests/test_filter_scatter_bool.rs
git commit -m "Kernel: filter_scatter_bool with atomic OR on both data and validity"
```

### Task 14: Compaction pipeline — combine predicate kernel, MLX cumsum, and scatter

**Files:**
- Create: `crates/polars-metal-kernels/src/pipeline.rs`
- Modify: `crates/polars-metal-kernels/src/lib.rs`
- Create: `crates/polars-metal-kernels/tests/test_compaction_pipeline.rs`

- [ ] **Step 1: Write the failing test**

```rust
// End-to-end: take a bool column + validity + an i64 column, run the three-pass
// compaction, assert output equals CPU reference.
use polars_metal_kernels::pipeline::compact_i64;
// ... see test pattern from Task 11.
```

The test should use a 10K-row column to ensure threadgroup boundaries are exercised in cumsum.

- [ ] **Step 2: Implement `pipeline::compact_*`**

```rust
// crates/polars-metal-kernels/src/pipeline.rs

use crate::command::CommandQueue;
use crate::filter::{dispatch_predicate_to_u8, dispatch_scatter_i64, dispatch_scatter_f64, dispatch_scatter_bool};
use polars_metal_buffer::MetalDevice;
use polars_metal_mlx_sys::cumsum_u8_to_u32;

pub struct CompactionResult<T> {
    pub data: Vec<T>,
    pub valid: Vec<u8>,
    pub n_out: usize,
}

pub fn compact_i64(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    src_data: &[i64],
    src_valid: &[u8],
    pred_data: &[u8],
    pred_valid: &[u8],
) -> Result<CompactionResult<i64>, crate::filter::FilterError> {
    let n_rows = src_data.len();
    let mut keep = vec![0u8; n_rows];
    dispatch_predicate_to_u8(device, queue, pred_data, pred_valid, n_rows, &mut keep)?;
    let mut prefix = vec![0u32; n_rows];
    cumsum_u8_to_u32(&keep, &mut prefix).map_err(|_| crate::filter::FilterError::Dispatch(crate::command::DispatchError::GpuError("cumsum".into())))?;
    let n_out = if n_rows == 0 { 0 } else { prefix[n_rows - 1] as usize };
    let mut dst_data = vec![0i64; n_out + 1];   // sentinel slot
    let mut dst_valid = vec![0u8; ((n_out + 7) / 8 + 4) & !3];  // 4-byte aligned
    dispatch_scatter_i64(device, queue, src_data, src_valid, &keep, &prefix, n_out, &mut dst_data, &mut dst_valid)?;
    dst_data.truncate(n_out);
    Ok(CompactionResult { data: dst_data, valid: dst_valid, n_out })
}

// Mirror compact_f64, compact_bool (bool's data is bit-packed input + output).
```

- [ ] **Step 3: Run the test**

Run: `cargo test -p polars-metal-kernels --test test_compaction_pipeline`
Expected: PASS, all sizes.

- [ ] **Step 4: Commit**

```bash
git add crates/polars-metal-kernels/src/{pipeline.rs,lib.rs} crates/polars-metal-kernels/tests/test_compaction_pipeline.rs
git commit -m "Compaction pipeline: predicate + cumsum + scatter for i64/f64/bool"
```

### Task 15: Filter handler in the walker (Column(bool) predicates only)

**Files:**
- Modify: `python/polars_metal/_walker.py`
- Modify: `crates/polars-metal-core/src/udf.rs`
- Create: `tests/python_integration/test_filter_bool_column.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/python_integration/test_filter_bool_column.py
import polars as pl
import polars_metal
from polars.testing import assert_frame_equal


def test_filter_with_precomputed_bool_column_runs_on_gpu():
    df = pl.DataFrame({
        "a": [1, 2, 3, 4, 5, None, 7, 8],
        "mask": [True, False, True, False, True, None, True, False],
    })
    cpu = df.lazy().filter(pl.col("mask")).select("a").collect()
    metal = df.lazy().filter(pl.col("mask")).select("a").collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_with_arithmetic_predicate_falls_back_in_phase_5():
    df = pl.DataFrame({"a": [1, 2, 3]})
    # `pl.col("a") > 0` is not yet accepted in Phase 5 (comparison kernels arrive in Phase 6)
    cpu = df.lazy().filter(pl.col("a") > 0).collect()
    metal = df.lazy().filter(pl.col("a") > 0).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)  # still produces a correct result via fallback
```

- [ ] **Step 2: Extend `_walker.py` to accept Filter when the predicate is a single `Column(Boolean)`**

```python
def _walk_filter(nt, node):
    pred = node.predicate  # or node.expression — depends on Polars rev
    # Phase 5: only accept `Column(bool)` predicates.
    if _is_column_bool(pred):
        col_name = pred.name
        nt.set_node(nt.get_inputs()[0])
        input_result = walk(nt)
        if isinstance(input_result, FallBack):
            return input_result
        return Handled(plan={
            "kind": "Filter",
            "input": input_result.plan,
            "predicate": {"kind": "Column", "name": col_name, "dtype": "Bool"},
        })
    return FallBack(reason="Filter predicate is not a single Boolean column (Phase 6+ needed)")


def _is_column_bool(expr) -> bool:
    cls = type(expr).__name__
    if cls != "Column":
        return False
    dt = expr.dtype if hasattr(expr, "dtype") else None
    return _map_dtype(dt) == "Bool"
```

Update `walk()` to call `_walk_filter` for Filter nodes.

- [ ] **Step 3: Extend Rust-side `deserialize_plan` and `execute_node` to handle Filter**

```rust
// In crates/polars-metal-core/src/udf.rs:
// - deserialize_plan: add "Filter" case, parse the predicate dict into PredicateAst.
// - execute_node for MetalPlanNode::Filter:
//   1. Resolve `input` to a DataFrame.
//   2. Resolve `predicate` Column(name) → df.get_column(name) Arrow buffers.
//   3. Wrap source columns + predicate column as MetalBuffers via the buffer bridge.
//   4. Call polars_metal_kernels::pipeline::compact_<dtype> for each surviving column.
//   5. Assemble a new pl.DataFrame from the compaction results.
```

This is the most substantial Rust task in the plan; estimate ~150 LoC. Key concerns:
- Iterating df columns: `df.iter_columns()` returns Arrow ChunkedArrays. Get the first chunk; assert single-chunk (M1 restriction — return `EngineError::MultiChunkUnsupported` if not).
- The predicate column's bit-packed data is in Arrow's Boolean array `values: Buffer`.
- Validity is `nulls: Option<NullBuffer>` → `Buffer`.
- For each surviving column, call `compact_i64`/`compact_f64`/`compact_bool` and rebuild the Arrow Buffer + Series.
- Reassemble into `pl.DataFrame` via the PyO3 Polars FFI.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/python_integration/test_filter_bool_column.py -v && make test-kernel`
Expected: PASS.

- [ ] **Step 5: Run conformance to confirm no regressions**

Run: `make test-conformance`
Expected: still matches M0 baseline (filter tests still fall back for non-bool-column predicates).

- [ ] **Step 6: Commit**

```bash
git add python/polars_metal/_walker.py crates/polars-metal-core/src/udf.rs tests/python_integration/test_filter_bool_column.py
git commit -m "Filter handler: precomputed bool-column predicates dispatch through GPU compaction"
```

---

## Phase 6 — Comparison kernels (cmp_i64, cmp_f64)

### Task 16: `cmp_i64.metal` — six comparison ops + scalar variants

**Files:**
- Create: `shaders/cmp_i64.metal`
- Create: `crates/polars-metal-kernels/src/cmp.rs`
- Modify: `crates/polars-metal-kernels/src/lib.rs`
- Create: `crates/polars-metal-kernels/tests/test_cmp_i64.rs`

- [ ] **Step 1: Write the failing test**

```rust
// crates/polars-metal-kernels/tests/test_cmp_i64.rs
use polars_metal_kernels::cmp::{CompareOp, dispatch_cmp_i64, dispatch_cmp_i64_scalar};
use polars_metal_kernels::command::CommandQueue;
use polars_metal_buffer::MetalDevice;
use proptest::prelude::*;

fn cpu_cmp_i64(lhs: &[i64], lhs_v: &[u8], rhs: &[i64], rhs_v: &[u8], op: CompareOp) -> (Vec<u8>, Vec<u8>) {
    let n = lhs.len();
    let mut out_data = vec![0u8; (n + 7) / 8];
    let mut out_valid = vec![0u8; (n + 7) / 8];
    for i in 0..n {
        let lv = (lhs_v[i >> 3] >> (i & 7)) & 1 == 1;
        let rv = (rhs_v[i >> 3] >> (i & 7)) & 1 == 1;
        if lv && rv {
            let r = match op {
                CompareOp::Eq => lhs[i] == rhs[i],
                CompareOp::Ne => lhs[i] != rhs[i],
                CompareOp::Lt => lhs[i] < rhs[i],
                CompareOp::Le => lhs[i] <= rhs[i],
                CompareOp::Gt => lhs[i] > rhs[i],
                CompareOp::Ge => lhs[i] >= rhs[i],
            };
            if r { out_data[i >> 3] |= 1 << (i & 7); }
            out_valid[i >> 3] |= 1 << (i & 7);
        }
    }
    (out_data, out_valid)
}

proptest! {
    #[test]
    fn cmp_i64_matches_cpu(
        n in 8usize..256,
        lhs_seed in any::<u64>(),
        rhs_seed in any::<u64>(),
        op_idx in 0u8..6,
    ) {
        let op = match op_idx {
            0 => CompareOp::Eq, 1 => CompareOp::Ne, 2 => CompareOp::Lt,
            3 => CompareOp::Le, 4 => CompareOp::Gt, _ => CompareOp::Ge,
        };
        let lhs: Vec<i64> = (0..n).map(|i| (lhs_seed.rotate_left(i as u32) as i64) % 100).collect();
        let rhs: Vec<i64> = (0..n).map(|i| (rhs_seed.rotate_left(i as u32) as i64) % 100).collect();
        let mut lhs_v = vec![0u8; (n + 7) / 8];
        let mut rhs_v = vec![0u8; (n + 7) / 8];
        for i in 0..n {
            if (lhs_seed.rotate_left((i as u32 ^ 7)) & 1) == 1 {
                lhs_v[i >> 3] |= 1 << (i & 7);
            }
            if (rhs_seed.rotate_left((i as u32 ^ 13)) & 1) == 1 {
                rhs_v[i >> 3] |= 1 << (i & 7);
            }
        }
        let device = MetalDevice::system_default().unwrap();
        let mut queue = CommandQueue::new(&device).unwrap();
        let mut got_data = vec![0u8; (n + 7) / 8];
        let mut got_valid = vec![0u8; (n + 7) / 8];
        dispatch_cmp_i64(&device, &mut queue, &lhs, &lhs_v, &rhs, &rhs_v, n, op, &mut got_data, &mut got_valid).unwrap();
        let (exp_data, exp_valid) = cpu_cmp_i64(&lhs, &lhs_v, &rhs, &rhs_v, op);
        // Compare only bits within n (trailing bits are unspecified).
        for i in 0..n {
            let g = (got_data[i >> 3] >> (i & 7)) & 1;
            let e = (exp_data[i >> 3] >> (i & 7)) & 1;
            let gv = (got_valid[i >> 3] >> (i & 7)) & 1;
            let ev = (exp_valid[i >> 3] >> (i & 7)) & 1;
            prop_assert_eq!(gv, ev, "validity mismatch at {i}");
            if ev == 1 {
                prop_assert_eq!(g, e, "data mismatch at {i}");
            }
        }
    }
}

#[test]
fn cmp_i64_lt_scalar_matches_cpu() {
    // ... mirror for the scalar form.
}
```

- [ ] **Step 2: Write the MSL kernel**

```msl
// shaders/cmp_i64.metal
#include "_validity.metal"

// Define one templated body, six entry points via macros.
#define CMP_KERNEL(name, op) \
kernel void name( \
    device const int64_t* lhs_data  [[buffer(0)]], \
    device const uint8_t* lhs_valid [[buffer(1)]], \
    device const int64_t* rhs_data  [[buffer(2)]], \
    device const uint8_t* rhs_valid [[buffer(3)]], \
    device       uint8_t* out_data  [[buffer(4)]], \
    device       uint8_t* out_valid [[buffer(5)]], \
    constant     uint32_t& n_rows   [[buffer(6)]], \
    uint                  gid       [[thread_position_in_grid]]) \
{ \
    if (gid >= n_rows) return; \
    bool lv = get_valid(lhs_valid, gid); \
    bool rv = get_valid(rhs_valid, gid); \
    bool valid = lv && rv; \
    bool result = valid && (lhs_data[gid] op rhs_data[gid]); \
    /* Note: each thread writes its own row's bit. We do NOT atomic here */ \
    /* because each thread owns its own bit position. But threads writing */ \
    /* to the same byte race. So we DO need atomic OR on the data and */ \
    /* validity. Zero-init out_data and out_valid before dispatch.        */ \
    /* (Same shape as filter_scatter_bool.)                                */ \
    threadgroup_barrier(mem_flags::mem_device); \
    if (valid) { \
        set_valid_atomic_or((device atomic_uint*)out_valid, gid); \
    } \
    if (result) { \
        set_valid_atomic_or((device atomic_uint*)out_data, gid); \
    } \
}

CMP_KERNEL(cmp_i64_eq, ==)
CMP_KERNEL(cmp_i64_ne, !=)
CMP_KERNEL(cmp_i64_lt, <)
CMP_KERNEL(cmp_i64_le, <=)
CMP_KERNEL(cmp_i64_gt, >)
CMP_KERNEL(cmp_i64_ge, >=)

// Scalar variants: rhs is a 1-element buffer broadcast.
#define CMP_KERNEL_SCALAR(name, op) \
kernel void name( \
    device const int64_t* lhs_data    [[buffer(0)]], \
    device const uint8_t* lhs_valid   [[buffer(1)]], \
    constant     int64_t& rhs_scalar  [[buffer(2)]], \
    device       uint8_t* out_data    [[buffer(3)]], \
    device       uint8_t* out_valid   [[buffer(4)]], \
    constant     uint32_t& n_rows     [[buffer(5)]], \
    uint                  gid         [[thread_position_in_grid]]) \
{ \
    if (gid >= n_rows) return; \
    bool lv = get_valid(lhs_valid, gid); \
    if (lv) { \
        set_valid_atomic_or((device atomic_uint*)out_valid, gid); \
        if (lhs_data[gid] op rhs_scalar) { \
            set_valid_atomic_or((device atomic_uint*)out_data, gid); \
        } \
    } \
}

CMP_KERNEL_SCALAR(cmp_i64_eq_scalar, ==)
CMP_KERNEL_SCALAR(cmp_i64_ne_scalar, !=)
CMP_KERNEL_SCALAR(cmp_i64_lt_scalar, <)
CMP_KERNEL_SCALAR(cmp_i64_le_scalar, <=)
CMP_KERNEL_SCALAR(cmp_i64_gt_scalar, >)
CMP_KERNEL_SCALAR(cmp_i64_ge_scalar, >=)
```

- [ ] **Step 3: Implement `cmp::dispatch_cmp_i64` and `cmp::dispatch_cmp_i64_scalar`**

```rust
// crates/polars-metal-kernels/src/cmp.rs

use crate::command::CommandQueue;
use crate::shader_lib::shared_library;
use polars_metal_buffer::MetalDevice;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp { Eq, Ne, Lt, Le, Gt, Ge }

impl CompareOp {
    fn entry_point_i64(self) -> &'static str {
        match self {
            CompareOp::Eq => "cmp_i64_eq", CompareOp::Ne => "cmp_i64_ne",
            CompareOp::Lt => "cmp_i64_lt", CompareOp::Le => "cmp_i64_le",
            CompareOp::Gt => "cmp_i64_gt", CompareOp::Ge => "cmp_i64_ge",
        }
    }
    fn entry_point_i64_scalar(self) -> &'static str {
        match self {
            CompareOp::Eq => "cmp_i64_eq_scalar", CompareOp::Ne => "cmp_i64_ne_scalar",
            CompareOp::Lt => "cmp_i64_lt_scalar", CompareOp::Le => "cmp_i64_le_scalar",
            CompareOp::Gt => "cmp_i64_gt_scalar", CompareOp::Ge => "cmp_i64_ge_scalar",
        }
    }
}

pub fn dispatch_cmp_i64(/* ... see Task 16 test signature */) -> Result<(), crate::filter::FilterError> {
    // Buffers: lhs_data, lhs_valid, rhs_data, rhs_valid, out_data (zero-init), out_valid (zero-init), n_rows.
    // Dispatch n_rows threads.
    todo!()
}

pub fn dispatch_cmp_i64_scalar(/* ... */) -> Result<(), crate::filter::FilterError> {
    todo!()
}
```

Implement the dispatchers; they share the binding pattern from `dispatch_predicate_to_u8`. Out buffers must be zero-initialized (the atomic OR semantics require it).

- [ ] **Step 4: Run the test**

Run: `cargo test -p polars-metal-kernels --test test_cmp_i64`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add shaders/cmp_i64.metal crates/polars-metal-kernels/src/{cmp.rs,lib.rs} crates/polars-metal-kernels/tests/test_cmp_i64.rs
git commit -m "Kernel: cmp_i64 — six ops + scalar variants, null-aware"
```

### Task 17: `cmp_f64.metal` — mirror with Polars-conformant NaN semantics

**Files:**
- Create: `shaders/cmp_f64.metal`
- Modify: `crates/polars-metal-kernels/src/cmp.rs`
- Create: `crates/polars-metal-kernels/tests/test_cmp_f64.rs`

- [ ] **Step 1: Write the failing test**

```rust
// Per spec, Polars NaN semantics:
//   NaN < x, NaN <= x, NaN > x, NaN >= x, NaN == x  → false
//   NaN != x                                         → true
// And NaN's validity bit is still set (NaN is a value, not a null).

#[test]
fn nan_vs_value_comparisons_match_polars() {
    let device = MetalDevice::system_default().unwrap();
    let mut queue = CommandQueue::new(&device).unwrap();
    let lhs: Vec<f64> = vec![f64::NAN, 1.0, 2.0, f64::NAN];
    let rhs: Vec<f64> = vec![1.0, f64::NAN, 2.0, f64::NAN];
    let lhs_v = vec![0x0Fu8];
    let rhs_v = vec![0x0Fu8];
    // Test all 6 ops; verify NaN-involving rows produce the documented result.
    // ...
}

proptest! {
    #[test]
    fn cmp_f64_matches_cpu_with_nan_injection(
        n in 8usize..256,
        seed in any::<u64>(),
    ) {
        // 30% NaN density. Reference: pure-Rust matching Polars rules.
        // ...
    }
}
```

- [ ] **Step 2: Write the MSL kernel**

Same macro pattern as `cmp_i64.metal`, with `int64_t` → `float`. Crucial: Metal's `<`, `>`, `==`, `!=` on `float` already follow IEEE 754 — which means `NaN < x` returns `false`, `NaN == x` returns `false`, `NaN != x` returns `true`. So the macro body is identical; we don't need special NaN handling in MSL.

Validity is set as-input — NaN has its bit set in the input, and we propagate it via `lv && rv`. The result of a comparison involving a NaN-with-valid-bit-set is `(true validity, false data)` for `<,<=,>,>=,==` and `(true validity, true data)` for `!=`. The IEEE semantics give us this for free.

- [ ] **Step 3: Implement `dispatch_cmp_f64` and `dispatch_cmp_f64_scalar`**

Analogous to i64.

- [ ] **Step 4: Run the test**

Run: `cargo test -p polars-metal-kernels --test test_cmp_f64`
Expected: PASS, including NaN cases.

- [ ] **Step 5: Commit**

```bash
git add shaders/cmp_f64.metal crates/polars-metal-kernels/src/cmp.rs crates/polars-metal-kernels/tests/test_cmp_f64.rs
git commit -m "Kernel: cmp_f64 — six ops + scalar variants, Polars NaN semantics"
```

### Task 18: Walker + UDF accept comparison predicates; end-to-end test

**Files:**
- Modify: `python/polars_metal/_walker.py`
- Modify: `crates/polars-metal-core/src/udf.rs`
- Create: `tests/python_integration/test_filter_comparison.py`

- [ ] **Step 1: Write the failing test**

```python
import polars as pl
import polars_metal
from polars.testing import assert_frame_equal


def test_filter_col_gt_scalar_runs_on_gpu():
    df = pl.DataFrame({"a": [-1, 0, 1, 2, None, 4], "b": [10, 20, 30, 40, 50, 60]})
    cpu = df.lazy().filter(pl.col("a") > 0).collect()
    metal = df.lazy().filter(pl.col("a") > 0).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_col_lt_col_runs_on_gpu():
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
    cpu = df.lazy().filter(pl.col("a") < pl.col("b")).collect()
    metal = df.lazy().filter(pl.col("a") < pl.col("b")).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_with_nan_f64():
    df = pl.DataFrame({"x": [1.0, float("nan"), 3.0, float("nan")]})
    cpu = df.lazy().filter(pl.col("x") > 0).collect()
    metal = df.lazy().filter(pl.col("x") > 0).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)  # NaN is value, not null; > 0 is false
```

- [ ] **Step 2: Extend `_walker.py` predicate validator**

Replace the Phase-5 single-Column-bool check with the closed-set rule from the spec (Section 1):
- `Column(c)` where c is i64/f64/Boolean
- `Literal(scalar)` of i64/f64/Boolean
- `BinaryExpr(==,!=,<,<=,>,>=)` numeric leaves matching dtypes
- (AND/OR arrives in Phase 7)

```python
def _walk_predicate(expr) -> dict | None:
    """Returns a serialized predicate dict, or None if the shape is rejected."""
    cls = type(expr).__name__
    if cls == "Column":
        dt = _map_dtype(expr.dtype)
        if dt is None: return None
        return {"kind": "Column", "name": expr.name, "dtype": dt}
    if cls == "Literal":
        # Polars wraps Python literals. Inspect expr.value type.
        v = expr.value
        if isinstance(v, bool):
            return {"kind": "LiteralBool", "value": v}
        if isinstance(v, int):
            return {"kind": "LiteralI64", "value": int(v)}
        if isinstance(v, float):
            return {"kind": "LiteralF64", "value": float(v)}
        return None
    if cls == "BinaryExpr":
        op_name = type(expr.op).__name__   # e.g. 'Gt', 'Lt', ...
        cmp_ops = {"Eq", "NotEq", "Lt", "LtEq", "Gt", "GtEq"}
        if op_name in cmp_ops:
            lhs = _walk_predicate(expr.left)
            rhs = _walk_predicate(expr.right)
            if lhs is None or rhs is None: return None
            if _leaf_dtype(lhs) != _leaf_dtype(rhs): return None  # cross-type cmp not in M1
            op_map = {"Eq":"Eq","NotEq":"Ne","Lt":"Lt","LtEq":"Le","Gt":"Gt","GtEq":"Ge"}
            return {"kind": "Compare", "op": op_map[op_name], "lhs": lhs, "rhs": rhs}
    return None


def _leaf_dtype(pred: dict) -> str | None:
    if pred["kind"] == "Column": return pred["dtype"]
    if pred["kind"] == "LiteralI64": return "I64"
    if pred["kind"] == "LiteralF64": return "F64"
    if pred["kind"] == "LiteralBool": return "Bool"
    return None  # nested expression: skip for now
```

Update `_walk_filter` to call `_walk_predicate` and accept any returned dict.

- [ ] **Step 3: Extend Rust `deserialize_plan` and `udf::execute_node`**

```rust
// crates/polars-metal-core/src/udf.rs:
// - deserialize_plan: PredicateAst now includes Compare, LiteralI64/F64/Bool.
// - execute_node for Filter:
//   1. Evaluate the predicate AST → produces a bit-packed bool column (data + validity).
//      - Leaf: Column → wrap the source df's column.
//      - Leaf: Literal → 1-element materialization (for column-column path) OR
//        use the *_scalar kernel variant if the BinaryExpr has a scalar rhs.
//      - Compare: dispatch_cmp_*64 (or _scalar variant).
//   2. Pass the bit-packed result column to the compaction pipeline (Task 14).
```

Implementation note: When a predicate is `Compare { lhs: Column, rhs: Literal }`, prefer the scalar variant — avoids materializing a length-N broadcast.

- [ ] **Step 4: Run the test**

Run: `pytest tests/python_integration/test_filter_comparison.py -v && make test-kernel`
Expected: PASS.

- [ ] **Step 5: Run conformance**

Run: `make test-conformance`
Expected: matches M0 baseline. Polars' filter tests with comparison predicates now exercise our GPU path; any regression here is a bug.

- [ ] **Step 6: Commit**

```bash
git add python/polars_metal/_walker.py crates/polars-metal-core/src/udf.rs tests/python_integration/test_filter_comparison.py
git commit -m "Filter handler accepts comparison predicates; GPU dispatch end-to-end"
```

---

## Phase 7 — Logical AND/OR kernels

### Task 19: `logical_bool.metal` — 3-valued AND, OR

**Files:**
- Create: `shaders/logical_bool.metal`
- Create: `crates/polars-metal-kernels/src/logical.rs`
- Modify: `crates/polars-metal-kernels/src/lib.rs`
- Create: `crates/polars-metal-kernels/tests/test_logical_bool.rs`

- [ ] **Step 1: Write the failing test, including the exhaustive 3×3 truth table**

```rust
// crates/polars-metal-kernels/tests/test_logical_bool.rs
use polars_metal_kernels::logical::{dispatch_bool_and, dispatch_bool_or};
use polars_metal_kernels::command::CommandQueue;
use polars_metal_buffer::MetalDevice;

#[test]
fn and_truth_table_exhaustive() {
    // 9 row pairs covering every (lhs, rhs) in {T, F, Null} × {T, F, Null}.
    // Row 0: (T, T) → T (valid)
    // Row 1: (T, F) → F (valid)
    // Row 2: (T, Null) → Null
    // Row 3: (F, T) → F (valid)
    // Row 4: (F, F) → F (valid)
    // Row 5: (F, Null) → F (valid)        ← key 3-valued case
    // Row 6: (Null, T) → Null
    // Row 7: (Null, F) → F (valid)        ← symmetric 3-valued case
    // Row 8: (Null, Null) → Null
    let lhs_data  = 0b00000111u8; // T T T F F F N N N → bits 0..8: T T T F F F N N N
    let lhs_valid = 0b00111111u8; // valid in 0..5, null in 6..8
    let rhs_data  = 0b00001001u8; // ...
    let rhs_valid = 0b00011011u8; // ...
    // (Compute the exact masks for the table above; this is illustrative.)
    let exp_data  = /* per truth table */ 0u8;
    let exp_valid = /* per truth table */ 0u8;
    let device = MetalDevice::system_default().unwrap();
    let mut queue = CommandQueue::new(&device).unwrap();
    let mut got_data = vec![0u8; 2];
    let mut got_valid = vec![0u8; 2];
    dispatch_bool_and(&device, &mut queue, &[lhs_data], &[lhs_valid], &[rhs_data], &[rhs_valid], 9, &mut got_data, &mut got_valid).unwrap();
    assert_eq!(got_data[0] & 0x1FF, exp_data & 0x1FF);
    assert_eq!(got_valid[0] & 0x1FF, exp_valid & 0x1FF);
}

// Plus an `or_truth_table_exhaustive`, plus proptest matching CPU reference.
```

- [ ] **Step 2: Write the MSL kernel**

```msl
// shaders/logical_bool.metal
#include "_validity.metal"

kernel void bool_and(
    device const uint8_t* lhs_data   [[buffer(0)]],
    device const uint8_t* lhs_valid  [[buffer(1)]],
    device const uint8_t* rhs_data   [[buffer(2)]],
    device const uint8_t* rhs_valid  [[buffer(3)]],
    device       uint8_t* out_data   [[buffer(4)]],
    device       uint8_t* out_valid  [[buffer(5)]],
    constant     uint32_t& n_rows    [[buffer(6)]],
    uint                  gid        [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    bool ld = get_valid(lhs_data, gid);
    bool lv = get_valid(lhs_valid, gid);
    bool rd = get_valid(rhs_data, gid);
    bool rv = get_valid(rhs_valid, gid);

    // 3-valued AND truth table:
    //  - (false, *) and (*, false) → false (valid)
    //  - (true, true) → true (valid)
    //  - (null, true) and (true, null) → null
    //  - (null, null) → null
    bool result_valid;
    bool result_data;
    if (!lv && !rv) {                 result_valid = false; result_data = false; }
    else if (lv && !ld) {              result_valid = true;  result_data = false; }   // false dominates
    else if (rv && !rd) {              result_valid = true;  result_data = false; }
    else if (lv && ld && rv && rd) {   result_valid = true;  result_data = true; }
    else {                             result_valid = false; result_data = false; }

    // Atomic OR (same race shape as comparison kernels).
    if (result_valid) {
        set_valid_atomic_or((device atomic_uint*)out_valid, gid);
    }
    if (result_data) {
        set_valid_atomic_or((device atomic_uint*)out_data, gid);
    }
}

kernel void bool_or(/* ... mirror with OR truth table */) { /* ... */ }
```

- [ ] **Step 3: Implement Rust dispatchers**

Same pattern as Task 16. Zero-init out buffers before dispatch.

- [ ] **Step 4: Run the test**

Run: `cargo test -p polars-metal-kernels --test test_logical_bool`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add shaders/logical_bool.metal crates/polars-metal-kernels/src/{logical.rs,lib.rs} crates/polars-metal-kernels/tests/test_logical_bool.rs
git commit -m "Kernel: logical_bool — 3-valued AND/OR"
```

### Task 20: Walker + UDF accept AND/OR predicates; end-to-end compound test

**Files:**
- Modify: `python/polars_metal/_walker.py`
- Modify: `crates/polars-metal-core/src/udf.rs`
- Create: `tests/python_integration/test_filter_compound.py`

- [ ] **Step 1: Write the failing test**

```python
def test_filter_compound_predicate_runs_on_gpu():
    df = pl.DataFrame({
        "a": [1, 2, 3, 4, 5, None],
        "b": [10, 20, 30, 40, 50, 60],
        "c": [50, 40, 30, 20, 10, 5],
    })
    cpu = df.lazy().filter((pl.col("a") > 0) & (pl.col("b") < pl.col("c"))).collect()
    metal = df.lazy().filter((pl.col("a") > 0) & (pl.col("b") < pl.col("c"))).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_or_predicate_3valued_logic():
    df = pl.DataFrame({"a": [True, False, None], "b": [None, False, True]})
    cpu = df.lazy().filter(pl.col("a") | pl.col("b")).collect()
    metal = df.lazy().filter(pl.col("a") | pl.col("b")).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)
```

- [ ] **Step 2: Extend `_walker.py::_walk_predicate` to accept BinaryExpr(`&`/`|`)**

```python
# In _walk_predicate, after the cmp_ops branch:
if op_name in {"And", "Or"}:
    lhs = _walk_predicate(expr.left)
    rhs = _walk_predicate(expr.right)
    if lhs is None or rhs is None: return None
    if _result_dtype(lhs) != "Bool" or _result_dtype(rhs) != "Bool":
        return None
    return {"kind": op_name, "lhs": lhs, "rhs": rhs}
```

`_result_dtype` returns the boolean output type for Compare/And/Or, or the column/literal dtype for leaves.

- [ ] **Step 3: Extend Rust `udf::execute_node` to evaluate AND/OR**

In the predicate AST evaluator, add `PredicateAst::And/Or` arms: recursively evaluate lhs and rhs (each returns a bit-packed bool column), then `dispatch_bool_and` / `dispatch_bool_or`.

- [ ] **Step 4: Run the test**

Run: `pytest tests/python_integration/test_filter_compound.py -v && make test-kernel && make test-conformance`
Expected: PASS. Conformance still matches M0 baseline.

- [ ] **Step 5: Commit**

```bash
git add python/polars_metal/_walker.py crates/polars-metal-core/src/udf.rs tests/python_integration/test_filter_compound.py
git commit -m "Filter handler accepts AND/OR predicates with 3-valued logic"
```

---

## Phase 8 — Differential coverage extensions

### Task 21: Hypothesis strategies for M1 op set

**Files:**
- Modify: `tests/diff/strategies.py`
- Create: `tests/diff/test_filter_random.py`
- Create: `tests/diff/test_filter_edges.py`

- [ ] **Step 1: Write the strategy generators**

```python
# tests/diff/strategies.py — additions

from hypothesis import strategies as st
import polars as pl


@st.composite
def m1_null_density_dataframe(draw):
    """DataFrame with i64/f64/bool columns; null density biased toward
    0% and 100% (most likely to expose bit-packing bugs)."""
    n_rows = draw(st.integers(min_value=0, max_value=1000))
    n_cols = draw(st.integers(min_value=1, max_value=4))
    cols = {}
    for i in range(n_cols):
        dtype = draw(st.sampled_from(["i64", "f64", "bool"]))
        null_density = draw(st.sampled_from([0.0, 0.0, 0.3, 0.7, 1.0]))  # biased
        # ... generate the column
    return pl.DataFrame(cols)


@st.composite
def m1_predicate_expr(draw, schema):
    """Generate predicates from the closed M1 set, depth ≤ 3."""
    # Pick from: Column(bool), Compare(col, lit), Compare(col, col), And/Or of any of the above.
    ...


@st.composite
def m1_projection_subset(draw, schema):
    """Generate a random subset of columns in random order."""
    ...
```

- [ ] **Step 2: Write the differential tests**

```python
# tests/diff/test_filter_random.py
import polars as pl
import polars_metal
from polars.testing import assert_frame_equal
from hypothesis import given, settings, strategies as st
from .strategies import m1_null_density_dataframe, m1_predicate_expr, m1_projection_subset


# Use hypothesis `data()` so we can derive the predicate strategy from the
# DataFrame's schema, rather than calling `.example()` (which is for REPL use
# only and ignores hypothesis state).
@given(df=m1_null_density_dataframe(), data=st.data())
@settings(max_examples=200, deadline=None)
def test_filter_select_random_inputs_match_cpu(df, data):
    schema = df.schema
    pred = data.draw(m1_predicate_expr(schema))
    cols = data.draw(m1_projection_subset(schema))
    cpu = df.lazy().filter(pred).select(cols).collect()
    metal = df.lazy().filter(pred).select(cols).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)
```

- [ ] **Step 3: Write the edge-case tests**

Per spec Section 6 Layer 2: empty DF, single-row, all-null, all-NaN, all-true predicate, all-false predicate, all-null predicate, predicate that returns empty result.

```python
# tests/diff/test_filter_edges.py — one named function per case
def test_empty_dataframe(): ...
def test_single_row_predicate_true(): ...
def test_single_row_predicate_false(): ...
def test_single_row_predicate_null(): ...
def test_all_null_column(): ...
def test_all_nan_f64(): ...
def test_predicate_all_true(): ...
def test_predicate_all_false(): ...
def test_predicate_all_null(): ...
```

Each test runs both engines, asserts byte-exact equality.

- [ ] **Step 4: Run the tests**

Run: `make test-diff`
Expected: all green. Hypothesis runs 200 examples per property.

- [ ] **Step 5: Commit**

```bash
git add tests/diff/strategies.py tests/diff/test_filter_random.py tests/diff/test_filter_edges.py
git commit -m "M1 differential coverage: hypothesis strategies + explicit edge cases"
```

---

## Phase 9 — Conformance extensions and benchmarks

### Task 22: Wire additional Polars test files into the conformance suite

**Files:**
- Modify: `tests/conformance/test_polars_suite.py` (or wherever the suite-discovery lives)
- Modify: `tests/conformance/_skips.toml`

- [ ] **Step 1: Identify the file paths in `references/polars`**

Run: `ls references/polars/py-polars/tests/unit/operations/test_filter.py references/polars/py-polars/tests/unit/operations/test_comparison.py references/polars/py-polars/tests/unit/operations/test_select.py references/polars/py-polars/tests/unit/expr/test_binary.py`
(Exact names may differ — pin them based on what exists in the `py-1.40.1` checkout.)

- [ ] **Step 2: Extend the conformance harness to include the new files**

```python
# tests/conformance/test_polars_suite.py — add to the discovery list:
M1_SUITE_PATHS = [
    "tests/unit/operations/test_filter.py",
    "tests/unit/operations/test_comparison.py",
    "tests/unit/operations/test_select.py",
    "tests/unit/expr/test_binary.py",
]
```

- [ ] **Step 3: Run the expanded conformance**

Run: `make test-conformance`
Expected: pass count under `engine=MetalEngine()` matches pure-CPU pass count for the same files. Any new failure is a release-blocker and goes back to the kernel/walker that caused it.

- [ ] **Step 4: Document any new skips**

If any new test must be skipped: open an issue, add to `_skips.toml` with a PR-comment-style note about why (M1-supported shapes only — anything outside is "skip by design"). If a test inside M1's claimed scope fails, FIX THE BUG — do not skip.

- [ ] **Step 5: Commit**

```bash
git add tests/conformance/{test_polars_suite.py,_skips.toml}
git commit -m "Conformance: wire filter / comparison / select / binary-expr Polars tests"
```

### Task 23: Criterion benchmarks for each kernel

**Files:**
- Create: `crates/polars-metal-kernels/benches/cmp_i64.rs`
- Create: `crates/polars-metal-kernels/benches/cmp_f64.rs`
- Create: `crates/polars-metal-kernels/benches/logical_bool.rs`
- Create: `crates/polars-metal-kernels/benches/filter_predicate.rs`
- Create: `crates/polars-metal-kernels/benches/filter_scatter.rs`
- Modify: `crates/polars-metal-kernels/Cargo.toml`

- [ ] **Step 1: Write a benchmark per kernel**

```rust
// crates/polars-metal-kernels/benches/cmp_i64.rs
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use polars_metal_kernels::cmp::{CompareOp, dispatch_cmp_i64};
use polars_metal_kernels::command::CommandQueue;
use polars_metal_buffer::MetalDevice;

fn bench_cmp_i64(c: &mut Criterion) {
    let device = MetalDevice::system_default().unwrap();
    let mut group = c.benchmark_group("cmp_i64_lt");
    for &n in &[1_000usize, 100_000, 10_000_000] {
        for &null_density in &[0.0_f64, 0.5, 1.0] {
            let mut queue = CommandQueue::new(&device).unwrap();
            // ... build lhs, rhs, validity per null_density.
            group.bench_with_input(
                BenchmarkId::new(format!("nulls={null_density}"), n),
                &(n, null_density),
                |b, _| b.iter(|| dispatch_cmp_i64(/* ... */)),
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_cmp_i64);
criterion_main!(benches);
```

- [ ] **Step 2: Add `[[bench]]` entries**

```toml
# crates/polars-metal-kernels/Cargo.toml
[[bench]]
name = "cmp_i64"
harness = false

[[bench]]
name = "cmp_f64"
harness = false

[[bench]]
name = "logical_bool"
harness = false

[[bench]]
name = "filter_predicate"
harness = false

[[bench]]
name = "filter_scatter"
harness = false

[dev-dependencies]
criterion = "0.5"
```

- [ ] **Step 3: Run the benchmarks**

Run: `cargo bench -p polars-metal-kernels`
Expected: produces `target/criterion/` HTML reports. Numbers vary per machine; what matters is they run without errors.

- [ ] **Step 4: Commit**

```bash
git add crates/polars-metal-kernels/benches/ crates/polars-metal-kernels/Cargo.toml
git commit -m "Criterion benchmarks for M1 kernels"
```

### Task 24: pytest-benchmark end-to-end queries + baseline.json

**Files:**
- Create: `tests/bench/test_filter_e2e.py`
- Create: `tests/bench/baseline.json` (initial, post-M1 numbers)

- [ ] **Step 1: Write the queries**

```python
# tests/bench/test_filter_e2e.py
import polars as pl
import polars_metal
import pytest


@pytest.fixture(scope="module")
def big_frame():
    n = 10_000_000
    return pl.DataFrame({
        "a": pl.Series([(i % 200) - 100 for i in range(n)], dtype=pl.Int64),
        "b": pl.Series([i % 100 for i in range(n)], dtype=pl.Int64),
        "c": pl.Series([(i * 7) % 100 for i in range(n)], dtype=pl.Int64),
    })


@pytest.mark.benchmark(group="filter_simple_cpu")
def test_bench_filter_simple_cpu(benchmark, big_frame):
    benchmark(lambda: big_frame.lazy().filter(pl.col("a") > 0).collect())


@pytest.mark.benchmark(group="filter_simple_metal")
def test_bench_filter_simple_metal(benchmark, big_frame):
    benchmark(lambda: big_frame.lazy().filter(pl.col("a") > 0).collect(engine=polars_metal.MetalEngine()))


# Repeat for: compound, then-project, then-project-high-selectivity, then-project-low-selectivity.
```

- [ ] **Step 2: Run, then capture baseline**

Run: `make bench`
Expected: prints wall-clock for each query. Pair them up: for each Metal/CPU duo, compute the ratio.

- [ ] **Step 3: Generate `baseline.json`**

```json
{
  "machine": "M2 Ultra",
  "git_sha": "<HEAD>",
  "date": "2026-05-20",
  "queries": {
    "filter_simple": {
      "cpu_ms": 12.3,
      "metal_ms": 11.8,
      "ratio_metal_over_cpu": 0.96
    },
    "filter_compound": { ... },
    ...
  }
}
```

If any `ratio_metal_over_cpu > 1.05` (Metal >5% slower than CPU): debug the kernel, optimize, repeat.

- [ ] **Step 4: Commit**

```bash
git add tests/bench/test_filter_e2e.py tests/bench/baseline.json
git commit -m "End-to-end perf benchmarks + M1 baseline on M2 Ultra"
```

---

## Phase 10 — Docs and retrospective

### Task 25: Update `docs/architecture.md` with M1's walker + plan IR + kernel dispatch picture

**Files:**
- Modify: `docs/architecture.md`

- [ ] **Step 1: Replace the M0 stub with an M1 description**

Add sections describing:
- The walker's bottom-up Handled/FallBack model and partial-dispatch policy.
- The `MetalPlanNode` intermediate IR and why it exists.
- Kernel dispatch path: walker → UDF → `_native.execute_plan` → `polars-metal-kernels::pipeline`.
- Per-query arena and the keep-alive deallocator pattern.
- Shader build pipeline (build.rs → metallib → ShaderLibrary cache).

Keep concrete. Reference cuDF-polars where we mirror it; note divergences.

- [ ] **Step 2: Commit**

```bash
git add docs/architecture.md
git commit -m "Document M1 walker / plan IR / kernel dispatch architecture"
```

### Task 26: Update `docs/kernel-authoring.md` with the null-aware MSL idiom

**Files:**
- Modify: `docs/kernel-authoring.md`

- [ ] **Step 1: Write the kernel-authoring guide**

Cover:
- One MSL kernel family per file (`shaders/cmp_i64.metal` = the i64 comparison family).
- The leading-underscore = header convention (`_validity.metal`).
- The atomic OR pattern for any output where 8 rows share a byte (bit-packed bool data + validity bitmaps).
- Zero-init outputs before atomic dispatch.
- 4-byte alignment for any buffer used as `device atomic_uint*`.
- The macro pattern for generating six comparison entry points from one template.
- Threadgroup sizing: query at runtime, do not hardcode (per CLAUDE.md gotcha).
- Reading the matching cuDF CUDA kernel before writing MSL (CLAUDE.md convention).
- The Polars NaN / null semantics rules (link to the spec for the canonical list).

- [ ] **Step 2: Commit**

```bash
git add docs/kernel-authoring.md
git commit -m "Document M1 null-aware MSL idiom for future kernel authors"
```

### Task 27: Update `docs/open-questions.md`

**Files:**
- Modify: `docs/open-questions.md`

- [ ] **Step 1: Edit**

- Resolve / move: the MLX FFI revisit note now has data (one new MLX op used, friction noted in Task 5 Step 5).
- Resolve: the differential-harness "bare scans only" gap (Task 21 closed it for the M1 op set).
- Add new entries M1 surfaced (anything from the retrospective).

- [ ] **Step 2: Commit**

```bash
git add docs/open-questions.md
git commit -m "Update open-questions.md after M1: MLX FFI revisit data, diff-harness gap closed"
```

### Task 28: Write the M1 retrospective in the design spec

**Files:**
- Modify: `docs/superpowers/specs/2026-05-20-m1-design.md` (the retrospective stub at the bottom)

- [ ] **Step 1: Fill in the retrospective**

Following M0's template:
- **Outcome.** Per-exit-criterion pass/fail with numbers (`make gate` wall-clock, kernel count, conformance pass count, perf-bench ratios from `baseline.json`).
- **Surprises during execution.** Anything that diverged from this plan — API quirks (objc2-metal atomic API, MLX cumsum interface, Polars NodeTraverser shape), threadgroup-size discoveries, etc.
- **Resolved in PR follow-up commits.** Anything that was scoped to "after this milestone PR" but landed before merge.
- **Still to revisit at M2.** Things this milestone surfaced that M2 should handle (likely: MLX FFI revisit, predicate AST → kernel-fusion question, multi-chunk frame support).
- **Portability gate results.** Once the user has run `make gate` on M2 (16GB) and M1 (8GB), record git SHA + pass/fail.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-05-20-m1-design.md
git commit -m "M1 retrospective: outcome, surprises, follow-ups, M2 hand-off"
```

### Task 29: Full gate + portability gate + push

**Files:** none (verification + push only).

- [ ] **Step 1: Run the full local gate**

Run: `make gate`
Expected: all phases pass. Wall-clock estimate: ~30s (M0 was ~6s; M1 adds kernel tests + benches not in the gate target — confirm `make gate` doesn't accidentally run benches).

- [ ] **Step 2: Run the portability gate on M2 (16 GB)**

User-action: run `make gate` on the small-M2 machine. Capture the result (date, git SHA, pass/fail summary). Paste into the retrospective.

- [ ] **Step 3: Run the portability gate on M1 (8 GB)**

Same. The 10M-row bench in particular should fit in 8 GB — that's the test of the arena sizing.

- [ ] **Step 4: Push the branch and open a PR**

Run:
```bash
git push -u origin m1-scan-project-filter
gh pr create --title "M1: scan / project / filter on GPU" --body "$(cat <<'EOF'
## Summary

- Bottom-up IR walker with `Handled`/`FallBack` per-node dispatch
- Five MSL kernels: cmp_{i64,f64}, logical_bool, filter_predicate, filter_scatter_{i64,f64,bool}
- MLX cumsum binding for stream-compaction prefix-sum
- Custom null-aware kernels throughout — pattern established for all subsequent milestones
- Conformance: zero new failures vs M0 baseline
- Perf: engine=MetalEngine() within 5% of engine="cpu" on M2 Ultra across five queries

See spec: `docs/superpowers/specs/2026-05-20-m1-design.md`
See plan: `docs/superpowers/plans/2026-05-20-m1-scan-project-filter.md`

## Test plan

- [x] `make gate` on M2 Ultra
- [x] Portability gate on M2 (16 GB)
- [x] Portability gate on M1 (8 GB)
- [x] `make bench` numbers checked into `tests/bench/baseline.json`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: Done**

M1 ships when the PR merges.

---

## Notes for the implementer

- **Read the spec before each phase.** Phases here implement specific spec sections — when in doubt about scope or semantics, the spec wins.
- **Read the matching cuDF kernel first.** CLAUDE.md is firm on this for kernel work. Specifically: `copy_if.cu` for compaction, `binaryop.cu` for cmp variants.
- **The MLX cumsum copy-in/copy-out via CxxVector is known-suboptimal.** Acceptable for M1; M2 reconsiders. Don't optimize within M1 — that's scope creep.
- **Atomics on `device atomic_uint*` require 4-byte alignment of the underlying buffer.** Allocate validity (and bit-packed-bool data) buffers with `((n_bits + 7) / 8 + 4) & !3` byte sizes.
- **Zero-init buffers before any atomic-OR dispatch.** This is the discipline; forgetting it produces sporadic, hard-to-reproduce data corruption.
- **Don't introduce new dependencies without a written justification in the PR description** (per CLAUDE.md). The five files this plan creates use only what's already in the workspace.
- **Don't speculatively optimize.** Land correct + tested first; profile second; optimize third. M1's bar is correctness + no regression, not "faster than CPU."
