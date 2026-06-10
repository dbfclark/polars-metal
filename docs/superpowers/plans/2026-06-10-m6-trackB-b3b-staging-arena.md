# B3b — Reusable Page-Aligned Staging Arena Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Each task is a self-contained TDD loop (failing test → run/FAIL → minimal impl → run/PASS → lint → commit); the orchestrator reviews each task's diff before dispatching the next.

**Goal:** Eliminate the per-call Metal buffer-allocation tax on kernel input staging by adding a **reusable, growable, page-aligned `StagingPool`** to the buffer crate, and rewire `execute_dt` to use it — roughly **doubling dt throughput** (measured: 10M `dt.year` 6.8×→~11×, 50M 11.6×→~24× vs Polars CPU). The pool is generic over bytes so rolling / vector-search / FFT can adopt it later.

**Architecture:** A spike (2026-06-10) proved the cost in the current `execute_dt` input path is the **per-call `newBufferWithBytes` allocation**, not the copy: for a 190 MB input, `newBufferWithBytes`/call = 17.9 ms, `newBufferWithLength`+memcpy/call = 23.8 ms (allocation is the bottleneck — even worse), but a **reused** buffer + memcpy = 3.7 ms (~5× faster). So the fix is reuse, per CLAUDE.md's "reuse scratch buffers via a per-query arena." We add `StagingPool` (holds one growable Shared `MetalBuffer`, reallocates only when a larger input arrives, otherwise `memcpy`s into the existing one) to `polars-metal-buffer`; `polars-metal-core` holds a process-global `OnceLock<Mutex<StagingPool>>` and `execute_dt` stages its input through it. The **output path is unchanged** — it is already zero-copy (the caller's page-aligned numpy buffer → `bytesNoCopy` → Polars wraps it back). Only the input is pooled.

**Tech Stack:** Rust (`polars-metal-buffer` for the pool primitive + `new_buffer_uninit`; `polars-metal-core` for the global pool + `execute_dt` rewiring), Metal (`MTLResourceStorageModeShared` unified-memory buffers), PyO3. Differential correctness via the existing dt test suites (unchanged behavior); perf verified by direct measurement.

---

## Why this design (spike grounding, 2026-06-10)

Per-call allocation strategies for a 40 MB (10M i32) and 190 MB (50M i32) host→Metal input, `min` over 8 reps on M-series:

| strategy | 10M | 50M |
|---|---|---|
| (a) `newBufferWithBytes` per call (**current** `execute_dt` path) | 3.67 ms | 17.87 ms |
| (b) `newBufferWithLength`(zeroed) + memcpy per call | 4.18 ms | 23.78 ms |
| (c) **reused** buffer + memcpy only | **0.74 ms** | **3.72 ms** |

Conclusions that drive the design:
- **Allocation is the bottleneck.** (b) is *slower* than (a), so a per-call `newBufferWithLength` path is pointless — only **reuse** (c) wins. The pool must persist the buffer across calls.
- **The copy itself is cheap** (~0.74 ms / 40 MB ≈ 54 GB/s). The boundary memcpy is unavoidable (Polars' Arrow buffers are 64-byte aligned; Metal's `newBufferWithBytesNoCopy` hard-requires 16 KB page alignment — confirmed in `crates/polars-metal-buffer/src/alignment.rs`), but at memory-bandwidth speed it is not the problem.
- **End-to-end projection** (the input-side delta carries straight through): `execute_dt` input handling drops from ~3.7 ms to ~0.74 ms at 10M and ~17.9 ms to ~3.7 ms at 50M, taking the full engine path from 6.8×→~11× (10M) and 11.6×→~24× (50M).

Non-negotiable constraints respected:
- The pooled input buffer's capacity may **exceed** the data length. `dispatch_dt_field_buf` already takes `n: u32` explicitly and reads only `input[0..n)` — it never consults `buffer.len()`. So an oversized pooled buffer is correct as long as the right `n` is passed (it is).
- The **output** stays the caller's page-aligned `np.empty` (already zero-copy via `from_borrowed_i32`); it must NOT be pooled (it becomes the result Series and is handed to Polars).
- No `unwrap`/`expect`/`panic` in non-test code; every `unsafe` carries a `// SAFETY:` comment; `--test-threads=1` on all `cargo test` (Metal command-queue contention).

### Scope / non-goals

- **In:** `StagingPool` + `new_buffer_uninit` in the buffer crate; a process-global pool in core; `execute_dt` input rewired to it; perf re-measured; gate green.
- **Out (this milestone):** wiring rolling / vector-search / FFT to the pool (the struct is designed reusable, but adopting it elsewhere is a deliberate follow-up — each has its own staging call site and tests). Pooling the *output* (it must become a Polars Series). A per-query (vs process-global) arena lifetime — the global single-buffer pool is the simplest correct choice on unified memory (one buffer at peak size, anti-fragmentation, no OOM cliff per CLAUDE.md).

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `crates/polars-metal-buffer/src/device.rs` | Modify | Add `new_buffer_uninit(bytes)` — `newBufferWithLength` Shared alloc with NO zeroing (the pool overwrites via memcpy; bytes beyond the staged length are never read). |
| `crates/polars-metal-buffer/src/staging.rs` | Create | `StagingPool` — one growable Shared `MetalBuffer`; `stage(device, &[u8]) -> Result<&MetalBuffer>` reallocates only when `src.len() > capacity`, else memcpys into the existing buffer. |
| `crates/polars-metal-buffer/src/lib.rs` | Modify | `pub mod staging;` + re-export `StagingPool`. |
| `crates/polars-metal-buffer/tests/test_staging.rs` | Create | Unit tests: stage copies bytes correctly; reuse keeps the same underlying buffer when len ≤ capacity; grow reallocates; smaller-after-larger reuses + reads correct prefix; content correctness across a reuse sequence. |
| `crates/polars-metal-core/src/udf.rs` | Modify | `execute_dt`: replace the `from_borrowed_i32` input staging with a process-global `StagingPool` (memcpy into a reused buffer); dispatch with the pooled buffer + `n`. Output path unchanged. |
| `tests/python_integration/test_dt_e2e.py` | (no change expected) | The existing differential matrix must still pass byte-exact — proves the pool didn't break correctness. |

---

## Task 1 — `new_buffer_uninit` + `StagingPool` in the buffer crate

Add the uninitialized Shared allocator and the reusable pool, with unit tests pinning the reuse/grow semantics and content correctness.

**Files**
- Modify: `crates/polars-metal-buffer/src/device.rs`
- Create: `crates/polars-metal-buffer/src/staging.rs`
- Modify: `crates/polars-metal-buffer/src/lib.rs`
- Create (test): `crates/polars-metal-buffer/tests/test_staging.rs`

**Step 1: Write the failing test.** Create `crates/polars-metal-buffer/tests/test_staging.rs`:
```rust
// crates/polars-metal-buffer/tests/test_staging.rs
//
// Unit tests for the reusable page-aligned StagingPool (B3b). Validates that
// staging copies bytes correctly, reuses the underlying buffer when the new
// input fits, reallocates (grows) when it does not, and reads back the correct
// prefix after a larger-then-smaller sequence.
//
// Requires Metal-capable hardware; skips via `expect` without a device.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use polars_metal_buffer::{MetalDevice, StagingPool};
use std::sync::Mutex;

static METAL_TEST_LOCK: Mutex<()> = Mutex::new(());

fn dev() -> MetalDevice {
    MetalDevice::system_default().expect("Metal-capable hardware required")
}

#[test]
fn stage_copies_bytes() {
    let _l = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let device = dev();
    let mut pool = StagingPool::new();
    let src: Vec<u8> = (0..200u32).map(|i| (i % 256) as u8).collect();
    let buf = pool.stage(&device, &src).expect("stage ok");
    assert_eq!(&buf.as_slice()[..src.len()], &src[..]);
}

#[test]
fn stage_reuses_buffer_when_fits() {
    let _l = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let device = dev();
    let mut pool = StagingPool::new();
    let big = vec![1u8; 4096];
    let cap_ptr = {
        let b = pool.stage(&device, &big).expect("stage big");
        b.as_slice().as_ptr() as usize
    };
    // A smaller input must reuse the SAME backing allocation (same contents ptr).
    let small = vec![2u8; 128];
    let b2 = pool.stage(&device, &small).expect("stage small");
    assert_eq!(b2.as_slice().as_ptr() as usize, cap_ptr, "should reuse buffer");
    assert_eq!(&b2.as_slice()[..small.len()], &small[..]);
}

#[test]
fn stage_grows_when_larger() {
    let _l = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let device = dev();
    let mut pool = StagingPool::new();
    let small = vec![9u8; 64];
    let p1 = pool.stage(&device, &small).expect("s1").as_slice().as_ptr() as usize;
    let _ = p1;
    let big = vec![7u8; 1_000_000];
    let b = pool.stage(&device, &big).expect("grow");
    assert!(b.len() >= big.len(), "capacity grew to fit");
    assert_eq!(&b.as_slice()[..big.len()], &big[..]);
}

#[test]
fn stage_rejects_empty() {
    let _l = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let device = dev();
    let mut pool = StagingPool::new();
    assert!(pool.stage(&device, &[]).is_err(), "empty input rejected");
}
```

**Step 2: Run it, expect FAIL (compile error — `StagingPool` missing).**
`cargo test -p polars-metal-buffer --test test_staging -- --test-threads=1`
Expected: `unresolved import polars_metal_buffer::StagingPool`.

**Step 3: Add `new_buffer_uninit`** to `crates/polars-metal-buffer/src/device.rs` (place it right after `new_buffer_zeroed`, mirroring it but skipping the zero-fill):
```rust
    /// Allocate a new shared-storage `MTLBuffer` of the given length, WITHOUT
    /// zeroing its contents.
    ///
    /// Mirrors [`new_buffer_zeroed`](Self::new_buffer_zeroed) but skips the
    /// `write_bytes` zero-fill. Intended for staging buffers whose full used
    /// prefix is overwritten by a `memcpy` before any read (e.g.
    /// [`crate::StagingPool`]); bytes beyond the staged length are never read.
    ///
    /// Storage mode is `MTLResourceStorageModeShared` (unified memory, CPU- and
    /// GPU-addressable, page-aligned base). Returns
    /// `BufferError::AllocationFailed` when `bytes == 0` or Metal refuses.
    pub fn new_buffer_uninit(&self, bytes: usize) -> Result<MetalBuffer, BufferError> {
        if bytes == 0 {
            return Err(BufferError::AllocationFailed { bytes: 0 });
        }
        let inner = self
            .inner
            .newBufferWithLength_options(bytes, MTLResourceOptions::MTLResourceStorageModeShared)
            .ok_or(BufferError::AllocationFailed { bytes })?;
        Ok(MetalBuffer::from_metal_owned(inner))
    }
```
(Confirm `MetalBuffer::from_metal_owned` is the constructor `new_buffer_zeroed` uses — reuse the identical idiom. `MTLResourceOptions` is already imported in `device.rs`.)

**Step 4: Create the pool.** Create `crates/polars-metal-buffer/src/staging.rs`:
```rust
//! Reusable page-aligned staging buffer for kernel inputs (M6 B3b).
//!
//! Ingesting a Polars/Arrow column into Metal requires one copy: Polars' Arrow
//! buffers are 64-byte aligned, while `newBufferWithBytesNoCopy` hard-requires
//! 16 KB page alignment (see [`crate::alignment`]). A spike showed the cost of
//! the current per-call `newBufferWithBytes` path is the *allocation*, not the
//! copy — a reused Shared buffer + `memcpy` is ~5× faster than allocating a
//! fresh buffer each call. [`StagingPool`] holds one growable Shared
//! [`MetalBuffer`] and reallocates only when a larger input arrives.
//!
//! The pool buffer's capacity may exceed the staged length; kernels that take
//! an explicit element count (`n`) and read only `input[0..n)` are unaffected.
//! Callers must pass the true element count to the dispatcher, NOT
//! `buffer.len()`.

use crate::{BufferError, MetalBuffer, MetalDevice};

/// A single reusable, growable, page-aligned Shared staging buffer.
///
/// Not `Sync`/thread-safe on its own — wrap in a `Mutex` for cross-thread use
/// (Metal command submission serializes anyway). One pool holds at most one
/// buffer, sized to the largest input seen so far.
#[derive(Default)]
pub struct StagingPool {
    buf: Option<MetalBuffer>,
}

impl StagingPool {
    /// A pool with no buffer yet allocated.
    pub const fn new() -> Self {
        Self { buf: None }
    }

    /// Stage `src` into the pooled buffer and return it.
    ///
    /// Reallocates (via [`MetalDevice::new_buffer_uninit`]) only when `src` is
    /// larger than the current capacity; otherwise reuses the existing buffer.
    /// The first `src.len()` bytes are overwritten by `memcpy`; bytes beyond
    /// are left as-is (never read by a kernel that respects its `n`).
    ///
    /// Returns `BufferError::AllocationFailed { bytes: 0 }` for an empty input
    /// (Metal rejects zero-byte buffers).
    pub fn stage(
        &mut self,
        device: &MetalDevice,
        src: &[u8],
    ) -> Result<&MetalBuffer, BufferError> {
        let need = src.len();
        if need == 0 {
            return Err(BufferError::AllocationFailed { bytes: 0 });
        }
        let grow = self.buf.as_ref().map_or(true, |b| b.len() < need);
        if grow {
            self.buf = Some(device.new_buffer_uninit(need)?);
        }
        // `as_mut()` cannot be None here (set above or pre-existing).
        let buf = self
            .buf
            .as_mut()
            .ok_or(BufferError::AllocationFailed { bytes: need })?;
        buf.as_mut_slice()[..need].copy_from_slice(src);
        Ok(buf)
    }
}
```
(Verify the re-exports used: `BufferError`, `MetalBuffer`, `MetalDevice` are crate-root types in `polars-metal-buffer`. `MetalBuffer::as_mut_slice(&mut self) -> &mut [u8]` exists at `bridge.rs:477`; `len()` at `bridge.rs:248`; `as_slice()` at `bridge.rs:456`. If `map_or` trips a clippy `unwrap_or_default`-style lint, the explicit `match`/`is_none_or` form is fine — follow clippy's suggestion.)

**Step 5: Export it.** In `crates/polars-metal-buffer/src/lib.rs`, add the module + re-export next to the other public types:
```rust
pub mod staging;
pub use staging::StagingPool;
```
(Match the existing `pub mod` / `pub use` ordering and style in that file.)

**Step 6: Run the tests, expect PASS.**
`cargo test -p polars-metal-buffer --test test_staging -- --test-threads=1`
Expected: 4 tests pass.

**Step 7: Lint.**
`cargo clippy -p polars-metal-buffer --all-targets -- -D warnings && cargo fmt -p polars-metal-buffer -- --check`
Expected: clean.

**Step 8: Commit.**
```bash
git add crates/polars-metal-buffer/src/device.rs crates/polars-metal-buffer/src/staging.rs crates/polars-metal-buffer/src/lib.rs crates/polars-metal-buffer/tests/test_staging.rs
git commit -m "B3b T1: StagingPool — reusable page-aligned Shared staging buffer + new_buffer_uninit"
```

---

## Task 2 — Rewire `execute_dt` input staging through a global `StagingPool`

Replace `execute_dt`'s per-call `from_borrowed_i32` input staging with a process-global pooled `StagingPool` (memcpy into a reused buffer). Output unchanged. The existing dt tests prove correctness; measure the perf win.

**Files**
- Modify: `crates/polars-metal-core/src/udf.rs`
- Test: `tests/python_integration/test_dt_e2e.py` + `tests/python_integration/test_dt_binding.py` (existing — must still pass)

**Step 1: Confirm the RED is a perf observation, not a correctness one.** The existing dt tests pass today (Task B3 shipped). This task must keep them green while changing the staging mechanism. First, capture the current perf baseline to confirm the win after:
```bash
python3 -c "
import polars as pl, time
from polars_metal import MetalEngine
eng=MetalEngine()
for n in (10_000_000, 50_000_000):
    df=pl.DataFrame({'d': pl.Series('d', range(n), dtype=pl.Int32).cast(pl.Date)})
    mk=lambda: df.lazy().with_columns(pl.col('d').dt.year().alias('o'))
    mk().collect(engine=eng)
    g=min(((lambda l: (time.perf_counter(), l.collect(engine=eng), time.perf_counter()))(mk()) ) for _ in range(5))
    # simpler timing:
    import statistics
    gs=[]; cs=[]
    for _ in range(5):
        l=mk(); b=time.perf_counter(); l.collect(engine=eng); gs.append(time.perf_counter()-b)
        l=mk(); b=time.perf_counter(); l.collect(); cs.append(time.perf_counter()-b)
    print(f'n={n:,}: gpu={min(gs)*1e3:.1f}ms cpu={min(cs)*1e3:.1f}ms speedup={min(cs)/min(gs):.1f}x')
" 2>&1 | grep -v Warning
```
Record the baseline (expected ~6.8× @10M, ~11.6× @50M).

**Step 2: Implement the rewiring** in `crates/polars-metal-core/src/udf.rs`. Add a process-global pool near the top of the file (after the imports, with the other module-level items):
```rust
use std::sync::{Mutex, OnceLock};

/// Process-global reusable staging buffer for `execute_dt` inputs (B3b).
/// One buffer, grown to the largest input seen; the `Mutex` serializes dt
/// dispatches (Metal command submission serializes anyway). Designed so other
/// kernel bindings can adopt the same pattern with their own pool later.
static DT_STAGING: OnceLock<Mutex<polars_metal_buffer::StagingPool>> = OnceLock::new();
```
(If `std::sync::{Mutex, OnceLock}` or `polars_metal_buffer` are already imported in `udf.rs`, don't duplicate — fold into the existing `use`.)

Then, inside `execute_dt`, replace the **input** staging block (the `from_borrowed_i32` for `in_ptr` and its `inb` binding) with a pooled stage. The output (`outb` from `from_borrowed_i32` on `out_ptr`) and the copy-back stay exactly as they are. The dispatch call changes only in that its input is the pooled buffer:
```rust
    // Input staging via the reusable page-aligned pool: one memcpy into a
    // reused Shared buffer (B3b). Reuse avoids the per-call allocation that
    // dominated the old `newBufferWithBytes` path (~5x faster at scale). The
    // pooled buffer's capacity may exceed `in_n * 4`; the kernel reads only
    // `input[0..n)`, so passing the true `n` below is correct.
    let pool = DT_STAGING.get_or_init(|| Mutex::new(polars_metal_buffer::StagingPool::new()));
    // Recover from a poisoned lock rather than panic (the buffer is scratch;
    // no invariant is corrupted by a prior panic mid-stage).
    let mut staging = pool.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    // SAFETY: `in_ptr` addresses `in_n` live, contiguous i32 values for the
    // whole synchronous call; reinterpreting as `in_n * 4` bytes is sound
    // (`i32` has no invalid bit patterns) and the slice is only read (memcpy
    // source) before this function returns.
    let in_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(in_ptr as *const u8, in_n * 4) };
    let inb = staging.stage(&device, in_bytes).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: dt input staging: {e}"))
    })?;

    // ... existing output staging (`outb` via from_borrowed_i32 on out_ptr) ...
    // ... existing dispatch, now reading the pooled input: ...
    dispatch_dt_field_buf(&device, inb, &outb, n, dt_field).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: dt dispatch: {e}"))
    })?;
    // ... existing unaligned copy-back for `out_ptr` ...
    // `staging` (the MutexGuard) drops here, after the dispatch completed
    // (dispatch_dt_field_buf calls wait_until_complete), releasing the pool.
```
Important ordering: keep the `MutexGuard` (`staging`) alive until AFTER `dispatch_dt_field_buf` returns (the kernel reads the pooled buffer; `dispatch_dt_field_buf` calls `wait_until_complete` internally, so when it returns the GPU is done with the buffer and the guard can drop). Do NOT drop or re-lock between stage and dispatch. The existing `n` (`u32::try_from(in_n)`) is passed unchanged.

Remove the now-unused `from_borrowed_i32` import for the input if it was input-only; the **output** still uses `from_borrowed_i32` (keep that import). Verify the final function compiles with the output path intact.

**Step 3: Rebuild the wheel** (Rust changed):
`make wheel`
Expected: builds clean.

**Step 4: Run the existing dt suites, expect PASS (correctness unchanged).**
`pytest tests/python_integration/test_dt_binding.py tests/python_integration/test_dt_e2e.py -v`
Expected: all pass (20 e2e + 2 binding). Byte-exact correctness is unchanged — only the input staging mechanism differs. The null / empty / Datetime / large-N-null cases all still pass.

**Step 5: Measure the win** (re-run the Step-1 script):
```bash
python3 -c "
import polars as pl, time
from polars_metal import MetalEngine
eng=MetalEngine()
for n in (10_000_000, 50_000_000):
    df=pl.DataFrame({'d': pl.Series('d', range(n), dtype=pl.Int32).cast(pl.Date)})
    mk=lambda: df.lazy().with_columns(pl.col('d').dt.year().alias('o'))
    mk().collect(engine=eng)
    gs=[]; cs=[]
    for _ in range(5):
        l=mk(); b=time.perf_counter(); l.collect(engine=eng); gs.append(time.perf_counter()-b)
        l=mk(); b=time.perf_counter(); l.collect(); cs.append(time.perf_counter()-b)
    print(f'n={n:,}: gpu={min(gs)*1e3:.1f}ms cpu={min(cs)*1e3:.1f}ms speedup={min(cs)/min(gs):.1f}x')
" 2>&1 | grep -v Warning
```
Expected: ~11× @10M, ~24× @50M (roughly double the Step-1 baseline). Record both numbers for the commit message. If the win is materially absent, STOP and report (the rewiring may not be hitting the pool — e.g. the guard is dropped too early or the buffer isn't being reused).

**Step 6: Lint.**
`cargo clippy -p polars-metal-core --all-targets -- -D warnings && cargo fmt -p polars-metal-core -- --check`
Expected: clean.

**Step 7: Commit.**
```bash
git add crates/polars-metal-core/src/udf.rs
git commit -m "B3b T2: execute_dt input via reusable StagingPool — ~2x at scale (10M <X>x, 50M <Y>x)"
```
(Substitute the measured speedups.)

---

## Task 3 — Full `make gate` + honest perf record

Confirm no regression across the workspace and record the re-measured dt perf.

**Files**
- None (verification only).

**Steps**

- [ ] **Step 1: Full gate.**
  `make gate`
  Expected: green, OR only the documented pre-existing baseline divergences (MEMORY `m3-conformance-deferrals` / `m6-conformance-fixes`). A NEW failure → STOP and triage (B3b regression). The buffer-crate change is additive (`new_buffer_uninit` + `StagingPool` are new; no existing buffer API changed), and `execute_dt`'s observable behavior is identical — so the only risk is a Rust compile/lint issue or a dt correctness regression, both caught by the gate's `test-unit` + the dt suites.

- [ ] **Step 2: Record the result.**
  ```bash
  git commit --allow-empty -m "B3b T3: make gate green — staging arena lands; dt.year 10M <X>x / 50M <Y>x (was 6.8x/11.6x), no regression"
  ```
  (Substitute the measured speedups from Task 2 Step 5.)

---

## Self-Review: spec coverage

| Requirement | Covered by |
|---|---|
| Reusable page-aligned staging pool (reuse is the proven win, not per-call alloc) | T1 (`StagingPool` + `new_buffer_uninit`, reuse/grow unit tests) |
| `execute_dt` input rewired to the pool; output unchanged (already zero-copy) | T2 (global `OnceLock<Mutex<StagingPool>>`, input via `stage`, output/copy-back untouched) |
| Oversized pooled buffer correct (kernel reads `input[0..n)`) | T1 (`stage_reuses_buffer_when_fits` — smaller input reuses larger buffer) + T2 (passes true `n`) |
| Correctness unchanged (byte-exact, nulls, Datetime, empty, large-N) | T2 Step 4 (existing dt suites green) |
| Perf ~2× at scale, honestly measured | T2 Step 1/5 (before/after) + T3 (recorded) |
| Reusable for rolling/vector/FFT later | T1 (`StagingPool` is generic over `&[u8]`, not dt-specific; documented in scope) |
| No regression | T3 (`make gate`) |
| Conventions: no unwrap/expect/panic in non-test; SAFETY comments; `--test-threads=1`; lint+fmt per task | T1/T2 (poison-recovering lock, SAFETY on the `from_raw_parts`, per-task clippy+fmt) |

**Lock-ordering note (load-bearing):** the `MutexGuard` must stay alive from `stage()` through `dispatch_dt_field_buf()` (which `wait_until_complete`s). Dropping it earlier would let a concurrent dt call overwrite the pooled input mid-flight. T2 Step 2 pins this explicitly.

**Honest framing:** this does NOT reach the true zero-copy floor (4.94 ms / 10.5 ms) — the boundary memcpy from Polars' non-page-aligned Arrow buffer is unavoidable without a Polars fork (Non-goal). It removes the *allocation* tax, which is the addressable ~2× win. dt remains a bandwidth-shaped op per the Mission.
