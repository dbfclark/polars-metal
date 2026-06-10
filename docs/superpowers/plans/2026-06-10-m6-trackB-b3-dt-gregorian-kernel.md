# B3 — `dt` Gregorian MSL Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Each task is a self-contained TDD loop (failing test → run/FAIL → minimal impl → run/PASS → lint → commit); the orchestrator reviews each task's diff before dispatching the next.

**Goal:** Accelerate native `pl.col(d).dt.year()` / `.dt.month()` / `.dt.day()` under `engine="metal"` with a custom branchless gregorian (civil-from-days) Metal kernel — the flagship compute-bound consumer of Track B — matching Polars CPU byte-exact (incl. output dtypes Int32/Int8/Int8) over `Date` and `Datetime` (all three time units), pre-1970 negatives, leap/century boundaries, and nulls.

**Architecture:** `dt.year/month/day` are `NodeTraverser`-opaque, so recognize them the proven rolling/FFT way: a pre-optimization `lf.serialize` plan inspection (`{"Function":{"input":[{"Column":c}],"function":{"TemporalExpr":"Year"|"Month"|"Day"}}}`) with an O(1) `with_columns`-capture fast path, then collect-and-stitch (drop the dt output columns so projection pushdown elides them from the CPU collect, compute each dt binding on the GPU, stitch back in schema order). A single MSL kernel `dt_field_from_days` takes Int32 days-since-epoch + a field selector (0=year,1=month,2=day) and writes Int32; the host narrows month/day to Int8 and restores nulls positionally. `Date` (physically Int32 days) feeds the kernel directly; `Datetime` (Int64) is converted to days host-side via `value // units_per_day` (numpy floor-division, correct toward −∞ for pre-epoch values) before the same kernel runs.

**Tech Stack:** MSL (Metal Shading Language) for the kernel; Rust (`polars-metal-kernels` dispatch wrapper + `polars-metal-core` PyO3 binding) bridged via the existing zero-copy `from_borrowed_i32` buffer path; Python (`polars_metal` detect/dispatch + the `__init__.py` collect hook); Polars CPU as the differential oracle.

---

## Why this design (grounding — all spike-verified 2026-06-10)

Every load-bearing fact below was confirmed empirically before this plan was written (per the "spike unknowns during brainstorm" discipline; Python MLX/Polars are importable in-repo):

1. **Serialize shape (Date AND Datetime, all time units, identical):**
   ```json
   {"Alias": [{"Function": {"input": [{"Column": "d"}],
     "function": {"TemporalExpr": "Year"}}}, "o"]}
   ```
   `"TemporalExpr"` is `"Year"` / `"Month"` / `"Day"`. **The `time_unit` is NOT in the expression JSON** — it comes from the column's schema dtype (`Datetime(time_unit=...)`), exactly like rolling reads the column dtype from the schema.

2. **Output dtypes (must match Polars exactly):** `dt.year`→**Int32**, `dt.month`→**Int8**, `dt.day`→**Int8**.

3. **Physical representation:**
   - `Date` → **Int32** days-since-1970 (can be negative; `1969-12-31` → `-1`, `1900-01-01` → `-25567`).
   - `Datetime` → **Int64** since-epoch in the unit; `units_per_day` = `86_400_000` (ms) / `86_400_000_000` (us) / `86_400_000_000_000` (ns). `days = value // units_per_day` with numpy floor-division: `-82_800_000 // 86_400_000 == -1` (correct — `1969-12-31 01:00` is day `-1`, Polars `.dt.day()` returns 31). **numpy `//` on integers floors toward −∞, matching Polars; MSL integer `/` truncates toward zero, which is why the divide is done host-side, not in the kernel.**

4. **Algorithm — Howard Hinnant `civil_from_days`** (the settled branchless approach): differentially verified against Polars over **11,429 dates from ~1860 to ~2079 (step 7), 0 mismatches**, including all pre-1970 negatives and leap/century cases. Reference implementation (Python form, transliterate to MSL with integer ops):
   ```
   z += 719468
   era = (z >= 0 ? z : z - 146096) / 146097
   doe = z - era*146097                                  // [0, 146096]
   yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365 // [0, 399]
   y   = yoe + era*400
   doy = doe - (365*yoe + yoe/4 - yoe/100)               // [0, 365]
   mp  = (5*doy + 2) / 153                                // [0, 11]
   d   = doy - (153*mp + 2)/5 + 1                         // [1, 31]
   m   = mp < 10 ? mp + 3 : mp - 9                        // [1, 12]
   year_out = y + (m <= 2 ? 1 : 0)
   ```
   All intermediates fit in **Int32** for the supported date range (Polars `Date` is i32 days; the era arithmetic `z+719468` then `*146097` peaks well within i32 for any representable `Date`). The kernel computes in Int32.

5. **Null restore (positional, GPU-preserving — no CPU fallback):** dt is element-wise, so nulls are handled by staging the dense physical column with `to_physical().fill_null(0).to_numpy()`, running the kernel, then restoring via `dense.zip_with(src.is_not_null(), null_fill)` — verified byte-exact vs Polars on a null-bearing Date column. (Mirrors B1's `fill_null(0)` staging lesson: the sentinel 0 is harmless because civil-from-days never traps and the validity mask is restored afterward.)

6. **Collect-and-stitch split works:** `lf.drop("y").explain()` shows the `dt.year()` computation fully elided (projection pushdown) — the rolling/FFT template applies unchanged.

7. **Shaders auto-discovered:** `crates/polars-metal-kernels/build.rs` compiles every `shaders/*.metal` (stems not starting with `_`) into the metallib. Dropping `shaders/dt_gregorian.metal` is sufficient; no build.rs edit.

8. **Buffer/FFI ready:** `MetalBuffer::{from_i32_slice,to_i32_vec,from_borrowed_i32}` already exist (B1, via the `impl_typed_accessors!` macro at `crates/polars-metal-buffer/src/bridge.rs:552`). The kernel is Int32-in / Int32-out, so no new buffer width is needed.

### Scope / non-goals (explicit)

- **In:** `dt.year/month/day` over `Date` and `Datetime` (ms/us/ns), null-bearing, empty, pre-1970, leap/century. Always routes to GPU when detected (compute-bound consumer; no FLOPs/row gating).
- **Out (→ native Polars/CPU, correct by fallback):** time-zone-aware `Datetime` (`time_zone is not None`); `Duration`/`Time` dtypes; every other `dt.*` accessor (`hour`, `weekday`, `ordinal_day`, `quarter`, …); `dt.*` whose input is a sub-expression rather than a bare `Column`; streaming mode; the kernel-side Datetime→days divide (done host-side this milestone — a later optimization may push it into the kernel). B4 owns formal benchmarks/baselines; B3 only sanity-checks the win exists.

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `shaders/dt_gregorian.metal` | Create | MSL kernel `dt_field_from_days`: Int32 days-since-epoch in, field selector scalar (0/1/2), Int32 field out. Branchless Hinnant civil-from-days. Threadgroup/grid assumptions documented at top. |
| `crates/polars-metal-kernels/src/dt.rs` | Create | `DtError`, `dispatch_dt_field_buf` (zero-copy core over pre-staged `MetalBuffer`s — the PyO3 path), `dispatch_dt_field` (test-ergonomics slice wrapper). Mirrors `rolling.rs` structure. |
| `crates/polars-metal-kernels/src/lib.rs` | Modify | Add `pub mod dt;`. |
| `crates/polars-metal-kernels/tests/test_dt_gregorian.rs` | Create | Rust kernel correctness test vs a CPU Hinnant reference: multi-tile, every field, negatives, leap/century, n=0/1. |
| `crates/polars-metal-core/src/udf.rs` | Modify | `execute_dt` PyO3 entry point — stages i32 in/out via `from_borrowed_i32` (zero-copy when page-aligned; copy-back fallback when not), dispatches the kernel. Mirrors `execute_rolling`. |
| `crates/polars-metal-core/src/lib.rs` | Modify | Register `udf::execute_dt`. |
| `python/polars_metal/_dt_detect.py` | Create | Serialize-detect with an INDEPENDENT `with_columns` patch + cache; `DtBinding`, `find_dt_bindings(lf)`. Mirrors `_rolling_detect.py`. |
| `python/polars_metal/_dt_dispatch.py` | Create | `apply_dt(lf, bindings, collect_fn)`: collect-and-stitch; host floor-div for Datetime; kernel call; Int8 narrow for month/day; positional null restore. Mirrors `_rolling_dispatch.py`. |
| `python/polars_metal/__init__.py` | Modify | Import `_dt_detect` (installs the patch eagerly) + add the dt collect-hook block alongside rolling/vector/fft. |
| `tests/python_integration/test_dt_detect.py` | Create | Detect-layer unit tests (no engine call): handleable shapes recognized, non-handleable omitted. |
| `tests/python_integration/test_dt_e2e.py` | Create | Engine-level differential matrix vs Polars CPU, byte-exact incl. dtypes; ≥1 genuine GPU-path case via an `execute_dt` dispatch counter; fallback-correctness cases. |

---

## Task 1 — MSL kernel + Rust dispatch wrapper + Rust kernel correctness test

Build the kernel and its Rust dispatcher together (the natural unit, mirroring how `rolling.rs` + `rolling.metal` + `test_rolling.rs` were built). The Rust integration test is the RED.

**Files**
- Create: `shaders/dt_gregorian.metal`
- Create: `crates/polars-metal-kernels/src/dt.rs`
- Modify: `crates/polars-metal-kernels/src/lib.rs`
- Create (test): `crates/polars-metal-kernels/tests/test_dt_gregorian.rs`

**Steps**

- [ ] **Step 1: Write the failing Rust kernel test.** Create `crates/polars-metal-kernels/tests/test_dt_gregorian.rs`:
  ```rust
  // crates/polars-metal-kernels/tests/test_dt_gregorian.rs
  //
  // Correctness tests for the gregorian civil-from-days kernel
  // (`dt_field_from_days` in `shaders/dt_gregorian.metal`). Validates year /
  // month / day extraction from Int32 days-since-1970 against a CPU Hinnant
  // reference, across multi-tile inputs, pre-1970 negatives, leap/century
  // boundaries, and n=0/1.
  //
  // Requires Metal-capable hardware; skips via `expect` on machines without a
  // discoverable system-default MTLDevice.
  #![allow(clippy::expect_used, clippy::unwrap_used)]

  use polars_metal_buffer::MetalDevice;
  use polars_metal_kernels::dt::{dispatch_dt_field, DtField};
  use std::sync::Mutex;

  static METAL_TEST_LOCK: Mutex<()> = Mutex::new(());

  /// CPU reference: Howard Hinnant civil_from_days (days since 1970-01-01).
  fn civil_from_days(z0: i64) -> (i32, i32, i32) {
      let z = z0 + 719468;
      let era = if z >= 0 { z } else { z - 146096 } / 146097;
      let doe = z - era * 146097; // [0, 146096]
      let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0,399]
      let y = yoe + era * 400;
      let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0,365]
      let mp = (5 * doy + 2) / 153; // [0,11]
      let d = doy - (153 * mp + 2) / 5 + 1; // [1,31]
      let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1,12]
      ((y + if m <= 2 { 1 } else { 0 }) as i32, m as i32, d as i32)
  }

  fn run(days: &[i32], field: DtField) -> Vec<i32> {
      let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
      let device = MetalDevice::system_default().expect("Metal-capable hardware required");
      let mut out = vec![0i32; days.len()];
      dispatch_dt_field(&device, days, &mut out, field).expect("dispatch succeeds");
      out
  }

  #[test]
  fn dt_fields_match_reference_multitile_and_negatives() {
      // ~1500 dates crossing tile boundaries (TG_SIZE=256), incl. pre-1970.
      let days: Vec<i32> = (-25567..-25567 + 1500).collect(); // from 1900-01-01
      let more: Vec<i32> = (0..1500).map(|i| i * 37).collect(); // sparse forward
      for set in [days, more] {
          for (field, idx) in [(DtField::Year, 0), (DtField::Month, 1), (DtField::Day, 2)] {
              let got = run(&set, field);
              for (i, &z) in set.iter().enumerate() {
                  let want = civil_from_days(z as i64);
                  let w = [want.0, want.1, want.2][idx];
                  assert_eq!(got[i], w, "field {field:?} z={z}: got {} want {w}", got[i]);
              }
          }
      }
  }

  #[test]
  fn dt_leap_and_century_boundaries() {
      // 2000-02-29 (leap), 1900-02-28 then 1900-03-01 (NOT leap), 2020-12-31,
      // 2021-01-01, epoch 1970-01-01 (day 0).
      let cases: [(i32, (i32, i32, i32)); 6] = [
          (10956, (2000, 2, 29)),  // 2000-02-29
          (-25538, (1900, 2, 28)), // 1900-02-28
          (-25537, (1900, 3, 1)),  // 1900-03-01 (no Feb 29 in 1900)
          (18627, (2020, 12, 31)),
          (18628, (2021, 1, 1)),
          (0, (1970, 1, 1)),
      ];
      let days: Vec<i32> = cases.iter().map(|c| c.0).collect();
      let y = run(&days, DtField::Year);
      let m = run(&days, DtField::Month);
      let d = run(&days, DtField::Day);
      for (i, (_, (wy, wm, wd))) in cases.iter().enumerate() {
          assert_eq!((y[i], m[i], d[i]), (*wy, *wm, *wd), "case {i}");
      }
  }

  #[test]
  fn dt_n0_is_noop_and_n1_works() {
      let device = MetalDevice::system_default().expect("Metal-capable hardware required");
      let mut empty: Vec<i32> = vec![];
      dispatch_dt_field(&device, &[], &mut empty, DtField::Year).expect("n=0 ok");
      assert!(empty.is_empty());
      let single = run(&[18336], DtField::Year); // 2020-03-15
      assert_eq!(single[0], 2020);
  }
  ```
  (`DtField::{Year,Month,Day}` and `dispatch_dt_field` are defined in Step 3. The CPU reference is the same algorithm spike-verified against Polars — the test pins kernel-vs-reference; the engine-level Polars differential is Task 4.)

- [ ] **Step 2: Run the test, expect FAIL (does not compile — `dt` module missing).**
  Run: `cargo test -p polars-metal-kernels --test test_dt_gregorian -- --test-threads=1`
  Expected: compile error `unresolved import polars_metal_kernels::dt`.

- [ ] **Step 3: Write the MSL kernel.** Create `shaders/dt_gregorian.metal`:
  ```metal
  // shaders/dt_gregorian.metal
  //
  // Branchless gregorian civil-from-days kernel. Extracts the year, month, or
  // day field from a 1-D Int32 column of days-since-1970-01-01 (the physical
  // layout of a Polars `Date`; a `Datetime` is converted to days host-side
  // before dispatch).
  //
  // ## Algorithm — Howard Hinnant `civil_from_days`
  //
  // The settled branchless approach (http://howardhinnant.github.io/date_algorithms.html).
  // Differentially verified against Polars over 11,429 dates (~1860..2079)
  // including pre-1970 negatives and leap/century cases: 0 mismatches. All
  // intermediates fit in Int32 for any representable Polars `Date`.
  //
  // ## Field selector
  //
  //   field == 0 -> year   (Int32)
  //   field == 1 -> month  (Int32; host narrows to Int8)
  //   field == 2 -> day    (Int32; host narrows to Int8)
  //
  // ## Grid
  //
  //   One thread per element; dispatch `n` threads, threadgroup width 256.
  //   The `if (gid >= n) return;` guard exits the partial trailing
  //   threadgroup. No threadgroup memory, no cooperation (element-wise) — the
  //   tile machinery rolling.metal needs does not apply here.
  //
  // ## Scalar parameters
  //
  //   buffer(2): n      — element count
  //   buffer(3): field  — 0=year, 1=month, 2=day
  //
  // NOTE on signed division: MSL integer `/` truncates toward zero. The `era`
  // term handles negatives explicitly (`z >= 0 ? z : z - 146096`), reproducing
  // floor-division for the only place it matters; all other dividends are
  // non-negative ([0,146096] etc.) so truncation == floor there.

  #include <metal_stdlib>
  using namespace metal;

  constant constexpr uint TG_SIZE = 256;

  kernel void dt_field_from_days(
      device const int*  input  [[buffer(0)]],
      device       int*  output [[buffer(1)]],
      constant     uint& n      [[buffer(2)]],
      constant     uint& field  [[buffer(3)]],
      uint gid [[thread_position_in_grid]])
  {
      if (gid >= n) return;

      int z = input[gid] + 719468;
      int era = (z >= 0 ? z : z - 146096) / 146097;
      int doe = z - era * 146097;                                  // [0,146096]
      int yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;   // [0,399]
      int y   = yoe + era * 400;
      int doy = doe - (365*yoe + yoe/4 - yoe/100);                 // [0,365]
      int mp  = (5*doy + 2) / 153;                                 // [0,11]
      int d   = doy - (153*mp + 2)/5 + 1;                          // [1,31]
      int m   = (mp < 10) ? (mp + 3) : (mp - 9);                   // [1,12]
      int year = y + ((m <= 2) ? 1 : 0);

      int result;
      if (field == 0u)      result = year;
      else if (field == 1u) result = m;
      else                  result = d;
      output[gid] = result;
  }
  ```
  (`TG_SIZE` is declared for parity with sibling kernels / readability; the dispatcher chooses the threadgroup width.)

- [ ] **Step 4: Write the Rust dispatch wrapper.** Create `crates/polars-metal-kernels/src/dt.rs`:
  ```rust
  //! Gregorian civil-from-days kernel dispatcher (M6 B3).
  //!
  //! Element-wise extraction of year / month / day from an Int32 column of
  //! days-since-1970. `Date` columns feed their physical i32 directly; the host
  //! converts `Datetime` to days (floor-div by units-per-day) before dispatch.
  //!
  //! The kernel (`shaders/dt_gregorian.metal`) computes every field in Int32;
  //! the host narrows month/day to Int8 and restores the validity bitmap.

  use crate::command::{CommandQueue, DispatchError};
  use crate::shader_lib::{shared_library, ShaderError};
  use polars_metal_buffer::{BufferError, MetalBuffer, MetalDevice};

  /// Threadgroup width — kept in sync with `TG_SIZE` in `shaders/dt_gregorian.metal`.
  pub const TG_SIZE: usize = 256;

  /// Which gregorian field the kernel extracts.
  #[derive(Debug, Clone, Copy, PartialEq, Eq)]
  pub enum DtField {
      Year,
      Month,
      Day,
  }

  impl DtField {
      /// Scalar selector passed to the kernel (`buffer(3)`).
      fn code(self) -> u32 {
          match self {
              DtField::Year => 0,
              DtField::Month => 1,
              DtField::Day => 2,
          }
      }
  }

  /// Errors raised by the gregorian kernel dispatcher.
  #[derive(Debug, thiserror::Error)]
  pub enum DtError {
      #[error("shader library: {0}")]
      Shader(#[from] ShaderError),
      #[error("dispatch: {0}")]
      Dispatch(#[from] DispatchError),
      #[error("buffer: {0}")]
      Buffer(#[from] BufferError),
      #[error("output length {got} does not match input length {expected}")]
      OutputLengthMismatch { got: usize, expected: usize },
      #[error("n_rows {n_rows} exceeds u32::MAX")]
      RowCountOverflow { n_rows: usize },
  }

  /// Core dispatch over pre-staged Int32 `MetalBuffer`s (zero-copy when the
  /// caller staged via `MetalBuffer::from_borrowed_i32` on page-aligned
  /// memory). `input` and `output` are length-`n` Int32 buffers; `output` is
  /// written in place. This is the path the PyO3 `execute_dt` binding uses.
  ///
  /// ## Caller contract
  /// - `input` holds exactly `n * 4` bytes of Int32 days-since-1970.
  /// - `output` holds exactly `n * 4` bytes; fully overwritten.
  /// - `n <= u32::MAX` (enforced).
  ///
  /// ## n == 0
  /// No-op; Metal rejects zero-byte buffers / zero-grid dispatches, so this is
  /// handled on the host without touching Metal.
  pub fn dispatch_dt_field_buf(
      device: &MetalDevice,
      input: &MetalBuffer,
      output: &MetalBuffer,
      n: u32,
      field: DtField,
  ) -> Result<(), DtError> {
      if n == 0 {
          return Ok(());
      }
      let lib = shared_library(device)?;
      let pso = lib.pipeline("dt_field_from_days")?;

      let n_buf = device.new_buffer_from_bytes(&n.to_le_bytes())?;
      let field_buf = device.new_buffer_from_bytes(&field.code().to_le_bytes())?;

      let n_padded = (n as usize).div_ceil(TG_SIZE) * TG_SIZE;
      let mut queue = CommandQueue::new(device)?;
      queue.dispatch_1d_with_tg(
          &pso,
          &[input, output, &n_buf, &field_buf],
          n_padded,
          TG_SIZE,
      )?;
      queue.wait_until_complete()?;
      Ok(())
  }

  /// Test-ergonomics wrapper: stages the caller's slices into Metal buffers,
  /// calls [`dispatch_dt_field_buf`], copies the result back. For the zero-copy
  /// PyO3 path call [`dispatch_dt_field_buf`] directly with pre-staged buffers.
  ///
  /// ## Caller contract
  /// - `output.len() == input.len()`.
  /// - `n <= u32::MAX` (enforced).
  ///
  /// ## n == 0
  /// No-op; `Ok(())` without touching Metal.
  pub fn dispatch_dt_field(
      device: &MetalDevice,
      input: &[i32],
      output: &mut [i32],
      field: DtField,
  ) -> Result<(), DtError> {
      let n = input.len();
      if output.len() != n {
          return Err(DtError::OutputLengthMismatch {
              got: output.len(),
              expected: n,
          });
      }
      if n == 0 {
          return Ok(());
      }
      let n_u32 = u32::try_from(n).map_err(|_| DtError::RowCountOverflow { n_rows: n })?;

      let input_buf = MetalBuffer::from_i32_slice(device, input)?;
      let output_buf = device.new_buffer_zeroed(std::mem::size_of_val(output))?;

      dispatch_dt_field_buf(device, &input_buf, &output_buf, n_u32, field)?;

      let out_vec = output_buf.to_i32_vec();
      output.copy_from_slice(&out_vec[..n]);
      Ok(())
  }
  ```
  (Confirm `MetalBuffer::from_i32_slice` / `to_i32_vec` signatures against `crates/polars-metal-buffer/src/bridge.rs:552` — they were added in B1; `new_buffer_zeroed` and `new_buffer_from_bytes` are on `MetalDevice`. `dispatch_1d_with_tg` / `CommandQueue` / `shared_library` / `pipeline` are the same APIs `rolling.rs` uses — copy the exact call shapes from there if any signature differs.)

- [ ] **Step 5: Register the module.** In `crates/polars-metal-kernels/src/lib.rs`, add `pub mod dt;` in module-declaration order (alphabetical-ish; it sits after `pub mod command;` / before `pub mod fft;` — place to satisfy the `mod`-ordering the gate's lint expects; if clippy complains about order, follow its suggestion):
  ```rust
  pub mod dt;
  ```

- [ ] **Step 6: Run the kernel test, expect PASS.**
  Run: `cargo test -p polars-metal-kernels --test test_dt_gregorian -- --test-threads=1`
  Expected: 3 tests pass.

- [ ] **Step 7: Lint (the gate's stricter `--all-targets -D warnings`).**
  Run: `cargo clippy -p polars-metal-kernels --all-targets -- -D warnings && cargo fmt -p polars-metal-kernels -- --check`
  Expected: clean. (If fmt reports diffs, run `cargo fmt -p polars-metal-kernels` and re-stage.)

- [ ] **Step 8: Commit.**
  ```bash
  git add shaders/dt_gregorian.metal crates/polars-metal-kernels/src/dt.rs crates/polars-metal-kernels/src/lib.rs crates/polars-metal-kernels/tests/test_dt_gregorian.rs
  git commit -m "B3 T1: gregorian civil-from-days MSL kernel + Rust dispatcher + kernel correctness test"
  ```

---

## Task 2 — PyO3 `execute_dt` binding + native registration + wheel rebuild

Expose the kernel to Python as `polars_metal._native.execute_dt`, mirroring `execute_rolling` (raw `(ptr, n)` tuples, `from_borrowed_i32` staging with a copy-back fallback for unaligned `out`). Rebuild the wheel and smoke-test the binding directly.

**Files**
- Modify: `crates/polars-metal-core/src/udf.rs`
- Modify: `crates/polars-metal-core/src/lib.rs`
- Create (test): `tests/python_integration/test_dt_binding.py`

**Steps**

- [ ] **Step 1: Write the failing Python smoke test.** Create `tests/python_integration/test_dt_binding.py`:
  ```python
  """Direct smoke test of the execute_dt native binding (no engine/detect)."""

  import numpy as np

  from polars_metal import _native


  def _hinnant(z: int) -> tuple[int, int, int]:
      z += 719468
      era = (z if z >= 0 else z - 146096) // 146097
      doe = z - era * 146097
      yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
      y = yoe + era * 400
      doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
      mp = (5 * doy + 2) // 153
      d = doy - (153 * mp + 2) // 5 + 1
      m = mp + 3 if mp < 10 else mp - 9
      return (y + (1 if m <= 2 else 0), m, d)


  def test_execute_dt_year_month_day():
      days = np.array([18336, -1, -25567, 0, 10956], dtype=np.int32)  # incl negatives, leap
      for field in (0, 1, 2):  # year, month, day
          inp = np.ascontiguousarray(days, dtype=np.int32)
          out = np.empty(inp.size, dtype=np.int32)
          _native.execute_dt(
              inp=(inp.ctypes.data, inp.size),
              out=(out.ctypes.data, out.size),
              field=field,
          )
          want = np.array([_hinnant(int(z))[field] for z in days], dtype=np.int32)
          np.testing.assert_array_equal(out, want)


  def test_execute_dt_empty_is_noop():
      inp = np.empty(0, dtype=np.int32)
      out = np.empty(0, dtype=np.int32)
      _native.execute_dt(inp=(inp.ctypes.data, 0), out=(out.ctypes.data, 0), field=0)
  ```

- [ ] **Step 2: Run it, expect FAIL (binding missing).**
  Run: `pytest tests/python_integration/test_dt_binding.py -v`
  Expected: FAIL — `AttributeError: module 'polars_metal._native' has no attribute 'execute_dt'`.

- [ ] **Step 3: Implement the binding.** In `crates/polars-metal-core/src/udf.rs`, add after `execute_rolling` (reuse its zero-copy/copy-back structure; the difference is i32 not f32, and the copy-back path reads `outb` back into the caller's slice when `out` is unaligned). Add:
  ```rust
  // ── M6 B3: execute_dt ────────────────────────────────────────────────────────
  //
  // PyO3 entry point dispatching the gregorian civil-from-days kernel over a
  // caller-supplied Int32 days-since-1970 column. Mirrors `execute_rolling`:
  // raw (ptr, n) tuples, `from_borrowed_i32` staging (zero-copy when page-
  // aligned, copy-back fallback for an unaligned output).

  /// PyO3 entry point exposed as `polars_metal._native.execute_dt`.
  ///
  /// # Arguments
  /// * `inp` — `(ptr, n)`: address + element count of a live, C-contiguous
  ///   Int32 days-since-1970 array.
  /// * `out` — `(ptr, n)`: address + element count of a writable C-contiguous
  ///   Int32 array of the same length. Overwritten in place.
  /// * `field` — `0` = year, `1` = month, `2` = day.
  #[pyfunction]
  #[pyo3(signature = (inp, out, field))]
  pub fn execute_dt(inp: (usize, usize), out: (usize, usize), field: u32) -> PyResult<()> {
      use polars_metal_buffer::is_ptr_page_aligned;
      use polars_metal_kernels::dt::{dispatch_dt_field_buf, DtField};

      let device = MetalDevice::system_default().map_err(|e| {
          pyo3::exceptions::PyRuntimeError::new_err(format!(
              "polars_metal: metal device unavailable: {e}"
          ))
      })?;

      let (in_ptr, in_n) = inp;
      let (out_ptr, out_n) = out;
      if in_n != out_n {
          return Err(pyo3::exceptions::PyValueError::new_err(
              "polars_metal: dt input/output length mismatch",
          ));
      }
      if in_n == 0 {
          return Ok(());
      }
      let n = u32::try_from(in_n).map_err(|_| {
          pyo3::exceptions::PyValueError::new_err("polars_metal: dt column exceeds u32::MAX rows")
      })?;

      let dt_field = match field {
          0 => DtField::Year,
          1 => DtField::Month,
          2 => DtField::Day,
          other => {
              return Err(pyo3::exceptions::PyValueError::new_err(format!(
                  "polars_metal: unknown dt field {other}"
              )))
          }
      };

      // SAFETY: `in_ptr` addresses `in_n` live, contiguous i32 values for the
      // whole synchronous call. Page-aligned uses bytesNoCopy (read-only);
      // otherwise copied in. `i32` has no invalid bit patterns.
      let inb = unsafe {
          polars_metal_buffer::MetalBuffer::from_borrowed_i32(&device, in_ptr as *const i32, in_n)
      }
      .map_err(|e| {
          pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: dt input staging: {e}"))
      })?;

      let out_ptr_is_aligned = is_ptr_page_aligned(out_ptr);
      // SAFETY: `out_ptr` addresses `out_n` writable, contiguous i32 values for
      // the whole call. Page-aligned: bytesNoCopy shares host memory. Unaligned:
      // bytes copied in; read back below.
      let outb = unsafe {
          polars_metal_buffer::MetalBuffer::from_borrowed_i32(&device, out_ptr as *const i32, out_n)
      }
      .map_err(|e| {
          pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: dt output staging: {e}"))
      })?;

      dispatch_dt_field_buf(&device, &inb, &outb, n, dt_field).map_err(|e| {
          pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: dt dispatch: {e}"))
      })?;

      // Copy-back fallback for an unaligned output (kernel wrote a GPU-private
      // copy that does not alias the caller's slice).
      if !out_ptr_is_aligned {
          let produced = outb.to_i32_vec();
          // SAFETY: `out_ptr` addresses `out_n` writable i32 for the call; we
          // write exactly `out_n` elements. `produced.len() == out_n`.
          let dst = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut i32, out_n) };
          dst.copy_from_slice(&produced[..out_n]);
      }
      Ok(())
  }
  ```
  (Match the exact `from_borrowed_i32` return/`?`-handling and the `is_ptr_page_aligned` import path against how `execute_rolling` does it — copy its idioms verbatim where they differ. If `outb.to_i32_vec()` is not the available accessor, use the same read-back call `execute_rolling` uses for f32, swapped to i32.)

- [ ] **Step 4: Register the function.** In `crates/polars-metal-core/src/lib.rs`, after the `execute_rolling` registration line:
  ```rust
  m.add_function(wrap_pyfunction!(udf::execute_dt, m)?)?;
  ```

- [ ] **Step 5: Rebuild the wheel** (Rust changed):
  Run: `make wheel`
  Expected: builds and installs editable; no compile/clippy errors.

- [ ] **Step 6: Run the smoke test, expect PASS.**
  Run: `pytest tests/python_integration/test_dt_binding.py -v`
  Expected: both tests pass.

- [ ] **Step 7: Lint.**
  Run: `cargo clippy -p polars-metal-core --all-targets -- -D warnings && cargo fmt -p polars-metal-core -- --check && ruff check tests/python_integration/test_dt_binding.py`
  Expected: clean.

- [ ] **Step 8: Commit.**
  ```bash
  git add crates/polars-metal-core/src/udf.rs crates/polars-metal-core/src/lib.rs tests/python_integration/test_dt_binding.py
  git commit -m "B3 T2: execute_dt PyO3 binding (i32 in/out, zero-copy + copy-back) + native registration"
  ```

---

## Task 3 — `_dt_detect.py`: serialize-detect with an independent `with_columns` patch

Recognize handleable `dt.year/month/day` bindings from the LazyFrame, mirroring `_rolling_detect.py` (O(1) `with_columns`-capture fast path + serialize slow fallback) with its own patch + cache so it chains safely with rolling/vector/fft. Pure detection — no engine call, no GPU.

**Files**
- Create: `python/polars_metal/_dt_detect.py`
- Create (test): `tests/python_integration/test_dt_detect.py`

**Steps**

- [ ] **Step 1: Write the failing detect test.** Create `tests/python_integration/test_dt_detect.py`:
  ```python
  """Unit tests for dt.year/month/day serialize-detection (no engine call)."""

  import datetime

  import polars as pl

  from polars_metal._dt_detect import DtBinding, find_dt_bindings


  def test_detect_date_year_month_day():
      df = pl.DataFrame({"d": [datetime.date(2020, 3, 15)], "v": [1.0]})
      lf = df.lazy().with_columns(
          pl.col("d").dt.year().alias("y"),
          pl.col("d").dt.month().alias("mo"),
          pl.col("d").dt.day().alias("da"),
      )
      got = {(b.field, b.out_name, b.column) for b in find_dt_bindings(lf)}
      assert got == {("year", "y", "d"), ("month", "mo", "d"), ("day", "da", "d")}


  def test_detect_datetime_carries_time_unit():
      s = pl.Series("t", [datetime.datetime(2020, 3, 15, 1, 0)], dtype=pl.Datetime("us"))
      lf = pl.DataFrame({"t": s}).lazy().with_columns(pl.col("t").dt.year().alias("y"))
      bindings = find_dt_bindings(lf)
      assert len(bindings) == 1
      b = bindings[0]
      assert b.field == "year" and b.column == "t"
      assert b.units_per_day == 86_400_000_000  # us
      assert b.is_date is False


  def test_detect_date_has_no_units():
      lf = (
          pl.DataFrame({"d": [datetime.date(2020, 1, 1)]})
          .lazy()
          .with_columns(pl.col("d").dt.day().alias("o"))
      )
      b = find_dt_bindings(lf)[0]
      assert b.is_date is True and b.units_per_day is None


  def test_non_handleable_omitted():
      df = pl.DataFrame(
          {
              "d": [datetime.date(2020, 1, 1)],
              "v": [1.0],
              "t": pl.Series([datetime.datetime(2020, 1, 1)], dtype=pl.Datetime("us", "UTC")),
          }
      )
      # Unsupported accessor, sub-expression input, tz-aware datetime, non-date col.
      lf = df.lazy().with_columns(
          pl.col("d").dt.weekday().alias("wd"),       # unsupported field
          (pl.col("d") + pl.duration(days=1)).dt.year().alias("expr_in"),  # sub-expr input
          pl.col("t").dt.year().alias("tz"),          # tz-aware -> CPU
          (pl.col("v") * 2).alias("plain"),           # not a dt expr
      )
      assert find_dt_bindings(lf) == []


  def test_out_name_shadowing_source_rejected():
      # An output that overwrites a source column the kernel must read -> []
      lf = (
          pl.DataFrame({"d": [datetime.date(2020, 1, 1)]})
          .lazy()
          .with_columns(pl.col("d").dt.year().alias("d"))
      )
      assert find_dt_bindings(lf) == []
  ```

- [ ] **Step 2: Run it, expect FAIL (module missing).**
  Run: `pytest tests/python_integration/test_dt_detect.py -v`
  Expected: collection error `No module named 'polars_metal._dt_detect'`.

- [ ] **Step 3: Implement the detect module.** Create `python/polars_metal/_dt_detect.py`:
  ```python
  """M6 B3: detect native dt.year/month/day bindings from a LazyFrame's
  outermost with_columns layer, for the gregorian custom-kernel path.

  dt.* expressions are NodeTraverser-opaque, so (like rolling/FFT) we inspect
  the pre-optimization serialized plan. Serialized expr shape (py-1.40.1):

    {"Function": {"input": [{"Column": "d"}],
                  "function": {"TemporalExpr": "Year"}}}   // or "Month" / "Day"

  wrapped as {"Alias": [<Function>, "out_name"]}. The time_unit (Datetime) is
  NOT in the expr JSON — it comes from the column's schema dtype.

  Independent with_columns patch + cache (each detector pops only its own
  cache) so this chains safely with the rolling / vector / fft patches.
  """

  from __future__ import annotations

  import json
  import warnings
  from dataclasses import dataclass

  import polars as pl
  import polars.lazyframe.frame as _plf

  _TEMPORAL_FN_MAP = {"Year": "year", "Month": "month", "Day": "day"}

  # Slow-path pre-filter tags (appear in lf.explain()).
  _DT_EXPLAIN_TAGS = (".dt.year(", ".dt.month(", ".dt.day(")

  _UNITS_PER_DAY = {
      "ms": 86_400_000,
      "us": 86_400_000_000,
      "ns": 86_400_000_000_000,
  }

  # ── with_columns expression capture (independent patch + cache) ──────────────
  _dt_lf_exprs_cache: dict[int, list[pl.Expr]] = {}
  _PATCH_ATTR = "_polars_metal_dt_original_with_columns"

  if not hasattr(_plf.LazyFrame, _PATCH_ATTR):
      _orig_wc = _plf.LazyFrame.with_columns
      setattr(_plf.LazyFrame, _PATCH_ATTR, _orig_wc)

      def _patched_wc(self, *exprs, **named):  # type: ignore[no-untyped-def]
          result = _orig_wc(self, *exprs, **named)
          try:
              flat: list[pl.Expr] = [e for e in exprs if isinstance(e, pl.Expr)]
              flat += [e.alias(n) for n, e in named.items() if isinstance(e, pl.Expr)]
              if flat:
                  _dt_lf_exprs_cache[id(result)] = flat
          except Exception:
              pass
          return result

      _plf.LazyFrame.with_columns = _patched_wc  # type: ignore[method-assign]


  @dataclass(frozen=True)
  class DtBinding:
      field: str  # "year" | "month" | "day"
      column: str
      out_name: str
      is_date: bool  # True -> physical i32 days; False -> Datetime i64
      units_per_day: int | None  # None for Date; ms/us/ns count for Datetime


  def _binding_for_column(field: str, col_name: str, col_dtype) -> DtBinding | None:
      """Build a DtBinding from a recognized field + the column's schema dtype,
      or None if the dtype is not a supported temporal type."""
      if col_dtype == pl.Date:
          return DtBinding(field, col_name, "", is_date=True, units_per_day=None)
      if isinstance(col_dtype, pl.Datetime):
          # Time-zone-aware datetimes -> CPU (wall-clock semantics differ).
          if getattr(col_dtype, "time_zone", None) is not None:
              return None
          upd = _UNITS_PER_DAY.get(col_dtype.time_unit)
          if upd is None:
              return None
          return DtBinding(field, col_name, "", is_date=False, units_per_day=upd)
      return None


  def _parse_dt_expr(expr_json: dict, schema: dict) -> DtBinding | None:
      """Return a DtBinding if expr_json is a handleable TemporalExpr Function
      over a bare Column, else None. Never raises."""
      try:
          fn_node = expr_json.get("Function")
          if not isinstance(fn_node, dict):
              return None
          inputs = fn_node.get("input")
          if not isinstance(inputs, list) or len(inputs) != 1:
              return None
          col_node = inputs[0]
          if not isinstance(col_node, dict) or list(col_node.keys()) != ["Column"]:
              return None
          col_name = col_node["Column"]
          if not isinstance(col_name, str):
              return None
          function = fn_node.get("function")
          if not isinstance(function, dict):
              return None
          temporal = function.get("TemporalExpr")
          field = _TEMPORAL_FN_MAP.get(temporal) if isinstance(temporal, str) else None
          if field is None:
              return None
          return _binding_for_column(field, col_name, schema.get(col_name))
      except Exception:
          return None


  def _bindings_from_polars_exprs(exprs: list[pl.Expr], schema: dict) -> list[DtBinding]:
      results: list[DtBinding] = []
      for expr in exprs:
          try:
              with warnings.catch_warnings():
                  warnings.simplefilter("ignore")
                  ser = expr.meta.serialize(format="json")
              expr_json = json.loads(ser)
              inner, out_name = expr_json, ""
              alias = expr_json.get("Alias")
              if isinstance(alias, list) and len(alias) == 2:
                  inner, out_name = alias[0], alias[1]
              if not isinstance(out_name, str):
                  continue
              b = _parse_dt_expr(inner, schema)
              if b is not None:
                  results.append(
                      DtBinding(b.field, b.column, out_name or b.column, b.is_date, b.units_per_day)
                  )
          except Exception:
              continue
      return results


  def find_dt_bindings(lf: pl.LazyFrame) -> list[DtBinding]:
      """Return handleable dt.year/month/day bindings in the outermost
      with_columns layer. Never raises (returns [] on any failure)."""
      try:
          cached = _dt_lf_exprs_cache.pop(id(lf), None)
          if cached is not None:
              schema = dict(lf.collect_schema())
              results = _bindings_from_polars_exprs(cached, schema)
              if results:
                  sources = {b.column for b in results}
                  if any(b.out_name in sources for b in results):
                      return []
                  return results

          # Slow fallback: explain() pre-filter, then serialize + bounded parse.
          with warnings.catch_warnings():
              warnings.simplefilter("ignore", category=UserWarning)
              explain_text = lf.explain()
          if not any(tag in explain_text for tag in _DT_EXPLAIN_TAGS):
              return []

          with warnings.catch_warnings():
              warnings.simplefilter("ignore", category=UserWarning)
              plan_str = lf.serialize(format="json")
          schema = dict(lf.collect_schema())

          exprs_key = '"exprs":['
          idx = plan_str.rfind(exprs_key)
          if idx == -1:
              return []
          start = idx + len(exprs_key) - 1
          opts_idx = plan_str.rfind(',"options":', start)
          if opts_idx == -1:
              return []
          alias_nodes = json.loads(plan_str[start:opts_idx])
          if not isinstance(alias_nodes, list):
              return []

          results = []
          for node in alias_nodes:
              try:
                  if not isinstance(node, dict):
                      continue
                  alias = node.get("Alias")
                  if not isinstance(alias, list) or len(alias) != 2:
                      continue
                  expr_json, out_name = alias[0], alias[1]
                  if not isinstance(out_name, str):
                      continue
                  b = _parse_dt_expr(expr_json, schema)
                  if b is not None:
                      results.append(
                          DtBinding(b.field, b.column, out_name, b.is_date, b.units_per_day)
                      )
              except Exception:
                  continue

          sources = {b.column for b in results}
          if any(b.out_name in sources for b in results):
              return []
          return results
      except Exception:
          return []
  ```

- [ ] **Step 4: Run the detect tests, expect PASS.**
  Run: `pytest tests/python_integration/test_dt_detect.py -v`
  Expected: all 5 tests pass. (If `pl.Datetime.time_unit` access differs, the spike confirmed `col_dtype.time_unit` returns `"ms"`/`"us"`/`"ns"`; `pl.Date` equality and `isinstance(dt, pl.Datetime)` are the correct dtype checks.)

- [ ] **Step 5: Lint.**
  Run: `ruff check python/polars_metal/_dt_detect.py tests/python_integration/test_dt_detect.py`
  Expected: clean.

- [ ] **Step 6: Commit.**
  ```bash
  git add python/polars_metal/_dt_detect.py tests/python_integration/test_dt_detect.py
  git commit -m "B3 T3: _dt_detect — serialize-detect dt.year/month/day (Date + Datetime time_unit), independent with_columns patch"
  ```

---

## Task 4 — `_dt_dispatch.py` + collect hook: end-to-end on the GPU

Compute detected dt bindings on the GPU and stitch them back, then wire the dispatch into the `collect` hook. Datetime is floor-divided to days host-side; month/day are narrowed to Int8; nulls are restored positionally. Lock the exit bar with an engine-level differential matrix that proves the GPU path runs.

**Files**
- Create: `python/polars_metal/_dt_dispatch.py`
- Modify: `python/polars_metal/__init__.py`
- Create (test): `tests/python_integration/test_dt_e2e.py`

**Steps**

- [ ] **Step 1: Write the failing end-to-end test.** Create `tests/python_integration/test_dt_e2e.py`:
  ```python
  """Engine-level differential tests for GPU-accelerated dt.year/month/day."""

  import datetime

  import polars as pl
  import pytest

  from polars_metal import MetalEngine, _native

  _FIELDS = ["year", "month", "day"]
  _OUT_DTYPE = {"year": pl.Int32, "month": pl.Int8, "day": pl.Int8}


  def _dt_dispatches(lf, eng) -> int:
      """Count execute_dt dispatches (proves the GPU kernel path runs)."""
      n = {"c": 0}
      orig = _native.execute_dt

      def cnt(inp, out, field):
          n["c"] += 1
          return orig(inp=inp, out=out, field=field)

      _native.execute_dt = cnt
      try:
          lf.collect(engine=eng)
      finally:
          _native.execute_dt = orig
      return n["c"]


  def _date_range(start: datetime.date, n: int, step: int = 1) -> list[datetime.date]:
      return [start + datetime.timedelta(days=i * step) for i in range(n)]


  @pytest.mark.parametrize("field", _FIELDS)
  def test_date_field_byte_exact_and_gpu(field):
      eng = MetalEngine()
      # Span pre-1970 to 2080, includes leap/century via the wide range.
      dates = _date_range(datetime.date(1900, 1, 1), 2000, step=33)
      df = pl.DataFrame({"d": dates, "v": list(range(len(dates)))})
      expr = getattr(pl.col("d").dt, field)().alias("o")
      lf = df.lazy().with_columns(expr)
      assert _dt_dispatches(lf, eng) == 1, f"{field} should use the GPU kernel"
      got = lf.collect(engine=eng)
      want = lf.collect()
      assert got.equals(want), f"{field}: mismatch"
      assert got["o"].dtype == _OUT_DTYPE[field]


  @pytest.mark.parametrize("tu", ["ms", "us", "ns"])
  @pytest.mark.parametrize("field", _FIELDS)
  def test_datetime_all_time_units(tu, field):
      eng = MetalEngine()
      base = [
          datetime.datetime(2020, 3, 15, 12, 30),
          datetime.datetime(1969, 12, 31, 1, 0),   # pre-epoch (day -1)
          datetime.datetime(2000, 2, 29, 23, 59),  # leap
          datetime.datetime(1970, 1, 1, 0, 0),     # epoch
      ]
      s = pl.Series("t", base, dtype=pl.Datetime(tu))
      lf = pl.DataFrame({"t": s}).lazy().with_columns(getattr(pl.col("t").dt, field)().alias("o"))
      assert _dt_dispatches(lf, eng) == 1
      got, want = lf.collect(engine=eng), lf.collect()
      assert got.equals(want), f"{tu}/{field}: mismatch"
      assert got["o"].dtype == _OUT_DTYPE[field]


  @pytest.mark.parametrize("field", _FIELDS)
  def test_nulls_preserved(field):
      eng = MetalEngine()
      dates = [datetime.date(2020, 3, 15), None, datetime.date(1969, 12, 31), None]
      df = pl.DataFrame({"d": pl.Series("d", dates)})
      lf = df.lazy().with_columns(getattr(pl.col("d").dt, field)().alias("o"))
      got, want = lf.collect(engine=eng), lf.collect()
      assert got.equals(want), f"{field} nulls: mismatch"
      assert got["o"].dtype == _OUT_DTYPE[field]


  def test_empty_frame():
      eng = MetalEngine()
      df = pl.DataFrame({"d": pl.Series("d", [], dtype=pl.Date)})
      lf = df.lazy().with_columns(pl.col("d").dt.year().alias("o"))
      got, want = lf.collect(engine=eng), lf.collect()
      assert got.equals(want)


  def test_multiple_fields_one_collect():
      eng = MetalEngine()
      df = pl.DataFrame({"d": [datetime.date(2020, 3, 15), datetime.date(1999, 7, 4)]})
      lf = df.lazy().with_columns(
          pl.col("d").dt.year().alias("y"),
          pl.col("d").dt.month().alias("mo"),
          pl.col("d").dt.day().alias("da"),
      )
      assert _dt_dispatches(lf, eng) == 3  # one kernel call per field
      got, want = lf.collect(engine=eng), lf.collect()
      assert got.equals(want)


  def test_unsupported_field_falls_back_and_matches():
      eng = MetalEngine()
      df = pl.DataFrame({"d": [datetime.date(2020, 3, 15)]})
      lf = df.lazy().with_columns(pl.col("d").dt.weekday().alias("wd"))
      assert _dt_dispatches(lf, eng) == 0  # not handled -> CPU
      got, want = lf.collect(engine=eng), lf.collect()
      assert got.equals(want)
  ```

- [ ] **Step 2: Run it, expect FAIL.**
  Run: `pytest tests/python_integration/test_dt_e2e.py -v`
  Expected: FAIL — `_dt_dispatches` returns 0 (dt not routed yet; the dispatch module + collect hook don't exist), so the `== 1`/`== 3` assertions fail (the `got.equals(want)` checks may pass via CPU). This RED proves the GPU path is not yet wired.

- [ ] **Step 3: Implement the dispatch module.** Create `python/polars_metal/_dt_dispatch.py`:
  ```python
  """Execute detected dt bindings via the gregorian Metal kernel and stitch
  results onto the collected frame. Collect-and-stitch over whole, materialized
  columns (chunk-safe), mirroring _rolling_dispatch.

  Datetime columns are converted to days-since-1970 host-side via integer
  floor-division (numpy `//` floors toward -inf, matching Polars for pre-epoch
  values); Date columns feed their physical i32 directly. The kernel computes
  every field in Int32; month/day are narrowed to Int8 to match Polars, and
  nulls are restored positionally.
  """

  from __future__ import annotations

  import numpy as np
  import polars as pl

  from polars_metal import _native
  from polars_metal._dt_detect import DtBinding

  _FIELD_CODE = {"year": 0, "month": 1, "day": 2}
  _FIELD_DTYPE = {"year": pl.Int32, "month": pl.Int8, "day": pl.Int8}


  def _dt_series(src: pl.Series, b: DtBinding) -> pl.Series:
      """Run the gregorian kernel on *src* and return a named Series of the
      Polars-matching dtype (Int32 for year, Int8 for month/day), with nulls
      restored positionally."""
      n = src.len()
      out_dtype = _FIELD_DTYPE[b.field]
      if n == 0:
          return pl.Series(b.out_name, [], dtype=out_dtype)

      mask = src.is_not_null()
      # Dense physical days-since-1970 (Int32). Date -> direct; Datetime -> floor-div.
      phys = src.to_physical().fill_null(0).to_numpy()
      if b.is_date:
          days = np.ascontiguousarray(phys, dtype=np.int32)
      else:
          # phys is int64 since-epoch in the unit; floor-div to days (toward -inf).
          days = np.ascontiguousarray((phys // b.units_per_day).astype(np.int32))

      out = np.empty(days.shape[0], dtype=np.int32)
      _native.execute_dt(
          inp=(days.ctypes.data, days.size),
          out=(out.ctypes.data, out.size),
          field=_FIELD_CODE[b.field],
      )

      dense = pl.Series(b.out_name, out, dtype=pl.Int32).cast(out_dtype)
      if src.null_count() == 0:
          return dense
      null_fill = pl.Series(b.out_name, [None] * n, dtype=out_dtype)
      return dense.zip_with(mask, null_fill)


  def apply_dt(lf: pl.LazyFrame, bindings: list[DtBinding], collect_fn) -> pl.DataFrame:
      """Dispatch dt bindings to the Metal kernel and stitch into a DataFrame.

      *collect_fn(rest_lf)* runs the existing collect path on the non-dt
      columns; projection pushdown on ``lf.drop(out_names)`` elides the dt
      computation from the CPU path so each dt column is computed once, on GPU.
      Column order matches the original LazyFrame's schema.
      """
      out_names = [b.out_name for b in bindings]
      order = lf.collect_schema().names()
      rest_lf = lf.drop(out_names)
      df = collect_fn(rest_lf)

      cols: dict[str, pl.Series] = {c: df.get_column(c) for c in df.columns}
      for b in bindings:
          cols[b.out_name] = _dt_series(df.get_column(b.column), b)
      return pl.DataFrame([cols[c] for c in order])
  ```
  (Verify `dense.cast(out_dtype)` and `Series.zip_with(mask, other)` against the spike — both confirmed working: month/day values are in `[1,31]`, well within Int8; `zip_with` takes `dense` where `mask` is True, `null_fill` where False. The `cast` from Int32→Int8 is lossless for the bounded month/day range.)

- [ ] **Step 4: Wire the collect hook.** In `python/polars_metal/__init__.py`:
  - Near the other detect-module imports (around line 13-18, where `_fft_detect`/`_rolling_detect`/`_vector_detect` are imported to install their patches), add:
    ```python
    from polars_metal import _dt_detect as _dt_detect_module  # noqa: F401  (installs with_columns patch)
    ```
  - In `collect_wrapper`, add the dt block AFTER the fft block (around line 297, before the final `return original_collect(...)`):
    ```python
            # M6 B3 dt: serialize-detected dt.year/month/day run on the gregorian
            # Metal kernel via the same collect-and-stitch template. Independent
            # with_columns patch/cache; coexists with rolling/vector/fft.
            from polars_metal import _dt_detect, _dt_dispatch

            dt_bindings = [] if streaming else _dt_detect.find_dt_bindings(self)
            if dt_bindings:

                def _collect_rest_dt(rest_lf: Any) -> Any:
                    return original_collect(rest_lf, engine="cpu", post_opt_callback=cb, **kwargs)

                return _dt_dispatch.apply_dt(self, dt_bindings, _collect_rest_dt)
    ```
    (`streaming` is already computed earlier in the wrapper. Mirror the exact shape of the fft block immediately above it.)

- [ ] **Step 5: Run the e2e tests, expect PASS.**
  Run: `pytest tests/python_integration/test_dt_e2e.py -v`
  Expected: all parametrizations pass (GPU dispatch counts == 1 / == 3, byte-exact, correct dtypes; unsupported field falls back with 0 dispatches).

- [ ] **Step 6: Confirm no detector cross-talk.** Run the sibling detect/e2e suites to ensure the new `with_columns` patch chains cleanly:
  Run: `pytest tests/python_integration/test_rolling_e2e.py tests/python_integration/test_dt_detect.py tests/python_integration/test_dt_binding.py -v`
  Expected: green (rolling still works; patches stack — each pops only its own cache).

- [ ] **Step 7: Lint.**
  Run: `ruff check python/polars_metal/_dt_dispatch.py python/polars_metal/__init__.py tests/python_integration/test_dt_e2e.py`
  Expected: clean.

- [ ] **Step 8: Commit.**
  ```bash
  git add python/polars_metal/_dt_dispatch.py python/polars_metal/__init__.py tests/python_integration/test_dt_e2e.py
  git commit -m "B3 T4: _dt_dispatch + collect hook — dt.year/month/day on GPU end-to-end (Datetime floor-div, Int8 narrow, null restore)"
  ```

---

## Task 5 — Perf sanity check + full `make gate` + conformance no-regression

Confirm the flagship win exists (sanity, not a gate — B4 owns formal baselines) and that B3 introduces no Rust/Python/conformance regression.

**Files**
- None (verification only; a doc note if a baseline shifts).

**Steps**

- [ ] **Step 1: Perf sanity (not gated).** Measure the kernel vs Polars CPU at 10M rows, record the number in the commit message. Run:
  ```bash
  python3 -c "
  import polars as pl, datetime, time
  from polars_metal import MetalEngine
  n = 10_000_000
  base = datetime.date(1970,1,1)
  df = pl.DataFrame({'d': pl.Series('d', range(n), dtype=pl.Int32).cast(pl.Date)})
  eng = MetalEngine()
  lf = df.lazy().with_columns(pl.col('d').dt.year().alias('y'))
  lf.collect(engine=eng)  # warm
  t=time.perf_counter(); lf.collect(engine=eng); gpu=time.perf_counter()-t
  t=time.perf_counter(); lf.collect(); cpu=time.perf_counter()-t
  print(f'gpu={gpu*1e3:.1f}ms cpu={cpu*1e3:.1f}ms speedup={cpu/gpu:.1f}x')
  "
  ```
  Expected: a speedup (target ~30-40×; if it lands lower, e.g. <10×, note it honestly — the collect-and-stitch fold-back overhead is real and B4 will profile it; correctness is the B3 gate, perf is the headline claim to record). Do NOT block on the exact ratio.

- [ ] **Step 2: Full gate.**
  Run: `make gate`
  Expected: green, OR ONLY the documented pre-existing baseline divergences (per MEMORY `m3-conformance-deferrals` / `m6-conformance-fixes`: the F32-mean-returns-F32 case and the known lazyframe/group_by set). If a NEW failure appears, STOP and triage — it is a B3 regression.

- [ ] **Step 3: Record the result.** If green with no new failures and no files changed:
  ```bash
  git commit --allow-empty -m "B3 T5: make gate green — dt gregorian kernel lands with no regression; perf <RECORD speedup>x at 10M"
  ```
  (Substitute the measured speedup from Step 1.)

---

## Self-Review: B3 spec coverage

Mapping each B3 spec item (umbrella spec §B3) to a task:

| B3 spec item | Covered by |
|---|---|
| Recognition via `lf.serialize` + collect-and-stitch (NodeTraverser-opaque dt.*) | T3 (`_dt_detect`, with_columns fast path + serialize fallback) + T4 (collect hook + `apply_dt`) |
| Kernel `shaders/dt_gregorian.metal`, branchless civil-from-days, field selector, threadgroup/grid documented | T1 |
| `tests/kernel/`-style kernel test (per shaders convention; rolling precedent places it at the crate level) | T1 (`crates/polars-metal-kernels/tests/test_dt_gregorian.rs`) |
| Date (i32 days) fed directly | T4 (`_dt_series` `is_date` branch) |
| Datetime (i64, ms/us/ns) → `days = floor_div(value, units_per_day)`, negatives handled | T3 (carries `units_per_day` from schema) + T4 (numpy `//` floor) |
| Output dtypes: year→Int32, month/day→Int8 (narrowed host-side) | T4 (`_FIELD_DTYPE` + `.cast`) + asserted in T4 e2e dtype checks |
| Correctness: epoch, leap/century (2000 leap, 1900 not), pre-1970 negatives, boundaries, all 3 time units, nulls | T1 (kernel vs reference: negatives, leap/century, n=0/1) + T4 (engine differential: all units, nulls, empty, pre-epoch) |
| Perf target ~30-40×, always routes | T4 (no FLOPs gating — always routes when detected) + T5 (perf sanity record) |
| No new public API (native dt.* just gets faster) | T3/T4 (no new verb; recognized natively) |
| No conformance/Rust regression | T5 (`make gate`) |
| time-zone-aware / unsupported fields → CPU fallback | T3 (`_binding_for_column` tz guard; `_TEMPORAL_FN_MAP` admits only year/month/day) + T4 (`test_unsupported_field_falls_back`) |

**Conventions honored:** one MSL kernel per file (`dt_field_from_days` ↔ `dt_gregorian.metal`); kernel test accompanies the shader; `--test-threads=1` on all `cargo test` runs (Metal command-queue contention, MEMORY `polars-metal-test-threading-gotcha`); per-task `ruff` + `cargo clippy --all-targets -D warnings` + `cargo fmt --check` (MEMORY `subagent-driven-fmt-discipline`); differential vs Polars CPU byte-exact incl. dtypes; GPU path proven via the `execute_dt` dispatch counter (mirrors `_reduction_dispatches`); independent `with_columns` patch/cache (mirrors fft/vector); zero-copy i32 staging via B1's `from_borrowed_i32`; no `unwrap`/`expect`/`panic` in non-test Rust (errors via `DtError`/`PyResult`).

**Deliberate MVP choices (documented, not gaps):** the Datetime→days floor-divide is host-side (numpy) rather than in the kernel — trivial O(N) arithmetic vs the branchy civil computation that stays on GPU; a later optimization may push it into the kernel (would need i64-in + explicit MSL floor for negatives). Null-bearing dt columns stay on the GPU with positional mask-restore (better than rolling's CPU-fallback-on-null, because dt is element-wise, not windowed).
