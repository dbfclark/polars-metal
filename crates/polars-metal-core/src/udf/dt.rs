//! `dt.year/month/day` PyO3 entry point: branchless Howard-Hinnant
//! civil-from-days gregorian extraction over an Int32 days-since-1970 column,
//! with a reusable page-aligned input staging pool (B3b).

use polars_metal_buffer::MetalDevice;
use pyo3::prelude::*;
use std::sync::{Mutex, OnceLock};

/// Process-global reusable staging buffer for `execute_dt` inputs (B3b).
/// One buffer, grown to the largest input seen; the `Mutex` serializes dt
/// dispatches (Metal command submission serializes anyway). Designed so other
/// kernel bindings can adopt the same pattern with their own pool later.
static DT_STAGING: OnceLock<Mutex<polars_metal_buffer::StagingPool>> = OnceLock::new();

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

    // Output staging: two paths (see execute_rolling for the rationale).
    let out_ptr_is_aligned = is_ptr_page_aligned(out_ptr);

    // SAFETY: `out_ptr` addresses `out_n` writable, contiguous i32 values
    // for the whole call. Page-aligned: bytesNoCopy shares host memory
    // (writable). Unaligned: bytes are copied in; the MTLBuffer is later
    // read back.
    let outb = unsafe {
        polars_metal_buffer::MetalBuffer::from_borrowed_i32(&device, out_ptr as *const i32, out_n)
    }
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: dt output staging: {e}"))
    })?;

    // Input staging: two paths based on pointer alignment.
    //   Aligned (Datetime at large N): numpy mmap buffer is page-aligned →
    //   zero-copy `bytesNoCopy` path via `from_borrowed_i32`, saving ~1ms at
    //   10M rows.
    //   Unaligned (Date / small N): Arrow 64-byte-aligned only → stage
    //   through the reusable pool (one memcpy into a reused Shared buffer,
    //   B3b). The pool avoids the per-call newBufferWithBytes allocation that
    //   dominated the old path (~5x faster at scale).
    // dispatch is called inside each arm to avoid juggling guard lifetimes.
    let in_aligned = is_ptr_page_aligned(in_ptr);
    if in_aligned {
        // SAFETY: `in_ptr` addresses `in_n` live, contiguous i32 values for
        // the whole synchronous call; page-aligned → `bytesNoCopy` shares
        // host memory (read-only here). `from_borrowed_i32` borrows without
        // copying.
        let inb = unsafe {
            polars_metal_buffer::MetalBuffer::from_borrowed_i32(&device, in_ptr as *const i32, in_n)
        }
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "polars_metal: dt input staging: {e}"
            ))
        })?;
        dispatch_dt_field_buf(&device, &inb, &outb, n, dt_field).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: dt dispatch: {e}"))
        })?;
    } else {
        // Recover from a poisoned lock rather than panic (the buffer is
        // scratch; no invariant is corrupted by a prior panic mid-stage).
        let pool = DT_STAGING.get_or_init(|| Mutex::new(polars_metal_buffer::StagingPool::new()));
        let mut staging = pool.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        // SAFETY: `in_ptr` addresses `in_n` live, contiguous i32 values for
        // the whole synchronous call; reinterpreting as `in_n * 4` bytes is
        // sound (`i32` has no invalid bit patterns) and the slice is only
        // read (memcpy source) before this function returns. `in_n * 4`
        // cannot overflow: the `u32::try_from(in_n)` check above bounds
        // `in_n <= u32::MAX`, so `in_n * 4 <= 4 * (2^32 - 1) < usize::MAX`
        // on 64-bit targets.
        let in_bytes: &[u8] = unsafe { std::slice::from_raw_parts(in_ptr as *const u8, in_n * 4) };
        let inb = staging.stage(&device, in_bytes).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "polars_metal: dt input staging: {e}"
            ))
        })?;
        dispatch_dt_field_buf(&device, inb, &outb, n, dt_field).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: dt dispatch: {e}"))
        })?;
    }

    // Copy-back path: if `out` was not page-aligned, `outb` holds a
    // GPU-private copy of the kernel results. Read it back via `as_slice()`
    // and cast to i32, mirroring execute_rolling's f32 read-back.
    if !out_ptr_is_aligned {
        let out_bytes = outb.as_slice();
        // SAFETY: `out_ptr` is valid for `out_n * 4` bytes (caller contract).
        // We just dispatched into `outb` and the GPU is idle; the result bytes
        // are stable. `i32` has no invalid bit patterns.
        let dst: &mut [i32] = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut i32, out_n) };
        // SAFETY: Metal `StorageModeShared` allocations are page-aligned,
        // satisfying i32's 4-byte alignment requirement; `out_n` i32 occupy
        // exactly `out_bytes.len()` bytes.
        let src_i32: &[i32] =
            unsafe { std::slice::from_raw_parts(out_bytes.as_ptr() as *const i32, out_n) };
        dst.copy_from_slice(src_i32);
    }

    Ok(())
}
