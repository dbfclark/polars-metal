//! `rolling_{sum,mean,var,std}` PyO3 entry point: tile-blocked, numerically
//! stable F32 rolling-window kernel over a caller-supplied (ptr, n) column,
//! with `from_borrowed_f32` staging (zero-copy when page-aligned).

use polars_metal_buffer::MetalDevice;
use pyo3::prelude::*;

// ── M5 Task 3: execute_rolling ───────────────────────────────────────────────
//
// PyO3 entry point that dispatches the Metal rolling-statistics kernels
// (sum / mean / var / std) over a caller-supplied F32 column. The caller
// passes raw pointer + length tuples (from numpy `ctypes.data` + `size`)
// so the buffer protocol isn't needed. The implementation stages both
// buffers via `MetalBuffer::from_borrowed_f32`:
//
//   - Page-aligned pointers (the common case for numpy arrays) → zero-copy
//     `newBufferWithBytesNoCopy`; GPU writes go directly to host memory and
//     are visible after `wait_until_complete`.
//   - Unaligned pointers → `newBufferWithBytes` (copies in). For the output
//     buffer this means the kernel writes to a GPU-private copy that is NOT
//     reflected back to the caller's slice. We detect this case and use the
//     allocate-and-copy-back fallback so correctness is always preserved.
//
// The Python caller must keep both arrays alive for the duration of this
// synchronous call (trivially satisfied for numpy locals).

/// PyO3 entry point exposed as `polars_metal._native.execute_rolling`.
///
/// # Arguments
/// * `inp` — `(ptr, n_elements)`: address and element count of a live,
///   C-contiguous F32 input array (e.g. `x.ctypes.data, x.size`).
/// * `out` — `(ptr, n_elements)`: address and element count of a writable
///   C-contiguous F32 output array of the same length. Overwritten in place.
/// * `w` — window size (1 ≤ w ≤ 4096).
/// * `op` — operation selector: `0` = sum, `1` = mean, `2` = var, `3` = std.
/// * `ddof` — degrees-of-freedom correction (only used for op 2/3; default 1).
///
/// The first `w-1` output elements are zero-filled (structural nulls). The
/// caller is responsible for setting the Arrow validity bitmap accordingly
/// before presenting the result to Polars.
#[pyfunction]
#[pyo3(signature = (inp, out, w, op, ddof=1))]
pub fn execute_rolling(
    inp: (usize, usize),
    out: (usize, usize),
    w: u32,
    op: u32,
    ddof: u32,
) -> PyResult<()> {
    use polars_metal_buffer::is_ptr_page_aligned;
    use polars_metal_kernels::rolling::{
        dispatch_rolling_sum_f32_buf, dispatch_rolling_var_f32_buf,
    };

    let device = MetalDevice::system_default().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "polars_metal: metal device unavailable: {e}"
        ))
    })?;

    let (in_ptr, in_n) = inp;
    let (out_ptr, out_n) = out;

    if in_n != out_n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "polars_metal: rolling input/output length mismatch",
        ));
    }

    // Fix 1: n=0 is a no-op — staging a 0-byte MTLBuffer would error.
    if in_n == 0 {
        return Ok(());
    }

    // Fix 2: guard against silent truncation on pathological inputs.
    let n = u32::try_from(in_n).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(
            "polars_metal: rolling column exceeds u32::MAX rows",
        )
    })?;

    // Stage input buffer. Zero-copy when page-aligned (numpy arrays are),
    // single copy otherwise. The source array is kept alive by the caller
    // for the duration of this synchronous call.
    //
    // SAFETY: `in_ptr` addresses `in_n` live, contiguous f32 values for the
    // whole call. Page-aligned pointers use bytesNoCopy (read-only from the
    // kernel's perspective); others are copied in. `f32` has no invalid bit
    // patterns.
    let inb = unsafe {
        polars_metal_buffer::MetalBuffer::from_borrowed_f32(&device, in_ptr as *const f32, in_n)
    }
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "polars_metal: rolling input staging: {e}"
        ))
    })?;

    // Output staging: two paths.
    //
    // Zero-copy path (page-aligned `out`): `newBufferWithBytesNoCopy` yields a
    // Shared-storage MTLBuffer that aliases the numpy allocation. The kernel
    // writes through the MTLBuffer and `wait_until_complete` (inside the
    // dispatcher) ensures the writes are visible on the host before we return.
    //
    // Copy-back path (unaligned `out`): `newBufferWithBytes` copies the
    // (zero) initial bytes into a GPU-private allocation. The kernel writes
    // to that private buffer; we must read it back via `as_slice()` and copy
    // into the caller's slice before returning.
    let out_ptr_is_aligned = is_ptr_page_aligned(out_ptr);

    // SAFETY: `out_ptr` addresses `out_n` writable, contiguous f32 values
    // for the whole call. Page-aligned: bytesNoCopy shares host memory
    // (writable). Unaligned: bytes are copied in; the MTLBuffer is later
    // read back.
    let outb = unsafe {
        polars_metal_buffer::MetalBuffer::from_borrowed_f32(&device, out_ptr as *const f32, out_n)
    }
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "polars_metal: rolling output staging: {e}"
        ))
    })?;

    // Dispatch the kernel into `outb`. The dispatcher calls
    // `wait_until_complete` internally before returning.
    let res = match op {
        0 => dispatch_rolling_sum_f32_buf(&device, &inb, &outb, n, w, false),
        1 => dispatch_rolling_sum_f32_buf(&device, &inb, &outb, n, w, true),
        2 => dispatch_rolling_var_f32_buf(&device, &inb, &outb, n, w, ddof, false),
        3 => dispatch_rolling_var_f32_buf(&device, &inb, &outb, n, w, ddof, true),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "polars_metal: unknown rolling op {other}"
            )))
        }
    };
    res.map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: rolling dispatch: {e}"))
    })?;

    // Copy-back path: if `out` was not page-aligned, `outb` holds a
    // GPU-private copy of the kernel results. Read it back into the caller's
    // slice now that the GPU is idle.
    if !out_ptr_is_aligned {
        let out_bytes = outb.as_slice();
        // SAFETY: `out_ptr` is valid for `out_n * 4` bytes (caller contract).
        // We just dispatched into `outb` and the GPU is idle; the result bytes
        // are stable. `f32` has no invalid bit patterns.
        let dst: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_n) };
        // SAFETY: `out_ptr` is valid for `out_n * 4` bytes (caller contract).
        // Metal `StorageModeShared` allocations are page-aligned, satisfying
        // f32's 4-byte alignment requirement; `out_n` f32 occupy exactly
        // `out_bytes.len()` bytes.
        let src_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(out_bytes.as_ptr() as *const f32, out_n) };
        dst.copy_from_slice(src_f32);
    }

    Ok(())
}
