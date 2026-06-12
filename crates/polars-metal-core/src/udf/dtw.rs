//! `.metal.dtw` PyO3 entry point: custom MSL DTW kernel (one threadgroup per
//! pair, Euclidean local cost, optional Sakoe-Chiba band) over a pair-major
//! F32 query matrix against a broadcast reference sequence.

use polars_metal_buffer::MetalDevice;
use pyo3::prelude::*;

// ── M6 A4: execute_dtw ───────────────────────────────────────────────────────

/// PyO3 entry exposed as `polars_metal._native.execute_dtw`.
///
/// * `queries` — `(ptr, n_pairs*seq_len)`: pair-major F32 query matrix.
/// * `reference` — `(ptr, seq_len)`: the broadcast reference sequence.
/// * `out` — `(ptr, n_pairs)`: writable F32 output (overwritten in place).
/// * `n_pairs`, `seq_len` — dimensions.
/// * `window` — Sakoe-Chiba radius; negative => unconstrained full DTW.
///
/// Staging mirrors `execute_rolling`: `from_borrowed_f32` (zero-copy when
/// page-aligned; copy-back for an unaligned output). The caller keeps all
/// arrays alive for the synchronous call.
#[pyfunction]
#[pyo3(signature = (queries, reference, out, n_pairs, seq_len, window))]
pub fn execute_dtw(
    queries: (usize, usize),
    reference: (usize, usize),
    out: (usize, usize),
    n_pairs: usize,
    seq_len: usize,
    window: i32,
) -> PyResult<()> {
    use polars_metal_buffer::is_ptr_page_aligned;
    use polars_metal_kernels::dtw::dispatch_dtw_buf;

    let device = MetalDevice::system_default().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "polars_metal: metal device unavailable: {e}"
        ))
    })?;

    let (q_ptr, q_n) = queries;
    let (r_ptr, r_n) = reference;
    let (out_ptr, out_n) = out;

    if r_n != seq_len || out_n != n_pairs || q_n != n_pairs.saturating_mul(seq_len) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "polars_metal: dtw dimension mismatch",
        ));
    }
    if n_pairs == 0 || seq_len == 0 {
        return Ok(());
    }
    let n_u32 = u32::try_from(n_pairs).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("polars_metal: dtw n_pairs exceeds u32::MAX")
    })?;
    let l_u32 = u32::try_from(seq_len).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("polars_metal: dtw seq_len exceeds u32::MAX")
    })?;

    // SAFETY: each ptr addresses its stated count of live, contiguous f32 for
    // the whole call; page-aligned pointers use bytesNoCopy, others copy in.
    let qb = unsafe {
        polars_metal_buffer::MetalBuffer::from_borrowed_f32(&device, q_ptr as *const f32, q_n)
    }
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: dtw query staging: {e}"))
    })?;
    let rb = unsafe {
        polars_metal_buffer::MetalBuffer::from_borrowed_f32(&device, r_ptr as *const f32, r_n)
    }
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: dtw ref staging: {e}"))
    })?;
    let out_aligned = is_ptr_page_aligned(out_ptr);
    // SAFETY: out_ptr addresses out_n writable contiguous f32 for the call.
    let ob = unsafe {
        polars_metal_buffer::MetalBuffer::from_borrowed_f32(&device, out_ptr as *const f32, out_n)
    }
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: dtw out staging: {e}"))
    })?;

    dispatch_dtw_buf(&device, &qb, &rb, &ob, n_u32, l_u32, window).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: dtw dispatch: {e}"))
    })?;

    if !out_aligned {
        let out_bytes = ob.as_slice();
        // SAFETY: out_ptr valid for out_n*4 bytes (caller contract); GPU idle.
        let dst: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_n) };
        // SAFETY: Shared allocations are page-aligned (≥4-byte); out_n f32
        // occupy exactly out_bytes.len() bytes; f32 has no invalid patterns.
        let src: &[f32] =
            unsafe { std::slice::from_raw_parts(out_bytes.as_ptr() as *const f32, out_n) };
        dst.copy_from_slice(src);
    }
    Ok(())
}
