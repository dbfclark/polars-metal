//! M6 A3: 1-D FFT over a whole column, run on the hand-rolled MSL FFT kernel.
//! The kernel handles all sizes on-GPU (pow2 four-step/recursive, mixed-radix
//! composites, Bluestein for primes / non-smooth), replacing the MLX FFT FFI
//! (which was correct only to 2^20).
//!
//! `fft_core` stages inputs as planar re/im buffers directly — no host
//! interleave/split. Real input gets a zeroed imaginary plane. The planar
//! kernel (`fft_gpu_planar_core`) reads/writes separate re/im MTLBuffers, so
//! the result comes back planar with no host gather.

use polars_metal_buffer::{MetalBuffer, MetalDevice};
use polars_metal_mlx_sys::FfiError;

/// Input to `fft_core`: a real F32 signal, or a complex signal as two F32 streams.
pub enum FftInput<'a> {
    Real(&'a [f32]),
    Complex(&'a [f32], &'a [f32]),
}

/// Run a 1-D FFT (or inverse) over the whole signal. Returns `(real_out, imag_out)`,
/// each length `n`, row order = bin order (matches numpy.fft).
///
/// Inputs are staged as planar re/im MTLBuffers — no host interleave/split.
/// Real input receives a zeroed imaginary plane. The planar kernel handles all
/// sizes on-GPU (pow2 four-step/recursive, mixed-radix, Bluestein).
pub fn fft_core(
    input: FftInput<'_>,
    n: i64,
    inverse: bool,
) -> Result<(Vec<f32>, Vec<f32>), FfiError> {
    let len = n as usize;
    let device = MetalDevice::system_default()
        .map_err(|e| FfiError::Runtime(format!("metal device unavailable: {e}")))?;

    // Stage planar inputs directly — no host interleave.
    // Real input gets a zeroed imaginary plane (zero imaginary input is correct
    // for a real signal). The planar kernel reads/writes separate re/im buffers,
    // so the result comes back planar with no host split.
    let (re_in, im_in) = match input {
        FftInput::Real(re) => {
            let re_buf = MetalBuffer::from_f32_slice(&device, re)
                .map_err(|e| FfiError::Runtime(format!("fft re staging: {e}")))?;
            let im_buf = device
                .new_buffer_zeroed(len * std::mem::size_of::<f32>())
                .map_err(|e| FfiError::Runtime(format!("fft im staging: {e}")))?;
            (re_buf, im_buf)
        }
        FftInput::Complex(re, im) => {
            let re_buf = MetalBuffer::from_f32_slice(&device, re)
                .map_err(|e| FfiError::Runtime(format!("fft re staging: {e}")))?;
            let im_buf = MetalBuffer::from_f32_slice(&device, im)
                .map_err(|e| FfiError::Runtime(format!("fft im staging: {e}")))?;
            (re_buf, im_buf)
        }
    };
    let re_out = device
        .new_buffer_zeroed(len * std::mem::size_of::<f32>())
        .map_err(|e| FfiError::Runtime(format!("fft re_out: {e}")))?;
    let im_out = device
        .new_buffer_zeroed(len * std::mem::size_of::<f32>())
        .map_err(|e| FfiError::Runtime(format!("fft im_out: {e}")))?;
    polars_metal_kernels::fft::fft_gpu_planar_core(
        &device, &re_in, &im_in, &re_out, &im_out, n, inverse,
    )
    .map_err(|e| FfiError::Runtime(format!("fft kernel: {e}")))?;
    Ok((re_out.to_f32_vec(), im_out.to_f32_vec()))
}

use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// PyO3 entry: `_native.execute_fft(real, imag, n, inverse)`.
/// `real` is `(ptr, len)` of a contiguous F32 stream; `imag` is `Some((ptr,len))` for complex
/// input (struct column) or `None` for a real signal. Returns the real and imaginary streams as
/// raw little-endian f32 bytes; the Python layer reconstructs them with np.frombuffer.
#[pyfunction]
#[pyo3(signature = (real, imag, n, inverse))]
pub fn execute_fft(
    py: Python<'_>,
    real: (usize, usize),
    imag: Option<(usize, usize)>,
    n: i64,
    inverse: bool,
) -> PyResult<(Bound<'_, PyBytes>, Bound<'_, PyBytes>)> {
    let (rptr, rlen) = real;
    // Guard: the signal length must match the declared n (Python supplies n independently; a
    // mismatch would make the FFT kernel read past the buffer). Reject clearly instead of risking UB.
    if rlen != n as usize {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "fft: real stream len {rlen} != n {n}"
        )));
    }
    if let Some((_, ilen)) = imag {
        if ilen != n as usize {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "fft: imag stream len {ilen} != n {n}"
            )));
        }
    }
    // SAFETY: Python guarantees these point to contiguous F32 arrays of the given lengths, kept
    // alive (rechunked Series / numpy arrays) for this synchronous call. Read-only; `f32` has no
    // invalid bit patterns. Mirrors `vector_search::execute_vector_search`.
    let rslice = unsafe { std::slice::from_raw_parts(rptr as *const f32, rlen) };
    let result = match imag {
        None => fft_core(FftInput::Real(rslice), n, inverse),
        Some((iptr, ilen)) => {
            let islice = unsafe { std::slice::from_raw_parts(iptr as *const f32, ilen) };
            fft_core(FftInput::Complex(rslice, islice), n, inverse)
        }
    };
    let (re_out, im_out) =
        result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("fft: {e}")))?;
    // SAFETY: `f32` is plain-old-data with no padding and no invalid bit patterns; reinterpreting
    // a &[f32] as &[u8] of 4x the length is sound and read-only. The slices do not outlive
    // `re_out` / `im_out`, which are alive for the duration of `PyBytes::new_bound` (which copies).
    let re_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(re_out.as_ptr() as *const u8, re_out.len() * 4) };
    let im_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(im_out.as_ptr() as *const u8, im_out.len() * 4) };
    Ok((
        PyBytes::new_bound(py, re_bytes),
        PyBytes::new_bound(py, im_bytes),
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn fft_of_dc_signal_is_single_bin() {
        // FFT of a constant signal [2,2,2,2]: bin0 = sum = 8 (real), all other bins 0.
        let sig = [2.0f32, 2.0, 2.0, 2.0];
        let (re, im) = fft_core(FftInput::Real(&sig), 4, /*inverse=*/ false).unwrap();
        assert_eq!(re.len(), 4);
        assert_eq!(im.len(), 4);
        assert!((re[0] - 8.0).abs() < 1e-4, "bin0 real = {}", re[0]);
        for k in 1..4 {
            assert!(re[k].abs() < 1e-4 && im[k].abs() < 1e-4, "bin {k} not ~0");
        }
    }

    #[test]
    fn fft_then_ifft_round_trips_via_complex_input() {
        // Forward FFT a real signal, then inverse-FFT the complex result; recover the signal.
        let sig = [1.0f32, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0];
        let (fre, fim) = fft_core(FftInput::Real(&sig), 8, false).unwrap();
        let (ire, _iim) = fft_core(FftInput::Complex(&fre, &fim), 8, true).unwrap();
        for (a, b) in ire.iter().zip(sig.iter()) {
            assert!((a - b).abs() < 1e-4, "round-trip differs: {a} vs {b}");
        }
    }
}
