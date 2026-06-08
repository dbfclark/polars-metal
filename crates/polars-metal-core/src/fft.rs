//! M6 A3: 1-D FFT over a whole column, composed from MLX FFT FFI.
use std::sync::Arc;

use polars_metal_buffer::{MetalBuffer, MetalDevice};
use polars_metal_mlx_sys::array::{
    mlx_array_eval, mlx_array_to_f32_vec, mlx_array_view_metal_buffer, MlxArrayHandle, MlxDtype,
};
use polars_metal_mlx_sys::fft::{mlx_complex, mlx_fft, mlx_ifft, mlx_imag, mlx_real};
use polars_metal_mlx_sys::FfiError;

/// Input to `fft_core`: a real F32 signal, or a complex signal as two F32 streams.
// Task 4 wires these to the `execute_fft` PyO3 binding; allow until then.
#[allow(dead_code)]
pub enum FftInput<'a> {
    Real(&'a [f32]),
    // Task 4: Complex input path used by execute_fft for pre-split real/imag columns.
    Complex(&'a [f32], &'a [f32]),
}

/// View a host F32 slice as a 1-D `(n,)` MLX array (mirrors `vector_search::view2d`).
// Task 4: used by execute_fft; allow until the pyfunction registration lands.
#[allow(dead_code)]
fn view1d(data: &[f32], n: i64) -> Result<MlxArrayHandle, FfiError> {
    let device = MetalDevice::system_default()
        .map_err(|e| FfiError::Runtime(format!("metal device unavailable: {e}")))?;
    // SAFETY: `data` outlives every use of the returned handle within this fn's callers,
    // which eval and read back before returning. MetalBuffer borrows, does not own.
    // `f32` has no invalid bit patterns.
    let buf = unsafe { MetalBuffer::from_borrowed_f32(&device, data.as_ptr(), data.len()) }
        .map(Arc::new)
        .map_err(|e| FfiError::Runtime(format!("metal buffer staging: {e}")))?;
    mlx_array_view_metal_buffer(buf, &[n], MlxDtype::F32)
}

/// Run a 1-D FFT (or inverse) over the whole signal. Returns `(real_out, imag_out)`,
/// each length `n`, row order = MLX bin order (matches numpy.fft).
// Task 4: execute_fft calls this from PyO3; allow until then.
#[allow(dead_code)]
pub fn fft_core(
    input: FftInput<'_>,
    n: i64,
    inverse: bool,
) -> Result<(Vec<f32>, Vec<f32>), FfiError> {
    let arr = match input {
        FftInput::Real(re) => view1d(re, n)?,
        FftInput::Complex(re, im) => {
            let r = view1d(re, n)?;
            let i = view1d(im, n)?;
            mlx_complex(&r, &i)?
        }
    };
    let transformed = if inverse { mlx_ifft(&arr)? } else { mlx_fft(&arr)? };
    let re_out = mlx_real(&transformed)?;
    let im_out = mlx_imag(&transformed)?;
    mlx_array_eval(&[re_out.clone(), im_out.clone()])?;
    Ok((mlx_array_to_f32_vec(&re_out)?, mlx_array_to_f32_vec(&im_out)?))
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
