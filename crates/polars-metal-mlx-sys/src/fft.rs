//! M4 Phase 1 Task 10: 1-D FFT bindings.
//!
//! `mlx_fft(a)` and `mlx_ifft(a)` produce complex64 arrays (interleaved
//! real / imag F32 pairs in memory). For readback as separate F32 streams,
//! use `mlx_real(c)` and `mlx_imag(c)` which return F32 arrays.

use crate::array::MlxArrayHandle;
use crate::error::FfiError;
use crate::ffi;

/// 1-D FFT over the last axis. Input may be F32 or complex64; output is
/// always complex64. The result requires `mlx_real`/`mlx_imag` for F32
/// readback.
pub fn mlx_fft(a: &MlxArrayHandle) -> Result<MlxArrayHandle, FfiError> {
    let ptr = ffi::mlx_op_fft_1d(&a.ptr).map_err(FfiError::from)?;
    Ok(MlxArrayHandle {
        ptr,
        _input_refs: a._input_refs.clone(),
    })
}

/// 1-D inverse FFT over the last axis.
pub fn mlx_ifft(a: &MlxArrayHandle) -> Result<MlxArrayHandle, FfiError> {
    let ptr = ffi::mlx_op_ifft_1d(&a.ptr).map_err(FfiError::from)?;
    Ok(MlxArrayHandle {
        ptr,
        _input_refs: a._input_refs.clone(),
    })
}

/// Extract the real part of a complex array as an F32 array.
pub fn mlx_real(a: &MlxArrayHandle) -> Result<MlxArrayHandle, FfiError> {
    let ptr = ffi::mlx_op_real(&a.ptr).map_err(FfiError::from)?;
    Ok(MlxArrayHandle {
        ptr,
        _input_refs: a._input_refs.clone(),
    })
}

/// Extract the imaginary part of a complex array as an F32 array.
pub fn mlx_imag(a: &MlxArrayHandle) -> Result<MlxArrayHandle, FfiError> {
    let ptr = ffi::mlx_op_imag(&a.ptr).map_err(FfiError::from)?;
    Ok(MlxArrayHandle {
        ptr,
        _input_refs: a._input_refs.clone(),
    })
}
