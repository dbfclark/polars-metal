//! M4 Phase 1 Task 10: cumulative scan bindings.
//! M5 rolling Task 1: mlx_shift (forward shift, zero-fill).
//!
//! All four cumulative ops require an `axis` argument. For 1-D arrays, use
//! `axis = 0`. Defaults match Polars conventions (`reverse=false,
//! inclusive=true`).

use crate::array::MlxArrayHandle;
use crate::error::FfiError;
use crate::ffi;

macro_rules! scan_op {
    ($rs:ident, $cpp:ident) => {
        pub fn $rs(a: &MlxArrayHandle, axis: i32) -> Result<MlxArrayHandle, FfiError> {
            let ptr = ffi::$cpp(&a.ptr, axis).map_err(FfiError::from)?;
            Ok(MlxArrayHandle {
                ptr,
                _input_refs: a._input_refs.clone(),
            })
        }
    };
}

scan_op!(mlx_cumsum, mlx_op_cumsum);
scan_op!(mlx_cumprod, mlx_op_cumprod);
scan_op!(mlx_cummax, mlx_op_cummax);
scan_op!(mlx_cummin, mlx_op_cummin);

/// Forward-shift a 1-D F32 array along axis 0 by `shift` positions,
/// zero-filling the vacated front positions.
///
/// Output length equals input length. `shift < 0` is treated as 0;
/// `shift >= n` produces an all-zero result (both clamped on the C++ side).
///
/// Implemented as `pad(a, (shift, 0), 0.0f)[:n]` using MLX's
/// `mlx::core::pad` (pair<int,int> overload) and `mlx::core::slice`
/// (stride-1 overload). API verified against `vendor/mlx/mlx/ops.h`.
///
/// Returns `Err` only if the C++ side throws unexpectedly (e.g. allocation
/// failure); for all normal inputs this succeeds.
pub fn mlx_shift(a: &MlxArrayHandle, shift: i64) -> Result<MlxArrayHandle, FfiError> {
    let ptr = ffi::mlx_shift(&a.ptr, shift).map_err(FfiError::from)?;
    Ok(MlxArrayHandle {
        ptr,
        _input_refs: a._input_refs.clone(),
    })
}
