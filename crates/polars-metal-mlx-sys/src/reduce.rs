//! M4 Phase 1 Task 8: reduction bindings.
//!
//! Global reductions collapse a multi-element array to a single scalar:
//!   sum, mean, min, max, std, var, argmin, argmax
//!
//! Per-axis variants reduce along one dimension:
//!   sum_axis, mean_axis
//!
//! `std` and `var` use MLX's default `ddof=0` (population variance). Polars
//! defaults to sample variance (`ddof=1`); the analyzer / engine layer
//! handles the Bessel correction (n / (n-1) factor) when needed.
//!
//! `argmin` / `argmax` return I32 arrays containing indices. Readback via
//! `mlx_array_to_f32_vec` returns `FfiError::DtypeMismatch`; callers must
//! cast to F32 via `mlx_cast` first.

use crate::array::MlxArrayHandle;
use crate::error::FfiError;
use crate::ffi;

macro_rules! global_reduce {
    ($rs:ident, $cpp:ident) => {
        pub fn $rs(a: &MlxArrayHandle) -> Result<MlxArrayHandle, FfiError> {
            let ptr = ffi::$cpp(&a.ptr).map_err(FfiError::from)?;
            Ok(MlxArrayHandle {
                ptr,
                _input_refs: a._input_refs.clone(),
            })
        }
    };
}

global_reduce!(mlx_sum, mlx_op_sum_all);
global_reduce!(mlx_mean, mlx_op_mean_all);
global_reduce!(mlx_min, mlx_op_min_all);
global_reduce!(mlx_max, mlx_op_max_all);
global_reduce!(mlx_std, mlx_op_std_all);
global_reduce!(mlx_var, mlx_op_var_all);
global_reduce!(mlx_argmin, mlx_op_argmin_all);
global_reduce!(mlx_argmax, mlx_op_argmax_all);

/// Reduce along a specific axis. Result has rank `a.ndim() - 1`.
pub fn mlx_sum_axis(a: &MlxArrayHandle, axis: i32) -> Result<MlxArrayHandle, FfiError> {
    let ptr = ffi::mlx_op_sum_axis(&a.ptr, axis).map_err(FfiError::from)?;
    Ok(MlxArrayHandle {
        ptr,
        _input_refs: a._input_refs.clone(),
    })
}

/// Mean along a specific axis. Result has rank `a.ndim() - 1`.
pub fn mlx_mean_axis(a: &MlxArrayHandle, axis: i32) -> Result<MlxArrayHandle, FfiError> {
    let ptr = ffi::mlx_op_mean_axis(&a.ptr, axis).map_err(FfiError::from)?;
    Ok(MlxArrayHandle {
        ptr,
        _input_refs: a._input_refs.clone(),
    })
}
