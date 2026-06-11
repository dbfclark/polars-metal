//! M4 Phase 1 Task 9: sort + argpartition bindings.
//!
//! `mlx_sort` returns a sorted copy of the input (ascending).
//! `mlx_argpartition` returns I32 indices partitioned at the `kth` position
//! (the `kth+1` smallest elements occupy positions `0..=kth` in some order).
//!
//! Use argpartition with `mlx_neg` for top-K (largest) and slice the first
//! K indices.
//!
//! MLX 0.22.0 `sort` is NOT stable — equal elements may appear in any
//! relative order. The engine's analyzer must reject Polars sort operations
//! that require stability and fall back to CPU.

use crate::array::MlxArrayHandle;
use crate::error::FfiError;
use crate::ffi;

/// Sort an array ascending. Output dtype matches input.
///
/// # Errors
/// Returns `FfiError::Runtime` if MLX rejects the input (e.g. unsupported dtype).
pub fn mlx_sort(a: &MlxArrayHandle) -> Result<MlxArrayHandle, FfiError> {
    let ptr = ffi::mlx_op_sort(&a.ptr).map_err(FfiError::from)?;
    Ok(MlxArrayHandle {
        ptr,
        _input_refs: a._input_refs.clone(),
    })
}

/// Argpartition: returns I32 indices such that positions `0..=kth` hold the
/// `kth+1` smallest elements (unordered among themselves) and positions
/// `kth+1..` hold the rest.
///
/// For top-K largest: apply `mlx_neg` first.
///
/// # Errors
/// Returns `FfiError::Runtime` if `kth` is out of range or MLX rejects.
pub fn mlx_argpartition(a: &MlxArrayHandle, kth: i32) -> Result<MlxArrayHandle, FfiError> {
    let ptr = ffi::mlx_op_argpartition(&a.ptr, kth).map_err(FfiError::from)?;
    Ok(MlxArrayHandle {
        ptr,
        _input_refs: a._input_refs.clone(),
    })
}

/// Argpartition along `axis` (use `-1` for the last axis). Returns integer indices,
/// same shape as `a`, with the `0..=kth` positions along `axis` holding the kth-smallest.
///
/// Unlike [`mlx_argpartition`], this preserves the input shape (per-row top-k)
/// rather than flattening to 1-D.
///
/// # Errors
/// Returns `FfiError::Runtime` if `kth`/`axis` is out of range or MLX rejects.
pub fn mlx_argpartition_axis(
    a: &MlxArrayHandle,
    kth: i32,
    axis: i32,
) -> Result<MlxArrayHandle, FfiError> {
    let ptr = ffi::mlx_op_argpartition_axis(&a.ptr, kth, axis).map_err(FfiError::from)?;
    Ok(MlxArrayHandle {
        ptr,
        _input_refs: a._input_refs.clone(),
    })
}
