//! M6 vector search: shape-manipulation wrappers (transpose/reshape/slice/take_along_axis).
use crate::array::MlxArrayHandle;
use crate::ffi;
use crate::FfiError;

/// Transpose `a` according to `axes` (a permutation of `0..ndim`).
pub fn mlx_transpose(a: &MlxArrayHandle, axes: &[i32]) -> Result<MlxArrayHandle, FfiError> {
    let ptr = ffi::mlx_op_transpose(&a.ptr, axes).map_err(FfiError::from)?;
    Ok(MlxArrayHandle {
        ptr,
        _input_refs: a._input_refs.clone(),
    })
}

/// Reshape `a` to `shape` (total element count must match).
pub fn mlx_reshape(a: &MlxArrayHandle, shape: &[i32]) -> Result<MlxArrayHandle, FfiError> {
    let ptr = ffi::mlx_op_reshape(&a.ptr, shape).map_err(FfiError::from)?;
    Ok(MlxArrayHandle {
        ptr,
        _input_refs: a._input_refs.clone(),
    })
}

/// Slice `a` with per-axis `start`/`stop`/`strides` (NumPy-style half-open).
pub fn mlx_slice(
    a: &MlxArrayHandle,
    start: &[i32],
    stop: &[i32],
    strides: &[i32],
) -> Result<MlxArrayHandle, FfiError> {
    let ptr = ffi::mlx_op_slice(&a.ptr, start, stop, strides).map_err(FfiError::from)?;
    Ok(MlxArrayHandle {
        ptr,
        _input_refs: a._input_refs.clone(),
    })
}

/// Gather along `axis`: `out[i,j] = a[i, indices[i,j]]` for `axis=1`.
pub fn mlx_take_along_axis(
    a: &MlxArrayHandle,
    indices: &MlxArrayHandle,
    axis: i32,
) -> Result<MlxArrayHandle, FfiError> {
    let ptr = ffi::mlx_op_take_along_axis(&a.ptr, &indices.ptr, axis).map_err(FfiError::from)?;
    let mut refs = a._input_refs.clone();
    refs.extend(indices._input_refs.iter().cloned());
    Ok(MlxArrayHandle {
        ptr,
        _input_refs: refs,
    })
}
