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
