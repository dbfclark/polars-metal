//! M4 Phase 1 Task 10: matmul binding.
//!
//! `mlx_matmul(a, b)` wraps `mlx::core::matmul`. Standard NumPy-like broadcast
//! rules apply: `(M, K) @ (K, N) -> (M, N)`, with broadcasting on leading
//! dimensions.

use crate::array::MlxArrayHandle;
use crate::error::FfiError;
use crate::ffi;

pub fn mlx_matmul(a: &MlxArrayHandle, b: &MlxArrayHandle) -> Result<MlxArrayHandle, FfiError> {
    let mut refs = a._input_refs.clone();
    refs.extend(b._input_refs.iter().cloned());
    let ptr = ffi::mlx_op_matmul(&a.ptr, &b.ptr).map_err(FfiError::from)?;
    Ok(MlxArrayHandle {
        ptr,
        _input_refs: refs,
    })
}
