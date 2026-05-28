//! M4 Phase 1 Task 10: cumulative scan bindings.
//!
//! All four ops require an `axis` argument. For 1-D arrays, use `axis = 0`.
//! Defaults match Polars conventions (`reverse=false, inclusive=true`).

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
