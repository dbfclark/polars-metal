// crates/polars-metal-mlx-sys/src/array.rs
//! MLX array construction, eval, and readback bindings.
//!
//! `MlxArrayHandle` wraps a `cxx::SharedPtr<mlx::core::array>`; clone is cheap
//! (refcount increment), and MLX's refcount drives drop (refcount decrement).
//!
//! # Construction path
//! `mlx_array_from_f32_slice` is a COPY path: the MLX `array(ptr, shape, dtype)`
//! constructor copies the input bytes into MLX-owned memory. A zero-copy
//! MTLBuffer view is added in Task 5.
//!
//! # Eval
//! `mlx_array_eval` accepts a slice of handles and evaluates them one at a time.
//! True batch eval via `mlx::core::eval(vector<array>)` is deferred: cxx's
//! `SharedPtr<T>` cannot be placed in a `CxxVector`, so batching requires a
//! bespoke C++ helper that takes raw pointers. One-at-a-time is correct and
//! sufficient for Task 4; the perf cost is an extra kernel-dispatch overhead
//! per array, which only matters when evaluating tens-of-arrays tight loops.

use cxx::SharedPtr;

use crate::error::FfiError;
use crate::ffi;

/// A ref-counted handle to an `mlx::core::array`.
///
/// Clone is O(1) (shared-pointer refcount bump). The underlying array is freed
/// when the last handle drops.
#[derive(Clone)]
pub struct MlxArrayHandle(pub(crate) SharedPtr<ffi::MlxArray>);

impl MlxArrayHandle {
    /// Return the shape of the array as a `Vec<usize>`.
    pub fn shape(&self) -> Vec<usize> {
        ffi::mlx_array_shape(&self.0)
            .into_iter()
            .map(|x| x as usize)
            .collect()
    }

    /// Return `true` iff the array's dtype is `float32`.
    pub fn dtype_is_f32(&self) -> bool {
        ffi::mlx_array_is_f32(&self.0)
    }
}

/// Construct a 1-D F32 `MlxArrayHandle` from a Rust slice.
///
/// The input bytes are **copied** into MLX-owned memory by the MLX array
/// constructor. Empty slices produce a valid, zero-element handle.
///
/// # Errors
/// Returns `FfiError::ConstructionFailed` if the C++ side returns a null
/// `SharedPtr` (should not happen under normal conditions, but defensive).
pub fn mlx_array_from_f32_slice(data: &[f32]) -> Result<MlxArrayHandle, FfiError> {
    // Empty slice: pass null so the invariant is self-contained in Rust.
    // The C++ bridge already short-circuits on `n == 0` and never dereferences
    // the pointer in that case (see `mlx_array_from_f32_data` in mlx_bridge.cc).
    let ptr = if data.is_empty() {
        std::ptr::null()
    } else {
        data.as_ptr()
    };
    // SAFETY: `ptr` is either a valid pointer to at least `data.len()` f32
    // values, or `std::ptr::null()` with `n == 0`. The C++ side never
    // dereferences a null pointer when `n == 0`.
    let handle = unsafe { ffi::mlx_array_from_f32_data(ptr, data.len()) };
    if handle.is_null() {
        return Err(FfiError::ConstructionFailed);
    }
    Ok(MlxArrayHandle(handle))
}

/// Force evaluation (materialization) of each handle in `handles`.
///
/// Iterates one at a time. See module-level doc for why batch eval is deferred.
///
/// # Errors
/// Returns the first `FfiError` encountered (wrapping the MLX exception).
pub fn mlx_array_eval(handles: &[MlxArrayHandle]) -> Result<(), FfiError> {
    for h in handles {
        ffi::mlx_array_eval_one(&h.0).map_err(FfiError::from)?;
    }
    Ok(())
}

/// Copy the materialized values out of `handle` into a new `Vec<f32>`.
///
/// Must be called after [`mlx_array_eval`] (or equivalent). Returns an empty
/// `Vec` for a zero-element array without touching the C++ side.
///
/// # Errors
/// Returns `FfiError::DtypeMismatch` if the array's dtype is not F32.
/// Returns `FfiError::Runtime` if the copy fails on the C++ side.
pub fn mlx_array_to_f32_vec(handle: &MlxArrayHandle) -> Result<Vec<f32>, FfiError> {
    if !handle.dtype_is_f32() {
        return Err(FfiError::DtypeMismatch);
    }
    let n: usize = handle.shape().iter().product();
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut out = vec![0.0_f32; n];
    // SAFETY: `out.as_mut_ptr()` points to a live allocation of exactly `n`
    // f32 values. The C++ function writes exactly `n * sizeof(float)` bytes
    // into that buffer via `std::memcpy`. The array has been eval'd (caller
    // contract), so `arr->data<float>()` is valid. The dtype check above
    // guarantees the underlying buffer is correctly typed as float32.
    unsafe {
        ffi::mlx_array_copy_to_f32(&handle.0, out.as_mut_ptr(), n);
    }
    Ok(out)
}
