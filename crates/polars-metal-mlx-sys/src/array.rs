// crates/polars-metal-mlx-sys/src/array.rs
//! MLX array construction, eval, and readback bindings.
//!
//! `MlxArrayHandle` wraps a `cxx::SharedPtr<mlx::core::array>`; clone is cheap
//! (refcount increment), and MLX's refcount drives drop (refcount decrement).
//!
//! # Construction paths
//! - `mlx_array_from_f32_slice` — COPY path: the MLX `array(ptr, shape, dtype)`
//!   constructor copies the input bytes into MLX-owned memory.
//! - `mlx_array_view_metal_buffer` — ZERO-COPY path (Task 5): wraps an existing
//!   `MetalBuffer` via `mlx::core::allocator::Buffer` + a no-op Deleter. The
//!   `Arc<MetalBuffer>` is stashed in `_input_refs` so the buffer cannot be
//!   freed while MLX holds the pointer.
//!
//! # Eval
//! `mlx_array_eval` accepts a slice of handles and evaluates them one at a time.
//! True batch eval via `mlx::core::eval(vector<array>)` is deferred: cxx's
//! `SharedPtr<T>` cannot be placed in a `CxxVector`, so batching requires a
//! bespoke C++ helper that takes raw pointers. One-at-a-time is correct and
//! sufficient for Tasks 4–5; the perf cost is an extra kernel-dispatch overhead
//! per array, which only matters when evaluating tens-of-arrays tight loops.

use std::sync::Arc;

use cxx::SharedPtr;

use polars_metal_buffer::MetalBuffer;

use crate::error::FfiError;
use crate::ffi;

/// Element dtype tag passed through the FFI boundary as a `u32`.
///
/// The integer values must match the `switch (dtype)` in
/// `mlx_bridge.cc::mlx_array_view_mtl_buffer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MlxDtype {
    F32 = 0,
    F64 = 1,
    I32 = 2,
    Bool = 3,
}

impl MlxDtype {
    /// Size of one element in bytes.
    pub fn element_size(self) -> usize {
        match self {
            MlxDtype::F32 | MlxDtype::I32 => 4,
            MlxDtype::F64 => 8,
            MlxDtype::Bool => 1,
        }
    }
}

/// A ref-counted handle to an `mlx::core::array`.
///
/// Clone is O(1) (shared-pointer refcount bump). The underlying array is freed
/// when the last handle drops.
///
/// `_input_refs` keeps `Arc<MetalBuffer>` instances alive for the lifetime of
/// view-based handles (zero-copy path). It is empty for copy-path handles.
/// `Vec<Arc<MetalBuffer>>` is `Clone` because `Arc<MetalBuffer>` is `Clone`.
#[derive(Clone)]
pub struct MlxArrayHandle {
    pub(crate) ptr: SharedPtr<ffi::MlxArray>,
    /// Keep-alives for any `MetalBuffer`s this handle views into (zero-copy
    /// path). Empty for copy-path handles. Cloning shares the Arcs.
    _input_refs: Vec<Arc<MetalBuffer>>,
}

impl MlxArrayHandle {
    /// Return the shape of the array as a `Vec<usize>`.
    pub fn shape(&self) -> Vec<usize> {
        ffi::mlx_array_shape(&self.ptr)
            .into_iter()
            .map(|x| x as usize)
            .collect()
    }

    /// Return `true` iff the array's dtype is `float32`.
    pub fn dtype_is_f32(&self) -> bool {
        ffi::mlx_array_is_f32(&self.ptr)
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
    Ok(MlxArrayHandle {
        ptr: handle,
        _input_refs: Vec::new(),
    })
}

/// Force evaluation (materialization) of each handle in `handles`.
///
/// Iterates one at a time. See module-level doc for why batch eval is deferred.
///
/// # Errors
/// Returns the first `FfiError` encountered (wrapping the MLX exception).
pub fn mlx_array_eval(handles: &[MlxArrayHandle]) -> Result<(), FfiError> {
    for h in handles {
        ffi::mlx_array_eval_one(&h.ptr).map_err(FfiError::from)?;
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
        ffi::mlx_array_copy_to_f32(&handle.ptr, out.as_mut_ptr(), n);
    }
    Ok(out)
}

/// Construct a zero-copy MLX array view over an existing `MetalBuffer`.
///
/// The buffer's contents are exposed to MLX without copying: MLX receives the
/// `MTL::Buffer*` pointer wrapped in `mlx::core::allocator::Buffer`, and the
/// array is given a no-op Deleter so MLX never tries to free it.
///
/// `buf` is cloned into `MlxArrayHandle::_input_refs` so the buffer stays
/// alive at least as long as the returned handle (and any clones of it).
///
/// `shape` must be consistent with the element count implied by
/// `buf.len() / dtype.element_size()`. This is not checked here; an
/// inconsistency would produce undefined behaviour on the C++ side.
///
/// # Errors
/// Returns `FfiError::ConstructionFailed` if the C++ bridge returns a null
/// `SharedPtr`, which happens when an unknown `dtype` tag is passed (throws
/// `std::invalid_argument` on the C++ side, which cxx converts to an error).
pub fn mlx_array_view_metal_buffer(
    buf: Arc<MetalBuffer>,
    shape: &[i64],
    dtype: MlxDtype,
) -> Result<MlxArrayHandle, FfiError> {
    // Obtain a thin ObjC pointer to the MTL::Buffer object.  This is the same
    // address that metal-cpp uses as `MTL::Buffer*`.  The cast is safe because
    // `ProtocolObject<dyn MTLBuffer>` is a zero-sized newtype around the ObjC
    // `id`; a reference to it IS the instance pointer.
    let mtl_ptr = buf.as_mtl_buffer_raw_ptr() as *const u8;

    // SAFETY:
    // - `mtl_ptr` points to a live `MTL::Buffer` (Retained keeps it alive).
    // - `shape` is a valid slice for the duration of the call.
    // - The C++ side uses a no-op Deleter, so it never calls free on this ptr.
    // - `buf` is stored in `_input_refs` below so the buffer outlives the handle.
    let handle = unsafe { ffi::mlx_array_view_mtl_buffer(mtl_ptr, shape, dtype as u32) };
    if handle.is_null() {
        return Err(FfiError::ConstructionFailed);
    }
    Ok(MlxArrayHandle {
        ptr: handle,
        _input_refs: vec![buf],
    })
}
