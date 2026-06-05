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
    /// Not supported in MLX 0.22.0; returns `Err(FfiError::Runtime)` if
    /// passed to `mlx_array_view_metal_buffer`. Kept for forward compatibility
    /// when MLX gains F64 support.
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
    pub(crate) _input_refs: Vec<Arc<MetalBuffer>>,
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
/// Returns `FfiError::Runtime` on MLX-side construction throws (e.g.
/// allocation failure), or `FfiError::ConstructionFailed` if the C++
/// returns a null `SharedPtr` without throwing.
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
    let handle =
        unsafe { ffi::mlx_array_from_f32_data(ptr, data.len()) }.map_err(FfiError::from)?;
    if handle.is_null() {
        return Err(FfiError::ConstructionFailed);
    }
    Ok(MlxArrayHandle {
        ptr: handle,
        _input_refs: Vec::new(),
    })
}

/// Construct a 1-D Bool `MlxArrayHandle` from a Rust `&[bool]` slice.
///
/// Each `bool` value is passed to the C++ side as a `u8` byte (Rust guarantees
/// `false == 0u8`, `true == 1u8`). The C++ implementation copies these bytes
/// into MLX-owned memory and constructs an array with `mlx::core::bool_` dtype.
///
/// Empty slices produce a valid zero-element handle.
///
/// # Errors
/// Returns `FfiError::ConstructionFailed` if the C++ side returns a null
/// `SharedPtr` (should not happen under normal conditions, but defensive).
pub fn mlx_array_from_bool_slice(data: &[bool]) -> Result<MlxArrayHandle, FfiError> {
    // SAFETY: `bool` is exactly 1 byte with bit pattern 0 (false) or 1 (true)
    // per the Rust reference. Reinterpreting as `&[u8]` is sound because:
    //   - size_of::<bool>() == size_of::<u8>() == 1
    //   - align_of::<bool>() == align_of::<u8>() == 1
    //   - Every bool value is a valid u8 value (0 or 1)
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len()) };
    let ptr = if data.is_empty() {
        std::ptr::null()
    } else {
        bytes.as_ptr()
    };
    // SAFETY: `ptr` is either a valid pointer to `data.len()` u8 values (each
    // representing a bool: 0=false, non-zero=true) or null with `n == 0`. The
    // C++ side never dereferences a null pointer when `n == 0`.
    let handle =
        unsafe { ffi::mlx_array_from_bool_data(ptr, data.len()) }.map_err(FfiError::from)?;
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

/// Read a materialized I32 array back to a host `Vec<i32>`. Call after `mlx_array_eval`.
///
/// # Errors
/// Returns `FfiError::Runtime` if the copy fails on the C++ side.
pub fn mlx_array_to_i32_vec(handle: &MlxArrayHandle) -> Result<Vec<i32>, FfiError> {
    let n: usize = handle.shape().iter().product();
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut out = vec![0i32; n];
    // SAFETY: `out` has exactly `n` i32 slots; matches the array element count.
    // The array is eval'd (caller contract) and I32 (caller contract), so
    // `arr->data<int32_t>()` is valid for `n` elements.
    unsafe { ffi::mlx_array_copy_to_i32(&handle.ptr, out.as_mut_ptr(), n) };
    Ok(out)
}

/// Copy an eval'd F32 array's contents directly into a caller-owned slice,
/// returning the number of elements written. This is the output-zero-copy
/// readback: the destination is the final buffer (e.g. a numpy output array),
/// so no intermediate `Vec` is allocated.
///
/// `dst` must hold at least `handle.shape().product()` elements; a shorter
/// slice is an error (we never write past `dst`).
///
/// # Errors
/// `FfiError::DtypeMismatch` if the array is not F32; `FfiError::Runtime` if
/// `dst` is too small.
pub fn mlx_array_copy_to_f32_slice(
    handle: &MlxArrayHandle,
    dst: &mut [f32],
) -> Result<usize, FfiError> {
    if !handle.dtype_is_f32() {
        return Err(FfiError::DtypeMismatch);
    }
    let n: usize = handle.shape().iter().product();
    if n == 0 {
        return Ok(0);
    }
    if dst.len() < n {
        return Err(FfiError::Runtime(format!(
            "destination slice too small: have {}, need {n}",
            dst.len()
        )));
    }
    // SAFETY: `dst` holds at least `n` f32 (checked above); the C++ function
    // writes exactly `n * sizeof(float)` bytes. The array is eval'd (caller
    // contract) and F32 (checked), so `arr->data<float>()` is valid.
    unsafe {
        ffi::mlx_array_copy_to_f32(&handle.ptr, dst.as_mut_ptr(), n);
    }
    Ok(n)
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
/// `buf.len() / dtype.element_size()`. A `debug_assert_eq!` enforces this in
/// debug builds; release builds do not check (zero-cost).
///
/// # Errors
/// Returns `FfiError::Runtime` if the C++ side throws (e.g. unknown `dtype`
/// tag causes `std::invalid_argument`; cxx propagates the throw as `Err`).
/// Returns `FfiError::ConstructionFailed` if the C++ bridge returns a null
/// `SharedPtr` without throwing (belt-and-braces; should not occur in practice).
pub fn mlx_array_view_metal_buffer(
    buf: Arc<MetalBuffer>,
    shape: &[i64],
    dtype: MlxDtype,
) -> Result<MlxArrayHandle, FfiError> {
    let expected_bytes: usize = shape.iter().product::<i64>() as usize * dtype.element_size();
    debug_assert_eq!(
        buf.len(),
        expected_bytes,
        "MetalBuffer len ({}) != shape.product ({}) * dtype.element_size ({})",
        buf.len(),
        shape.iter().product::<i64>(),
        dtype.element_size(),
    );

    // Obtain a thin ObjC pointer to the MTL::Buffer object.  This is the same
    // address that metal-cpp uses as `MTL::Buffer*`.  See the SAFETY comment on
    // `MetalBuffer::as_mtl_buffer_raw_ptr` for why the pointer is valid.
    let mtl_ptr = buf.as_mtl_buffer_raw_ptr() as *const u8;

    // SAFETY:
    // - `mtl_ptr` points to a live `MTL::Buffer` (Retained keeps it alive).
    // - `shape` is a valid slice for the duration of the call.
    // - The C++ side uses a no-op Deleter, so it never calls free on this ptr.
    // - `buf` is stored in `_input_refs` below so the buffer outlives the handle.
    let handle = unsafe { ffi::mlx_array_view_mtl_buffer(mtl_ptr, shape, dtype as u32) }
        .map_err(FfiError::from)?;
    if handle.is_null() {
        return Err(FfiError::ConstructionFailed);
    }
    Ok(MlxArrayHandle {
        ptr: handle,
        _input_refs: vec![buf],
    })
}
