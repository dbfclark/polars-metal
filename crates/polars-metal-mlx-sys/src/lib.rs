// crates/polars-metal-mlx-sys/src/lib.rs

// Allow unwrap() in tests; forbid it in production code.
#![cfg_attr(not(test), forbid(clippy::unwrap_used))]

//! FFI bindings to MLX C++.
//!
//! In M0 this crate exposes one trivial operation (`add_one` for the cxx
//! smoke test) and `add_f32` (real MLX elementwise add) to validate the FFI
//! strategy end to end. Real op coverage arrives in later milestones.

mod error;
pub use error::FfiError;

pub mod array;
pub mod elementwise;
pub mod matmul;
pub mod reduce;
pub mod scan;
pub mod shape;
pub mod sort;

// cxx's SharedPtr<T> implementation expands a panic! macro in the generated
// Rust glue (inside SharedPtr::is_null()'s unreachable branch). This is
// internal to the cxx crate and cannot be suppressed at the call site.
// Allow clippy::panic for this module only.
#[allow(clippy::panic)]
#[cxx::bridge(namespace = "polars_metal_mlx")]
mod ffi {
    unsafe extern "C++" {
        include!("polars-metal-mlx-sys/cxx/mlx_bridge.h");

        fn add_one(x: i64) -> i64;
        fn add_f32(a: &CxxVector<f32>, b: &CxxVector<f32>) -> Result<UniquePtr<CxxVector<f32>>>;
        fn add_f32_on_gpu(
            a: &CxxVector<f32>,
            b: &CxxVector<f32>,
        ) -> Result<UniquePtr<CxxVector<f32>>>;
        // Slice-based cumsum: cxx maps `&[u8]` / `&mut [u32]` to
        // `rust::Slice<const uint8_t>` / `rust::Slice<uint32_t>`, which are
        // thin pointer+length pairs. This eliminates the per-element
        // `CxxVector` push/iter loops on both sides of the FFI. The internal
        // MLX `array(ptr, shape, dtype)` constructor still copies the input
        // into MLX-owned memory (one memcpy) and we memcpy the result back
        // into the caller's output slice — those are the only data-touching
        // passes left in this call. T30 Step 3 / `docs/open-questions.md`.
        fn cumsum_u8_to_u32(input: &[u8], output: &mut [u32]) -> Result<()>;

        // M4 Phase 1: array construction + eval + readback.
        //
        // MlxArray is a type alias for mlx::core::array on the C++ side,
        // exposed here as an opaque cxx type. All access is via SharedPtr
        // so the MLX refcount manages lifetime (drop is refcount decrement).
        type MlxArray;

        // Construct a 1-D F32 array from a raw pointer + length. The MLX
        // `array(ptr, shape, dtype)` constructor copies the input bytes into
        // MLX-owned memory (one memcpy). Returns a null SharedPtr on failure.
        // SAFETY: `data` must point to at least `n` valid f32 values.
        unsafe fn mlx_array_from_f32_data(
            data: *const f32,
            n: usize,
        ) -> Result<SharedPtr<MlxArray>>;

        // Return the shape of `arr` as a Vec<u64>. Wraps `arr->shape()`.
        fn mlx_array_shape(arr: &SharedPtr<MlxArray>) -> Vec<u64>;

        // Return true iff `arr`'s dtype is mlx::core::float32.
        fn mlx_array_is_f32(arr: &SharedPtr<MlxArray>) -> bool;

        // Copy `n` f32 values from the materialized array into the
        // caller-provided buffer. Must be called after `mlx_array_eval_one`.
        // SAFETY: `out` must point to a buffer of at least `n` f32 values.
        unsafe fn mlx_array_copy_to_f32(arr: &SharedPtr<MlxArray>, out: *mut f32, n: usize);

        // M6 vector search: I32 readback. Copy `n` i32 values from the
        // materialized (eval'd) array into the caller buffer. Array must be I32.
        // SAFETY: `out` must point to a buffer of at least `n` i32 values.
        unsafe fn mlx_array_copy_to_i32(arr: &SharedPtr<MlxArray>, out: *mut i32, n: usize);

        // M6 Track B (B1): integer dtype query + per-width readback.
        //
        // Return the MlxDtype tag of `arr`'s dtype (0=f32, 2=i32, 3=bool,
        // 4=i8, 5=i16, 6=i64, 7=u8, 8=u16, 9=u32, 10=u64). Throws on an
        // unmapped dtype (e.g. float64), which cxx surfaces as Err.
        fn mlx_array_dtype(arr: &SharedPtr<MlxArray>) -> Result<u32>;

        // Per-width integer readback. Each copies `n` values of the matching
        // width into the caller buffer. Array must be eval'd and have the
        // matching dtype (caller contract).
        // SAFETY: `out` must point to a buffer of at least `n` elements.
        unsafe fn mlx_array_copy_to_i8(arr: &SharedPtr<MlxArray>, out: *mut i8, n: usize);
        unsafe fn mlx_array_copy_to_i16(arr: &SharedPtr<MlxArray>, out: *mut i16, n: usize);
        unsafe fn mlx_array_copy_to_i64(arr: &SharedPtr<MlxArray>, out: *mut i64, n: usize);
        unsafe fn mlx_array_copy_to_u8(arr: &SharedPtr<MlxArray>, out: *mut u8, n: usize);
        unsafe fn mlx_array_copy_to_u16(arr: &SharedPtr<MlxArray>, out: *mut u16, n: usize);
        unsafe fn mlx_array_copy_to_u32(arr: &SharedPtr<MlxArray>, out: *mut u32, n: usize);
        unsafe fn mlx_array_copy_to_u64(arr: &SharedPtr<MlxArray>, out: *mut u64, n: usize);

        // Force evaluation (materialize) of a single array. Wraps
        // `mlx::core::eval(*arr)`. Returns Err on any MLX exception.
        fn mlx_array_eval_one(arr: &SharedPtr<MlxArray>) -> Result<()>;

        // Construct a zero-copy view of an existing MTL::Buffer as an MLX array.
        //
        // `mtl_ptr` must be a valid `MTL::Buffer*` cast to `*const u8` (cxx maps
        // `*const u8` cleanly; we use it as an opaque pointer carrier — the C++
        // side casts it back to `const void*` before wrapping in
        // `mlx::core::allocator::Buffer`).  `shape` specifies the array
        // dimensions; their product must equal the element count implied by the
        // buffer's byte length and `dtype`. `dtype` is the `MlxDtype` tag cast to
        // `u32` (0=F32, 1=F64, 2=I32, 3=Bool).
        //
        // MLX is given a no-op Deleter so it never tries to free the buffer; the
        // Rust side (via `_input_refs` in `MlxArrayHandle`) holds the keep-alive.
        //
        // SAFETY: `mtl_ptr` must remain valid for the lifetime of every
        // `MlxArrayHandle` that was built from it (enforced by `_input_refs`).
        unsafe fn mlx_array_view_mtl_buffer(
            mtl_ptr: *const u8,
            shape: &[i64],
            dtype: u32,
        ) -> Result<SharedPtr<MlxArray>>;

        // M4 Phase 1 Task 6: elementwise op bindings.
        // Each takes one or more SharedPtr<MlxArray> args and returns a fresh
        // SharedPtr<MlxArray> representing the graph node (lazy; eval to materialize).
        // Operations throw on dtype/shape mismatch, which propagates via Result<>.

        fn mlx_op_add(
            a: &SharedPtr<MlxArray>,
            b: &SharedPtr<MlxArray>,
        ) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_sub(
            a: &SharedPtr<MlxArray>,
            b: &SharedPtr<MlxArray>,
        ) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_mul(
            a: &SharedPtr<MlxArray>,
            b: &SharedPtr<MlxArray>,
        ) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_div(
            a: &SharedPtr<MlxArray>,
            b: &SharedPtr<MlxArray>,
        ) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_mod(
            a: &SharedPtr<MlxArray>,
            b: &SharedPtr<MlxArray>,
        ) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_pow(
            a: &SharedPtr<MlxArray>,
            b: &SharedPtr<MlxArray>,
        ) -> Result<SharedPtr<MlxArray>>;

        fn mlx_op_eq(
            a: &SharedPtr<MlxArray>,
            b: &SharedPtr<MlxArray>,
        ) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_ne(
            a: &SharedPtr<MlxArray>,
            b: &SharedPtr<MlxArray>,
        ) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_lt(
            a: &SharedPtr<MlxArray>,
            b: &SharedPtr<MlxArray>,
        ) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_le(
            a: &SharedPtr<MlxArray>,
            b: &SharedPtr<MlxArray>,
        ) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_gt(
            a: &SharedPtr<MlxArray>,
            b: &SharedPtr<MlxArray>,
        ) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_ge(
            a: &SharedPtr<MlxArray>,
            b: &SharedPtr<MlxArray>,
        ) -> Result<SharedPtr<MlxArray>>;

        fn mlx_op_logical_and(
            a: &SharedPtr<MlxArray>,
            b: &SharedPtr<MlxArray>,
        ) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_logical_or(
            a: &SharedPtr<MlxArray>,
            b: &SharedPtr<MlxArray>,
        ) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_logical_not(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;

        fn mlx_op_neg(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_abs(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_square(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;

        fn mlx_op_where(
            cond: &SharedPtr<MlxArray>,
            then_v: &SharedPtr<MlxArray>,
            else_v: &SharedPtr<MlxArray>,
        ) -> Result<SharedPtr<MlxArray>>;

        // SAFETY: `data` must point to at least `n` valid u8 values (each representing
        // a bool: 0=false, non-zero=true), or be null when `n == 0`.
        unsafe fn mlx_array_from_bool_data(
            data: *const u8,
            n: usize,
        ) -> Result<SharedPtr<MlxArray>>;

        // M4 Phase 1 Task 7: transcendentals + roots + rounding + atan2 + cast.

        fn mlx_op_sin(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_cos(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_tan(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_sinh(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_cosh(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_tanh(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_asin(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_acos(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_atan(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_log(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_log2(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_log10(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_log1p(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_exp(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_exp2(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_sqrt(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_cbrt(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_floor(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_ceil(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_round(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;

        fn mlx_op_atan2(
            a: &SharedPtr<MlxArray>,
            b: &SharedPtr<MlxArray>,
        ) -> Result<SharedPtr<MlxArray>>;

        fn mlx_op_cast(a: &SharedPtr<MlxArray>, dtype: u32) -> Result<SharedPtr<MlxArray>>;

        // M4 Phase 1 Task 8: reduction bindings.

        fn mlx_op_sum_all(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_mean_all(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_min_all(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_max_all(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_std_all(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_var_all(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_argmin_all(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_argmax_all(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;

        fn mlx_op_sum_axis(a: &SharedPtr<MlxArray>, axis: i32) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_mean_axis(a: &SharedPtr<MlxArray>, axis: i32) -> Result<SharedPtr<MlxArray>>;

        // M4 Phase 1 Task 9: sort + argpartition.

        fn mlx_op_sort(a: &SharedPtr<MlxArray>) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_argpartition(a: &SharedPtr<MlxArray>, kth: i32) -> Result<SharedPtr<MlxArray>>;

        // M6 vector search: axis-aware argpartition (per-row top-k). Unlike the
        // flattening `mlx_op_argpartition` above, this preserves the input shape
        // and partitions along `axis` (use -1 for the last axis).
        fn mlx_op_argpartition_axis(
            a: &SharedPtr<MlxArray>,
            kth: i32,
            axis: i32,
        ) -> Result<SharedPtr<MlxArray>>;

        // M5 rolling Task 1: forward shift (zero-fill).
        //
        // Shifts array `a` forward by `shift` positions along axis 0, prepending
        // zeros. Output length == input length. `shift` is clamped to [0, n] on
        // the C++ side. Infallible by construction; Result<> is kept for
        // consistency with the rest of the bridge so callers use the same `?`
        // idiom.
        fn mlx_shift(a: &SharedPtr<MlxArray>, shift: i64) -> Result<SharedPtr<MlxArray>>;

        // M5 rolling Task 4b: row-index (iota) generator.
        //
        // Produces a 1-D F32 array [0.0, 1.0, …, n-1.0] via
        // mlx::core::arange. n <= 0 yields an empty array. Takes no input
        // MlxArray; Result<> kept for `?`-idiom consistency.
        fn mlx_iota_f32(n: i64) -> Result<SharedPtr<MlxArray>>;

        // M4 Phase 1 Task 10: cumulative scans + matmul.

        fn mlx_op_cumsum(a: &SharedPtr<MlxArray>, axis: i32) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_cumprod(a: &SharedPtr<MlxArray>, axis: i32) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_cummax(a: &SharedPtr<MlxArray>, axis: i32) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_cummin(a: &SharedPtr<MlxArray>, axis: i32) -> Result<SharedPtr<MlxArray>>;

        fn mlx_op_matmul(
            a: &SharedPtr<MlxArray>,
            b: &SharedPtr<MlxArray>,
        ) -> Result<SharedPtr<MlxArray>>;

        // M6 vector search: shape ops (transpose/reshape/slice/take_along_axis).
        fn mlx_op_transpose(a: &SharedPtr<MlxArray>, axes: &[i32]) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_reshape(a: &SharedPtr<MlxArray>, shape: &[i32]) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_slice(
            a: &SharedPtr<MlxArray>,
            start: &[i32],
            stop: &[i32],
            strides: &[i32],
        ) -> Result<SharedPtr<MlxArray>>;
        fn mlx_op_take_along_axis(
            a: &SharedPtr<MlxArray>,
            indices: &SharedPtr<MlxArray>,
            axis: i32,
        ) -> Result<SharedPtr<MlxArray>>;

        fn mlx_op_take(
            a: &SharedPtr<MlxArray>,
            indices: &SharedPtr<MlxArray>,
        ) -> Result<SharedPtr<MlxArray>>;
    }
}

pub fn add_one(x: i64) -> i64 {
    ffi::add_one(x)
}

/// Elementwise add forced to the Metal GPU device.
///
/// Uses MLX's `StreamContext` RAII to switch the default device to
/// `Device::gpu` for the duration of the call. If Metal is unavailable on
/// this host (e.g. the linked `libmlx.a` was built without
/// `-DMLX_BUILD_METAL=ON`), MLX throws `std::invalid_argument` and this
/// function returns `Err`. Callers can use this to gate GPU-only code paths.
pub fn add_f32_on_gpu(a: &[f32], b: &[f32]) -> Result<Vec<f32>, FfiError> {
    if a.len() != b.len() {
        return Err(FfiError::ShapeMismatch {
            lhs: a.len(),
            rhs: b.len(),
        });
    }
    let mut va = cxx::CxxVector::new();
    let mut vb = cxx::CxxVector::new();
    for &x in a {
        va.pin_mut().push(x);
    }
    for &x in b {
        vb.pin_mut().push(x);
    }
    let result = ffi::add_f32_on_gpu(&va, &vb).map_err(FfiError::from)?;
    Ok(result.iter().copied().collect())
}

pub fn add_f32(a: &[f32], b: &[f32]) -> Result<Vec<f32>, FfiError> {
    if a.len() != b.len() {
        return Err(FfiError::ShapeMismatch {
            lhs: a.len(),
            rhs: b.len(),
        });
    }
    let mut va = cxx::CxxVector::new();
    let mut vb = cxx::CxxVector::new();
    for &x in a {
        va.pin_mut().push(x);
    }
    for &x in b {
        vb.pin_mut().push(x);
    }
    let result = ffi::add_f32(&va, &vb).map_err(FfiError::from)?;
    Ok(result.iter().copied().collect())
}

/// Inclusive cumulative sum over a u8 keep-flag input, writing u32 offsets
/// to `output`. The u32 codomain is wide enough for ~4B-row inputs.
///
/// Forces MLX onto `Device::gpu` via `StreamContext` on the C++ side. Empty
/// input is short-circuited and never enters MLX. Input/output length must
/// match or `FfiError::ShapeMismatch` is returned.
///
/// Used by the M1 filter compaction pipeline: bit-packed predicate column ->
/// dense u8 keep-flags -> cumsum -> scatter indices.
///
/// FFI shape: the cxx bridge passes `input` / `output` as `rust::Slice`s —
/// thin pointer+length pairs — so there is no per-element marshalling on
/// either side of the call. The MLX `array(ptr, shape, dtype)` constructor
/// still copies the input bytes into MLX-managed memory and we memcpy the
/// result back into `output`; eliminating those two memcpys requires
/// dropping down to MLX's lower-level Metal-buffer API and is deferred to
/// the M2 MLX-FFI revisit (see `docs/open-questions.md`).
pub fn cumsum_u8_to_u32(input: &[u8], output: &mut [u32]) -> Result<(), FfiError> {
    if input.len() != output.len() {
        return Err(FfiError::ShapeMismatch {
            lhs: input.len(),
            rhs: output.len(),
        });
    }
    if input.is_empty() {
        return Ok(());
    }
    ffi::cumsum_u8_to_u32(input, output).map_err(FfiError::from)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn add_one_smoke() {
        assert_eq!(add_one(41), 42);
    }

    #[test]
    fn add_f32_on_gpu_dispatches_to_metal() {
        // 4096 elements: large enough that MLX would route to GPU even without
        // explicit forcing. We force explicitly via StreamContext so that a
        // broken Metal toolchain produces a hard error rather than a silent
        // CPU fallback.
        let len = 4096;
        let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..len).map(|i| i as f32 * -0.25).collect();
        let result = add_f32_on_gpu(&a, &b).expect("Metal GPU dispatch should work on this host");
        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        assert_eq!(result.len(), expected.len());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-4, "GPU result {r} != CPU expected {e}");
        }
    }

    proptest! {
        #[test]
        fn add_f32_matches_rust(
            len in 0usize..256,
            seed_a: u64,
            seed_b: u64,
        ) {
            let a: Vec<f32> = (0..len)
                .map(|i| ((seed_a.wrapping_add(i as u64) & 0xffff) as f32) / 1024.0)
                .collect();
            let b: Vec<f32> = (0..len)
                .map(|i| ((seed_b.wrapping_add(i as u64) & 0xffff) as f32) / 1024.0)
                .collect();

            let result = add_f32(&a, &b).expect("add_f32 must succeed for same-length inputs");
            let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

            prop_assert_eq!(result.len(), expected.len());
            for (r, e) in result.iter().zip(expected.iter()) {
                prop_assert!((r - e).abs() < 1e-5);
            }
        }
    }
}
