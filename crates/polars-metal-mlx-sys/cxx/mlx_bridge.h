// crates/polars-metal-mlx-sys/cxx/mlx_bridge.h
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "rust/cxx.h"
#include "mlx/array.h"

namespace polars_metal_mlx {

// MlxArray is the opaque type exposed through the cxx::bridge.
// cxx requires that the type appear in the same namespace as the functions;
// we define it here as a simple struct that wraps mlx::core::array, using
// inheritance so that all mlx::core::array member functions are available
// in the implementation without extra casting.
struct MlxArray : mlx::core::array {
    using mlx::core::array::array;
};

// Smoke-test from the cxx hello-world (kept for regression).
int64_t add_one(int64_t x);

// Elementwise addition of two f32 arrays. Returns the result as a
// heap-allocated std::vector<float> via unique_ptr (required by cxx when
// the Rust declaration uses UniquePtr<CxxVector<f32>>).
// Throws std::runtime_error on shape mismatch or any MLX-side failure;
// cxx converts that to a Rust error.
std::unique_ptr<std::vector<float>> add_f32(
    const std::vector<float>& a, const std::vector<float>& b);

// Same elementwise add as add_f32, but explicitly forces execution on the
// Metal GPU device via StreamContext RAII. If Metal is unavailable on this
// host (e.g. libmlx.a built without -DMLX_BUILD_METAL=ON), this throws
// std::invalid_argument so the caller learns loudly rather than silently
// falling back to CPU.
std::unique_ptr<std::vector<float>> add_f32_on_gpu(
    const std::vector<float>& a, const std::vector<float>& b);

// Inclusive cumulative sum over a uint8 keep-flag column, producing uint32
// output offsets. Forces execution on Device::gpu via StreamContext (same
// pattern as add_f32_on_gpu). The u32 output domain is sized so that a
// 4B-row input cannot overflow; callers in the filter compaction pipeline
// can read the final element as the total kept-row count.
//
// `input` and `output` are passed as `rust::Slice`s (thin pointer+length
// pairs) so there is no per-element marshalling. The input is copied once
// into MLX-managed storage by the `array(ptr, shape, dtype)` constructor;
// the scan result is memcpy'd once into the caller's output buffer.
//
// Caller must ensure non-empty input. Empty input is short-circuited on the
// Rust side and does not call into this function. Throws std::runtime_error
// on shape mismatch (cxx converts that to a Rust error).
void cumsum_u8_to_u32(
    rust::Slice<const uint8_t> input, rust::Slice<uint32_t> output);

// ── M4 Phase 1: MlxArray handle ──────────────────────────────────────────────
//
// MlxArray is a type alias for mlx::core::array (defined in the .cc file after
// the MLX headers are available). We expose it through shared_ptr so the MLX
// refcount drives lifetime on both sides of the FFI boundary.

// Construct a 1-D float32 MlxArray from a raw pointer + element count.
// Copies `n * sizeof(float)` bytes into MLX-owned memory. Caller must ensure
// `data` points to at least `n` valid floats. `n == 0` is allowed.
// Returns an empty shared_ptr on failure (should not occur under normal use;
// the Rust wrapper checks for null and returns FfiError::ConstructionFailed).
std::shared_ptr<MlxArray> mlx_array_from_f32_data(const float* data, size_t n);

// Return the shape of `arr` as a rust::Vec<uint64_t>.
rust::Vec<uint64_t> mlx_array_shape(const std::shared_ptr<MlxArray>& arr);

// Return true iff arr->dtype() == mlx::core::float32.
bool mlx_array_is_f32(const std::shared_ptr<MlxArray>& arr);

// Copy `n` floats from the materialized array into the caller's buffer.
// Must be called after mlx_array_eval_one (or equivalent). The caller is
// responsible for allocating a buffer of at least `n` floats.
void mlx_array_copy_to_f32(
    const std::shared_ptr<MlxArray>& arr, float* out, size_t n);

// Force evaluation (materialize) of a single array by calling
// mlx::core::eval(*arr). Throws std::runtime_error on MLX failure;
// cxx converts that to a Rust error.
void mlx_array_eval_one(const std::shared_ptr<MlxArray>& arr);

// ── Zero-copy MTLBuffer view (Task 5) ────────────────────────────────────────
//
// Construct an MLX array that views an existing Metal buffer without copying.
//
// `mtl_buffer_ptr` is a `MTL::Buffer*` cast to `const uint8_t*` (the Rust
// side passes it as `*const u8` because cxx maps that cleanly; the C++ side
// immediately casts it back to `const void*` and then to `MTL::Buffer*` for
// use in mlx::core::allocator::Buffer).
//
// `shape` describes the array dimensions (product must match element count).
// `dtype` is the MlxDtype tag: 0=float32, 1=float64, 2=int32, 3=bool_.
//
// MLX is given a no-op Deleter so it never frees the buffer; lifetime is
// enforced on the Rust side by `MlxArrayHandle::_input_refs`.
//
// Throws std::invalid_argument for unknown dtype tags (cxx converts to Err).
std::shared_ptr<MlxArray> mlx_array_view_mtl_buffer(
    const uint8_t* mtl_buffer_ptr,
    rust::Slice<const int64_t> shape,
    uint32_t dtype);

// ── M4 Phase 1 Task 6: elementwise op declarations ───────────────────────────
//
// All wrap mlx::core::* ops via the same aliasing-shared_ptr pattern as
// Task 4/5. All throw on dtype/shape errors; cxx maps to Rust Err.
// MLX function names verified against vendor/mlx/mlx/ops.h:
//   subtract (not sub), multiply (not mul), divide (not div),
//   remainder (not mod), power (not pow), negative (not neg),
//   logical_not/logical_and/logical_or, abs, square, where (in mlx::core::).

std::shared_ptr<MlxArray> mlx_op_add(const std::shared_ptr<MlxArray>& a, const std::shared_ptr<MlxArray>& b);
std::shared_ptr<MlxArray> mlx_op_sub(const std::shared_ptr<MlxArray>& a, const std::shared_ptr<MlxArray>& b);
std::shared_ptr<MlxArray> mlx_op_mul(const std::shared_ptr<MlxArray>& a, const std::shared_ptr<MlxArray>& b);
std::shared_ptr<MlxArray> mlx_op_div(const std::shared_ptr<MlxArray>& a, const std::shared_ptr<MlxArray>& b);
std::shared_ptr<MlxArray> mlx_op_mod(const std::shared_ptr<MlxArray>& a, const std::shared_ptr<MlxArray>& b);
std::shared_ptr<MlxArray> mlx_op_pow(const std::shared_ptr<MlxArray>& a, const std::shared_ptr<MlxArray>& b);

std::shared_ptr<MlxArray> mlx_op_eq(const std::shared_ptr<MlxArray>& a, const std::shared_ptr<MlxArray>& b);
std::shared_ptr<MlxArray> mlx_op_ne(const std::shared_ptr<MlxArray>& a, const std::shared_ptr<MlxArray>& b);
std::shared_ptr<MlxArray> mlx_op_lt(const std::shared_ptr<MlxArray>& a, const std::shared_ptr<MlxArray>& b);
std::shared_ptr<MlxArray> mlx_op_le(const std::shared_ptr<MlxArray>& a, const std::shared_ptr<MlxArray>& b);
std::shared_ptr<MlxArray> mlx_op_gt(const std::shared_ptr<MlxArray>& a, const std::shared_ptr<MlxArray>& b);
std::shared_ptr<MlxArray> mlx_op_ge(const std::shared_ptr<MlxArray>& a, const std::shared_ptr<MlxArray>& b);

std::shared_ptr<MlxArray> mlx_op_logical_and(const std::shared_ptr<MlxArray>& a, const std::shared_ptr<MlxArray>& b);
std::shared_ptr<MlxArray> mlx_op_logical_or(const std::shared_ptr<MlxArray>& a, const std::shared_ptr<MlxArray>& b);
std::shared_ptr<MlxArray> mlx_op_logical_not(const std::shared_ptr<MlxArray>& a);

std::shared_ptr<MlxArray> mlx_op_neg(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_abs(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_square(const std::shared_ptr<MlxArray>& a);

std::shared_ptr<MlxArray> mlx_op_where(
    const std::shared_ptr<MlxArray>& cond,
    const std::shared_ptr<MlxArray>& then_v,
    const std::shared_ptr<MlxArray>& else_v);

// Construct a 1-D bool MlxArray from a raw pointer + element count.
// Each non-zero byte becomes true. `n == 0` is allowed (pass null for data).
std::shared_ptr<MlxArray> mlx_array_from_bool_data(const uint8_t* data, size_t n);

// ── M4 Phase 1 Task 7: transcendentals + roots + rounding + atan2 + cast ───
//
// MLX 0.22.0 function name divergences from our Rust naming:
//   asin/acos/atan/atan2 -> arcsin/arccos/arctan/arctan2
//   cast -> astype
//
// MLX 0.22.0 does NOT have cbrt or exp2; we compose them:
//   cbrt(x)  = power(x, 1/3)
//   exp2(x)  = exp(x * ln(2))

std::shared_ptr<MlxArray> mlx_op_sin(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_cos(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_tan(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_sinh(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_cosh(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_tanh(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_asin(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_acos(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_atan(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_log(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_log2(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_log10(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_log1p(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_exp(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_exp2(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_sqrt(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_cbrt(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_floor(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_ceil(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_round(const std::shared_ptr<MlxArray>& a);

std::shared_ptr<MlxArray> mlx_op_atan2(
    const std::shared_ptr<MlxArray>& a, const std::shared_ptr<MlxArray>& b);

std::shared_ptr<MlxArray> mlx_op_cast(const std::shared_ptr<MlxArray>& a, uint32_t dtype);

// ── M4 Phase 1 Task 8: reduction op declarations ─────────────────────────────
//
// Global reductions collapse the whole array to a 0-d scalar. std/var use
// MLX's default `ddof=0` (population variance); Polars' default sample
// variance (ddof=1) is handled by the engine analyzer layer.
// argmin/argmax return I32 arrays of indices (readback as F32 will fail
// the dtype guard; callers must cast).

std::shared_ptr<MlxArray> mlx_op_sum_all(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_mean_all(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_min_all(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_max_all(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_std_all(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_var_all(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_argmin_all(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_argmax_all(const std::shared_ptr<MlxArray>& a);

std::shared_ptr<MlxArray> mlx_op_sum_axis(const std::shared_ptr<MlxArray>& a, int32_t axis);
std::shared_ptr<MlxArray> mlx_op_mean_axis(const std::shared_ptr<MlxArray>& a, int32_t axis);

// ── M4 Phase 1 Task 9: sort + argpartition ───────────────────────────────────

std::shared_ptr<MlxArray> mlx_op_sort(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_argpartition(
    const std::shared_ptr<MlxArray>& a, int32_t kth);

// ── M4 Phase 1 Task 10: cumulative scans + matmul + fft + real/imag ─────────
//
// MLX 0.22.0 cumulative ops require an `axis` argument (no default). For
// 1-D arrays, pass `axis = 0`. Defaults: reverse=false, inclusive=true.
//
// FFT output is complex64 (interleaved real / imag F32 pairs). Use real/imag
// to extract F32 streams for readback.

// ── M5 rolling Task 1: mlx_shift ─────────────────────────────────────────────
//
// Forward-shift a 1-D array along axis 0 by `shift` positions, zero-filling
// the vacated front positions. Output shape equals input shape.
// `shift` is clamped to [0, n] so that shift >= n produces an all-zero result.
//
// Implementation: mlx::core::pad(a, {s, 0}, 0.0f) prepends s zeros to the
// front, then mlx::core::slice(..., {0}, {n}) discards the last s elements.
// API verified against vendor/mlx/mlx/ops.h: pad(array, pair<int,int>,
// pad_value) and slice(array, Shape start, Shape stop).
std::shared_ptr<MlxArray> mlx_shift(const std::shared_ptr<MlxArray>& a, int64_t shift);

// ── M5 rolling Task 4b: mlx_iota_f32 ────────────────────────────────────────
//
// Produce a 1-D F32 array [0.0, 1.0, …, n-1.0] — the row-index (iota)
// generator used by the rolling rewrite to build index arrays on-GPU.
//
// Implementation: mlx::core::arange(0.0, n, float32) using the
// (double start, double stop, Dtype, StreamOrDevice) overload verified against
// vendor/mlx/mlx/ops.h. n <= 0 yields an empty array (arange start==stop).
std::shared_ptr<MlxArray> mlx_iota_f32(int64_t n);

std::shared_ptr<MlxArray> mlx_op_cumsum(const std::shared_ptr<MlxArray>& a, int32_t axis);
std::shared_ptr<MlxArray> mlx_op_cumprod(const std::shared_ptr<MlxArray>& a, int32_t axis);
std::shared_ptr<MlxArray> mlx_op_cummax(const std::shared_ptr<MlxArray>& a, int32_t axis);
std::shared_ptr<MlxArray> mlx_op_cummin(const std::shared_ptr<MlxArray>& a, int32_t axis);

std::shared_ptr<MlxArray> mlx_op_matmul(
    const std::shared_ptr<MlxArray>& a, const std::shared_ptr<MlxArray>& b);

// ── M6 vector search: shape ops ──────────────────────────────────────────────
std::shared_ptr<MlxArray> mlx_op_transpose(
    const std::shared_ptr<MlxArray>& a,
    rust::Slice<const int32_t> axes);
std::shared_ptr<MlxArray> mlx_op_reshape(
    const std::shared_ptr<MlxArray>& a,
    rust::Slice<const int32_t> shape);

std::shared_ptr<MlxArray> mlx_op_fft_1d(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_ifft_1d(const std::shared_ptr<MlxArray>& a);

std::shared_ptr<MlxArray> mlx_op_real(const std::shared_ptr<MlxArray>& a);
std::shared_ptr<MlxArray> mlx_op_imag(const std::shared_ptr<MlxArray>& a);

}  // namespace polars_metal_mlx
