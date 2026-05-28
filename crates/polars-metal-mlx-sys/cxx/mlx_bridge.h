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

}  // namespace polars_metal_mlx
