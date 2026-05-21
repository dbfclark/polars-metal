// crates/polars-metal-mlx-sys/cxx/mlx_bridge.h
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "rust/cxx.h"

namespace polars_metal_mlx {

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

}  // namespace polars_metal_mlx
