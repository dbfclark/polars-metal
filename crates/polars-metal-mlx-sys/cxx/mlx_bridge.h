// crates/polars-metal-mlx-sys/cxx/mlx_bridge.h
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

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

}  // namespace polars_metal_mlx
