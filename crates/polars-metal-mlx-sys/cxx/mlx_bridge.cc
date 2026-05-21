// crates/polars-metal-mlx-sys/cxx/mlx_bridge.cc
#include "mlx_bridge.h"
#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "mlx/utils.h"

#include <cstring>
#include <memory>
#include <stdexcept>

namespace polars_metal_mlx {

int64_t add_one(int64_t x) {
    return x + 1;
}

std::unique_ptr<std::vector<float>> add_f32(
    const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("add_f32: shape mismatch");
    }
    // Shape is std::vector<int32_t> in MLX v0.22.0.
    int32_t n = static_cast<int32_t>(a.size());
    auto arr_a = mlx::core::array(a.data(), {n}, mlx::core::float32);
    auto arr_b = mlx::core::array(b.data(), {n}, mlx::core::float32);
    auto out = mlx::core::add(arr_a, arr_b);
    mlx::core::eval(out);
    const float* data = out.data<float>();
    return std::make_unique<std::vector<float>>(data, data + n);
}


std::unique_ptr<std::vector<float>> add_f32_on_gpu(
    const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("add_f32_on_gpu: shape mismatch");
    }
    // StreamContext is RAII: saves the current default device/stream, switches
    // to Device::gpu for the scope, then restores on destruction. If Metal is
    // unavailable, set_default_device throws std::invalid_argument before any
    // computation begins — the caller gets an explicit error, not a silent
    // CPU fallback.
    // Use brace-initialization to avoid the "most vexing parse" ambiguity.
    mlx::core::Device gpu_device{mlx::core::Device::gpu};
    mlx::core::StreamContext gpu_ctx(gpu_device);

    int32_t n = static_cast<int32_t>(a.size());
    auto arr_a = mlx::core::array(a.data(), {n}, mlx::core::float32);
    auto arr_b = mlx::core::array(b.data(), {n}, mlx::core::float32);
    auto out = mlx::core::add(arr_a, arr_b);
    mlx::core::eval(out);
    const float* data = out.data<float>();
    return std::make_unique<std::vector<float>>(data, data + n);
}

void cumsum_u8_to_u32(
    rust::Slice<const uint8_t> input, rust::Slice<uint32_t> output) {
    if (input.size() != output.size()) {
        throw std::runtime_error("cumsum_u8_to_u32: shape mismatch");
    }
    if (input.empty()) {
        return;
    }
    // Force execution on Metal GPU. See add_f32_on_gpu for the RAII pattern;
    // brace-initialization avoids the most-vexing-parse ambiguity.
    mlx::core::Device gpu_device{mlx::core::Device::gpu};
    mlx::core::StreamContext gpu_ctx(gpu_device);

    int32_t n = static_cast<int32_t>(input.size());
    // Construct a uint8 MLX array from the input slice. `mlx::core::array`
    // copies the data into MLX-managed memory (one memcpy of n bytes) —
    // there's no zero-copy slice constructor at MLX v0.22.0's public API
    // surface. Cast to uint32 before the scan so the running total has
    // headroom (a 4B-row keep-all input would overflow uint8 immediately).
    auto in_u8 = mlx::core::array(input.data(), {n}, mlx::core::uint8);
    auto in_u32 = mlx::core::astype(in_u8, mlx::core::uint32);
    // axis=0, reverse=false, inclusive=true: classic prefix-sum.
    auto scanned = mlx::core::cumsum(in_u32, /*axis=*/0, /*reverse=*/false,
                                     /*inclusive=*/true);
    mlx::core::eval(scanned);
    // Memcpy the scan result directly into the Rust-owned output slice.
    // This replaces the old `std::vector<uint32_t>` round-trip and the
    // Rust-side per-element copy back through CxxVector.
    std::memcpy(output.data(), scanned.data<uint32_t>(),
                static_cast<size_t>(n) * sizeof(uint32_t));
}

}  // namespace polars_metal_mlx
