// crates/polars-metal-mlx-sys/cxx/mlx_bridge.cc
#include "mlx_bridge.h"
#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "mlx/utils.h"

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

}  // namespace polars_metal_mlx
