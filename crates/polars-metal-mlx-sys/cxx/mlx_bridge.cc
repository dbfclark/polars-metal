// crates/polars-metal-mlx-sys/cxx/mlx_bridge.cc
#include "mlx_bridge.h"
#include "mlx/allocator.h"
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

// ── M4 Phase 1: MlxArray handle ──────────────────────────────────────────────

std::shared_ptr<MlxArray> mlx_array_from_f32_data(const float* data, size_t n) {
    // Shape is std::vector<int32_t> in MLX v0.22.0 (same as add_f32 above).
    // n == 0 produces an empty shape; MLX accepts this and creates a length-0
    // 1-D array (shape = {0}).
    int32_t shape_n = static_cast<int32_t>(n);
    // When n == 0 the pointer may be dangling (Rust passes NonNull::dangling());
    // MLX must not dereference it. We pass nullptr explicitly in that case.
    const float* src = (n == 0) ? nullptr : data;
    // Construct a mlx::core::array then up-cast to MlxArray via shared_ptr
    // aliasing. MlxArray inherits from mlx::core::array so the static_cast
    // is well-defined; we use the aliasing constructor so the control block
    // is shared and lifetime is correct.
    auto base = std::make_shared<mlx::core::array>(src, std::vector<int>{shape_n}, mlx::core::float32);
    return std::shared_ptr<MlxArray>(base, static_cast<MlxArray*>(base.get()));
}

rust::Vec<uint64_t> mlx_array_shape(const std::shared_ptr<MlxArray>& arr) {
    rust::Vec<uint64_t> out;
    for (auto d : arr->shape()) {
        out.push_back(static_cast<uint64_t>(d));
    }
    return out;
}

bool mlx_array_is_f32(const std::shared_ptr<MlxArray>& arr) {
    return arr->dtype() == mlx::core::float32;
}

void mlx_array_copy_to_f32(
    const std::shared_ptr<MlxArray>& arr, float* out, size_t n) {
    // Caller guarantees the array has been eval'd and `out` holds at least n
    // floats. data<float>() returns a pointer into MLX-managed memory.
    const float* src = arr->data<float>();
    std::memcpy(out, src, n * sizeof(float));
}

void mlx_array_eval_one(const std::shared_ptr<MlxArray>& arr) {
    // mlx::core::eval's variadic template checks is_arrays_v<T> which only
    // matches mlx::core::array exactly. Downcast to the base type so the
    // template constraint is satisfied. The static_cast is safe because
    // MlxArray publicly inherits from mlx::core::array.
    mlx::core::array& base = static_cast<mlx::core::array&>(*arr);
    mlx::core::eval(base);
}

// ── Zero-copy MTLBuffer view (Task 5) ────────────────────────────────────────

std::shared_ptr<MlxArray> mlx_array_view_mtl_buffer(
    const uint8_t* mtl_buffer_ptr,
    rust::Slice<const int64_t> shape,
    uint32_t dtype)
{
    // Convert shape from rust::Slice<const int64_t> to std::vector<int32_t>
    // (mlx::core::Shape = std::vector<ShapeElem> = std::vector<int32_t>).
    std::vector<int32_t> shape_vec;
    shape_vec.reserve(shape.size());
    for (auto d : shape) {
        shape_vec.push_back(static_cast<int32_t>(d));
    }

    // Wrap the MTL::Buffer* into mlx::core::allocator::Buffer.
    // The Rust side passes the ObjC instance pointer (same as MTL::Buffer*)
    // as *const uint8_t. We cast const away because allocator::Buffer stores
    // a non-const void*; MLX treats the data as logically immutable for a
    // view constructed this way (we never write through this pointer from
    // the MLX side — it was created from a read-only Rust &[f32]).
    mlx::core::allocator::Buffer mlx_buf(
        const_cast<void*>(static_cast<const void*>(mtl_buffer_ptr)));

    // No-op Deleter: Rust owns the MetalBuffer lifetime. MLX will call this
    // function when it decides to release the backing buffer; the empty body
    // ensures it does nothing and leaves the MTLBuffer untouched.
    mlx::core::Deleter no_op_deleter = [](mlx::core::allocator::Buffer) {};

    // Map the dtype tag to mlx::core::Dtype.
    // MLX 0.22.0 has no float64; tag 1 throws rather than silently mapping
    // to a wrong type.  Tags: 0=float32, 1=float64 (unsupported), 2=int32,
    // 3=bool_.
    mlx::core::Dtype dt = mlx::core::float32; // initialise to satisfy compiler
    switch (dtype) {
        case 0: dt = mlx::core::float32; break;
        case 1:
            throw std::invalid_argument(
                "mlx_array_view_mtl_buffer: float64 is not supported by "
                "MLX 0.22.0; use float32");
        case 2: dt = mlx::core::int32;   break;
        case 3: dt = mlx::core::bool_;   break;
        default:
            throw std::invalid_argument(
                "mlx_array_view_mtl_buffer: unknown dtype tag");
    }

    // Construct via the buffer-accepting array constructor:
    //   explicit array(allocator::Buffer data, Shape shape, Dtype dtype,
    //                  Deleter deleter = allocator::free);
    // (see vendor/mlx/mlx/array.h ~line 60).
    // Use the aliasing shared_ptr constructor to upcast mlx::core::array* to
    // MlxArray* — same pattern as mlx_array_from_f32_data.
    auto base = std::make_shared<mlx::core::array>(
        mlx_buf, std::move(shape_vec), dt, no_op_deleter);
    return std::shared_ptr<MlxArray>(base, static_cast<MlxArray*>(base.get()));
}

}  // namespace polars_metal_mlx
