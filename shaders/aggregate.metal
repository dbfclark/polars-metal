// shaders/aggregate.metal
//
// Aggregation kernels for 32-bit dtypes. One thread per source row.
// Each entry point reads one value + its validity bit, looks up the
// row's group from row_to_group, atomic-OPs into out[group_id].
//
// Why only 32-bit dtypes here:
//   Apple Silicon Metal toolchain 32023.883 does NOT support
//   atomic_fetch_add_explicit / atomic_min/max for 64-bit types
//   (atomic_long / atomic_ulong). The 64-bit aggregation path is
//   in Rust (CPU finalize over GPU-produced row_to_group), not MSL.
//   See docs/kernel-authoring.md for the dispatch matrix.
//
// Null handling: skip rows where validity bit is 0.
//
// Dispatcher contract: caller must seed out[g] with the operator's
// identity element before launch:
//   sum_i32:  0
//   sum_u32:  0
//   sum_f32:  0.0 (as uint bit pattern: 0u)
//   min_i32:  INT32_MAX  (0x7FFFFFFF)
//   max_i32:  INT32_MIN  (0x80000000 interpreted as int)
//   min_u32:  UINT32_MAX (0xFFFFFFFF)
//   max_u32:  0
//   min_f32:  +INFINITY  (as uint bit pattern: 0x7F800000)
//   max_f32:  -INFINITY  (as uint bit pattern: 0xFF800000)
//   count:    0
//   len:      0

#include "_validity.metal"
#include <metal_stdlib>
using namespace metal;

// ---- sum ----

kernel void agg_sum_i32(
    device const int*           values        [[buffer(0)]],
    device const uint8_t*       valid         [[buffer(1)]],
    device const uint32_t*      row_to_group  [[buffer(2)]],
    device       atomic_int*    out           [[buffer(3)]],
    constant     uint32_t&      n_rows        [[buffer(4)]],
    uint                        gid           [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    if (!get_valid(valid, gid)) return;
    uint g = row_to_group[gid];
    atomic_fetch_add_explicit(&out[g], values[gid], memory_order_relaxed);
}

kernel void agg_sum_u32(
    device const uint*          values        [[buffer(0)]],
    device const uint8_t*       valid         [[buffer(1)]],
    device const uint32_t*      row_to_group  [[buffer(2)]],
    device       atomic_uint*   out           [[buffer(3)]],
    constant     uint32_t&      n_rows        [[buffer(4)]],
    uint                        gid           [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    if (!get_valid(valid, gid)) return;
    uint g = row_to_group[gid];
    atomic_fetch_add_explicit(&out[g], values[gid], memory_order_relaxed);
}

// f32 sum via CAS-loop on atomic_uint (bit-pattern container).
// atomic_float exists on this toolchain but atomic_fetch_add_explicit on it
// is not supported in compute kernels on Apple Silicon Metal 32023.883.
// The CAS-loop pattern works universally.
kernel void agg_sum_f32(
    device const float*         values        [[buffer(0)]],
    device const uint8_t*       valid         [[buffer(1)]],
    device const uint32_t*      row_to_group  [[buffer(2)]],
    device       atomic_uint*   out           [[buffer(3)]],
    constant     uint32_t&      n_rows        [[buffer(4)]],
    uint                        gid           [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    if (!get_valid(valid, gid)) return;
    uint g = row_to_group[gid];
    float delta = values[gid];

    uint old_bits = atomic_load_explicit(&out[g], memory_order_relaxed);
    while (true) {
        float cur = as_type<float>(old_bits);
        uint next_bits = as_type<uint>(cur + delta);
        if (atomic_compare_exchange_weak_explicit(
                &out[g], &old_bits, next_bits,
                memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

// ---- min / max ----

kernel void agg_min_i32(
    device const int*           values        [[buffer(0)]],
    device const uint8_t*       valid         [[buffer(1)]],
    device const uint32_t*      row_to_group  [[buffer(2)]],
    device       atomic_int*    out           [[buffer(3)]],
    constant     uint32_t&      n_rows        [[buffer(4)]],
    uint                        gid           [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    if (!get_valid(valid, gid)) return;
    uint g = row_to_group[gid];
    atomic_fetch_min_explicit(&out[g], values[gid], memory_order_relaxed);
}

kernel void agg_max_i32(
    device const int*           values        [[buffer(0)]],
    device const uint8_t*       valid         [[buffer(1)]],
    device const uint32_t*      row_to_group  [[buffer(2)]],
    device       atomic_int*    out           [[buffer(3)]],
    constant     uint32_t&      n_rows        [[buffer(4)]],
    uint                        gid           [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    if (!get_valid(valid, gid)) return;
    uint g = row_to_group[gid];
    atomic_fetch_max_explicit(&out[g], values[gid], memory_order_relaxed);
}

kernel void agg_min_u32(
    device const uint*          values        [[buffer(0)]],
    device const uint8_t*       valid         [[buffer(1)]],
    device const uint32_t*      row_to_group  [[buffer(2)]],
    device       atomic_uint*   out           [[buffer(3)]],
    constant     uint32_t&      n_rows        [[buffer(4)]],
    uint                        gid           [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    if (!get_valid(valid, gid)) return;
    uint g = row_to_group[gid];
    atomic_fetch_min_explicit(&out[g], values[gid], memory_order_relaxed);
}

kernel void agg_max_u32(
    device const uint*          values        [[buffer(0)]],
    device const uint8_t*       valid         [[buffer(1)]],
    device const uint32_t*      row_to_group  [[buffer(2)]],
    device       atomic_uint*   out           [[buffer(3)]],
    constant     uint32_t&      n_rows        [[buffer(4)]],
    uint                        gid           [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    if (!get_valid(valid, gid)) return;
    uint g = row_to_group[gid];
    atomic_fetch_max_explicit(&out[g], values[gid], memory_order_relaxed);
}

// f32 min via CAS-loop on atomic_uint bit pattern.
// Seeded with +INFINITY (0x7F800000); any real value wins the first CAS.
kernel void agg_min_f32(
    device const float*         values        [[buffer(0)]],
    device const uint8_t*       valid         [[buffer(1)]],
    device const uint32_t*      row_to_group  [[buffer(2)]],
    device       atomic_uint*   out           [[buffer(3)]],
    constant     uint32_t&      n_rows        [[buffer(4)]],
    uint                        gid           [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    if (!get_valid(valid, gid)) return;
    uint g = row_to_group[gid];
    float v = values[gid];
    uint old_bits = atomic_load_explicit(&out[g], memory_order_relaxed);
    while (true) {
        float cur = as_type<float>(old_bits);
        if (!(v < cur)) break;
        uint new_bits = as_type<uint>(v);
        if (atomic_compare_exchange_weak_explicit(
                &out[g], &old_bits, new_bits,
                memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

// f32 max via CAS-loop on atomic_uint bit pattern.
// Seeded with -INFINITY (0xFF800000); any real value wins the first CAS.
kernel void agg_max_f32(
    device const float*         values        [[buffer(0)]],
    device const uint8_t*       valid         [[buffer(1)]],
    device const uint32_t*      row_to_group  [[buffer(2)]],
    device       atomic_uint*   out           [[buffer(3)]],
    constant     uint32_t&      n_rows        [[buffer(4)]],
    uint                        gid           [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    if (!get_valid(valid, gid)) return;
    uint g = row_to_group[gid];
    float v = values[gid];
    uint old_bits = atomic_load_explicit(&out[g], memory_order_relaxed);
    while (true) {
        float cur = as_type<float>(old_bits);
        if (!(v > cur)) break;
        uint new_bits = as_type<uint>(v);
        if (atomic_compare_exchange_weak_explicit(
                &out[g], &old_bits, new_bits,
                memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

// ---- count (non-null per group) ----

kernel void agg_count(
    device const uint8_t*       valid         [[buffer(0)]],
    device const uint32_t*      row_to_group  [[buffer(1)]],
    device       atomic_uint*   out           [[buffer(2)]],
    constant     uint32_t&      n_rows        [[buffer(3)]],
    uint                        gid           [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    if (!get_valid(valid, gid)) return;
    uint g = row_to_group[gid];
    atomic_fetch_add_explicit(&out[g], 1u, memory_order_relaxed);
}

// ---- len (row count per group, ignoring validity) ----

kernel void agg_len(
    device const uint32_t*      row_to_group  [[buffer(0)]],
    device       atomic_uint*   out           [[buffer(1)]],
    constant     uint32_t&      n_rows        [[buffer(2)]],
    uint                        gid           [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    uint g = row_to_group[gid];
    atomic_fetch_add_explicit(&out[g], 1u, memory_order_relaxed);
}
