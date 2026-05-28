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

// f32 sum: per-row CAS-loop variant (the "high-cardinality" path).
// Each row issues an atomic-CAS-add against `out[row_to_group[row]]`.
// At HIGH cardinality (≥ ~100 groups) contention is low and CAS retries
// are rare; at LOW cardinality this kernel WILL trip the GPU watchdog
// — the dispatcher routes low-cardinality through `agg_sum_f32_prereduce`
// instead (the original Phase-13 bug).
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

// f32 sum via per-thread register pre-reduce + simdgroup reduce + per-TG
// final CAS-add to device output.
//
// Low-cardinality groupbys (1-16 groups) used to send every row through a
// CAS loop on a handful of atomic slots — at 10M rows and ~4 groups,
// retry contention is O(N^2/2) and trips the Metal GPU watchdog
// (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`). This kernel
// collapses 10M atomic-CAS attempts to (n_threadgroups * n_groups) by
// pre-reducing per-thread → per-simdgroup → per-threadgroup, then a
// single CAS-add per (TG, group) on the device output.
//
// MAX_GROUPS = 16 caps the per-thread register array. The dispatcher
// routes higher cardinality to `agg_sum_f32` (per-row CAS), which is
// fast at high cardinality due to low slot contention.
//
// Dispatch contract:
//   - threadgroup width must be a power-of-two multiple of 32 (simd width
//     on Apple Silicon). The dispatcher uses 256.
//   - threads_per_grid is sized small enough that each thread strides
//     over many rows (~`n_rows / threads_per_grid` each).
//   - n_groups arrives as a constant buffer (slot 5).
#define MAX_GROUPS 16u
// Threadgroup width is 256 on Apple Silicon; simd width is 32, so up to
// 8 simdgroups per threadgroup. Pre-reduction TGSM stages each
// simdgroup's per-group partial in a 8 * MAX_GROUPS slab. The value 8 is
// fixed at compile time — the dispatcher must respect a 256-thread TG.
#define MAX_SIMDS_PER_TG 8u

kernel void agg_sum_f32_prereduce(
    device const float*         values        [[buffer(0)]],
    device const uint8_t*       valid         [[buffer(1)]],
    device const uint32_t*      row_to_group  [[buffer(2)]],
    device       atomic_uint*   out           [[buffer(3)]],
    constant     uint32_t&      n_rows        [[buffer(4)]],
    constant     uint32_t&      n_groups      [[buffer(5)]],
    uint                        gid                  [[thread_position_in_grid]],
    uint                        grid_size            [[threads_per_grid]],
    uint                        tid_in_tg            [[thread_index_in_threadgroup]],
    uint                        sg_index             [[simdgroup_index_in_threadgroup]],
    uint                        lane                 [[thread_index_in_simdgroup]],
    uint                        n_simdgroups         [[simdgroups_per_threadgroup]])
{
    // Per-thread accumulators (one slot per group, capped at MAX_GROUPS).
    float local_sum[MAX_GROUPS];
    for (uint g = 0u; g < MAX_GROUPS; ++g) {
        local_sum[g] = 0.0f;
    }

    // Strided main loop: each thread covers rows gid, gid+grid_size, …
    for (uint row = gid; row < n_rows; row += grid_size) {
        if (!get_valid(valid, row)) continue;
        uint g = row_to_group[row];
        if (g >= n_groups) continue; // defensive — kernel contract is g < n_groups
        local_sum[g] += values[row];
    }

    // TGSM staging: one float per (simdgroup, group).
    threadgroup float tg_partial[MAX_SIMDS_PER_TG * MAX_GROUPS];

    // simd-level reduce + write the per-simdgroup partial into TGSM.
    // The compile-time loop bound (MAX_GROUPS) is required because MSL
    // disallows uniform control flow over simd ops on non-uniform indices.
    for (uint g = 0u; g < MAX_GROUPS; ++g) {
        if (g >= n_groups) break;
        float sgsum = simd_sum(local_sum[g]);
        if (lane == 0u) {
            tg_partial[sg_index * MAX_GROUPS + g] = sgsum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 of the TG sums the simd-partials and CAS-adds to device.
    if (tid_in_tg == 0u) {
        for (uint g = 0u; g < MAX_GROUPS; ++g) {
            if (g >= n_groups) break;
            float total = 0.0f;
            for (uint s = 0u; s < n_simdgroups; ++s) {
                total += tg_partial[s * MAX_GROUPS + g];
            }
            if (total == 0.0f) continue; // identity ⇒ skip atomic
            uint old_bits = atomic_load_explicit(&out[g], memory_order_relaxed);
            while (true) {
                float cur = as_type<float>(old_bits);
                uint next_bits = as_type<uint>(cur + total);
                if (atomic_compare_exchange_weak_explicit(
                        &out[g], &old_bits, next_bits,
                        memory_order_relaxed, memory_order_relaxed)) {
                    break;
                }
            }
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

// f32 min: per-row CAS variant. See `agg_sum_f32` for the dispatcher
// routing rationale. Watchdog risk at low cardinality is the same
// motivation; the pre-reduce variant lives below.
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

// f32 max: per-row CAS variant (counterpart to `agg_min_f32` above).
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

// f32 min via per-thread register pre-reduce + simdgroup reduce + per-TG
// CAS-min on device. Same contention motivation as `agg_sum_f32` above —
// at low n_groups, 10M threads CAS-looping on 4 slots hit the GPU
// watchdog. Pre-reduction collapses to (n_threadgroups * n_groups) CASes.
// Output seeded with +INFINITY; a TG that saw no valid rows skips the
// atomic via the `total == +INF` check.
kernel void agg_min_f32_prereduce(
    device const float*         values        [[buffer(0)]],
    device const uint8_t*       valid         [[buffer(1)]],
    device const uint32_t*      row_to_group  [[buffer(2)]],
    device       atomic_uint*   out           [[buffer(3)]],
    constant     uint32_t&      n_rows        [[buffer(4)]],
    constant     uint32_t&      n_groups      [[buffer(5)]],
    uint                        gid                  [[thread_position_in_grid]],
    uint                        grid_size            [[threads_per_grid]],
    uint                        tid_in_tg            [[thread_index_in_threadgroup]],
    uint                        sg_index             [[simdgroup_index_in_threadgroup]],
    uint                        lane                 [[thread_index_in_simdgroup]],
    uint                        n_simdgroups         [[simdgroups_per_threadgroup]])
{
    float local_min[MAX_GROUPS];
    for (uint g = 0u; g < MAX_GROUPS; ++g) {
        local_min[g] = INFINITY;
    }

    for (uint row = gid; row < n_rows; row += grid_size) {
        if (!get_valid(valid, row)) continue;
        uint g = row_to_group[row];
        if (g >= n_groups) continue;
        float v = values[row];
        local_min[g] = min(local_min[g], v);
    }

    threadgroup float tg_partial[MAX_SIMDS_PER_TG * MAX_GROUPS];
    for (uint g = 0u; g < MAX_GROUPS; ++g) {
        if (g >= n_groups) break;
        float sgmin = simd_min(local_min[g]);
        if (lane == 0u) {
            tg_partial[sg_index * MAX_GROUPS + g] = sgmin;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid_in_tg == 0u) {
        for (uint g = 0u; g < MAX_GROUPS; ++g) {
            if (g >= n_groups) break;
            float total = INFINITY;
            for (uint s = 0u; s < n_simdgroups; ++s) {
                total = min(total, tg_partial[s * MAX_GROUPS + g]);
            }
            if (isinf(total) && total > 0.0f) continue; // identity ⇒ skip
            uint old_bits = atomic_load_explicit(&out[g], memory_order_relaxed);
            while (true) {
                float cur = as_type<float>(old_bits);
                if (!(total < cur)) break;
                uint new_bits = as_type<uint>(total);
                if (atomic_compare_exchange_weak_explicit(
                        &out[g], &old_bits, new_bits,
                        memory_order_relaxed, memory_order_relaxed)) {
                    break;
                }
            }
        }
    }
}

// f32 max via per-thread register pre-reduce + simdgroup reduce + per-TG
// CAS-max on device. Mirrors `agg_min_f32_prereduce` above. Seeded with
// -INFINITY.
kernel void agg_max_f32_prereduce(
    device const float*         values        [[buffer(0)]],
    device const uint8_t*       valid         [[buffer(1)]],
    device const uint32_t*      row_to_group  [[buffer(2)]],
    device       atomic_uint*   out           [[buffer(3)]],
    constant     uint32_t&      n_rows        [[buffer(4)]],
    constant     uint32_t&      n_groups      [[buffer(5)]],
    uint                        gid                  [[thread_position_in_grid]],
    uint                        grid_size            [[threads_per_grid]],
    uint                        tid_in_tg            [[thread_index_in_threadgroup]],
    uint                        sg_index             [[simdgroup_index_in_threadgroup]],
    uint                        lane                 [[thread_index_in_simdgroup]],
    uint                        n_simdgroups         [[simdgroups_per_threadgroup]])
{
    float local_max[MAX_GROUPS];
    for (uint g = 0u; g < MAX_GROUPS; ++g) {
        local_max[g] = -INFINITY;
    }

    for (uint row = gid; row < n_rows; row += grid_size) {
        if (!get_valid(valid, row)) continue;
        uint g = row_to_group[row];
        if (g >= n_groups) continue;
        float v = values[row];
        local_max[g] = max(local_max[g], v);
    }

    threadgroup float tg_partial[MAX_SIMDS_PER_TG * MAX_GROUPS];
    for (uint g = 0u; g < MAX_GROUPS; ++g) {
        if (g >= n_groups) break;
        float sgmax = simd_max(local_max[g]);
        if (lane == 0u) {
            tg_partial[sg_index * MAX_GROUPS + g] = sgmax;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid_in_tg == 0u) {
        for (uint g = 0u; g < MAX_GROUPS; ++g) {
            if (g >= n_groups) break;
            float total = -INFINITY;
            for (uint s = 0u; s < n_simdgroups; ++s) {
                total = max(total, tg_partial[s * MAX_GROUPS + g]);
            }
            if (isinf(total) && total < 0.0f) continue; // identity ⇒ skip
            uint old_bits = atomic_load_explicit(&out[g], memory_order_relaxed);
            while (true) {
                float cur = as_type<float>(old_bits);
                if (!(total > cur)) break;
                uint new_bits = as_type<uint>(total);
                if (atomic_compare_exchange_weak_explicit(
                        &out[g], &old_bits, new_bits,
                        memory_order_relaxed, memory_order_relaxed)) {
                    break;
                }
            }
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
