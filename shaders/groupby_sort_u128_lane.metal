// shaders/groupby_sort_u128_lane.metal
//
// Per-lane (8-bit) radix-sort pass for u128 keys paired with u32 row
// indices. Two kernels per lane:
//   1. lane_histogram: count occurrences of each 0..255 digit globally.
//   2. (CPU exclusive scan turns counts → offsets; offsets are loaded
//      into a `cursors` buffer.)
//   3. lane_scatter: each thread reads its digit and atomic-increments
//      the per-digit cursor to claim a target position; writes the
//      (key, row_idx) tuple there.
//
// One full u128 sort = 16 8-bit-lane passes (Task 26 chains them).

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

static inline uchar extract_digit(uint64_t key_lo, uint64_t key_hi, uint lane_idx) {
    if (lane_idx < 8) return (uchar)((key_lo >> (lane_idx * 8)) & 0xFFul);
    return (uchar)((key_hi >> ((lane_idx - 8) * 8)) & 0xFFul);
}

kernel void lane_histogram(
    device const uint64_t* keys_lo  [[buffer(0)]],
    device const uint64_t* keys_hi  [[buffer(1)]],
    device atomic_uint*    bins     [[buffer(2)]],   // [256], pre-zeroed
    constant uint&         n_rows   [[buffer(3)]],
    constant uint&         lane_idx [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    uchar d = extract_digit(keys_lo[gid], keys_hi[gid], lane_idx);
    atomic_fetch_add_explicit(&bins[(uint)d], 1u, memory_order_relaxed);
}

kernel void lane_scatter(
    device const uint64_t* keys_lo_in  [[buffer(0)]],
    device const uint64_t* keys_hi_in  [[buffer(1)]],
    device const uint*     row_idx_in  [[buffer(2)]],
    device uint64_t*       keys_lo_out [[buffer(3)]],
    device uint64_t*       keys_hi_out [[buffer(4)]],
    device uint*           row_idx_out [[buffer(5)]],
    device atomic_uint*    cursors     [[buffer(6)]],  // seeded with offsets
    constant uint&         n_rows      [[buffer(7)]],
    constant uint&         lane_idx    [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    uchar d = extract_digit(keys_lo_in[gid], keys_hi_in[gid], lane_idx);
    uint write_pos = atomic_fetch_add_explicit(&cursors[(uint)d], 1u, memory_order_relaxed);
    keys_lo_out[write_pos] = keys_lo_in[gid];
    keys_hi_out[write_pos] = keys_hi_in[gid];
    row_idx_out[write_pos] = row_idx_in[gid];
}
