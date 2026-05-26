// shaders/groupby_sort_u128_lane.metal
//
// STABLE per-lane (8-bit) radix-sort pass for u128 keys paired with u32
// row indices. LSD radix sort requires each pass to be stable: keys with
// the same digit at this lane must preserve their relative order from
// earlier passes. The previous design used an atomic-add cursor in the
// scatter, which raced within digit buckets and broke stability.
//
// Two kernels per lane pass:
//   1. lane_tile_hist: each 256-thread tile (= threadgroup) builds a
//      per-tile histogram in TGSM via atomic adds, then writes the 256
//      counts to global tile_hist[tg_id * 256 + d].
//   2. (CPU computes per-tile-per-digit prefix and global per-digit
//      offsets from the tile histograms.)
//   3. lane_stable_scatter: each active thread computes its output
//      position as
//        global_offset[d] + tile_prefix[tg_id][d] + tile_local_rank
//      where tile_local_rank is built inside the kernel via SIMD
//      match-and-rank (over the 32-lane simdgroup) plus a cross-simdgroup
//      prefix sum in TGSM (8 simdgroups per tile).
//
// Stability proof sketch: for two rows r1 < r2 sharing digit d,
//   * different tile  -> tile_prefix monotonic in tile id;
//   * same tile, different simdgroup -> cross-simdgroup prefix monotonic;
//   * same simdgroup -> match-and-rank counts contributing lanes with
//     lane < my lane, so r2's rank strictly exceeds r1's.
// In all three cases r2's output position is strictly greater than r1's.
//
// One full u128 sort = 16 such 8-bit-lane passes (Task 26 chains them).

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// Tile = 256 threads = 8 simdgroups x 32 lanes (Apple Silicon).
// (TILE_SIZE = 256 is implicit in the dispatch threadgroup width; the
// kernel reads it from `threads_per_threadgroup` so we do not hardcode.)
constant constexpr uint SIMDS_PER_TILE = 8;
constant constexpr uint SIMD_WIDTH = 32;

static inline uchar extract_digit(uint64_t key_lo, uint64_t key_hi, uint lane_idx) {
    if (lane_idx < 8) return (uchar)((key_lo >> (lane_idx * 8)) & 0xFFul);
    return (uchar)((key_hi >> ((lane_idx - 8) * 8)) & 0xFFul);
}

// ---- Kernel A: per-tile histogram --------------------------------------
//
// Each tile (= one threadgroup of 256 threads) builds a 256-bucket
// histogram of its slice of the input. The tg writes its 256 counts to
// `tile_hist[tg_id * 256 + d]`. The CPU consumes that and produces the
// per-tile, per-digit prefix needed by the scatter kernel.
kernel void lane_tile_hist(
    device const uint64_t* keys_lo  [[buffer(0)]],
    device const uint64_t* keys_hi  [[buffer(1)]],
    device uint*           tile_hist [[buffer(2)]],   // [n_tiles * 256]
    constant uint&         n_rows   [[buffer(3)]],
    constant uint&         lane_idx [[buffer(4)]],
    uint gid     [[thread_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_id   [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup atomic_uint tgsm_hist[256];
    for (uint i = tid; i < 256; i += tg_size) {
        atomic_store_explicit(&tgsm_hist[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid < n_rows) {
        uint d = (uint)extract_digit(keys_lo[gid], keys_hi[gid], lane_idx);
        atomic_fetch_add_explicit(&tgsm_hist[d], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write per-tile histogram to global storage.
    for (uint i = tid; i < 256; i += tg_size) {
        tile_hist[tg_id * 256 + i] =
            atomic_load_explicit(&tgsm_hist[i], memory_order_relaxed);
    }
}

// ---- Kernel B: stable scatter ------------------------------------------
//
// Each active thread:
//   1. Reads its digit d from the input key.
//   2. Computes its rank within its simdgroup among lanes that share d
//      ("match-and-rank") via a loop of simd_broadcast calls. Also
//      records the simdgroup's total count of d, contributed by the
//      first matching lane (rank == 0) to TGSM.
//   3. After a tg barrier, sums earlier simdgroups' counts of d to get
//      a cross-simdgroup prefix.
//   4. Total local rank in the tile = cross_rank + simd_rank.
//   5. Output pos = global_offset[d] + tile_prefix[tg_id][d] + local_rank.
kernel void lane_stable_scatter(
    device const uint64_t* keys_lo_in     [[buffer(0)]],
    device const uint64_t* keys_hi_in     [[buffer(1)]],
    device const uint*     row_idx_in     [[buffer(2)]],
    device uint64_t*       keys_lo_out    [[buffer(3)]],
    device uint64_t*       keys_hi_out    [[buffer(4)]],
    device uint*           row_idx_out    [[buffer(5)]],
    device const uint*     tile_prefix    [[buffer(6)]],   // [n_tiles * 256]
    device const uint*     global_offset  [[buffer(7)]],   // [256]
    constant uint&         n_rows         [[buffer(8)]],
    constant uint&         lane_idx       [[buffer(9)]],
    uint gid       [[thread_position_in_grid]],
    uint tid       [[thread_position_in_threadgroup]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tg_size   [[threads_per_threadgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]])
{
    // Per-simdgroup, per-digit count buffer. 8 simdgroups * 256 buckets *
    // 4 bytes = 8 KB; well under the 32 KB TGSM budget per threadgroup.
    threadgroup uint simd_digit_counts[SIMDS_PER_TILE * 256];
    for (uint i = tid; i < SIMDS_PER_TILE * 256; i += tg_size) {
        simd_digit_counts[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool active = (gid < n_rows);
    uint my_digit = 0;
    if (active) {
        my_digit = (uint)extract_digit(keys_lo_in[gid], keys_hi_in[gid], lane_idx);
    }

    // SIMD match-and-rank within the 32-lane simdgroup. We iterate k =
    // 0..31 and pull the (digit, active) of lane k via simd_broadcast.
    // All 32 lanes execute the broadcast in lockstep, so this works even
    // for inactive lanes (their broadcasted values are still valid; we
    // just gate the match with `other_active`). Inactive lanes also do
    // the loop (and harmlessly produce simd_rank == simd_total == 0).
    uint simd_rank = 0;
    uint simd_total = 0;
    for (uint k = 0; k < SIMD_WIDTH; k++) {
        uint other_digit  = simd_broadcast(my_digit, k);
        uint other_active = simd_broadcast(active ? 1u : 0u, k);
        bool match = active && (other_active == 1u) && (other_digit == my_digit);
        if (match) {
            if (k < simd_lane) simd_rank += 1;
            simd_total += 1;
        }
    }

    // First active matching lane (rank == 0) in this simdgroup publishes
    // the simdgroup's count for its digit. Every other matching lane in
    // the same simdgroup also has simd_total == simd_rank's matching
    // count, but writing only from rank-0 avoids redundant TGSM writes
    // (multiple lanes writing the same value is also safe, just wasteful).
    if (active && simd_rank == 0 && simd_total > 0) {
        simd_digit_counts[simd_id * 256 + my_digit] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-simdgroup prefix sum for my digit: how many earlier
    // simdgroups (in this tile) carried this digit.
    uint cross_rank = 0;
    if (active) {
        for (uint s = 0; s < simd_id; s++) {
            cross_rank += simd_digit_counts[s * 256 + my_digit];
        }
    }
    uint my_local_rank = cross_rank + simd_rank;

    if (active) {
        uint pos = global_offset[my_digit]
                 + tile_prefix[tg_id * 256 + my_digit]
                 + my_local_rank;
        keys_lo_out[pos] = keys_lo_in[gid];
        keys_hi_out[pos] = keys_hi_in[gid];
        row_idx_out[pos] = row_idx_in[gid];
    }
}
