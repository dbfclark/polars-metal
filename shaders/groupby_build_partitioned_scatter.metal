// shaders/groupby_build_partitioned_scatter.metal
//
// Capability A1, phase 1: per-row, compute partition_id from a hash of
// the encoded composite key, then scatter row indices into per-partition
// lanes. The scatter uses a two-pass approach:
//   Pass A: count per-partition row counts (atomic).
//   (CPU between A and B: exclusive scan -> partition_offsets[].)
//   Pass B: scatter row_idx into per-partition slot using atomic-add
//           on a write cursor seeded by partition_offsets.
//
// Both passes are 32-bit atomics -- well within Apple Silicon's set.
//
// Keys are passed as `ulong2` (16 bytes, 16-byte aligned). The host
// reinterprets `&[u128]` directly as bytes — u128 has the same layout
// as ulong2 on little-endian Apple Silicon, so no CPU lo/hi split is
// needed.

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant uint TGSM_SLOTS_PER_PARTITION = 1024;

static inline uint64_t hash_u128(ulong2 key) {
    uint64_t h = 0x9E3779B97F4A7C15ull;
    h ^= key.x * 0xBF58476D1CE4E5B9ull;
    h ^= key.y * 0x94D049BB133111EBull;
    h ^= h >> 31;
    return h * 0x9E3779B97F4A7C15ull;
}

static inline uint partition_id(ulong2 key, uint n_partitions, uint log2_tgsm_slots) {
    uint64_t h = hash_u128(key);
    return (uint)((h >> log2_tgsm_slots) & (uint64_t)(n_partitions - 1u));
}

kernel void partition_count(
    device const ulong2*   keys           [[buffer(0)]],
    device atomic_uint*    counts         [[buffer(1)]],
    constant uint&         n_rows         [[buffer(2)]],
    constant uint&         n_partitions   [[buffer(3)]],
    constant uint&         log2_tgsm      [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    uint p = partition_id(keys[gid], n_partitions, log2_tgsm);
    atomic_fetch_add_explicit(&counts[p], 1u, memory_order_relaxed);
}

// Writes both `row_indices_out` and `partition_id_per_row` so the host's
// final group_id derivation doesn't need to re-hash each key.
kernel void partition_scatter(
    device const ulong2*   keys                  [[buffer(0)]],
    device const uint*     partition_offsets     [[buffer(1)]],
    device atomic_uint*    write_cursors         [[buffer(2)]],
    device uint*           row_indices_out       [[buffer(3)]],
    device uint*           partition_id_per_row  [[buffer(4)]],
    constant uint&         n_rows                [[buffer(5)]],
    constant uint&         n_partitions          [[buffer(6)]],
    constant uint&         log2_tgsm             [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    uint p = partition_id(keys[gid], n_partitions, log2_tgsm);
    uint slot = atomic_fetch_add_explicit(&write_cursors[p], 1u, memory_order_relaxed);
    row_indices_out[partition_offsets[p] + slot] = gid;
    partition_id_per_row[gid] = p;
}
