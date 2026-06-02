// shaders/groupby_build_partitioned_build.metal
//
// Capability A1, phase 2: per-partition hash-table build in TGSM.
// One threadgroup per partition. Each threadgroup:
//   - Loads its row_indices_in_partition slice.
//   - Builds a local hash table in TGSM (open addressing, linear probe).
//   - Assigns per-partition local group_ids 0..k_p.
//   - Writes row_to_local_group + n_groups_in_partition out to global.
//
// TGSM_SLOTS = 1024 slots x (16 bytes key + 4 bytes state) = 20 KB; fits
// in the 32 KB threadgroup memory limit. Load factor capped at 75% via the
// probe-limit heuristic (768 unique keys before probe distance climbs).
//
// Overflow detection: if any slot insertion's probe distance > 64
// (heuristic), set the global overflow flag and write UINT_MAX to
// row_to_local_group for that row.
//
// ⚠ SIMD-lockstep risk:
// Per the M2 retrospective (crates/polars-metal-kernels/src/groupby.rs:380-413),
// CAS-claim + spin-wait designs deadlock when multiple threads in the same
// SIMD-group race on the same slot: the claim winner needs to write its
// key words and publish gid+1, but its sibling threads are spinning on
// that slot in lockstep, starving the winner of execution slots.
//
// Phase 4's bet is that at the per-threadgroup level, with hash-spread
// keys, the spin-wait is salvageable because: (a) the scatter (Task 20)
// used high hash bits to assign partitions, and (b) this kernel uses
// low bits to assign slots — so same-warp slot collisions should be rare.
//
// Overflow flag flips on probe-limit exhaustion; caller falls back to A2.

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant uint TGSM_SLOTS = 1024;
constant uint TGSM_PROBE_LIMIT = 64;

static inline uint64_t hash_u128_again(ulong2 key) {
    uint64_t h = 0x9E3779B97F4A7C15ull;
    h ^= key.x * 0xBF58476D1CE4E5B9ull;
    h ^= key.y * 0x94D049BB133111EBull;
    h ^= h >> 31;
    return h * 0x9E3779B97F4A7C15ull;
}

kernel void partition_build(
    device const ulong2*   keys               [[buffer(0)]],
    device const uint*     row_indices        [[buffer(1)]],
    device const uint*     partition_offsets  [[buffer(2)]],
    device uint*           row_to_local_group [[buffer(3)]],
    device atomic_uint*    n_groups_per_part  [[buffer(4)]],
    device atomic_uint*    overflow_flag      [[buffer(5)]],
    constant uint&         n_rows             [[buffer(6)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    (void)n_rows;

    threadgroup ulong2      slot_key[TGSM_SLOTS];
    // slot_state: 0 = empty, UINT_MAX = claiming, else = group_id + 1.
    threadgroup atomic_uint slot_state[TGSM_SLOTS];
    threadgroup atomic_uint next_local_id;

    // Initialize TGSM.
    for (uint i = tid; i < TGSM_SLOTS; i += tg_size) {
        slot_key[i] = ulong2(0, 0);
        atomic_store_explicit(&slot_state[i], 0u, memory_order_relaxed);
    }
    if (tid == 0) {
        atomic_store_explicit(&next_local_id, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint start = partition_offsets[tg_id];
    uint end   = partition_offsets[tg_id + 1];
    uint count = end - start;

    for (uint i = tid; i < count; i += tg_size) {
        uint r = row_indices[start + i];
        ulong2 k = keys[r];
        uint64_t h = hash_u128_again(k);
        uint slot = (uint)(h & (uint64_t)(TGSM_SLOTS - 1u));
        uint probe = 0;
        uint group_id = UINT_MAX;
        while (probe < TGSM_PROBE_LIMIT) {
            uint state = atomic_load_explicit(&slot_state[slot], memory_order_relaxed);
            if (state == 0u) {
                uint expected = 0u;
                if (atomic_compare_exchange_weak_explicit(
                        &slot_state[slot], &expected, UINT_MAX,
                        memory_order_relaxed, memory_order_relaxed)) {
                    // We won the claim. Publish key, then state.
                    slot_key[slot] = k;
                    uint gid = atomic_fetch_add_explicit(&next_local_id, 1u, memory_order_relaxed);
                    atomic_store_explicit(&slot_state[slot], gid + 1u, memory_order_relaxed);
                    group_id = gid;
                    break;
                } else {
                    // CAS failed; `expected` now holds the actual state.
                    state = expected;
                }
            }
            // Spin while another thread is in the claiming phase.
            // See top-of-file note re: SIMD-lockstep risk.
            while (state == UINT_MAX) {
                state = atomic_load_explicit(&slot_state[slot], memory_order_relaxed);
            }
            // state is now a published gid+1 (non-zero, non-UINT_MAX).
            ulong2 sk = slot_key[slot];
            if (sk.x == k.x && sk.y == k.y) {
                group_id = state - 1u;
                break;
            }
            slot = (slot + 1u) & (TGSM_SLOTS - 1u);
            probe += 1;
        }
        if (group_id == UINT_MAX) {
            atomic_store_explicit(overflow_flag, 1u, memory_order_relaxed);
            row_to_local_group[r] = UINT_MAX;
        } else {
            row_to_local_group[r] = group_id;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint final_count = atomic_load_explicit(&next_local_id, memory_order_relaxed);
        atomic_store_explicit(
            &n_groups_per_part[tg_id],
            final_count,
            memory_order_relaxed);
    }
}
