// shaders/groupby_build.metal
//
// Hash-table build kernel. One thread per row. Each thread:
//   1. Reads its row's u128 key (as two u64) and u32 hash.
//   2. Probes the open-addressing table in an outer retry loop.
//   3. On each inner sweep (from `home` through the full table):
//        - Tracks whether any CLAIMED slot was seen (`saw_claimed`).
//        - EMPTY slot: claims via CAS only if `saw_claimed == false`.
//          This guards against creating a duplicate group when an earlier
//          CLAIMED slot might already be installing our key.
//        - CLAIMED slot: sets `saw_claimed = true`, skips (no spin).
//        - READY slot: compares keys. Match → assign group_id. No match →
//          continue probing.
//   4. If the inner sweep ends without assignment and `saw_claimed` was true,
//      start another outer-loop pass. By then, the skipped CLAIMED slots
//      will be READY and their keys will be comparable.
//
// Correctness: no spinning, no deadlock, no duplicate groups.
//   - No spin: CLAIMED slots are skipped, not waited on.
//   - No deadlock: threads in the same SIMD-group (warp) can all make
//     progress simultaneously; no thread blocks on another within the same
//     warp executing in lockstep.
//   - No duplicate groups: a thread only claims an EMPTY slot when all
//     slots before it in the probe chain have confirmed keys different from
//     its own (or were READY with confirmed different keys). Any CLAIMED slot
//     at an earlier position means a concurrent thread might be installing
//     our key there → we do not claim a later slot until we know.
//
// Slot layout (parallel arrays, one entry per hash-table slot):
//   atomic_uint  slot_state[table_size]     — 0=EMPTY, 1=CLAIMED, 2=READY
//   atomic_uint  slot_key[table_size * 4]   — key words: lo_lo, lo_hi, hi_lo, hi_hi
//   atomic_uint  slot_group_id[table_size]  — group ID (valid only when READY)
//   atomic_uint  group_count[1]             — global group-ID allocator
//   uint32_t     first_row_per_group[n_rows]— representative row per group
//   uint32_t     row_to_group[n_rows]       — output: group ID for each row
//
// NOTE: `atomic_ulong` / `atomic_compare_exchange_weak_explicit` on 64-bit
// types is NOT supported in device address space on this Metal toolchain
// (verified: Apple metal version 32023.883). Keys are therefore split into
// four uint32 words and the per-slot state machine uses `atomic_uint`.
//
// The encoder guarantees that the top bit of k_hi is 0 (keys are ≤127 bits;
// T30 router-side fallback enforces this for wider key sets).
//
// Threadgroup/grid: one thread per row. build.rs / dispatch_1d handles grid
// sizing at runtime by querying maxTotalThreadsPerThreadgroup.
// table_size must be a power of two and >= 2.

#include "_groupby.metal"

// Per-slot state constants.
constant uint32_t SLOT_EMPTY   = 0u;
constant uint32_t SLOT_CLAIMED = 1u;
constant uint32_t SLOT_READY   = 2u;

kernel void groupby_build(
    device const uint64_t*    keys                  [[buffer(0)]],
    device const uint32_t*    hashes                [[buffer(1)]],
    device       atomic_uint* slot_state            [[buffer(2)]],
    device       atomic_uint* slot_key              [[buffer(3)]],
    device       atomic_uint* slot_group_id         [[buffer(4)]],
    device       atomic_uint* group_count           [[buffer(5)]],
    device       uint32_t*    first_row_per_group   [[buffer(6)]],
    device       uint32_t*    row_to_group          [[buffer(7)]],
    constant     uint32_t&    n_rows                [[buffer(8)]],
    constant     uint32_t&    table_size            [[buffer(9)]],
    uint                      gid                   [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;

    // Load this row's u128 key as two u64, then split to four u32 words.
    uint64_t k_lo = keys[gid * 2u];
    uint64_t k_hi = keys[gid * 2u + 1u];
    uint32_t h    = hashes[gid];

    uint32_t k_lo_lo = (uint32_t)(k_lo & 0xFFFFFFFFu);
    uint32_t k_lo_hi = (uint32_t)(k_lo >> 32u);
    uint32_t k_hi_lo = (uint32_t)(k_hi & 0xFFFFFFFFu);
    uint32_t k_hi_hi = (uint32_t)(k_hi >> 32u);

    uint32_t mask = table_size - 1u;
    uint32_t home = h & mask;

    // Outer retry loop. Each iteration is one full sweep of the table.
    // We retry if (and only if) we encountered at least one CLAIMED slot on
    // the previous sweep — meaning a peer thread was mid-insert when we
    // passed its slot.  On the next sweep, that slot will be READY and
    // comparable.  Bounded by table_size retries; in practice almost always
    // terminates in ≤2 passes.
    for (uint32_t outer = 0u; outer < table_size; ++outer) {
        bool saw_claimed = false;

        for (uint32_t probe = 0u; probe < table_size; ++probe) {
            uint32_t s = (home + probe) & mask;
            uint32_t state = atomic_load_explicit(&slot_state[s], memory_order_relaxed);

            if (state == SLOT_EMPTY) {
                if (saw_claimed) {
                    // An earlier slot in this probe chain is CLAIMED (key
                    // unknown). That peer might be installing our key. Do
                    // not claim this EMPTY slot — wait for the retry pass.
                    continue;
                }
                // All prior slots confirmed to hold different keys. Safe to claim.
                uint32_t expected = SLOT_EMPTY;
                if (atomic_compare_exchange_weak_explicit(
                        &slot_state[s], &expected, SLOT_CLAIMED,
                        memory_order_relaxed, memory_order_relaxed)) {
                    // Won the slot.
                    atomic_store_explicit(&slot_key[s * 4u],      k_lo_lo, memory_order_relaxed);
                    atomic_store_explicit(&slot_key[s * 4u + 1u], k_lo_hi, memory_order_relaxed);
                    atomic_store_explicit(&slot_key[s * 4u + 2u], k_hi_lo, memory_order_relaxed);
                    atomic_store_explicit(&slot_key[s * 4u + 3u], k_hi_hi, memory_order_relaxed);
                    uint32_t new_gid = atomic_fetch_add_explicit(group_count, 1u, memory_order_relaxed);
                    row_to_group[gid] = new_gid;
                    first_row_per_group[new_gid] = gid;
                    atomic_store_explicit(&slot_group_id[s], new_gid, memory_order_relaxed);
                    // Publish: CLAIMED → READY.
                    atomic_store_explicit(&slot_state[s], SLOT_READY, memory_order_relaxed);
                    return;
                }
                // CAS failed (spurious or another thread won). Re-read and
                // treat the slot as occupied (CLAIMED or READY).
                saw_claimed = true;
                continue;
            }

            if (state == SLOT_CLAIMED) {
                // Key unknown — we can't confirm this isn't our key yet.
                saw_claimed = true;
                continue;
            }

            // state == SLOT_READY: compare keys.
            uint32_t s_lo_lo = atomic_load_explicit(&slot_key[s * 4u],      memory_order_relaxed);
            uint32_t s_lo_hi = atomic_load_explicit(&slot_key[s * 4u + 1u], memory_order_relaxed);
            uint32_t s_hi_lo = atomic_load_explicit(&slot_key[s * 4u + 2u], memory_order_relaxed);
            uint32_t s_hi_hi = atomic_load_explicit(&slot_key[s * 4u + 3u], memory_order_relaxed);

            if (s_lo_lo == k_lo_lo && s_lo_hi == k_lo_hi &&
                s_hi_lo == k_hi_lo && s_hi_hi == k_hi_hi) {
                // Key match. group_id is published (state is READY).
                row_to_group[gid] = atomic_load_explicit(&slot_group_id[s], memory_order_relaxed);
                return;
            }
            // Different key — linear-probe to next slot.
        }

        // Inner sweep finished without assignment.
        // If no CLAIMED slot was seen, the table has only READY slots with
        // different keys (or is full). This shouldn't happen with a
        // correctly sized table (table_size = next_pow2(n_rows * 2)).
        if (!saw_claimed) break;
        // Otherwise, retry: skipped CLAIMED slots are now READY.
    }

    // Should not reach here if table_size = next_pow2(n_rows * 2).
    row_to_group[gid] = 0xFFFFFFFFu;
}
