// shaders/groupby_build.metal
//
// Hash-table build kernel. One thread per row. Each thread:
//   1. Reads its row's u128 key (as two u64) and u32 hash.
//   2. Probes the open-addressing table starting at `hash % table_size`.
//   3. For each slot it visits:
//        - If EMPTY: atomic-CAS slot_state EMPTY → CLAIMED to take ownership.
//          On win: store all four u32 key words, fetch_add group_count to get
//          a fresh group_id, store it in slot_group_id, write first_row and
//          row_to_group, then store slot_state CLAIMED → READY to publish.
//        - If CLAIMED or READY with same key: spin until READY, then read
//          slot_group_id and assign to row_to_group.
//        - If READY with different key: linear-probe to next slot.
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
    device const uint64_t*   keys                  [[buffer(0)]],
    device const uint32_t*   hashes                [[buffer(1)]],
    device       atomic_uint* slot_state            [[buffer(2)]],
    device       atomic_uint* slot_key              [[buffer(3)]],
    device       atomic_uint* slot_group_id         [[buffer(4)]],
    device       atomic_uint* group_count           [[buffer(5)]],
    device       uint32_t*   first_row_per_group    [[buffer(6)]],
    device       uint32_t*   row_to_group           [[buffer(7)]],
    constant     uint32_t&   n_rows                 [[buffer(8)]],
    constant     uint32_t&   table_size             [[buffer(9)]],
    uint                     gid                    [[thread_position_in_grid]])
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

    for (uint32_t probe = 0u; probe < table_size; ++probe) {
        uint32_t s = (home + probe) & mask;

        uint32_t state = atomic_load_explicit(&slot_state[s], memory_order_relaxed);

        if (state == SLOT_EMPTY) {
            // Attempt to claim the empty slot.
            uint32_t expected = SLOT_EMPTY;
            if (atomic_compare_exchange_weak_explicit(
                    &slot_state[s], &expected, SLOT_CLAIMED,
                    memory_order_relaxed, memory_order_relaxed)) {
                // We own the slot. Write key words, allocate group_id,
                // record first_row and row_to_group, then publish.
                atomic_store_explicit(&slot_key[s * 4u],      k_lo_lo, memory_order_relaxed);
                atomic_store_explicit(&slot_key[s * 4u + 1u], k_lo_hi, memory_order_relaxed);
                atomic_store_explicit(&slot_key[s * 4u + 2u], k_hi_lo, memory_order_relaxed);
                atomic_store_explicit(&slot_key[s * 4u + 3u], k_hi_hi, memory_order_relaxed);

                uint32_t new_gid = atomic_fetch_add_explicit(group_count, 1u, memory_order_relaxed);
                row_to_group[gid] = new_gid;
                first_row_per_group[new_gid] = gid;
                atomic_store_explicit(&slot_group_id[s], new_gid, memory_order_relaxed);

                // Publish: transition CLAIMED → READY. Other threads spinning
                // on this slot will unblock once they see SLOT_READY.
                atomic_store_explicit(&slot_state[s], SLOT_READY, memory_order_relaxed);
                return;
            }
            // CAS failed — another thread claimed the slot. Fall through
            // to spin until READY before reading the key.
            state = atomic_load_explicit(&slot_state[s], memory_order_relaxed);
        }

        // Spin until the slot owner has fully published (state == READY).
        for (uint32_t spin = 0u; spin < 65536u && state != SLOT_READY; ++spin) {
            state = atomic_load_explicit(&slot_state[s], memory_order_relaxed);
        }
        if (state != SLOT_READY) {
            // Spin timed out — defensive; should not occur in practice.
            row_to_group[gid] = 0xFFFFFFFFu;
            return;
        }

        // Slot is ready. Read key words and compare.
        uint32_t s_lo_lo = atomic_load_explicit(&slot_key[s * 4u],      memory_order_relaxed);
        uint32_t s_lo_hi = atomic_load_explicit(&slot_key[s * 4u + 1u], memory_order_relaxed);
        uint32_t s_hi_lo = atomic_load_explicit(&slot_key[s * 4u + 2u], memory_order_relaxed);
        uint32_t s_hi_hi = atomic_load_explicit(&slot_key[s * 4u + 3u], memory_order_relaxed);

        if (s_lo_lo == k_lo_lo && s_lo_hi == k_lo_hi &&
            s_hi_lo == k_hi_lo && s_hi_hi == k_hi_hi) {
            // Key match. Read group_id (safe: state is READY, so group_id
            // is fully published before the READY store above).
            row_to_group[gid] = atomic_load_explicit(&slot_group_id[s], memory_order_relaxed);
            return;
        }

        // Different key — linear-probe to next slot.
    }

    // Table full. Should not occur: table_size = next_pow2(n_rows * 2).
    row_to_group[gid] = 0xFFFFFFFFu;
}
