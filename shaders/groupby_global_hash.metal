// shaders/groupby_global_hash.metal
//
// Capability A3 spike (Phase 5b): single-pass global-atomic GPU hash
// table for groupby's build phase. Targets the regime where A1 overflows
// (>~16K groups) and where A2's 16-pass radix sort proved too slow on
// Apple Silicon (~2s at 10M rows).
//
// Design:
//   - One global open-addressing hash table of `table_size` slots
//     (table_size is power-of-two; caller sizes ≥ 2 × expected unique).
//   - Each slot: a 16-byte key (ulong2) + a 4-byte state (atomic_uint).
//   - State encoding:
//        0          — empty
//        UINT_MAX   — claiming (some thread has won the CAS and is in
//                     the middle of writing the key + publishing gid+1)
//        otherwise  — (group_id + 1)
//   - Each thread: hash → linear-probe until it finds either its key
//     (match → reuse gid) or an empty slot it can CAS-claim.
//
// ⚠ Phase 5b spike result (2026-05-26): THIS KERNEL DOES NOT PRODUCE
// CORRECT RESULTS on the current MSL toolchain (32023.883). See
// `tests/test_groupby_global_hash_smoke.rs` — the
// `ten_thousand_unique_keys_in_hundred_thousand_rows` test observed
// 20,697 groups for 10K unique keys (~2× inflation), meaning the same
// key got registered in multiple slots.
//
// Root cause: MSL only supports `memory_order_relaxed`. The non-atomic
// `slot_key` write between CAS-claim and state-publish is not reliably
// visible to other threads before they observe the state publish. A
// peer thread can read state=gid+1 but read stale slot_key (zeros),
// fail the key match, and probe to the next slot — where it will then
// also write its key.
//
// The kernel did NOT deadlock (Phase 4's spin-wait risk was
// successfully mitigated by hash spreading) — the correctness issue is
// purely memory-ordering.
//
// Unworkable mitigations on this toolchain:
//   - acquire/release/acq_rel orderings: rejected at MSL compile
//   - atomic_thread_fence: only accepts memory_order_relaxed (no fence semantics)
//   - threadgroup_barrier(mem_flags::mem_device): works but requires
//     non-divergent control flow; our threads return at different points
//   - 128-bit atomic key+gid pack: MSL has no atomic_ulong2
//
// Path forward: M4 when Apple ships acquire/release in MSL, OR a
// fundamentally different algorithm (e.g. 2-pass with TGSM reduce).
//
// The kernel is kept in the tree as a documented experiment + reference
// for the M4 follow-up. The smoke test that fails serves as the
// regression-detection signal if/when Apple ships better atomics.

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant uint SLOT_EMPTY    = 0u;
constant uint SLOT_CLAIMING = 0xFFFFFFFFu;

static inline uint64_t hash_u128(ulong2 key) {
    uint64_t h = 0x9E3779B97F4A7C15ull;
    h ^= key.x * 0xBF58476D1CE4E5B9ull;
    h ^= key.y * 0x94D049BB133111EBull;
    h ^= h >> 31;
    return h * 0x9E3779B97F4A7C15ull;
}

kernel void global_hash_build(
    device const ulong2*   keys           [[buffer(0)]],
    device ulong2*         slot_key       [[buffer(1)]],   // [table_size]
    device atomic_uint*    slot_state     [[buffer(2)]],   // [table_size]
    device atomic_uint*    next_group_id  [[buffer(3)]],   // [1]
    device atomic_uint*    overflow_flag  [[buffer(4)]],   // [1]
    device uint*           row_to_group   [[buffer(5)]],   // [n_rows]
    constant uint&         n_rows         [[buffer(6)]],
    constant uint&         table_size     [[buffer(7)]],   // power-of-two
    constant uint&         max_probe      [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;

    ulong2 key = keys[gid];
    uint slot = (uint)(hash_u128(key) & (uint64_t)(table_size - 1u));

    for (uint probe = 0; probe < max_probe; probe++) {
        uint state = atomic_load_explicit(&slot_state[slot], memory_order_relaxed);

        if (state == SLOT_EMPTY) {
            // Try to claim this slot via CAS. MSL only has weak CAS which
            // can spuriously fail even when the expected value matches —
            // retry until we either win or observe a real non-EMPTY state.
            uint expected = SLOT_EMPTY;
            bool won = false;
            while (state == SLOT_EMPTY) {
                expected = SLOT_EMPTY;
                if (atomic_compare_exchange_weak_explicit(
                        &slot_state[slot], &expected, SLOT_CLAIMING,
                        memory_order_relaxed, memory_order_relaxed)) {
                    won = true;
                    break;
                }
                state = expected;
            }
            if (won) {
                // We won. Write the key, allocate a group_id, publish.
                slot_key[slot] = key;
                uint gid_local = atomic_fetch_add_explicit(
                    &next_group_id[0], 1u, memory_order_relaxed);
                atomic_store_explicit(
                    &slot_state[slot], gid_local + 1u, memory_order_relaxed);
                row_to_group[gid] = gid_local;
                return;
            }
        }

        // Spin while a peer is in the middle of claiming this slot.
        // ⚠ SIMD-lockstep deadlock risk if multiple lanes in the same
        // warp hash to the same slot (the claimer is stalled by the
        // spinners). Mitigated by: at high cardinality the hash spreads
        // collisions across slots so within-warp slot contention is rare.
        while (state == SLOT_CLAIMING) {
            state = atomic_load_explicit(&slot_state[slot], memory_order_relaxed);
        }

        // State now holds a published gid+1. Compare keys.
        ulong2 sk = slot_key[slot];
        if (sk.x == key.x && sk.y == key.y) {
            row_to_group[gid] = state - 1u;
            return;
        }

        // Different key; probe next slot.
        slot = (slot + 1u) & (table_size - 1u);
    }

    // Probe budget exhausted. Flag overflow; downstream falls back to CPU.
    atomic_store_explicit(overflow_flag, 1u, memory_order_relaxed);
    row_to_group[gid] = UINT_MAX;
}
