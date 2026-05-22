// shaders/groupby_build.metal
//
// Hash-table build kernel (REFERENCE IMPLEMENTATION — not dispatched).
//
// As of the CPU-build pivot (see dispatch_build doc comment in groupby.rs),
// this kernel is compiled into the Metal library but not invoked at runtime.
// It is retained for:
//   (a) Keeping the build system consistent (build.rs compiles all .metal
//       files; removing this file would require a build.rs change).
//   (b) Future work: a correct GPU build phase would use a different
//       algorithm (e.g. sort-then-scan, or warp-cooperative insert) that
//       avoids the CAS-spin / skip-and-retry failure modes documented in
//       dispatch_build's doc comment.
//
// -----------------------------------------------------------------------
// Why the naive CAS approach fails on Metal
// -----------------------------------------------------------------------
//
// Two competing failure modes prevent a simple concurrent CAS hash table
// from working correctly here:
//
// 1. SIMD-group deadlock (spin-on-CLAIMED design):
//    Threads in the same SIMD-group (warp) that hash to the same slot can
//    deadlock.  The CAS winner (thread X) needs to write its key words and
//    publish READY.  Its SIMD siblings (threads Y, Z, ...) are spinning on
//    the same slot waiting for READY.  On Apple Silicon's scalar GPU, the
//    winning thread CAN execute while siblings spin via hardware predication,
//    but in practice the Metal compiler's loop optimisation and the warp
//    scheduler's prioritisation cause X's stores to be delayed until after
//    the spin timeout fires, producing 0xFFFFFFFF sentinels.
//
// 2. Livelock (skip-and-retry design):
//    Replacing the spin with "skip CLAIMED slots and retry" avoids the
//    deadlock but creates a livelock.  A thread that skips a CLAIMED slot
//    will retry when the slot becomes READY.  BUT: on each retry pass, new
//    CLAIMED slots may have appeared (from other thread-groups), so the
//    thread perpetually sees CLAIMED slots in its probe chain and never
//    finds a claimable EMPTY slot.  This exhausts the retry budget and
//    again produces 0xFFFFFFFF sentinels.
//
// The correct fix for v0.1: run the build phase on CPU (see dispatch_build
// in groupby.rs).  The build phase is not the aggregation bottleneck, and
// CPU HashMap is both correct and sufficient for realistic cardinalities.
//
// -----------------------------------------------------------------------
// Slot layout (for future GPU build work)
// -----------------------------------------------------------------------
//   atomic_uint  slot_state[table_size]     — 0=EMPTY, 1=CLAIMED, 2=READY
//   atomic_uint  slot_key[table_size * 4]   — key words: lo_lo, lo_hi, hi_lo, hi_hi
//   atomic_uint  slot_group_id[table_size]  — group ID (valid only when READY)
//   atomic_uint  group_count[1]             — global group-ID allocator
//   uint32_t     first_row_per_group[n_rows]— representative row per group
//   uint32_t     row_to_group[n_rows]       — output: group ID for each row

#include "_groupby.metal"

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
    // NOT DISPATCHED — see file header for rationale.
    if (gid >= n_rows) return;
    row_to_group[gid] = 0xFFFFFFFFu;
}
