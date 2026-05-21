// shaders/groupby_hash.metal
//
// Hash kernel — one thread per row, reads a u128-encoded composite key
// from `keys`, writes a u32 hash to `hashes[gid]`.
//
// Deterministic: same input bytes always produce the same output hash.
//
// Buffer layout:
//   buffer(0)  keys      device const uint64_t*   — 2 × u64 per row (lo, hi)
//   buffer(1)  hashes    device       uint32_t*   — one hash per row (output)
//   buffer(2)  n_rows    constant uint32_t&       — total row count
//
// u128 encoding: the Rust side packs each u128 key as a contiguous 16-byte
// little-endian blob. Reading as `uint64_t[2*gid]` gives lo at index
// `2*gid` and hi at index `2*gid + 1` (LE, same byte order as u64 on
// Apple Silicon).
//
// Threadgroup/grid: one thread per row; build.rs / dispatch_1d handles
// grid sizing at runtime by querying maxTotalThreadsPerThreadgroup.

#include "_groupby.metal"

kernel void groupby_hash(
    device const uint64_t*   keys    [[buffer(0)]],
    device       uint32_t*   hashes  [[buffer(1)]],
    constant     uint32_t&   n_rows  [[buffer(2)]],
    uint                     gid     [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    uint64_t lo = keys[gid * 2u];
    uint64_t hi = keys[gid * 2u + 1u];
    hashes[gid] = hash_u128(lo, hi);
}
