// shaders/groupby_segments.metal
//
// Given sorted (key_lo, key_hi) arrays, mark each row whose key differs
// from the previous row (or row 0) as a "segment start". Bits are written
// into a packed byte buffer; the CPU then scans the bit mask to derive
// per-row group_ids.
//
// Each segment-start bit is OR'd into the buffer via a 32-bit atomic on
// the word containing that bit. This avoids a write-write race when two
// adjacent threads land in the same byte but is conservative — only
// segment-start bits are written, so for sparse-boundary inputs the
// atomic contention is negligible.

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

kernel void segment_starts(
    device const uint64_t* sorted_lo [[buffer(0)]],
    device const uint64_t* sorted_hi [[buffer(1)]],
    device atomic_uint*    starts    [[buffer(2)]],  // bit-packed; size = ceil(n_rows/8) padded to 4
    constant uint&         n_rows    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    bool is_start = (gid == 0) ||
                    (sorted_lo[gid] != sorted_lo[gid - 1]) ||
                    (sorted_hi[gid] != sorted_hi[gid - 1]);
    if (is_start) {
        // The byte holding our bit is starts_bytes[gid >> 3]; bit index
        // is gid & 7. The containing 32-bit word starts at byte
        // (gid >> 5) << 2, and our bit position within that word is
        // gid & 31 — for little-endian byte ordering (Apple Silicon),
        // bit (gid & 7) of byte (gid >> 3) is the same physical bit
        // as bit (gid & 31) of the 32-bit word at byte (gid >> 5) << 2.
        uint word_idx = gid >> 5;
        uint bit_in_word = gid & 31u;
        atomic_fetch_or_explicit(&starts[word_idx], 1u << bit_in_word, memory_order_relaxed);
    }
}
