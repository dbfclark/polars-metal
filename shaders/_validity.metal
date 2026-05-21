// shaders/_validity.metal
//
// Shared MSL helpers for bit-packed Arrow validity bitmaps.
// Each row's validity bit lives at byte (row / 8), bit (row % 8).
// Round-up byte count for n_rows is `(n_rows + 7) / 8`.
//
// This file is a HEADER, not a kernel. The leading underscore signals
// to build.rs that it should not be compiled into the metallib directly;
// its definitions are inlined into the kernels that #include it.
//
// Matches the Rust-side conventions in
// `crates/polars-metal-buffer/src/null_bitmap.rs` (little-endian,
// bit-packed, Arrow-aligned).

#pragma once
#include <metal_stdlib>
using namespace metal;

/// Read the validity bit for `row` from a bit-packed bitmap.
inline bool get_valid(device const uint8_t* bitmap, uint row) {
    return ((bitmap[row >> 3u] >> (row & 7u)) & 1u) != 0u;
}

/// Non-atomic single-bit set. Use only when no other thread can race the
/// containing byte — typically for pre-zeroed scratch buffers being filled
/// sequentially. For scatter / multi-thread writes use `set_valid_atomic_or`.
inline void set_valid_nonatomic(device uint8_t* bitmap, uint row, bool v) {
    uint byte = row >> 3u;
    uint bit  = row & 7u;
    if (v) {
        bitmap[byte] |= uint8_t(1u << bit);
    } else {
        bitmap[byte] &= uint8_t(~(1u << bit));
    }
}

/// Atomically OR a single validity bit into a u32-view of the bitmap.
///
/// Callers must:
///   - Allocate the validity buffer with size rounded up to a multiple of 4
///     bytes so the `device atomic_uint*` cast is well-aligned.
///   - Zero-initialize the buffer before dispatch.
///   - Pass the buffer as `device atomic_uint*` (cast at bind time or via
///     reinterpret in the kernel).
///
/// The function never clears bits — it's append-only. Multiple threads may
/// write to the same row's bit (the OR is idempotent at the bit level).
inline void set_valid_atomic_or(device atomic_uint* atomic_words, uint row) {
    uint byte_idx     = row >> 3u;
    uint word_idx     = byte_idx >> 2u;             // 4 bytes per u32 word
    uint bit_in_word  = ((byte_idx & 3u) << 3u) | (row & 7u);
    atomic_fetch_or_explicit(&atomic_words[word_idx],
                             1u << bit_in_word,
                             memory_order_relaxed);
}
