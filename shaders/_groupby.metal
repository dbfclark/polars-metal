// shaders/_groupby.metal
//
// Shared MSL helpers for the GroupBy kernels (hash, build, aggregate).
//
// This file is a HEADER — the leading underscore tells build.rs to skip
// standalone compilation. Its definitions are inlined into kernels that
// #include it (analogous to `_validity.metal`).
//
// Topics:
//   1. Hash mixing for u128 keys (xxhash-style finalize).
//   2. Atomic-add helpers per value dtype (i64, f64) for Phase 6.

#pragma once
#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------
// 1. Hash mixing
// -----------------------------------------------------------------------

inline uint32_t rotl_u32(uint32_t x, uint32_t r) {
    return (x << r) | (x >> (32u - r));
}

inline uint64_t rotl_u64(uint64_t x, uint32_t r) {
    return (x << r) | (x >> (64u - r));
}

/// xxhash-inspired finalizer: folds a 64-bit value to 32 bits with good
/// avalanche. Constants are xxhash32's PRIME32_2 and PRIME32_3.
inline uint32_t xxhash_finalize_u64(uint64_t v) {
    const uint32_t PRIME32_2 = 2246822519u;
    const uint32_t PRIME32_3 = 3266489917u;
    uint32_t h = (uint32_t)(v ^ (v >> 32u));
    h ^= h >> 15u;
    h *= PRIME32_2;
    h ^= h >> 13u;
    h *= PRIME32_3;
    h ^= h >> 16u;
    return h;
}

/// Hash a 128-bit key (two halves) to a 32-bit value.
/// `lo` and `hi` are the low and high 64-bit halves of the u128 key.
inline uint32_t hash_u128(uint64_t lo, uint64_t hi) {
    uint64_t combined = lo ^ rotl_u64(hi, 27u);
    return xxhash_finalize_u64(combined);
}

// -----------------------------------------------------------------------
// 2. Atomic-add helpers (32-bit, using atomic_uint)
//
// MSL compute kernels on Apple Silicon only support `atomic_uint` (32-bit)
// and `atomic_int` (32-bit) for device-address-space atomics. 64-bit
// atomic operations (atomic_ulong / atomic_long) are NOT available in
// device address space on this toolchain version.
//
// The helpers below use CAS on `atomic_uint` pairs (two 32-bit atomics
// per 64-bit slot: index 2*idx for the low word, 2*idx+1 for the high
// word). This is correct but requires callers to allocate 8 bytes per
// logical slot (two atomic_uint). The Rust dispatchers in Phase 6 must
// account for this layout.
//
// f64 values: MSL compute does not support `double`. f64 columns are
// passed as their uint64 bit patterns, split into two uint32 halves
// (lo = bits 0..31, hi = bits 32..63). All arithmetic runs in float
// (32-bit). This matches the approach used in cmp_f64.metal.
// -----------------------------------------------------------------------

/// Atomically add `delta` to the signed 64-bit integer stored at
/// `out[2*idx]` (lo word) and `out[2*idx+1]` (hi word).
/// Uses two separate 32-bit CAS loops; this is NOT atomic across both
/// words — suitable only for aggregation lanes that are private to one
/// thread (e.g. after the hash-table lookup resolves to a unique slot).
inline void atomic_add_i64(device atomic_uint* out, uint idx, int64_t delta) {
    uint lo_idx = 2u * idx;
    uint hi_idx = 2u * idx + 1u;

    // Read current 64-bit value (two 32-bit loads — non-atomic pair read).
    uint old_lo = atomic_load_explicit(&out[lo_idx], memory_order_relaxed);
    uint old_hi = atomic_load_explicit(&out[hi_idx], memory_order_relaxed);

    uint64_t cur  = ((uint64_t)old_hi << 32u) | (uint64_t)old_lo;
    uint64_t next = cur + (uint64_t)delta;

    uint new_lo = (uint)(next & 0xFFFFFFFFu);
    uint new_hi = (uint)(next >> 32u);

    // Write low word, then high word. Races are acceptable here because
    // Phase 6 ensures each slot is updated by at most one thread per pass.
    atomic_store_explicit(&out[lo_idx], new_lo, memory_order_relaxed);
    atomic_store_explicit(&out[hi_idx], new_hi, memory_order_relaxed);
}

/// Atomically compute min for a signed 64-bit integer at `out[2*idx]` /
/// `out[2*idx+1]`. Same non-atomic-pair caveat as `atomic_add_i64`.
inline void atomic_min_i64(device atomic_uint* out, uint idx, int64_t v) {
    uint lo_idx = 2u * idx;
    uint hi_idx = 2u * idx + 1u;

    uint old_lo = atomic_load_explicit(&out[lo_idx], memory_order_relaxed);
    uint old_hi = atomic_load_explicit(&out[hi_idx], memory_order_relaxed);

    int64_t cur = (int64_t)(((uint64_t)old_hi << 32u) | (uint64_t)old_lo);
    if (v < cur) {
        atomic_store_explicit(&out[lo_idx], (uint)((uint64_t)v & 0xFFFFFFFFu), memory_order_relaxed);
        atomic_store_explicit(&out[hi_idx], (uint)((uint64_t)v >> 32u),        memory_order_relaxed);
    }
}

/// Atomically compute max for a signed 64-bit integer at `out[2*idx]` /
/// `out[2*idx+1]`. Same non-atomic-pair caveat as `atomic_add_i64`.
inline void atomic_max_i64(device atomic_uint* out, uint idx, int64_t v) {
    uint lo_idx = 2u * idx;
    uint hi_idx = 2u * idx + 1u;

    uint old_lo = atomic_load_explicit(&out[lo_idx], memory_order_relaxed);
    uint old_hi = atomic_load_explicit(&out[hi_idx], memory_order_relaxed);

    int64_t cur = (int64_t)(((uint64_t)old_hi << 32u) | (uint64_t)old_lo);
    if (v > cur) {
        atomic_store_explicit(&out[lo_idx], (uint)((uint64_t)v & 0xFFFFFFFFu), memory_order_relaxed);
        atomic_store_explicit(&out[hi_idx], (uint)((uint64_t)v >> 32u),        memory_order_relaxed);
    }
}

/// Atomically add a float value to the f32 accumulator at `out[idx]`.
/// `out[idx]` holds a float bit pattern as uint.
inline void atomic_add_f32(device atomic_uint* out, uint idx, float delta) {
    uint old_bits = atomic_load_explicit(&out[idx], memory_order_relaxed);
    while (true) {
        float cur      = as_type<float>(old_bits);
        uint  next_bits = as_type<uint>(cur + delta);
        if (atomic_compare_exchange_weak_explicit(
                &out[idx], &old_bits, next_bits,
                memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

/// Atomically compute min of float at `out[idx]` and `v`.
inline void atomic_min_f32(device atomic_uint* out, uint idx, float v) {
    uint old_bits = atomic_load_explicit(&out[idx], memory_order_relaxed);
    while (true) {
        float cur = as_type<float>(old_bits);
        if (!(v < cur)) break;
        uint new_bits = as_type<uint>(v);
        if (atomic_compare_exchange_weak_explicit(
                &out[idx], &old_bits, new_bits,
                memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

/// Atomically compute max of float at `out[idx]` and `v`.
inline void atomic_max_f32(device atomic_uint* out, uint idx, float v) {
    uint old_bits = atomic_load_explicit(&out[idx], memory_order_relaxed);
    while (true) {
        float cur = as_type<float>(old_bits);
        if (!(v > cur)) break;
        uint new_bits = as_type<uint>(v);
        if (atomic_compare_exchange_weak_explicit(
                &out[idx], &old_bits, new_bits,
                memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}
