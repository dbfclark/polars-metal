// shaders/filter_scatter.metal
//
// Pass 3 of the filter compaction pipeline (Tasks 10-13). Following Pass 1
// (`filter_predicate_to_u8`, dense keep flags) and Pass 2 (MLX inclusive
// cumsum over the keep flags, producing the prefix sum), this kernel walks
// the source column row-by-row and, for every row where keep is set,
// writes the row's value into the dense output at offset
// `prefix_sum[row] - 1`. The same thread atomically ORs the row's
// validity bit into the output validity bitmap.
//
// Validity write: 8 output rows share a u8 in the output validity bitmap,
// so multiple threads (one per surviving source row) can race the same
// byte. The non-atomic `set_valid_nonatomic` in `_validity.metal` would
// corrupt the bitmap; we use `set_valid_atomic_or` instead, which casts
// the bitmap as a `device atomic_uint*` and uses
// `atomic_fetch_or_explicit`. The caller's contract:
//   - `dst_valid` allocated as a multiple of 4 bytes so the u32 cast is
//     well-aligned (caller-side check in `dispatch_scatter_i64`).
//   - `dst_valid` zero-initialised before dispatch (the OR is
//     append-only; it never clears bits).
//
// Sentinel overrun check: the kernel performs `out_idx = prefix_sum[gid]
// - 1` and a sanity bounds check against `n_out`. If a kernel-logic bug
// (e.g. a buggy prefix sum) computes `out_idx >= n_out`, the kernel
// writes a recognizable sentinel value at `dst_data[n_out]`. The host
// allocates `n_out + 1` slots for exactly this reason and checks the
// sentinel slot post-dispatch. If the sentinel matches, the host returns
// `FilterError::ScatterOverrun` instead of producing silently corrupt
// output.
//
// Threadgroup / grid:
//   - One thread per source row (`n_rows` threads).
//   - `thread_position_in_grid` is bounds-checked against `n_rows`.
//   - Threadgroup width selected by the dispatcher
//     (`CommandQueue::dispatch_1d`).

#include "_validity.metal"

// Sentinel value written at `dst_data[n_out]` if a kernel-logic bug
// produces an out-of-range output index. Chosen to be unlikely to occur
// in real data; the host checks for this exact value post-dispatch.
constant int64_t SCATTER_SENTINEL_I64 = (int64_t)0xDEADBEEFCAFEBABEll;

// f64 variant sentinel: an explicit NaN payload. NaN itself is a valid
// f64 value (the host bit-compares against this exact pattern to
// disambiguate "user's NaN" from "kernel overrun"). Apple Silicon MSL
// does not support `double` in compute kernels, so the f64 scatter
// treats each 8-byte slot as an opaque `ulong` — the sentinel is
// therefore declared as a `ulong` bit pattern alongside its i64 sibling.
constant ulong SCATTER_SENTINEL_F64_BITS = 0x7FFDEADBEEFCAFE0ull;

kernel void filter_scatter_i64(
    device const int64_t*  src_data         [[buffer(0)]],
    device const uint8_t*  src_valid        [[buffer(1)]],
    device const uint8_t*  keep             [[buffer(2)]],
    device const uint32_t* prefix_sum       [[buffer(3)]],
    device       int64_t*  dst_data         [[buffer(4)]],
    device       atomic_uint* dst_valid     [[buffer(5)]],
    constant     uint32_t& n_rows           [[buffer(6)]],
    constant     uint32_t& n_out            [[buffer(7)]],
    uint                   gid              [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    if (keep[gid] == 0u) return;

    // Inclusive prefix sum: row gid (kept) lands at offset prefix_sum[gid] - 1.
    uint32_t out_idx = prefix_sum[gid] - 1u;

    if (out_idx >= n_out) {
        // Kernel-logic bug: write the sentinel at slot n_out (host allocates
        // n_out + 1). The host checks dst_data[n_out] post-dispatch and
        // raises FilterError::ScatterOverrun if it matches.
        dst_data[n_out] = SCATTER_SENTINEL_I64;
        return;
    }

    dst_data[out_idx] = src_data[gid];
    if (get_valid(src_valid, gid)) {
        set_valid_atomic_or(dst_valid, out_idx);
    }
}

// f64 scatter: byte-identical to `filter_scatter_i64` except that the
// source/destination type is f64. We bind these buffers as `device
// const ulong*` / `device ulong*` for two reasons:
//
//   1. Apple Silicon MSL compute kernels do not support `double`. Any
//      use of `double` would fail to compile (or worse, silently coerce
//      to `float` on older toolchains). Treating each slot as an opaque
//      8-byte `ulong` sidesteps the issue entirely.
//   2. The kernel performs no arithmetic on the values — it just copies
//      8 bytes from one location to another and writes a sentinel on
//      overrun. Bit-identical copy preserves NaN payloads, ±Inf, ±0.0,
//      subnormals, and everything else f64 can represent.
//
// The Rust host (`dispatch_scatter_f64`) casts its `&[f64]` slices to
// byte slices before constructing the MTLBuffers; the GPU side reads
// those bytes as `ulong`. Reinterpretation is a no-op at the buffer
// level — the same eight bytes either way.
kernel void filter_scatter_f64(
    device const ulong*    src_data         [[buffer(0)]],
    device const uint8_t*  src_valid        [[buffer(1)]],
    device const uint8_t*  keep             [[buffer(2)]],
    device const uint32_t* prefix_sum       [[buffer(3)]],
    device       ulong*    dst_data         [[buffer(4)]],
    device       atomic_uint* dst_valid     [[buffer(5)]],
    constant     uint32_t& n_rows           [[buffer(6)]],
    constant     uint32_t& n_out            [[buffer(7)]],
    uint                   gid              [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    if (keep[gid] == 0u) return;

    uint32_t out_idx = prefix_sum[gid] - 1u;

    if (out_idx >= n_out) {
        // Same overrun protocol as the i64 variant, but the sentinel is
        // a NaN bit pattern interpreted as `ulong` on the GPU and bit-
        // compared on the host (`f64::to_bits() == SCATTER_SENTINEL_F64_BITS`).
        dst_data[n_out] = SCATTER_SENTINEL_F64_BITS;
        return;
    }

    // Bit-identical copy: the 8-byte `ulong` is whatever the host wrote
    // in (f64, including NaN payloads).
    dst_data[out_idx] = src_data[gid];
    if (get_valid(src_valid, gid)) {
        set_valid_atomic_or(dst_valid, out_idx);
    }
}

// Bool scatter: the source data is bit-packed (one bit per row, Arrow
// layout), AND the destination data is also bit-packed. This is the
// only scatter variant where the data buffer has the multi-thread-same-
// byte race that the validity buffer always has — 8 surviving rows
// share a byte in the output data buffer, and each of those rows is
// written by a distinct thread. The non-atomic `set_valid_nonatomic`
// would race; we use `set_valid_atomic_or` for the data buffer for the
// same reason we use it for the validity buffer.
//
// Caller contract for BOTH `dst_data` and `dst_valid`:
//   - Allocated as a multiple of 4 bytes (for the u32 atomic cast).
//   - Zero-initialised before dispatch (atomic OR is append-only).
//
// There is NO sentinel slot for the data buffer. Every bit value is a
// legitimate data value (0 or 1), so we cannot reserve a "this means
// overrun" pattern. The host instead pre-verifies the prefix-sum
// invariant `prefix_sum[n_rows - 1] == n_out` before dispatch; if the
// invariant holds, no thread can compute `out_idx >= n_out`. The
// `out_idx >= n_out` check below is a defensive early-return — it
// never writes a sentinel, and it never fires under a valid prefix
// sum.
kernel void filter_scatter_bool(
    device const uint8_t*     src_data         [[buffer(0)]],  // bit-packed
    device const uint8_t*     src_valid        [[buffer(1)]],  // bit-packed
    device const uint8_t*     keep             [[buffer(2)]],
    device const uint32_t*    prefix_sum       [[buffer(3)]],
    device       atomic_uint* dst_data         [[buffer(4)]],  // bit-packed, atomic
    device       atomic_uint* dst_valid        [[buffer(5)]],  // bit-packed, atomic
    constant     uint32_t&    n_rows           [[buffer(6)]],
    constant     uint32_t&    n_out            [[buffer(7)]],
    uint                      gid              [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    if (keep[gid] == 0u) return;

    uint32_t out_idx = prefix_sum[gid] - 1u;
    if (out_idx >= n_out) {
        // Defensive: host pre-verifies `prefix_sum[n_rows - 1] == n_out`
        // before dispatch, so this branch should be unreachable. There
        // is no sentinel to write (every bit value is legitimate);
        // early-return silently and rely on the host-side invariant
        // check for safety.
        return;
    }

    // Set output data bit iff source data bit is set.
    if (get_valid(src_data, gid)) {
        set_valid_atomic_or(dst_data, out_idx);
    }
    // Set output validity bit iff source validity bit is set.
    if (get_valid(src_valid, gid)) {
        set_valid_atomic_or(dst_valid, out_idx);
    }
}
