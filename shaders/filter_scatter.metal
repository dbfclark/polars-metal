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
