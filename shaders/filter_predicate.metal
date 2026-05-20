// shaders/filter_predicate.metal
//
// Pass 1 of the filter compaction pipeline (see docs/superpowers/specs/
// 2026-05-20-m1-design.md §"Predicate compaction kernels").
//
// Reads a bit-packed Boolean column + its bit-packed validity bitmap and
// produces a dense u8 array where keep[i] = 1 iff (data[i] is true) AND
// (valid[i] is true), else 0. The output feeds MLX cumsum for the prefix
// sum that drives the scatter pass.
//
// Why dense u8 output? MLX cumsum requires contiguous numeric input.
// Bit-packed input would save 8x the memory but cost us MLX's well-tuned
// scan. Trade-off documented in the M1 spec.
//
// Threadgroup / grid:
//   - One thread per row.
//   - `thread_position_in_grid` is bounds-checked against `n_rows` so
//     `dispatchThreads:` padding does not write out-of-range.
//   - Threadgroup width is selected by the dispatcher via the default
//     auto-sizing path (`CommandQueue::dispatch_1d`).

#include "_validity.metal"

kernel void filter_predicate_to_u8(
    device const uint8_t* pred_data   [[buffer(0)]],
    device const uint8_t* pred_valid  [[buffer(1)]],
    device       uint8_t* keep_flags  [[buffer(2)]],
    constant     uint32_t& n_rows     [[buffer(3)]],
    uint                  gid         [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;
    // Both inputs are bit-packed bitmaps with identical byte/bit layout, so
    // the validity helper applies to BOTH the data column and the validity
    // mask. The ternary forces the output to exactly 0/1 — required because
    // MLX cumsum sums these bytes; any value >1 would corrupt the prefix
    // sum.
    bool d = get_valid(pred_data, gid);
    bool v = get_valid(pred_valid, gid);
    keep_flags[gid] = (d && v) ? 1u : 0u;
}
