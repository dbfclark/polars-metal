// shaders/rolling.metal
//
// Rolling windowed statistics over a 1-D F32 column — tile-blocked variant.
//
// ## Numerical stability
//
// The naive global cumsum-diff approach (`output[i] = cumsum[i] -
// cumsum[i-w]`) suffers catastrophic cancellation when w << N: the two
// cumsum values grow to ~N·mean in magnitude while their difference is
// only ~w·mean, losing ~log2(N/w) bits of precision. This kernel avoids
// that by scoping each accumulation to a per-threadgroup tile: each
// threadgroup processes TG_SIZE consecutive outputs from one input tile
// loaded into threadgroup memory. Accumulation magnitudes stay ~w·mean
// regardless of N.
//
// ## Algorithm (tile-blocked, O(w) per output)
//
// One thread per output element. Each threadgroup owns TG_SIZE consecutive
// outputs [base, base+TG_SIZE) and needs inputs from
// [base-halo, base+TG_SIZE) where halo = w-1. A tile of
// `TG_SIZE + halo` floats is cooperatively loaded into threadgroup memory
// (each thread loads one or more elements in a strided loop), then each
// thread sums its w-element window from the tile. The first w-1 outputs
// (i < w-1) read windows whose left edge would fall before index 0; those
// positions are zero-filled by the loader, so their sum is structurally
// meaningless. The host masks them (sets validity bitmap to null) and
// ignores the kernel output for those rows.
//
// ## Kernel constants
//
//   TG_SIZE = 256   — outputs per threadgroup (≤ 1024 thread cap; 256
//                     chosen for warp-friendly occupancy on all M-series)
//   MAX_W   = 4096  — max supported window; host returns an error if
//                     `w > MAX_W`. Tile size = TG_SIZE + MAX_W = 4352
//                     floats = 17 408 bytes < 32 KB threadgroup limit.
//
// ## Grid
//
//   Dispatch n threads; threadgroup width = TG_SIZE (or less on the last
//   threadgroup). Metal pads the trailing threadgroup with out-of-range
//   `thread_position_in_grid`; the `if (gid >= n) return;` guard exits
//   those threads early.
//
// ## Scalar parameters
//
//   Scalars (n, w, is_mean) are passed as 1-element MetalBuffers bound as
//   `constant uint& x [[buffer(k)]]`, matching the convention used by
//   filter_scatter.metal, cmp_i64.metal, and other kernels in this repo.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint TG_SIZE = 256;   // outputs per threadgroup (<=1024)
constant constexpr uint MAX_W   = 4096;  // host guarantees w <= MAX_W (else CPU)

kernel void rolling_sum_f32(
    device const float* input   [[buffer(0)]],
    device       float* output  [[buffer(1)]],
    constant     uint&  n       [[buffer(2)]],
    constant     uint&  w       [[buffer(3)]],
    constant     uint&  is_mean [[buffer(4)]],
    uint gid  [[thread_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    // ## Tile-local cooperative inclusive prefix-scan, O(N) rolling sum.
    //
    // The old O(N·w) serial loop summed `w` values per output element. This
    // rewrite builds a tile-local inclusive prefix sum P[] over the L-element
    // tile in three cooperative phases (A/B/C), then each output does one O(1)
    // difference: window_sum = P[lid+w-1] - (lid==0 ? 0 : P[lid-1]).
    //
    // Magnitudes stay tile-scale (~L·mean, not N·mean) — same F32-stability
    // property as before; we just avoid the per-thread w-iteration.
    //
    // ## Threadgroup memory budget
    //   tile      : TG_SIZE + MAX_W = 4352 floats = 17 408 B
    //   seg_a/b   : TG_SIZE = 256 floats each      =  2 048 B  (×2 = 4 096 B)
    //   Total                                       = 21 504 B < 32 KB limit
    //
    // ## Grid
    //   Dispatched with n_padded threads (n rounded up to TG_SIZE multiple).
    //   Surplus threads (gid >= n) participate in ALL cooperative phases so
    //   every barrier is reached by every thread in the group.
    //   Only the final output store is guarded by `if (gid < n)`.

    // Shared tile: L = TG_SIZE + (w-1) elements.
    threadgroup float tile[TG_SIZE + MAX_W];

    // Two scratch buffers for the Hillis-Steele ping-pong scan in Phase B.
    // seg_a is the "read" buffer; seg_b is the "write" buffer. We swap roles
    // each pass to avoid read/write races across threads.
    threadgroup float seg_a[TG_SIZE];
    threadgroup float seg_b[TG_SIZE];

    uint halo       = w - 1u;
    uint base       = tgid * TG_SIZE;         // first output index of this group
    uint load_count = TG_SIZE + halo;         // L = inputs needed

    // ── Cooperative tile load ────────────────────────────────────────────────
    // Each thread loads a stripe of tile slots spaced TG_SIZE apart.
    // Slots that map before index 0 or beyond n are zero-filled.
    for (uint j = lid; j < load_count; j += TG_SIZE) {
        long src = (long)base - (long)halo + (long)j;
        tile[j] = (src >= 0L && src < (long)n) ? input[src] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase A: per-thread serial inclusive prefix over each thread's segment ─
    //
    // Divide the L-element tile among TG_SIZE threads.
    // Thread `lid` owns tile indices [lid*seg, min((lid+1)*seg, L)).
    // Running the serial prefix in-place transforms that segment so that each
    // element holds the cumulative sum of all preceding elements in the segment.
    // seg_a[lid] = total of the segment (for the Phase B scan).
    //
    // `seg` = ceil(L / TG_SIZE) — constant across the group for a given w.
    uint seg = (load_count + TG_SIZE - 1u) / TG_SIZE;   // ceil(L / TG_SIZE)
    uint seg_start = lid * seg;
    uint seg_end   = min(seg_start + seg, load_count);   // exclusive

    if (seg_start < load_count) {
        // In-place inclusive prefix over tile[seg_start..seg_end).
        for (uint j = seg_start + 1u; j < seg_end; ++j) {
            tile[j] += tile[j - 1u];
        }
        seg_a[lid] = tile[seg_end - 1u];   // segment total
    } else {
        // This thread's segment is empty (trailing threads when L < TG_SIZE*seg).
        seg_a[lid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase B: Hillis-Steele inclusive scan of seg_a → compute exclusive offsets
    //
    // After Phase A, seg_a[k] holds the total of segment k. We need the
    // exclusive prefix of these totals: off[k] = sum(seg_a[0..k)).
    // We use Hillis-Steele (log2(TG_SIZE) = 8 passes) on TG_SIZE elements.
    // Double-buffering between seg_a and seg_b prevents read/write races.
    //
    // After this phase, seg_a holds the INCLUSIVE scan of the original
    // segment totals. The exclusive offset for thread lid is then:
    //   off = seg_a[lid] - (original) seg_a[lid]
    // We save the original totals into seg_b first, then compute.

    // Save original segment totals into seg_b (used to derive exclusive offset).
    seg_b[lid] = seg_a[lid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis-Steele inclusive scan: at each pass p, every element reads
    // from the element `stride` positions to its left and adds it.
    // We ping-pong: even passes read seg_a, write seg_b (and vice versa),
    // except that after saving originals we need a clear scheme.
    // Cleaner: use seg_a as the "current" scan buffer (starts as the totals),
    // and seg_b as the scratch (ping-pong), carrying the result back to seg_a.
    //
    // Pass structure (8 passes for TG_SIZE=256):
    //   stride = 1, 2, 4, 8, 16, 32, 64, 128
    //   read from seg_a, write to seg_b; then swap.
    // After all passes seg_a holds the inclusive scan of the original totals.
    // The exclusive offset for thread `lid` is: off = (lid == 0) ? 0 : seg_a[lid-1].
    //
    // We need to restore the original totals so we can compute the exclusive
    // offset correctly. They were saved in seg_b at the top of Phase B.
    // Reload seg_a from the saved originals so seg_a starts as totals again.
    seg_a[lid] = seg_b[lid];   // seg_b still holds originals at this point
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis-Steele inclusive scan over seg_a (8 passes).
    for (uint stride = 1u; stride < TG_SIZE; stride <<= 1u) {
        float val = (lid >= stride) ? seg_a[lid - stride] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        seg_b[lid] = seg_a[lid] + val;   // write to seg_b
        threadgroup_barrier(mem_flags::mem_threadgroup);
        seg_a[lid] = seg_b[lid];         // copy back to seg_a
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // seg_a[lid] now holds the inclusive prefix sum of original segment totals.
    // Exclusive offset for thread lid: off = (lid == 0) ? 0.0 : seg_a[lid-1].
    float off = (lid == 0u) ? 0.0f : seg_a[lid - 1u];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase C: add each thread's segment offset to its local prefixes ──────
    //
    // After Phase A, tile[seg_start..seg_end) holds the LOCAL inclusive prefix
    // of that segment. Adding `off` (= sum of all earlier segments) converts
    // each element to the GLOBAL tile-local inclusive prefix P[j].
    if (seg_start < load_count) {
        for (uint j = seg_start; j < seg_end; ++j) {
            tile[j] += off;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Output (one thread per output, O(1) window sum via prefix difference) ─
    //
    // Thread `lid` owns global output index `gid = tgid*TG_SIZE + lid`.
    // Its window covers input[gid-halo .. gid], which maps to
    // tile[lid .. lid+halo], i.e. tile indices [lid, lid+w-1].
    //
    // With the inclusive prefix P now in tile[0..L):
    //   window_sum = P[lid+w-1] - (lid == 0 ? 0 : P[lid-1])
    //             = P[hi] - P_prev
    // where hi = lid + halo = lid + w - 1.
    //
    // Guard: surplus threads (gid >= n) skip the write but reach this point
    // AFTER all barriers — no hang risk.
    if (gid < n) {
        uint hi = lid + halo;        // == lid + w - 1; always < L (= TG_SIZE + halo)
        float p_prev = (lid == 0u) ? 0.0f : tile[lid - 1u];
        float window_sum = tile[hi] - p_prev;
        output[gid] = (is_mean != 0u) ? (window_sum / float(w)) : window_sum;
    }
}

// Rolling windowed variance/std over a 1-D F32 column — centered two-pass
// variant.
//
// ## Numerical stability
//
// A single-pass variance accumulation over a window whose values have a large
// common offset (e.g. all values near 1000.0) suffers catastrophic
// cancellation when the squared-mean term is nearly as large as the
// sum-of-squares term. The centered two-pass approach avoids this: pass 1
// computes the window mean; pass 2 accumulates (x_k - mu)^2 over the same
// tile window. The subtracted residuals are small regardless of the absolute
// magnitude of the input, so F32 precision is preserved.
//
// ## Algorithm
//
// Identical tile-load preamble to `rolling_sum_f32`. After the barrier:
//   Pass 1: sum tile[t0..t0+w] → mu = sum / w
//   Pass 2: sum (tile[t0+k] - mu)^2 for k in [0,w) → ss
//   Output: ss / (w - ddof), or sqrt(that) when is_std != 0
//
// ## Scalar parameters
//
//   buffer(2): n      — row count
//   buffer(3): w      — window width
//   buffer(4): ddof   — degrees-of-freedom correction (1 = sample variance)
//   buffer(5): is_std — nonzero → output sqrt(var)

kernel void rolling_var_f32(
    device const float* input   [[buffer(0)]],
    device       float* output  [[buffer(1)]],
    constant     uint&  n       [[buffer(2)]],
    constant     uint&  w       [[buffer(3)]],
    constant     uint&  ddof    [[buffer(4)]],
    constant     uint&  is_std  [[buffer(5)]],
    uint gid  [[thread_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    // Tile: TG_SIZE outputs + (w-1) left-halo elements.
    // Same compile-time sizing as rolling_sum_f32; 17 408 bytes < 32 KB limit.
    threadgroup float tile[TG_SIZE + MAX_W];

    uint halo       = w - 1u;
    uint base       = tgid * TG_SIZE;
    uint load_count = TG_SIZE + halo;

    // Cooperative tile load — identical to rolling_sum_f32.
    for (uint j = lid; j < load_count; j += TG_SIZE) {
        long src = (long)base - (long)halo + (long)j;
        tile[j] = (src >= 0L && src < (long)n) ? input[src] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid >= n) return;

    // `t0`: index into tile[] for the first element of this thread's window.
    // Thread `lid` within threadgroup `tgid` owns global output at `gid =
    // tgid*TG_SIZE + lid`; its window is input[gid-halo .. gid+1), which maps
    // to tile[lid .. lid+w).
    uint t0 = lid;

    // Pass 1: window mean (keeps all arithmetic in the centered frame).
    float s = 0.0f;
    for (uint k = 0u; k < w; ++k) {
        s += tile[t0 + k];
    }
    float mu = s / float(w);

    // Pass 2: sum of centered squares — cancellation-free because each term
    // (tile[t0+k] - mu) is at most ~max_range of the input, not ~mean.
    float ss = 0.0f;
    for (uint k = 0u; k < w; ++k) {
        float d = tile[t0 + k] - mu;
        ss += d * d;
    }

    // Caller guarantees w > ddof; denom is strictly positive.
    float denom = float(w - ddof);
    float var   = ss / denom;
    output[gid] = (is_std != 0u) ? sqrt(var) : var;
}
