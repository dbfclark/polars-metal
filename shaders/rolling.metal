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
    // Tile: TG_SIZE outputs + (w-1) left-halo elements.
    // Stack-allocated; size is compile-time constant (TG_SIZE + MAX_W
    // = 4352 floats = 17 408 bytes, well within the 32 KB threadgroup limit).
    threadgroup float tile[TG_SIZE + MAX_W];

    uint halo       = w - 1u;
    uint base       = tgid * TG_SIZE;         // first output index of this group
    uint load_count = TG_SIZE + halo;         // inputs needed: [base-halo, base+TG_SIZE)

    // Cooperative load into threadgroup memory: each thread loads a stripe
    // of elements spaced TG_SIZE apart. For the first halo elements, the
    // global source index may be negative (i.e. before the start of the
    // array); those positions are zero-filled (reflecting the "the window
    // doesn't exist yet" contract — the host marks those outputs as null).
    for (uint j = lid; j < load_count; j += TG_SIZE) {
        long src = (long)base - (long)halo + (long)j;
        tile[j] = (src >= 0L && src < (long)n) ? input[src] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid >= n) return;

    // Each thread sums tile[lid .. lid+w) — its w-element window within the
    // shared tile. `lid` = lid of this thread within its threadgroup, so
    // tile[lid] corresponds to input[base], and tile[lid + k] to
    // input[base + k] for k in [0, TG_SIZE). The left halo occupies
    // tile[0..halo], so thread 0 reads tile[0..w) which is
    // input[base-halo .. base+1).
    float s = 0.0f;
    for (uint k = 0u; k < w; ++k) {
        s += tile[lid + k];
    }

    output[gid] = (is_mean != 0u) ? (s / float(w)) : s;
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
