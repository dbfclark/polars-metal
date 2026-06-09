// shaders/fft.metal — hand-rolled FFT kernels (M6 A3).
#include "_fft_radix.metal"
using namespace metal;

// 1024 complex points × 8 B × 2 ping-pong buffers = 16 KB threadgroup memory,
// safely under Apple Silicon's 32 KB limit. (4096 would need 64 KB — too big.)
constant uint FFT_BASE_MAX = 1024;

// One threadgroup transforms one length-n pow2 signal (n <= FFT_BASE_MAX),
// iterative Stockham radix-2. Input/output interleaved float2 in global mem.
// `n` and `inv` (0/1) are scalar buffers. tg memory holds 2 working buffers.
//
// This is the self-sorting (auto-sort) Stockham layout: each stage reads the
// butterfly pair at a fixed half-n stride from `src` and scatters the results
// into contiguous sub-transform slots of `dst`, so the final pass lands in
// natural order — no bit-reversal permutation. `ns` is the current
// sub-transform size (1,2,4,…,n/2); `j` is the twiddle index within it.
kernel void fft_stockham_pow2_f32(
    device const float2* in   [[buffer(0)]],
    device float2*       out  [[buffer(1)]],
    constant uint&       n    [[buffer(2)]],
    constant uint&       inv  [[buffer(3)]],
    uint tid                  [[thread_position_in_threadgroup]],
    uint tg_size              [[threads_per_threadgroup]]) {
    threadgroup float2 a[FFT_BASE_MAX];
    threadgroup float2 b[FFT_BASE_MAX];
    // cooperative load
    for (uint i = tid; i < n; i += tg_size) a[i] = in[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float2* src = a;
    threadgroup float2* dst = b;
    bool inverse = inv != 0;
    uint nhalf = n >> 1;
    for (uint ns = 1; ns < n; ns <<= 1) {
        for (uint i = tid; i < nhalf; i += tg_size) {
            uint j = i & (ns - 1);        // index within the sub-transform
            uint block = i / ns;          // which sub-transform
            // read pair at stride n/2 from src
            float2 u = src[i];
            float2 v = cmul(src[i + nhalf], twiddle(int(j), int(2 * ns), inverse));
            // scatter contiguously into dst's 2*ns-sized output slot
            uint out0 = block * (2 * ns) + j;
            dst[out0]      = u + v;
            dst[out0 + ns] = u - v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float2* t = src; src = dst; dst = t;
    }
    float scale = inverse ? (1.0f / float(n)) : 1.0f;
    for (uint i = tid; i < n; i += tg_size) out[i] = src[i] * scale;
}
