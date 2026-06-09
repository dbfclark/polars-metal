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

// Dispatch the forward radix-r butterfly codelet (r in {2,3,4,5,6,7,8}).
inline void radix_dispatch(uint r, thread float2* x, thread float2* y) {
    switch (r) {
        case 2: radix2(x, y); break;
        case 3: radix3(x, y); break;
        case 4: radix4(x, y); break;
        case 5: radix5(x, y); break;
        case 6: radix6(x, y); break;
        case 7: radix7(x, y); break;
        case 8: radix8(x, y); break;
        default: break;  // unreachable: factorize only ever emits radices in 2..8
    }
}

// One threadgroup transforms one length-n composite signal (n <= FFT_BASE_MAX),
// mixed-radix self-sorting Stockham. `radices` lists the per-stage radices in
// order (product == n), `n_radices` their count. Each radix is in {2,..,8}.
//
// This generalizes the verified radix-2 kernel above: with all radices == 2 it
// reduces exactly to fft_stockham_pow2_f32. `p` tracks the size of completed
// sub-transforms; each stage with radix r does m = n/r butterflies, gathering r
// inputs at stride m, applying forward twiddles W(t*k, r*p), running the codelet,
// and scattering r outputs into contiguous (r*p)-sized slots.
//
// Inverse handled by conjugation at the boundary (ifft(x)=conj(fft(conj(x)))/n);
// the butterflies stay forward-sign.
kernel void fft_mixed_radix_f32(
    device const float2* in        [[buffer(0)]],
    device float2*       out       [[buffer(1)]],
    constant uint&       n         [[buffer(2)]],
    constant uint&       inv       [[buffer(3)]],
    constant uint*       radices   [[buffer(4)]],
    constant uint&       n_radices [[buffer(5)]],
    uint tid                       [[thread_position_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]) {
    threadgroup float2 a[FFT_BASE_MAX];
    threadgroup float2 b[FFT_BASE_MAX];
    bool inverse = inv != 0;
    // cooperative load; conjugate on load if inverse
    for (uint i = tid; i < n; i += tg_size) {
        float2 v = in[i];
        a[i] = inverse ? float2(v.x, -v.y) : v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float2* src = a;
    threadgroup float2* dst = b;
    uint p = 1;
    for (uint s = 0; s < n_radices; ++s) {
        uint r = radices[s];
        uint m = n / r;  // number of butterflies this stage
        for (uint i = tid; i < m; i += tg_size) {
            uint k = i % p;     // twiddle index within sub-transform
            uint blk = i / p;   // which sub-transform block
            float2 x[8];
            float2 y[8];
            // gather + forward twiddle (t=0 is multiply-by-1)
            for (uint t = 0; t < r; ++t) {
                float2 v = src[i + t * m];
                if (t != 0) {
                    v = cmul(v, twiddle(int(t * k), int(r * p), false));
                }
                x[t] = v;
            }
            radix_dispatch(r, x, y);
            // scatter into contiguous (r*p)-sized output slot
            uint base = blk * r * p + k;
            for (uint t = 0; t < r; ++t) {
                dst[base + t * p] = y[t];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float2* tmp = src; src = dst; dst = tmp;
        p *= r;
    }

    // store; conjugate + 1/n scale if inverse
    float scale = inverse ? (1.0f / float(n)) : 1.0f;
    for (uint i = tid; i < n; i += tg_size) {
        float2 v = src[i];
        out[i] = inverse ? float2(v.x, -v.y) * scale : v * scale;
    }
}
