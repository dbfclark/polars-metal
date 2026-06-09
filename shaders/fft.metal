// shaders/fft.metal — hand-rolled FFT kernels (M6 A3).
#include "_fft_radix.metal"
using namespace metal;

// 1024 complex points × 8 B × 2 ping-pong buffers = 16 KB threadgroup memory,
// safely under Apple Silicon's 32 KB limit. (4096 would need 64 KB — too big.)
constant uint FFT_BASE_MAX = 1024;

// ---- Shared forward radix-2 Stockham core (used by the four-step kernels). ----
// Transforms `len` (pow2, <= FFT_BASE_MAX) interleaved float2 points already
// cooperatively loaded into `a[0..len)`, using `b[0..len)` as ping-pong scratch.
// FORWARD only (no inverse flag, no 1/n scaling) — the four-step driver handles
// inverse purely by boundary conjugation + a single 1/N at the very end.
//
// NOTE: this butterfly body is intentionally duplicated from the standalone
// kernel `fft_stockham_pow2_f32` below (which adds the inverse flag + scaling).
// A change to the radix-2 Stockham butterfly math must touch BOTH.
//
// Contract: on return the result is always in `a[0..len)`. After log2(len)
// stages the data may land in `a` or `b` depending on parity; we copy back to
// `a` when it ends in `b` so the caller has one fixed read location. The caller
// must barrier after its load into `a` before calling, and barrier after this
// returns before reading `a` (this routine ends each stage with a barrier, and
// the final copy-back ends with a barrier too).
inline void stockham_pow2_tg(threadgroup float2* a, threadgroup float2* b,
                             uint len, uint tid, uint tg_size) {
    threadgroup float2* src = a;
    threadgroup float2* dst = b;
    uint lhalf = len >> 1;
    uint stages = 0;
    for (uint ns = 1; ns < len; ns <<= 1) {
        for (uint i = tid; i < lhalf; i += tg_size) {
            uint j = i & (ns - 1);
            uint block = i / ns;
            float2 u = src[i];
            float2 v = cmul(src[i + lhalf], twiddle(int(j), int(2 * ns), false));
            uint out0 = block * (2 * ns) + j;
            dst[out0]      = u + v;
            dst[out0 + ns] = u - v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float2* t = src; src = dst; dst = t;
        ++stages;
    }
    // After the swap, `src` points at the buffer holding the result. If that is
    // `b` (odd number of stages), copy back to `a`.
    if ((stages & 1u) != 0u) {
        for (uint i = tid; i < len; i += tg_size) a[i] = b[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ============================ Four-step FFT ============================
// For N = n1*n2 (both pow2, each <= FFT_BASE_MAX), view interleaved input as an
// n1 x n2 row-major matrix M[i][j] = x[i*n2 + j]. Bailey's four-step:
//   1. fft_fourstep_cols : length-n1 FFT down each of the n2 columns (stride n2)
//   2. fft_twiddle_mul   : multiply element at flat idx=i*n2+j by W_N^{i*j}
//   3. fft_fourstep_rows : length-n2 FFT across each of the n1 rows (contiguous)
//   4. fft_transpose     : out[j*n1 + i] = in[i*n2 + j]   (n1 x n2 -> n2 x n1)
// All sub-FFTs are FORWARD-only; the driver applies inverse via boundary
// conjugation (host-side) + a single 1/N on readback.
//
// Each of cols/rows is dispatched as (n_groups * tg_width) threads with
// threadgroup width tg_width, so threadgroup_position_in_grid selects the
// column/row. Each threadgroup owns a disjoint column/row, loads it fully into
// tg memory before writing, so in-place over the shared data buffer is safe.

// One threadgroup per column j (0..n2). Loads its n1 elements at stride n2,
// base offset j, runs a forward length-n1 Stockham, writes back strided.
kernel void fft_fourstep_cols(
    device float2*  data [[buffer(0)]],
    constant uint&  n1   [[buffer(1)]],
    constant uint&  n2   [[buffer(2)]],
    uint tgid           [[threadgroup_position_in_grid]],
    uint tid            [[thread_position_in_threadgroup]],
    uint tg_size        [[threads_per_threadgroup]]) {
    threadgroup float2 a[FFT_BASE_MAX];
    threadgroup float2 b[FFT_BASE_MAX];
    uint col = tgid;  // column index in [0, n2)
    // strided load: M[i][col] = data[i*n2 + col]
    for (uint i = tid; i < n1; i += tg_size) a[i] = data[i * n2 + col];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    stockham_pow2_tg(a, b, n1, tid, tg_size);
    // strided store back
    for (uint i = tid; i < n1; i += tg_size) data[i * n2 + col] = a[i];
}

// Elementwise twiddle: data[idx] *= W_N^{i*j}, i=idx/n2, j=idx%n2, forward sign.
// Grid-strided one thread per element over all N = n1*n2 elements.
kernel void fft_twiddle_mul(
    device float2*  data [[buffer(0)]],
    constant uint&  n2   [[buffer(1)]],
    constant uint&  ntot [[buffer(2)]],
    uint gid            [[thread_position_in_grid]],
    uint grid_size      [[threads_per_grid]]) {
    for (uint idx = gid; idx < ntot; idx += grid_size) {
        uint i = idx / n2;
        uint j = idx % n2;
        data[idx] = cmul(data[idx], twiddle(int(i * j), int(ntot), false));
    }
}

// One threadgroup per row i (0..n1). Loads its contiguous n2 elements at
// base i*n2, runs a forward length-n2 Stockham, writes back contiguous.
kernel void fft_fourstep_rows(
    device float2*  data [[buffer(0)]],
    constant uint&  n1   [[buffer(1)]],
    constant uint&  n2   [[buffer(2)]],
    uint tgid           [[threadgroup_position_in_grid]],
    uint tid            [[thread_position_in_threadgroup]],
    uint tg_size        [[threads_per_threadgroup]]) {
    threadgroup float2 a[FFT_BASE_MAX];
    threadgroup float2 b[FFT_BASE_MAX];
    uint row = tgid;          // row index in [0, n1)
    uint base = row * n2;
    for (uint j = tid; j < n2; j += tg_size) a[j] = data[base + j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    stockham_pow2_tg(a, b, n2, tid, tg_size);
    for (uint j = tid; j < n2; j += tg_size) data[base + j] = a[j];
}

// Transpose n1 x n2 -> n2 x n1: out[j*n1 + i] = in[i*n2 + j]. One thread per
// element, grid-strided over the n1*n2 source elements (simple, correct).
kernel void fft_transpose(
    device const float2* in   [[buffer(0)]],
    device float2*       out  [[buffer(1)]],
    constant uint&       n1   [[buffer(2)]],
    constant uint&       n2   [[buffer(3)]],
    uint gid                  [[thread_position_in_grid]],
    uint grid_size            [[threads_per_grid]]) {
    uint ntot = n1 * n2;
    for (uint idx = gid; idx < ntot; idx += grid_size) {
        uint i = idx / n2;   // source row
        uint j = idx % n2;   // source col
        out[j * n1 + i] = in[idx];
    }
}

// One threadgroup transforms one length-n pow2 signal (n <= FFT_BASE_MAX),
// iterative Stockham radix-2. Input/output interleaved float2 in global mem.
// `n` and `inv` (0/1) are scalar buffers. tg memory holds 2 working buffers.
//
// This is the self-sorting (auto-sort) Stockham layout: each stage reads the
// butterfly pair at a fixed half-n stride from `src` and scatters the results
// into contiguous sub-transform slots of `dst`, so the final pass lands in
// natural order — no bit-reversal permutation. `ns` is the current
// sub-transform size (1,2,4,…,n/2); `j` is the twiddle index within it.
//
// NOTE: the butterfly body below is intentionally duplicated by the forward-only
// device variant `stockham_pow2_tg` (above), used by the four-step kernels. A
// change to the radix-2 Stockham butterfly math must touch BOTH.
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
