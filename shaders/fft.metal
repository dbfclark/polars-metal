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

// ===================== Recursive batched four-step FFT =====================
// The Rust driver (`fft_pass` in fft.rs) transforms `batch` contiguous signals,
// each length `len` (pow2). When len <= FFT_BASE_MAX it dispatches the batched
// base kernel `fft_stockham_pow2_batched`. Otherwise it four-steps each signal
// with l1 x l2 (l1 = min(1024, ...) <= FFT_BASE_MAX, l2 = len/l1), recursing on
// the length-l2 row FFTs. All sub-FFTs here are FORWARD-only; the driver applies
// inverse via boundary conjugation (host-side) + a single 1/N on readback.
//
// LAYOUT per recursion level (one signal s of length len = l1*l2, row-major):
//   element (i,j) lives at  buf[s*len + i*l2 + j]   (i in [0,l1), j in [0,l2))
//   1. COLUMN FFTs  (fft_fourstep_cols)  length l1, stride l2: batch*l2 groups,
//      group g -> signal s=g/l2, col c=g%l2.
//   2. TWIDDLE      (fft_twiddle_mul)    over batch*len: (s,i,j) -> *= W_len^{i*j}
//      with modulus = len (the CURRENT sub-signal length, NOT the top-level N).
//   3. ROW FFTs     (recurse)            the l1 contiguous rows of length l2 of
//      every signal form batch*l1 contiguous length-l2 signals at buf[0].
//   4. TRANSPOSE    (fft_transpose)      per signal l1 x l2 -> l2 x l1, NOT
//      in-place: out[s*len + j*l1 + i] = in[s*len + i*l2 + j], into scratch,
//      then fft_copy scratch -> buf.
//
// Each of cols dispatches (n_groups * tg_width) threads with threadgroup width
// tg_width, so threadgroup_position_in_grid selects (signal, column). Each
// threadgroup owns a disjoint column, loads it fully into tg memory before
// writing, so in-place over the shared data buffer is safe.

// Batched base case: one threadgroup per signal s (0..batch). Transforms the
// contiguous length-`len` signal at buf[s*len .. s*len+len] via a forward
// Stockham. FORWARD only — no inverse/scale (the boundary handles inverse).
kernel void fft_stockham_pow2_batched(
    device float2*  data [[buffer(0)]],
    constant uint&  len  [[buffer(1)]],
    uint tgid           [[threadgroup_position_in_grid]],
    uint tid            [[thread_position_in_threadgroup]],
    uint tg_size        [[threads_per_threadgroup]]) {
    threadgroup float2 a[FFT_BASE_MAX];
    threadgroup float2 b[FFT_BASE_MAX];
    uint base = tgid * len;  // signal s = tgid
    for (uint i = tid; i < len; i += tg_size) a[i] = data[base + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    stockham_pow2_tg(a, b, len, tid, tg_size);
    for (uint i = tid; i < len; i += tg_size) data[base + i] = a[i];
}

// COLUMN FFTs (batched). One threadgroup per (signal s, column c): tgid = s*l2 + c.
// Loads the l1 elements at base s*len + c, stride l2; forward length-l1 Stockham;
// writes back strided. l1 <= FFT_BASE_MAX (caller guarantees).
kernel void fft_fourstep_cols(
    device float2*  data [[buffer(0)]],
    constant uint&  l1   [[buffer(1)]],
    constant uint&  l2   [[buffer(2)]],
    constant uint&  batch [[buffer(3)]],
    uint tgid           [[threadgroup_position_in_grid]],
    uint tid            [[thread_position_in_threadgroup]],
    uint tg_size        [[threads_per_threadgroup]]) {
    (void)batch;  // unused: signal index is tgid/l2; arg kept for uniform binding layout across batched kernels
    threadgroup float2 a[FFT_BASE_MAX];
    threadgroup float2 b[FFT_BASE_MAX];
    uint s   = tgid / l2;       // signal index
    uint col = tgid % l2;       // column index in [0, l2)
    uint base = s * (l1 * l2) + col;
    for (uint i = tid; i < l1; i += tg_size) a[i] = data[base + i * l2];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    stockham_pow2_tg(a, b, l1, tid, tg_size);
    for (uint i = tid; i < l1; i += tg_size) data[base + i * l2] = a[i];
}

// TWIDDLE (batched). For element at global idx in [0, batch*len): s=idx/len,
// loc=idx%len, i=loc/l2, j=loc%l2; data[idx] *= W_len^{i*j} (forward sign,
// modulus = len = l1*l2 of the CURRENT sub-signal). Grid-strided.
kernel void fft_twiddle_mul(
    device float2*  data [[buffer(0)]],
    constant uint&  l2   [[buffer(1)]],
    constant uint&  len  [[buffer(2)]],
    constant uint&  batch [[buffer(3)]],
    uint gid            [[thread_position_in_grid]],
    uint grid_size      [[threads_per_grid]]) {
    uint total = batch * len;
    for (uint idx = gid; idx < total; idx += grid_size) {
        uint loc = idx % len;
        uint i = loc / l2;
        uint j = loc % l2;
        data[idx] = cmul(data[idx], twiddle(int(i * j), int(len), false));
    }
}

// TRANSPOSE (batched). Per signal s, l1 x l2 -> l2 x l1:
//   out[s*len + j*l1 + i] = in[s*len + i*l2 + j].
// One thread per source element, grid-strided over batch*len elements. NOT
// in-place (out must differ from in).
kernel void fft_transpose(
    device const float2* in    [[buffer(0)]],
    device float2*       out   [[buffer(1)]],
    constant uint&       l1    [[buffer(2)]],
    constant uint&       l2    [[buffer(3)]],
    constant uint&       batch [[buffer(4)]],
    uint gid                   [[thread_position_in_grid]],
    uint grid_size             [[threads_per_grid]]) {
    uint len = l1 * l2;
    uint total = batch * len;
    for (uint idx = gid; idx < total; idx += grid_size) {
        uint s   = idx / len;
        uint loc = idx % len;
        uint i = loc / l2;   // source row
        uint j = loc % l2;   // source col
        out[s * len + j * l1 + i] = in[idx];
    }
}

// Straight copy of `count` float2 elements (in -> out). Grid-strided. Used to
// fold the transpose scratch buffer back into the data buffer.
kernel void fft_copy(
    device const float2* in    [[buffer(0)]],
    device float2*       out   [[buffer(1)]],
    constant uint&       count [[buffer(2)]],
    uint gid                   [[thread_position_in_grid]],
    uint grid_size             [[threads_per_grid]]) {
    for (uint idx = gid; idx < count; idx += grid_size) {
        out[idx] = in[idx];
    }
}

// ============ PLANAR (SoA) four-step kernels (M5c-2) ============
// One-for-one planar twins of the interleaved four-step kernels above. The
// index math (base, stride, transpose location, twiddle modulus) is IDENTICAL
// to the interleaved versions — the ONLY change is the global (device) buffer
// I/O: separate re/im planes instead of one interleaved float2 buffer. The
// internal threadgroup `float2` working buffers and the (shared, UNCHANGED)
// `stockham_pow2_tg` helper are byte-for-byte the same. The Rust four-step
// driver (`fft_pass_planar` in fft.rs) threads both re/im data planes and both
// re/im scratch planes through the recursion; a differential test verifies the
// planar core matches the interleaved `fft_gpu` to L2 < 1e-3.

// PLANAR batched base case. Mirrors fft_stockham_pow2_batched: one threadgroup
// per signal s = tgid, contiguous length-`len` forward Stockham. The cooperative
// load reads data_re[base+i]/data_im[base+i] into a float2; the store splits the
// result back. Buffers: data_re=0, data_im=1, len=2.
kernel void fft_stockham_pow2_batched_planar(
    device float*   data_re [[buffer(0)]],
    device float*   data_im [[buffer(1)]],
    constant uint&  len     [[buffer(2)]],
    uint tgid              [[threadgroup_position_in_grid]],
    uint tid              [[thread_position_in_threadgroup]],
    uint tg_size          [[threads_per_threadgroup]]) {
    threadgroup float2 a[FFT_BASE_MAX];
    threadgroup float2 b[FFT_BASE_MAX];
    uint base = tgid * len;  // signal s = tgid
    for (uint i = tid; i < len; i += tg_size) {
        a[i] = float2(data_re[base + i], data_im[base + i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    stockham_pow2_tg(a, b, len, tid, tg_size);
    for (uint i = tid; i < len; i += tg_size) {
        data_re[base + i] = a[i].x;
        data_im[base + i] = a[i].y;
    }
}

// PLANAR column FFTs. Mirrors fft_fourstep_cols: one threadgroup per (signal s,
// column c), tgid = s*l2 + c; loads the l1 elements at base s*len + c, stride l2,
// runs a forward length-l1 Stockham, writes back strided. The strided load reads
// data_re[base+i*l2]/data_im[base+i*l2] into a float2; the store splits back.
// Buffers: data_re=0, data_im=1, l1=2, l2=3, batch=4.
kernel void fft_fourstep_cols_planar(
    device float*   data_re [[buffer(0)]],
    device float*   data_im [[buffer(1)]],
    constant uint&  l1      [[buffer(2)]],
    constant uint&  l2      [[buffer(3)]],
    constant uint&  batch   [[buffer(4)]],
    uint tgid             [[threadgroup_position_in_grid]],
    uint tid             [[thread_position_in_threadgroup]],
    uint tg_size         [[threads_per_threadgroup]]) {
    (void)batch;  // unused: signal index is tgid/l2; arg kept for uniform binding layout
    threadgroup float2 a[FFT_BASE_MAX];
    threadgroup float2 b[FFT_BASE_MAX];
    uint s   = tgid / l2;       // signal index
    uint col = tgid % l2;       // column index in [0, l2)
    uint base = s * (l1 * l2) + col;
    for (uint i = tid; i < l1; i += tg_size) {
        a[i] = float2(data_re[base + i * l2], data_im[base + i * l2]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    stockham_pow2_tg(a, b, l1, tid, tg_size);
    for (uint i = tid; i < l1; i += tg_size) {
        data_re[base + i * l2] = a[i].x;
        data_im[base + i * l2] = a[i].y;
    }
}

// PLANAR cross-twiddle. Mirrors fft_twiddle_mul: in-place read-modify-write,
// data[idx] *= W_len^{i*j} (forward sign, modulus = len of the CURRENT
// sub-signal). Reads the planar pair into a float2, multiplies, splits back.
// Buffers: data_re=0, data_im=1, l2=2, len=3, batch=4.
kernel void fft_twiddle_mul_planar(
    device float*   data_re [[buffer(0)]],
    device float*   data_im [[buffer(1)]],
    constant uint&  l2      [[buffer(2)]],
    constant uint&  len     [[buffer(3)]],
    constant uint&  batch   [[buffer(4)]],
    uint gid              [[thread_position_in_grid]],
    uint grid_size        [[threads_per_grid]]) {
    uint total = batch * len;
    for (uint idx = gid; idx < total; idx += grid_size) {
        uint loc = idx % len;
        uint i = loc / l2;
        uint j = loc % l2;
        float2 v = float2(data_re[idx], data_im[idx]);
        v = cmul(v, twiddle(int(i * j), int(len), false));
        data_re[idx] = v.x;
        data_im[idx] = v.y;
    }
}

// PLANAR transpose. Mirrors fft_transpose: per signal s, l1 x l2 -> l2 x l1,
//   out[s*len + j*l1 + i] = in[s*len + i*l2 + j]
// applied to BOTH planes. One thread per source element, grid-strided over
// batch*len. NOT in-place (out planes must differ from in planes). Buffers:
// in_re=0, in_im=1, out_re=2, out_im=3, l1=4, l2=5, batch=6.
kernel void fft_transpose_planar(
    device const float* in_re  [[buffer(0)]],
    device const float* in_im  [[buffer(1)]],
    device float*       out_re [[buffer(2)]],
    device float*       out_im [[buffer(3)]],
    constant uint&      l1     [[buffer(4)]],
    constant uint&      l2     [[buffer(5)]],
    constant uint&      batch  [[buffer(6)]],
    uint gid                   [[thread_position_in_grid]],
    uint grid_size             [[threads_per_grid]]) {
    uint len = l1 * l2;
    uint total = batch * len;
    for (uint idx = gid; idx < total; idx += grid_size) {
        uint s   = idx / len;
        uint loc = idx % len;
        uint i = loc / l2;   // source row
        uint j = loc % l2;   // source col
        uint dst = s * len + j * l1 + i;
        out_re[dst] = in_re[idx];
        out_im[dst] = in_im[idx];
    }
}

// PLANAR copy. Mirrors fft_copy: straight copy of `count` elements (in -> out),
// grid-strided, applied to BOTH planes. Used to fold the transpose scratch
// planes back into the data planes. Buffers: in_re=0, in_im=1, out_re=2,
// out_im=3, count=4.
kernel void fft_copy_planar(
    device const float* in_re  [[buffer(0)]],
    device const float* in_im  [[buffer(1)]],
    device float*       out_re [[buffer(2)]],
    device float*       out_im [[buffer(3)]],
    constant uint&      count  [[buffer(4)]],
    uint gid                   [[thread_position_in_grid]],
    uint grid_size             [[threads_per_grid]]) {
    for (uint idx = gid; idx < count; idx += grid_size) {
        out_re[idx] = in_re[idx];
        out_im[idx] = in_im[idx];
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

// PLANAR (SoA) variant of fft_stockham_pow2_f32. Identical butterfly math; the
// ONLY difference is the global (device) buffer I/O: separate re/im planes
// instead of one interleaved float2 buffer. The internal threadgroup `float2`
// working buffers and the radix-2 Stockham butterfly loop are byte-for-byte the
// same as the interleaved kernel above — only the cooperative load (reads
// in_re[i]/in_im[i] into a float2) and the final store (splits the float2 back
// into out_re[i]/out_im[i]) change. Buffer indices shift +2 vs interleaved
// (two extra planes): in_re=0, in_im=1, out_re=2, out_im=3, n=4, inv=5.
kernel void fft_stockham_pow2_planar_f32(
    device const float* in_re  [[buffer(0)]],
    device const float* in_im  [[buffer(1)]],
    device float*       out_re [[buffer(2)]],
    device float*       out_im [[buffer(3)]],
    constant uint&      n      [[buffer(4)]],
    constant uint&      inv    [[buffer(5)]],
    uint tid                   [[thread_position_in_threadgroup]],
    uint tg_size               [[threads_per_threadgroup]]) {
    threadgroup float2 a[FFT_BASE_MAX];
    threadgroup float2 b[FFT_BASE_MAX];
    // cooperative load (planar -> float2)
    for (uint i = tid; i < n; i += tg_size) a[i] = float2(in_re[i], in_im[i]);
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
    for (uint i = tid; i < n; i += tg_size) {
        float2 vv = src[i] * scale;
        out_re[i] = vv.x;
        out_im[i] = vv.y;
    }
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

// PLANAR (SoA) variant of fft_mixed_radix_f32. Identical butterfly math (gather +
// forward twiddle, radix_dispatch codelet, scatter) and identical inverse-by-
// conjugation; the ONLY difference is the global buffer I/O — separate re/im
// planes instead of one interleaved float2 buffer. The cooperative load reads
// in_re[i]/in_im[i] into a float2 (conjugating on load if inverse), and the store
// splits the float2 back into out_re[i]/out_im[i] (conjugating + scaling if
// inverse). The threadgroup working buffers and the mixed-radix Stockham loop are
// byte-for-byte the same as the interleaved kernel above. Buffer indices shift +2
// vs interleaved (two extra planes): in_re=0, in_im=1, out_re=2, out_im=3, n=4,
// inv=5, radices=6, n_radices=7.
kernel void fft_mixed_radix_planar_f32(
    device const float* in_re     [[buffer(0)]],
    device const float* in_im     [[buffer(1)]],
    device float*       out_re    [[buffer(2)]],
    device float*       out_im    [[buffer(3)]],
    constant uint&      n         [[buffer(4)]],
    constant uint&      inv       [[buffer(5)]],
    constant uint*      radices   [[buffer(6)]],
    constant uint&      n_radices [[buffer(7)]],
    uint tid                      [[thread_position_in_threadgroup]],
    uint tg_size                  [[threads_per_threadgroup]]) {
    threadgroup float2 a[FFT_BASE_MAX];
    threadgroup float2 b[FFT_BASE_MAX];
    bool inverse = inv != 0;
    // cooperative load (planar -> float2); conjugate on load if inverse
    for (uint i = tid; i < n; i += tg_size) {
        float2 v = float2(in_re[i], in_im[i]);
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

    // store; conjugate + 1/n scale if inverse (planar split)
    float scale = inverse ? (1.0f / float(n)) : 1.0f;
    for (uint i = tid; i < n; i += tg_size) {
        float2 v = src[i];
        float2 vv = inverse ? float2(v.x, -v.y) * scale : v * scale;
        out_re[i] = vv.x;
        out_im[i] = vv.y;
    }
}
