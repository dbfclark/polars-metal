// shaders/fft_pack.metal
//
// GPU pack/unpack helpers for the FFT host path.
//
// The FFT kernel (fft.metal) operates on interleaved-complex buffers
// ([re0,im0,re1,im1,...]). The host path in fft_core currently builds and
// splits this layout on the CPU; these kernels move that O(N) scatter/gather
// work to the GPU so M5b can keep the entire FFT pipeline on-device.
//
// ## Kernels
//
//   fft_pack_real_to_interleaved   — real signal -> interleaved complex (im=0)
//   fft_pack_complex_to_interleaved — separate re+im planes -> interleaved
//   fft_unpack_interleaved_to_planar — interleaved -> separate re+im planes
//
// ## Grid
//
//   One thread per sample; dispatch `n` threads, threadgroup width chosen by
//   dispatch_1d (default 256). The `if (gid >= n) return;` guard handles the
//   trailing partial threadgroup. No threadgroup memory, no cooperation —
//   purely element-wise scatter/gather.
//
// ## Scalar parameters
//
//   Each kernel receives `n` (element count, NOT byte count) as a
//   `constant uint&` in the last buffer slot, matching the pattern used by
//   fft.metal's kernels.

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// fft_pack_real_to_interleaved
//
// Pack a real signal into interleaved complex: out[2i] = re[i], out[2i+1] = 0.
// Input:  re   — length n floats
// Output: out  — length 2n floats
// ---------------------------------------------------------------------------
kernel void fft_pack_real_to_interleaved(
    device const float* re  [[buffer(0)]],
    device       float* out [[buffer(1)]],   // length 2n
    constant     uint&  n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[2 * gid]     = re[gid];
    out[2 * gid + 1] = 0.0f;
}

// ---------------------------------------------------------------------------
// fft_pack_complex_to_interleaved
//
// Pack separate real and imaginary planes into interleaved complex:
//   out[2i] = re[i], out[2i+1] = im[i].
// Input:  re, im — length n floats each
// Output: out    — length 2n floats
// ---------------------------------------------------------------------------
kernel void fft_pack_complex_to_interleaved(
    device const float* re  [[buffer(0)]],
    device const float* im  [[buffer(1)]],
    device       float* out [[buffer(2)]],   // length 2n
    constant     uint&  n   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    out[2 * gid]     = re[gid];
    out[2 * gid + 1] = im[gid];
}

// ---------------------------------------------------------------------------
// fft_unpack_interleaved_to_planar
//
// Unpack interleaved complex into separate real and imaginary planes:
//   re_out[i] = in[2i], im_out[i] = in[2i+1].
// Input:  in     — length 2n floats
// Output: re_out, im_out — length n floats each
// ---------------------------------------------------------------------------
kernel void fft_unpack_interleaved_to_planar(
    device const float* in     [[buffer(0)]],   // length 2n
    device       float* re_out [[buffer(1)]],
    device       float* im_out [[buffer(2)]],
    constant     uint&  n      [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    re_out[gid] = in[2 * gid];
    im_out[gid] = in[2 * gid + 1];
}
