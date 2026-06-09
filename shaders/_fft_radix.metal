// shaders/_fft_radix.metal — FFT codelets (adapted from vendor/mlx/.../fft/radix.h).
// Header-only ('_' prefix): #included by fft.metal, not compiled standalone.
#pragma once
#include <metal_stdlib>
using namespace metal;

// Complex multiply: (a.x+i a.y)(b.x+i b.y).
inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
// Twiddle e^{-2πi k / p} (forward). For inverse, pass +sign via `inv`.
inline float2 twiddle(int k, int p, bool inv) {
    float theta = (inv ? 2.0f : -2.0f) * float(k) * M_PI_F / float(p);
    return float2(fast::cos(theta), fast::sin(theta));
}
