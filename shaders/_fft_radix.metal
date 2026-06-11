// shaders/_fft_radix.metal — FFT codelets (adapted from vendor/mlx/.../fft/radix.h).
// Header-only ('_' prefix): #included by fft.metal, not compiled standalone.
#pragma once
#include <metal_stdlib>
using namespace metal;

// Complex multiply: (a.x+i a.y)(b.x+i b.y).
inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
// Complex multiply followed by conjugate (MLX complex_mul_conj); used by radix7.
inline float2 cmulconj(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, -a.x * b.y - a.y * b.x);
}
// Twiddle e^{-2πi k / p} (forward). For inverse, pass +sign via `inv`.
inline float2 twiddle(int k, int p, bool inv) {
    float theta = (inv ? 2.0f : -2.0f) * float(k) * M_PI_F / float(p);
    return float2(fast::cos(theta), fast::sin(theta));
}

// ---- Forward radix butterfly codelets (ported from MLX fft/radix.h). ----
// Each takes r inputs x[0..r) and writes r outputs y[0..r). FORWARD-sign only;
// the driver handles inverse by conjugation at the kernel boundary. cmul == MLX
// complex_mul; cmulconj == MLX complex_mul_conj.

inline void radix2(thread float2* x, thread float2* y) {
    y[0] = x[0] + x[1];
    y[1] = x[0] - x[1];
}

inline void radix3(thread float2* x, thread float2* y) {
    float pi_2_3 = -0.8660254037844387f;

    float2 a_1 = x[1] + x[2];
    float2 a_2 = x[1] - x[2];

    y[0] = x[0] + a_1;
    float2 b_1 = x[0] - 0.5f * a_1;
    float2 b_2 = pi_2_3 * a_2;

    float2 b_2_j = {-b_2.y, b_2.x};
    y[1] = b_1 + b_2_j;
    y[2] = b_1 - b_2_j;
}

inline void radix4(thread float2* x, thread float2* y) {
    float2 z_0 = x[0] + x[2];
    float2 z_1 = x[0] - x[2];
    float2 z_2 = x[1] + x[3];
    float2 z_3 = x[1] - x[3];
    float2 z_3_i = {z_3.y, -z_3.x};

    y[0] = z_0 + z_2;
    y[1] = z_1 + z_3_i;
    y[2] = z_0 - z_2;
    y[3] = z_1 - z_3_i;
}

inline void radix5(thread float2* x, thread float2* y) {
    float root_5_4 = 0.5590169943749475f;
    float sin_2pi_5 = 0.9510565162951535f;
    float sin_1pi_5 = 0.5877852522924731f;

    float2 a_1 = x[1] + x[4];
    float2 a_2 = x[2] + x[3];
    float2 a_3 = x[1] - x[4];
    float2 a_4 = x[2] - x[3];

    float2 a_5 = a_1 + a_2;
    float2 a_6 = root_5_4 * (a_1 - a_2);
    float2 a_7 = x[0] - a_5 / 4;
    float2 a_8 = a_7 + a_6;
    float2 a_9 = a_7 - a_6;
    float2 a_10 = sin_2pi_5 * a_3 + sin_1pi_5 * a_4;
    float2 a_11 = sin_1pi_5 * a_3 - sin_2pi_5 * a_4;
    float2 a_10_j = {a_10.y, -a_10.x};
    float2 a_11_j = {a_11.y, -a_11.x};

    y[0] = x[0] + a_5;
    y[1] = a_8 + a_10_j;
    y[2] = a_9 + a_11_j;
    y[3] = a_9 - a_11_j;
    y[4] = a_8 - a_10_j;
}

inline void radix6(thread float2* x, thread float2* y) {
    float sin_pi_3 = 0.8660254037844387f;
    float2 a_1 = x[2] + x[4];
    float2 a_2 = x[0] - a_1 / 2;
    float2 a_3 = sin_pi_3 * (x[2] - x[4]);
    float2 a_4 = x[5] + x[1];
    float2 a_5 = x[3] - a_4 / 2;
    float2 a_6 = sin_pi_3 * (x[5] - x[1]);
    float2 a_7 = x[0] + a_1;

    float2 a_3_i = {a_3.y, -a_3.x};
    float2 a_6_i = {a_6.y, -a_6.x};
    float2 a_8 = a_2 + a_3_i;
    float2 a_9 = a_2 - a_3_i;
    float2 a_10 = x[3] + a_4;
    float2 a_11 = a_5 + a_6_i;
    float2 a_12 = a_5 - a_6_i;

    y[0] = a_7 + a_10;
    y[1] = a_8 - a_11;
    y[2] = a_9 + a_12;
    y[3] = a_7 - a_10;
    y[4] = a_8 + a_11;
    y[5] = a_9 - a_12;
}

inline void radix7(thread float2* x, thread float2* y) {
    // Rader's algorithm (decomposes 7 into a length-6 codelet).
    float2 inv = {1 / 6.0f, -1 / 6.0f};

    // fft
    float2 in1[6] = {x[1], x[3], x[2], x[6], x[4], x[5]};
    radix6(in1, y + 1);

    y[0] = y[1] + x[0];

    // b_q
    y[1] = cmulconj(y[1], float2(-1, 0));
    y[2] = cmulconj(y[2], float2(2.44013336f, -1.02261879f));
    y[3] = cmulconj(y[3], float2(2.37046941f, -1.17510629f));
    y[4] = cmulconj(y[4], float2(0, -2.64575131f));
    y[5] = cmulconj(y[5], float2(2.37046941f, 1.17510629f));
    y[6] = cmulconj(y[6], float2(-2.44013336f, -1.02261879f));

    // ifft
    radix6(y + 1, x + 1);

    y[1] = x[1] * inv + x[0];
    y[5] = x[2] * inv + x[0];
    y[4] = x[3] * inv + x[0];
    y[6] = x[4] * inv + x[0];
    y[2] = x[5] * inv + x[0];
    y[3] = x[6] * inv + x[0];
}

inline void radix8(thread float2* x, thread float2* y) {
    float cos_pi_4 = 0.7071067811865476f;
    float2 w_0 = {cos_pi_4, -cos_pi_4};
    float2 w_1 = {-cos_pi_4, -cos_pi_4};
    float2 temp[8] = {x[0], x[2], x[4], x[6], x[1], x[3], x[5], x[7]};
    radix4(temp, x);
    radix4(temp + 4, x + 4);

    y[0] = x[0] + x[4];
    y[4] = x[0] - x[4];
    float2 x_5 = cmul(x[5], w_0);
    y[1] = x[1] + x_5;
    y[5] = x[1] - x_5;
    float2 x_6 = {x[6].y, -x[6].x};
    y[2] = x[2] + x_6;
    y[6] = x[2] - x_6;
    float2 x_7 = cmul(x[7], w_1);
    y[3] = x[3] + x_7;
    y[7] = x[3] - x_7;
}
