//! M6 A3: differential tests for the hand-rolled MSL FFT.
//! Populated by later tasks (radix kernel, four-step, Bluestein) against the
//! CPU DFT oracle in `polars_metal_kernels::fft`.

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::fft::{dft_reference, fft_gpu, l2_rel_err};

fn interleaved_signal(n: usize, seed: u64) -> Vec<f32> {
    // deterministic pseudo-random complex signal, interleaved re,im
    let mut v = vec![0f32; 2 * n];
    let mut s = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    for x in v.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *x = ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
    }
    v
}

#[test]
fn radix2_small_pow2_forward_matches_dft() {
    let device = MetalDevice::system_default().expect("device");
    for &n in &[2usize, 4, 8, 16, 64, 256, 1024] {
        let sig = interleaved_signal(n, n as u64);
        let got = fft_gpu(&device, &sig, n as i64, /*inverse=*/ false).expect("fft_gpu");
        let exp = dft_reference(&sig, n, false);
        let err = l2_rel_err(&got, &exp);
        assert!(err < 1e-4, "n={n}: L2 rel err {err} too high");
    }
}

#[test]
fn mixed_radix_small_composite_matches_dft() {
    let device = MetalDevice::system_default().expect("device");
    // composite smooth sizes exercising radix 3,4,5,6,7,8 factor paths
    for &n in &[3usize, 5, 6, 7, 9, 12, 15, 24, 35, 120, 360, 1000] {
        let sig = interleaved_signal(n, 100 + n as u64);
        let got = fft_gpu(&device, &sig, n as i64, false).expect("fft");
        let exp = dft_reference(&sig, n, false);
        assert!(
            l2_rel_err(&got, &exp) < 1e-4,
            "n={n}: L2 {}",
            l2_rel_err(&got, &exp)
        );
    }
}

#[test]
fn mixed_radix_inverse_and_roundtrip() {
    let device = MetalDevice::system_default().expect("device");
    // composite (non-pow2) sizes route through fft_mixed_radix_f32, whose
    // inverse-by-conjugation path is otherwise untested.
    for &n in &[6usize, 12, 35, 360, 1000] {
        let sig = interleaved_signal(n, 500 + n as u64);
        // inverse vs the CPU idft oracle
        let inv = fft_gpu(&device, &sig, n as i64, true).expect("ifft");
        let exp = dft_reference(&sig, n, true);
        assert!(
            l2_rel_err(&inv, &exp) < 1e-4,
            "n={n} ifft L2 {}",
            l2_rel_err(&inv, &exp)
        );
        // round-trip ifft(fft(x)) ≈ x
        let fwd = fft_gpu(&device, &sig, n as i64, false).expect("fft");
        let back = fft_gpu(&device, &fwd, n as i64, true).expect("ifft");
        assert!(
            l2_rel_err(&back, &sig) < 1e-4,
            "n={n} roundtrip L2 {}",
            l2_rel_err(&back, &sig)
        );
    }
}

/// Iterative in-place radix-2 Cooley-Tukey FFT in f64 over interleaved-complex
/// input; returns interleaved f32. Correct reference for large pow2 where the
/// O(N^2) DFT is too slow. `inverse` applies +sign twiddles and 1/N scaling.
/// Requires `n` be a power of two.
fn host_fft_f64(input: &[f32], n: usize, inverse: bool) -> Vec<f32> {
    assert!(n.is_power_of_two(), "host_fft_f64 requires pow2 n");
    let mut re = vec![0f64; n];
    let mut im = vec![0f64; n];
    for i in 0..n {
        re[i] = input[2 * i] as f64;
        im[i] = input[2 * i + 1] as f64;
    }
    // bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    // butterflies
    let sign = if inverse { 1.0f64 } else { -1.0f64 };
    let mut len = 2usize;
    while len <= n {
        let ang = sign * 2.0 * std::f64::consts::PI / (len as f64);
        let (wre, wim) = (ang.cos(), ang.sin());
        let half = len / 2;
        let mut start = 0;
        while start < n {
            let (mut cre, mut cim) = (1.0f64, 0.0f64);
            for k in 0..half {
                let a = start + k;
                let b = start + k + half;
                let tre = re[b] * cre - im[b] * cim;
                let tim = re[b] * cim + im[b] * cre;
                re[b] = re[a] - tre;
                im[b] = im[a] - tim;
                re[a] += tre;
                im[a] += tim;
                let ncre = cre * wre - cim * wim;
                cim = cre * wim + cim * wre;
                cre = ncre;
            }
            start += len;
        }
        len <<= 1;
    }
    let scale = if inverse { 1.0 / n as f64 } else { 1.0 };
    let mut out = vec![0f32; 2 * n];
    for i in 0..n {
        out[2 * i] = (re[i] * scale) as f32;
        out[2 * i + 1] = (im[i] * scale) as f32;
    }
    out
}

#[test]
fn fourstep_large_pow2_matches_reference() {
    let device = MetalDevice::system_default().expect("device");
    // Odd powers (13, 17) make the planner's split give p1 != p2, so n1 != n2 —
    // exercising the transpose dim-swap / row<->col paths that an even-only pow
    // list (n1 == n2 always) would never catch. 13 -> n1=64,n2=128;
    // 17 -> n1=256,n2=512 (both factors <= 1024).
    for pow in [12u32, 13, 16, 17, 20] {
        // (2^10, 2^20]: n1, n2 <= 1024
        let n = 1usize << pow;
        let sig = interleaved_signal(n, pow as u64);
        let got = fft_gpu(&device, &sig, n as i64, false).expect("fft");
        let exp = host_fft_f64(&sig, n, false);
        let err = l2_rel_err(&got, &exp);
        assert!(err < 1e-3, "n=2^{pow}: L2 {err}");
    }
}

#[test]
fn fourstep_inverse_and_roundtrip() {
    let device = MetalDevice::system_default().expect("device");
    // Include odd powers (13, 15) so the inverse path is checked on a
    // non-square n1 != n2 shape too.
    for pow in [12u32, 13, 15, 16] {
        let n = 1usize << pow;
        let sig = interleaved_signal(n, 1000 + pow as u64);
        // inverse vs the f64 oracle
        let inv = fft_gpu(&device, &sig, n as i64, true).expect("ifft");
        let exp = host_fft_f64(&sig, n, true);
        let err = l2_rel_err(&inv, &exp);
        assert!(err < 1e-3, "n=2^{pow} ifft L2 {err}");
        // round-trip ifft(fft(x)) ≈ x
        let fwd = fft_gpu(&device, &sig, n as i64, false).expect("fft");
        let back = fft_gpu(&device, &fwd, n as i64, true).expect("ifft");
        let rerr = l2_rel_err(&back, &sig);
        assert!(rerr < 1e-3, "n=2^{pow} roundtrip L2 {rerr}");
    }
}

#[test]
fn radix2_inverse_and_roundtrip() {
    let device = MetalDevice::system_default().expect("device");
    // sizes within the single-threadgroup base path (n <= FFT_BASE_MAX = 1024);
    // n > 1024 is exercised once the four-step path lands (Task 4).
    for &n in &[8usize, 64, 256, 1024] {
        let sig = interleaved_signal(n, 7 + n as u64);
        // inverse matches reference idft
        let inv = fft_gpu(&device, &sig, n as i64, true).expect("ifft");
        let exp = dft_reference(&sig, n, true);
        assert!(l2_rel_err(&inv, &exp) < 1e-4, "n={n} ifft");
        // round-trip: ifft(fft(x)) ≈ x
        let fwd = fft_gpu(&device, &sig, n as i64, false).expect("fft");
        let back = fft_gpu(&device, &fwd, n as i64, true).expect("ifft");
        assert!(l2_rel_err(&back, &sig) < 1e-4, "n={n} roundtrip");
    }
}
