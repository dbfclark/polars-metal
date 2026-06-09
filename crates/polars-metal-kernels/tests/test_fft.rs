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
