//! M6 A3: differential tests for the hand-rolled MSL FFT.
//! Populated by later tasks (radix kernel, four-step, Bluestein) against the
//! CPU DFT oracle in `polars_metal_kernels::fft`.
#![allow(clippy::expect_used)]

use polars_metal_buffer::{MetalBuffer, MetalDevice};
use polars_metal_kernels::fft::{
    dft_reference, dispatch_pack_complex, dispatch_pack_real, dispatch_unpack, fft_gpu,
    fft_gpu_planar, fft_gpu_planar_core, l2_rel_err,
};

/// The PLANAR four-step path (`n > FFT_BASE_MAX`, pow2) must match the proven
/// interleaved `fft_gpu`, including the recursive (2^21+) band that was once
/// broken in MLX. COMPLEX input, forward + inverse. Compared via the crate's
/// L2 relative error (per-element abs is noisy at large N).
#[test]
fn fft_planar_core_matches_interleaved_fourstep() {
    let device = MetalDevice::system_default().expect("device");
    for &n in &[2048i64, 4096, 1 << 13, 1 << 16, 1 << 20, 1 << 21, 1 << 22] {
        for &inverse in &[false, true] {
            let nn = n as usize;
            let re: Vec<f32> = (0..nn).map(|i| ((i as f32) * 0.0013).sin()).collect();
            let im: Vec<f32> = (0..nn).map(|i| ((i as f32) * 0.0007).cos() * 0.3).collect();
            let mut inter = vec![0.0f32; 2 * nn];
            for i in 0..nn {
                inter[2 * i] = re[i];
                inter[2 * i + 1] = im[i];
            }
            let ref_out = fft_gpu(&device, &inter, n, inverse).expect("fft_gpu");
            let re_in = MetalBuffer::from_f32_slice(&device, &re).expect("re_in");
            let im_in = MetalBuffer::from_f32_slice(&device, &im).expect("im_in");
            let re_out = device.new_buffer_zeroed(nn * 4).expect("re_out");
            let im_out = device.new_buffer_zeroed(nn * 4).expect("im_out");
            fft_gpu_planar_core(&device, &re_in, &im_in, &re_out, &im_out, n, inverse)
                .expect("planar core");
            let ro = re_out.to_f32_vec();
            let io = im_out.to_f32_vec();
            let mut got = vec![0.0f32; 2 * nn];
            for i in 0..nn {
                got[2 * i] = ro[i];
                got[2 * i + 1] = io[i];
            }
            let err = l2_rel_err(&got, &ref_out);
            assert!(err < 1e-3, "n={n} inv={inverse} L2={err}");
        }
    }
}

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
fn recursive_fourstep_broken_band_and_beyond() {
    let device = MetalDevice::system_default().expect("device");
    for pow in [21u32, 22, 23, 24, 25] {
        // > 2^20 → needs recursion; MLX is broken here (ml-explore/mlx#1800).
        let n = 1usize << pow;
        let sig = interleaved_signal(n, pow as u64);
        let got = fft_gpu(&device, &sig, n as i64, false).expect("fft");
        let exp = host_fft_f64(&sig, n, false);
        let err = l2_rel_err(&got, &exp);
        assert!(err < 1e-3, "2^{pow} L2 {err}");
    }
}

#[test]
fn recursive_fourstep_inverse_broken_band() {
    let device = MetalDevice::system_default().expect("device");
    // Inverse / round-trip in the recursion band (one and two recursion levels).
    for pow in [21u32, 22] {
        let n = 1usize << pow;
        let sig = interleaved_signal(n, 2000 + pow as u64);
        // inverse vs the f64 oracle
        let inv = fft_gpu(&device, &sig, n as i64, true).expect("ifft");
        let exp = host_fft_f64(&sig, n, true);
        let err = l2_rel_err(&inv, &exp);
        assert!(err < 1e-3, "2^{pow} ifft L2 {err}");
        // round-trip ifft(fft(x)) ≈ x
        let fwd = fft_gpu(&device, &sig, n as i64, false).expect("fft");
        let back = fft_gpu(&device, &fwd, n as i64, true).expect("ifft");
        let rerr = l2_rel_err(&back, &sig);
        assert!(rerr < 1e-3, "2^{pow} roundtrip L2 {rerr}");
    }
}

// ---------------------------------------------------------------------------
// M5a: GPU pack/unpack kernel tests
// ---------------------------------------------------------------------------

/// Round-trip: dispatch_pack_real -> dispatch_unpack reproduces the original
/// real signal and all-zero imaginary plane.
#[test]
fn fft_pack_unpack_roundtrip() {
    let device = MetalDevice::system_default().expect("device");
    let n = 1000usize;
    let re: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 13.0).collect();

    // Pack real -> interleaved (length 2n, imaginary = 0).
    let inter = dispatch_pack_real(&device, &re, n).expect("dispatch_pack_real");
    assert_eq!(inter.len(), 2 * n, "interleaved length must be 2n");

    // Sanity-check a few packed values before the unpack.
    assert_eq!(inter[0], re[0]);
    assert_eq!(inter[1], 0.0f32);
    assert_eq!(inter[2], re[1]);
    assert_eq!(inter[3], 0.0f32);

    // Unpack back to planar.
    let (ro, io) = dispatch_unpack(&device, &inter, n).expect("dispatch_unpack");
    assert_eq!(ro.len(), n);
    assert_eq!(io.len(), n);
    assert_eq!(ro, re, "real plane must match the original signal");
    assert!(
        io.iter().all(|&x| x == 0.0f32),
        "imaginary plane must be all-zero after pack_real -> unpack"
    );
}

/// Round-trip: dispatch_pack_complex -> dispatch_unpack preserves both planes.
#[test]
fn fft_pack_complex_unpack_roundtrip() {
    let device = MetalDevice::system_default().expect("device");
    let n = 512usize;
    let re: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25).collect();
    let im: Vec<f32> = (0..n).map(|i| -((i as f32) * 0.1 - 25.6)).collect();

    let inter = dispatch_pack_complex(&device, &re, &im, n).expect("dispatch_pack_complex");
    assert_eq!(inter.len(), 2 * n);

    let (ro, io) = dispatch_unpack(&device, &inter, n).expect("dispatch_unpack");
    assert_eq!(ro, re, "real plane must be preserved");
    assert_eq!(io, im, "imaginary plane must be preserved");
}

/// Edge case: n == 1.
#[test]
fn fft_pack_unpack_single_element() {
    let device = MetalDevice::system_default().expect("device");
    let re = vec![42.0f32];
    let inter = dispatch_pack_real(&device, &re, 1).expect("pack single");
    assert_eq!(inter, vec![42.0f32, 0.0f32]);
    let (ro, io) = dispatch_unpack(&device, &inter, 1).expect("unpack single");
    assert_eq!(ro, vec![42.0f32]);
    assert_eq!(io, vec![0.0f32]);
}

/// Large n: exercises multi-threadgroup dispatch (n > DEFAULT_THREADGROUP_WIDTH = 256).
#[test]
fn fft_pack_unpack_large_n() {
    let device = MetalDevice::system_default().expect("device");
    let n = 1 << 20; // 1M samples
    let re: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
    let inter = dispatch_pack_real(&device, &re, n).expect("pack large");
    assert_eq!(inter.len(), 2 * n);
    let (ro, io) = dispatch_unpack(&device, &inter, n).expect("unpack large");
    assert_eq!(ro, re, "large n real plane must round-trip exactly");
    assert!(
        io.iter().all(|&x| x == 0.0f32),
        "large n imaginary plane must be all-zero"
    );
}

#[test]
fn bluestein_prime_matches_dft_and_roundtrips() {
    let device = MetalDevice::system_default().expect("device");
    // Moderate primes: forward vs the O(N^2) DFT oracle (feasible at these sizes).
    for &n in &[101usize, 251, 509, 1021] {
        let sig = interleaved_signal(n, 200 + n as u64);
        let got = fft_gpu(&device, &sig, n as i64, false).expect("fft");
        let exp = dft_reference(&sig, n, false);
        assert!(
            l2_rel_err(&got, &exp) < 1e-4,
            "n={n} fwd L2 {}",
            l2_rel_err(&got, &exp)
        );
        // inverse vs idft oracle too (exercises inverse-boundary Bluestein)
        let inv = fft_gpu(&device, &sig, n as i64, true).expect("ifft");
        let iexp = dft_reference(&sig, n, true);
        assert!(
            l2_rel_err(&inv, &iexp) < 1e-4,
            "n={n} inv L2 {}",
            l2_rel_err(&inv, &iexp)
        );
    }
    // Large prime: O(N^2) oracle is infeasible — verify via round-trip self-consistency.
    let n = 100003usize;
    let sig = interleaved_signal(n, 999);
    let fwd = fft_gpu(&device, &sig, n as i64, false).expect("fft");
    let back = fft_gpu(&device, &fwd, n as i64, true).expect("ifft");
    assert!(
        l2_rel_err(&back, &sig) < 1e-3,
        "n={n} roundtrip L2 {}",
        l2_rel_err(&back, &sig)
    );
}

// ---------------------------------------------------------------------------
// M5b-2: on-device planar pipeline (pack -> fft_gpu_buf -> unpack)
// ---------------------------------------------------------------------------

/// `fft_gpu_planar` (GPU pack -> fft_gpu_buf -> GPU unpack) must produce the
/// SAME result as the host path (CPU interleave + `fft_gpu` + CPU split), for
/// real input across several sizes including large pow2.
#[test]
fn fft_gpu_planar_matches_host_path() {
    let device = MetalDevice::system_default().expect("device");
    for &n in &[16i64, 1024, 4096, 65536] {
        let re: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).sin()).collect();
        // host reference: interleave, fft_gpu, split
        let mut inter = vec![0.0f32; 2 * n as usize];
        for i in 0..n as usize {
            inter[2 * i] = re[i];
        }
        let host = fft_gpu(&device, &inter, n, false).expect("fft_gpu host");
        let (ro, io) = fft_gpu_planar(&device, &re, None, n, false).expect("fft_gpu_planar");
        for i in 0..n as usize {
            assert!((ro[i] - host[2 * i]).abs() < 1e-3, "re[{i}] n={n}");
            assert!((io[i] - host[2 * i + 1]).abs() < 1e-3, "im[{i}] n={n}");
        }
    }
}

/// `fft_gpu_planar` complex input + inverse must match the host path.
#[test]
fn fft_gpu_planar_complex_and_inverse_match_host_path() {
    let device = MetalDevice::system_default().expect("device");
    for &n in &[16i64, 1024, 4096, 65536] {
        let re: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.1).sin()).collect();
        let im: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07).cos()).collect();
        for &inverse in &[false, true] {
            let mut inter = vec![0.0f32; 2 * n as usize];
            for i in 0..n as usize {
                inter[2 * i] = re[i];
                inter[2 * i + 1] = im[i];
            }
            let host = fft_gpu(&device, &inter, n, inverse).expect("fft_gpu host");
            let (ro, io) =
                fft_gpu_planar(&device, &re, Some(&im), n, inverse).expect("fft_gpu_planar");
            for i in 0..n as usize {
                assert!(
                    (ro[i] - host[2 * i]).abs() < 1e-3,
                    "re[{i}] n={n} inv={inverse}"
                );
                assert!(
                    (io[i] - host[2 * i + 1]).abs() < 1e-3,
                    "im[{i}] n={n} inv={inverse}"
                );
            }
        }
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

#[test]
fn fft_planar_core_matches_interleaved_bluestein() {
    let device = MetalDevice::system_default().expect("device");
    // non-smooth n (prime factor > 7): primes + composites with a large prime factor
    for &n in &[11i64, 13, 22, 26, 101, 251, 1009, 1021] {
        for &inverse in &[false, true] {
            let nn = n as usize;
            let re: Vec<f32> = (0..nn).map(|i| ((i as f32) * 0.11).sin()).collect();
            let im: Vec<f32> = (0..nn).map(|i| ((i as f32) * 0.05).cos() * 0.4).collect();
            let mut inter = vec![0.0f32; 2 * nn];
            for i in 0..nn {
                inter[2 * i] = re[i];
                inter[2 * i + 1] = im[i];
            }
            let ref_out = fft_gpu(&device, &inter, n, inverse).expect("fft_gpu");
            let re_in = MetalBuffer::from_f32_slice(&device, &re).expect("re_in");
            let im_in = MetalBuffer::from_f32_slice(&device, &im).expect("im_in");
            let re_out = device.new_buffer_zeroed(nn * 4).expect("re_out");
            let im_out = device.new_buffer_zeroed(nn * 4).expect("im_out");
            fft_gpu_planar_core(&device, &re_in, &im_in, &re_out, &im_out, n, inverse)
                .expect("planar core");
            let ro = re_out.to_f32_vec();
            let io = im_out.to_f32_vec();
            for i in 0..nn {
                assert!(
                    (ro[i] - ref_out[2 * i]).abs() < 1e-3,
                    "re n={n} inv={inverse} i={i}"
                );
                assert!(
                    (io[i] - ref_out[2 * i + 1]).abs() < 1e-3,
                    "im n={n} inv={inverse} i={i}"
                );
            }
        }
    }
}

#[test]
fn fft_planar_core_matches_interleaved_small() {
    // The PLANAR (SoA) single-threadgroup path must match the proven interleaved
    // `fft_gpu`. COMPLEX input (both re and im nonzero) exercises the full planar
    // load, not just the real-input zero-im case.
    let device = MetalDevice::system_default().expect("device");
    for &n in &[
        2i64, 4, 8, 16, 64, 256, 1024, /* smooth: */ 6, 12, 360, 720,
    ] {
        for &inverse in &[false, true] {
            let nn = n as usize;
            let re: Vec<f32> = (0..nn).map(|i| ((i as f32) * 0.13).sin()).collect();
            let im: Vec<f32> = (0..nn).map(|i| ((i as f32) * 0.07).cos() * 0.5).collect();
            // interleaved reference
            let mut inter = vec![0.0f32; 2 * nn];
            for i in 0..nn {
                inter[2 * i] = re[i];
                inter[2 * i + 1] = im[i];
            }
            let ref_out = fft_gpu(&device, &inter, n, inverse).expect("fft_gpu");
            // planar
            let re_in = MetalBuffer::from_f32_slice(&device, &re).expect("re_in");
            let im_in = MetalBuffer::from_f32_slice(&device, &im).expect("im_in");
            let re_out = device.new_buffer_zeroed(nn * 4).expect("re_out");
            let im_out = device.new_buffer_zeroed(nn * 4).expect("im_out");
            fft_gpu_planar_core(&device, &re_in, &im_in, &re_out, &im_out, n, inverse)
                .expect("planar core");
            let ro = re_out.to_f32_vec();
            let io = im_out.to_f32_vec();
            for i in 0..nn {
                assert!(
                    (ro[i] - ref_out[2 * i]).abs() < 1e-3,
                    "re n={n} inv={inverse} i={i}: {} vs {}",
                    ro[i],
                    ref_out[2 * i]
                );
                assert!(
                    (io[i] - ref_out[2 * i + 1]).abs() < 1e-3,
                    "im n={n} inv={inverse} i={i}: {} vs {}",
                    io[i],
                    ref_out[2 * i + 1]
                );
            }
        }
    }
}
