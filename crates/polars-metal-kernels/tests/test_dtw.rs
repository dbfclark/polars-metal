#![allow(clippy::expect_used, clippy::unwrap_used)]
//! DTW kernel correctness vs a CPU scalar reference (M6 A4).

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::dtw::{dispatch_dtw, DtwError, MAX_L};

/// Scalar DTW: Euclidean (squared-diff cost, sqrt of the final cumulative).
/// `window < 0` => unconstrained; else compute cell (i,j) iff |i-j| <= window.
fn dtw_ref(q: &[f32], r: &[f32], window: i32) -> f32 {
    let l = q.len();
    assert_eq!(l, r.len());
    let inf = f32::INFINITY;
    let mut prev = vec![inf; l + 1];
    let mut cur = vec![inf; l + 1];
    prev[0] = 0.0;
    for i in 1..=l {
        cur[0] = inf;
        for j in 1..=l {
            if window >= 0 && (i as i32 - j as i32).abs() > window {
                cur[j] = inf;
                continue;
            }
            let d = q[i - 1] - r[j - 1];
            let cost = d * d;
            let m = prev[j].min(cur[j - 1]).min(prev[j - 1]);
            cur[j] = cost + m;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[l].sqrt()
}

fn run(q: &[f32], r: &[f32], n_pairs: usize, window: i32) -> Vec<f32> {
    // queries laid out pair-major: n_pairs copies of q here for simplicity
    let device = MetalDevice::system_default().expect("metal device");
    let mut queries = Vec::with_capacity(n_pairs * q.len());
    for _ in 0..n_pairs {
        queries.extend_from_slice(q);
    }
    let mut out = vec![0.0f32; n_pairs];
    dispatch_dtw(&device, &queries, r, &mut out, q.len(), window).expect("dispatch");
    out
}

#[test]
fn matches_reference_full_and_banded() {
    let l = 17usize;
    let q: Vec<f32> = (0..l).map(|i| (i as f32 * 0.37).sin()).collect();
    let r: Vec<f32> = (0..l).map(|i| (i as f32 * 0.29).cos()).collect();
    for window in [-1i32, 0, 1, 3, l as i32] {
        let expect = dtw_ref(&q, &r, window);
        let got = run(&q, &r, 4, window);
        for g in got {
            assert!(
                (g - expect).abs() <= 1e-3 * (1.0 + expect.abs()),
                "window={window} got={g} expect={expect}"
            );
        }
    }
}

#[test]
fn identical_sequences_distance_zero() {
    let q: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let got = run(&q, &q, 2, -1);
    for g in got {
        assert!(g.abs() <= 1e-3, "identical seqs should be ~0, got {g}");
    }
}

#[test]
fn single_element() {
    let got = run(&[3.0], &[5.0], 1, -1);
    assert!((got[0] - 2.0).abs() <= 1e-4, "got {}", got[0]); // sqrt((3-5)^2)=2
}

#[test]
fn zero_pairs_is_ok() {
    let device = MetalDevice::system_default().expect("metal device");
    let r = [1.0f32, 2.0, 3.0];
    let mut out: Vec<f32> = vec![];
    dispatch_dtw(&device, &[], &r, &mut out, 3, -1).expect("n=0 ok");
}

#[test]
fn rejects_l_over_max() {
    let device = MetalDevice::system_default().expect("metal device");
    let l = MAX_L + 1;
    let q = vec![0.0f32; l];
    let mut out = vec![0.0f32; 1];
    let err = dispatch_dtw(&device, &q, &q, &mut out, l, -1).unwrap_err();
    assert!(matches!(err, DtwError::SeqLenOutOfRange { .. }));
}
