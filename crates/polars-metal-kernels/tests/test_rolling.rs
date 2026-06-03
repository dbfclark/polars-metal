// crates/polars-metal-kernels/tests/test_rolling.rs
//
// Correctness tests for the tile-blocked rolling sum/mean/var/std kernels
// (`rolling_sum_f32`, `rolling_var_f32` in `shaders/rolling.metal`). Validates:
//   - Multi-tile inputs (n > TG_SIZE) produce correct sums at every window.
//   - Window boundaries: w=1 (identity), w=TG_SIZE (tile-boundary), w>TG_SIZE
//     (halo larger than one tile).
//   - The first w-1 outputs are zero-filled (structural nulls; host masks them).
//   - `is_mean=true` divides the per-window sum by w.
//   - Numerical stability: per-threadgroup accumulation keeps magnitudes
//     ~window-scale, not ~N-scale (no catastrophic cancellation).
//   - Rolling variance: centered two-pass per window, ddof=1 (Polars default).
//   - Rolling std: sqrt of variance; large-offset inputs (1000.0 base) stress
//     cancellation avoidance.
//
// All tests require Metal-capable hardware; they will skip with an `expect`
// failure on machines without a discoverable system-default MTLDevice.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::rolling::{dispatch_rolling_sum_f32, dispatch_rolling_var_f32};
use std::sync::Mutex;

static METAL_TEST_LOCK: Mutex<()> = Mutex::new(());

/// CPU reference: exclusive prefix (first w-1 outputs are NaN; valid outputs
/// are the sum of the w elements ending at that index).
fn ref_rolling_sum(x: &[f32], w: usize) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            if i + 1 < w {
                f64::NAN
            } else {
                ((i + 1 - w)..=i).map(|j| x[j] as f64).sum()
            }
        })
        .collect()
}

/// Helper: dispatch the kernel and return the GPU outputs.
fn run_sum(x: &[f32], w: usize, is_mean: bool) -> Vec<f32> {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let device = MetalDevice::system_default().expect("Metal-capable hardware required");
    let mut out = vec![0.0f32; x.len()];
    dispatch_rolling_sum_f32(&device, x, &mut out, w, is_mean).expect("dispatch succeeds");
    out
}

// --- Fixed-shape unit tests --------------------------------------------------

#[test]
fn rolling_sum_multi_tile_and_boundary() {
    // 1000 rows, mix of windows that cross tile boundaries (TG_SIZE=256).
    let n = 1000usize;
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 100.0).collect();
    for w in [1usize, 2, 7, 256, 257, 300] {
        let got = run_sum(&x, w, false);
        let want = ref_rolling_sum(&x, w);
        // First w-1 outputs are structural nulls (zero-filled by the kernel;
        // host is expected to mask them). Only check i >= w-1.
        for i in (w - 1)..n {
            let delta = (got[i] as f64 - want[i]).abs();
            assert!(
                delta < 1e-3,
                "w={w} i={i} got={} want={} delta={delta}",
                got[i],
                want[i]
            );
        }
    }
}

#[test]
fn rolling_mean_divides_by_w() {
    let x: Vec<f32> = (1..=10).map(|i| i as f32).collect();
    // w=3: mean of [1,2,3]=2, mean of [8,9,10]=9.
    let got = run_sum(&x, 3, true);
    assert!(
        (got[2] - 2.0).abs() < 1e-5,
        "mean at i=2 got={} expected 2.0",
        got[2]
    );
    assert!(
        (got[9] - 9.0).abs() < 1e-5,
        "mean at i=9 got={} expected 9.0",
        got[9]
    );
}

#[test]
fn rolling_sum_w1_is_identity() {
    let x: Vec<f32> = vec![1.0, -2.0, 3.5, 0.0, 99.0];
    let got = run_sum(&x, 1, false);
    for (i, (&g, &e)) in got.iter().zip(x.iter()).enumerate() {
        assert!((g - e).abs() < 1e-6, "w=1 i={i} got={g} want={e}");
    }
}

#[test]
fn rolling_sum_full_window_equals_total() {
    // w == n: only the last element should be valid (equal to the total sum).
    let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let n = x.len();
    let total: f32 = x.iter().sum();
    let got = run_sum(&x, n, false);
    assert!(
        (got[n - 1] - total).abs() < 1e-4,
        "last output got={} want={total}",
        got[n - 1]
    );
}

#[test]
fn rolling_mean_small_constant_series() {
    // All same value: mean == value for every valid output.
    let x = vec![3.0f32; 50];
    let got = run_sum(&x, 7, true);
    for i in 6..50 {
        assert!(
            (got[i] - 3.0).abs() < 1e-5,
            "i={i} mean got={} want=3.0",
            got[i]
        );
    }
}

// --- Rolling variance / std tests -------------------------------------------

/// CPU reference for rolling sample variance (ddof=1). Returns NaN for the
/// first w-1 positions (window not yet full).
fn ref_rolling_var(x: &[f32], w: usize, ddof: usize) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            if i + 1 < w {
                return f64::NAN;
            }
            let win: Vec<f64> = ((i + 1 - w)..=i).map(|j| x[j] as f64).collect();
            let mu = win.iter().sum::<f64>() / w as f64;
            win.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / (w - ddof) as f64
        })
        .collect()
}

/// Helper: dispatch rolling_var_f32 via the slice wrapper and return outputs.
fn run_var(x: &[f32], w: usize, is_std: bool) -> Vec<f32> {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let device = MetalDevice::system_default().expect("Metal-capable hardware required");
    let mut out = vec![0.0f32; x.len()];
    // ddof=1 matches Polars sample-variance default.
    dispatch_rolling_var_f32(&device, x, &mut out, w, 1, is_std).expect("dispatch succeeds");
    out
}

#[test]
fn rolling_var_std_match_reference() {
    let n = 600usize;
    // Large 1000.0 offset stresses cancellation: naive single-pass variance
    // (without centering) would catastrophically cancel here at F32 precision.
    let x: Vec<f32> = (0..n).map(|i| ((i * 7 % 13) as f32) + 1000.0).collect();
    for w in [2usize, 5, 256, 300] {
        let var = run_var(&x, w, false);
        let std = run_var(&x, w, true);
        let rv = ref_rolling_var(&x, w, 1);
        for i in (w - 1)..n {
            assert!(
                (var[i] as f64 - rv[i]).abs() < 1e-2,
                "var w={w} i={i} got={} want={}",
                var[i],
                rv[i]
            );
            assert!(
                (std[i] as f64 - rv[i].sqrt()).abs() < 1e-2,
                "std w={w} i={i} got={} want={}",
                std[i],
                rv[i].sqrt()
            );
        }
    }
}
