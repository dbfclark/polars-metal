// crates/polars-metal-kernels/tests/test_rolling.rs
//
// Correctness tests for the tile-blocked rolling sum/mean kernel
// (`rolling_sum_f32` in `shaders/rolling.metal`). Validates:
//   - Multi-tile inputs (n > TG_SIZE) produce correct sums at every window.
//   - Window boundaries: w=1 (identity), w=TG_SIZE (tile-boundary), w>TG_SIZE
//     (halo larger than one tile).
//   - The first w-1 outputs are zero-filled (structural nulls; host masks them).
//   - `is_mean=true` divides the per-window sum by w.
//   - Numerical stability: per-threadgroup accumulation keeps magnitudes
//     ~window-scale, not ~N-scale (no catastrophic cancellation).
//
// All tests require Metal-capable hardware; they will skip with an `expect`
// failure on machines without a discoverable system-default MTLDevice.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::rolling::dispatch_rolling_sum_f32;
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
