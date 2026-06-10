// crates/polars-metal-buffer/tests/test_staging.rs
//
// Unit tests for the reusable page-aligned StagingPool (B3b). Validates that
// staging copies bytes correctly, reuses the underlying buffer when the new
// input fits, reallocates (grows) when it does not, and reads back the correct
// prefix after a larger-then-smaller sequence.
//
// Requires Metal-capable hardware; skips via `expect` without a device.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use polars_metal_buffer::{MetalDevice, StagingPool};
use std::sync::Mutex;

static METAL_TEST_LOCK: Mutex<()> = Mutex::new(());

fn dev() -> MetalDevice {
    MetalDevice::system_default().expect("Metal-capable hardware required")
}

#[test]
fn stage_copies_bytes() {
    let _l = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let device = dev();
    let mut pool = StagingPool::new();
    let src: Vec<u8> = (0..200u32).map(|i| (i % 256) as u8).collect();
    let buf = pool.stage(&device, &src).expect("stage ok");
    assert_eq!(&buf.as_slice()[..src.len()], &src[..]);
}

#[test]
fn stage_reuses_buffer_when_fits() {
    let _l = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let device = dev();
    let mut pool = StagingPool::new();
    let big = vec![1u8; 4096];
    let cap_ptr = {
        let b = pool.stage(&device, &big).expect("stage big");
        b.as_slice().as_ptr() as usize
    };
    // A smaller input must reuse the SAME backing allocation (same contents ptr).
    let small = vec![2u8; 128];
    let b2 = pool.stage(&device, &small).expect("stage small");
    assert_eq!(
        b2.as_slice().as_ptr() as usize,
        cap_ptr,
        "should reuse buffer"
    );
    assert_eq!(&b2.as_slice()[..small.len()], &small[..]);
}

#[test]
fn stage_grows_when_larger() {
    let _l = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let device = dev();
    let mut pool = StagingPool::new();
    let small = vec![9u8; 64];
    let p1 = pool.stage(&device, &small).expect("s1").as_slice().as_ptr() as usize;
    let _ = p1;
    let big = vec![7u8; 1_000_000];
    let b = pool.stage(&device, &big).expect("grow");
    assert!(b.len() >= big.len(), "capacity grew to fit");
    assert_eq!(&b.as_slice()[..big.len()], &big[..]);
}

#[test]
fn stage_rejects_empty() {
    let _l = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let device = dev();
    let mut pool = StagingPool::new();
    assert!(pool.stage(&device, &[]).is_err(), "empty input rejected");
}
