// Integration tests for `BumpArena`. Each test acquires a real Metal device
// (StorageModeShared backing buffer) and exercises a single arena invariant.

#![allow(clippy::expect_used)]

use polars_metal_buffer::MetalDevice;
// The `polars-metal-core` crate's library is exported under the name
// `polars_metal_native` (see its `[lib]` section) because the same artifact
// also serves as the Python `cdylib`. Integration tests therefore import
// from `polars_metal_native`.
use polars_metal_native::BumpArena;

#[test]
fn allocs_two_buffers_with_distinct_pointers() {
    let device = MetalDevice::system_default().expect("Metal hardware");
    let mut arena = BumpArena::with_capacity(&device, 1024 * 1024).expect("alloc backing");
    let a = arena.alloc(64).expect("first alloc");
    let b = arena.alloc(64).expect("second alloc");
    assert_ne!(a.as_slice().as_ptr(), b.as_slice().as_ptr());
}

#[test]
fn alignment_at_least_16_bytes() {
    let device = MetalDevice::system_default().expect("Metal hardware");
    let mut arena = BumpArena::with_capacity(&device, 1024).expect("alloc backing");
    let a = arena.alloc(1).expect("first alloc");
    let b = arena.alloc(1).expect("second alloc");
    let pa = a.as_slice().as_ptr() as usize;
    let pb = b.as_slice().as_ptr() as usize;
    assert_eq!(pa % 16, 0);
    assert_eq!(pb % 16, 0);
}

#[test]
fn exhaustion_returns_error_not_panic() {
    let device = MetalDevice::system_default().expect("Metal hardware");
    let mut arena = BumpArena::with_capacity(&device, 256).expect("alloc backing");
    let _ = arena.alloc(200).expect("fits");
    let result = arena.alloc(200);
    assert!(result.is_err(), "second 200B alloc must fail");
}

#[test]
fn alloc_returns_view_with_requested_byte_length() {
    let device = MetalDevice::system_default().expect("Metal hardware");
    let mut arena = BumpArena::with_capacity(&device, 1024).expect("alloc backing");
    let v = arena.alloc(100).expect("alloc");
    assert_eq!(v.as_slice().len(), 100);
}

#[test]
fn writes_to_view_visible_in_parent() {
    let device = MetalDevice::system_default().expect("Metal hardware");
    let mut arena = BumpArena::with_capacity(&device, 1024).expect("alloc backing");
    let v = arena.alloc(8).expect("alloc");
    // SAFETY: backing buffer is StorageModeShared, no GPU command in flight,
    // and the view length is 8 bytes. We obtain a raw mut pointer into the
    // shared mapping to scribble a recognizable pattern, then re-read via the
    // same view to confirm the bytes round-trip.
    unsafe {
        let ptr = v.as_slice().as_ptr() as *mut u8;
        for i in 0..8 {
            *ptr.add(i) = i as u8;
        }
    }
    for i in 0..8 {
        assert_eq!(v.as_slice()[i], i as u8);
    }
}
