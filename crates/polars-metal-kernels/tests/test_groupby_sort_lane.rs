//! GPU single-lane radix-sort pass: one 8-bit lane partitions rows by
//! their lane-byte digit AND preserves their relative input order within
//! each digit bucket (stability). Stability is required for LSD radix
//! correctness when Task 26 chains 16 lanes for a full u128 sort —
//! ties at byte k+1 must keep the byte-k ordering established by the
//! previous pass.
#![allow(clippy::expect_used)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::groupby_build_sort::gpu::run_radix_lane;

#[test]
fn lane_0_sorts_by_low_byte() {
    let device = MetalDevice::system_default().expect("metal device");
    // Keys whose lowest byte differs distinctly.
    let keys: Vec<u128> = vec![0x3, 0x1, 0x2, 0x4, 0x5];
    let idx: Vec<u32> = vec![0, 1, 2, 3, 4];
    let (sk, si) = run_radix_lane(&device, &keys, &idx, 0).expect("lane");
    assert_eq!(sk, vec![0x1, 0x2, 0x3, 0x4, 0x5]);
    // Each output key corresponds to the original key at sorted_idx[i].
    for i in 0..sk.len() {
        assert_eq!(sk[i], keys[si[i] as usize]);
    }
}

#[test]
fn lane_groups_equal_digits_together() {
    let device = MetalDevice::system_default().expect("metal device");
    // All keys share lane-0 digit = 0xAB. With the stable scatter the
    // pass must leave them in their original order (input order is the
    // stability target).
    let keys: Vec<u128> = vec![0x100AB, 0x200AB, 0x300AB, 0x400AB];
    let idx: Vec<u32> = vec![0, 1, 2, 3];
    let (sk, si) = run_radix_lane(&device, &keys, &idx, 0).expect("lane");
    // si is a permutation of 0..4.
    let mut perm = si.clone();
    perm.sort_unstable();
    assert_eq!(perm, vec![0, 1, 2, 3]);
    // sk and si remain consistent.
    for i in 0..sk.len() {
        assert_eq!(sk[i], keys[si[i] as usize]);
    }
}

#[test]
fn lane_pass_is_stable_within_a_bucket() {
    let device = MetalDevice::system_default().expect("metal device");
    // Construct 500 keys that all share the lane-0 digit (0xAB) but
    // differ in higher bytes. Stability means: within the single
    // resulting digit bucket, the original row indices appear in
    // increasing order (= input order preserved). 500 > 256, so this
    // spans multiple tiles and exercises the cross-tile prefix path.
    let keys: Vec<u128> = (0u128..500).map(|i| (i << 16) | 0xAB).collect();
    let idx: Vec<u32> = (0..500).collect();
    let (sk, si) = run_radix_lane(&device, &keys, &idx, 0).expect("lane");
    // All keys went into the 0xAB bucket; their order in `si` must be
    // exactly the input order 0,1,2,...,499.
    assert_eq!(si, idx);
    // sk and si remain consistent.
    for i in 0..sk.len() {
        assert_eq!(sk[i], keys[si[i] as usize]);
    }
}
