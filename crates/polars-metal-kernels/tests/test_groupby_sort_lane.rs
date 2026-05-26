//! GPU single-lane radix-sort pass: one 8-bit lane partitions rows by
//! their lane-byte digit. Stability within a digit is *not* guaranteed
//! (scatter uses a non-deterministic atomic cursor) — Task 26 chains 16
//! lanes for a full u128 sort; correctness then depends only on each
//! lane's grouping property, which these tests check.
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
    // All keys share lane-0 digit = 0xAB; the pass should leave them
    // adjacent (in *some* order — scatter is non-deterministic).
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
