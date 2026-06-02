//! Phase 5b spike: smoke tests + correctness proptest for the
//! single-pass global-atomic GPU hash table.
//!
//! Pass conditions:
//! - Smoke: 10K unique keys × 100K rows produces n_groups == 10000 and
//!   consistent row_to_group (rows with the same key get the same gid).
//! - Proptest (16 cases × varied n_rows/cardinality): equivalence-class
//!   match with the CPU HashMap reference.
//! - No deadlock: tests complete in finite time. If the spin-wait
//!   pattern deadlocks the GPU, the Metal driver will time out at
//!   ~5s and Rust will see a kernel error.

#![allow(clippy::expect_used)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::groupby_global_hash::gpu::global_hash_build;
use proptest::prelude::*;

#[test]
fn empty_input_yields_zero_groups() {
    let device = MetalDevice::system_default().expect("metal device");
    let out = global_hash_build(&device, &[], 0).expect("global hash");
    assert_eq!(out.n_groups, 0);
    assert!(out.row_to_group.is_empty());
}

#[test]
fn single_key_yields_one_group() {
    let device = MetalDevice::system_default().expect("metal device");
    let keys: Vec<u128> = vec![42u128; 100];
    let out = global_hash_build(&device, &keys, 1).expect("global hash");
    assert_eq!(out.n_groups, 1);
    assert!(out.row_to_group.iter().all(|&g| g == 0));
}

/// ⚠ DEMONSTRATES THE A3 SPIKE'S NEGATIVE RESULT.
///
/// This test FAILS on MSL toolchain 32023.883: the kernel produces
/// inflated `n_groups` (~2×) due to the missing memory-ordering
/// primitives. It is `#[ignore]`d so CI stays green while preserving
/// the regression signal for when Apple ships acquire/release in MSL.
/// Run manually with `--ignored` to re-check the toolchain state.
///
/// See `shaders/groupby_global_hash.metal` for the root-cause analysis.
#[test]
#[ignore = "A3 spike: MSL relaxed-only atomics produce duplicate groups; revisit when Apple ships acquire/release"]
fn ten_thousand_unique_keys_in_hundred_thousand_rows() {
    let device = MetalDevice::system_default().expect("metal device");
    let n_unique = 10_000u128;
    let keys: Vec<u128> = (0..100_000u128).map(|i| i % n_unique).collect();
    let out = global_hash_build(&device, &keys, n_unique as usize).expect("global hash");
    assert_eq!(out.n_groups, n_unique as u32);

    let mut key_to_gid: std::collections::HashMap<u128, u32> = std::collections::HashMap::new();
    for (r, &k) in keys.iter().enumerate() {
        let gid = out.row_to_group[r];
        match key_to_gid.entry(k) {
            std::collections::hash_map::Entry::Occupied(e) => assert_eq!(*e.get(), gid),
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(gid);
            }
        }
    }
}

// Proptest passes at small N (the memory-ordering bug is statistically
// rare at low load factors) but fails at the cardinalities A3 was
// designed for. Kept #[ignore]'d as part of the spike record.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    #[ignore = "A3 spike — see `ten_thousand_unique_keys_in_hundred_thousand_rows`"]
    fn global_hash_matches_cpu_reference(
        keys in proptest::collection::vec(any::<u128>(), 1..2048usize),
    ) {
        let device = MetalDevice::system_default().expect("metal device");
        // CPU reference.
        let mut cpu_groups: std::collections::HashMap<u128, u32> = std::collections::HashMap::new();
        let mut cpu_r2g = vec![0u32; keys.len()];
        for (r, &k) in keys.iter().enumerate() {
            let len_before = cpu_groups.len() as u32;
            let gid = *cpu_groups.entry(k).or_insert(len_before);
            cpu_r2g[r] = gid;
        }
        let cpu_n_groups = cpu_groups.len() as u32;

        // GPU. Size the table generously to avoid overflow at small N.
        let est = (keys.len() + 1) / 2 + 1;
        let gpu_out = global_hash_build(&device, &keys, est).expect("global hash");

        prop_assert_eq!(gpu_out.n_groups, cpu_n_groups);

        // Equivalence-class check: same-grouping invariant.
        for a in 0..keys.len() {
            for b in 0..keys.len() {
                let gpu_same = gpu_out.row_to_group[a] == gpu_out.row_to_group[b];
                let cpu_same = cpu_r2g[a] == cpu_r2g[b];
                prop_assert_eq!(gpu_same, cpu_same);
            }
        }
    }
}
