// crates/polars-metal-kernels/tests/test_groupby_build.rs
//
// Proptest the build phase against a pure-Rust hash-table reference.
#![allow(clippy::expect_used, clippy::panic)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::groupby::{dispatch_build, dispatch_hash};
use proptest::prelude::*;
use std::collections::HashMap;

fn build_reference(encoded: &[u128]) -> (Vec<u32>, u32) {
    let mut group_for_key: HashMap<u128, u32> = HashMap::new();
    let mut next_gid: u32 = 0;
    let mut row_to_group = Vec::with_capacity(encoded.len());
    for &k in encoded {
        let gid = *group_for_key.entry(k).or_insert_with(|| {
            let g = next_gid;
            next_gid += 1;
            g
        });
        row_to_group.push(gid);
    }
    (row_to_group, next_gid)
}

/// Check that two group-assignment vectors define the same equivalence classes.
/// The group IDs themselves may differ (both are valid labellings), but any two
/// rows that are in the same group in one assignment must be in the same group
/// in the other.
fn same_equivalence_classes(a: &[u32], b: &[u32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        for j in (i + 1)..a.len() {
            let same_a = a[i] == a[j];
            let same_b = b[i] == b[j];
            if same_a != same_b {
                return false;
            }
        }
    }
    true
}

fn run_build(encoded: &[u128]) -> (Vec<u32>, u32, Vec<u32>) {
    let device = MetalDevice::system_default().expect("Metal-capable hardware required");
    let mut queue = CommandQueue::new(&device).expect("queue");
    let mut hashes = vec![0u32; encoded.len()];
    dispatch_hash(&device, &mut queue, encoded, encoded.len(), &mut hashes).expect("dispatch_hash");
    let out = dispatch_build(&device, &mut queue, encoded, &hashes, encoded.len())
        .expect("dispatch_build");
    (out.row_to_group, out.group_count, out.first_row_per_group)
}

#[test]
fn all_distinct_keys_produce_one_group_per_row() {
    let encoded: Vec<u128> = (1..=128u128).collect();
    let (r2g, count, first_rows) = run_build(&encoded);
    assert_eq!(count, 128);
    assert_eq!(r2g.len(), 128);
    assert_eq!(first_rows.len(), 128);
    // All group IDs should be distinct.
    let mut sorted = r2g.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 128);
}

#[test]
fn all_same_keys_produce_one_group() {
    let encoded = vec![42u128; 1024];
    let (r2g, count, first_rows) = run_build(&encoded);
    assert_eq!(count, 1);
    assert!(
        r2g.iter().all(|&g| g == r2g[0]),
        "all rows with the same key must map to the same group"
    );
    assert_eq!(first_rows.len(), 1);
}

#[test]
fn four_groups_ten_thousand_rows_modeled_q1() {
    let mut encoded = Vec::with_capacity(10_000);
    for i in 0..10_000 {
        encoded.push((i % 4) as u128);
    }
    let (r2g, count, first_rows) = run_build(&encoded);
    assert_eq!(count, 4);
    assert_eq!(first_rows.len(), 4);
    // Equivalence check: two rows share a group iff they have the same key.
    for i in 0..10_000 {
        for j in (i + 1)..10_000 {
            let same_key = (i % 4) == (j % 4);
            let same_group = r2g[i] == r2g[j];
            assert_eq!(
                same_key, same_group,
                "row {i} vs row {j}: same_key={same_key}, same_group={same_group}"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn kernel_matches_reference_equivalence_classes(
        keys in prop::collection::vec(0u128..=16u128, 4..=512),
    ) {
        let (kernel_r2g, kernel_count, _first) = run_build(&keys);
        let (ref_r2g, ref_count) = build_reference(&keys);
        prop_assert_eq!(kernel_count, ref_count);
        prop_assert!(
            same_equivalence_classes(&kernel_r2g, &ref_r2g),
            "kernel: {:?} ref: {:?}",
            kernel_r2g,
            ref_r2g
        );
    }
}
