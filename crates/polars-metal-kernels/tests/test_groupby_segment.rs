#![allow(clippy::expect_used)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::groupby_build_sort::gpu::sort_and_segment;
use polars_metal_kernels::groupby_build_sort::reference::cpu_sort_segment;
use proptest::prelude::*;

#[test]
fn empty_input_yields_zero_groups() {
    let device = MetalDevice::system_default().expect("metal device");
    let out = sort_and_segment(&device, &[]).expect("sort_and_segment");
    assert_eq!(out.n_groups, 0);
    assert!(out.row_to_group.is_empty());
}

#[test]
fn all_same_yields_one_group() {
    let device = MetalDevice::system_default().expect("metal device");
    let keys: Vec<u128> = vec![42u128; 100];
    let out = sort_and_segment(&device, &keys).expect("sort_and_segment");
    assert_eq!(out.n_groups, 1);
    assert!(out.row_to_group.iter().all(|&g| g == 0));
}

#[test]
fn all_distinct_yields_n_groups() {
    let device = MetalDevice::system_default().expect("metal device");
    let keys: Vec<u128> = (0u128..200).collect();
    let out = sort_and_segment(&device, &keys).expect("sort_and_segment");
    assert_eq!(out.n_groups, 200);
}

// Reduced proptest budget — sort dispatches 32 kernels per case (16 lanes ×
// 2 kernels) plus 1 for the segment pass. Keep total under ~500 dispatches.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn sort_and_segment_matches_cpu_reference(
        keys in proptest::collection::vec(any::<u128>(), 1..256usize),
    ) {
        let device = MetalDevice::system_default().expect("metal device");
        let gpu_out = sort_and_segment(&device, &keys).expect("sort_and_segment");
        let cpu_out = cpu_sort_segment(&keys);

        prop_assert_eq!(gpu_out.n_groups, cpu_out.n_groups);
        // Equivalence-class check: both row_to_group's induce the same
        // partition of rows. (Numbering may differ; we already proved
        // sort is stable so they SHOULD match exactly but the partition
        // property is what really matters for groupby correctness.)
        for a in 0..keys.len() {
            for b in 0..keys.len() {
                let gpu_same = gpu_out.row_to_group[a] == gpu_out.row_to_group[b];
                let cpu_same = cpu_out.row_to_group[a] == cpu_out.row_to_group[b];
                prop_assert_eq!(gpu_same, cpu_same);
            }
        }
    }
}
