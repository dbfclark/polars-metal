//! GPU `partition_and_scatter` must produce a layout that matches the
//! CPU reference (`cpu_partition_layout`) bucket-for-bucket. Rows within
//! a partition may appear in any order — the GPU uses an atomic write
//! cursor — so we compare sorted slices per partition.
#![allow(clippy::expect_used)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::groupby_build_partitioned::gpu::partition_and_scatter;
use polars_metal_kernels::groupby_build_partitioned::reference::cpu_partition_layout;
use proptest::prelude::*;

#[test]
fn scatter_produces_partition_layout_matching_cpu_reference() {
    let device =
        MetalDevice::system_default().expect("Metal-capable hardware required for this test");
    let keys: Vec<u128> = vec![10, 20, 30, 10, 20, 30, 10, 50];
    let n_partitions = 4;
    let out = partition_and_scatter(&device, &keys, n_partitions)
        .expect("partition_and_scatter dispatch must succeed");
    let mut seen = vec![false; keys.len()];
    for &r in &out.row_indices {
        seen[r as usize] = true;
    }
    assert!(seen.iter().all(|&b| b));
    for w in out.partition_offsets.windows(2) {
        assert!(w[0] <= w[1]);
    }
    assert_eq!(
        *out.partition_offsets
            .last()
            .expect("partition_offsets has n_partitions+1 >= 1 elements"),
        keys.len() as u32
    );
    // Each row's partition_id matches the partition it landed in.
    use polars_metal_kernels::groupby_build_partitioned::reference::partition_id;
    for (r, &k) in keys.iter().enumerate() {
        assert_eq!(out.partition_id_per_row[r], partition_id(k, n_partitions));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn scatter_proptest_matches_cpu_reference(
        keys in proptest::collection::vec(any::<u128>(), 1..1024usize),
        n_part in proptest::sample::select(vec![2u32, 4, 8, 16]),
    ) {
        let device =
            MetalDevice::system_default().expect("Metal-capable hardware required for this test");
        let out = partition_and_scatter(&device, &keys, n_part)
            .expect("partition_and_scatter dispatch must succeed");
        let cpu = cpu_partition_layout(&keys, n_part);
        prop_assert_eq!(&out.partition_offsets, &cpu.partition_offsets);
        for p in 0..n_part as usize {
            let s = out.partition_offsets[p] as usize;
            let e = out.partition_offsets[p + 1] as usize;
            let mut gpu_slice: Vec<u32> = out.row_indices[s..e].to_vec();
            let mut cpu_slice: Vec<u32> = cpu.row_indices[s..e].to_vec();
            gpu_slice.sort_unstable();
            cpu_slice.sort_unstable();
            prop_assert_eq!(gpu_slice, cpu_slice);
        }
        // partition_id_per_row matches the CPU reference for every row.
        use polars_metal_kernels::groupby_build_partitioned::reference::partition_id;
        for (r, &k) in keys.iter().enumerate() {
            prop_assert_eq!(out.partition_id_per_row[r], partition_id(k, n_part));
        }
    }
}
