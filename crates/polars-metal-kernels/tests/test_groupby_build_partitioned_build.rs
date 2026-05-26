//! GPU `partition_and_build` must induce the same equivalence classes
//! over rows as the CPU reference (`cpu_partitioned_hash`). Group
//! numbering may differ between CPU and GPU because the GPU's
//! per-partition local-id assignment depends on thread scheduling, so
//! direct `==` on `row_to_group` is wrong; we check that "rows a and b
//! land in the same group" agrees CPU/GPU for every pair.
#![allow(clippy::expect_used)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::groupby_build_partitioned::gpu::partition_and_build;
use polars_metal_kernels::groupby_build_partitioned::reference::cpu_partitioned_hash;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn build_matches_cpu_reference(
        keys in proptest::collection::vec(any::<u128>(), 1..2048usize),
        n_partitions in proptest::sample::select(vec![4u32, 8, 16, 32]),
    ) {
        let device =
            MetalDevice::system_default().expect("Metal-capable hardware required for this test");
        let gpu_out = partition_and_build(&device, &keys, n_partitions)
            .expect("partition_and_build dispatch must succeed");
        let cpu_out = cpu_partitioned_hash(&keys, n_partitions);

        prop_assert_eq!(gpu_out.n_groups, cpu_out.n_groups);
        // Equivalence-class check: GPU's grouping must induce the same
        // partition of rows as CPU's grouping (numbering may differ).
        for a in 0..keys.len() {
            for b in 0..keys.len() {
                let gpu_same = gpu_out.row_to_group[a] == gpu_out.row_to_group[b];
                let cpu_same = cpu_out.row_to_group[a] == cpu_out.row_to_group[b];
                prop_assert_eq!(gpu_same, cpu_same);
            }
        }
    }

    #[test]
    fn build_handles_extreme_collision_case(
        seed in any::<u64>(),
    ) {
        let device =
            MetalDevice::system_default().expect("Metal-capable hardware required for this test");
        // 100 keys constructed to share the low 10 bits -> same TGSM slot
        // in the same partition. Stresses the linear-probe chain.
        let keys: Vec<u128> = (0..100u128)
            .map(|i| (i << 10) | (seed as u128))
            .collect();
        let gpu_out = partition_and_build(&device, &keys, 4)
            .expect("partition_and_build dispatch must succeed");
        let cpu_out = cpu_partitioned_hash(&keys, 4);
        prop_assert_eq!(gpu_out.n_groups, cpu_out.n_groups);
    }
}
