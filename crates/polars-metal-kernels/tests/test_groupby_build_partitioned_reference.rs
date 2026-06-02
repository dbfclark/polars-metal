//! The CPU reference is the ground truth for proptest comparisons.
//! It implements the *same algorithm* the GPU runs — not a high-level
//! HashMap — so that any algorithmic bug surfaces equally on both sides.

use polars_metal_kernels::groupby_build_partitioned::reference::cpu_partitioned_hash;

#[test]
fn empty_input_yields_zero_groups() {
    let out = cpu_partitioned_hash(&[], /*n_partitions=*/ 4);
    assert_eq!(out.n_groups, 0);
    assert!(out.row_to_group.is_empty());
}

#[test]
fn all_same_key_yields_one_group() {
    let keys: Vec<u128> = vec![0xdeadbeef_cafebabe; 100];
    let out = cpu_partitioned_hash(&keys, 4);
    assert_eq!(out.n_groups, 1);
    assert!(out.row_to_group.iter().all(|&g| g == 0));
}

#[test]
fn all_distinct_keys_yields_n_groups() {
    let keys: Vec<u128> = (0u128..256).collect();
    let out = cpu_partitioned_hash(&keys, 4);
    assert_eq!(out.n_groups, 256);
    let mut seen = std::collections::HashSet::new();
    for &g in &out.row_to_group {
        assert!(seen.insert(g));
    }
}

#[test]
fn round_trip_first_row_per_group_indexes_original_rows() {
    let keys: Vec<u128> = vec![10, 20, 10, 30, 20, 10];
    let out = cpu_partitioned_hash(&keys, 4);
    for g in 0..out.n_groups {
        let fr = out.first_row_per_group[g as usize] as usize;
        for (r, &group_of_r) in out.row_to_group.iter().enumerate() {
            if group_of_r == g {
                assert_eq!(keys[r], keys[fr]);
            }
        }
    }
}
