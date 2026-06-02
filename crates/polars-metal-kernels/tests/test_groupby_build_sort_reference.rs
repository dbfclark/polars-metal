use polars_metal_kernels::groupby_build_sort::reference::cpu_sort_segment;

#[test]
fn empty_input_yields_zero_groups() {
    let out = cpu_sort_segment(&[]);
    assert_eq!(out.n_groups, 0);
}

#[test]
fn all_unique_yields_n_groups() {
    let keys: Vec<u128> = (0u128..1000).collect();
    let out = cpu_sort_segment(&keys);
    assert_eq!(out.n_groups, 1000);
}

#[test]
fn all_same_yields_one_group() {
    let keys: Vec<u128> = vec![42u128; 1000];
    let out = cpu_sort_segment(&keys);
    assert_eq!(out.n_groups, 1);
    assert!(out.row_to_group.iter().all(|&g| g == 0));
}

#[test]
fn duplicates_collapsed_in_arbitrary_order() {
    let keys: Vec<u128> = vec![10, 20, 10, 30, 20, 10];
    let out = cpu_sort_segment(&keys);
    assert_eq!(out.n_groups, 3);
    let g_for = |original_idx: usize| out.row_to_group[original_idx];
    assert_eq!(g_for(0), g_for(2));
    assert_eq!(g_for(0), g_for(5));
    assert_eq!(g_for(1), g_for(4));
}
