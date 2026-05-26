use crate::groupby_build_partitioned::BuildOutput;

pub fn cpu_sort_segment(keys: &[u128]) -> BuildOutput {
    if keys.is_empty() {
        return BuildOutput {
            row_to_group: vec![],
            first_row_per_group: vec![],
            n_groups: 0,
        };
    }
    let mut pairs: Vec<(u128, u32)> = keys
        .iter()
        .enumerate()
        .map(|(i, &k)| (k, i as u32))
        .collect();
    pairs.sort_unstable_by_key(|(k, _)| *k);

    let mut row_to_group = vec![0u32; keys.len()];
    let mut first_row_per_group: Vec<u32> = Vec::new();
    let mut cur_group: u32 = 0;
    first_row_per_group.push(pairs[0].1);
    row_to_group[pairs[0].1 as usize] = 0;
    for i in 1..pairs.len() {
        if pairs[i].0 != pairs[i - 1].0 {
            cur_group += 1;
            first_row_per_group.push(pairs[i].1);
        }
        row_to_group[pairs[i].1 as usize] = cur_group;
    }
    BuildOutput {
        row_to_group,
        first_row_per_group,
        n_groups: cur_group + 1,
    }
}
