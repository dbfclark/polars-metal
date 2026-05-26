use super::BuildOutput;

/// xxhash-style mixing function. Same constants as the MSL implementation.
pub fn hash_u128(key: u128) -> u64 {
    let mut h = 0x9E3779B97F4A7C15u64;
    h ^= (key as u64).wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= ((key >> 64) as u64).wrapping_mul(0x94D049BB133111EB);
    h ^= h >> 31;
    h.wrapping_mul(0x9E3779B97F4A7C15)
}

pub const TGSM_SLOTS_PER_PARTITION: u32 = 1024;

/// Public partition-id helper. Matches the MSL `partition_id` exactly so the
/// CPU reference and the GPU scatter produce identical buckets.
pub fn partition_id(key: u128, n_partitions: u32) -> u32 {
    let h = hash_u128(key);
    ((h >> TGSM_SLOTS_PER_PARTITION.trailing_zeros()) & (n_partitions as u64 - 1)) as u32
}

/// Layout produced by `cpu_partition_layout`. Matches GPU's output shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartitionLayout {
    /// Row indices grouped by partition (length = keys.len()).
    pub row_indices: Vec<u32>,
    /// Partition boundaries (length = n_partitions + 1). `row_indices[offsets[p]..offsets[p+1]]`
    /// holds partition `p`.
    pub partition_offsets: Vec<u32>,
}

/// Compute the per-partition row layout deterministically on CPU.
/// Rows within a partition are emitted in input order (the GPU's
/// `atomic_fetch_add` cursor produces an arbitrary order — callers must
/// compare sets-per-partition, not order-per-partition).
pub fn cpu_partition_layout(keys: &[u128], n_partitions: u32) -> PartitionLayout {
    assert!(n_partitions.is_power_of_two() && n_partitions > 0);
    if keys.is_empty() {
        return PartitionLayout {
            row_indices: vec![],
            partition_offsets: vec![0u32; n_partitions as usize + 1],
        };
    }
    let np = n_partitions as usize;
    let mut counts = vec![0u32; np];
    for &k in keys {
        let p = partition_id(k, n_partitions) as usize;
        counts[p] += 1;
    }
    let mut partition_offsets = vec![0u32; np + 1];
    for p in 0..np {
        partition_offsets[p + 1] = partition_offsets[p] + counts[p];
    }
    let mut cursors = partition_offsets[..np].to_vec();
    let mut row_indices = vec![0u32; keys.len()];
    for (r, &k) in keys.iter().enumerate() {
        let p = partition_id(k, n_partitions) as usize;
        row_indices[cursors[p] as usize] = r as u32;
        cursors[p] += 1;
    }
    PartitionLayout {
        row_indices,
        partition_offsets,
    }
}

pub fn cpu_partitioned_hash(keys: &[u128], n_partitions: u32) -> BuildOutput {
    if keys.is_empty() {
        return BuildOutput {
            row_to_group: vec![],
            first_row_per_group: vec![],
            n_groups: 0,
        };
    }
    assert!(n_partitions.is_power_of_two() && n_partitions > 0);

    let mut rows_by_partition: Vec<Vec<u32>> = vec![Vec::new(); n_partitions as usize];
    for (r, &k) in keys.iter().enumerate() {
        let part = partition_id(k, n_partitions) as usize;
        rows_by_partition[part].push(r as u32);
    }

    let mut per_partition_groups: Vec<Vec<(u128, u32)>> = vec![Vec::new(); n_partitions as usize];
    let mut row_local_group: Vec<u32> = vec![0; keys.len()];
    for (p, rows) in rows_by_partition.iter().enumerate() {
        let table = &mut per_partition_groups[p];
        let cap = (rows.len() * 2).next_power_of_two().max(8);
        let mut slots: Vec<Option<(u128, u32)>> = vec![None; cap];
        let mut local_next = 0u32;
        for &r in rows {
            let k = keys[r as usize];
            let h = hash_u128(k) as usize;
            let mut idx = h & (cap - 1);
            loop {
                match slots[idx] {
                    None => {
                        slots[idx] = Some((k, local_next));
                        table.push((k, local_next));
                        row_local_group[r as usize] = local_next;
                        local_next += 1;
                        break;
                    }
                    Some((existing_k, gid)) if existing_k == k => {
                        row_local_group[r as usize] = gid;
                        break;
                    }
                    Some(_) => {
                        idx = (idx + 1) & (cap - 1);
                    }
                }
            }
        }
    }

    let mut partition_offset = vec![0u32; n_partitions as usize + 1];
    for (p, table) in per_partition_groups.iter().enumerate() {
        partition_offset[p + 1] = partition_offset[p] + table.len() as u32;
    }
    // SAFETY: partition_offset has n_partitions+1 elements; n_partitions > 0.
    let n_groups = partition_offset[n_partitions as usize];
    let mut row_to_group = vec![0u32; keys.len()];
    let mut first_row_per_group = vec![u32::MAX; n_groups as usize];
    for (r, &k) in keys.iter().enumerate() {
        let part = partition_id(k, n_partitions) as usize;
        let local = row_local_group[r];
        let global = partition_offset[part] + local;
        row_to_group[r] = global;
        if first_row_per_group[global as usize] == u32::MAX {
            first_row_per_group[global as usize] = r as u32;
        }
    }

    BuildOutput {
        row_to_group,
        first_row_per_group,
        n_groups,
    }
}
