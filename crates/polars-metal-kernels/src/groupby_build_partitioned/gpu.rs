//! GPU dispatch for capability A1's partition-and-scatter phase.
//!
//! Two-pass scatter (see `shaders/groupby_build_partitioned_scatter.metal`):
//!   1. `partition_count`: each row atomically increments the count of
//!      its destination partition.
//!   2. CPU exclusive-scan over counts -> `partition_offsets`.
//!   3. `partition_scatter`: each row atomically claims a slot inside its
//!      partition lane and writes its row index into
//!      `row_indices_out[partition_offsets[p] + slot]`.
//!
//! Rows within a partition appear in arbitrary order (depends on atomic-
//! cursor scheduling). The CPU reference in
//! [`crate::groupby_build_partitioned::reference::cpu_partition_layout`]
//! emits rows in input order; tests must compare sets-per-partition, not
//! order-per-partition.

use std::mem::size_of;

use polars_metal_buffer::MetalDevice;

use crate::command::CommandQueue;
use crate::shader_lib::shared_library;

use super::PartitionedBuildError;

/// Dispatch `partition_count` + `partition_scatter` and return
/// `(row_indices, partition_offsets)`.
///
/// `n_partitions` must be a power of two and > 0. Empty `keys` returns
/// `(vec![], vec![0; n_partitions + 1])`.
pub fn partition_and_scatter(
    device: &MetalDevice,
    keys: &[u128],
    n_partitions: u32,
) -> Result<(Vec<u32>, Vec<u32>), PartitionedBuildError> {
    if keys.is_empty() {
        return Ok((vec![], vec![0u32; n_partitions as usize + 1]));
    }
    assert!(n_partitions.is_power_of_two() && n_partitions > 0);

    let keys_lo: Vec<u64> = keys.iter().map(|k| *k as u64).collect();
    let keys_hi: Vec<u64> = keys.iter().map(|k| (*k >> 64) as u64).collect();
    let n_rows: u32 = keys
        .len()
        .try_into()
        .map_err(|_| PartitionedBuildError::RowOverflow)?;
    let log2_tgsm = 10u32;
    let np = n_partitions;

    // SAFETY: u64 / u32 are POD; reinterpret as bytes for the synchronous
    // copy performed inside `new_buffer_from_bytes`. Slices remain valid
    // for the call's duration.
    let u64_bytes = |s: &[u64]| unsafe {
        std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s))
    };
    let u32_bytes = |s: &[u32]| unsafe {
        std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s))
    };

    let lib = shared_library(device)?;
    let pso_count = lib.pipeline("partition_count")?;
    let pso_scatter = lib.pipeline("partition_scatter")?;

    let buf_keys_lo = device.new_buffer_from_bytes(u64_bytes(&keys_lo))?;
    let buf_keys_hi = device.new_buffer_from_bytes(u64_bytes(&keys_hi))?;
    let buf_counts = device.new_buffer_zeroed(np as usize * size_of::<u32>())?;
    let buf_n_rows = device.new_buffer_from_bytes(&n_rows.to_le_bytes())?;
    let buf_n_part = device.new_buffer_from_bytes(&np.to_le_bytes())?;
    let buf_log2 = device.new_buffer_from_bytes(&log2_tgsm.to_le_bytes())?;

    let mut queue = CommandQueue::new(device)?;
    queue.dispatch_1d(
        &pso_count,
        &[
            &buf_keys_lo,
            &buf_keys_hi,
            &buf_counts,
            &buf_n_rows,
            &buf_n_part,
            &buf_log2,
        ],
        n_rows as usize,
    )?;
    queue.wait_until_complete()?;

    let counts_bytes = buf_counts.as_slice();
    let mut counts = vec![0u32; np as usize];
    for (i, c) in counts.iter_mut().enumerate() {
        let b = &counts_bytes[i * 4..(i + 1) * 4];
        *c = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
    let mut partition_offsets = vec![0u32; np as usize + 1];
    for i in 0..np as usize {
        partition_offsets[i + 1] = partition_offsets[i] + counts[i];
    }

    let buf_offsets = device.new_buffer_from_bytes(u32_bytes(&partition_offsets))?;
    let buf_cursors = device.new_buffer_zeroed(np as usize * size_of::<u32>())?;
    let buf_row_idx = device.new_buffer_zeroed(n_rows as usize * size_of::<u32>())?;

    queue.dispatch_1d(
        &pso_scatter,
        &[
            &buf_keys_lo,
            &buf_keys_hi,
            &buf_offsets,
            &buf_cursors,
            &buf_row_idx,
            &buf_n_rows,
            &buf_n_part,
            &buf_log2,
        ],
        n_rows as usize,
    )?;
    queue.wait_until_complete()?;

    let row_bytes = buf_row_idx.as_slice();
    let mut row_indices = vec![0u32; n_rows as usize];
    for (i, r) in row_indices.iter_mut().enumerate() {
        let b = &row_bytes[i * 4..(i + 1) * 4];
        *r = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
    Ok((row_indices, partition_offsets))
}
