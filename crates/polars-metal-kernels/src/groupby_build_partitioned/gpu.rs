//! GPU dispatch for capability A1's partition-and-scatter phase.
//!
//! Two-pass scatter (see `shaders/groupby_build_partitioned_scatter.metal`):
//!   1. `partition_count`: each row atomically increments the count of
//!      its destination partition.
//!   2. CPU exclusive-scan over counts -> `partition_offsets`.
//!   3. `partition_scatter`: each row atomically claims a slot inside its
//!      partition lane and writes its row index into
//!      `row_indices_out[partition_offsets[p] + slot]`. Also persists its
//!      `partition_id` into `partition_id_per_row` so the build phase's
//!      final row → global-group derivation doesn't need to re-hash.
//!
//! Rows within a partition appear in arbitrary order (depends on atomic-
//! cursor scheduling). The CPU reference in
//! [`crate::groupby_build_partitioned::reference::cpu_partition_layout`]
//! emits rows in input order; tests must compare sets-per-partition, not
//! order-per-partition.
//!
//! Keys cross the FFI boundary as raw bytes — `&[u128]` is layout-
//! compatible with MSL `ulong2` on little-endian Apple Silicon (16 bytes,
//! 16-byte aligned, x = low 64 bits, y = high 64 bits).

use std::mem::size_of;

use polars_metal_buffer::MetalDevice;

use crate::command::CommandQueue;
use crate::shader_lib::shared_library;

use super::PartitionedBuildError;

/// Reinterpret `&[u128]` as a flat `&[u8]` for buffer upload. The u128
/// layout on Apple Silicon (LE, 16-byte aligned) matches MSL `ulong2`
/// (`.x` = low 64 bits, `.y` = high 64 bits), so the kernel sees the
/// same bytes the host wrote with no swizzling.
///
/// SAFETY: u128 is POD; the resulting slice is valid for the duration of
/// the borrow. `new_buffer_from_bytes` copies synchronously.
#[inline]
fn keys_as_bytes(keys: &[u128]) -> &[u8] {
    // SAFETY: see doc comment above.
    unsafe { std::slice::from_raw_parts(keys.as_ptr() as *const u8, std::mem::size_of_val(keys)) }
}

#[inline]
fn u32_bytes(s: &[u32]) -> &[u8] {
    // SAFETY: u32 is POD.
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

/// Output of [`partition_and_scatter`]. `row_indices` is grouped by
/// partition; `partition_offsets[p]..partition_offsets[p+1]` is partition
/// `p`'s slice. `partition_id_per_row[r]` is the partition that row `r`
/// belongs to (persisted by the scatter kernel so callers don't re-hash).
pub struct ScatterOutput {
    pub row_indices: Vec<u32>,
    pub partition_offsets: Vec<u32>,
    pub partition_id_per_row: Vec<u32>,
}

/// Dispatch `partition_count` + `partition_scatter`.
///
/// `n_partitions` must be a power of two and > 0. Empty `keys` returns
/// zero-length vectors with `partition_offsets = vec![0; n_partitions + 1]`.
pub fn partition_and_scatter(
    device: &MetalDevice,
    keys: &[u128],
    n_partitions: u32,
) -> Result<ScatterOutput, PartitionedBuildError> {
    if keys.is_empty() {
        return Ok(ScatterOutput {
            row_indices: vec![],
            partition_offsets: vec![0u32; n_partitions as usize + 1],
            partition_id_per_row: vec![],
        });
    }
    assert!(n_partitions.is_power_of_two() && n_partitions > 0);

    let n_rows: u32 = keys
        .len()
        .try_into()
        .map_err(|_| PartitionedBuildError::RowOverflow)?;
    let log2_tgsm = 10u32;
    let np = n_partitions;

    let lib = shared_library(device)?;
    let pso_count = lib.pipeline("partition_count")?;
    let pso_scatter = lib.pipeline("partition_scatter")?;

    let buf_keys = device.new_buffer_from_bytes(keys_as_bytes(keys))?;
    let buf_counts = device.new_buffer_zeroed(np as usize * size_of::<u32>())?;
    let buf_n_rows = device.new_buffer_from_bytes(&n_rows.to_le_bytes())?;
    let buf_n_part = device.new_buffer_from_bytes(&np.to_le_bytes())?;
    let buf_log2 = device.new_buffer_from_bytes(&log2_tgsm.to_le_bytes())?;

    let mut queue = CommandQueue::new(device)?;
    queue.dispatch_1d(
        &pso_count,
        &[&buf_keys, &buf_counts, &buf_n_rows, &buf_n_part, &buf_log2],
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
    let buf_pid = device.new_buffer_zeroed(n_rows as usize * size_of::<u32>())?;

    queue.dispatch_1d(
        &pso_scatter,
        &[
            &buf_keys,
            &buf_offsets,
            &buf_cursors,
            &buf_row_idx,
            &buf_pid,
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
    let pid_bytes = buf_pid.as_slice();
    let mut partition_id_per_row = vec![0u32; n_rows as usize];
    for (i, p) in partition_id_per_row.iter_mut().enumerate() {
        let b = &pid_bytes[i * 4..(i + 1) * 4];
        *p = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
    Ok(ScatterOutput {
        row_indices,
        partition_offsets,
        partition_id_per_row,
    })
}

/// Capability A1 build phase: partition rows, then build one TGSM hash
/// table per partition.
///
/// Returns the same [`BuildOutput`] shape as the CPU
/// [`reference::cpu_partitioned_hash`] (`row_to_group`,
/// `first_row_per_group`, `n_groups`). Numbering of groups may differ
/// from the CPU reference because the GPU's per-partition local-id
/// assignment depends on thread scheduling; callers that diff against
/// the reference must use equivalence-class checks.
///
/// On TGSM overflow (any single probe chain exceeded the kernel's probe
/// limit), returns [`PartitionedBuildError::Overflow`]; the caller is
/// expected to fall back to the CPU build phase (capability A2 is
/// shipped but, per the Phase 5 retrospective, slower than CPU at every
/// tested cardinality).
pub fn partition_and_build(
    device: &MetalDevice,
    keys: &[u128],
    n_partitions: u32,
) -> Result<crate::groupby_build_partitioned::BuildOutput, PartitionedBuildError> {
    use crate::groupby_build_partitioned::BuildOutput;

    if keys.is_empty() {
        return Ok(BuildOutput {
            row_to_group: vec![],
            first_row_per_group: vec![],
            n_groups: 0,
        });
    }
    assert!(n_partitions.is_power_of_two() && n_partitions > 0);
    let n_rows: u32 = keys
        .len()
        .try_into()
        .map_err(|_| PartitionedBuildError::RowOverflow)?;
    let np = n_partitions;

    // 1. Scatter (also yields per-row partition_id).
    let scatter = partition_and_scatter(device, keys, np)?;

    // 2. Build dispatch.
    let lib = shared_library(device)?;
    let pso_build = lib.pipeline("partition_build")?;

    let buf_keys = device.new_buffer_from_bytes(keys_as_bytes(keys))?;
    let buf_row_idx = device.new_buffer_from_bytes(u32_bytes(&scatter.row_indices))?;
    let buf_offsets = device.new_buffer_from_bytes(u32_bytes(&scatter.partition_offsets))?;
    let buf_r2lg = device.new_buffer_zeroed(n_rows as usize * size_of::<u32>())?;
    let buf_ng_per_part = device.new_buffer_zeroed(np as usize * size_of::<u32>())?;
    let buf_overflow = device.new_buffer_zeroed(size_of::<u32>())?;
    let buf_n_rows = device.new_buffer_from_bytes(&n_rows.to_le_bytes())?;

    // One threadgroup per partition. tg_width=256 chosen as a generic
    // worker count; the inner loop is `for i in tid; ... ; i += tg_size`,
    // so any reasonable tg_width works.
    let tg_width = 256usize;
    let mut queue = CommandQueue::new(device)?;
    queue.dispatch_1d_with_tg(
        &pso_build,
        &[
            &buf_keys,
            &buf_row_idx,
            &buf_offsets,
            &buf_r2lg,
            &buf_ng_per_part,
            &buf_overflow,
            &buf_n_rows,
        ],
        np as usize * tg_width,
        tg_width,
    )?;
    queue.wait_until_complete()?;

    // 3. Overflow check.
    let of_bytes = buf_overflow.as_slice();
    let overflow = u32::from_le_bytes([of_bytes[0], of_bytes[1], of_bytes[2], of_bytes[3]]);
    if overflow != 0 {
        return Err(PartitionedBuildError::Overflow);
    }

    // 4. Readback.
    let ngp_bytes = buf_ng_per_part.as_slice();
    let mut n_groups_per_part = vec![0u32; np as usize];
    for (i, n) in n_groups_per_part.iter_mut().enumerate() {
        let b = &ngp_bytes[i * 4..(i + 1) * 4];
        *n = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
    let r2lg_bytes = buf_r2lg.as_slice();
    let mut row_to_local_group = vec![0u32; n_rows as usize];
    for (i, v) in row_to_local_group.iter_mut().enumerate() {
        let b = &r2lg_bytes[i * 4..(i + 1) * 4];
        *v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }

    // 5. Exclusive scan -> global group ids + first_row_per_group.
    // Uses the per-row partition_id that the scatter kernel already
    // computed; avoids re-hashing each key on the CPU.
    let mut partition_group_offset = vec![0u32; np as usize + 1];
    for i in 0..np as usize {
        partition_group_offset[i + 1] = partition_group_offset[i] + n_groups_per_part[i];
    }
    let n_groups = partition_group_offset[np as usize];
    let mut row_to_group = vec![0u32; n_rows as usize];
    let mut first_row_per_group = vec![u32::MAX; n_groups as usize];
    for r in 0..n_rows as usize {
        let p = scatter.partition_id_per_row[r] as usize;
        let local = row_to_local_group[r];
        let global = partition_group_offset[p] + local;
        row_to_group[r] = global;
        if first_row_per_group[global as usize] == u32::MAX {
            first_row_per_group[global as usize] = r as u32;
        }
    }
    Ok(BuildOutput {
        row_to_group,
        first_row_per_group,
        n_groups,
    })
}
