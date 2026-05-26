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
//! Then `partition_build` builds one TGSM hash table per partition.
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
//!
//! ## Two entry points
//!
//! - [`partition_and_build`]: one-shot. Allocates a [`BuildScratch`]
//!   internally per call. Convenient for tests/benches.
//! - [`partition_and_build_with_scratch`]: persistent scratch. Callers
//!   that invoke the build repeatedly (e.g. the engine's UDF dispatch)
//!   should keep a `BuildScratch` and pass it in to avoid per-call
//!   MTLBuffer allocation (~10ms at 10M rows).

use std::mem::size_of;

use polars_metal_buffer::MetalDevice;

use crate::command::CommandQueue;
use crate::shader_lib::shared_library;

use super::scratch::BuildScratch;
use super::{BuildOutput, PartitionedBuildError};

/// Output of [`partition_and_scatter`]. `row_indices` is grouped by
/// partition; `partition_offsets[p]..partition_offsets[p+1]` is partition
/// `p`'s slice. `partition_id_per_row[r]` is the partition that row `r`
/// belongs to (persisted by the scatter kernel so callers don't re-hash).
pub struct ScatterOutput {
    pub row_indices: Vec<u32>,
    pub partition_offsets: Vec<u32>,
    pub partition_id_per_row: Vec<u32>,
}

/// One-shot scatter dispatch — primarily for tests. Production callers
/// should use [`partition_and_build_with_scratch`] which fuses scatter +
/// build into a single pipeline with persistent scratch.
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

    let mut scratch = BuildScratch::new(device)?;
    scratch.ensure(device, keys.len(), n_partitions)?;
    let n_rows: u32 = keys
        .len()
        .try_into()
        .map_err(|_| PartitionedBuildError::RowOverflow)?;
    scratch.reset(n_rows, n_partitions, LOG2_TGSM);
    scratch.write_keys(keys);

    let mut queue = CommandQueue::new(device)?;
    let partition_offsets = scatter_phase(device, &mut scratch, &mut queue, n_rows, n_partitions)?;

    let row_indices = readback_u32(&scratch.row_idx, n_rows as usize);
    let partition_id_per_row = readback_u32(&scratch.pid, n_rows as usize);
    Ok(ScatterOutput {
        row_indices,
        partition_offsets,
        partition_id_per_row,
    })
}

/// Capability A1 build phase (one-shot, allocates a scratch per call).
///
/// See [`partition_and_build_with_scratch`] for the persistent-scratch
/// entry point preferred by the engine.
pub fn partition_and_build(
    device: &MetalDevice,
    keys: &[u128],
    n_partitions: u32,
) -> Result<BuildOutput, PartitionedBuildError> {
    let mut scratch = BuildScratch::new(device)?;
    partition_and_build_with_scratch(device, &mut scratch, keys, n_partitions)
}

/// Capability A1 build phase with caller-owned scratch buffers.
///
/// Returns [`BuildOutput`] (`row_to_group`, `first_row_per_group`,
/// `n_groups`). On TGSM overflow returns
/// [`PartitionedBuildError::Overflow`]; the caller is expected to fall
/// back to the CPU build phase (capability A2 is shipped but, per the
/// Phase 5 retrospective, slower than CPU at every tested cardinality).
///
/// `scratch` grows on demand if its capacity is insufficient; reuse the
/// same scratch across calls to amortize MTLBuffer allocation.
pub fn partition_and_build_with_scratch(
    device: &MetalDevice,
    scratch: &mut BuildScratch,
    keys: &[u128],
    n_partitions: u32,
) -> Result<BuildOutput, PartitionedBuildError> {
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

    scratch.ensure(device, keys.len(), np)?;
    scratch.reset(n_rows, np, LOG2_TGSM);
    scratch.write_keys(keys);

    let mut queue = CommandQueue::new(device)?;

    // 1. Scatter — populates scratch.row_idx, scratch.pid, scratch.offsets.
    let partition_offsets = scatter_phase(device, scratch, &mut queue, n_rows, np)?;

    // 2. Build dispatch.
    let lib = shared_library(device)?;
    let pso_build = lib.pipeline("partition_build")?;

    // tg_width=256: any reasonable width works (inner loop strides by tg_size).
    let tg_width = 256usize;
    queue.dispatch_1d_with_tg(
        &pso_build,
        &[
            &scratch.keys,
            &scratch.row_idx,
            &scratch.offsets,
            &scratch.r2lg,
            &scratch.ng_per_part,
            &scratch.overflow,
            &scratch.n_rows_buf,
        ],
        np as usize * tg_width,
        tg_width,
    )?;
    queue.wait_until_complete()?;

    // 3. Overflow check.
    let overflow = read_u32_le(&scratch.overflow, 0);
    if overflow != 0 {
        return Err(PartitionedBuildError::Overflow);
    }

    // 4. Readback.
    let n_groups_per_part = readback_u32(&scratch.ng_per_part, np as usize);
    let row_to_local_group = readback_u32(&scratch.r2lg, n_rows as usize);
    let partition_id_per_row = readback_u32(&scratch.pid, n_rows as usize);

    // 5. Exclusive scan → global group ids + first_row_per_group.
    let mut partition_group_offset = vec![0u32; np as usize + 1];
    for i in 0..np as usize {
        partition_group_offset[i + 1] = partition_group_offset[i] + n_groups_per_part[i];
    }
    let n_groups = partition_group_offset[np as usize];
    let mut row_to_group = vec![0u32; n_rows as usize];
    let mut first_row_per_group = vec![u32::MAX; n_groups as usize];
    for r in 0..n_rows as usize {
        let p = partition_id_per_row[r] as usize;
        let local = row_to_local_group[r];
        let global = partition_group_offset[p] + local;
        row_to_group[r] = global;
        if first_row_per_group[global as usize] == u32::MAX {
            first_row_per_group[global as usize] = r as u32;
        }
    }
    let _ = partition_offsets; // consumed by scatter_phase; not part of BuildOutput
    Ok(BuildOutput {
        row_to_group,
        first_row_per_group,
        n_groups,
    })
}

const LOG2_TGSM: u32 = 10; // 1024 TGSM slots per partition

/// Dispatches `partition_count` + CPU-side prefix scan + `partition_scatter`.
/// Side effects: populates scratch's counts, offsets, row_idx, pid, cursors.
/// Returns the partition_offsets host-side vector (used by callers that
/// need it; the device buffer is also populated in `scratch.offsets`).
fn scatter_phase(
    device: &MetalDevice,
    scratch: &mut BuildScratch,
    queue: &mut CommandQueue,
    n_rows: u32,
    n_partitions: u32,
) -> Result<Vec<u32>, PartitionedBuildError> {
    let np = n_partitions;
    let lib = shared_library(device)?;
    let pso_count = lib.pipeline("partition_count")?;
    let pso_scatter = lib.pipeline("partition_scatter")?;

    queue.dispatch_1d(
        &pso_count,
        &[
            &scratch.keys,
            &scratch.counts,
            &scratch.n_rows_buf,
            &scratch.n_part_buf,
            &scratch.log2_buf,
        ],
        n_rows as usize,
    )?;
    queue.wait_until_complete()?;

    // CPU exclusive scan: counts → partition_offsets[np + 1].
    let counts = readback_u32(&scratch.counts, np as usize);
    let mut partition_offsets = vec![0u32; np as usize + 1];
    for i in 0..np as usize {
        partition_offsets[i + 1] = partition_offsets[i] + counts[i];
    }
    scratch.write_offsets(&partition_offsets);

    queue.dispatch_1d(
        &pso_scatter,
        &[
            &scratch.keys,
            &scratch.offsets,
            &scratch.cursors,
            &scratch.row_idx,
            &scratch.pid,
            &scratch.n_rows_buf,
            &scratch.n_part_buf,
            &scratch.log2_buf,
        ],
        n_rows as usize,
    )?;
    queue.wait_until_complete()?;

    Ok(partition_offsets)
}

fn readback_u32(buf: &polars_metal_buffer::MetalBuffer, n: usize) -> Vec<u32> {
    let bytes = buf.as_slice();
    let mut out = vec![0u32; n];
    for (i, v) in out.iter_mut().enumerate() {
        let b = &bytes[i * size_of::<u32>()..(i + 1) * size_of::<u32>()];
        *v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
    out
}

fn read_u32_le(buf: &polars_metal_buffer::MetalBuffer, idx: usize) -> u32 {
    let bytes = buf.as_slice();
    let off = idx * size_of::<u32>();
    u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
}
