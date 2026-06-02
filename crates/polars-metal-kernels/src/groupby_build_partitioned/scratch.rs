//! Reusable scratch buffers for capability A1's GPU dispatch.
//!
//! Per the Phase 5 retrospective profiling, fresh MTLBuffer allocation
//! dominated A1's runtime at 10M rows (~12ms of a 50ms total = 23%).
//! A `BuildScratch` is allocated once and reused across calls; only
//! capacity growth triggers new MTLBuffer allocations.
//!
//! Usage:
//! ```ignore
//! let mut scratch = BuildScratch::new(&device, 0, 0)?;  // empty
//! for batch in batches {
//!     let out = partition_and_build_with_scratch(
//!         &device, &mut scratch, batch.keys, 16)?;
//!     // ...
//! }
//! ```

use std::mem::size_of;

use polars_metal_buffer::{MetalBuffer, MetalDevice};

use super::PartitionedBuildError;

/// Owns the device-side scratch buffers used by `partition_and_build`.
///
/// Capacity grows on demand (`ensure`); buffers persist across calls and
/// only the cheap zero-fill of accumulator buffers (counts / cursors /
/// overflow) happens each call.
pub struct BuildScratch {
    // Input-data buffer (host writes into it each call via as_mut_slice).
    pub(crate) keys: MetalBuffer,
    // Accumulator buffers that must be zeroed each call.
    pub(crate) counts: MetalBuffer,
    pub(crate) cursors: MetalBuffer,
    pub(crate) overflow: MetalBuffer,
    // Buffers fully written by the kernel/CPU each call (no zero needed).
    pub(crate) offsets: MetalBuffer,
    pub(crate) row_idx: MetalBuffer,
    pub(crate) pid: MetalBuffer,
    pub(crate) r2lg: MetalBuffer,
    pub(crate) ng_per_part: MetalBuffer,
    // Scalar constant buffers (rewritten each call with new values).
    pub(crate) n_rows_buf: MetalBuffer,
    pub(crate) n_part_buf: MetalBuffer,
    pub(crate) log2_buf: MetalBuffer,
    // Capacity tracking.
    capacity_rows: usize,
    capacity_partitions: u32,
}

impl BuildScratch {
    /// Allocate empty scratch (zero capacity). Callers grow via [`ensure`].
    pub fn new(device: &MetalDevice) -> Result<Self, PartitionedBuildError> {
        // Seed with 16 bytes (one u128 slot) so the small constant
        // buffers and ulong2-shaped buffer agree on minimum alignment.
        let seed = 16;
        Ok(Self {
            keys: device.new_buffer_zeroed(seed)?,
            counts: device.new_buffer_zeroed(seed)?,
            cursors: device.new_buffer_zeroed(seed)?,
            overflow: device.new_buffer_zeroed(seed)?,
            offsets: device.new_buffer_zeroed(seed)?,
            row_idx: device.new_buffer_zeroed(seed)?,
            pid: device.new_buffer_zeroed(seed)?,
            r2lg: device.new_buffer_zeroed(seed)?,
            ng_per_part: device.new_buffer_zeroed(seed)?,
            n_rows_buf: device.new_buffer_zeroed(size_of::<u32>())?,
            n_part_buf: device.new_buffer_zeroed(size_of::<u32>())?,
            log2_buf: device.new_buffer_zeroed(size_of::<u32>())?,
            capacity_rows: 0,
            capacity_partitions: 0,
        })
    }

    /// Resize scratch to fit `(n_rows, n_partitions)`. Idempotent; only
    /// reallocates the buffers that need to grow. New buffers are
    /// initialized to all-zero on allocation.
    pub fn ensure(
        &mut self,
        device: &MetalDevice,
        n_rows: usize,
        n_partitions: u32,
    ) -> Result<(), PartitionedBuildError> {
        let np = n_partitions as usize;
        if n_partitions > self.capacity_partitions {
            self.counts = device.new_buffer_zeroed(np * size_of::<u32>())?;
            self.cursors = device.new_buffer_zeroed(np * size_of::<u32>())?;
            self.offsets = device.new_buffer_zeroed((np + 1) * size_of::<u32>())?;
            self.ng_per_part = device.new_buffer_zeroed(np * size_of::<u32>())?;
            self.capacity_partitions = n_partitions;
        }
        if n_rows > self.capacity_rows {
            // Each key is 16 bytes (ulong2).
            self.keys = device.new_buffer_zeroed(n_rows * 16)?;
            self.row_idx = device.new_buffer_zeroed(n_rows * size_of::<u32>())?;
            self.pid = device.new_buffer_zeroed(n_rows * size_of::<u32>())?;
            self.r2lg = device.new_buffer_zeroed(n_rows * size_of::<u32>())?;
            self.capacity_rows = n_rows;
        }
        Ok(())
    }

    /// Copy `keys` into the scratch's keys buffer. Caller must `ensure`
    /// sufficient capacity first.
    pub fn write_keys(&mut self, keys: &[u128]) {
        let n_bytes = std::mem::size_of_val(keys);
        let dst = self.keys.as_mut_slice();
        // SAFETY: u128 is POD; reinterpret as bytes.
        let src: &[u8] = unsafe { std::slice::from_raw_parts(keys.as_ptr() as *const u8, n_bytes) };
        dst[..n_bytes].copy_from_slice(src);
    }

    /// Write `partition_offsets` into the scratch's offsets buffer.
    pub(crate) fn write_offsets(&mut self, partition_offsets: &[u32]) {
        let n_bytes = std::mem::size_of_val(partition_offsets);
        let dst = self.offsets.as_mut_slice();
        // SAFETY: u32 is POD.
        let src: &[u8] =
            unsafe { std::slice::from_raw_parts(partition_offsets.as_ptr() as *const u8, n_bytes) };
        dst[..n_bytes].copy_from_slice(src);
    }

    /// Zero the accumulator buffers (counts, cursors, overflow) and
    /// write the per-call scalar values (n_rows, n_partitions, log2_tgsm).
    /// Cheap; <1 µs for the sizes used in A1.
    pub fn reset(&mut self, n_rows: u32, n_partitions: u32, log2_tgsm: u32) {
        let np = n_partitions as usize;
        zero_prefix(&mut self.counts, np * size_of::<u32>());
        zero_prefix(&mut self.cursors, np * size_of::<u32>());
        zero_prefix(&mut self.overflow, size_of::<u32>());
        // n_groups_per_part is fully written by the kernel; no zero needed.
        // row_idx / pid / r2lg are fully written by the kernel; no zero needed.
        write_u32_le(&mut self.n_rows_buf, n_rows);
        write_u32_le(&mut self.n_part_buf, n_partitions);
        write_u32_le(&mut self.log2_buf, log2_tgsm);
    }

    pub fn capacity_rows(&self) -> usize {
        self.capacity_rows
    }

    pub fn capacity_partitions(&self) -> u32 {
        self.capacity_partitions
    }
}

fn zero_prefix(buf: &mut MetalBuffer, n_bytes: usize) {
    let bytes = buf.as_mut_slice();
    for b in &mut bytes[..n_bytes] {
        *b = 0;
    }
}

fn write_u32_le(buf: &mut MetalBuffer, value: u32) {
    let bytes = buf.as_mut_slice();
    let v = value.to_le_bytes();
    bytes[0] = v[0];
    bytes[1] = v[1];
    bytes[2] = v[2];
    bytes[3] = v[3];
}
