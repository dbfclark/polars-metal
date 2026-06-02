//! Rust dispatch for the A3 global-atomic GPU hash kernel.

use std::mem::size_of;

use polars_metal_buffer::MetalDevice;

use crate::command::CommandQueue;
use crate::groupby_build_partitioned::BuildOutput;
use crate::shader_lib::shared_library;

use super::GlobalHashError;

/// Default linear-probe limit. 128 is generous at <50% load factor;
/// the spike measures whether contention-driven clustering pushes
/// real probe lengths past this. Raise if `Overflow` triggers when the
/// table is provably under-loaded.
const DEFAULT_MAX_PROBE: u32 = 128;

/// Default table-sizing oversubscription. Caller-supplied
/// `est_cardinality * OVERSUBSCRIBE` rounded up to a power of two.
/// 4× keeps load factor under 25%, making probe-length variance low.
const OVERSUBSCRIBE: usize = 4;

/// Build groupby IDs via the single-pass global-atomic GPU hash table.
///
/// `est_cardinality` is the host's best estimate of distinct keys; the
/// table is sized `next_power_of_two(est_cardinality * OVERSUBSCRIBE)`
/// to keep load factor < 50%. Underestimates inflate probe chains and
/// eventually trip `GlobalHashError::Overflow`.
///
/// `n_groups` in the returned [`BuildOutput`] is the count of distinct
/// keys observed. `first_row_per_group` is derived host-side from the
/// returned per-row group ids (one pass; trivial cost vs the GPU build).
pub fn global_hash_build(
    device: &MetalDevice,
    keys: &[u128],
    est_cardinality: usize,
) -> Result<BuildOutput, GlobalHashError> {
    if keys.is_empty() {
        return Ok(BuildOutput {
            row_to_group: vec![],
            first_row_per_group: vec![],
            n_groups: 0,
        });
    }
    let n_rows: u32 = keys
        .len()
        .try_into()
        .map_err(|_| GlobalHashError::RowOverflow)?;
    // Size the table for the larger of (a) est × OVERSUBSCRIBE and (b)
    // some minimum (1024) so very small inputs still have probe slack.
    let target = (est_cardinality.max(1))
        .saturating_mul(OVERSUBSCRIBE)
        .max(1024);
    let table_size: u32 = target
        .next_power_of_two()
        .try_into()
        .map_err(|_| GlobalHashError::RowOverflow)?;

    let lib = shared_library(device)?;
    let pso = lib.pipeline("global_hash_build")?;

    // SAFETY: u128 is layout-compatible with MSL ulong2 on Apple Silicon.
    let keys_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(keys.as_ptr() as *const u8, std::mem::size_of_val(keys))
    };

    let buf_keys = device.new_buffer_from_bytes(keys_bytes)?;
    let buf_slot_key = device.new_buffer_zeroed(table_size as usize * 16)?;
    let buf_slot_state = device.new_buffer_zeroed(table_size as usize * size_of::<u32>())?;
    let buf_next_gid = device.new_buffer_zeroed(size_of::<u32>())?;
    let buf_overflow = device.new_buffer_zeroed(size_of::<u32>())?;
    let buf_r2g = device.new_buffer_zeroed(n_rows as usize * size_of::<u32>())?;
    let buf_n_rows = device.new_buffer_from_bytes(&n_rows.to_le_bytes())?;
    let buf_table_size = device.new_buffer_from_bytes(&table_size.to_le_bytes())?;
    let buf_max_probe = device.new_buffer_from_bytes(&DEFAULT_MAX_PROBE.to_le_bytes())?;

    let mut queue = CommandQueue::new(device)?;
    queue.dispatch_1d(
        &pso,
        &[
            &buf_keys,
            &buf_slot_key,
            &buf_slot_state,
            &buf_next_gid,
            &buf_overflow,
            &buf_r2g,
            &buf_n_rows,
            &buf_table_size,
            &buf_max_probe,
        ],
        n_rows as usize,
    )?;
    queue.wait_until_complete()?;

    let of_bytes = buf_overflow.as_slice();
    if u32::from_le_bytes([of_bytes[0], of_bytes[1], of_bytes[2], of_bytes[3]]) != 0 {
        return Err(GlobalHashError::Overflow);
    }

    let ng_bytes = buf_next_gid.as_slice();
    let n_groups = u32::from_le_bytes([ng_bytes[0], ng_bytes[1], ng_bytes[2], ng_bytes[3]]);

    let r2g_bytes = buf_r2g.as_slice();
    let mut row_to_group = vec![0u32; n_rows as usize];
    for (i, v) in row_to_group.iter_mut().enumerate() {
        let b = &r2g_bytes[i * size_of::<u32>()..(i + 1) * size_of::<u32>()];
        *v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }

    let mut first_row_per_group = vec![u32::MAX; n_groups as usize];
    for (r, &g) in row_to_group.iter().enumerate() {
        if first_row_per_group[g as usize] == u32::MAX {
            first_row_per_group[g as usize] = r as u32;
        }
    }

    Ok(BuildOutput {
        row_to_group,
        first_row_per_group,
        n_groups,
    })
}
