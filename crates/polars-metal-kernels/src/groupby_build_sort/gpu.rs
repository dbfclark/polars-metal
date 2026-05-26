use crate::command::CommandQueue;
use crate::shader_lib::shared_library;
use polars_metal_buffer::MetalDevice;
use std::mem::size_of_val;

use super::SortError;

pub fn run_radix_lane(
    device: &MetalDevice,
    keys: &[u128],
    row_idx_in: &[u32],
    lane: u32,
) -> Result<(Vec<u128>, Vec<u32>), SortError> {
    assert_eq!(
        keys.len(),
        row_idx_in.len(),
        "keys and idx must have same length"
    );
    if keys.is_empty() {
        return Ok((vec![], vec![]));
    }
    let keys_lo: Vec<u64> = keys.iter().map(|k| *k as u64).collect();
    let keys_hi: Vec<u64> = keys.iter().map(|k| (*k >> 64) as u64).collect();
    let n_rows: u32 = keys.len().try_into().map_err(|_| SortError::RowOverflow)?;

    let lib = shared_library(device)?;
    let pso_hist = lib.pipeline("lane_histogram")?;
    let pso_scat = lib.pipeline("lane_scatter")?;

    // SAFETY: u64/u32 are POD; reinterpret as bytes for the synchronous
    // copy inside `new_buffer_from_bytes`. Slices remain valid for the
    // duration of the call.
    let u64_bytes =
        |s: &[u64]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, size_of_val(s)) };
    let u32_bytes =
        |s: &[u32]| unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, size_of_val(s)) };

    let buf_lo = device.new_buffer_from_bytes(u64_bytes(&keys_lo))?;
    let buf_hi = device.new_buffer_from_bytes(u64_bytes(&keys_hi))?;
    let buf_idx = device.new_buffer_from_bytes(u32_bytes(row_idx_in))?;
    let buf_bins = device.new_buffer_zeroed(256 * std::mem::size_of::<u32>())?;
    let buf_n = device.new_buffer_from_bytes(&n_rows.to_le_bytes())?;
    let buf_lane = device.new_buffer_from_bytes(&lane.to_le_bytes())?;

    let mut queue = CommandQueue::new(device)?;
    queue.dispatch_1d(
        &pso_hist,
        &[&buf_lo, &buf_hi, &buf_bins, &buf_n, &buf_lane],
        n_rows as usize,
    )?;
    queue.wait_until_complete()?;

    // Read bins; CPU exclusive scan.
    let bins_bytes = buf_bins.as_slice();
    let mut bins = [0u32; 256];
    for i in 0..256 {
        let b = &bins_bytes[i * 4..(i + 1) * 4];
        bins[i] = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
    let mut offsets = [0u32; 256];
    for i in 1..256 {
        offsets[i] = offsets[i - 1] + bins[i - 1];
    }

    let buf_cursors = device.new_buffer_from_bytes(u32_bytes(&offsets))?;
    let buf_lo_out = device.new_buffer_zeroed(keys.len() * std::mem::size_of::<u64>())?;
    let buf_hi_out = device.new_buffer_zeroed(keys.len() * std::mem::size_of::<u64>())?;
    let buf_idx_out = device.new_buffer_zeroed(keys.len() * std::mem::size_of::<u32>())?;

    queue.dispatch_1d(
        &pso_scat,
        &[
            &buf_lo,
            &buf_hi,
            &buf_idx,
            &buf_lo_out,
            &buf_hi_out,
            &buf_idx_out,
            &buf_cursors,
            &buf_n,
            &buf_lane,
        ],
        n_rows as usize,
    )?;
    queue.wait_until_complete()?;

    let lo_b = buf_lo_out.as_slice();
    let hi_b = buf_hi_out.as_slice();
    let idx_b = buf_idx_out.as_slice();
    let mut sorted_keys = Vec::with_capacity(keys.len());
    let mut sorted_idx = Vec::with_capacity(keys.len());
    for i in 0..keys.len() {
        let lo_slice = &lo_b[i * 8..(i + 1) * 8];
        let hi_slice = &hi_b[i * 8..(i + 1) * 8];
        let id_slice = &idx_b[i * 4..(i + 1) * 4];
        let lo = u64::from_le_bytes([
            lo_slice[0],
            lo_slice[1],
            lo_slice[2],
            lo_slice[3],
            lo_slice[4],
            lo_slice[5],
            lo_slice[6],
            lo_slice[7],
        ]);
        let hi = u64::from_le_bytes([
            hi_slice[0],
            hi_slice[1],
            hi_slice[2],
            hi_slice[3],
            hi_slice[4],
            hi_slice[5],
            hi_slice[6],
            hi_slice[7],
        ]);
        let id = u32::from_le_bytes([id_slice[0], id_slice[1], id_slice[2], id_slice[3]]);
        sorted_keys.push(((hi as u128) << 64) | (lo as u128));
        sorted_idx.push(id);
    }
    Ok((sorted_keys, sorted_idx))
}
