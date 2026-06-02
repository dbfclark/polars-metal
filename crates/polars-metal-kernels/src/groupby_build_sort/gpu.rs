use crate::command::CommandQueue;
use crate::shader_lib::shared_library;
use polars_metal_buffer::MetalDevice;
use std::mem::size_of_val;

use super::SortError;

/// Threads per tile / threadgroup for the lane pass. The MSL kernel
/// assumes a 32-lane simdgroup (Apple Silicon) and 8 simdgroups per tile;
/// keep this in sync with the `SIMDS_PER_TILE * SIMD_WIDTH` constants in
/// `shaders/groupby_sort_u128_lane.metal`.
const TILE_SIZE: usize = 256;

/// Run one 8-bit-lane stable radix-sort pass over (key, row_idx) pairs.
///
/// The pass is stable: keys with the same digit at `lane` preserve their
/// relative input order. This is required for LSD radix to produce a
/// correctly sorted result after chaining 16 lane passes (Task 26).
///
/// Implementation: per-tile (256-thread) histogram, then a CPU prefix
/// scan to compute per-tile-per-digit prefix and global per-digit
/// offsets, then a stable scatter that ranks within each digit bucket
/// using SIMD match-and-rank + cross-simdgroup TGSM prefix.
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
    let n_tiles = keys.len().div_ceil(TILE_SIZE);
    let grid = n_tiles * TILE_SIZE;

    let lib = shared_library(device)?;
    let pso_hist = lib.pipeline("lane_tile_hist")?;
    let pso_scat = lib.pipeline("lane_stable_scatter")?;

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
    let buf_tile_hist = device.new_buffer_zeroed(n_tiles * 256 * std::mem::size_of::<u32>())?;
    let buf_n = device.new_buffer_from_bytes(&n_rows.to_le_bytes())?;
    let buf_lane = device.new_buffer_from_bytes(&lane.to_le_bytes())?;

    let mut queue = CommandQueue::new(device)?;
    queue.dispatch_1d_with_tg(
        &pso_hist,
        &[&buf_lo, &buf_hi, &buf_tile_hist, &buf_n, &buf_lane],
        grid,
        TILE_SIZE,
    )?;
    queue.wait_until_complete()?;

    // CPU per-digit prefix scan over the per-tile histograms.
    //
    // For each digit d, walk tiles top-to-bottom and accumulate counts:
    //   tile_prefix[t * 256 + d] = sum_{u < t} tile_hist[u * 256 + d]
    //   bucket_total[d]          = sum_t tile_hist[t * 256 + d]
    // Then exclusive-scan bucket_total to get the per-digit base offset.
    //
    // Loop order (d outer, t inner) is slightly less cache-friendly than
    // the alternative because it strides 256 entries per inner step, but
    // CPU-side cost is dominated by the GPU dispatches; correctness is
    // identical to t-outer.
    let tile_hist_bytes = buf_tile_hist.as_slice();
    let mut tile_prefix = vec![0u32; n_tiles * 256];
    let mut bucket_total = [0u32; 256];
    for (d, total) in bucket_total.iter_mut().enumerate() {
        let mut acc = 0u32;
        for t in 0..n_tiles {
            let off = t * 256 + d;
            let bytes = &tile_hist_bytes[off * 4..(off + 1) * 4];
            let h = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            tile_prefix[off] = acc;
            acc += h;
        }
        *total = acc;
    }
    let mut global_offset = [0u32; 256];
    for d in 1..256 {
        global_offset[d] = global_offset[d - 1] + bucket_total[d - 1];
    }

    let buf_tile_prefix = device.new_buffer_from_bytes(u32_bytes(&tile_prefix))?;
    let buf_global_offset = device.new_buffer_from_bytes(u32_bytes(&global_offset))?;
    let buf_lo_out = device.new_buffer_zeroed(keys.len() * std::mem::size_of::<u64>())?;
    let buf_hi_out = device.new_buffer_zeroed(keys.len() * std::mem::size_of::<u64>())?;
    let buf_idx_out = device.new_buffer_zeroed(keys.len() * std::mem::size_of::<u32>())?;

    queue.dispatch_1d_with_tg(
        &pso_scat,
        &[
            &buf_lo,
            &buf_hi,
            &buf_idx,
            &buf_lo_out,
            &buf_hi_out,
            &buf_idx_out,
            &buf_tile_prefix,
            &buf_global_offset,
            &buf_n,
            &buf_lane,
        ],
        grid,
        TILE_SIZE,
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

/// Sort a slice of `u128` keys using a stable LSD radix sort.
///
/// Chains 16 calls of [`run_radix_lane`] (lanes 0..15, least-significant
/// byte first). Because each lane pass is stable, the composition produces
/// a fully-stable sort: equal keys retain their original input order.
///
/// Returns `(sorted_keys, sorted_idx)` where `sorted_keys[i] ==
/// keys[sorted_idx[i]]` for all `i`.
pub fn sort_u128(device: &MetalDevice, keys: &[u128]) -> Result<(Vec<u128>, Vec<u32>), SortError> {
    if keys.is_empty() {
        return Ok((vec![], vec![]));
    }
    let n: u32 = keys.len().try_into().map_err(|_| SortError::RowOverflow)?;
    let mut current_keys: Vec<u128> = keys.to_vec();
    let mut current_idx: Vec<u32> = (0..n).collect();
    for lane in 0u32..16 {
        let (next_keys, next_idx) = run_radix_lane(device, &current_keys, &current_idx, lane)?;
        current_keys = next_keys;
        current_idx = next_idx;
    }
    Ok((current_keys, current_idx))
}

use crate::groupby_build_partitioned::BuildOutput;

/// A2 entry point: GPU-sort u128 keys, then derive per-row group ids
/// via a segment-boundary kernel + CPU scan. Mirrors A1's `BuildOutput`
/// so the router (Phase 6) can dispatch either build interchangeably.
pub fn sort_and_segment(device: &MetalDevice, keys: &[u128]) -> Result<BuildOutput, SortError> {
    if keys.is_empty() {
        return Ok(BuildOutput {
            row_to_group: vec![],
            first_row_per_group: vec![],
            n_groups: 0,
        });
    }
    let n_rows: u32 = keys.len().try_into().map_err(|_| SortError::RowOverflow)?;

    // 1. GPU radix sort.
    let (sorted_keys, sorted_idx) = sort_u128(device, keys)?;
    let sorted_lo: Vec<u64> = sorted_keys.iter().map(|k| *k as u64).collect();
    let sorted_hi: Vec<u64> = sorted_keys.iter().map(|k| (*k >> 64) as u64).collect();

    // SAFETY: u64 is POD; reinterpret as bytes for synchronous copy.
    let u64_bytes = |s: &[u64]| unsafe {
        std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s))
    };

    // 2. Segment-starts kernel: pack 1 bit per row into a byte buffer.
    // Pad to a multiple of 4 bytes so 32-bit atomic OR writes stay
    // in-bounds at the trailing edge.
    let starts_size_bytes = ((keys.len() + 7) >> 3).next_multiple_of(4).max(4);
    let lib = shared_library(device)?;
    let pso = lib.pipeline("segment_starts")?;
    let buf_lo = device.new_buffer_from_bytes(u64_bytes(&sorted_lo))?;
    let buf_hi = device.new_buffer_from_bytes(u64_bytes(&sorted_hi))?;
    let buf_starts = device.new_buffer_zeroed(starts_size_bytes)?;
    let buf_n = device.new_buffer_from_bytes(&n_rows.to_le_bytes())?;

    let mut queue = CommandQueue::new(device)?;
    queue.dispatch_1d(
        &pso,
        &[&buf_lo, &buf_hi, &buf_starts, &buf_n],
        n_rows as usize,
    )?;
    queue.wait_until_complete()?;

    let starts = buf_starts.as_slice();

    // 3. CPU scan: derive group ids in sorted order, then permute back
    // to original row order via sorted_idx.
    let mut row_to_group = vec![0u32; keys.len()];
    let mut first_row_per_group: Vec<u32> = Vec::new();
    let mut cur_group: u32 = 0;
    for i in 0..keys.len() {
        let bit = (starts[i >> 3] >> (i & 7)) & 1u8;
        if i > 0 && bit == 1 {
            cur_group += 1;
        }
        if i == 0 || bit == 1 {
            first_row_per_group.push(sorted_idx[i]);
        }
        row_to_group[sorted_idx[i] as usize] = cur_group;
    }
    Ok(BuildOutput {
        row_to_group,
        first_row_per_group,
        n_groups: cur_group + 1,
    })
}
