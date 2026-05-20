// crates/polars-metal-kernels/tests/test_filter_scatter.rs
//
// Correctness tests for `filter_scatter_i64` — pass 3 of the filter
// compaction pipeline. Validates:
//   - Surviving source rows are written into the dense output at offsets
//     given by `prefix_sum[i] - 1`.
//   - Source validity bits land at the surviving rows' output positions,
//     even when 8 threads concurrently OR into the same output byte
//     (atomic correctness).
//   - Round-tripping the entire input (every row kept) produces an
//     identical column.
//   - Property-based comparison against a CPU reference over random
//     combinations of keep flags and source validity.
//
// All tests require Metal-capable hardware; they will skip with an
// `expect` failure on machines without a discoverable system-default
// MTLDevice.
#![allow(clippy::expect_used)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::filter::dispatch_scatter_i64;
use proptest::prelude::*;

fn device_and_queue() -> (MetalDevice, CommandQueue) {
    let device = MetalDevice::system_default().expect("Metal-capable hardware");
    let queue = CommandQueue::new(&device).expect("queue creation");
    (device, queue)
}

/// Round up to the kernel's required validity-buffer size: `ceil(n / 8)`
/// rounded up to 4 bytes (for the u32 atomic cast), minimum 4 bytes.
fn dst_valid_bytes(n_out: usize) -> usize {
    let raw = (n_out + 7) / 8;
    let padded = (raw + 3) & !3;
    padded.max(4)
}

/// Inclusive prefix sum of the keep flags. Mirrors what MLX cumsum
/// produces in the real pipeline.
fn prefix_sum_inclusive(keep: &[u8]) -> Vec<u32> {
    let mut prefix = Vec::with_capacity(keep.len());
    let mut acc: u32 = 0;
    for &k in keep {
        acc += k as u32;
        prefix.push(acc);
    }
    prefix
}

/// Pre-allocate the host-side `dst_data` and `dst_valid` slices. The data
/// slice is `n_out + 1` long to hold the sentinel slot.
fn dst_alloc(n_out: usize) -> (Vec<i64>, Vec<u8>) {
    (vec![0i64; n_out + 1], vec![0u8; dst_valid_bytes(n_out)])
}

/// Pure-Rust compaction reference: produce the expected `(data, valid)`
/// for a given source column + keep mask. The `valid` slice is padded to
/// match the kernel's allocation requirement.
fn cpu_compact_i64(src: &[i64], src_valid: &[u8], keep: &[u8]) -> (Vec<i64>, Vec<u8>) {
    let mut data = Vec::new();
    let mut bits = Vec::new();
    for (i, &k) in keep.iter().enumerate() {
        if k == 1 {
            data.push(src[i]);
            bits.push(((src_valid[i >> 3] >> (i & 7)) & 1) == 1);
        }
    }
    let n_out = data.len();
    let mut valid = vec![0u8; dst_valid_bytes(n_out)];
    for (i, &b) in bits.iter().enumerate() {
        if b {
            valid[i >> 3] |= 1u8 << (i & 7);
        }
    }
    (data, valid)
}

#[test]
fn alternating_keep_compacts_correctly() {
    let (device, mut queue) = device_and_queue();
    let src: Vec<i64> = (0..16).collect();
    let src_valid = vec![0xFFu8, 0xFFu8];
    let keep = vec![1u8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = *prefix.last().expect("non-empty input") as usize;
    let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
    dispatch_scatter_i64(
        &device,
        &mut queue,
        &src,
        &src_valid,
        &keep,
        &prefix,
        n_out,
        &mut dst_data,
        &mut dst_valid,
    )
    .expect("dispatch succeeds");
    let (exp_data, exp_valid) = cpu_compact_i64(&src, &src_valid, &keep);
    assert_eq!(&dst_data[..n_out], &exp_data[..]);
    for r in 0..n_out {
        let got = (dst_valid[r >> 3] >> (r & 7)) & 1;
        let exp = (exp_valid[r >> 3] >> (r & 7)) & 1;
        assert_eq!(got, exp, "validity at row {r}");
    }
}

#[test]
fn all_keep_round_trips_input() {
    let (device, mut queue) = device_and_queue();
    let src: Vec<i64> = (0..32).collect();
    let src_valid = vec![0xFFu8; 4];
    let keep = vec![1u8; 32];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = 32;
    let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
    dispatch_scatter_i64(
        &device,
        &mut queue,
        &src,
        &src_valid,
        &keep,
        &prefix,
        n_out,
        &mut dst_data,
        &mut dst_valid,
    )
    .expect("dispatch succeeds");
    assert_eq!(&dst_data[..n_out], &src[..]);
    // All 32 inputs are valid; every output bit should be set.
    for r in 0..32 {
        let bit = (dst_valid[r >> 3] >> (r & 7)) & 1;
        assert_eq!(bit, 1, "row {r} should be valid");
    }
}

#[test]
fn null_inputs_produce_null_outputs() {
    let (device, mut queue) = device_and_queue();
    // 8 source rows, all kept. Source validity: rows 2 and 5 are null
    // (bits 2 and 5 cleared in the byte).
    let src: Vec<i64> = vec![10, 20, 30, 40, 50, 60, 70, 80];
    let src_valid = vec![0b1101_1011u8];
    let keep = vec![1u8; 8];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = 8;
    let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
    dispatch_scatter_i64(
        &device,
        &mut queue,
        &src,
        &src_valid,
        &keep,
        &prefix,
        n_out,
        &mut dst_data,
        &mut dst_valid,
    )
    .expect("dispatch succeeds");
    // Output validity mirrors source validity: same 8 rows, same null
    // positions.
    assert_eq!(dst_valid[0], 0b1101_1011u8);
    // Data values themselves are written unconditionally for kept rows
    // (Polars convention: null payloads are unspecified but the slot is
    // still occupied).
    assert_eq!(&dst_data[..n_out], &src[..]);
}

#[test]
fn forces_multi_thread_same_byte_writes() {
    // 8 source rows, all kept, all valid. All 8 output validity bits land
    // in one byte (and the same atomic u32 word). The atomic OR is the
    // only thing that prevents bits being lost — a non-atomic
    // read-modify-write would race here.
    let (device, mut queue) = device_and_queue();
    let src: Vec<i64> = (0..8).collect();
    let src_valid = vec![0xFFu8];
    let keep = vec![1u8; 8];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = 8;
    let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
    dispatch_scatter_i64(
        &device,
        &mut queue,
        &src,
        &src_valid,
        &keep,
        &prefix,
        n_out,
        &mut dst_data,
        &mut dst_valid,
    )
    .expect("dispatch succeeds");
    assert_eq!(dst_valid[0], 0xFFu8, "all 8 bits must be set");
}

#[test]
fn nothing_kept_leaves_outputs_empty() {
    // n_rows > 0 but keep is all-zero → n_out == 0. The kernel runs
    // (every thread short-circuits) and writes nothing; the validity
    // buffer stays zero and the only data slot (the sentinel) stays
    // zero.
    let (device, mut queue) = device_and_queue();
    let src: Vec<i64> = (0..8).collect();
    let src_valid = vec![0xFFu8];
    let keep = vec![0u8; 8];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = 0;
    let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
    dispatch_scatter_i64(
        &device,
        &mut queue,
        &src,
        &src_valid,
        &keep,
        &prefix,
        n_out,
        &mut dst_data,
        &mut dst_valid,
    )
    .expect("dispatch succeeds");
    assert!(dst_valid.iter().all(|&b| b == 0));
}

#[test]
fn dst_data_too_small_errors() {
    // Caller forgot the +1 sentinel slot.
    let (device, mut queue) = device_and_queue();
    let src: Vec<i64> = vec![1, 2, 3, 4];
    let src_valid = vec![0xFFu8];
    let keep = vec![1u8; 4];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = 4;
    let mut dst_data = vec![0i64; n_out]; // missing +1 sentinel slot
    let mut dst_valid = vec![0u8; dst_valid_bytes(n_out)];
    let err = dispatch_scatter_i64(
        &device,
        &mut queue,
        &src,
        &src_valid,
        &keep,
        &prefix,
        n_out,
        &mut dst_data,
        &mut dst_valid,
    )
    .expect_err("missing sentinel slot must error");
    let msg = format!("{err}");
    assert!(msg.contains("dst_data too small"), "got {msg}");
}

#[test]
fn dst_valid_too_small_errors() {
    // Caller passed a 1-byte validity slice but we require 4-byte
    // alignment.
    let (device, mut queue) = device_and_queue();
    let src: Vec<i64> = vec![1, 2, 3, 4];
    let src_valid = vec![0xFFu8];
    let keep = vec![1u8; 4];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = 4;
    let mut dst_data = vec![0i64; n_out + 1];
    let mut dst_valid = vec![0u8; 1]; // too small
    let err = dispatch_scatter_i64(
        &device,
        &mut queue,
        &src,
        &src_valid,
        &keep,
        &prefix,
        n_out,
        &mut dst_data,
        &mut dst_valid,
    )
    .expect_err("undersized dst_valid must error");
    let msg = format!("{err}");
    assert!(msg.contains("dst_valid too small"), "got {msg}");
}

proptest! {
    #[test]
    fn matches_cpu_reference(
        n in 8usize..256,
        data_seed in any::<u64>(),
        valid_seed in any::<u64>(),
        keep_seed in any::<u64>(),
    ) {
        // Build a deterministic-ish mix of src values, src validity, and
        // keep flags. The exact bit layout matters less than covering a
        // wide range of (keep, valid) combinations.
        let src: Vec<i64> = (0..n as i64)
            .map(|i| i.wrapping_mul(data_seed as i64))
            .collect();
        let mut src_valid = vec![0u8; (n + 7) / 8];
        let mut keep = vec![0u8; n];
        for r in 0..n {
            if (valid_seed.rotate_left((r as u32) & 63) & 1) == 1 {
                src_valid[r >> 3] |= 1u8 << (r & 7);
            }
            if (keep_seed.rotate_left((r as u32) & 63) & 1) == 1 {
                keep[r] = 1;
            }
        }
        let prefix = prefix_sum_inclusive(&keep);
        let n_out = *prefix.last().expect("non-empty input") as usize;
        if n_out == 0 {
            // Skip the empty-output case: dispatch is a no-op and the
            // CPU reference is trivially equal.
            return Ok(());
        }
        let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
        let (device, mut queue) = device_and_queue();
        dispatch_scatter_i64(
            &device, &mut queue,
            &src, &src_valid, &keep, &prefix,
            n_out, &mut dst_data, &mut dst_valid,
        ).expect("dispatch succeeds");

        let (exp_data, exp_valid) = cpu_compact_i64(&src, &src_valid, &keep);
        prop_assert_eq!(&dst_data[..n_out], &exp_data[..]);
        for r in 0..n_out {
            let got = (dst_valid[r >> 3] >> (r & 7)) & 1;
            let exp = (exp_valid[r >> 3] >> (r & 7)) & 1;
            prop_assert_eq!(got, exp, "row {} validity", r);
        }
    }
}
