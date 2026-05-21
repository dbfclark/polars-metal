// crates/polars-metal-kernels/tests/test_filter_scatter_bool.rs
//
// Correctness tests for `filter_scatter_bool` — pass 3 of the filter
// compaction pipeline, bool variant. Bool is the only scatter variant
// where BOTH the data buffer and the validity buffer are bit-packed
// (one bit per row), so both buffers have the multi-thread-same-byte
// write race that the i64/f64 variants only have on validity. The
// kernel uses an atomic OR for both.
//
// There is no sentinel slot for the data buffer (every bit value is a
// legitimate bool), so the dispatcher instead pre-verifies the
// prefix-sum invariant `prefix_sum[n_rows - 1] == n_out` as the
// safety net.
//
// All tests require Metal-capable hardware; they will skip with an
// `expect` failure on machines without a discoverable system-default
// MTLDevice.
#![allow(clippy::expect_used)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::filter::dispatch_scatter_bool;
use proptest::prelude::*;

fn device_and_queue() -> (MetalDevice, CommandQueue) {
    let device = MetalDevice::system_default().expect("Metal-capable hardware");
    let queue = CommandQueue::new(&device).expect("queue creation");
    (device, queue)
}

/// Required allocation in bytes for a bit-packed bool buffer holding
/// `n_out` rows: `ceil(n_out / 8)` rounded up to a 4-byte multiple,
/// minimum 4 bytes (the kernel binds these buffers as
/// `device atomic_uint*` and requires u32 alignment).
fn dst_bytes(n_out: usize) -> usize {
    let raw = (n_out + 7) / 8;
    let padded = (raw + 3) & !3;
    padded.max(4)
}

fn prefix_sum_inclusive(keep: &[u8]) -> Vec<u32> {
    let mut prefix = Vec::with_capacity(keep.len());
    let mut acc: u32 = 0;
    for &k in keep {
        acc += k as u32;
        prefix.push(acc);
    }
    prefix
}

/// Pre-allocate (`dst_data`, `dst_valid`) for `n_out` output rows.
fn dst_alloc(n_out: usize) -> (Vec<u8>, Vec<u8>) {
    let b = dst_bytes(n_out);
    (vec![0u8; b], vec![0u8; b])
}

/// CPU reference: produce the expected bit-packed `(data, valid)` for
/// a given source column + keep mask. Mirrors the kernel's behaviour
/// exactly, including the convention that the data bit is taken from
/// `src_data` regardless of the row's validity (same as i64/f64
/// variants: null payloads land in the output slot unchanged; only the
/// validity bit signals null-ness).
fn cpu_compact_bool(src_data: &[u8], src_valid: &[u8], keep: &[u8]) -> (Vec<u8>, Vec<u8>, usize) {
    let n = keep.len();
    let mut data_bits = Vec::new();
    let mut valid_bits = Vec::new();
    for i in 0..n {
        if keep[i] == 1 {
            data_bits.push(((src_data[i >> 3] >> (i & 7)) & 1) == 1);
            valid_bits.push(((src_valid[i >> 3] >> (i & 7)) & 1) == 1);
        }
    }
    let n_out = data_bits.len();
    let bytes = dst_bytes(n_out);
    let mut data = vec![0u8; bytes];
    let mut valid = vec![0u8; bytes];
    for (i, &b) in data_bits.iter().enumerate() {
        if b {
            data[i >> 3] |= 1u8 << (i & 7);
        }
    }
    for (i, &b) in valid_bits.iter().enumerate() {
        if b {
            valid[i >> 3] |= 1u8 << (i & 7);
        }
    }
    (data, valid, n_out)
}

#[test]
fn forces_multi_thread_same_byte_writes_for_data_and_validity() {
    // 8 rows all kept, all true, all valid. 8 threads race ONE byte in
    // the DATA buffer AND one byte in the VALIDITY buffer. The atomic
    // OR on BOTH buffers must hold or bits will be lost.
    //
    // This is the bool-specific stress test: the i64/f64 variants only
    // race the validity byte (their data writes are 8-byte-unique).
    let (device, mut queue) = device_and_queue();
    let src_data = vec![0xFFu8]; // 8 true rows
    let src_valid = vec![0xFFu8];
    let keep = vec![1u8; 8];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = 8;
    let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
    dispatch_scatter_bool(
        &device,
        &mut queue,
        &src_data,
        &src_valid,
        &keep,
        &prefix,
        8,
        n_out,
        &mut dst_data,
        &mut dst_valid,
    )
    .expect("dispatch succeeds");
    assert_eq!(dst_data[0], 0xFFu8, "all 8 data bits must be set");
    assert_eq!(dst_valid[0], 0xFFu8, "all 8 validity bits must be set");
}

#[test]
fn alternating_data_with_all_kept() {
    // Rows 1, 3, 5, 7 are true; rows 0, 2, 4, 6 are false. All kept,
    // all valid. The output data byte must mirror the input bit-by-bit.
    let (device, mut queue) = device_and_queue();
    let src_data = vec![0b1010_1010u8];
    let src_valid = vec![0xFFu8];
    let keep = vec![1u8; 8];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = 8;
    let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
    dispatch_scatter_bool(
        &device,
        &mut queue,
        &src_data,
        &src_valid,
        &keep,
        &prefix,
        8,
        n_out,
        &mut dst_data,
        &mut dst_valid,
    )
    .expect("dispatch succeeds");
    assert_eq!(dst_data[0], 0b1010_1010u8);
    assert_eq!(dst_valid[0], 0xFFu8);
}

#[test]
fn null_propagates_through_scatter() {
    // 8 rows, all true, all kept; validity has rows 2 and 5 cleared
    // (null). Validity must propagate exactly. The data bits at null
    // rows still land in the output (Polars convention: the payload
    // of a null slot is unspecified but the slot is present); since
    // the source data is all-1s, we expect the output data byte to be
    // 0xFF — but we only assert on the bits that have defined
    // semantics: validity bits, and data bits where validity is 1.
    let (device, mut queue) = device_and_queue();
    let src_data = vec![0xFFu8];
    let src_valid = vec![0b1101_1011u8]; // rows 2 and 5 null
    let keep = vec![1u8; 8];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = 8;
    let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
    dispatch_scatter_bool(
        &device,
        &mut queue,
        &src_data,
        &src_valid,
        &keep,
        &prefix,
        8,
        n_out,
        &mut dst_data,
        &mut dst_valid,
    )
    .expect("dispatch succeeds");
    // Validity bits land exactly where the source has them.
    assert_eq!(dst_valid[0], 0b1101_1011u8);
    // Data bits at valid rows are the source value (all-1 here).
    for r in 0..n_out {
        let v = (dst_valid[0] >> r) & 1;
        if v == 1 {
            let d = (dst_data[0] >> r) & 1;
            assert_eq!(d, 1, "valid row {r} should carry the source data bit");
        }
    }
}

#[test]
fn alternating_keep_compacts_bool() {
    // 16 source rows: first byte 0xFF (all true), second byte 0x00
    // (all false). Keep every even row → 8 surviving rows. The first
    // 4 land in output bits 0..3 (true), the next 4 in bits 4..7
    // (false). Expected output data byte: 0b0000_1111.
    let (device, mut queue) = device_and_queue();
    let src_data = vec![0xFFu8, 0x00u8];
    let src_valid = vec![0xFFu8, 0xFFu8];
    let keep = vec![1u8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = *prefix.last().expect("non-empty input") as usize;
    assert_eq!(n_out, 8);
    let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
    dispatch_scatter_bool(
        &device,
        &mut queue,
        &src_data,
        &src_valid,
        &keep,
        &prefix,
        16,
        n_out,
        &mut dst_data,
        &mut dst_valid,
    )
    .expect("dispatch succeeds");
    let (exp_data, exp_valid, _) = cpu_compact_bool(&src_data, &src_valid, &keep);
    assert_eq!(&dst_data[..exp_data.len()], &exp_data[..]);
    assert_eq!(&dst_valid[..exp_valid.len()], &exp_valid[..]);
    assert_eq!(
        dst_data[0], 0b0000_1111u8,
        "evens of first byte then evens of second byte"
    );
}

#[test]
fn all_kept_round_trips_input() {
    // 32 rows of mixed true/false, all kept, all valid. Output must be
    // bit-identical to the input data.
    let (device, mut queue) = device_and_queue();
    let src_data = vec![0xA3u8, 0x5Cu8, 0xF0u8, 0x0Fu8];
    let src_valid = vec![0xFFu8; 4];
    let n_rows = 32;
    let keep = vec![1u8; n_rows];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = n_rows;
    let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
    dispatch_scatter_bool(
        &device,
        &mut queue,
        &src_data,
        &src_valid,
        &keep,
        &prefix,
        n_rows,
        n_out,
        &mut dst_data,
        &mut dst_valid,
    )
    .expect("dispatch succeeds");
    assert_eq!(&dst_data[..4], &src_data[..]);
    assert_eq!(&dst_valid[..4], &src_valid[..]);
}

#[test]
fn nothing_kept_leaves_outputs_empty() {
    // n_rows > 0, keep all zero → n_out == 0. Every thread
    // short-circuits; no writes happen. Both output buffers stay at
    // their zero-init state.
    let (device, mut queue) = device_and_queue();
    let src_data = vec![0xFFu8];
    let src_valid = vec![0xFFu8];
    let keep = vec![0u8; 8];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = 0;
    let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
    dispatch_scatter_bool(
        &device,
        &mut queue,
        &src_data,
        &src_valid,
        &keep,
        &prefix,
        8,
        n_out,
        &mut dst_data,
        &mut dst_valid,
    )
    .expect("dispatch succeeds");
    assert!(dst_data.iter().all(|&b| b == 0));
    assert!(dst_valid.iter().all(|&b| b == 0));
}

#[test]
fn dst_data_too_small_errors() {
    // Caller passed a 1-byte data slice; kernel requires 4-byte
    // alignment (for the u32 atomic cast).
    let (device, mut queue) = device_and_queue();
    let src_data = vec![0xFFu8];
    let src_valid = vec![0xFFu8];
    let keep = vec![1u8; 4];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = 4;
    let mut dst_data = vec![0u8; 1]; // too small
    let mut dst_valid = vec![0u8; dst_bytes(n_out)];
    let err = dispatch_scatter_bool(
        &device,
        &mut queue,
        &src_data,
        &src_valid,
        &keep,
        &prefix,
        4,
        n_out,
        &mut dst_data,
        &mut dst_valid,
    )
    .expect_err("undersized dst_data must error");
    let msg = format!("{err}");
    assert!(
        msg.contains("dst_data (bool, bit-packed) too small"),
        "got {msg}"
    );
}

#[test]
fn dst_valid_too_small_errors() {
    let (device, mut queue) = device_and_queue();
    let src_data = vec![0xFFu8];
    let src_valid = vec![0xFFu8];
    let keep = vec![1u8; 4];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = 4;
    let mut dst_data = vec![0u8; dst_bytes(n_out)];
    let mut dst_valid = vec![0u8; 1]; // too small
    let err = dispatch_scatter_bool(
        &device,
        &mut queue,
        &src_data,
        &src_valid,
        &keep,
        &prefix,
        4,
        n_out,
        &mut dst_data,
        &mut dst_valid,
    )
    .expect_err("undersized dst_valid must error");
    let msg = format!("{err}");
    assert!(msg.contains("dst_valid too small"), "got {msg}");
}

#[test]
fn prefix_sum_mismatch_errors() {
    // Caller's `n_out` disagrees with the last element of the
    // prefix sum. The bool scatter has no data-buffer sentinel, so the
    // host-side invariant check is the only safety net; it must
    // refuse to dispatch.
    let (device, mut queue) = device_and_queue();
    let src_data = vec![0xFFu8];
    let src_valid = vec![0xFFu8];
    let keep = vec![1u8; 4];
    let prefix = prefix_sum_inclusive(&keep); // last element = 4
    let bad_n_out = 5; // caller claims 5 survivors but prefix says 4
    let mut dst_data = vec![0u8; dst_bytes(bad_n_out)];
    let mut dst_valid = vec![0u8; dst_bytes(bad_n_out)];
    let err = dispatch_scatter_bool(
        &device,
        &mut queue,
        &src_data,
        &src_valid,
        &keep,
        &prefix,
        4,
        bad_n_out,
        &mut dst_data,
        &mut dst_valid,
    )
    .expect_err("prefix-sum mismatch must error");
    let msg = format!("{err}");
    assert!(msg.contains("prefix-sum invariant violated"), "got {msg}");
}

#[test]
fn n_rows_zero_is_no_op() {
    // Zero rows: no kernel runs, no writes happen. The caller's
    // buffers stay at their initial state.
    let (device, mut queue) = device_and_queue();
    let src_data: Vec<u8> = Vec::new();
    let src_valid: Vec<u8> = Vec::new();
    let keep: Vec<u8> = Vec::new();
    let prefix: Vec<u32> = Vec::new();
    let n_out = 0;
    let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
    dispatch_scatter_bool(
        &device,
        &mut queue,
        &src_data,
        &src_valid,
        &keep,
        &prefix,
        0,
        n_out,
        &mut dst_data,
        &mut dst_valid,
    )
    .expect("zero rows is a no-op");
    assert!(dst_data.iter().all(|&b| b == 0));
    assert!(dst_valid.iter().all(|&b| b == 0));
}

#[test]
fn unaligned_row_count_handles_partial_byte() {
    // 13 rows in 2 bytes (3 padding bits in the second byte). All
    // kept, mixed data. Tests the explicit `n_rows` parameter — the
    // kernel reads exactly 13 rows, not 16.
    let (device, mut queue) = device_and_queue();
    let n_rows = 13;
    let src_data = vec![0b1010_1010u8, 0b0000_0111u8]; // rows 8,9,10 true; row 11,12 false (padding bits 13-15 ignored)
    let src_valid = vec![0xFFu8, 0b0001_1111u8]; // all 13 rows valid; padding bits zero
    let keep = vec![1u8; n_rows];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = n_rows;
    let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
    dispatch_scatter_bool(
        &device,
        &mut queue,
        &src_data,
        &src_valid,
        &keep,
        &prefix,
        n_rows,
        n_out,
        &mut dst_data,
        &mut dst_valid,
    )
    .expect("dispatch succeeds");
    let (exp_data, exp_valid, _) = cpu_compact_bool(&src_data, &src_valid, &keep);
    // Compare the bits we asserted, ignoring any padding bits past
    // n_out in either buffer.
    for r in 0..n_out {
        let d_got = (dst_data[r >> 3] >> (r & 7)) & 1;
        let d_exp = (exp_data[r >> 3] >> (r & 7)) & 1;
        assert_eq!(d_got, d_exp, "data bit at row {r}");
        let v_got = (dst_valid[r >> 3] >> (r & 7)) & 1;
        let v_exp = (exp_valid[r >> 3] >> (r & 7)) & 1;
        assert_eq!(v_got, v_exp, "validity bit at row {r}");
    }
}

proptest! {
    #[test]
    fn matches_cpu_reference_bool(n in 8usize..256, seed in any::<u64>()) {
        // Random src_data, src_valid, and keep; assert the kernel's
        // output matches the CPU reference bit-by-bit on the surviving
        // rows. Per Polars convention, data bits at NULL output rows
        // are unspecified — we only assert data bits where validity
        // is set.
        let bytes = (n + 7) / 8;
        let mut src_data = vec![0u8; bytes];
        let mut src_valid = vec![0u8; bytes];
        let mut keep = vec![0u8; n];
        for r in 0..n {
            if (seed.rotate_left((r as u32) & 63) & 1) == 1 {
                src_data[r >> 3] |= 1u8 << (r & 7);
            }
            if (seed.rotate_left(((r as u32) ^ 13) & 63) & 1) == 1 {
                src_valid[r >> 3] |= 1u8 << (r & 7);
            }
            if (seed.rotate_left(((r as u32).wrapping_mul(7)) & 63) & 1) == 1 {
                keep[r] = 1;
            }
        }
        let prefix = prefix_sum_inclusive(&keep);
        let n_out = *prefix.last().expect("non-empty input") as usize;
        if n_out == 0 {
            return Ok(());
        }
        let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
        let (device, mut queue) = device_and_queue();
        dispatch_scatter_bool(
            &device, &mut queue,
            &src_data, &src_valid, &keep, &prefix,
            n, n_out, &mut dst_data, &mut dst_valid,
        ).expect("dispatch succeeds");

        let (exp_data, exp_valid, exp_n_out) = cpu_compact_bool(&src_data, &src_valid, &keep);
        prop_assert_eq!(exp_n_out, n_out, "CPU and host agree on survivor count");
        for r in 0..n_out {
            let v_got = (dst_valid[r >> 3] >> (r & 7)) & 1;
            let v_exp = (exp_valid[r >> 3] >> (r & 7)) & 1;
            prop_assert_eq!(v_got, v_exp, "validity row {}", r);
            // For valid rows, data must match. For null rows, data is
            // unspecified per Polars convention (mirrors the i64/f64
            // tests, which don't assert on data at null rows either).
            if v_exp == 1 {
                let d_got = (dst_data[r >> 3] >> (r & 7)) & 1;
                let d_exp = (exp_data[r >> 3] >> (r & 7)) & 1;
                prop_assert_eq!(d_got, d_exp, "data row {}", r);
            }
        }
    }
}
