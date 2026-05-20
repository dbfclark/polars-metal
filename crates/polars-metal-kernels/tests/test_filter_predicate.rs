// crates/polars-metal-kernels/tests/test_filter_predicate.rs
//
// Correctness tests for `filter_predicate_to_u8` — the first production MSL
// kernel in M1. Validates:
//   - Bit-packed bool + validity reduce to dense u8 0/1 with no off-by-one
//     errors at the byte/bit boundary.
//   - Null rows mask to zero even when the data bit is set.
//   - False rows mask to zero even when the validity bit is set.
//   - The output is exactly 0 or 1 — never 0xFF — so MLX cumsum can sum it
//     without corruption.
//   - Property-based comparison against a CPU reference over 256 random
//     mixes of data and validity bits.
//
// All tests require Metal-capable hardware; they will skip with an
// `expect` failure on machines without a discoverable system-default
// MTLDevice.
#![allow(clippy::expect_used)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::filter::dispatch_predicate_to_u8;
use proptest::prelude::*;

fn device_and_queue() -> (MetalDevice, CommandQueue) {
    let device = MetalDevice::system_default().expect("Metal-capable hardware");
    let queue = CommandQueue::new(&device).expect("queue creation");
    (device, queue)
}

#[test]
fn all_true_no_nulls_outputs_all_ones() {
    let (device, mut queue) = device_and_queue();
    // 16 rows, all bits set, all valid.
    let data = vec![0xFFu8, 0xFFu8];
    let valid = vec![0xFFu8, 0xFFu8];
    let mut out = vec![0u8; 16];
    dispatch_predicate_to_u8(&device, &mut queue, &data, &valid, 16, &mut out)
        .expect("dispatch succeeds");
    assert_eq!(out, vec![1u8; 16]);
    // Explicitly guard against the kernel writing 0xFF instead of 1; MLX
    // cumsum would silently overflow if any byte were >1.
    assert!(out.iter().all(|&b| b <= 1), "byte must be exactly 0 or 1");
}

#[test]
fn null_rows_mask_to_zero() {
    let (device, mut queue) = device_and_queue();
    let data = vec![0xFFu8]; // all 8 "true"
    let valid = vec![0b0000_1111u8]; // only rows 0..3 valid
    let mut out = vec![0u8; 8];
    dispatch_predicate_to_u8(&device, &mut queue, &data, &valid, 8, &mut out)
        .expect("dispatch succeeds");
    assert_eq!(out, vec![1, 1, 1, 1, 0, 0, 0, 0]);
}

#[test]
fn false_rows_mask_to_zero_even_when_valid() {
    let (device, mut queue) = device_and_queue();
    let data = vec![0b0000_1111u8]; // rows 0..3 true, 4..7 false
    let valid = vec![0xFFu8]; // all valid
    let mut out = vec![0u8; 8];
    dispatch_predicate_to_u8(&device, &mut queue, &data, &valid, 8, &mut out)
        .expect("dispatch succeeds");
    assert_eq!(out, vec![1, 1, 1, 1, 0, 0, 0, 0]);
}

#[test]
fn empty_input_does_not_crash() {
    let (device, mut queue) = device_and_queue();
    let data: Vec<u8> = vec![];
    let valid: Vec<u8> = vec![];
    let mut out: Vec<u8> = vec![];
    // Empty input is allowed; the dispatcher short-circuits before
    // hitting Metal (which rejects zero-byte buffers and zero-grid
    // dispatches). Either Ok or a specific Err is fine — the important
    // contract is "do not panic".
    let _ = dispatch_predicate_to_u8(&device, &mut queue, &data, &valid, 0, &mut out);
}

#[test]
fn non_multiple_of_8_rows_ignores_padding_bits() {
    // Exercise the boundary where the predicate byte holds 8 bits but
    // only 7 are addressed by the grid. The kernel's bounds check
    // (`gid >= n_rows` -> return) must drop the high bit of the input
    // byte; the output buffer must be exactly 7 bytes long.
    let (device, mut queue) = device_and_queue();
    // Row 7 (the high bit of byte 0) is "true" but n_rows = 7, so it
    // must NOT appear in the output.
    let data = vec![0b1011_1101u8];
    let valid = vec![0xFFu8];
    let mut out = vec![0u8; 7];
    dispatch_predicate_to_u8(&device, &mut queue, &data, &valid, 7, &mut out)
        .expect("dispatch succeeds");
    // Bit 0 = 1, bit 1 = 0, bit 2 = 1, bit 3 = 1, bit 4 = 1, bit 5 = 1, bit 6 = 0.
    assert_eq!(out, vec![1, 0, 1, 1, 1, 1, 0]);
}

#[test]
fn output_length_mismatch_errors() {
    let (device, mut queue) = device_and_queue();
    let data = vec![0xFFu8];
    let valid = vec![0xFFu8];
    let mut out = vec![0u8; 4]; // mismatched — n_rows says 8
    let err = dispatch_predicate_to_u8(&device, &mut queue, &data, &valid, 8, &mut out)
        .expect_err("output length mismatch must error");
    let msg = format!("{err}");
    assert!(msg.contains("keep_flags=4"), "got {msg}");
    assert!(msg.contains("expected 8"), "got {msg}");
}

#[test]
fn predicate_too_short_errors() {
    let (device, mut queue) = device_and_queue();
    let data = vec![0xFFu8]; // 8 bits, but caller asks for 16 rows
    let valid = vec![0xFFu8, 0xFFu8];
    let mut out = vec![0u8; 16];
    let err = dispatch_predicate_to_u8(&device, &mut queue, &data, &valid, 16, &mut out)
        .expect_err("undersized predicate must error");
    let msg = format!("{err}");
    assert!(msg.contains("expected at least 2"), "got {msg}");
}

proptest! {
    #[test]
    fn matches_reference(
        n in 1usize..1024,
        data_seed in any::<u64>(),
        valid_seed in any::<u64>(),
    ) {
        let bytes = (n + 7) / 8;
        let mut data = vec![0u8; bytes];
        let mut valid = vec![0u8; bytes];
        for r in 0..n {
            // Rotate the seed by `r` to scatter bits across the input space.
            // This is intentionally crude; the goal is to cover a wide mix
            // of true/false and valid/invalid combinations, not statistical
            // uniformity.
            if (data_seed.rotate_left((r as u32) & 63) & 1) == 1 {
                data[r >> 3] |= 1u8 << (r & 7);
            }
            if (valid_seed.rotate_left((r as u32) & 63) & 1) == 1 {
                valid[r >> 3] |= 1u8 << (r & 7);
            }
        }
        let (device, mut queue) = device_and_queue();
        let mut got = vec![0u8; n];
        dispatch_predicate_to_u8(&device, &mut queue, &data, &valid, n, &mut got)
            .expect("dispatch succeeds");
        for r in 0..n {
            let d_bit = (data[r >> 3] >> (r & 7)) & 1;
            let v_bit = (valid[r >> 3] >> (r & 7)) & 1;
            let expected = d_bit & v_bit;
            prop_assert_eq!(got[r], expected, "row {} mismatch", r);
        }
    }
}
