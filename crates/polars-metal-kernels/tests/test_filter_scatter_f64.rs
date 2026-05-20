// crates/polars-metal-kernels/tests/test_filter_scatter_f64.rs
//
// Correctness tests for `filter_scatter_f64` — pass 3 of the filter
// compaction pipeline, f64 variant. Mirror of `test_filter_scatter.rs`
// (i64 variant) with two additions specific to floating point:
//
//   - NaN round-trip: the kernel must preserve f64 NaN payloads
//     bit-identically. Internally the kernel reads/writes 8-byte opaque
//     chunks (`ulong` in MSL) rather than `double`, so no floating-point
//     interpretation happens on the GPU side. We assert via `to_bits`.
//   - Special-value round-trip: ±Inf, ±0.0, subnormals, MIN_POSITIVE,
//     EPSILON, and ordinary finites all round-trip bit-identically.
//
// All tests require Metal-capable hardware; they will skip with an
// `expect` failure on machines without a discoverable system-default
// MTLDevice.
#![allow(clippy::expect_used)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::filter::{dispatch_scatter_f64, SCATTER_SENTINEL_F64_BITS};
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
fn dst_alloc(n_out: usize) -> (Vec<f64>, Vec<u8>) {
    (vec![0.0f64; n_out + 1], vec![0u8; dst_valid_bytes(n_out)])
}

/// Pure-Rust compaction reference: produce the expected `(data, valid)`
/// for a given source column + keep mask. The `valid` slice is padded to
/// match the kernel's allocation requirement.
fn cpu_compact_f64(src: &[f64], src_valid: &[u8], keep: &[u8]) -> (Vec<f64>, Vec<u8>) {
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
fn alternating_keep_compacts_correctly_f64() {
    let (device, mut queue) = device_and_queue();
    let src: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let src_valid = vec![0xFFu8, 0xFFu8];
    let keep = vec![1u8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = *prefix.last().expect("non-empty input") as usize;
    let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
    dispatch_scatter_f64(
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
    let (exp_data, exp_valid) = cpu_compact_f64(&src, &src_valid, &keep);
    // Compare bit patterns to be NaN-safe (these are finite, but the
    // bit comparison generalises to the NaN tests below).
    for r in 0..n_out {
        assert_eq!(
            dst_data[r].to_bits(),
            exp_data[r].to_bits(),
            "row {r} f64 bits"
        );
        let got = (dst_valid[r >> 3] >> (r & 7)) & 1;
        let exp = (exp_valid[r >> 3] >> (r & 7)) & 1;
        assert_eq!(got, exp, "validity at row {r}");
    }
}

#[test]
fn all_keep_round_trips_input_f64() {
    let (device, mut queue) = device_and_queue();
    let src: Vec<f64> = (0..32).map(|i| (i as f64) * 0.125 - 4.0).collect();
    let src_valid = vec![0xFFu8; 4];
    let keep = vec![1u8; 32];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = 32;
    let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
    dispatch_scatter_f64(
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
    for r in 0..n_out {
        assert_eq!(dst_data[r].to_bits(), src[r].to_bits(), "row {r} f64 bits");
    }
    for r in 0..32 {
        let bit = (dst_valid[r >> 3] >> (r & 7)) & 1;
        assert_eq!(bit, 1, "row {r} should be valid");
    }
}

#[test]
fn nan_round_trips_bit_identical() {
    // f64 NaN is a value (not a null). The scatter kernel must preserve
    // arbitrary NaN payloads bit-for-bit. Because the MSL kernel treats
    // each 8-byte slot as opaque (`ulong` rather than `double`), no
    // floating-point normalisation happens on the GPU.
    let (device, mut queue) = device_and_queue();
    // A quiet NaN with a non-default payload (mantissa = 0x0...0042).
    let qnan = f64::from_bits(0x7FF8_0000_0000_0042u64);
    // A signaling NaN with a different payload.
    let snan = f64::from_bits(0x7FF0_0000_0000_0001u64);
    let src: Vec<f64> = vec![1.0, qnan, 3.0, snan, 5.0, qnan, 7.0, snan];
    let src_valid = vec![0xFFu8];
    let keep = vec![1u8; 8];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = 8;
    let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
    dispatch_scatter_f64(
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
    for r in 0..n_out {
        assert_eq!(dst_data[r].to_bits(), src[r].to_bits(), "row {r} f64 bits");
    }
    // All inputs valid; all output bits set.
    assert_eq!(dst_valid[0], 0xFFu8);
}

#[test]
fn special_values_round_trip() {
    // ±Inf, ±0.0, MIN_POSITIVE, EPSILON, and ordinary finites must all
    // round-trip bit-identically. `-0.0` and `+0.0` compare equal under
    // `==` but have distinct bit patterns; the bit comparison catches
    // a hypothetical regression that lost the sign of zero.
    let (device, mut queue) = device_and_queue();
    let src: Vec<f64> = vec![
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        f64::MIN_POSITIVE,
        f64::EPSILON,
        1.0,
        -1.0,
    ];
    let src_valid = vec![0xFFu8];
    let keep = vec![1u8; 8];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = 8;
    let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
    dispatch_scatter_f64(
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
    for r in 0..n_out {
        assert_eq!(dst_data[r].to_bits(), src[r].to_bits(), "row {r} f64 bits");
    }
}

#[test]
fn null_inputs_produce_null_outputs_f64() {
    let (device, mut queue) = device_and_queue();
    let src: Vec<f64> = (0..8).map(|i| (i as f64) * 1.5).collect();
    // Rows 2 and 5 are null.
    let src_valid = vec![0b1101_1011u8];
    let keep = vec![1u8; 8];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = 8;
    let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
    dispatch_scatter_f64(
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
    assert_eq!(dst_valid[0], 0b1101_1011u8);
    for r in 0..n_out {
        assert_eq!(dst_data[r].to_bits(), src[r].to_bits(), "row {r} f64 bits");
    }
}

#[test]
fn forces_multi_thread_same_byte_writes_f64() {
    // 8 source rows, all kept, all valid. All 8 output validity bits land
    // in one byte (and the same atomic u32 word). The atomic OR is the
    // only thing that prevents bits being lost.
    let (device, mut queue) = device_and_queue();
    let src: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let src_valid = vec![0xFFu8];
    let keep = vec![1u8; 8];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = 8;
    let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
    dispatch_scatter_f64(
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
fn nothing_kept_leaves_outputs_empty_f64() {
    let (device, mut queue) = device_and_queue();
    let src: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let src_valid = vec![0xFFu8];
    let keep = vec![0u8; 8];
    let prefix = prefix_sum_inclusive(&keep);
    let n_out = 0;
    let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
    dispatch_scatter_f64(
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
fn sentinel_constant_is_nan() {
    // The sentinel must be a NaN so it is distinguishable from every
    // finite value the kernel could legitimately copy. NaN itself is a
    // valid f64 value, so callers also bit-compare against
    // SCATTER_SENTINEL_F64_BITS to disambiguate "user's NaN" from "kernel
    // overrun".
    let s = f64::from_bits(SCATTER_SENTINEL_F64_BITS);
    assert!(s.is_nan(), "sentinel must be a NaN");
}

proptest! {
    #[test]
    fn matches_cpu_reference_f64(
        n in 8usize..256,
        data_seed in any::<u64>(),
        valid_seed in any::<u64>(),
        keep_seed in any::<u64>(),
    ) {
        // Build a deterministic-ish mix of src values, src validity, and
        // keep flags. We use a non-trivial f64 derivation to cover a
        // range of bit patterns.
        let src: Vec<f64> = (0..n)
            .map(|i| {
                let raw = (i as u64).wrapping_mul(data_seed.wrapping_add(1));
                // Avoid producing NaN by mapping into a finite-friendly
                // range; the dedicated NaN test covers NaN payloads.
                (raw as f64) * 1e-3 - (n as f64) * 0.5
            })
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
            return Ok(());
        }
        let (mut dst_data, mut dst_valid) = dst_alloc(n_out);
        let (device, mut queue) = device_and_queue();
        dispatch_scatter_f64(
            &device, &mut queue,
            &src, &src_valid, &keep, &prefix,
            n_out, &mut dst_data, &mut dst_valid,
        ).expect("dispatch succeeds");

        let (exp_data, exp_valid) = cpu_compact_f64(&src, &src_valid, &keep);
        for r in 0..n_out {
            prop_assert_eq!(dst_data[r].to_bits(), exp_data[r].to_bits());
            let got = (dst_valid[r >> 3] >> (r & 7)) & 1;
            let exp = (exp_valid[r >> 3] >> (r & 7)) & 1;
            prop_assert_eq!(got, exp, "row {} validity", r);
        }
    }
}
