// crates/polars-metal-kernels/tests/test_compaction_pipeline.rs
//
// End-to-end correctness tests for the three-pass filter compaction
// pipeline (Task 14): bit-packed predicate -> dense u8 keep flags
// (Task 10) -> inclusive prefix sum via MLX (Task 5) -> scattered
// outputs (Tasks 11-13). The kernels and the cumsum are individually
// tested elsewhere; here we verify that the orchestration in
// `pipeline.rs` composes them correctly across all three dtypes.
//
// The 10K-row test is the real stress: it forces MLX cumsum across
// multiple threadgroups, where a buggy partial-sum reduction would
// silently produce wrong scatter offsets. All other tests exercise
// edge cases (empty result, all kept, NaN payload preservation,
// bit-packed bool round-trip).
//
// All tests require Metal-capable hardware; they will fail with an
// `expect` error on machines without a discoverable system-default
// MTLDevice.
#![allow(clippy::expect_used)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::pipeline::{compact_bool, compact_f64, compact_i64};
use proptest::prelude::*;
use std::sync::Mutex;

fn device_and_queue() -> (MetalDevice, CommandQueue) {
    let device = MetalDevice::system_default().expect("Metal-capable hardware");
    let queue = CommandQueue::new(&device).expect("queue creation");
    (device, queue)
}

/// Serialises every test in this file. Without this mutex, the Rust
/// test runner runs multiple tests in parallel, each of which
/// allocates its own `MetalDevice` + `CommandQueue` and routes a
/// cumsum through MLX. The combination (~3 proptests of 256 cases
/// each plus 8 explicit tests, all racing for Metal resources) makes
/// the host hit Metal's "Internal Error 00000206" — not a bug in the
/// pipeline, just resource pressure. The serialisation is test-only;
/// the real engine sees one query at a time. Production code does
/// not need this lock.
///
/// `Mutex::lock` returns `PoisonError` when a prior holder panicked;
/// in tests we still want to proceed — the next case re-creates its
/// own MetalDevice and so does not depend on any state guarded by the
/// mutex — hence the `unwrap_or_else(|e| e.into_inner())` pattern at
/// every acquisition site.
static METAL_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Convenience: acquire the global test mutex, tolerating poison from
/// a prior test's panic. We never poke at state behind the lock; it
/// only serialises Metal device/queue creation across tests.
fn lock_metal() -> std::sync::MutexGuard<'static, ()> {
    METAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[test]
fn compact_i64_basic() {
    let _guard = lock_metal();
    let (device, mut queue) = device_and_queue();
    // 16 rows; predicate is true for even rows.
    let src: Vec<i64> = (0..16).collect();
    let src_valid = vec![0xFFu8, 0xFFu8];
    let pred_data = vec![0b01010101u8, 0b01010101u8]; // rows 0, 2, 4, ..., 14
    let pred_valid = vec![0xFFu8, 0xFFu8];
    let result = compact_i64(
        &device,
        &mut queue,
        &src,
        &src_valid,
        &pred_data,
        &pred_valid,
        16,
    )
    .expect("compaction succeeds");
    assert_eq!(result.n_out, 8);
    assert_eq!(result.data, vec![0i64, 2, 4, 6, 8, 10, 12, 14]);
    // First byte of validity: bits 0..7 all set (all 8 outputs were valid).
    assert_eq!(result.valid[0], 0xFFu8);
}

#[test]
fn compact_i64_empty_result() {
    let _guard = lock_metal();
    let (device, mut queue) = device_and_queue();
    let src: Vec<i64> = (0..8).collect();
    let src_valid = vec![0xFFu8];
    let pred_data = vec![0u8]; // no row passes
    let pred_valid = vec![0xFFu8];
    let result = compact_i64(
        &device,
        &mut queue,
        &src,
        &src_valid,
        &pred_data,
        &pred_valid,
        8,
    )
    .expect("succeeds");
    assert_eq!(result.n_out, 0);
    assert_eq!(result.data.len(), 0);
}

#[test]
fn compact_i64_all_kept() {
    let _guard = lock_metal();
    let (device, mut queue) = device_and_queue();
    let src: Vec<i64> = (0..16).collect();
    let src_valid = vec![0xFFu8, 0xFFu8];
    let pred_data = vec![0xFFu8, 0xFFu8];
    let pred_valid = vec![0xFFu8, 0xFFu8];
    let result = compact_i64(
        &device,
        &mut queue,
        &src,
        &src_valid,
        &pred_data,
        &pred_valid,
        16,
    )
    .expect("succeeds");
    assert_eq!(result.n_out, 16);
    assert_eq!(result.data, src);
}

#[test]
fn compact_i64_threadgroup_boundary_10k() {
    // 10K rows — exercises cumsum across multiple threadgroups.
    let _guard = lock_metal();
    let (device, mut queue) = device_and_queue();
    let n = 10_000usize;
    let src: Vec<i64> = (0..n as i64).collect();
    let src_valid_bytes = (n + 7) / 8;
    let src_valid = vec![0xFFu8; src_valid_bytes];
    // Predicate: keep rows where index % 3 == 0 → roughly n/3 outputs.
    let mut pred_data = vec![0u8; src_valid_bytes];
    for r in 0..n {
        if r % 3 == 0 {
            pred_data[r >> 3] |= 1u8 << (r & 7);
        }
    }
    let pred_valid = vec![0xFFu8; src_valid_bytes];
    let result = compact_i64(
        &device,
        &mut queue,
        &src,
        &src_valid,
        &pred_data,
        &pred_valid,
        n,
    )
    .expect("succeeds");
    let expected: Vec<i64> = (0..n as i64).filter(|i| i % 3 == 0).collect();
    assert_eq!(result.n_out, expected.len());
    assert_eq!(result.data, expected);
}

#[test]
fn compact_f64_preserves_nan() {
    let _guard = lock_metal();
    let (device, mut queue) = device_and_queue();
    // Deliberately non-canonical NaN payload — verifies the scatter
    // copies the exact bit pattern.
    let nan = f64::from_bits(0x7FF8_0000_0000_BEEFu64);
    let src: Vec<f64> = vec![1.0, nan, 3.0, nan];
    let src_valid = vec![0xFFu8];
    let pred_data = vec![0xFFu8]; // keep all 4
    let pred_valid = vec![0xFFu8];
    let result = compact_f64(
        &device,
        &mut queue,
        &src,
        &src_valid,
        &pred_data,
        &pred_valid,
        4,
    )
    .expect("succeeds");
    assert_eq!(result.n_out, 4);
    for (i, (got, want)) in result.data.iter().zip(src.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "row {i}: bit pattern must round-trip exactly"
        );
    }
}

#[test]
fn compact_bool_round_trips() {
    let _guard = lock_metal();
    let (device, mut queue) = device_and_queue();
    let src_data = vec![0b10110100u8]; // rows 2, 4, 5, 7 are true
    let src_valid = vec![0xFFu8];
    let pred_data = vec![0xFFu8]; // keep all 8
    let pred_valid = vec![0xFFu8];
    let result = compact_bool(
        &device,
        &mut queue,
        &src_data,
        &src_valid,
        &pred_data,
        &pred_valid,
        8,
    )
    .expect("succeeds");
    assert_eq!(result.n_out, 8);
    assert_eq!(result.data[0], 0b10110100u8);
    assert_eq!(result.valid[0], 0xFFu8);
}

#[test]
fn compact_i64_single_row() {
    // n=1 case. The smallest possible non-empty input; ensures the
    // pipeline does not trip on grid sizes below the threadgroup width.
    let _guard = lock_metal();
    let (device, mut queue) = device_and_queue();
    let src = vec![42i64];
    let src_valid = vec![0xFFu8];
    let pred_data = vec![0xFFu8];
    let pred_valid = vec![0xFFu8];
    let result = compact_i64(
        &device,
        &mut queue,
        &src,
        &src_valid,
        &pred_data,
        &pred_valid,
        1,
    )
    .expect("single-row dispatch succeeds");
    assert_eq!(result.n_out, 1);
    assert_eq!(result.data, vec![42]);
    assert_eq!(result.valid[0] & 1, 1);
}

#[test]
fn compact_i64_null_predicate_drops_row() {
    // Polars semantics: a row whose predicate is null is dropped, even
    // if the predicate's data bit happens to be 1. The pipeline's pass 1
    // (filter_predicate_to_u8) folds (data AND valid) into the keep
    // flag, so we get a clean check here that the orchestration honors
    // the null mask.
    let _guard = lock_metal();
    let (device, mut queue) = device_and_queue();
    let src: Vec<i64> = vec![10, 20, 30, 40];
    let src_valid = vec![0xFFu8];
    let pred_data = vec![0b00001111u8]; // bits 0..3 all "true"
    let pred_valid = vec![0b00001010u8]; // but only rows 1 and 3 are non-null
    let result = compact_i64(
        &device,
        &mut queue,
        &src,
        &src_valid,
        &pred_data,
        &pred_valid,
        4,
    )
    .expect("succeeds");
    assert_eq!(result.n_out, 2);
    assert_eq!(result.data, vec![20, 40]);
}

proptest! {
    // Cap cases per proptest function. Each case allocates a fresh
    // `MetalDevice` + `CommandQueue` and runs an MLX cumsum, which
    // accumulates GPU state inside MLX's internal arena across calls
    // within a single test process. Past ~80 sequential cases the
    // host hits Metal "Internal Error 00000206" from internal
    // accumulation; 64 cases give us good shrink-coverage of (n,
    // seed) without tripping the wedge. This is a *test infra*
    // accommodation, not a code-correctness compromise — the
    // engine itself only runs one query per process invocation.
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn compact_i64_matches_cpu_reference(n in 1usize..512, seed in any::<u64>()) {
        let _guard = lock_metal();
        let bytes = (n + 7) / 8;
        let src: Vec<i64> = (0..n as i64).collect();
        let mut src_valid = vec![0u8; bytes];
        let mut pred_data = vec![0u8; bytes];
        let mut pred_valid = vec![0u8; bytes];
        for r in 0..n {
            if (seed.rotate_left((r as u32) & 63) & 1) == 1 {
                src_valid[r >> 3] |= 1u8 << (r & 7);
            }
            if (seed.rotate_left((r as u32 ^ 13) & 63) & 1) == 1 {
                pred_data[r >> 3] |= 1u8 << (r & 7);
            }
            if (seed.rotate_left((r as u32 ^ 31) & 63) & 1) == 1 {
                pred_valid[r >> 3] |= 1u8 << (r & 7);
            }
        }
        let (device, mut queue) = device_and_queue();
        let result = compact_i64(
            &device, &mut queue,
            &src, &src_valid, &pred_data, &pred_valid, n,
        ).expect("succeeds");
        // Compute CPU reference.
        let mut exp_data = Vec::new();
        let mut exp_valid_bits = Vec::new();
        for r in 0..n {
            let d = ((pred_data[r >> 3] >> (r & 7)) & 1) == 1;
            let v = ((pred_valid[r >> 3] >> (r & 7)) & 1) == 1;
            if d && v {
                exp_data.push(src[r]);
                exp_valid_bits.push(((src_valid[r >> 3] >> (r & 7)) & 1) == 1);
            }
        }
        prop_assert_eq!(result.n_out, exp_data.len());
        prop_assert_eq!(&result.data[..], &exp_data[..]);
        for (i, b) in exp_valid_bits.iter().enumerate() {
            let got = (result.valid[i >> 3] >> (i & 7)) & 1;
            let exp = if *b { 1 } else { 0 };
            prop_assert_eq!(got, exp, "validity bit at output row {} should be {}", i, exp);
        }
    }

    #[test]
    fn compact_f64_matches_cpu_reference(n in 1usize..256, seed in any::<u64>()) {
        let _guard = lock_metal();
        let bytes = (n + 7) / 8;
        // Construct a source covering finite values, NaN with payload,
        // ±Inf, and ±0.0 — exactly the cases where a bit-level copy is
        // required and an FP-arithmetic copy would diverge.
        let src: Vec<f64> = (0..n).map(|i| match i % 5 {
            0 => i as f64 * 1.5,
            1 => f64::from_bits(0x7FF8_0000_0000_0000u64 | (i as u64)),
            2 => f64::INFINITY,
            3 => f64::NEG_INFINITY,
            _ => if i & 1 == 0 { -0.0 } else { 0.0 },
        }).collect();
        let mut src_valid = vec![0u8; bytes];
        let mut pred_data = vec![0u8; bytes];
        let mut pred_valid = vec![0u8; bytes];
        for r in 0..n {
            if (seed.rotate_left((r as u32) & 63) & 1) == 1 {
                src_valid[r >> 3] |= 1u8 << (r & 7);
            }
            if (seed.rotate_left((r as u32 ^ 7) & 63) & 1) == 1 {
                pred_data[r >> 3] |= 1u8 << (r & 7);
            }
            if (seed.rotate_left((r as u32 ^ 19) & 63) & 1) == 1 {
                pred_valid[r >> 3] |= 1u8 << (r & 7);
            }
        }
        let (device, mut queue) = device_and_queue();
        let result = compact_f64(
            &device, &mut queue,
            &src, &src_valid, &pred_data, &pred_valid, n,
        ).expect("succeeds");
        let mut exp_data = Vec::new();
        let mut exp_valid_bits = Vec::new();
        for r in 0..n {
            let d = ((pred_data[r >> 3] >> (r & 7)) & 1) == 1;
            let v = ((pred_valid[r >> 3] >> (r & 7)) & 1) == 1;
            if d && v {
                exp_data.push(src[r]);
                exp_valid_bits.push(((src_valid[r >> 3] >> (r & 7)) & 1) == 1);
            }
        }
        prop_assert_eq!(result.n_out, exp_data.len());
        // Bit-level equality: NaN != NaN under IEEE 754, and we want to
        // verify the exact payload round-trip.
        for (i, (g, e)) in result.data.iter().zip(exp_data.iter()).enumerate() {
            prop_assert_eq!(g.to_bits(), e.to_bits(), "f64 bit pattern at row {}", i);
        }
        for (i, b) in exp_valid_bits.iter().enumerate() {
            let got = (result.valid[i >> 3] >> (i & 7)) & 1;
            let exp = if *b { 1 } else { 0 };
            prop_assert_eq!(got, exp, "validity bit at output row {}", i);
        }
    }

    #[test]
    fn compact_bool_matches_cpu_reference(n in 1usize..256, seed in any::<u64>()) {
        let _guard = lock_metal();
        let bytes = (n + 7) / 8;
        let mut src_data = vec![0u8; bytes];
        let mut src_valid = vec![0u8; bytes];
        let mut pred_data = vec![0u8; bytes];
        let mut pred_valid = vec![0u8; bytes];
        for r in 0..n {
            if (seed.rotate_left((r as u32) & 63) & 1) == 1 {
                src_data[r >> 3] |= 1u8 << (r & 7);
            }
            if (seed.rotate_left((r as u32 ^ 3) & 63) & 1) == 1 {
                src_valid[r >> 3] |= 1u8 << (r & 7);
            }
            if (seed.rotate_left((r as u32 ^ 11) & 63) & 1) == 1 {
                pred_data[r >> 3] |= 1u8 << (r & 7);
            }
            if (seed.rotate_left((r as u32 ^ 23) & 63) & 1) == 1 {
                pred_valid[r >> 3] |= 1u8 << (r & 7);
            }
        }
        let (device, mut queue) = device_and_queue();
        let result = compact_bool(
            &device, &mut queue,
            &src_data, &src_valid, &pred_data, &pred_valid, n,
        ).expect("succeeds");
        // Expected: iterate rows, push data bit + validity bit when the
        // predicate (data AND valid) keeps the row.
        let mut exp_data_bits = Vec::new();
        let mut exp_valid_bits = Vec::new();
        for r in 0..n {
            let pd = ((pred_data[r >> 3] >> (r & 7)) & 1) == 1;
            let pv = ((pred_valid[r >> 3] >> (r & 7)) & 1) == 1;
            if pd && pv {
                exp_data_bits.push(((src_data[r >> 3] >> (r & 7)) & 1) == 1);
                exp_valid_bits.push(((src_valid[r >> 3] >> (r & 7)) & 1) == 1);
            }
        }
        prop_assert_eq!(result.n_out, exp_data_bits.len());
        for (i, b) in exp_data_bits.iter().enumerate() {
            let got = (result.data[i >> 3] >> (i & 7)) & 1;
            let exp = if *b { 1 } else { 0 };
            prop_assert_eq!(got, exp, "data bit at output row {}", i);
        }
        for (i, b) in exp_valid_bits.iter().enumerate() {
            let got = (result.valid[i >> 3] >> (i & 7)) & 1;
            let exp = if *b { 1 } else { 0 };
            prop_assert_eq!(got, exp, "validity bit at output row {}", i);
        }
    }
}
