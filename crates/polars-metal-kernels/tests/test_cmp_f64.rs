// crates/polars-metal-kernels/tests/test_cmp_f64.rs
//
// Correctness tests for the `cmp_f64_*` kernels (Task 17).
//
// Twelve entry points: six column-column ops (eq, ne, lt, le, gt, ge)
// and the matching six column-scalar variants. Each kernel reads one or
// two f64 columns (bound as `ulong` 8-byte payloads on the GPU because
// Apple Silicon MSL compute kernels do not support `double`) plus their
// bit-packed validity bitmaps, and writes a bit-packed bool column plus
// its validity bitmap.
//
// Output semantics under Polars/IEEE 754 NaN rules:
//   - `out_valid[i]` = `lhs_v[i] AND rhs_v[i]` (column-column) or
//     `lhs_v[i]` (column-scalar). NaN's validity bit stays set — NaN is
//     a value, not null.
//   - `out_data[i]` reflects the IEEE-ordered comparison: `NaN <op> x`
//     is **false** for `==/</<=/>/>=` and **true** for `!=`. Data bits
//     at null rows are zero.
//
// Like the other null-aware kernels, 8 output rows share one byte and
// multiple threads can race the same byte, so the kernels use atomic OR.
// All tests require Metal-capable hardware and serialise on a process-
// wide `Mutex` (see Task 16's test file for the rationale).
#![allow(clippy::expect_used)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::cmp::{dispatch_cmp_f64, dispatch_cmp_f64_scalar, CompareOp};
use polars_metal_kernels::command::CommandQueue;
use proptest::prelude::*;
use std::sync::Mutex;

/// Serialise Metal tests to avoid shader-cache thrash across parallel
/// `cargo test` workers (mirrors `test_cmp_i64.rs`).
static METAL_TEST_LOCK: Mutex<()> = Mutex::new(());

fn device_and_queue() -> (MetalDevice, CommandQueue) {
    let device = MetalDevice::system_default().expect("Metal-capable hardware");
    let queue = CommandQueue::new(&device).expect("queue creation");
    (device, queue)
}

/// Minimum bytes for a bit-packed output bitmap (matches `out_min_bytes`
/// in the dispatcher).
fn out_bytes(n: usize) -> usize {
    let raw = (n + 7) / 8;
    let padded = (raw + 3) & !3;
    padded.max(4)
}

fn out_alloc(n: usize) -> (Vec<u8>, Vec<u8>) {
    let b = out_bytes(n);
    (vec![0u8; b], vec![0u8; b])
}

/// CPU reference: produce expected (data, valid) bit-packed buffers for
/// a column-column f64 comparison. Mirrors the kernel exactly, including
/// the IEEE-ordered NaN rules. Rust's `f64` operators already follow
/// IEEE 754 (Polars' spec), so we just transcribe them.
fn cpu_cmp_f64_cc(
    lhs: &[f64],
    lhs_v: &[u8],
    rhs: &[f64],
    rhs_v: &[u8],
    op: CompareOp,
) -> (Vec<u8>, Vec<u8>) {
    let n = lhs.len();
    let b = out_bytes(n);
    let mut out_data = vec![0u8; b];
    let mut out_valid = vec![0u8; b];
    for i in 0..n {
        let lv = (lhs_v[i >> 3] >> (i & 7)) & 1 == 1;
        let rv = (rhs_v[i >> 3] >> (i & 7)) & 1 == 1;
        if lv && rv {
            // IEEE 754 rules: NaN <op> x is false for ==/</<=/>/>= and
            // true for !=. Rust gives us this for free.
            let r = match op {
                CompareOp::Eq => lhs[i] == rhs[i],
                CompareOp::Ne => lhs[i] != rhs[i],
                CompareOp::Lt => lhs[i] < rhs[i],
                CompareOp::Le => lhs[i] <= rhs[i],
                CompareOp::Gt => lhs[i] > rhs[i],
                CompareOp::Ge => lhs[i] >= rhs[i],
            };
            if r {
                out_data[i >> 3] |= 1u8 << (i & 7);
            }
            out_valid[i >> 3] |= 1u8 << (i & 7);
        }
    }
    (out_data, out_valid)
}

/// CPU reference for column-scalar (the scalar is always-valid; output
/// validity = lhs_v).
fn cpu_cmp_f64_cs(lhs: &[f64], lhs_v: &[u8], rhs: f64, op: CompareOp) -> (Vec<u8>, Vec<u8>) {
    let n = lhs.len();
    let b = out_bytes(n);
    let mut out_data = vec![0u8; b];
    let mut out_valid = vec![0u8; b];
    for i in 0..n {
        let lv = (lhs_v[i >> 3] >> (i & 7)) & 1 == 1;
        if lv {
            let r = match op {
                CompareOp::Eq => lhs[i] == rhs,
                CompareOp::Ne => lhs[i] != rhs,
                CompareOp::Lt => lhs[i] < rhs,
                CompareOp::Le => lhs[i] <= rhs,
                CompareOp::Gt => lhs[i] > rhs,
                CompareOp::Ge => lhs[i] >= rhs,
            };
            if r {
                out_data[i >> 3] |= 1u8 << (i & 7);
            }
            out_valid[i >> 3] |= 1u8 << (i & 7);
        }
    }
    (out_data, out_valid)
}

fn ops() -> [CompareOp; 6] {
    [
        CompareOp::Eq,
        CompareOp::Ne,
        CompareOp::Lt,
        CompareOp::Le,
        CompareOp::Gt,
        CompareOp::Ge,
    ]
}

#[test]
fn cmp_f64_lt_basic() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let rhs = vec![5.0f64; 8];
    let lhs_v = vec![0xFFu8];
    let rhs_v = vec![0xFFu8];
    let n = 8;
    let (mut got_data, mut got_valid) = out_alloc(n);
    dispatch_cmp_f64(
        &device,
        &mut queue,
        &lhs,
        &lhs_v,
        &rhs,
        &rhs_v,
        n,
        CompareOp::Lt,
        &mut got_data,
        &mut got_valid,
    )
    .expect("dispatch ok");
    // Rows 0..3 (1,2,3,4) < 5 -> true; rows 4..7 (5,6,7,8) < 5 -> false.
    assert_eq!(got_data[0] & 0x0Fu8, 0x0Fu8, "bits 0..3 set");
    assert_eq!(got_data[0] & 0xF0u8, 0u8, "bits 4..7 clear");
    assert_eq!(got_valid[0], 0xFFu8, "all valid");
}

/// Polars/IEEE 754 NaN rule check, column-column. `NaN <op> x` should
/// produce `false` for `==/</<=/>/>=` and `true` for `!=`. Crucially,
/// NaN's validity bit stays set (NaN is a value, not null).
#[test]
fn nan_vs_value_comparisons_match_polars() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    // Row 0: NaN vs finite. Row 1: finite vs NaN. Row 2: finite vs finite.
    // Row 3: NaN vs NaN.
    let lhs = vec![f64::NAN, 1.0, 2.0, f64::NAN];
    let rhs = vec![1.0f64, f64::NAN, 2.0, f64::NAN];
    let lhs_v = vec![0x0Fu8];
    let rhs_v = vec![0x0Fu8];
    let n = 4;

    for op in ops() {
        let (mut got_data, mut got_valid) = out_alloc(n);
        dispatch_cmp_f64(
            &device,
            &mut queue,
            &lhs,
            &lhs_v,
            &rhs,
            &rhs_v,
            n,
            op,
            &mut got_data,
            &mut got_valid,
        )
        .expect("dispatch ok");
        for i in 0..n {
            let expected = match op {
                CompareOp::Eq => lhs[i] == rhs[i],
                CompareOp::Ne => lhs[i] != rhs[i],
                CompareOp::Lt => lhs[i] < rhs[i],
                CompareOp::Le => lhs[i] <= rhs[i],
                CompareOp::Gt => lhs[i] > rhs[i],
                CompareOp::Ge => lhs[i] >= rhs[i],
            };
            let g = (got_data[i >> 3] >> (i & 7)) & 1;
            let gv = (got_valid[i >> 3] >> (i & 7)) & 1;
            assert_eq!(
                gv, 1,
                "op={:?} row={} validity (NaN is still a valid value)",
                op, i
            );
            assert_eq!(g == 1, expected, "op={:?} row={}", op, i);
        }
    }
}

/// Same NaN check for the column-scalar path, with a NaN scalar. With
/// any NaN involved, IEEE 754 says `==/</<=/>/>=` are all false and
/// `!=` is true, regardless of the row value. We assert that directly
/// rather than feeding `f64::NAN` through CPU comparison operators
/// (clippy flags those as suspicious, since they always evaluate to
/// the same constant — which is exactly the point we want to test).
#[test]
fn nan_scalar_comparisons_match_polars() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs = vec![1.0f64, f64::NAN, -1.0, f64::INFINITY];
    let lhs_v = vec![0x0Fu8];
    let n = 4;

    for op in ops() {
        // For any NaN scalar: every valid row's expected bit follows
        // the IEEE rule. != is the only op that returns true.
        let expected_bit = matches!(op, CompareOp::Ne);
        let (mut got_data, mut got_valid) = out_alloc(n);
        dispatch_cmp_f64_scalar(
            &device,
            &mut queue,
            &lhs,
            &lhs_v,
            f64::NAN,
            n,
            op,
            &mut got_data,
            &mut got_valid,
        )
        .expect("dispatch ok");
        for i in 0..n {
            let g = (got_data[i >> 3] >> (i & 7)) & 1;
            let gv = (got_valid[i >> 3] >> (i & 7)) & 1;
            assert_eq!(gv, 1, "op={:?} scalar=NaN row={} validity", op, i);
            assert_eq!(
                g == 1,
                expected_bit,
                "op={:?} scalar=NaN row={} data",
                op,
                i
            );
        }
    }
}

#[test]
fn cmp_f64_eq_with_nulls() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs = vec![1.0f64, 2.0, 3.0, 4.0];
    let rhs = vec![1.0f64, 2.0, 3.0, 4.0];
    // Rows 0,1 null on lhs.
    let lhs_v = vec![0b0000_1100u8];
    let rhs_v = vec![0xFFu8];
    let n = 4;
    let (mut got_data, mut got_valid) = out_alloc(n);
    dispatch_cmp_f64(
        &device,
        &mut queue,
        &lhs,
        &lhs_v,
        &rhs,
        &rhs_v,
        n,
        CompareOp::Eq,
        &mut got_data,
        &mut got_valid,
    )
    .expect("dispatch ok");
    // Output validity = lhs AND rhs => bits 2,3 only.
    assert_eq!(got_valid[0] & 0x0Fu8, 0b0000_1100u8);
    // Output data: only at valid rows; values match => bits 2,3 set.
    assert_eq!(got_data[0] & 0x0Fu8, 0b0000_1100u8);
}

#[test]
fn cmp_f64_lt_scalar_basic() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs = vec![0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let lhs_v = vec![0xFFu8];
    let n = 8;
    let (mut got_data, mut got_valid) = out_alloc(n);
    dispatch_cmp_f64_scalar(
        &device,
        &mut queue,
        &lhs,
        &lhs_v,
        4.0f64,
        n,
        CompareOp::Lt,
        &mut got_data,
        &mut got_valid,
    )
    .expect("dispatch ok");
    // Rows 0..3 < 4 -> true; rows 4..7 -> false.
    assert_eq!(got_data[0], 0x0Fu8);
    assert_eq!(got_valid[0], 0xFFu8);
}

/// Spot-check IEEE 754 corner cases: ±Inf, ±0.0, and the signed-zero
/// equality rule. Rust's CPU `f64` comparisons follow IEEE exactly, so
/// we just compare against the CPU reference rather than hand-coding
/// expected bits — the test is here to catch regressions specifically
/// in these tricky cases.
#[test]
fn cmp_f64_special_values() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    // ±Inf, ±0.0, tiniest subnormals, ±1.0.
    let lhs = vec![
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        1.0,
        -1.0,
    ];
    let rhs = vec![0.0f64, 0.0, -0.0, 0.0, 0.0, 0.0, -1.0, 1.0];
    let lhs_v = vec![0xFFu8];
    let rhs_v = vec![0xFFu8];
    let n = 8;

    for op in ops() {
        let (mut got_data, mut got_valid) = out_alloc(n);
        dispatch_cmp_f64(
            &device,
            &mut queue,
            &lhs,
            &lhs_v,
            &rhs,
            &rhs_v,
            n,
            op,
            &mut got_data,
            &mut got_valid,
        )
        .expect("dispatch ok");
        let (exp_data, exp_valid) = cpu_cmp_f64_cc(&lhs, &lhs_v, &rhs, &rhs_v, op);
        for r in 0..n {
            let g = (got_data[r >> 3] >> (r & 7)) & 1;
            let e = (exp_data[r >> 3] >> (r & 7)) & 1;
            let gv = (got_valid[r >> 3] >> (r & 7)) & 1;
            let ev = (exp_valid[r >> 3] >> (r & 7)) & 1;
            assert_eq!(gv, ev, "op={:?} row={} validity", op, r);
            assert_eq!(
                g, e,
                "op={:?} row={} data (lhs={}, rhs={})",
                op, r, lhs[r], rhs[r]
            );
        }
    }
}

/// `0.0 == -0.0` is the one IEEE 754 case where two distinct bit
/// patterns must compare equal. Explicit spot-check rather than relying
/// on the proptest harness to hit it.
#[test]
fn signed_zero_equality() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs = vec![0.0f64, -0.0, 0.0, -0.0];
    let rhs = vec![-0.0f64, 0.0, 0.0, -0.0];
    let lhs_v = vec![0x0Fu8];
    let rhs_v = vec![0x0Fu8];
    let n = 4;

    // All four pairs must compare ==-true and <-false (no signed-zero
    // ordering under IEEE), with full validity.
    let (mut got_data, mut got_valid) = out_alloc(n);
    dispatch_cmp_f64(
        &device,
        &mut queue,
        &lhs,
        &lhs_v,
        &rhs,
        &rhs_v,
        n,
        CompareOp::Eq,
        &mut got_data,
        &mut got_valid,
    )
    .expect("dispatch ok");
    assert_eq!(got_data[0] & 0x0Fu8, 0x0Fu8, "all ±0 pairs equal");
    assert_eq!(got_valid[0] & 0x0Fu8, 0x0Fu8);

    let (mut got_data, mut got_valid) = out_alloc(n);
    dispatch_cmp_f64(
        &device,
        &mut queue,
        &lhs,
        &lhs_v,
        &rhs,
        &rhs_v,
        n,
        CompareOp::Lt,
        &mut got_data,
        &mut got_valid,
    )
    .expect("dispatch ok");
    assert_eq!(got_data[0] & 0x0Fu8, 0u8, "no ±0 pair is strictly less");
    assert_eq!(got_valid[0] & 0x0Fu8, 0x0Fu8);

    let (mut got_data, mut got_valid) = out_alloc(n);
    dispatch_cmp_f64(
        &device,
        &mut queue,
        &lhs,
        &lhs_v,
        &rhs,
        &rhs_v,
        n,
        CompareOp::Le,
        &mut got_data,
        &mut got_valid,
    )
    .expect("dispatch ok");
    assert_eq!(got_data[0] & 0x0Fu8, 0x0Fu8, "all ±0 pairs <=");
}

#[test]
fn cmp_f64_all_six_ops_smoke() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs = vec![1.0f64, 2.0, 3.0, 4.0];
    let rhs = vec![2.0f64, 2.0, 2.0, 2.0];
    let lhs_v = vec![0xFFu8];
    let rhs_v = vec![0xFFu8];
    let n = 4;
    for op in ops() {
        let (mut got_data, mut got_valid) = out_alloc(n);
        dispatch_cmp_f64(
            &device,
            &mut queue,
            &lhs,
            &lhs_v,
            &rhs,
            &rhs_v,
            n,
            op,
            &mut got_data,
            &mut got_valid,
        )
        .expect("dispatch ok");
        let (exp_data, exp_valid) = cpu_cmp_f64_cc(&lhs, &lhs_v, &rhs, &rhs_v, op);
        for r in 0..n {
            let g = (got_data[r >> 3] >> (r & 7)) & 1;
            let e = (exp_data[r >> 3] >> (r & 7)) & 1;
            let gv = (got_valid[r >> 3] >> (r & 7)) & 1;
            let ev = (exp_valid[r >> 3] >> (r & 7)) & 1;
            assert_eq!(gv, ev, "op={:?} row={} validity", op, r);
            if ev == 1 {
                assert_eq!(g, e, "op={:?} row={} data", op, r);
            }
        }
    }
}

#[test]
fn cmp_f64_all_six_scalar_ops_smoke() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs = vec![1.0f64, 2.0, 3.0, 4.0];
    let scalar = 3.0f64;
    let lhs_v = vec![0xFFu8];
    let n = 4;
    for op in ops() {
        let (mut got_data, mut got_valid) = out_alloc(n);
        dispatch_cmp_f64_scalar(
            &device,
            &mut queue,
            &lhs,
            &lhs_v,
            scalar,
            n,
            op,
            &mut got_data,
            &mut got_valid,
        )
        .expect("dispatch ok");
        let (exp_data, exp_valid) = cpu_cmp_f64_cs(&lhs, &lhs_v, scalar, op);
        for r in 0..n {
            let g = (got_data[r >> 3] >> (r & 7)) & 1;
            let e = (exp_data[r >> 3] >> (r & 7)) & 1;
            let gv = (got_valid[r >> 3] >> (r & 7)) & 1;
            let ev = (exp_valid[r >> 3] >> (r & 7)) & 1;
            assert_eq!(gv, ev, "op={:?} (scalar) row={} validity", op, r);
            if ev == 1 {
                assert_eq!(g, e, "op={:?} (scalar) row={} data", op, r);
            }
        }
    }
}

#[test]
fn cmp_f64_multi_thread_same_byte_stress() {
    // 8 threads race ONE byte in BOTH output buffers. With atomic OR
    // the result must be all-1s for both data and validity; a non-
    // atomic write would lose at least one bit.
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs = vec![5.0f64; 8];
    let rhs = vec![5.0f64; 8];
    let lhs_v = vec![0xFFu8];
    let rhs_v = vec![0xFFu8];
    let n = 8;
    let (mut got_data, mut got_valid) = out_alloc(n);
    dispatch_cmp_f64(
        &device,
        &mut queue,
        &lhs,
        &lhs_v,
        &rhs,
        &rhs_v,
        n,
        CompareOp::Eq,
        &mut got_data,
        &mut got_valid,
    )
    .expect("dispatch ok");
    assert_eq!(got_data[0], 0xFFu8, "all 8 data bits must be set");
    assert_eq!(got_valid[0], 0xFFu8, "all 8 validity bits must be set");
}

#[test]
fn cmp_f64_unaligned_row_count() {
    // 13 rows in two bytes (3 padding bits in the second byte). The
    // kernel reads exactly 13 rows; the trailing 3 bits of byte 1 in
    // the output stay zero.
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs: Vec<f64> = (0..13).map(|i| i as f64).collect();
    let rhs = vec![6.0f64; 13];
    let lhs_v = vec![0xFFu8, 0b0001_1111u8];
    let rhs_v = vec![0xFFu8, 0b0001_1111u8];
    let n = 13;
    let (mut got_data, mut got_valid) = out_alloc(n);
    dispatch_cmp_f64(
        &device,
        &mut queue,
        &lhs,
        &lhs_v,
        &rhs,
        &rhs_v,
        n,
        CompareOp::Lt,
        &mut got_data,
        &mut got_valid,
    )
    .expect("dispatch ok");
    let (exp_data, exp_valid) = cpu_cmp_f64_cc(&lhs, &lhs_v, &rhs, &rhs_v, CompareOp::Lt);
    for r in 0..n {
        let g = (got_data[r >> 3] >> (r & 7)) & 1;
        let e = (exp_data[r >> 3] >> (r & 7)) & 1;
        let gv = (got_valid[r >> 3] >> (r & 7)) & 1;
        let ev = (exp_valid[r >> 3] >> (r & 7)) & 1;
        assert_eq!(gv, ev, "row {r} validity");
        if ev == 1 {
            assert_eq!(g, e, "row {r} data");
        }
    }
    // Padding bits at rows 13,14,15 must stay zero.
    for r in n..16 {
        let g = (got_data[r >> 3] >> (r & 7)) & 1;
        let gv = (got_valid[r >> 3] >> (r & 7)) & 1;
        assert_eq!(g, 0, "padding row {r} data must be zero");
        assert_eq!(gv, 0, "padding row {r} validity must be zero");
    }
}

#[test]
fn cmp_f64_n_rows_zero_is_no_op() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs: Vec<f64> = Vec::new();
    let rhs: Vec<f64> = Vec::new();
    let lhs_v: Vec<u8> = Vec::new();
    let rhs_v: Vec<u8> = Vec::new();
    let (mut got_data, mut got_valid) = out_alloc(0);
    dispatch_cmp_f64(
        &device,
        &mut queue,
        &lhs,
        &lhs_v,
        &rhs,
        &rhs_v,
        0,
        CompareOp::Lt,
        &mut got_data,
        &mut got_valid,
    )
    .expect("zero rows is a no-op");
    assert!(got_data.iter().all(|&b| b == 0));
    assert!(got_valid.iter().all(|&b| b == 0));
}

#[test]
fn cmp_f64_input_length_mismatch_errors() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs = vec![1.0f64, 2.0, 3.0];
    let rhs = vec![1.0f64, 2.0]; // shorter
    let lhs_v = vec![0xFFu8];
    let rhs_v = vec![0xFFu8];
    let n = 3;
    let (mut got_data, mut got_valid) = out_alloc(n);
    let err = dispatch_cmp_f64(
        &device,
        &mut queue,
        &lhs,
        &lhs_v,
        &rhs,
        &rhs_v,
        n,
        CompareOp::Lt,
        &mut got_data,
        &mut got_valid,
    )
    .expect_err("mismatched lengths must error");
    let msg = format!("{err}");
    assert!(msg.contains("input length mismatch"), "got: {msg}");
}

#[test]
fn cmp_f64_output_too_short_errors() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs = vec![1.0f64; 8];
    let rhs = vec![1.0f64; 8];
    let lhs_v = vec![0xFFu8];
    let rhs_v = vec![0xFFu8];
    let n = 8;
    let mut got_data = vec![0u8; 1]; // too small (need >= 4)
    let mut got_valid = vec![0u8; 4];
    let err = dispatch_cmp_f64(
        &device,
        &mut queue,
        &lhs,
        &lhs_v,
        &rhs,
        &rhs_v,
        n,
        CompareOp::Eq,
        &mut got_data,
        &mut got_valid,
    )
    .expect_err("undersized output must error");
    let msg = format!("{err}");
    assert!(msg.contains("output buffer too short"), "got: {msg}");
}

#[test]
fn cmp_f64_validity_too_short_errors() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs = vec![1.0f64; 16];
    let rhs = vec![1.0f64; 16];
    let lhs_v = vec![0xFFu8]; // only 1 byte but need 2
    let rhs_v = vec![0xFFu8, 0xFFu8];
    let n = 16;
    let (mut got_data, mut got_valid) = out_alloc(n);
    let err = dispatch_cmp_f64(
        &device,
        &mut queue,
        &lhs,
        &lhs_v,
        &rhs,
        &rhs_v,
        n,
        CompareOp::Eq,
        &mut got_data,
        &mut got_valid,
    )
    .expect_err("undersized validity must error");
    let msg = format!("{err}");
    assert!(msg.contains("validity buffer too short"), "got: {msg}");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Differential: dispatch_cmp_f64 must match Rust's IEEE 754 CPU
    /// comparison for arbitrary mixtures of finite values, NaNs, and
    /// infinities, across random null patterns.
    #[test]
    fn cmp_f64_cc_matches_cpu(
        n in 8usize..256,
        seed in any::<u64>(),
        op_idx in 0u8..6,
    ) {
        let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let op = ops()[op_idx as usize];
        // Mix of finite, NaN, infinities for ~30% NaN density. Each
        // NaN carries a unique payload (so we exercise non-canonical
        // NaN bit patterns, not just the canonical 0x7FF8_..._0000).
        let lhs: Vec<f64> = (0..n).map(|i| {
            let s = seed.rotate_left(i as u32);
            if (s % 10) < 3 { f64::from_bits(0x7FF8_0000_0000_0000u64 | (s & 0xFFFF)) }
            else if (s % 100) == 0 { f64::INFINITY }
            else { (s as i64 as f64) * 0.001 }
        }).collect();
        let rhs: Vec<f64> = (0..n).map(|i| {
            let s = seed.rotate_left((i as u32) ^ 13);
            if (s % 10) < 3 { f64::from_bits(0x7FF8_0000_0000_0000u64 | (s & 0xFFFF)) }
            else if (s % 100) == 0 { f64::NEG_INFINITY }
            else { (s as i64 as f64) * 0.001 }
        }).collect();
        let bytes = (n + 7) / 8;
        let mut lhs_v = vec![0u8; bytes];
        let mut rhs_v = vec![0u8; bytes];
        for i in 0..n {
            if (seed.rotate_left((i as u32) ^ 7) & 1) == 1 {
                lhs_v[i >> 3] |= 1u8 << (i & 7);
            }
            if (seed.rotate_left((i as u32) ^ 11) & 1) == 1 {
                rhs_v[i >> 3] |= 1u8 << (i & 7);
            }
        }
        let (device, mut queue) = device_and_queue();
        let (mut got_data, mut got_valid) = out_alloc(n);
        dispatch_cmp_f64(
            &device, &mut queue,
            &lhs, &lhs_v, &rhs, &rhs_v,
            n, op, &mut got_data, &mut got_valid,
        ).expect("dispatch ok");
        let (exp_data, exp_valid) = cpu_cmp_f64_cc(&lhs, &lhs_v, &rhs, &rhs_v, op);
        for i in 0..n {
            let g = (got_data[i >> 3] >> (i & 7)) & 1;
            let e = (exp_data[i >> 3] >> (i & 7)) & 1;
            let gv = (got_valid[i >> 3] >> (i & 7)) & 1;
            let ev = (exp_valid[i >> 3] >> (i & 7)) & 1;
            prop_assert_eq!(gv, ev, "row {} validity (op={:?})", i, op);
            if ev == 1 {
                prop_assert_eq!(g, e, "row {} data (op={:?}, lhs={}, rhs={})", i, op, lhs[i], rhs[i]);
            }
        }
    }

    /// Differential for the column-scalar dispatcher. Scalar may itself
    /// be NaN or ±Inf to exercise the all-rows-false / all-rows-true
    /// behaviour.
    #[test]
    fn cmp_f64_cs_matches_cpu(
        n in 8usize..256,
        seed in any::<u64>(),
        op_idx in 0u8..6,
        scalar_kind in 0u8..4,
    ) {
        let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let op = ops()[op_idx as usize];
        let scalar = match scalar_kind {
            0 => f64::NAN,
            1 => f64::INFINITY,
            2 => f64::NEG_INFINITY,
            _ => (seed as i64 as f64) * 0.001,
        };
        let lhs: Vec<f64> = (0..n).map(|i| {
            let s = seed.rotate_left(i as u32);
            if (s % 10) < 3 { f64::from_bits(0x7FF8_0000_0000_0000u64 | (s & 0xFFFF)) }
            else { (s as i64 as f64) * 0.001 }
        }).collect();
        let bytes = (n + 7) / 8;
        let mut lhs_v = vec![0u8; bytes];
        for i in 0..n {
            if (seed.rotate_left((i as u32) ^ 7) & 1) == 1 {
                lhs_v[i >> 3] |= 1u8 << (i & 7);
            }
        }
        let (device, mut queue) = device_and_queue();
        let (mut got_data, mut got_valid) = out_alloc(n);
        dispatch_cmp_f64_scalar(
            &device, &mut queue,
            &lhs, &lhs_v, scalar,
            n, op, &mut got_data, &mut got_valid,
        ).expect("dispatch ok");
        let (exp_data, exp_valid) = cpu_cmp_f64_cs(&lhs, &lhs_v, scalar, op);
        for i in 0..n {
            let g = (got_data[i >> 3] >> (i & 7)) & 1;
            let e = (exp_data[i >> 3] >> (i & 7)) & 1;
            let gv = (got_valid[i >> 3] >> (i & 7)) & 1;
            let ev = (exp_valid[i >> 3] >> (i & 7)) & 1;
            prop_assert_eq!(gv, ev, "row {} validity (op={:?}, scalar={})", i, op, scalar);
            if ev == 1 {
                prop_assert_eq!(g, e, "row {} data (op={:?}, scalar={}, lhs={})", i, op, scalar, lhs[i]);
            }
        }
    }
}
