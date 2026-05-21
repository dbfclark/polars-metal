// crates/polars-metal-kernels/tests/test_cmp_i64.rs
//
// Correctness tests for the `cmp_i64_*` kernels (Task 16).
//
// Twelve entry points: six column-column ops (eq, ne, lt, le, gt, ge) and
// the matching six column-scalar variants. Each kernel reads one or two
// i64 columns + their bit-packed validity bitmaps and writes a bit-packed
// bool column + its validity bitmap. Output validity is the AND of the
// input validity bits (column-scalar treats the scalar as always-valid);
// output data is the comparison result only where both inputs are valid.
//
// Like the bool scatter (Task 13), 8 output rows share one byte and
// multiple threads can race the same byte, so the kernels use atomic OR.
// Tests therefore include a multi-thread-same-byte stress to confirm no
// bits are lost.
//
// All tests require Metal-capable hardware. A process-wide `Mutex`
// serialises them to avoid the "Internal Error 00000206" we saw in Task
// 14 when multiple test binaries hammered the system shader cache in
// parallel — `cargo test` schedules tests across threads by default.
#![allow(clippy::expect_used)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::cmp::{dispatch_cmp_i64, dispatch_cmp_i64_scalar, CompareOp};
use polars_metal_kernels::command::CommandQueue;
use proptest::prelude::*;
use std::sync::Mutex;

/// Serialise Metal tests to avoid shader-cache thrash across parallel
/// `cargo test` workers (see comment at top of file).
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

/// CPU reference: produce expected (data, valid) bit-packed buffers for a
/// column-column comparison. Mirrors the kernel exactly:
/// out_valid[i] = lhs_v[i] AND rhs_v[i]
/// out_data[i]  = out_valid[i] AND (lhs[i] OP rhs[i])
fn cpu_cmp_cc(
    lhs: &[i64],
    lhs_v: &[u8],
    rhs: &[i64],
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

/// CPU reference for column-scalar (the scalar is always-valid, so
/// validity = lhs_v).
fn cpu_cmp_cs(lhs: &[i64], lhs_v: &[u8], rhs: i64, op: CompareOp) -> (Vec<u8>, Vec<u8>) {
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
fn cmp_i64_lt_basic() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs = vec![1i64, 2, 3, 4, 5, 6, 7, 8];
    let rhs = vec![5i64; 8];
    let lhs_v = vec![0xFFu8];
    let rhs_v = vec![0xFFu8];
    let n = 8;
    let (mut got_data, mut got_valid) = out_alloc(n);
    dispatch_cmp_i64(
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

#[test]
fn cmp_i64_eq_with_nulls() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs = vec![10i64, 10, 10, 10];
    let rhs = vec![10i64, 10, 10, 10];
    // Rows 0,1 null on lhs.
    let lhs_v = vec![0b0000_1100u8];
    let rhs_v = vec![0xFFu8];
    let n = 4;
    let (mut got_data, mut got_valid) = out_alloc(n);
    dispatch_cmp_i64(
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
    // Output validity = lhs & rhs => bits 2,3 only.
    assert_eq!(got_valid[0] & 0x0Fu8, 0b0000_1100u8);
    // Output data: only at valid rows; 10 == 10 is true => bits 2,3 set.
    assert_eq!(got_data[0] & 0x0Fu8, 0b0000_1100u8);
}

#[test]
fn cmp_i64_lt_scalar_basic() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs = vec![0i64, 1, 2, 3, 4, 5, 6, 7];
    let lhs_v = vec![0xFFu8];
    let n = 8;
    let (mut got_data, mut got_valid) = out_alloc(n);
    dispatch_cmp_i64_scalar(
        &device,
        &mut queue,
        &lhs,
        &lhs_v,
        4i64,
        n,
        CompareOp::Lt,
        &mut got_data,
        &mut got_valid,
    )
    .expect("dispatch ok");
    // Rows 0..3 < 4 -> true. Rows 4..7 -> false.
    assert_eq!(got_data[0], 0x0Fu8);
    assert_eq!(got_valid[0], 0xFFu8);
}

#[test]
fn cmp_i64_scalar_with_nulls() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    // 8 rows, scalar = 5; we expect lhs < 5 at every valid row.
    let lhs = vec![0i64, 1, 2, 3, 4, 5, 6, 7];
    // Rows 1, 3, 5, 7 are null.
    let lhs_v = vec![0b0101_0101u8];
    let n = 8;
    let (mut got_data, mut got_valid) = out_alloc(n);
    dispatch_cmp_i64_scalar(
        &device,
        &mut queue,
        &lhs,
        &lhs_v,
        5i64,
        n,
        CompareOp::Lt,
        &mut got_data,
        &mut got_valid,
    )
    .expect("dispatch ok");
    // Output validity mirrors lhs_v.
    assert_eq!(got_valid[0], 0b0101_0101u8);
    // Output data: valid rows are 0,2,4,6; values 0,2,4,6; only 0,2,4 < 5.
    // So data bits set at rows 0,2,4 (mask 0b0001_0101).
    assert_eq!(got_data[0], 0b0001_0101u8);
}

#[test]
fn cmp_i64_all_six_ops_smoke() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs = vec![1i64, 2, 3, 4];
    let rhs = vec![2i64, 2, 2, 2];
    let lhs_v = vec![0xFFu8];
    let rhs_v = vec![0xFFu8];
    let n = 4;
    for op in ops() {
        let (mut got_data, mut got_valid) = out_alloc(n);
        dispatch_cmp_i64(
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
        let (exp_data, exp_valid) = cpu_cmp_cc(&lhs, &lhs_v, &rhs, &rhs_v, op);
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
fn cmp_i64_all_six_scalar_ops_smoke() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs = vec![1i64, 2, 3, 4];
    let scalar = 3i64;
    let lhs_v = vec![0xFFu8];
    let n = 4;
    for op in ops() {
        let (mut got_data, mut got_valid) = out_alloc(n);
        dispatch_cmp_i64_scalar(
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
        let (exp_data, exp_valid) = cpu_cmp_cs(&lhs, &lhs_v, scalar, op);
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
fn cmp_i64_multi_thread_same_byte_stress() {
    // 8 threads race ONE byte in BOTH output buffers. With atomic OR the
    // result must be all-1s for both data and validity; a non-atomic
    // write would lose at least one bit.
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs = vec![5i64; 8];
    let rhs = vec![5i64; 8];
    let lhs_v = vec![0xFFu8];
    let rhs_v = vec![0xFFu8];
    let n = 8;
    let (mut got_data, mut got_valid) = out_alloc(n);
    dispatch_cmp_i64(
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
fn cmp_i64_unaligned_row_count() {
    // 13 rows in two bytes (3 padding bits in the second byte). The
    // kernel reads exactly 13 rows; the trailing 3 bits of byte 1 in
    // the output stay zero.
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs: Vec<i64> = (0..13i64).collect();
    let rhs = vec![6i64; 13];
    let lhs_v = vec![0xFFu8, 0b0001_1111u8];
    let rhs_v = vec![0xFFu8, 0b0001_1111u8];
    let n = 13;
    let (mut got_data, mut got_valid) = out_alloc(n);
    dispatch_cmp_i64(
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
    let (exp_data, exp_valid) = cpu_cmp_cc(&lhs, &lhs_v, &rhs, &rhs_v, CompareOp::Lt);
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
fn cmp_i64_n_rows_zero_is_no_op() {
    // n_rows == 0: dispatcher must not invoke Metal (Metal rejects
    // zero-byte buffers / zero-grid dispatches). Outputs stay at the
    // zeroed alloc.
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs: Vec<i64> = Vec::new();
    let rhs: Vec<i64> = Vec::new();
    let lhs_v: Vec<u8> = Vec::new();
    let rhs_v: Vec<u8> = Vec::new();
    let (mut got_data, mut got_valid) = out_alloc(0);
    dispatch_cmp_i64(
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
fn cmp_i64_input_length_mismatch_errors() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs = vec![1i64, 2, 3];
    let rhs = vec![1i64, 2]; // shorter
    let lhs_v = vec![0xFFu8];
    let rhs_v = vec![0xFFu8];
    let n = 3;
    let (mut got_data, mut got_valid) = out_alloc(n);
    let err = dispatch_cmp_i64(
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
fn cmp_i64_output_too_short_errors() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs = vec![1i64; 8];
    let rhs = vec![1i64; 8];
    let lhs_v = vec![0xFFu8];
    let rhs_v = vec![0xFFu8];
    let n = 8;
    let mut got_data = vec![0u8; 1]; // too small (need >= 4)
    let mut got_valid = vec![0u8; 4];
    let err = dispatch_cmp_i64(
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
fn cmp_i64_validity_too_short_errors() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs = vec![1i64; 16];
    let rhs = vec![1i64; 16];
    let lhs_v = vec![0xFFu8]; // only 1 byte but need ceil(16/8) = 2
    let rhs_v = vec![0xFFu8, 0xFFu8];
    let n = 16;
    let (mut got_data, mut got_valid) = out_alloc(n);
    let err = dispatch_cmp_i64(
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

#[test]
fn cmp_i64_extreme_values() {
    // i64::MIN and i64::MAX should compare correctly (sign-aware).
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs = vec![i64::MIN, -1, 0, 1, i64::MAX];
    let rhs = vec![0i64; 5];
    let lhs_v = vec![0b0001_1111u8];
    let rhs_v = vec![0b0001_1111u8];
    let n = 5;
    let (mut got_data, mut got_valid) = out_alloc(n);
    dispatch_cmp_i64(
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
    // < 0: i64::MIN (true), -1 (true), 0 (false), 1 (false), i64::MAX (false)
    let expected = 0b0000_0011u8;
    assert_eq!(got_data[0] & 0b0001_1111u8, expected);
    assert_eq!(got_valid[0] & 0b0001_1111u8, 0b0001_1111u8);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn cmp_i64_cc_matches_cpu(
        n in 8usize..256,
        seed in any::<u64>(),
        op_idx in 0u8..6,
    ) {
        let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let op = ops()[op_idx as usize];
        // Build inputs deterministically from `seed`. Modulo 100 keeps
        // values dense enough to exercise all six ops on a single sample
        // (otherwise random i64s rarely repeat / cross zero).
        let lhs: Vec<i64> = (0..n)
            .map(|i| (seed.rotate_left(i as u32) as i64).rem_euclid(100) - 50)
            .collect();
        let rhs: Vec<i64> = (0..n)
            .map(|i| (seed.rotate_left((i as u32) ^ 13) as i64).rem_euclid(100) - 50)
            .collect();
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
        dispatch_cmp_i64(
            &device, &mut queue,
            &lhs, &lhs_v, &rhs, &rhs_v,
            n, op, &mut got_data, &mut got_valid,
        ).expect("dispatch ok");
        let (exp_data, exp_valid) = cpu_cmp_cc(&lhs, &lhs_v, &rhs, &rhs_v, op);
        for i in 0..n {
            let g = (got_data[i >> 3] >> (i & 7)) & 1;
            let e = (exp_data[i >> 3] >> (i & 7)) & 1;
            let gv = (got_valid[i >> 3] >> (i & 7)) & 1;
            let ev = (exp_valid[i >> 3] >> (i & 7)) & 1;
            prop_assert_eq!(gv, ev, "row {} validity (op={:?})", i, op);
            if ev == 1 {
                prop_assert_eq!(g, e, "row {} data (op={:?})", i, op);
            }
        }
    }

    #[test]
    fn cmp_i64_cs_matches_cpu(
        n in 8usize..256,
        seed in any::<u64>(),
        op_idx in 0u8..6,
        scalar in -50i64..50,
    ) {
        let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let op = ops()[op_idx as usize];
        let lhs: Vec<i64> = (0..n)
            .map(|i| (seed.rotate_left(i as u32) as i64).rem_euclid(100) - 50)
            .collect();
        let bytes = (n + 7) / 8;
        let mut lhs_v = vec![0u8; bytes];
        for i in 0..n {
            if (seed.rotate_left((i as u32) ^ 7) & 1) == 1 {
                lhs_v[i >> 3] |= 1u8 << (i & 7);
            }
        }
        let (device, mut queue) = device_and_queue();
        let (mut got_data, mut got_valid) = out_alloc(n);
        dispatch_cmp_i64_scalar(
            &device, &mut queue,
            &lhs, &lhs_v, scalar,
            n, op, &mut got_data, &mut got_valid,
        ).expect("dispatch ok");
        let (exp_data, exp_valid) = cpu_cmp_cs(&lhs, &lhs_v, scalar, op);
        for i in 0..n {
            let g = (got_data[i >> 3] >> (i & 7)) & 1;
            let e = (exp_data[i >> 3] >> (i & 7)) & 1;
            let gv = (got_valid[i >> 3] >> (i & 7)) & 1;
            let ev = (exp_valid[i >> 3] >> (i & 7)) & 1;
            prop_assert_eq!(gv, ev, "row {} validity (op={:?}, scalar={})", i, op, scalar);
            if ev == 1 {
                prop_assert_eq!(g, e, "row {} data (op={:?}, scalar={})", i, op, scalar);
            }
        }
    }
}
