// crates/polars-metal-kernels/tests/test_logical_bool.rs
//
// Correctness tests for the `bool_and` / `bool_or` kernels (Task 19).
//
// 3-valued logic on nullable bool columns, matching Polars CPU exactly:
//   AND: false dominates (false ∧ null = false, valid).
//   OR : true  dominates (true  ∨ null = true , valid).
// Result is null only when neither side can short-circuit to the dominating
// value AND at least one side is null.
//
// Inputs and outputs are bit-packed bool columns + bit-packed validity
// bitmaps (Arrow layout). Outputs use atomic OR because 8 rows share a
// byte; the dispatchers zero-initialise the device buffers.
//
// A process-wide `Mutex` serialises Metal tests across parallel `cargo
// test` workers (same pattern as `test_cmp_i64`).
#![allow(clippy::expect_used)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::logical::{dispatch_bool_and, dispatch_bool_or};
use proptest::prelude::*;
use std::sync::Mutex;

static METAL_TEST_LOCK: Mutex<()> = Mutex::new(());

fn device_and_queue() -> (MetalDevice, CommandQueue) {
    let device = MetalDevice::system_default().expect("Metal-capable hardware");
    let queue = CommandQueue::new(&device).expect("queue creation");
    (device, queue)
}

/// Minimum bytes for a bit-packed output bitmap (mirrors the dispatcher).
fn out_bytes(n: usize) -> usize {
    let raw = (n + 7) / 8;
    let padded = (raw + 3) & !3;
    padded.max(4)
}

fn out_alloc(n: usize) -> (Vec<u8>, Vec<u8>) {
    let b = out_bytes(n);
    (vec![0u8; b], vec![0u8; b])
}

/// Encode rows of `(data_bit, valid_bit)` into matching bit-packed buffers.
fn encode_rows(rows: &[(bool, bool)]) -> (Vec<u8>, Vec<u8>) {
    let n = rows.len();
    let bytes = (n + 7) / 8;
    let mut data = vec![0u8; bytes];
    let mut valid = vec![0u8; bytes];
    for (i, (d, v)) in rows.iter().enumerate() {
        if *d {
            data[i >> 3] |= 1u8 << (i & 7);
        }
        if *v {
            valid[i >> 3] |= 1u8 << (i & 7);
        }
    }
    (data, valid)
}

/// CPU reference for 3-valued AND (false dominates).
fn cpu_bool_and(
    lhs_d: &[u8],
    lhs_v: &[u8],
    rhs_d: &[u8],
    rhs_v: &[u8],
    n: usize,
) -> (Vec<u8>, Vec<u8>) {
    let b = out_bytes(n);
    let mut data = vec![0u8; b];
    let mut valid = vec![0u8; b];
    for i in 0..n {
        let ld = (lhs_d[i >> 3] >> (i & 7)) & 1 == 1;
        let lv = (lhs_v[i >> 3] >> (i & 7)) & 1 == 1;
        let rd = (rhs_d[i >> 3] >> (i & 7)) & 1 == 1;
        let rv = (rhs_v[i >> 3] >> (i & 7)) & 1 == 1;
        // false-and-valid on either side → false (valid). Both true-and-valid → true (valid).
        // Anything else with at least one null → null.
        let (out_d, out_v) = if (lv && !ld) || (rv && !rd) {
            (false, true)
        } else if lv && ld && rv && rd {
            (true, true)
        } else {
            (false, false)
        };
        if out_v {
            valid[i >> 3] |= 1u8 << (i & 7);
        }
        if out_d {
            data[i >> 3] |= 1u8 << (i & 7);
        }
    }
    (data, valid)
}

/// CPU reference for 3-valued OR (true dominates).
fn cpu_bool_or(
    lhs_d: &[u8],
    lhs_v: &[u8],
    rhs_d: &[u8],
    rhs_v: &[u8],
    n: usize,
) -> (Vec<u8>, Vec<u8>) {
    let b = out_bytes(n);
    let mut data = vec![0u8; b];
    let mut valid = vec![0u8; b];
    for i in 0..n {
        let ld = (lhs_d[i >> 3] >> (i & 7)) & 1 == 1;
        let lv = (lhs_v[i >> 3] >> (i & 7)) & 1 == 1;
        let rd = (rhs_d[i >> 3] >> (i & 7)) & 1 == 1;
        let rv = (rhs_v[i >> 3] >> (i & 7)) & 1 == 1;
        let (out_d, out_v) = if (lv && ld) || (rv && rd) {
            (true, true)
        } else if lv && !ld && rv && !rd {
            (false, true)
        } else {
            (false, false)
        };
        if out_v {
            valid[i >> 3] |= 1u8 << (i & 7);
        }
        if out_d {
            data[i >> 3] |= 1u8 << (i & 7);
        }
    }
    (data, valid)
}

#[test]
fn and_truth_table_exhaustive() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    // 9 row pairs covering every (lhs, rhs) in {T, F, null} × {T, F, null}.
    // For "null" we set both data and valid to false; the kernel must
    // ignore the data bit when valid=0.
    let lhs_rows = vec![
        (true, true),   // T
        (true, true),   // T
        (true, true),   // T
        (false, true),  // F
        (false, true),  // F
        (false, true),  // F
        (false, false), // null
        (false, false), // null
        (false, false), // null
    ];
    let rhs_rows = vec![
        (true, true),   // T
        (false, true),  // F
        (false, false), // null
        (true, true),   // T
        (false, true),  // F
        (false, false), // null
        (true, true),   // T
        (false, true),  // F
        (false, false), // null
    ];
    // AND truth table:
    //   T∧T = T(v), T∧F = F(v), T∧null = null
    //   F∧T = F(v), F∧F = F(v), F∧null = F(v)
    //   null∧T = null, null∧F = F(v), null∧null = null
    let expected_data = [true, false, false, false, false, false, false, false, false];
    let expected_valid = [true, true, false, true, true, true, false, true, false];

    let (lhs_d, lhs_v) = encode_rows(&lhs_rows);
    let (rhs_d, rhs_v) = encode_rows(&rhs_rows);
    let n = 9;
    let (mut got_data, mut got_valid) = out_alloc(n);
    let (device, mut queue) = device_and_queue();
    dispatch_bool_and(
        &device,
        &mut queue,
        &lhs_d,
        &lhs_v,
        &rhs_d,
        &rhs_v,
        n,
        &mut got_data,
        &mut got_valid,
    )
    .expect("dispatch ok");

    for i in 0..n {
        let g = (got_data[i >> 3] >> (i & 7)) & 1;
        let e = u8::from(expected_data[i]);
        let gv = (got_valid[i >> 3] >> (i & 7)) & 1;
        let ev = u8::from(expected_valid[i]);
        assert_eq!(
            gv, ev,
            "row {i} validity (lhs={:?}, rhs={:?})",
            lhs_rows[i], rhs_rows[i]
        );
        if ev == 1 {
            assert_eq!(
                g, e,
                "row {i} data (lhs={:?}, rhs={:?})",
                lhs_rows[i], rhs_rows[i]
            );
        }
    }
}

#[test]
fn or_truth_table_exhaustive() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let lhs_rows = vec![
        (true, true),
        (true, true),
        (true, true),
        (false, true),
        (false, true),
        (false, true),
        (false, false),
        (false, false),
        (false, false),
    ];
    let rhs_rows = vec![
        (true, true),
        (false, true),
        (false, false),
        (true, true),
        (false, true),
        (false, false),
        (true, true),
        (false, true),
        (false, false),
    ];
    // OR truth table:
    //   T∨T = T(v), T∨F = T(v), T∨null = T(v)
    //   F∨T = T(v), F∨F = F(v), F∨null = null
    //   null∨T = T(v), null∨F = null, null∨null = null
    let expected_data = [true, true, true, true, false, false, true, false, false];
    let expected_valid = [true, true, true, true, true, false, true, false, false];

    let (lhs_d, lhs_v) = encode_rows(&lhs_rows);
    let (rhs_d, rhs_v) = encode_rows(&rhs_rows);
    let n = 9;
    let (mut got_data, mut got_valid) = out_alloc(n);
    let (device, mut queue) = device_and_queue();
    dispatch_bool_or(
        &device,
        &mut queue,
        &lhs_d,
        &lhs_v,
        &rhs_d,
        &rhs_v,
        n,
        &mut got_data,
        &mut got_valid,
    )
    .expect("dispatch ok");

    for i in 0..n {
        let g = (got_data[i >> 3] >> (i & 7)) & 1;
        let e = u8::from(expected_data[i]);
        let gv = (got_valid[i >> 3] >> (i & 7)) & 1;
        let ev = u8::from(expected_valid[i]);
        assert_eq!(
            gv, ev,
            "row {i} validity (lhs={:?}, rhs={:?})",
            lhs_rows[i], rhs_rows[i]
        );
        if ev == 1 {
            assert_eq!(
                g, e,
                "row {i} data (lhs={:?}, rhs={:?})",
                lhs_rows[i], rhs_rows[i]
            );
        }
    }
}

#[test]
fn and_unaligned_row_count() {
    // 13 rows: kernel reads exactly 13, trailing bits of the second byte
    // must stay zero in both data and validity.
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let n = 13;
    // lhs: pattern of T/F/null repeated; rhs: pattern of T/null/F.
    let mut lhs_rows = Vec::with_capacity(n);
    let mut rhs_rows = Vec::with_capacity(n);
    for i in 0..n {
        lhs_rows.push(match i % 3 {
            0 => (true, true),
            1 => (false, true),
            _ => (false, false),
        });
        rhs_rows.push(match i % 3 {
            0 => (true, true),
            1 => (false, false),
            _ => (false, true),
        });
    }
    let (lhs_d, lhs_v) = encode_rows(&lhs_rows);
    let (rhs_d, rhs_v) = encode_rows(&rhs_rows);
    let (mut got_data, mut got_valid) = out_alloc(n);
    let (device, mut queue) = device_and_queue();
    dispatch_bool_and(
        &device,
        &mut queue,
        &lhs_d,
        &lhs_v,
        &rhs_d,
        &rhs_v,
        n,
        &mut got_data,
        &mut got_valid,
    )
    .expect("dispatch ok");
    let (exp_data, exp_valid) = cpu_bool_and(&lhs_d, &lhs_v, &rhs_d, &rhs_v, n);
    for i in 0..n {
        let g = (got_data[i >> 3] >> (i & 7)) & 1;
        let e = (exp_data[i >> 3] >> (i & 7)) & 1;
        let gv = (got_valid[i >> 3] >> (i & 7)) & 1;
        let ev = (exp_valid[i >> 3] >> (i & 7)) & 1;
        assert_eq!(gv, ev, "row {i} validity");
        if ev == 1 {
            assert_eq!(g, e, "row {i} data");
        }
    }
    // Trailing 3 bits in byte 1 (rows 13,14,15) must be zero.
    for r in n..16 {
        let g = (got_data[r >> 3] >> (r & 7)) & 1;
        let gv = (got_valid[r >> 3] >> (r & 7)) & 1;
        assert_eq!(g, 0, "padding row {r} data must be zero");
        assert_eq!(gv, 0, "padding row {r} validity must be zero");
    }
}

#[test]
fn or_multi_thread_same_byte_stress() {
    // 8 threads race ONE byte in both output buffers. With atomic OR all
    // 8 bits must be set; a non-atomic write would lose at least one.
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let n = 8;
    let lhs_rows = vec![(true, true); 8];
    let rhs_rows = vec![(false, true); 8];
    let (lhs_d, lhs_v) = encode_rows(&lhs_rows);
    let (rhs_d, rhs_v) = encode_rows(&rhs_rows);
    let (mut got_data, mut got_valid) = out_alloc(n);
    let (device, mut queue) = device_and_queue();
    dispatch_bool_or(
        &device,
        &mut queue,
        &lhs_d,
        &lhs_v,
        &rhs_d,
        &rhs_v,
        n,
        &mut got_data,
        &mut got_valid,
    )
    .expect("dispatch ok");
    // T∨F at every row → T (valid) at every row.
    assert_eq!(got_data[0], 0xFFu8, "all 8 data bits must be set");
    assert_eq!(got_valid[0], 0xFFu8, "all 8 validity bits must be set");
}

#[test]
fn n_rows_zero_is_no_op() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let empty: Vec<u8> = Vec::new();
    let (mut got_data, mut got_valid) = out_alloc(0);
    dispatch_bool_and(
        &device,
        &mut queue,
        &empty,
        &empty,
        &empty,
        &empty,
        0,
        &mut got_data,
        &mut got_valid,
    )
    .expect("zero rows is a no-op");
    assert!(got_data.iter().all(|&b| b == 0));
    assert!(got_valid.iter().all(|&b| b == 0));

    let (mut got_data, mut got_valid) = out_alloc(0);
    dispatch_bool_or(
        &device,
        &mut queue,
        &empty,
        &empty,
        &empty,
        &empty,
        0,
        &mut got_data,
        &mut got_valid,
    )
    .expect("zero rows is a no-op");
    assert!(got_data.iter().all(|&b| b == 0));
    assert!(got_valid.iter().all(|&b| b == 0));
}

#[test]
fn input_too_short_errors() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs_d = vec![0xFFu8]; // 1 byte
    let lhs_v = vec![0xFFu8];
    let rhs_d = vec![0xFFu8];
    let rhs_v = vec![0xFFu8];
    let n = 16; // needs 2 bytes per side
    let (mut got_data, mut got_valid) = out_alloc(n);
    let err = dispatch_bool_and(
        &device,
        &mut queue,
        &lhs_d,
        &lhs_v,
        &rhs_d,
        &rhs_v,
        n,
        &mut got_data,
        &mut got_valid,
    )
    .expect_err("undersized input must error");
    let msg = format!("{err}");
    assert!(msg.contains("input"), "got: {msg}");
}

#[test]
fn output_too_short_errors() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let (device, mut queue) = device_and_queue();
    let lhs_rows = vec![(true, true); 8];
    let rhs_rows = vec![(true, true); 8];
    let (lhs_d, lhs_v) = encode_rows(&lhs_rows);
    let (rhs_d, rhs_v) = encode_rows(&rhs_rows);
    let n = 8;
    let mut got_data = vec![0u8; 1]; // too small (need >= 4)
    let mut got_valid = vec![0u8; 4];
    let err = dispatch_bool_or(
        &device,
        &mut queue,
        &lhs_d,
        &lhs_v,
        &rhs_d,
        &rhs_v,
        n,
        &mut got_data,
        &mut got_valid,
    )
    .expect_err("undersized output must error");
    let msg = format!("{err}");
    assert!(msg.contains("output"), "got: {msg}");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn and_matches_cpu_ref(n in 8usize..256, seed in any::<u64>()) {
        let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let bytes = (n + 7) / 8;
        let mut lhs_d = vec![0u8; bytes];
        let mut lhs_v = vec![0u8; bytes];
        let mut rhs_d = vec![0u8; bytes];
        let mut rhs_v = vec![0u8; bytes];
        for r in 0..n {
            if (seed.rotate_left((r as u32) ^ 1) & 1) == 1 { lhs_d[r >> 3] |= 1u8 << (r & 7); }
            if (seed.rotate_left((r as u32) ^ 7) & 1) == 1 { lhs_v[r >> 3] |= 1u8 << (r & 7); }
            if (seed.rotate_left((r as u32) ^ 13) & 1) == 1 { rhs_d[r >> 3] |= 1u8 << (r & 7); }
            if (seed.rotate_left((r as u32) ^ 31) & 1) == 1 { rhs_v[r >> 3] |= 1u8 << (r & 7); }
        }
        let (device, mut queue) = device_and_queue();
        let (mut got_d, mut got_v) = out_alloc(n);
        dispatch_bool_and(
            &device, &mut queue,
            &lhs_d, &lhs_v, &rhs_d, &rhs_v,
            n, &mut got_d, &mut got_v,
        ).expect("dispatch ok");
        let (exp_d, exp_v) = cpu_bool_and(&lhs_d, &lhs_v, &rhs_d, &rhs_v, n);
        for i in 0..n {
            let g = (got_d[i >> 3] >> (i & 7)) & 1;
            let e = (exp_d[i >> 3] >> (i & 7)) & 1;
            let gv = (got_v[i >> 3] >> (i & 7)) & 1;
            let ev = (exp_v[i >> 3] >> (i & 7)) & 1;
            prop_assert_eq!(gv, ev, "row {} validity", i);
            if ev == 1 {
                prop_assert_eq!(g, e, "row {} data", i);
            }
        }
    }

    #[test]
    fn or_matches_cpu_ref(n in 8usize..256, seed in any::<u64>()) {
        let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let bytes = (n + 7) / 8;
        let mut lhs_d = vec![0u8; bytes];
        let mut lhs_v = vec![0u8; bytes];
        let mut rhs_d = vec![0u8; bytes];
        let mut rhs_v = vec![0u8; bytes];
        for r in 0..n {
            if (seed.rotate_left((r as u32) ^ 3) & 1) == 1 { lhs_d[r >> 3] |= 1u8 << (r & 7); }
            if (seed.rotate_left((r as u32) ^ 9) & 1) == 1 { lhs_v[r >> 3] |= 1u8 << (r & 7); }
            if (seed.rotate_left((r as u32) ^ 17) & 1) == 1 { rhs_d[r >> 3] |= 1u8 << (r & 7); }
            if (seed.rotate_left((r as u32) ^ 23) & 1) == 1 { rhs_v[r >> 3] |= 1u8 << (r & 7); }
        }
        let (device, mut queue) = device_and_queue();
        let (mut got_d, mut got_v) = out_alloc(n);
        dispatch_bool_or(
            &device, &mut queue,
            &lhs_d, &lhs_v, &rhs_d, &rhs_v,
            n, &mut got_d, &mut got_v,
        ).expect("dispatch ok");
        let (exp_d, exp_v) = cpu_bool_or(&lhs_d, &lhs_v, &rhs_d, &rhs_v, n);
        for i in 0..n {
            let g = (got_d[i >> 3] >> (i & 7)) & 1;
            let e = (exp_d[i >> 3] >> (i & 7)) & 1;
            let gv = (got_v[i >> 3] >> (i & 7)) & 1;
            let ev = (exp_v[i >> 3] >> (i & 7)) & 1;
            prop_assert_eq!(gv, ev, "row {} validity", i);
            if ev == 1 {
                prop_assert_eq!(g, e, "row {} data", i);
            }
        }
    }
}
