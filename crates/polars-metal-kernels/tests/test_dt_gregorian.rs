// crates/polars-metal-kernels/tests/test_dt_gregorian.rs
//
// Correctness tests for the gregorian civil-from-days kernel
// (`dt_field_from_days` in `shaders/dt_gregorian.metal`). Validates year /
// month / day extraction from Int32 days-since-1970 against a CPU Hinnant
// reference, across multi-tile inputs, pre-1970 negatives, leap/century
// boundaries, and n=0/1.
//
// Requires Metal-capable hardware; skips via `expect` on machines without a
// discoverable system-default MTLDevice.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::dt::{dispatch_dt_field, DtField};
use std::sync::Mutex;

static METAL_TEST_LOCK: Mutex<()> = Mutex::new(());

/// CPU reference: Howard Hinnant civil_from_days (days since 1970-01-01).
fn civil_from_days(z0: i64) -> (i32, i32, i32) {
    let z = z0 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0,399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0,365]
    let mp = (5 * doy + 2) / 153; // [0,11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1,31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1,12]
    ((y + if m <= 2 { 1 } else { 0 }) as i32, m as i32, d as i32)
}

fn run(days: &[i32], field: DtField) -> Vec<i32> {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let device = MetalDevice::system_default().expect("Metal-capable hardware required");
    let mut out = vec![0i32; days.len()];
    dispatch_dt_field(&device, days, &mut out, field).expect("dispatch succeeds");
    out
}

#[test]
fn dt_fields_match_reference_multitile_and_negatives() {
    // ~1500 dates crossing tile boundaries (TG_SIZE=256), incl. pre-1970.
    let days: Vec<i32> = (-25567..-25567 + 1500).collect(); // from 1900-01-01
    let more: Vec<i32> = (0..1500).map(|i| i * 37).collect(); // sparse forward
    for set in [days, more] {
        for (field, idx) in [(DtField::Year, 0), (DtField::Month, 1), (DtField::Day, 2)] {
            let got = run(&set, field);
            for (i, &z) in set.iter().enumerate() {
                let want = civil_from_days(z as i64);
                let w = [want.0, want.1, want.2][idx];
                assert_eq!(got[i], w, "field {field:?} z={z}: got {} want {w}", got[i]);
            }
        }
    }
}

#[test]
fn dt_leap_and_century_boundaries() {
    // 2000-02-29 (leap), 1900-02-28 then 1900-03-01 (NOT leap), 2020-12-31,
    // 2021-01-01, epoch 1970-01-01 (day 0).
    let cases: [(i32, (i32, i32, i32)); 6] = [
        (11016, (2000, 2, 29)),  // 2000-02-29
        (-25509, (1900, 2, 28)), // 1900-02-28
        (-25508, (1900, 3, 1)),  // 1900-03-01 (no Feb 29 in 1900)
        (18627, (2020, 12, 31)),
        (18628, (2021, 1, 1)),
        (0, (1970, 1, 1)),
    ];
    let days: Vec<i32> = cases.iter().map(|c| c.0).collect();
    let y = run(&days, DtField::Year);
    let m = run(&days, DtField::Month);
    let d = run(&days, DtField::Day);
    for (i, (_, (wy, wm, wd))) in cases.iter().enumerate() {
        assert_eq!((y[i], m[i], d[i]), (*wy, *wm, *wd), "case {i}");
    }
}

#[test]
fn dt_n0_is_noop_and_n1_works() {
    let device = MetalDevice::system_default().expect("Metal-capable hardware required");
    let mut empty: Vec<i32> = vec![];
    dispatch_dt_field(&device, &[], &mut empty, DtField::Year).expect("n=0 ok");
    assert!(empty.is_empty());
    let single = run(&[18336], DtField::Year); // 2020-03-15
    assert_eq!(single[0], 2020);
}
