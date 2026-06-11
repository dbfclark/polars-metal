// crates/polars-metal-kernels/tests/test_groupby_aggregate.rs
//
// Proptest + unit tests for the Phase 6 aggregation dispatchers.
//
// Coverage:
//   - GPU path (i32/u32/f32 sum/min/max/count/len) vs naive Rust reference.
//   - CPU-finalize path (i64/f64 sum/min/max/count/len) vs naive Rust reference.
//   - compute_mean_{f64,i64} unit tests.
//
// Null-bitmap convention: bit `i` of byte `i/8`, LSB-first (Arrow layout).
// The same convention is used by `_validity.metal` and all MSL kernels.
//
// Proptest strategy: 64 cases per property over varying cardinalities,
// null densities, and row counts.
#![allow(clippy::expect_used, clippy::panic, clippy::cast_possible_truncation)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::groupby::{
    aggregate_count_cpu, aggregate_len_cpu, aggregate_max_f64_cpu, aggregate_max_i64_cpu,
    aggregate_min_f64_cpu, aggregate_min_i64_cpu, aggregate_sum_f64_cpu, aggregate_sum_i64_cpu,
    compute_mean_f64, compute_mean_i64, dispatch_count_u32, dispatch_len_u32, dispatch_max_f32,
    dispatch_max_i32, dispatch_max_u32, dispatch_min_f32, dispatch_min_i32, dispatch_min_u32,
    dispatch_sum_f32, dispatch_sum_i32, dispatch_sum_u32,
};
use proptest::prelude::*;

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Pack a `Vec<bool>` validity into a bit-packed byte slice (Arrow layout).
fn pack_valid(valid: &[bool]) -> Vec<u8> {
    let n_bytes = (valid.len() + 7) / 8;
    let mut bytes = vec![0u8; n_bytes.max(1)];
    for (i, &v) in valid.iter().enumerate() {
        if v {
            bytes[i >> 3] |= 1 << (i & 7);
        }
    }
    bytes
}

/// Decode bit-packed byte slice back to Vec<bool>.
fn unpack_valid(bytes: &[u8], n: usize) -> Vec<bool> {
    (0..n)
        .map(|i| (bytes[i >> 3] >> (i & 7)) & 1 == 1)
        .collect()
}

// -----------------------------------------------------------------------
// Naive Rust reference implementations
// -----------------------------------------------------------------------

fn ref_sum_i32(values: &[i32], valid: &[bool], r2g: &[u32], n_groups: usize) -> Vec<i32> {
    let mut acc = vec![0i32; n_groups];
    for i in 0..values.len() {
        if valid[i] {
            acc[r2g[i] as usize] = acc[r2g[i] as usize].wrapping_add(values[i]);
        }
    }
    acc
}

fn ref_min_i32(values: &[i32], valid: &[bool], r2g: &[u32], n_groups: usize) -> Vec<i32> {
    let mut acc = vec![i32::MAX; n_groups];
    for i in 0..values.len() {
        if valid[i] {
            let g = r2g[i] as usize;
            acc[g] = acc[g].min(values[i]);
        }
    }
    acc
}

fn ref_max_i32(values: &[i32], valid: &[bool], r2g: &[u32], n_groups: usize) -> Vec<i32> {
    let mut acc = vec![i32::MIN; n_groups];
    for i in 0..values.len() {
        if valid[i] {
            let g = r2g[i] as usize;
            acc[g] = acc[g].max(values[i]);
        }
    }
    acc
}

fn ref_sum_u32(values: &[u32], valid: &[bool], r2g: &[u32], n_groups: usize) -> Vec<u32> {
    let mut acc = vec![0u32; n_groups];
    for i in 0..values.len() {
        if valid[i] {
            acc[r2g[i] as usize] = acc[r2g[i] as usize].wrapping_add(values[i]);
        }
    }
    acc
}

fn ref_min_u32(values: &[u32], valid: &[bool], r2g: &[u32], n_groups: usize) -> Vec<u32> {
    let mut acc = vec![u32::MAX; n_groups];
    for i in 0..values.len() {
        if valid[i] {
            let g = r2g[i] as usize;
            acc[g] = acc[g].min(values[i]);
        }
    }
    acc
}

fn ref_max_u32(values: &[u32], valid: &[bool], r2g: &[u32], n_groups: usize) -> Vec<u32> {
    let mut acc = vec![0u32; n_groups];
    for i in 0..values.len() {
        if valid[i] {
            let g = r2g[i] as usize;
            acc[g] = acc[g].max(values[i]);
        }
    }
    acc
}

fn ref_sum_f32(values: &[f32], valid: &[bool], r2g: &[u32], n_groups: usize) -> Vec<f32> {
    let mut acc = vec![0f32; n_groups];
    for i in 0..values.len() {
        if valid[i] {
            acc[r2g[i] as usize] += values[i];
        }
    }
    acc
}

fn ref_min_f32(values: &[f32], valid: &[bool], r2g: &[u32], n_groups: usize) -> Vec<f32> {
    let mut acc = vec![f32::INFINITY; n_groups];
    for i in 0..values.len() {
        if valid[i] {
            let g = r2g[i] as usize;
            if values[i] < acc[g] {
                acc[g] = values[i];
            }
        }
    }
    acc
}

fn ref_max_f32(values: &[f32], valid: &[bool], r2g: &[u32], n_groups: usize) -> Vec<f32> {
    let mut acc = vec![f32::NEG_INFINITY; n_groups];
    for i in 0..values.len() {
        if valid[i] {
            let g = r2g[i] as usize;
            if values[i] > acc[g] {
                acc[g] = values[i];
            }
        }
    }
    acc
}

// CPU-path references for 64-bit.
fn ref_sum_i64(values: &[i64], valid: &[bool], r2g: &[u32], n_groups: usize) -> Vec<i64> {
    let mut acc = vec![0i64; n_groups];
    for i in 0..values.len() {
        if valid[i] {
            acc[r2g[i] as usize] = acc[r2g[i] as usize].wrapping_add(values[i]);
        }
    }
    acc
}

fn ref_min_i64(
    values: &[i64],
    valid: &[bool],
    r2g: &[u32],
    n_groups: usize,
) -> (Vec<i64>, Vec<bool>) {
    let mut acc = vec![i64::MAX; n_groups];
    let mut has = vec![false; n_groups];
    for i in 0..values.len() {
        if valid[i] {
            let g = r2g[i] as usize;
            has[g] = true;
            acc[g] = acc[g].min(values[i]);
        }
    }
    (acc, has)
}

fn ref_max_i64(
    values: &[i64],
    valid: &[bool],
    r2g: &[u32],
    n_groups: usize,
) -> (Vec<i64>, Vec<bool>) {
    let mut acc = vec![i64::MIN; n_groups];
    let mut has = vec![false; n_groups];
    for i in 0..values.len() {
        if valid[i] {
            let g = r2g[i] as usize;
            has[g] = true;
            acc[g] = acc[g].max(values[i]);
        }
    }
    (acc, has)
}

fn ref_sum_f64(values: &[f64], valid: &[bool], r2g: &[u32], n_groups: usize) -> Vec<f64> {
    let mut acc = vec![0f64; n_groups];
    for i in 0..values.len() {
        if valid[i] {
            acc[r2g[i] as usize] += values[i];
        }
    }
    acc
}

fn ref_min_f64(
    values: &[f64],
    valid: &[bool],
    r2g: &[u32],
    n_groups: usize,
) -> (Vec<f64>, Vec<bool>) {
    let mut acc = vec![f64::INFINITY; n_groups];
    let mut has = vec![false; n_groups];
    for i in 0..values.len() {
        if valid[i] {
            let g = r2g[i] as usize;
            has[g] = true;
            if values[i].is_nan() {
                acc[g] = f64::NAN;
            } else if !acc[g].is_nan() && values[i] < acc[g] {
                acc[g] = values[i];
            }
        }
    }
    (acc, has)
}

fn ref_max_f64(
    values: &[f64],
    valid: &[bool],
    r2g: &[u32],
    n_groups: usize,
) -> (Vec<f64>, Vec<bool>) {
    let mut acc = vec![f64::NEG_INFINITY; n_groups];
    let mut has = vec![false; n_groups];
    for i in 0..values.len() {
        if valid[i] {
            let g = r2g[i] as usize;
            has[g] = true;
            if values[i].is_nan() {
                acc[g] = f64::NAN;
            } else if !acc[g].is_nan() && values[i] > acc[g] {
                acc[g] = values[i];
            }
        }
    }
    (acc, has)
}

fn ref_count_u64(valid: &[bool], r2g: &[u32], n_groups: usize) -> Vec<u64> {
    let mut acc = vec![0u64; n_groups];
    for i in 0..valid.len() {
        if valid[i] {
            acc[r2g[i] as usize] += 1;
        }
    }
    acc
}

fn ref_len_u64(r2g: &[u32], n_groups: usize) -> Vec<u64> {
    let mut acc = vec![0u64; n_groups];
    for &g in r2g {
        acc[g as usize] += 1;
    }
    acc
}

// -----------------------------------------------------------------------
// Compare f32 results by bit pattern (exact equality for finite values;
// NaN == NaN since both sides use the same seeded identity).
// -----------------------------------------------------------------------
fn f32_bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Compare f64 results: exact bit equality for non-NaN; both-NaN counts as equal.
fn f64_near_eq(a: f64, b: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    a.to_bits() == b.to_bits()
}

fn f64_vecs_near_eq(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| f64_near_eq(*x, *y))
}

// -----------------------------------------------------------------------
// GPU dispatch helpers (set up device + queue per call)
// -----------------------------------------------------------------------

fn device_and_queue() -> (MetalDevice, CommandQueue) {
    let device = MetalDevice::system_default().expect("Metal-capable hardware required");
    let queue = CommandQueue::new(&device).expect("queue creation");
    (device, queue)
}

// -----------------------------------------------------------------------
// Fixed-shape unit tests
// -----------------------------------------------------------------------

#[test]
fn gpu_sum_i32_four_groups() {
    let (device, mut queue) = device_and_queue();
    let values: Vec<i32> = vec![10, 20, 30, 40, -1, 5, 7, 8];
    let valid_bits: Vec<bool> = vec![true, true, true, true, false, true, true, true];
    let valid = pack_valid(&valid_bits);
    let r2g: Vec<u32> = vec![0, 1, 2, 3, 0, 1, 2, 3];
    let n_groups = 4;
    let mut out = vec![0i32; n_groups];
    dispatch_sum_i32(
        &device, &mut queue, &values, &valid, &r2g, 8, n_groups, &mut out,
    )
    .expect("dispatch_sum_i32");
    let expected = ref_sum_i32(&values, &valid_bits, &r2g, n_groups);
    assert_eq!(out, expected);
}

#[test]
fn gpu_sum_i32_all_nulls_gives_zero() {
    let (device, mut queue) = device_and_queue();
    let values: Vec<i32> = vec![100, 200, 300];
    let valid_bits: Vec<bool> = vec![false, false, false];
    let valid = pack_valid(&valid_bits);
    let r2g: Vec<u32> = vec![0, 0, 0];
    let mut out = vec![0i32; 1];
    dispatch_sum_i32(&device, &mut queue, &values, &valid, &r2g, 3, 1, &mut out)
        .expect("dispatch_sum_i32 all-null");
    assert_eq!(out[0], 0, "all-null group sum must be 0");
}

#[test]
fn gpu_count_and_len_differ_for_nulls() {
    let (device, mut queue) = device_and_queue();
    let valid_bits: Vec<bool> = vec![true, false, true, true, false];
    let valid = pack_valid(&valid_bits);
    let r2g: Vec<u32> = vec![0, 0, 0, 1, 1];
    let n_groups = 2;
    let mut counts = vec![0u32; n_groups];
    let mut lens = vec![0u32; n_groups];
    dispatch_count_u32(&device, &mut queue, &valid, &r2g, 5, n_groups, &mut counts)
        .expect("dispatch_count_u32");
    dispatch_len_u32(&device, &mut queue, &r2g, 5, n_groups, &mut lens).expect("dispatch_len_u32");
    assert_eq!(counts, vec![2, 1]); // group 0: 2 valid, group 1: 1 valid
    assert_eq!(lens, vec![3, 2]); // group 0: 3 rows, group 1: 2 rows
}

#[test]
fn gpu_min_max_f32_q1_shape() {
    let (device, mut queue) = device_and_queue();
    // 100 rows, 4 groups
    let n_rows = 100;
    let n_groups = 4;
    let values: Vec<f32> = (0..n_rows).map(|i| (i as f32) * 1.5 - 37.0).collect();
    let valid_bits: Vec<bool> = (0..n_rows).map(|i| i % 7 != 0).collect();
    let valid = pack_valid(&valid_bits);
    let r2g: Vec<u32> = (0..n_rows as u32).map(|i| i % n_groups as u32).collect();

    let mut min_out = vec![0f32; n_groups];
    let mut max_out = vec![0f32; n_groups];
    dispatch_min_f32(
        &device,
        &mut queue,
        &values,
        &valid,
        &r2g,
        n_rows,
        n_groups,
        &mut min_out,
    )
    .expect("dispatch_min_f32");
    dispatch_max_f32(
        &device,
        &mut queue,
        &values,
        &valid,
        &r2g,
        n_rows,
        n_groups,
        &mut max_out,
    )
    .expect("dispatch_max_f32");

    let ref_min = ref_min_f32(&values, &valid_bits, &r2g, n_groups);
    let ref_max = ref_max_f32(&values, &valid_bits, &r2g, n_groups);
    assert!(
        f32_bits_eq(&min_out, &ref_min),
        "min: {min_out:?} != {ref_min:?}"
    );
    assert!(
        f32_bits_eq(&max_out, &ref_max),
        "max: {max_out:?} != {ref_max:?}"
    );
}

#[test]
fn gpu_sum_i32_empty_input_is_noop() {
    let (device, mut queue) = device_and_queue();
    let mut out = vec![42i32; 2]; // pre-filled sentinel
    dispatch_sum_i32(&device, &mut queue, &[], &[], &[], 0, 2, &mut out).expect("empty dispatch");
    // Should be unchanged since n_rows == 0.
    assert_eq!(out, vec![42, 42]);
}

// -----------------------------------------------------------------------
// CPU-finalize unit tests
// -----------------------------------------------------------------------

#[test]
fn cpu_sum_i64_basic() {
    let values: Vec<i64> = vec![1, 2, 3, 4, 5, 6];
    let valid_bits: Vec<bool> = vec![true, true, false, true, true, false];
    let valid = pack_valid(&valid_bits);
    let r2g: Vec<u32> = vec![0, 0, 0, 1, 1, 1];
    let result = aggregate_sum_i64_cpu(&values, &valid, &r2g, 2);
    // group 0: 1+2 = 3 (row 2 is null); group 1: 4+5 = 9 (row 5 is null)
    let expected = ref_sum_i64(&values, &valid_bits, &r2g, 2);
    assert_eq!(result, expected);
}

#[test]
fn cpu_min_i64_all_null_group_has_no_valid() {
    let values: Vec<i64> = vec![100, 200];
    let valid_bits: Vec<bool> = vec![false, false];
    let valid = pack_valid(&valid_bits);
    let r2g: Vec<u32> = vec![0, 0];
    let (vals, has_value) = aggregate_min_i64_cpu(&values, &valid, &r2g, 1);
    let (ref_vals, ref_has) = ref_min_i64(&values, &valid_bits, &r2g, 1);
    assert_eq!(has_value, ref_has);
    assert_eq!(vals, ref_vals);
    assert!(!has_value[0], "all-null group must report has_value=false");
}

#[test]
fn cpu_max_i64_single_group() {
    let values: Vec<i64> = vec![-5, 100, -1000, 42, 0];
    let valid_bits: Vec<bool> = vec![true, true, false, true, true];
    let valid = pack_valid(&valid_bits);
    let r2g: Vec<u32> = vec![0, 0, 0, 0, 0];
    let (vals, has_value) = aggregate_max_i64_cpu(&values, &valid, &r2g, 1);
    assert!(has_value[0]);
    assert_eq!(vals[0], 100);
}

#[test]
fn cpu_sum_f64_basic() {
    let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let valid_bits: Vec<bool> = vec![true, false, true, true];
    let valid = pack_valid(&valid_bits);
    let r2g: Vec<u32> = vec![0, 0, 1, 1];
    let result = aggregate_sum_f64_cpu(&values, &valid, &r2g, 2);
    let expected = ref_sum_f64(&values, &valid_bits, &r2g, 2);
    assert!(
        f64_vecs_near_eq(&result, &expected),
        "{result:?} != {expected:?}"
    );
}

#[test]
fn cpu_min_f64_nan_poisoning() {
    let values: Vec<f64> = vec![1.0, f64::NAN, 3.0];
    let valid_bits: Vec<bool> = vec![true, true, true];
    let valid = pack_valid(&valid_bits);
    let r2g: Vec<u32> = vec![0, 0, 0];
    let (vals, has_value) = aggregate_min_f64_cpu(&values, &valid, &r2g, 1);
    assert!(has_value[0]);
    // NaN in group → result is NaN (poisoning).
    assert!(
        vals[0].is_nan(),
        "NaN poisoning: expected NaN, got {}",
        vals[0]
    );
}

#[test]
fn cpu_max_f64_nan_poisoning() {
    let values: Vec<f64> = vec![10.0, f64::NAN, 5.0];
    let valid_bits: Vec<bool> = vec![true, true, true];
    let valid = pack_valid(&valid_bits);
    let r2g: Vec<u32> = vec![0, 0, 0];
    let (vals, has_value) = aggregate_max_f64_cpu(&values, &valid, &r2g, 1);
    assert!(has_value[0]);
    assert!(
        vals[0].is_nan(),
        "NaN poisoning: expected NaN, got {}",
        vals[0]
    );
}

#[test]
fn cpu_count_and_len_cpu() {
    let valid_bits: Vec<bool> = vec![true, false, true, true, false];
    let valid = pack_valid(&valid_bits);
    let r2g: Vec<u32> = vec![0, 0, 0, 1, 1];
    let counts = aggregate_count_cpu(&valid, &r2g, 2);
    let lens = aggregate_len_cpu(&r2g, 2);
    let ref_c = ref_count_u64(&valid_bits, &r2g, 2);
    let ref_l = ref_len_u64(&r2g, 2);
    assert_eq!(counts, ref_c);
    assert_eq!(lens, ref_l);
}

// -----------------------------------------------------------------------
// compute_mean unit tests (T25)
// -----------------------------------------------------------------------

#[test]
fn compute_mean_f64_handles_empty_group() {
    let m = compute_mean_f64(&[10.0, 0.0], &[2, 0]);
    assert_eq!(m, vec![Some(5.0), None]);
}

#[test]
fn compute_mean_i64_returns_f64_with_division() {
    let m = compute_mean_i64(&[10, 7], &[4, 2]);
    assert_eq!(m, vec![Some(2.5), Some(3.5)]);
}

#[test]
fn compute_mean_all_null_groups() {
    let m = compute_mean_f64(&[0.0, 0.0, 0.0], &[0, 0, 0]);
    assert_eq!(m, vec![None, None, None]);
}

#[test]
fn compute_mean_single_row_per_group() {
    let m = compute_mean_i64(&[3, -6, 0], &[1, 1, 1]);
    assert_eq!(m, vec![Some(3.0), Some(-6.0), Some(0.0)]);
}

// -----------------------------------------------------------------------
// Proptest strategies
// -----------------------------------------------------------------------

/// Generate (values: Vec<i32>, valid: Vec<bool>, row_to_group: Vec<u32>, n_groups: usize)
/// for a given max cardinality and row count range.
fn input_strategy_i32(
    max_groups: usize,
    row_range: std::ops::Range<usize>,
) -> impl Strategy<Value = (Vec<i32>, Vec<bool>, Vec<u32>, usize)> {
    (1usize..=max_groups, row_range).prop_flat_map(|(n_groups, n_rows)| {
        let vals = prop::collection::vec(any::<i32>(), n_rows);
        let valid = prop::collection::vec(any::<bool>(), n_rows);
        let r2g = prop::collection::vec(0u32..(n_groups as u32), n_rows);
        (vals, valid, r2g, Just(n_groups))
    })
}

fn input_strategy_u32(
    max_groups: usize,
    row_range: std::ops::Range<usize>,
) -> impl Strategy<Value = (Vec<u32>, Vec<bool>, Vec<u32>, usize)> {
    (1usize..=max_groups, row_range).prop_flat_map(|(n_groups, n_rows)| {
        let vals = prop::collection::vec(any::<u32>(), n_rows);
        let valid = prop::collection::vec(any::<bool>(), n_rows);
        let r2g = prop::collection::vec(0u32..(n_groups as u32), n_rows);
        (vals, valid, r2g, Just(n_groups))
    })
}

fn input_strategy_f32(
    max_groups: usize,
    row_range: std::ops::Range<usize>,
) -> impl Strategy<Value = (Vec<f32>, Vec<bool>, Vec<u32>, usize)> {
    (1usize..=max_groups, row_range).prop_flat_map(|(n_groups, n_rows)| {
        // Avoid NaN/Inf in f32 tests (GPU CAS loop sum is not NaN-ordered).
        let vals = prop::collection::vec(-1e6f32..1e6f32, n_rows);
        let valid = prop::collection::vec(any::<bool>(), n_rows);
        let r2g = prop::collection::vec(0u32..(n_groups as u32), n_rows);
        (vals, valid, r2g, Just(n_groups))
    })
}

fn input_strategy_i64(
    max_groups: usize,
    row_range: std::ops::Range<usize>,
) -> impl Strategy<Value = (Vec<i64>, Vec<bool>, Vec<u32>, usize)> {
    (1usize..=max_groups, row_range).prop_flat_map(|(n_groups, n_rows)| {
        let vals = prop::collection::vec(any::<i64>(), n_rows);
        let valid = prop::collection::vec(any::<bool>(), n_rows);
        let r2g = prop::collection::vec(0u32..(n_groups as u32), n_rows);
        (vals, valid, r2g, Just(n_groups))
    })
}

fn input_strategy_f64(
    max_groups: usize,
    row_range: std::ops::Range<usize>,
) -> impl Strategy<Value = (Vec<f64>, Vec<bool>, Vec<u32>, usize)> {
    (1usize..=max_groups, row_range).prop_flat_map(|(n_groups, n_rows)| {
        // Avoid NaN in generic f64 sum/min/max tests for simplicity;
        // NaN poisoning is covered by dedicated unit tests above.
        let vals = prop::collection::vec(-1e12f64..1e12f64, n_rows);
        let valid = prop::collection::vec(any::<bool>(), n_rows);
        let r2g = prop::collection::vec(0u32..(n_groups as u32), n_rows);
        (vals, valid, r2g, Just(n_groups))
    })
}

// -----------------------------------------------------------------------
// GPU proptests
// -----------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn prop_gpu_sum_i32(
        (values, valid_bools, r2g, n_groups) in input_strategy_i32(16, 1..512),
    ) {
        let (device, mut queue) = device_and_queue();
        let valid = pack_valid(&valid_bools);
        let n = values.len();
        let mut out = vec![0i32; n_groups];
        dispatch_sum_i32(&device, &mut queue, &values, &valid, &r2g, n, n_groups, &mut out)
            .expect("dispatch_sum_i32");
        let expected = ref_sum_i32(&values, &valid_bools, &r2g, n_groups);
        prop_assert_eq!(out, expected);
    }

    #[test]
    fn prop_gpu_min_i32(
        (values, valid_bools, r2g, n_groups) in input_strategy_i32(16, 1..512),
    ) {
        let (device, mut queue) = device_and_queue();
        let valid = pack_valid(&valid_bools);
        let n = values.len();
        let mut out = vec![0i32; n_groups];
        dispatch_min_i32(&device, &mut queue, &values, &valid, &r2g, n, n_groups, &mut out)
            .expect("dispatch_min_i32");
        let expected = ref_min_i32(&values, &valid_bools, &r2g, n_groups);
        // All-null groups retain i32::MAX (identity). Reference also returns i32::MAX for them.
        prop_assert_eq!(out, expected);
    }

    #[test]
    fn prop_gpu_max_i32(
        (values, valid_bools, r2g, n_groups) in input_strategy_i32(16, 1..512),
    ) {
        let (device, mut queue) = device_and_queue();
        let valid = pack_valid(&valid_bools);
        let n = values.len();
        let mut out = vec![0i32; n_groups];
        dispatch_max_i32(&device, &mut queue, &values, &valid, &r2g, n, n_groups, &mut out)
            .expect("dispatch_max_i32");
        let expected = ref_max_i32(&values, &valid_bools, &r2g, n_groups);
        prop_assert_eq!(out, expected);
    }

    #[test]
    fn prop_gpu_sum_u32(
        (values, valid_bools, r2g, n_groups) in input_strategy_u32(16, 1..512),
    ) {
        let (device, mut queue) = device_and_queue();
        let valid = pack_valid(&valid_bools);
        let n = values.len();
        let mut out = vec![0u32; n_groups];
        dispatch_sum_u32(&device, &mut queue, &values, &valid, &r2g, n, n_groups, &mut out)
            .expect("dispatch_sum_u32");
        let expected = ref_sum_u32(&values, &valid_bools, &r2g, n_groups);
        prop_assert_eq!(out, expected);
    }

    #[test]
    fn prop_gpu_min_u32(
        (values, valid_bools, r2g, n_groups) in input_strategy_u32(16, 1..512),
    ) {
        let (device, mut queue) = device_and_queue();
        let valid = pack_valid(&valid_bools);
        let n = values.len();
        let mut out = vec![0u32; n_groups];
        dispatch_min_u32(&device, &mut queue, &values, &valid, &r2g, n, n_groups, &mut out)
            .expect("dispatch_min_u32");
        let expected = ref_min_u32(&values, &valid_bools, &r2g, n_groups);
        prop_assert_eq!(out, expected);
    }

    #[test]
    fn prop_gpu_max_u32(
        (values, valid_bools, r2g, n_groups) in input_strategy_u32(16, 1..512),
    ) {
        let (device, mut queue) = device_and_queue();
        let valid = pack_valid(&valid_bools);
        let n = values.len();
        let mut out = vec![0u32; n_groups];
        dispatch_max_u32(&device, &mut queue, &values, &valid, &r2g, n, n_groups, &mut out)
            .expect("dispatch_max_u32");
        let expected = ref_max_u32(&values, &valid_bools, &r2g, n_groups);
        prop_assert_eq!(out, expected);
    }

    #[test]
    fn prop_gpu_sum_f32(
        (values, valid_bools, r2g, n_groups) in input_strategy_f32(16, 1..256),
    ) {
        let (device, mut queue) = device_and_queue();
        let valid = pack_valid(&valid_bools);
        let n = values.len();
        let mut out = vec![0f32; n_groups];
        dispatch_sum_f32(&device, &mut queue, &values, &valid, &r2g, n, n_groups, &mut out)
            .expect("dispatch_sum_f32");
        let expected = ref_sum_f32(&values, &valid_bools, &r2g, n_groups);
        // The GPU f32 sum uses a CAS-loop in parallel (one thread per row), so the
        // accumulation order is non-deterministic. Both the GPU result and the sequential
        // reference are valid f32 summations of the same multiset, so each lies within the
        // standard backward-error bound γ_{n-1}·Σ|xᵢ| of the true sum; their difference is
        // therefore bounded by ~2(n-1)·ε·Σ|xᵢ|. We assert that ABSOLUTE bound rather than a
        // relative-to-result one: under catastrophic cancellation (large summands that nearly
        // cancel) the relative-to-result error is unbounded — it scales with the condition
        // number Σ|xᵢ|/|Σxᵢ| — even though every individual addition is correct to f32 ULP.
        // A real kernel bug (dropped/double-counted/misrouted value) injects error of order
        // |x| ≈ Σ|x|, which is ~4 orders of magnitude above this tolerance, so it stays caught.
        for g in 0..n_groups {
            let mut abssum = 0f32;
            let mut count = 0u32;
            for i in 0..n {
                if valid_bools[i] && r2g[i] == g as u32 {
                    abssum += values[i].abs();
                    count += 1;
                }
            }
            if count == 0 {
                // Truly empty group → identity 0.0, bit-exact.
                prop_assert_eq!(out[g].to_bits(), 0u32, "empty group g={} must be 0.0", g);
            } else {
                let tol = 8.0 * count as f32 * f32::EPSILON * abssum + f32::MIN_POSITIVE;
                let abs_err = (out[g] - expected[g]).abs();
                prop_assert!(
                    abs_err <= tol,
                    "g={}: out={} expected={} abs_err={} tol={} (n={}, sum|x|={})",
                    g, out[g], expected[g], abs_err, tol, count, abssum
                );
            }
        }
    }

    #[test]
    fn prop_gpu_min_f32(
        (values, valid_bools, r2g, n_groups) in input_strategy_f32(16, 1..256),
    ) {
        let (device, mut queue) = device_and_queue();
        let valid = pack_valid(&valid_bools);
        let n = values.len();
        let mut out = vec![0f32; n_groups];
        dispatch_min_f32(&device, &mut queue, &values, &valid, &r2g, n, n_groups, &mut out)
            .expect("dispatch_min_f32");
        let expected = ref_min_f32(&values, &valid_bools, &r2g, n_groups);
        prop_assert!(f32_bits_eq(&out, &expected), "min_f32: {out:?} != {expected:?}");
    }

    #[test]
    fn prop_gpu_max_f32(
        (values, valid_bools, r2g, n_groups) in input_strategy_f32(16, 1..256),
    ) {
        let (device, mut queue) = device_and_queue();
        let valid = pack_valid(&valid_bools);
        let n = values.len();
        let mut out = vec![0f32; n_groups];
        dispatch_max_f32(&device, &mut queue, &values, &valid, &r2g, n, n_groups, &mut out)
            .expect("dispatch_max_f32");
        let expected = ref_max_f32(&values, &valid_bools, &r2g, n_groups);
        prop_assert!(f32_bits_eq(&out, &expected), "max_f32: {out:?} != {expected:?}");
    }
}

// -----------------------------------------------------------------------
// CPU-finalize proptests
// -----------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn prop_cpu_sum_i64(
        (values, valid_bools, r2g, n_groups) in input_strategy_i64(16, 1..1024),
    ) {
        let valid = pack_valid(&valid_bools);
        let result = aggregate_sum_i64_cpu(&values, &valid, &r2g, n_groups);
        let expected = ref_sum_i64(&values, &valid_bools, &r2g, n_groups);
        prop_assert_eq!(result, expected);
    }

    #[test]
    fn prop_cpu_min_i64(
        (values, valid_bools, r2g, n_groups) in input_strategy_i64(16, 1..1024),
    ) {
        let valid = pack_valid(&valid_bools);
        let (vals, has_v) = aggregate_min_i64_cpu(&values, &valid, &r2g, n_groups);
        let (ref_vals, ref_has) = ref_min_i64(&values, &valid_bools, &r2g, n_groups);
        prop_assert_eq!(&has_v, &ref_has);
        // Only compare values for groups that have valid rows.
        for g in 0..n_groups {
            if ref_has[g] {
                prop_assert_eq!(vals[g], ref_vals[g], "min_i64 mismatch at group {}", g);
            }
        }
    }

    #[test]
    fn prop_cpu_max_i64(
        (values, valid_bools, r2g, n_groups) in input_strategy_i64(16, 1..1024),
    ) {
        let valid = pack_valid(&valid_bools);
        let (vals, has_v) = aggregate_max_i64_cpu(&values, &valid, &r2g, n_groups);
        let (ref_vals, ref_has) = ref_max_i64(&values, &valid_bools, &r2g, n_groups);
        prop_assert_eq!(&has_v, &ref_has);
        for g in 0..n_groups {
            if ref_has[g] {
                prop_assert_eq!(vals[g], ref_vals[g], "max_i64 mismatch at group {}", g);
            }
        }
    }

    #[test]
    fn prop_cpu_sum_f64(
        (values, valid_bools, r2g, n_groups) in input_strategy_f64(16, 1..1024),
    ) {
        let valid = pack_valid(&valid_bools);
        let result = aggregate_sum_f64_cpu(&values, &valid, &r2g, n_groups);
        let expected = ref_sum_f64(&values, &valid_bools, &r2g, n_groups);
        // The Rayon parallel CAS-loop accumulates f64 values in non-deterministic
        // order (threads race to CAS). This is not bit-exact against the sequential
        // reference but should be within a small relative tolerance. We verify:
        //   - empty groups give exactly 0.0 in both.
        //   - non-empty groups agree within 1e-9 relative error (numerical).
        for g in 0..n_groups {
            let r = result[g];
            let e = expected[g];
            if e == 0.0 {
                prop_assert_eq!(r.to_bits(), e.to_bits(), "sum_f64 g={}: zero mismatch", g);
            } else {
                let rel = (r - e).abs() / e.abs().max(1e-300);
                prop_assert!(
                    rel < 1e-9,
                    "sum_f64 g={}: {} != {} rel_err={}",
                    g, r, e, rel
                );
            }
        }
    }

    #[test]
    fn prop_cpu_min_f64(
        (values, valid_bools, r2g, n_groups) in input_strategy_f64(16, 1..1024),
    ) {
        let valid = pack_valid(&valid_bools);
        let (vals, has_v) = aggregate_min_f64_cpu(&values, &valid, &r2g, n_groups);
        let (ref_vals, ref_has) = ref_min_f64(&values, &valid_bools, &r2g, n_groups);
        prop_assert_eq!(&has_v, &ref_has);
        for g in 0..n_groups {
            if ref_has[g] {
                prop_assert!(
                    f64_near_eq(vals[g], ref_vals[g]),
                    "min_f64 mismatch at group {g}: {} != {}", vals[g], ref_vals[g]
                );
            }
        }
    }

    #[test]
    fn prop_cpu_max_f64(
        (values, valid_bools, r2g, n_groups) in input_strategy_f64(16, 1..1024),
    ) {
        let valid = pack_valid(&valid_bools);
        let (vals, has_v) = aggregate_max_f64_cpu(&values, &valid, &r2g, n_groups);
        let (ref_vals, ref_has) = ref_max_f64(&values, &valid_bools, &r2g, n_groups);
        prop_assert_eq!(&has_v, &ref_has);
        for g in 0..n_groups {
            if ref_has[g] {
                prop_assert!(
                    f64_near_eq(vals[g], ref_vals[g]),
                    "max_f64 mismatch at group {g}: {} != {}", vals[g], ref_vals[g]
                );
            }
        }
    }

    #[test]
    fn prop_cpu_count(
        (_, valid_bools, r2g, n_groups) in input_strategy_i64(16, 1..1024),
    ) {
        let valid = pack_valid(&valid_bools);
        let result = aggregate_count_cpu(&valid, &r2g, n_groups);
        let expected = ref_count_u64(&valid_bools, &r2g, n_groups);
        prop_assert_eq!(result, expected);
    }

    #[test]
    fn prop_cpu_len(
        (_, _, r2g, n_groups) in input_strategy_i64(16, 1..1024),
    ) {
        let result = aggregate_len_cpu(&r2g, n_groups);
        let expected = ref_len_u64(&r2g, n_groups);
        prop_assert_eq!(result, expected);
    }
}

// -----------------------------------------------------------------------
// Specific cardinality/density cases required by the spec
// -----------------------------------------------------------------------

#[test]
fn gpu_sum_i32_all_distinct_keys() {
    // n_groups = n_rows — each row is its own group
    let (device, mut queue) = device_and_queue();
    let n = 64;
    let values: Vec<i32> = (0..n as i32).collect();
    let valid_bits: Vec<bool> = (0..n).map(|i| i % 3 != 0).collect();
    let valid = pack_valid(&valid_bits);
    let r2g: Vec<u32> = (0..n as u32).collect();
    let mut out = vec![0i32; n];
    dispatch_sum_i32(&device, &mut queue, &values, &valid, &r2g, n, n, &mut out)
        .expect("all-distinct sum_i32");
    let expected = ref_sum_i32(&values, &valid_bits, &r2g, n);
    assert_eq!(out, expected);
}

#[test]
fn cpu_sum_i64_all_same_key() {
    let values: Vec<i64> = vec![1, 2, 3, 4, 5];
    let valid_bits: Vec<bool> = vec![true, true, true, true, false];
    let valid = pack_valid(&valid_bits);
    let r2g: Vec<u32> = vec![0, 0, 0, 0, 0];
    let result = aggregate_sum_i64_cpu(&values, &valid, &r2g, 1);
    // 1+2+3+4 = 10 (row 4 null)
    assert_eq!(result[0], 10);
}

#[test]
fn cpu_sum_i64_q1_shape_10k_rows() {
    let n = 10_000;
    let n_groups = 4;
    let values: Vec<i64> = (0..n as i64).collect();
    let valid_bits: Vec<bool> = (0..n).map(|i| i % 11 != 0).collect();
    let valid = pack_valid(&valid_bits);
    let r2g: Vec<u32> = (0..n as u32).map(|i| i % n_groups as u32).collect();

    let result = aggregate_sum_i64_cpu(&values, &valid, &r2g, n_groups);
    let expected = ref_sum_i64(&values, &valid_bits, &r2g, n_groups);
    assert_eq!(result, expected);
}

#[test]
fn cpu_sum_f64_100pct_null_density() {
    let values: Vec<f64> = vec![1.0, 2.0, 3.0];
    let valid_bits: Vec<bool> = vec![false, false, false];
    let valid = pack_valid(&valid_bits);
    let r2g: Vec<u32> = vec![0, 0, 0];
    let result = aggregate_sum_f64_cpu(&values, &valid, &r2g, 1);
    assert_eq!(result[0], 0.0, "all-null group sum must be 0.0");
}

/// Roundtrip the validity bitmap pack/unpack to confirm the helper is correct.
#[test]
fn valid_bitmap_roundtrip() {
    let original: Vec<bool> = vec![true, false, true, true, false, false, true, false, true];
    let packed = pack_valid(&original);
    let recovered = unpack_valid(&packed, original.len());
    assert_eq!(original, recovered);
}
