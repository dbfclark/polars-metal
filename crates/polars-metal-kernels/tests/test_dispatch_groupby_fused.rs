#![allow(clippy::expect_used, clippy::panic)]
//! End-to-end test for `dispatch_groupby_fused`: same shape of result
//! as `dispatch_groupby` across canonical Q1-shape aggs.
//!
//! Verifies correctness of the Task 15 wiring (kernel-side parallel
//! dispatcher that uses one fused MSL kernel per signature).

use std::collections::{BTreeMap, HashMap};

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::aggregate_fused::cache::FusedLibraryCache;
use polars_metal_kernels::aggregate_fused::signature::{
    AggExpr as KAggExpr, AggOp as KAggOp, AggSpec as KAggSpec, BinaryOp as KBinaryOp,
    MetalDtype as KMetalDtype,
};
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::groupby::{
    dispatch_groupby_fused, AggOutput, KeyColumn, KeyDtype, ValueColumn,
};

fn pack_valid(valid: &[bool]) -> Vec<u8> {
    let n_bytes = ((valid.len() + 7) / 8 + 3) & !3;
    let mut out = vec![0u8; n_bytes.max(4)];
    for (i, &b) in valid.iter().enumerate() {
        if b {
            out[i >> 3] |= 1 << (i & 7);
        }
    }
    out
}

fn setup() -> (MetalDevice, CommandQueue, FusedLibraryCache) {
    let device = MetalDevice::system_default().expect("Metal hardware required");
    let queue = CommandQueue::new(&device).expect("command queue");
    let cache = FusedLibraryCache::new(device.clone());
    (device, queue, cache)
}

/// CPU reference: group by k, sum + count + mean per group.
fn cpu_reference(keys: &[i32], values: &[f32], valid: &[bool]) -> BTreeMap<i32, (f32, u32, u32)> {
    let mut out: BTreeMap<i32, (f32, u32, u32)> = BTreeMap::new();
    for ((&k, &v), &is_valid) in keys.iter().zip(values.iter()).zip(valid.iter()) {
        let e = out.entry(k).or_insert((0.0f32, 0u32, 0u32));
        e.2 += 1; // length
        if is_valid {
            e.0 += v;
            e.1 += 1;
        }
    }
    out
}

#[test]
fn fused_groupby_sum_mean_count_len_f32_matches_cpu() {
    let (device, mut queue, cache) = setup();

    // 12 rows, 3 groups, mix of valid/null.
    let keys_i32: Vec<i32> = (0..12_i32).map(|i| i % 3).collect();
    let values_f32: Vec<f32> = (0..12).map(|i| (i as f32) + 0.5).collect();
    let valid_flags: Vec<bool> = (0..12).map(|i| i % 5 != 0).collect();
    let n_rows = keys_i32.len();

    let key_data_bytes: Vec<u8> = keys_i32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let key_valid_bytes = pack_valid(&vec![true; n_rows]);
    let val_data_bytes: Vec<u8> = values_f32.iter().flat_map(|v| v.to_le_bytes()).collect();
    let val_valid_bytes = pack_valid(&valid_flags);

    let key_cols = vec![KeyColumn {
        name: "k".into(),
        dtype: KeyDtype::I32,
        data: &key_data_bytes,
        valid: &key_valid_bytes,
        n_rows,
    }];

    // SAFETY: f32 is plain-old-data.
    let val_f32_typed: &[f32] =
        unsafe { std::slice::from_raw_parts(val_data_bytes.as_ptr() as *const f32, n_rows) };

    let mut value_columns: HashMap<String, ValueColumn<'_>> = HashMap::new();
    value_columns.insert(
        "v".into(),
        ValueColumn::F32 {
            data: val_f32_typed,
            valid: &val_valid_bytes,
        },
    );

    let aggs: Vec<KAggSpec> = vec![
        KAggSpec::Simple {
            input_col: "v".into(),
            op: KAggOp::Sum,
            output_alias: "sum_v".into(),
        },
        KAggSpec::Simple {
            input_col: "v".into(),
            op: KAggOp::Mean,
            output_alias: "mean_v".into(),
        },
        KAggSpec::Simple {
            input_col: "v".into(),
            op: KAggOp::Count,
            output_alias: "count_v".into(),
        },
        KAggSpec::Length {
            output_alias: "n".into(),
        },
    ];

    let result = dispatch_groupby_fused(
        &device,
        &mut queue,
        &cache,
        &key_cols,
        &aggs,
        &value_columns,
        n_rows,
    )
    .expect("fused dispatch must succeed");

    // Validate vs CPU reference.
    let cpu_ref = cpu_reference(&keys_i32, &values_f32, &valid_flags);
    assert_eq!(result.n_groups as usize, cpu_ref.len());

    // Build a map from each group's representative key → its (sum, count, len).
    use polars_metal_kernels::groupby::DecodedColumn;
    let mut by_key: BTreeMap<i32, (f32, f32, u64, u64)> = BTreeMap::new();
    for g in 0..(result.n_groups as usize) {
        let key = match &result.decoded_keys[0] {
            DecodedColumn::I32 { values, .. } => values[g],
            _ => panic!("unexpected key dtype"),
        };
        let sum = match &result.agg_outputs[0] {
            AggOutput::F32 { values, .. } => values[g],
            other => panic!("expected F32 sum, got {other:?}"),
        };
        let mean = match &result.agg_outputs[1] {
            AggOutput::F32 { values, .. } => values[g],
            other => panic!("expected F32 mean, got {other:?}"),
        };
        let count = match &result.agg_outputs[2] {
            AggOutput::U64 { values } => values[g],
            other => panic!("expected U64 count, got {other:?}"),
        };
        let len = match &result.agg_outputs[3] {
            AggOutput::U64 { values } => values[g],
            other => panic!("expected U64 len, got {other:?}"),
        };
        by_key.insert(key, (sum, mean, count, len));
    }

    for (&k, &(ref_sum, ref_count, ref_len)) in cpu_ref.iter() {
        let (got_sum, got_mean, got_count, got_len) = by_key
            .get(&k)
            .copied()
            .unwrap_or_else(|| panic!("missing group {k}"));
        assert!(
            (got_sum - ref_sum).abs() < 1e-3,
            "sum mismatch for k={k}: got {got_sum} vs ref {ref_sum}"
        );
        let ref_mean = if ref_count == 0 {
            0.0f32
        } else {
            ref_sum / ref_count as f32
        };
        if ref_count > 0 {
            assert!(
                (got_mean - ref_mean).abs() < 1e-3,
                "mean mismatch for k={k}: got {got_mean} vs ref {ref_mean}"
            );
        }
        assert_eq!(
            got_count, ref_count as u64,
            "count mismatch for k={k}: got {got_count} vs ref {ref_count}"
        );
        assert_eq!(
            got_len, ref_len as u64,
            "len mismatch for k={k}: got {got_len} vs ref {ref_len}"
        );
    }
}

#[test]
fn fused_groupby_expression_sum_a_times_b() {
    // Q1-shape: sum(a * b) per group. Must match `a_i * b_i` summed per group.
    let (device, mut queue, cache) = setup();

    let n_rows = 6_000;
    let keys: Vec<i32> = (0..n_rows).map(|i| (i % 4) as i32).collect();
    let a: Vec<f32> = (0..n_rows).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..n_rows).map(|i| (i as f32) * 0.05 + 1.0).collect();

    let key_data_bytes: Vec<u8> = keys.iter().flat_map(|v| v.to_le_bytes()).collect();
    let key_valid_bytes = pack_valid(&vec![true; n_rows]);
    let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
    let all_valid = pack_valid(&vec![true; n_rows]);

    let key_cols = vec![KeyColumn {
        name: "k".into(),
        dtype: KeyDtype::I32,
        data: &key_data_bytes,
        valid: &key_valid_bytes,
        n_rows,
    }];

    // SAFETY: f32 is plain-old-data.
    let a_typed: &[f32] =
        unsafe { std::slice::from_raw_parts(a_bytes.as_ptr() as *const f32, n_rows) };
    let b_typed: &[f32] =
        unsafe { std::slice::from_raw_parts(b_bytes.as_ptr() as *const f32, n_rows) };

    let mut value_columns: HashMap<String, ValueColumn<'_>> = HashMap::new();
    value_columns.insert(
        "a".into(),
        ValueColumn::F32 {
            data: a_typed,
            valid: &all_valid,
        },
    );
    value_columns.insert(
        "b".into(),
        ValueColumn::F32 {
            data: b_typed,
            valid: &all_valid,
        },
    );

    let aggs: Vec<KAggSpec> = vec![KAggSpec::Expression {
        expr: KAggExpr::Binary {
            op: KBinaryOp::Mul,
            lhs: Box::new(KAggExpr::Column("a".into())),
            rhs: Box::new(KAggExpr::Column("b".into())),
        },
        op: KAggOp::Sum,
        output_alias: "sum_ab".into(),
    }];

    let result = dispatch_groupby_fused(
        &device,
        &mut queue,
        &cache,
        &key_cols,
        &aggs,
        &value_columns,
        n_rows,
    )
    .expect("fused dispatch with expression must succeed");

    let _ = KMetalDtype::F32; // silence unused-import if all checks remain numeric
                              // CPU reference.
    let mut cpu_sums: BTreeMap<i32, f32> = BTreeMap::new();
    for ((&k, &av), &bv) in keys.iter().zip(a.iter()).zip(b.iter()) {
        *cpu_sums.entry(k).or_insert(0.0f32) += av * bv;
    }

    assert_eq!(result.n_groups as usize, cpu_sums.len());

    use polars_metal_kernels::groupby::DecodedColumn;
    let mut by_key: BTreeMap<i32, f32> = BTreeMap::new();
    for g in 0..(result.n_groups as usize) {
        let key = match &result.decoded_keys[0] {
            DecodedColumn::I32 { values, .. } => values[g],
            _ => panic!("unexpected key dtype"),
        };
        let sum = match &result.agg_outputs[0] {
            AggOutput::F32 { values, .. } => values[g],
            other => panic!("expected F32 sum, got {other:?}"),
        };
        by_key.insert(key, sum);
    }
    for (&k, &cpu_sum) in cpu_sums.iter() {
        let got = by_key
            .get(&k)
            .copied()
            .unwrap_or_else(|| panic!("missing group {k}"));
        // Atomic float CAS with floats accumulates in non-deterministic
        // order, so use a small relative tolerance.
        let rel = (got - cpu_sum).abs() / cpu_sum.abs().max(1.0);
        assert!(
            rel < 1e-3,
            "sum(a*b) mismatch for k={k}: got {got} vs cpu {cpu_sum} (rel={rel})"
        );
    }
}
