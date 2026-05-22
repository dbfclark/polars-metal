// crates/polars-metal-kernels/tests/test_groupby_pipeline.rs
//
// End-to-end proptest for dispatch_groupby.
// Verifies that the GPU pipeline (encode → hash → build → CPU aggregate → decode)
// produces the same result as a pure-Rust HashMap-based reference implementation,
// modulo group-id assignment order.
//
// Null-bitmap convention: bit `i` of byte `i/8`, LSB-first (Arrow layout).
#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::groupby::{
    dispatch_groupby, AggKind, AggOutput, AggRequest, DecodedColumn, KeyColumn, KeyDtype,
    ValueColumn,
};
use proptest::prelude::*;
use std::collections::BTreeMap;

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

fn bytes_i64(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Pack a `Vec<bool>` validity into a bit-packed byte slice (Arrow layout).
fn pack_valid(valid: &[bool]) -> Vec<u8> {
    let n_bytes = (valid.len() + 7) / 8;
    let mut out = vec![0u8; n_bytes.max(1)];
    for (i, &v) in valid.iter().enumerate() {
        if v {
            out[i >> 3] |= 1 << (i & 7);
        }
    }
    out
}

fn setup() -> (MetalDevice, CommandQueue) {
    let device = MetalDevice::system_default().expect("device");
    let queue = CommandQueue::new(&device).expect("queue");
    (device, queue)
}

// -----------------------------------------------------------------------
// Canonicalization
// -----------------------------------------------------------------------

/// Canonical key: one (valid, value) pair per key column (i64 always for our tests).
type KeyTuple = Vec<(bool, i64)>;

/// Canonical agg result: one entry per AggRequest, in order.
type AggTuple = Vec<AggOutputValue>;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum AggOutputValue {
    /// Stores raw i64 bit patterns so NaN-containing floats compare equal.
    I64 {
        valid: bool,
        value: i64,
    },
    /// Stores raw u64 bit patterns for exact f64 equality.
    F64 {
        valid: bool,
        value_bits: u64,
    },
    U64(u64),
}

fn canonicalize(
    decoded_keys: &[DecodedColumn],
    agg_outputs: &[AggOutput],
    n_groups: u32,
) -> BTreeMap<KeyTuple, AggTuple> {
    let mut out: BTreeMap<KeyTuple, AggTuple> = BTreeMap::new();
    for g in 0..(n_groups as usize) {
        let key_tuple: KeyTuple = decoded_keys
            .iter()
            .map(|dc| match dc {
                DecodedColumn::I64 { values, valid } => (valid[g], values[g]),
                DecodedColumn::F64 { values, valid } => (valid[g], values[g].to_bits() as i64),
                DecodedColumn::Bool { values, valid } => (valid[g], values[g] as i64),
            })
            .collect();
        let agg_tuple: AggTuple = agg_outputs
            .iter()
            .map(|ao| match ao {
                AggOutput::I64 { values, valid } => AggOutputValue::I64 {
                    valid: valid[g],
                    value: values[g],
                },
                AggOutput::F64 { values, valid } => AggOutputValue::F64 {
                    valid: valid[g],
                    value_bits: values[g].to_bits(),
                },
                AggOutput::U64 { values } => AggOutputValue::U64(values[g]),
            })
            .collect();
        out.insert(key_tuple, agg_tuple);
    }
    out
}

// -----------------------------------------------------------------------
// Pure-Rust reference groupby
// -----------------------------------------------------------------------

struct RefValI64<'a> {
    col_idx: usize,
    data: &'a [i64],
    valid: &'a [bool],
}

fn cpu_reference_groupby(
    keys: &[Vec<i64>],
    key_valids: &[Vec<bool>],
    val_i64: &[RefValI64<'_>],
    aggs: &[AggRequest],
    n_rows: usize,
) -> BTreeMap<KeyTuple, AggTuple> {
    use std::collections::HashMap;

    // Build groups: key_tuple → list of source row indices
    let mut groups: HashMap<KeyTuple, Vec<usize>> = HashMap::new();
    for r in 0..n_rows {
        let key: KeyTuple = (0..keys.len())
            .map(|k| (key_valids[k][r], keys[k][r]))
            .collect();
        groups.entry(key).or_default().push(r);
    }

    let mut out: BTreeMap<KeyTuple, AggTuple> = BTreeMap::new();
    for (key, rows) in groups {
        let agg_tuple: AggTuple = aggs
            .iter()
            .map(|req| match req.kind {
                AggKind::SumI64 => {
                    let v = val_i64
                        .iter()
                        .find(|v| v.col_idx == req.input_col_idx)
                        .expect("i64 col for SumI64");
                    let s: i64 = rows
                        .iter()
                        .filter(|&&r| v.valid[r])
                        .map(|&r| v.data[r])
                        .sum();
                    // Polars: sum of all-null group = 0 (not null)
                    AggOutputValue::I64 {
                        valid: true,
                        value: s,
                    }
                }
                AggKind::MinI64 => {
                    let v = val_i64
                        .iter()
                        .find(|v| v.col_idx == req.input_col_idx)
                        .expect("i64 col for MinI64");
                    let vals: Vec<i64> = rows
                        .iter()
                        .filter(|&&r| v.valid[r])
                        .map(|&r| v.data[r])
                        .collect();
                    if vals.is_empty() {
                        AggOutputValue::I64 {
                            valid: false,
                            value: 0,
                        }
                    } else {
                        AggOutputValue::I64 {
                            valid: true,
                            value: *vals.iter().min().expect("non-empty vals"),
                        }
                    }
                }
                AggKind::MaxI64 => {
                    let v = val_i64
                        .iter()
                        .find(|v| v.col_idx == req.input_col_idx)
                        .expect("i64 col for MaxI64");
                    let vals: Vec<i64> = rows
                        .iter()
                        .filter(|&&r| v.valid[r])
                        .map(|&r| v.data[r])
                        .collect();
                    if vals.is_empty() {
                        AggOutputValue::I64 {
                            valid: false,
                            value: 0,
                        }
                    } else {
                        AggOutputValue::I64 {
                            valid: true,
                            value: *vals.iter().max().expect("non-empty vals"),
                        }
                    }
                }
                AggKind::Count => {
                    let v = val_i64
                        .iter()
                        .find(|v| v.col_idx == req.input_col_idx)
                        .expect("i64 col for Count");
                    let c = rows.iter().filter(|&&r| v.valid[r]).count() as u64;
                    AggOutputValue::U64(c)
                }
                AggKind::Len => AggOutputValue::U64(rows.len() as u64),
                _ => panic!("agg kind {:?} not implemented in reference", req.kind),
            })
            .collect();
        out.insert(key, agg_tuple);
    }
    out
}

// -----------------------------------------------------------------------
// Unit tests
// -----------------------------------------------------------------------

#[test]
fn pipeline_four_groups_single_i64_key() {
    // 4 groups × 256 rows = 1024 rows, one i64 key in {0,1,2,3}, sum + len.
    // Each i64 key needs 65 bits (1 null + 64 data); a single i64 key fits
    // within the 128-bit budget. Two i64 keys would need 130 bits and hit
    // the TooWide error — that path is covered by T30 (key-width fallback).
    let (device, mut queue) = setup();
    let n_rows = 1024usize;
    let keys: Vec<i64> = (0..n_rows).map(|i| (i % 4) as i64).collect();
    let key_valid: Vec<bool> = vec![true; n_rows];
    let val: Vec<i64> = (0..n_rows).map(|i| i as i64).collect();
    let val_valid: Vec<bool> = vec![true; n_rows];

    let key_bytes = bytes_i64(&keys);
    let key_valid_p = pack_valid(&key_valid);
    let val_valid_p = pack_valid(&val_valid);

    let key_cols = vec![KeyColumn {
        name: "a".into(),
        dtype: KeyDtype::I64,
        data: &key_bytes,
        valid: &key_valid_p,
        n_rows,
    }];
    let agg_specs = vec![
        (
            AggRequest {
                kind: AggKind::SumI64,
                input_col_idx: 0,
            },
            ValueColumn::I64 {
                data: &val,
                valid: &val_valid_p,
            },
        ),
        (
            AggRequest {
                kind: AggKind::Len,
                input_col_idx: 0,
            },
            ValueColumn::I64 {
                data: &val,
                valid: &val_valid_p,
            },
        ),
    ];
    let result = dispatch_groupby(&device, &mut queue, &key_cols, &agg_specs, n_rows)
        .expect("dispatch_groupby");
    assert_eq!(result.n_groups, 4, "expected exactly 4 groups");

    let canonical = canonicalize(&result.decoded_keys, &result.agg_outputs, result.n_groups);
    let reference = cpu_reference_groupby(
        &[keys],
        &[key_valid],
        &[RefValI64 {
            col_idx: 0,
            data: &val,
            valid: &val_valid,
        }],
        &[
            AggRequest {
                kind: AggKind::SumI64,
                input_col_idx: 0,
            },
            AggRequest {
                kind: AggKind::Len,
                input_col_idx: 0,
            },
        ],
        n_rows,
    );
    assert_eq!(canonical, reference);
}

#[test]
fn pipeline_empty_input() {
    let (device, mut queue) = setup();
    let key_valid: Vec<u8> = vec![0u8; 1];
    let val_valid: Vec<u8> = vec![0u8; 1];

    let key_cols = vec![KeyColumn {
        name: "k".into(),
        dtype: KeyDtype::I64,
        data: &[],
        valid: &key_valid,
        n_rows: 0,
    }];
    let agg_specs = vec![(
        AggRequest {
            kind: AggKind::SumI64,
            input_col_idx: 0,
        },
        ValueColumn::I64 {
            data: &[],
            valid: &val_valid,
        },
    )];
    let result =
        dispatch_groupby(&device, &mut queue, &key_cols, &agg_specs, 0).expect("dispatch_groupby");
    assert_eq!(result.n_groups, 0);
    assert!(
        result.decoded_keys[0]
            == DecodedColumn::I64 {
                values: vec![],
                valid: vec![]
            }
    );
    assert!(
        result.agg_outputs[0]
            == AggOutput::I64 {
                values: vec![],
                valid: vec![]
            }
    );
}

#[test]
fn pipeline_single_row() {
    let (device, mut queue) = setup();
    let key = vec![42i64];
    let val = vec![100i64];
    let key_bytes = bytes_i64(&key);
    let val_valid_bool = vec![true];
    let key_valid_p = pack_valid(&[true]);
    let val_valid_p = pack_valid(&val_valid_bool);

    let key_cols = vec![KeyColumn {
        name: "k".into(),
        dtype: KeyDtype::I64,
        data: &key_bytes,
        valid: &key_valid_p,
        n_rows: 1,
    }];
    let agg_specs = vec![(
        AggRequest {
            kind: AggKind::SumI64,
            input_col_idx: 0,
        },
        ValueColumn::I64 {
            data: &val,
            valid: &val_valid_p,
        },
    )];
    let result =
        dispatch_groupby(&device, &mut queue, &key_cols, &agg_specs, 1).expect("dispatch_groupby");
    assert_eq!(result.n_groups, 1);
    match &result.agg_outputs[0] {
        AggOutput::I64 { values, valid } => {
            assert_eq!(values[0], 100);
            assert!(valid[0]);
        }
        other => panic!("expected I64 output, got {other:?}"),
    }
    // Decoded key should be 42
    match &result.decoded_keys[0] {
        DecodedColumn::I64 { values, valid } => {
            assert_eq!(values[0], 42);
            assert!(valid[0]);
        }
        other => panic!("expected I64 key column, got {other:?}"),
    }
}

// -----------------------------------------------------------------------
// Proptest
// -----------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn pipeline_matches_reference_random(
        n in 4usize..256,
        max_key in 1i64..=8,
    ) {
        let (device, mut queue) = setup();
        let keys: Vec<i64> = (0..n).map(|i| (i as i64) % max_key).collect();
        let key_valid: Vec<bool> = vec![true; n];
        let vals: Vec<i64> = (0..n).map(|i| i as i64 * 7).collect();
        // Every 5th row is null on the value column to exercise null paths.
        let val_valid: Vec<bool> = (0..n).map(|i| i % 5 != 0).collect();

        let key_bytes = bytes_i64(&keys);
        let key_valid_p = pack_valid(&key_valid);
        let val_valid_p = pack_valid(&val_valid);

        let key_cols = vec![KeyColumn {
            name: "k".into(),
            dtype: KeyDtype::I64,
            data: &key_bytes,
            valid: &key_valid_p,
            n_rows: n,
        }];
        let agg_specs = vec![
            (
                AggRequest { kind: AggKind::SumI64, input_col_idx: 0 },
                ValueColumn::I64 { data: &vals, valid: &val_valid_p },
            ),
            (
                AggRequest { kind: AggKind::Count, input_col_idx: 0 },
                ValueColumn::I64 { data: &vals, valid: &val_valid_p },
            ),
            (
                AggRequest { kind: AggKind::Len, input_col_idx: 0 },
                ValueColumn::I64 { data: &vals, valid: &val_valid_p },
            ),
        ];
        let result = dispatch_groupby(&device, &mut queue, &key_cols, &agg_specs, n)
            .expect("dispatch_groupby");

        let canonical = canonicalize(&result.decoded_keys, &result.agg_outputs, result.n_groups);
        let reference = cpu_reference_groupby(
            &[keys.clone()],
            &[key_valid.clone()],
            &[RefValI64 { col_idx: 0, data: &vals, valid: &val_valid }],
            &[
                AggRequest { kind: AggKind::SumI64, input_col_idx: 0 },
                AggRequest { kind: AggKind::Count, input_col_idx: 0 },
                AggRequest { kind: AggKind::Len, input_col_idx: 0 },
            ],
            n,
        );
        prop_assert_eq!(canonical, reference);
    }
}
