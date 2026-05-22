// crates/polars-metal-kernels/tests/test_key_encoding.rs
//
// Encoder + decoder unit tests. Covers single-key, multi-key,
// mixed dtypes, null patterns, and width-overflow.
#![allow(clippy::expect_used, clippy::panic)]

use polars_metal_kernels::groupby::{
    decode_keys, encode_keys, DecodedColumn, KeyColumn, KeyDtype, KeyEncodeError,
};

fn bytes_i64(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_f64(values: &[f64]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn all_valid(n_rows: usize) -> Vec<u8> {
    vec![0xFFu8; (n_rows + 7) / 8]
}

#[test]
fn single_i64_key_encodes_to_u128_per_row() {
    let data = bytes_i64(&[1, 2, 3, -1]);
    let valid = all_valid(4);
    let col = KeyColumn {
        name: "k".into(),
        dtype: KeyDtype::I64,
        data: &data,
        valid: &valid,
        n_rows: 4,
    };
    let (encoded, schema) = encode_keys(&[col]).expect("encode_keys");
    assert_eq!(encoded.len(), 4);
    assert_eq!(schema.total_bits(), 65);
    assert_eq!(schema.fields().len(), 1);
    let decoded = decode_keys(&encoded, &schema);
    match &decoded[0] {
        DecodedColumn::I64 { values, valid } => {
            assert_eq!(values, &vec![1i64, 2, 3, -1]);
            assert_eq!(valid, &vec![true, true, true, true]);
        }
        other => panic!("expected I64 decoded column, got {other:?}"),
    }
}

#[test]
fn two_bool_keys_pack_into_first_4_bits() {
    let a = vec![0b0000_0011u8]; // rows 0,1 true
    let b = vec![0b0000_0001u8]; // row 0 true, row 1 false
    let v = vec![0xFFu8];
    let cols = vec![
        KeyColumn {
            name: "a".into(),
            dtype: KeyDtype::Bool,
            data: &a,
            valid: &v,
            n_rows: 2,
        },
        KeyColumn {
            name: "b".into(),
            dtype: KeyDtype::Bool,
            data: &b,
            valid: &v,
            n_rows: 2,
        },
    ];
    let (encoded, schema) = encode_keys(&cols).expect("encode_keys");
    assert_eq!(encoded.len(), 2);
    assert_eq!(schema.total_bits(), 4);
    let decoded = decode_keys(&encoded, &schema);
    match &decoded[0] {
        DecodedColumn::Bool { values, valid } => {
            assert_eq!(values, &vec![true, true]);
            assert_eq!(valid, &vec![true, true]);
        }
        _ => panic!("expected Bool"),
    }
    match &decoded[1] {
        DecodedColumn::Bool { values, valid } => {
            assert_eq!(values, &vec![true, false]);
            assert_eq!(valid, &vec![true, true]);
        }
        _ => panic!("expected Bool"),
    }
}

#[test]
fn one_i64_plus_one_bool_packs_below_128_bits() {
    let i64_data = bytes_i64(&[42, -7]);
    let bool_data = vec![0b0000_0010u8]; // row 0 false, row 1 true
    let v = vec![0xFFu8];
    let cols = vec![
        KeyColumn {
            name: "i".into(),
            dtype: KeyDtype::I64,
            data: &i64_data,
            valid: &v,
            n_rows: 2,
        },
        KeyColumn {
            name: "b".into(),
            dtype: KeyDtype::Bool,
            data: &bool_data,
            valid: &v,
            n_rows: 2,
        },
    ];
    let (encoded, schema) = encode_keys(&cols).expect("encode_keys");
    assert_eq!(schema.total_bits(), 67);
    let decoded = decode_keys(&encoded, &schema);
    assert_eq!(decoded.len(), 2);
}

#[test]
fn null_value_clears_data_bits_in_decoded_output() {
    let data = bytes_i64(&[99, 0]);
    let valid = vec![0b0000_0001u8]; // row 0 valid, row 1 null
    let cols = vec![KeyColumn {
        name: "k".into(),
        dtype: KeyDtype::I64,
        data: &data,
        valid: &valid,
        n_rows: 2,
    }];
    let (encoded, schema) = encode_keys(&cols).expect("encode_keys");
    let decoded = decode_keys(&encoded, &schema);
    match &decoded[0] {
        DecodedColumn::I64 { values, valid } => {
            assert_eq!(valid, &vec![true, false]);
            assert_eq!(values[0], 99);
        }
        _ => panic!("expected I64"),
    }
}

#[test]
fn three_i64_keys_overflow_128_bits_returns_error() {
    let d = bytes_i64(&[1, 2]);
    let v = vec![0xFFu8];
    let cols = vec![
        KeyColumn {
            name: "a".into(),
            dtype: KeyDtype::I64,
            data: &d,
            valid: &v,
            n_rows: 2,
        },
        KeyColumn {
            name: "b".into(),
            dtype: KeyDtype::I64,
            data: &d,
            valid: &v,
            n_rows: 2,
        },
    ];
    let err = encode_keys(&cols).expect_err("expected TooWide");
    match err {
        KeyEncodeError::TooWide { total_bits } => assert_eq!(total_bits, 130),
        other => panic!("expected TooWide, got {other:?}"),
    }
}

#[test]
fn empty_keys_returns_error() {
    let err = encode_keys(&[]).expect_err("expected NoKeys");
    assert!(matches!(err, KeyEncodeError::NoKeys));
}

#[test]
fn f64_key_encodes_via_raw_bits() {
    let data = bytes_f64(&[1.5, -2.5, f64::INFINITY]);
    let v = all_valid(3);
    let cols = vec![KeyColumn {
        name: "f".into(),
        dtype: KeyDtype::F64,
        data: &data,
        valid: &v,
        n_rows: 3,
    }];
    let (encoded, schema) = encode_keys(&cols).expect("encode_keys");
    assert_eq!(schema.total_bits(), 65);
    let decoded = decode_keys(&encoded, &schema);
    match &decoded[0] {
        DecodedColumn::F64 { values, valid } => {
            assert_eq!(values, &vec![1.5, -2.5, f64::INFINITY]);
            assert_eq!(valid, &vec![true, true, true]);
        }
        _ => panic!("expected F64"),
    }
}

// === T15 proptest module ===

use proptest::prelude::*;

#[derive(Debug, Clone)]
struct ArbI64Col {
    values: Vec<i64>,
    valid: Vec<bool>,
}

#[derive(Debug, Clone)]
struct ArbBoolCol {
    values: Vec<bool>,
    valid: Vec<bool>,
}

fn pack_valid(valid: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; (valid.len() + 7) / 8];
    for (i, &v) in valid.iter().enumerate() {
        if v {
            out[i >> 3] |= 1 << (i & 7);
        }
    }
    out
}

fn pack_bool_data(values: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; (values.len() + 7) / 8];
    for (i, &v) in values.iter().enumerate() {
        if v {
            out[i >> 3] |= 1 << (i & 7);
        }
    }
    out
}

fn arb_i64_col(n: usize) -> impl Strategy<Value = ArbI64Col> {
    (
        prop::collection::vec(any::<i64>(), n..=n),
        prop::collection::vec(any::<bool>(), n..=n),
    )
        .prop_map(|(values, valid)| ArbI64Col { values, valid })
}

fn arb_bool_col(n: usize) -> impl Strategy<Value = ArbBoolCol> {
    (
        prop::collection::vec(any::<bool>(), n..=n),
        prop::collection::vec(any::<bool>(), n..=n),
    )
        .prop_map(|(values, valid)| ArbBoolCol { values, valid })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// One i64 key, varying row counts. Round-trip preserves valid rows'
    /// values byte-for-byte.
    #[test]
    fn roundtrip_single_i64(n in 1usize..256, col in arb_i64_col(256)) {
        let values: Vec<i64> = col.values.iter().take(n).copied().collect();
        let valid_bools: Vec<bool> = col.valid.iter().take(n).copied().collect();
        let data = bytes_i64(&values);
        let valid = pack_valid(&valid_bools);

        let kc = KeyColumn {
            name: "k".into(),
            dtype: KeyDtype::I64,
            data: &data,
            valid: &valid,
            n_rows: n,
        };
        let (encoded, schema) = encode_keys(&[kc]).expect("encode_keys");
        let decoded = decode_keys(&encoded, &schema);
        prop_assert_eq!(decoded.len(), 1);
        match &decoded[0] {
            DecodedColumn::I64 { values: dv, valid: dvalid } => {
                prop_assert_eq!(dv.len(), n);
                prop_assert_eq!(dvalid.len(), n);
                for i in 0..n {
                    prop_assert_eq!(dvalid[i], valid_bools[i]);
                    if valid_bools[i] {
                        prop_assert_eq!(dv[i], values[i]);
                    }
                }
            }
            _ => prop_assert!(false, "expected I64 decoded column"),
        }
    }

    /// One i64 + one bool key — composite case under 128 bits.
    #[test]
    fn roundtrip_i64_plus_bool(
        i64_col in arb_i64_col(32),
        bool_col in arb_bool_col(32),
    ) {
        let n = 32;
        let i64_values = i64_col.values;
        let i64_valid = i64_col.valid;
        let bool_values = bool_col.values;
        let bool_valid = bool_col.valid;

        let i64_data = bytes_i64(&i64_values);
        let i64_valid_packed = pack_valid(&i64_valid);
        let bool_data = pack_bool_data(&bool_values);
        let bool_valid_packed = pack_valid(&bool_valid);

        let cols = vec![
            KeyColumn { name: "i".into(), dtype: KeyDtype::I64, data: &i64_data, valid: &i64_valid_packed, n_rows: n },
            KeyColumn { name: "b".into(), dtype: KeyDtype::Bool, data: &bool_data, valid: &bool_valid_packed, n_rows: n },
        ];
        let (encoded, schema) = encode_keys(&cols).expect("encode_keys");
        prop_assert_eq!(schema.total_bits(), 1 + 64 + 1 + 1);
        let decoded = decode_keys(&encoded, &schema);
        prop_assert_eq!(decoded.len(), 2);
        match (&decoded[0], &decoded[1]) {
            (
                DecodedColumn::I64 { values: iv, valid: ivd },
                DecodedColumn::Bool { values: bv, valid: bvd },
            ) => {
                for i in 0..n {
                    prop_assert_eq!(ivd[i], i64_valid[i]);
                    prop_assert_eq!(bvd[i], bool_valid[i]);
                    if i64_valid[i] {
                        prop_assert_eq!(iv[i], i64_values[i]);
                    }
                    if bool_valid[i] {
                        prop_assert_eq!(bv[i], bool_values[i]);
                    }
                }
            }
            _ => prop_assert!(false, "unexpected decoded shape"),
        }
    }

    /// Equal rows in the source produce equal u128 lanes.
    #[test]
    fn equal_keys_encode_to_equal_lanes(a in any::<i64>(), b in any::<i64>()) {
        let n = 4;
        let values: Vec<i64> = vec![a, b, a, b];
        let valid_bools = vec![true, true, true, true];
        let data = bytes_i64(&values);
        let valid = pack_valid(&valid_bools);
        let kc = KeyColumn {
            name: "k".into(),
            dtype: KeyDtype::I64,
            data: &data,
            valid: &valid,
            n_rows: n,
        };
        let (encoded, _) = encode_keys(&[kc]).expect("encode_keys");
        prop_assert_eq!(encoded[0], encoded[2]);
        prop_assert_eq!(encoded[1], encoded[3]);
        if a != b {
            prop_assert_ne!(encoded[0], encoded[1]);
        }
    }
}
