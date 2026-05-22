// crates/polars-metal-kernels/tests/test_key_encoding_small_ints.rs
//! Proptest: encode → decode roundtrip for the M3-added KeyDtype variants.
//! Follows M2's pattern: struct-literal KeyColumn with `data: &[u8]` and
//! `valid: &[u8]` (no constructor helpers exist).
#![allow(clippy::expect_used)]

use polars_metal_kernels::groupby::{decode_keys, encode_keys, DecodedColumn, KeyColumn, KeyDtype};
use proptest::prelude::*;

fn bytes_i8(values: &[i8]) -> Vec<u8> {
    values.iter().map(|v| *v as u8).collect()
}
fn bytes_i16(values: &[i16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn bytes_i32(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn bytes_u8(values: &[u8]) -> Vec<u8> {
    values.to_vec()
}
fn bytes_u16(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn bytes_u32(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn all_valid(n_rows: usize) -> Vec<u8> {
    vec![0xFFu8; (n_rows + 7) / 8]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn i8_roundtrip(values in proptest::collection::vec(any::<i8>(), 1..256)) {
        let data = bytes_i8(&values);
        let valid = all_valid(values.len());
        let col = KeyColumn { name: "k".into(), dtype: KeyDtype::I8, data: &data, valid: &valid, n_rows: values.len() };
        let (encoded, schema) = encode_keys(&[col]).expect("encode");
        let decoded = decode_keys(&encoded, &schema);
        match &decoded[0] {
            DecodedColumn::I8 { values: out, valid: ovalid } => {
                prop_assert_eq!(out, &values);
                prop_assert!(ovalid.iter().all(|&v| v));
            }
            _ => prop_assert!(false, "expected DecodedColumn::I8"),
        }
    }

    #[test]
    fn i16_roundtrip(values in proptest::collection::vec(any::<i16>(), 1..256)) {
        let data = bytes_i16(&values);
        let valid = all_valid(values.len());
        let col = KeyColumn { name: "k".into(), dtype: KeyDtype::I16, data: &data, valid: &valid, n_rows: values.len() };
        let (encoded, schema) = encode_keys(&[col]).expect("encode");
        let decoded = decode_keys(&encoded, &schema);
        match &decoded[0] {
            DecodedColumn::I16 { values: out, .. } => prop_assert_eq!(out, &values),
            _ => prop_assert!(false, "expected DecodedColumn::I16"),
        }
    }

    #[test]
    fn u8_roundtrip(values in proptest::collection::vec(any::<u8>(), 1..256)) {
        let data = bytes_u8(&values);
        let valid = all_valid(values.len());
        let col = KeyColumn { name: "k".into(), dtype: KeyDtype::U8, data: &data, valid: &valid, n_rows: values.len() };
        let (encoded, schema) = encode_keys(&[col]).expect("encode");
        let decoded = decode_keys(&encoded, &schema);
        match &decoded[0] {
            DecodedColumn::U8 { values: out, .. } => prop_assert_eq!(out, &values),
            _ => prop_assert!(false, "expected DecodedColumn::U8"),
        }
    }

    #[test]
    fn u16_roundtrip(values in proptest::collection::vec(any::<u16>(), 1..256)) {
        let data = bytes_u16(&values);
        let valid = all_valid(values.len());
        let col = KeyColumn { name: "k".into(), dtype: KeyDtype::U16, data: &data, valid: &valid, n_rows: values.len() };
        let (encoded, schema) = encode_keys(&[col]).expect("encode");
        let decoded = decode_keys(&encoded, &schema);
        match &decoded[0] {
            DecodedColumn::U16 { values: out, .. } => prop_assert_eq!(out, &values),
            _ => prop_assert!(false, "expected DecodedColumn::U16"),
        }
    }

    #[test]
    fn u32_roundtrip(values in proptest::collection::vec(any::<u32>(), 1..256)) {
        let data = bytes_u32(&values);
        let valid = all_valid(values.len());
        let col = KeyColumn { name: "k".into(), dtype: KeyDtype::U32, data: &data, valid: &valid, n_rows: values.len() };
        let (encoded, schema) = encode_keys(&[col]).expect("encode");
        let decoded = decode_keys(&encoded, &schema);
        match &decoded[0] {
            DecodedColumn::U32 { values: out, .. } => prop_assert_eq!(out, &values),
            _ => prop_assert!(false, "expected DecodedColumn::U32"),
        }
    }

    #[test]
    fn multi_dtype_composite_under_128_bits(
        i8_vals  in proptest::collection::vec(any::<i8>(),  4..32),
        i16_vals in proptest::collection::vec(any::<i16>(), 4..32),
        u32_vals in proptest::collection::vec(any::<u32>(), 4..32),
    ) {
        let n = i8_vals.len().min(i16_vals.len()).min(u32_vals.len());
        let i8_vals = &i8_vals[..n];
        let i16_vals = &i16_vals[..n];
        let u32_vals = &u32_vals[..n];
        let valid = all_valid(n);
        let d8 = bytes_i8(i8_vals);
        let d16 = bytes_i16(i16_vals);
        let d32 = bytes_u32(u32_vals);
        let cols = vec![
            KeyColumn { name: "a".into(), dtype: KeyDtype::I8,  data: &d8,  valid: &valid, n_rows: n },
            KeyColumn { name: "b".into(), dtype: KeyDtype::I16, data: &d16, valid: &valid, n_rows: n },
            KeyColumn { name: "c".into(), dtype: KeyDtype::U32, data: &d32, valid: &valid, n_rows: n },
        ];
        let (encoded, schema) = encode_keys(&cols).expect("encode");
        let decoded = decode_keys(&encoded, &schema);
        match (&decoded[0], &decoded[1], &decoded[2]) {
            (DecodedColumn::I8 { values: out_a, .. },
             DecodedColumn::I16 { values: out_b, .. },
             DecodedColumn::U32 { values: out_c, .. }) => {
                prop_assert_eq!(out_a.as_slice(), i8_vals);
                prop_assert_eq!(out_b.as_slice(), i16_vals);
                prop_assert_eq!(out_c.as_slice(), u32_vals);
            }
            _ => prop_assert!(false, "unexpected decoded variants"),
        }
    }

    #[test]
    fn duplicate_signed_values_encode_identically(values in proptest::collection::vec(any::<i32>(), 2..64)) {
        let data = bytes_i32(&values);
        let valid = all_valid(values.len());
        let col = KeyColumn { name: "k".into(), dtype: KeyDtype::I32, data: &data, valid: &valid, n_rows: values.len() };
        let (encoded, _schema) = encode_keys(&[col]).expect("encode");
        for i in 0..values.len() {
            for j in 0..values.len() {
                if values[i] == values[j] {
                    prop_assert_eq!(encoded[i], encoded[j]);
                }
            }
        }
    }
}
