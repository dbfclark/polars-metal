//! KeyDtype::Utf8 round-trip + interaction with other dtypes.
#![allow(clippy::expect_used, clippy::panic)]

use polars_metal_kernels::groupby::{decode_keys, encode_keys, DecodedColumn, KeyColumn, KeyDtype};

#[test]
fn utf8_roundtrip_single_column() {
    // Build dict + codes for ["a", "b", "a", "c"].
    let dict = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let codes: Vec<u32> = vec![0, 1, 0, 2];
    // SAFETY: u32 is POD; reinterpret as bytes for the encoder.
    let data: &[u8] =
        unsafe { std::slice::from_raw_parts(codes.as_ptr() as *const u8, codes.len() * 4) };
    let col = KeyColumn {
        name: "s".into(),
        dtype: KeyDtype::Utf8,
        data,
        valid: &[0b1111u8],
        n_rows: 4,
        dict: Some(dict),
    };
    let (encoded, schema) = encode_keys(&[col]).expect("encode");
    let decoded = decode_keys(&encoded, &schema);
    assert_eq!(decoded.len(), 1);
    match &decoded[0] {
        DecodedColumn::Utf8 { values, valid } => {
            assert_eq!(
                values,
                &vec![
                    "a".to_string(),
                    "b".to_string(),
                    "a".to_string(),
                    "c".to_string()
                ]
            );
            assert_eq!(valid, &vec![true, true, true, true]);
        }
        _ => panic!("expected Utf8"),
    }
}

#[test]
fn utf8_combined_with_int_key() {
    let dict = vec!["x".into(), "y".into(), "z".into()];
    let codes: Vec<u32> = vec![0, 1, 0, 2];
    let s_data: &[u8] =
        unsafe { std::slice::from_raw_parts(codes.as_ptr() as *const u8, codes.len() * 4) };
    let ints: Vec<i32> = vec![1, 2, 1, 3];
    let i_data: &[u8] =
        unsafe { std::slice::from_raw_parts(ints.as_ptr() as *const u8, ints.len() * 4) };
    let cols = vec![
        KeyColumn {
            name: "s".into(),
            dtype: KeyDtype::Utf8,
            data: s_data,
            valid: &[0b1111u8],
            n_rows: 4,
            dict: Some(dict),
        },
        KeyColumn {
            name: "i".into(),
            dtype: KeyDtype::I32,
            data: i_data,
            valid: &[0b1111u8],
            n_rows: 4,
            dict: None,
        },
    ];
    let (encoded, schema) = encode_keys(&cols).expect("encode");
    let decoded = decode_keys(&encoded, &schema);
    assert_eq!(decoded.len(), 2);
    match &decoded[0] {
        DecodedColumn::Utf8 { values, .. } => {
            assert_eq!(
                values,
                &vec![
                    "x".to_string(),
                    "y".to_string(),
                    "x".to_string(),
                    "z".to_string()
                ]
            );
        }
        _ => panic!("expected Utf8 at index 0"),
    }
    match &decoded[1] {
        DecodedColumn::I32 { values, .. } => {
            assert_eq!(values, &vec![1i32, 2, 1, 3]);
        }
        _ => panic!("expected I32 at index 1"),
    }
}

#[test]
fn utf8_missing_dict_errors() {
    let codes: Vec<u32> = vec![0, 1];
    let data: &[u8] =
        unsafe { std::slice::from_raw_parts(codes.as_ptr() as *const u8, codes.len() * 4) };
    let col = KeyColumn {
        name: "s".into(),
        dtype: KeyDtype::Utf8,
        data,
        valid: &[0b11u8],
        n_rows: 2,
        dict: None, // intentional: should error
    };
    let result = encode_keys(&[col]);
    assert!(result.is_err());
}

#[test]
fn utf8_null_row_emits_empty_string() {
    let dict = vec!["a".to_string()];
    let codes: Vec<u32> = vec![0, 0, 0];
    let data: &[u8] =
        unsafe { std::slice::from_raw_parts(codes.as_ptr() as *const u8, codes.len() * 4) };
    let col = KeyColumn {
        name: "s".into(),
        dtype: KeyDtype::Utf8,
        data,
        valid: &[0b101u8], // rows 0, 2 valid; row 1 null
        n_rows: 3,
        dict: Some(dict),
    };
    let (encoded, schema) = encode_keys(&[col]).expect("encode");
    let decoded = decode_keys(&encoded, &schema);
    match &decoded[0] {
        DecodedColumn::Utf8 { values, valid } => {
            assert_eq!(valid, &vec![true, false, true]);
            // Null row's value is the dtype default (empty string).
            assert_eq!(values[1], "");
        }
        _ => panic!("expected Utf8"),
    }
}
