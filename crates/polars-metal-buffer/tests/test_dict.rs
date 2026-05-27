//! Tests for the CPU-side dictionary encoder.
#![allow(clippy::expect_used)]

use polars_metal_buffer::dict::{build_dict, build_dict_nullable, decode_dict};

#[test]
fn dict_roundtrip_simple() {
    let strings = vec!["apple", "banana", "apple", "cherry", "banana"];
    let (dict, codes) = build_dict(&strings);
    assert_eq!(dict.len(), 3);
    assert!(dict.contains(&"apple".to_string()));
    let decoded = decode_dict(&dict, &codes);
    assert_eq!(
        decoded,
        vec!["apple", "banana", "apple", "cherry", "banana"]
    );
}

#[test]
fn dict_first_seen_order_is_deterministic() {
    let strings = vec!["b", "a", "b", "c", "a"];
    let (dict, codes) = build_dict(&strings);
    assert_eq!(
        dict,
        vec!["b".to_string(), "a".to_string(), "c".to_string()]
    );
    assert_eq!(codes, vec![0, 1, 0, 2, 1]);
}

#[test]
fn dict_empty_input() {
    let strings: Vec<&str> = vec![];
    let (dict, codes) = build_dict(&strings);
    assert!(dict.is_empty());
    assert!(codes.is_empty());
}

#[test]
fn dict_handles_empty_strings_and_nulls() {
    let strings: Vec<Option<&str>> = vec![Some("a"), None, Some(""), Some("a"), None];
    let (dict, codes, valid) = build_dict_nullable(&strings);
    // "a" and "" — both are valid distinct strings.
    assert_eq!(dict.len(), 2);
    assert_eq!(valid, vec![true, false, true, true, false]);
    // codes for valid rows match dict order: a=0, ""=1.
    assert_eq!(codes[0], 0);
    assert_eq!(codes[2], 1);
    assert_eq!(codes[3], 0);
}

#[test]
fn dict_nullable_all_null() {
    let strings: Vec<Option<&str>> = vec![None, None, None];
    let (dict, codes, valid) = build_dict_nullable(&strings);
    assert!(dict.is_empty());
    assert_eq!(valid, vec![false, false, false]);
    assert_eq!(codes, vec![0, 0, 0]);
}
