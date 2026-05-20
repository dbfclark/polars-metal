// crates/polars-metal-mlx-sys/tests/test_cumsum.rs
//
// Integration tests for the cumsum_u8_to_u32 binding. This op underpins the
// filter compaction pipeline: a bit-packed keep-flag column is densified to
// u8, then inclusive-cumsum-ed to produce output offsets. The output domain
// is u32 so 4B-row inputs don't overflow.
#![allow(clippy::expect_used)]

use polars_metal_mlx_sys::cumsum_u8_to_u32;

#[test]
fn cumsum_basic_inclusive() {
    let input: Vec<u8> = vec![1, 0, 1, 1, 0, 1];
    let mut output = vec![0u32; input.len()];
    cumsum_u8_to_u32(&input, &mut output).expect("dispatch succeeds");
    assert_eq!(output, vec![1u32, 1, 2, 3, 3, 4]);
}

#[test]
fn cumsum_all_zeros() {
    let input = vec![0u8; 1024];
    let mut output = vec![0u32; 1024];
    cumsum_u8_to_u32(&input, &mut output).expect("dispatch succeeds");
    for v in &output {
        assert_eq!(*v, 0);
    }
}

#[test]
fn cumsum_all_ones_large() {
    let input = vec![1u8; 10_000];
    let mut output = vec![0u32; 10_000];
    cumsum_u8_to_u32(&input, &mut output).expect("dispatch succeeds");
    for (i, v) in output.iter().enumerate() {
        assert_eq!(*v, (i as u32) + 1, "row {i}");
    }
}

#[test]
fn cumsum_empty_input() {
    let input: Vec<u8> = Vec::new();
    let mut output: Vec<u32> = Vec::new();
    // Should succeed with no output writes; empty input is short-circuited
    // in the Rust wrapper before touching MLX.
    cumsum_u8_to_u32(&input, &mut output).expect("empty input is allowed");
}

#[test]
fn cumsum_length_mismatch_returns_error() {
    let input = vec![1u8; 10];
    let mut output = vec![0u32; 5];
    let result = cumsum_u8_to_u32(&input, &mut output);
    assert!(result.is_err(), "mismatched lengths must error");
}
