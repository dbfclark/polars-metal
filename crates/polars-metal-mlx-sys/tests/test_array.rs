// crates/polars-metal-mlx-sys/tests/test_array.rs
//! Construct an MlxArrayHandle from a raw F32 buffer; eval it; read the value back.
#![allow(clippy::expect_used)]

use polars_metal_mlx_sys::array::{mlx_array_from_f32_slice, mlx_array_to_f32_vec, mlx_eval};

#[test]
fn round_trip_f32_array() {
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let handle = mlx_array_from_f32_slice(&input).expect("construct");
    mlx_eval(&[handle.clone()]).expect("eval");
    let output = mlx_array_to_f32_vec(&handle).expect("read back");
    assert_eq!(output, input);
}

#[test]
fn empty_array_is_supported() {
    let input: Vec<f32> = vec![];
    let handle = mlx_array_from_f32_slice(&input).expect("construct empty");
    mlx_eval(&[handle.clone()]).expect("eval empty");
    let output = mlx_array_to_f32_vec(&handle).expect("read back empty");
    assert!(output.is_empty());
}
