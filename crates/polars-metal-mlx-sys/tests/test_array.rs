// crates/polars-metal-mlx-sys/tests/test_array.rs
//! Construct an MlxArrayHandle from a raw F32 buffer; eval it; read the value back.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use polars_metal_mlx_sys::array::{
    mlx_array_copy_to_f32_slice, mlx_array_eval, mlx_array_from_f32_slice, mlx_array_to_f32_vec,
};

#[test]
fn copy_to_f32_slice_writes_into_caller_buffer() {
    // The output-zero-copy path copies an eval'd MLX array directly into a
    // caller-owned destination (the numpy output array) with no intermediate
    // Vec. Returns the element count written.
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let handle = mlx_array_from_f32_slice(&input).expect("construct");
    mlx_array_eval(&[handle.clone()]).expect("eval");

    let mut dst = vec![0.0_f32; 5];
    let n = mlx_array_copy_to_f32_slice(&handle, &mut dst).expect("copy into slice");
    assert_eq!(n, 5);
    assert_eq!(dst, input);
}

#[test]
fn copy_to_f32_slice_rejects_too_small_dst() {
    let handle = mlx_array_from_f32_slice(&[1.0, 2.0, 3.0]).expect("construct");
    mlx_array_eval(&[handle.clone()]).expect("eval");
    let mut dst = vec![0.0_f32; 2];
    assert!(mlx_array_copy_to_f32_slice(&handle, &mut dst).is_err());
}

#[test]
fn round_trip_f32_array() {
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let handle = mlx_array_from_f32_slice(&input).expect("construct");
    mlx_array_eval(&[handle.clone()]).expect("eval");
    let output = mlx_array_to_f32_vec(&handle).expect("read back");
    assert_eq!(output, input);
}

#[test]
fn empty_array_is_supported() {
    let input: Vec<f32> = vec![];
    let handle = mlx_array_from_f32_slice(&input).expect("construct empty");
    mlx_array_eval(&[handle.clone()]).expect("eval empty");
    let output = mlx_array_to_f32_vec(&handle).expect("read back empty");
    assert!(output.is_empty());
}

#[test]
fn shape_and_dtype_accessors_work() {
    let h = mlx_array_from_f32_slice(&[1.0, 2.0, 3.0]).unwrap();
    assert_eq!(h.shape(), vec![3]);
    assert!(h.dtype_is_f32());
}
