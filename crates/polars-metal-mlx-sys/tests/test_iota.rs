// M5 rolling Task 4b: mlx_iota_f32 FFI binding.
//
// Produces a 1-D F32 array [0.0, 1.0, …, n-1.0] via mlx::core::arange.
// This is the row-index (iota) generator the rolling rewrite needs.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use polars_metal_mlx_sys::array::{mlx_array_eval, mlx_array_to_f32_vec};
use polars_metal_mlx_sys::scan::mlx_iota_f32;

#[test]
fn iota_f32_counts_from_zero() {
    let a = mlx_iota_f32(4).unwrap();
    mlx_array_eval(&[a.clone()]).unwrap();
    assert_eq!(mlx_array_to_f32_vec(&a).unwrap(), vec![0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn iota_f32_single_element() {
    let a = mlx_iota_f32(1).unwrap();
    mlx_array_eval(&[a.clone()]).unwrap();
    assert_eq!(mlx_array_to_f32_vec(&a).unwrap(), vec![0.0]);
}

#[test]
fn iota_f32_zero_length_is_empty() {
    let a = mlx_iota_f32(0).unwrap();
    mlx_array_eval(&[a.clone()]).unwrap();
    assert_eq!(mlx_array_to_f32_vec(&a).unwrap(), vec![]);
}
