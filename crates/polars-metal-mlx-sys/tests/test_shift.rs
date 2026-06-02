// M5 rolling Task 1: mlx_shift FFI binding.
//
// Forward-shift a 1-D F32 array by `shift` positions along axis 0, filling the
// vacated front positions with 0.0. Output length equals input length.
// `shift >= n` produces an all-zero array (clamped).
#![allow(clippy::expect_used, clippy::unwrap_used)]

use polars_metal_mlx_sys::array::{mlx_array_eval, mlx_array_from_f32_slice, mlx_array_to_f32_vec};
use polars_metal_mlx_sys::scan::mlx_shift;

#[test]
fn shift_by_2_zero_pads_front() {
    let a = mlx_array_from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let s = mlx_shift(&a, 2).unwrap(); // forward shift by 2, zero-fill front
    mlx_array_eval(&[s.clone()]).unwrap();
    assert_eq!(
        mlx_array_to_f32_vec(&s).unwrap(),
        vec![0.0, 0.0, 1.0, 2.0, 3.0]
    );
}

#[test]
fn shift_ge_len_is_all_zero() {
    let a = mlx_array_from_f32_slice(&[1.0, 2.0, 3.0]).unwrap();
    let s = mlx_shift(&a, 5).unwrap();
    mlx_array_eval(&[s.clone()]).unwrap();
    assert_eq!(mlx_array_to_f32_vec(&s).unwrap(), vec![0.0, 0.0, 0.0]);
}

#[test]
fn shift_by_zero_is_identity() {
    let a = mlx_array_from_f32_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let s = mlx_shift(&a, 0).unwrap();
    mlx_array_eval(&[s.clone()]).unwrap();
    assert_eq!(mlx_array_to_f32_vec(&s).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn shift_by_1_single_element() {
    let a = mlx_array_from_f32_slice(&[42.0]).unwrap();
    let s = mlx_shift(&a, 1).unwrap();
    mlx_array_eval(&[s.clone()]).unwrap();
    assert_eq!(mlx_array_to_f32_vec(&s).unwrap(), vec![0.0]);
}
