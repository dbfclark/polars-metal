// crates/polars-metal-mlx-sys/tests/test_elementwise.rs
//! Each elementwise op binding constructs a graph node; eval gives the right answer.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use polars_metal_mlx_sys::array::{
    mlx_array_eval, mlx_array_from_bool_slice, mlx_array_from_f32_slice, mlx_array_to_f32_vec,
};
use polars_metal_mlx_sys::elementwise::*;

#[test]
fn add_sub_mul_div() {
    let a = mlx_array_from_f32_slice(&[10.0, 20.0, 30.0]).unwrap();
    let b = mlx_array_from_f32_slice(&[3.0, 4.0, 5.0]).unwrap();
    let sum = mlx_add(&a, &b).unwrap();
    let diff = mlx_sub(&a, &b).unwrap();
    let prod = mlx_mul(&a, &b).unwrap();
    let quot = mlx_div(&a, &b).unwrap();
    mlx_array_eval(&[sum.clone(), diff.clone(), prod.clone(), quot.clone()]).unwrap();
    assert_eq!(mlx_array_to_f32_vec(&sum).unwrap(), vec![13.0, 24.0, 35.0]);
    assert_eq!(mlx_array_to_f32_vec(&diff).unwrap(), vec![7.0, 16.0, 25.0]);
    assert_eq!(
        mlx_array_to_f32_vec(&prod).unwrap(),
        vec![30.0, 80.0, 150.0]
    );
    let q = mlx_array_to_f32_vec(&quot).unwrap();
    assert!((q[0] - 10.0 / 3.0).abs() < 1e-6);
    assert!((q[1] - 5.0).abs() < 1e-6);
    assert!((q[2] - 6.0).abs() < 1e-6);
}

#[test]
fn neg_abs_square() {
    let a = mlx_array_from_f32_slice(&[1.0, -2.0, 3.0]).unwrap();
    let n = mlx_neg(&a).unwrap();
    let abs = mlx_abs(&a).unwrap();
    let sq = mlx_square(&a).unwrap();
    mlx_array_eval(&[n.clone(), abs.clone(), sq.clone()]).unwrap();
    assert_eq!(mlx_array_to_f32_vec(&n).unwrap(), vec![-1.0, 2.0, -3.0]);
    assert_eq!(mlx_array_to_f32_vec(&abs).unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(mlx_array_to_f32_vec(&sq).unwrap(), vec![1.0, 4.0, 9.0]);
}

#[test]
fn comparison_returns_bool() {
    // Construct two F32 arrays; the cmp result is Bool dtype. We can't
    // readback as F32 (dtype guard). Cast to F32 via a `where` against
    // 1.0/0.0 to verify.
    let a = mlx_array_from_f32_slice(&[1.0, 2.0, 3.0]).unwrap();
    let b = mlx_array_from_f32_slice(&[2.0, 2.0, 2.0]).unwrap();
    let lt_mask = mlx_lt(&a, &b).unwrap();
    let one = mlx_array_from_f32_slice(&[1.0, 1.0, 1.0]).unwrap();
    let zero = mlx_array_from_f32_slice(&[0.0, 0.0, 0.0]).unwrap();
    let result = mlx_where(&lt_mask, &one, &zero).unwrap();
    mlx_array_eval(&[result.clone()]).unwrap();
    // a < b: [1<2=true, 2<2=false, 3<2=false] -> [1.0, 0.0, 0.0]
    assert_eq!(mlx_array_to_f32_vec(&result).unwrap(), vec![1.0, 0.0, 0.0]);
}

#[test]
fn where_picks_per_element() {
    let cond = mlx_array_from_bool_slice(&[true, false, true]).unwrap();
    let then_v = mlx_array_from_f32_slice(&[10.0, 20.0, 30.0]).unwrap();
    let else_v = mlx_array_from_f32_slice(&[1.0, 2.0, 3.0]).unwrap();
    let r = mlx_where(&cond, &then_v, &else_v).unwrap();
    mlx_array_eval(&[r.clone()]).unwrap();
    assert_eq!(mlx_array_to_f32_vec(&r).unwrap(), vec![10.0, 2.0, 30.0]);
}

#[test]
fn logical_and_or_not() {
    // Closes the coverage gap on logical ops. Readback via where-cast since
    // the dtype guard in mlx_array_to_f32_vec rejects Bool arrays directly.
    let a = mlx_array_from_bool_slice(&[true, true, false, false]).unwrap();
    let b = mlx_array_from_bool_slice(&[true, false, true, false]).unwrap();
    let and = mlx_logical_and(&a, &b).unwrap();
    let or = mlx_logical_or(&a, &b).unwrap();
    let not_a = mlx_logical_not(&a).unwrap();

    let one = mlx_array_from_f32_slice(&[1.0, 1.0, 1.0, 1.0]).unwrap();
    let zero = mlx_array_from_f32_slice(&[0.0, 0.0, 0.0, 0.0]).unwrap();
    let and_f = mlx_where(&and, &one, &zero).unwrap();
    let or_f = mlx_where(&or, &one, &zero).unwrap();
    let not_a_f = mlx_where(&not_a, &one, &zero).unwrap();
    mlx_array_eval(&[and_f.clone(), or_f.clone(), not_a_f.clone()]).unwrap();

    assert_eq!(
        mlx_array_to_f32_vec(&and_f).unwrap(),
        vec![1.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        mlx_array_to_f32_vec(&or_f).unwrap(),
        vec![1.0, 1.0, 1.0, 0.0]
    );
    assert_eq!(
        mlx_array_to_f32_vec(&not_a_f).unwrap(),
        vec![0.0, 0.0, 1.0, 1.0]
    );
}
