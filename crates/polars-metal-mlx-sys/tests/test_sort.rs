//! M4 Phase 1 Task 9: sort + argpartition.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use polars_metal_mlx_sys::array::*;
use polars_metal_mlx_sys::elementwise::{mlx_cast, mlx_neg};
use polars_metal_mlx_sys::sort::*;

#[test]
fn sort_ascending() {
    let a = mlx_array_from_f32_slice(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]).unwrap();
    let s = mlx_sort(&a).unwrap();
    mlx_array_eval(&[s.clone()]).unwrap();
    assert_eq!(
        mlx_array_to_f32_vec(&s).unwrap(),
        vec![1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0]
    );
}

#[test]
fn argpartition_top_3_via_neg() {
    // Top-K = argpartition of negated values: the first kth+1 positions
    // hold indices of the largest kth+1 input values.
    let a = mlx_array_from_f32_slice(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]).unwrap();
    let neg = mlx_neg(&a).unwrap();
    let idx_i32 = mlx_argpartition(&neg, 2).unwrap();
    let idx_f32 = mlx_cast(&idx_i32, MlxDtype::F32).unwrap();
    mlx_array_eval(&[idx_f32.clone()]).unwrap();
    let idxs = mlx_array_to_f32_vec(&idx_f32).unwrap();

    // First 3 positions hold the indices of the 3 largest values.
    // Values: a[5]=9.0, a[7]=6.0, a[4]=5.0 are the top 3.
    use std::collections::HashSet;
    let top3: HashSet<i32> = idxs[..3].iter().map(|&f| f as i32).collect();
    let expected: HashSet<i32> = [5, 7, 4].iter().copied().collect();
    assert_eq!(top3, expected);
}
