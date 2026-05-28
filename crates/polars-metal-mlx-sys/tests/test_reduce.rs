//! M4 Phase 1 Task 8: reduction bindings.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use polars_metal_mlx_sys::array::*;
use polars_metal_mlx_sys::elementwise::mlx_cast;
use polars_metal_mlx_sys::reduce::*;

const TOL: f32 = 1e-3;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}

#[test]
fn sum_mean_min_max() {
    let a = mlx_array_from_f32_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let s = mlx_sum(&a).unwrap();
    let m = mlx_mean(&a).unwrap();
    let mn = mlx_min(&a).unwrap();
    let mx = mlx_max(&a).unwrap();
    mlx_array_eval(&[s.clone(), m.clone(), mn.clone(), mx.clone()]).unwrap();
    assert_eq!(mlx_array_to_f32_vec(&s).unwrap(), vec![15.0]);
    assert_eq!(mlx_array_to_f32_vec(&m).unwrap(), vec![3.0]);
    assert_eq!(mlx_array_to_f32_vec(&mn).unwrap(), vec![1.0]);
    assert_eq!(mlx_array_to_f32_vec(&mx).unwrap(), vec![5.0]);
}

#[test]
fn std_var_population() {
    // [2, 4, 4, 4, 5, 5, 7, 9]: pop var = 4, pop std = 2 (ddof=0)
    let a = mlx_array_from_f32_slice(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]).unwrap();
    let s = mlx_std(&a).unwrap();
    let v = mlx_var(&a).unwrap();
    mlx_array_eval(&[s.clone(), v.clone()]).unwrap();
    let sv = mlx_array_to_f32_vec(&s).unwrap()[0];
    let vv = mlx_array_to_f32_vec(&v).unwrap()[0];
    assert!(close(vv, 4.0, TOL), "var={vv}, expected ~4.0");
    assert!(close(sv, 2.0, TOL), "std={sv}, expected ~2.0");
}

#[test]
fn argmin_argmax_via_cast() {
    // Indices are I32; cast to F32 to read back.
    let a = mlx_array_from_f32_slice(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]).unwrap();
    let amin_i32 = mlx_argmin(&a).unwrap();
    let amax_i32 = mlx_argmax(&a).unwrap();
    let amin_f32 = mlx_cast(&amin_i32, MlxDtype::F32).unwrap();
    let amax_f32 = mlx_cast(&amax_i32, MlxDtype::F32).unwrap();
    mlx_array_eval(&[amin_f32.clone(), amax_f32.clone()]).unwrap();
    // argmin: first occurrence of 1.0 at index 1
    assert_eq!(mlx_array_to_f32_vec(&amin_f32).unwrap(), vec![1.0]);
    // argmax: 9.0 at index 5
    assert_eq!(mlx_array_to_f32_vec(&amax_f32).unwrap(), vec![5.0]);
}
