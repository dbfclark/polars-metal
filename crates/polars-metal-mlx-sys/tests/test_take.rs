#![allow(clippy::unwrap_used, clippy::expect_used)]
use polars_metal_mlx_sys::array::{
    mlx_array_eval, mlx_array_from_f32_slice, mlx_array_to_f32_vec, MlxArrayHandle, MlxDtype,
};
use polars_metal_mlx_sys::elementwise::mlx_cast;
use polars_metal_mlx_sys::shape::mlx_take;

fn make_f32(data: &[f32]) -> MlxArrayHandle {
    mlx_array_from_f32_slice(data).expect("build f32 array")
}

fn make_i32(data: &[i32]) -> MlxArrayHandle {
    // No direct i32 constructor in this crate; build f32 then cast to I32.
    let as_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let f = mlx_array_from_f32_slice(&as_f32).expect("build f32 array");
    mlx_cast(&f, MlxDtype::I32).expect("cast to i32")
}

#[test]
fn take_1d_gathers_by_index() {
    // source = [10,20,30,40]; idx = [3,0,0,2] -> [40,10,10,30]
    let src = make_f32(&[10.0, 20.0, 30.0, 40.0]);
    let idx = make_i32(&[3, 0, 0, 2]);
    let out = mlx_take(&src, &idx).expect("take dispatch");
    mlx_array_eval(&[out.clone()]).expect("eval");
    assert_eq!(
        mlx_array_to_f32_vec(&out).expect("readback"),
        vec![40.0, 10.0, 10.0, 30.0]
    );
}
