// crates/polars-metal-mlx-sys/tests/test_int_readback.rs
use polars_metal_mlx_sys::array::{mlx_array_eval, mlx_array_from_f32_slice, MlxDtype};

#[test]
fn dtype_query_reports_f32() {
    let a = mlx_array_from_f32_slice(&[1.0, 2.0, 3.0]).expect("build f32");
    mlx_array_eval(&[a.clone()]).expect("eval");
    assert_eq!(a.dtype(), MlxDtype::F32);
}
