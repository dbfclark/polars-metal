// crates/polars-metal-mlx-sys/tests/test_int_readback.rs
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Arc;

use polars_metal_buffer::{MetalBuffer, MetalDevice};
use polars_metal_mlx_sys::array::{
    mlx_array_eval, mlx_array_from_f32_slice, mlx_array_to_i64_vec, mlx_array_to_u64_vec,
    mlx_array_view_metal_buffer, MlxDtype,
};

#[test]
fn dtype_query_reports_f32() {
    let a = mlx_array_from_f32_slice(&[1.0, 2.0, 3.0]).expect("build f32");
    mlx_array_eval(&[a.clone()]).expect("eval");
    assert_eq!(a.dtype().expect("dtype"), MlxDtype::F32);
}

#[test]
fn i64_view_round_trips() {
    let device = MetalDevice::system_default().expect("metal");
    let vals: Vec<i64> = vec![-3, 0, 5, 3_000_000_000, -2_000_000_000];
    let buf = Arc::new(MetalBuffer::from_i64_slice(&device, &vals).expect("stage"));
    let h = mlx_array_view_metal_buffer(buf, &[vals.len() as i64], MlxDtype::I64).expect("view");
    mlx_array_eval(&[h.clone()]).expect("eval");
    assert_eq!(h.dtype().expect("dtype"), MlxDtype::I64);
    assert_eq!(mlx_array_to_i64_vec(&h).expect("readback"), vals);
}

#[test]
fn u64_view_round_trips_beyond_i64_range() {
    let device = MetalDevice::system_default().expect("metal");
    let vals: Vec<u64> = vec![0, 1, u64::MAX, 10_000_000_000_000_000_000];
    let buf = Arc::new(MetalBuffer::from_u64_slice(&device, &vals).expect("stage"));
    let h = mlx_array_view_metal_buffer(buf, &[vals.len() as i64], MlxDtype::U64).expect("view");
    mlx_array_eval(&[h.clone()]).expect("eval");
    assert_eq!(h.dtype().expect("dtype"), MlxDtype::U64);
    assert_eq!(mlx_array_to_u64_vec(&h).expect("readback"), vals);
}
