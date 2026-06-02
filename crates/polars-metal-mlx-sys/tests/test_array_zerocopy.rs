// crates/polars-metal-mlx-sys/tests/test_array_zerocopy.rs
//! Construct an MlxArrayHandle as a zero-copy view over an existing MetalBuffer.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Arc;

use polars_metal_buffer::{MetalBuffer, MetalDevice};
use polars_metal_mlx_sys::array::{
    mlx_array_eval, mlx_array_to_f32_vec, mlx_array_view_metal_buffer, MlxDtype,
};

#[test]
fn zero_copy_view_round_trips() {
    let device = MetalDevice::system_default().expect("metal device");
    let input: Vec<f32> = (0..1000).map(|i| i as f32).collect();
    let buf = Arc::new(MetalBuffer::from_f32_slice(&device, &input).expect("metal buffer"));

    let view =
        mlx_array_view_metal_buffer(buf.clone(), &[1000], MlxDtype::F32).expect("view construct");
    mlx_array_eval(&[view.clone()]).expect("eval");
    let out = mlx_array_to_f32_vec(&view).expect("readback");

    assert_eq!(out.len(), input.len());
    for (a, b) in out.iter().zip(input.iter()) {
        assert_eq!(a, b, "zero-copy view should produce identical bytes");
    }
}

#[test]
fn view_construction_is_fast() {
    let device = MetalDevice::system_default().expect("metal device");
    let zeros: Vec<f32> = vec![0.0; 10_000_000];
    let buf = Arc::new(MetalBuffer::from_f32_slice(&device, &zeros).expect("metal buffer"));
    let t0 = std::time::Instant::now();
    let _view = mlx_array_view_metal_buffer(buf, &[10_000_000], MlxDtype::F32).expect("view");
    let elapsed = t0.elapsed();
    assert!(
        elapsed.as_micros() < 10_000,
        "view construction took {:?} (expected < 10ms; a 40MB memcpy would take 5-50ms — anything <10ms means we're not copying)",
        elapsed
    );
}
