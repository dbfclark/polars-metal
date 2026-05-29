//! M4 Phase 4 Task 19: zero-copy MetalBuffer bridge.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use polars_metal_buffer::{MetalBuffer, MetalDevice};
use polars_metal_native::fusion::scope::{FusionScope, InputDtype};
use polars_metal_native::fusion::subgraph::MlxSubgraph;
use polars_metal_native::fusion::supported_ops::OpId;

#[test]
fn build_subgraph_over_metal_buffer_inputs() {
    let device = MetalDevice::system_default().expect("metal device");
    let buf = Arc::new(
        MetalBuffer::from_f32_slice(&device, &[1.0, 4.0, 9.0, 16.0]).expect("from_f32_slice"),
    );

    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::F32);
    let s = scope.push_op(OpId::Sqrt, vec![a]);
    scope.mark_output(s);

    let subgraph = MlxSubgraph::from_fusion_scope_buffers(&scope, &[buf]).expect("build");
    let outputs = subgraph.eval_to_metal_buffers(&device).expect("eval");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].to_f32_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn build_subgraph_with_two_inputs() {
    let device = MetalDevice::system_default().expect("metal device");
    let a = Arc::new(MetalBuffer::from_f32_slice(&device, &[1.0, 2.0, 3.0]).unwrap());
    let b = Arc::new(MetalBuffer::from_f32_slice(&device, &[10.0, 20.0, 30.0]).unwrap());

    let mut scope = FusionScope::new();
    let ai = scope.add_input("a", InputDtype::F32);
    let bi = scope.add_input("b", InputDtype::F32);
    let m = scope.push_op(OpId::Mul, vec![ai, bi]);
    scope.mark_output(m);

    let outputs = MlxSubgraph::from_fusion_scope_buffers(&scope, &[a, b])
        .expect("build")
        .eval_to_metal_buffers(&device)
        .expect("eval");
    assert_eq!(outputs[0].to_f32_vec(), vec![10.0, 40.0, 90.0]);
}

#[test]
fn input_count_mismatch_errors() {
    let device = MetalDevice::system_default().expect("metal device");
    let buf = Arc::new(MetalBuffer::from_f32_slice(&device, &[1.0]).unwrap());

    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::F32);
    let b = scope.add_input("b", InputDtype::F32);
    let s = scope.push_op(OpId::Add, vec![a, b]);
    scope.mark_output(s);

    let result = MlxSubgraph::from_fusion_scope_buffers(&scope, &[buf]);
    assert!(result.is_err());
}
