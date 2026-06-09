//! B1 Task 5: dtype-aware buffer-path subgraph (int inputs/outputs).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use polars_metal_buffer::{MetalBuffer, MetalDevice};
use polars_metal_native::fusion::scope::{FusionScope, InputDtype};
use polars_metal_native::fusion::subgraph::MlxSubgraph;
use polars_metal_native::fusion::supported_ops::OpId;

#[test]
fn i32_identity_subgraph_round_trips() {
    let device = MetalDevice::system_default().expect("metal");
    let vals: Vec<i32> = vec![-7, 0, 1, 100, 2_000_000_000];
    let buf = Arc::new(MetalBuffer::from_i32_slice(&device, &vals).expect("stage"));

    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::I32);
    scope.mark_output(a);

    let sg = MlxSubgraph::from_fusion_scope_buffers(&scope, &[buf]).expect("build");
    let outs = sg.eval_to_metal_buffers(&device).expect("eval");
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].to_i32_vec(), vals);
}

/// Exercises a real arithmetic op so the dtype actually matters: multiplying
/// two i32 columns must use integer semantics. With the old F32 hard-code the
/// input bit-patterns would be read as floats, multiplied as floats, then
/// read back as garbage — so this assertion forces the dtype-aware path.
#[test]
fn i32_mul_subgraph_uses_integer_semantics() {
    let device = MetalDevice::system_default().expect("metal");
    let a_vals: Vec<i32> = vec![-7, 0, 1, 100, 12345];
    let b_vals: Vec<i32> = vec![3, 9, 1000, -4, 2];
    let expect: Vec<i32> = a_vals.iter().zip(&b_vals).map(|(a, b)| a * b).collect();

    let a_buf = Arc::new(MetalBuffer::from_i32_slice(&device, &a_vals).expect("stage a"));
    let b_buf = Arc::new(MetalBuffer::from_i32_slice(&device, &b_vals).expect("stage b"));

    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::I32);
    let b = scope.add_input("b", InputDtype::I32);
    let m = scope.push_op(OpId::Mul, vec![a, b]);
    scope.mark_output(m);

    let sg = MlxSubgraph::from_fusion_scope_buffers(&scope, &[a_buf, b_buf]).expect("build");
    let outs = sg.eval_to_metal_buffers(&device).expect("eval");
    assert_eq!(outs.len(), 1);
    assert_eq!(outs[0].to_i32_vec(), expect);
}
