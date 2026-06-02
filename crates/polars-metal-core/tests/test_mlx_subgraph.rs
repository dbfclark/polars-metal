//! M4 Phase 4 Task 18 + M5 Task 3: build + eval a small MLX subgraph from a FusionScope.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use polars_metal_native::fusion::scope::{FusionScope, InputDtype};
use polars_metal_native::fusion::subgraph::{BuildError, ColumnBuffer, MlxSubgraph};
use polars_metal_native::fusion::supported_ops::OpId;

fn f32_input(data: Vec<f32>) -> ColumnBuffer {
    ColumnBuffer::from_f32_vec(data)
}

#[test]
fn build_and_eval_sin_cos_mul() {
    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::F32);
    let sin_a = scope.push_op(OpId::Sin, vec![a]);
    let cos_a = scope.push_op(OpId::Cos, vec![a]);
    let out = scope.push_op(OpId::Mul, vec![sin_a, cos_a]);
    scope.mark_output(out);

    let inputs = vec![f32_input((0..100).map(|i| i as f32 * 0.01).collect())];
    let subgraph = MlxSubgraph::from_fusion_scope(&scope, &inputs).expect("build");
    let outputs = subgraph.eval().expect("eval");
    assert_eq!(outputs.len(), 1);
    let out_vec = outputs[0].to_f32_vec().expect("read back");
    assert_eq!(out_vec.len(), 100);
    // sin(0)*cos(0) = 0
    assert!(out_vec[0].abs() < 1e-6, "got {}", out_vec[0]);
    // sin(0.5)*cos(0.5) = 0.5 * sin(1.0) ≈ 0.4207
    let expected_50 = (0.5_f32).sin() * (0.5_f32).cos();
    assert!((out_vec[50] - expected_50).abs() < 1e-5);
}

#[test]
fn build_with_two_inputs_and_add() {
    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::F32);
    let b = scope.add_input("b", InputDtype::F32);
    let sum = scope.push_op(OpId::Add, vec![a, b]);
    scope.mark_output(sum);

    let inputs = vec![
        f32_input(vec![1.0, 2.0, 3.0]),
        f32_input(vec![10.0, 20.0, 30.0]),
    ];
    let subgraph = MlxSubgraph::from_fusion_scope(&scope, &inputs).expect("build");
    let outputs = subgraph.eval().expect("eval");
    assert_eq!(outputs[0].to_f32_vec().unwrap(), vec![11.0, 22.0, 33.0]);
}

#[test]
fn build_a_log_sqrt_chain() {
    // log(x).sqrt() over [e, e^2, e^4]: results sqrt(1), sqrt(2), sqrt(4) = 1, ~1.414, 2.
    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::F32);
    let l = scope.push_op(OpId::Log, vec![a]);
    let s = scope.push_op(OpId::Sqrt, vec![l]);
    scope.mark_output(s);

    let e = std::f32::consts::E;
    let inputs = vec![f32_input(vec![e, e * e, e * e * e * e])];
    let outputs = MlxSubgraph::from_fusion_scope(&scope, &inputs)
        .expect("build")
        .eval()
        .expect("eval");
    let v = outputs[0].to_f32_vec().unwrap();
    assert!((v[0] - 1.0).abs() < 1e-4);
    assert!((v[1] - 2.0_f32.sqrt()).abs() < 1e-4);
    assert!((v[2] - 2.0).abs() < 1e-4);
}

#[test]
fn input_count_mismatch_errors() {
    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::F32);
    let _ = scope.push_op(OpId::Sqrt, vec![a]);
    scope.mark_output(a);

    let result = MlxSubgraph::from_fusion_scope(&scope, &[]);
    assert!(result.is_err());
}

#[test]
fn argpartition_is_unsupported_in_phase4() {
    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::F32);
    let p = scope.push_op(OpId::ArgPartition, vec![a]);
    scope.mark_output(p);

    let inputs = vec![f32_input(vec![3.0, 1.0, 2.0])];
    let result = MlxSubgraph::from_fusion_scope(&scope, &inputs);
    assert!(matches!(
        result,
        Err(BuildError::UnsupportedOp(OpId::ArgPartition))
    ));
}

#[test]
fn build_is_fast_for_large_chains() {
    // A 20-op chain on 10M F32 inputs. Build (graph construction) should be
    // sub-200µs - we're just chaining MLX FFI calls, no actual compute yet.
    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::F32);
    let mut cur = a;
    for op in &[
        OpId::Sin,
        OpId::Cos,
        OpId::Tan,
        OpId::Sqrt,
        OpId::Log,
        OpId::Exp,
        OpId::Abs,
        OpId::Floor,
        OpId::Ceil,
        OpId::Round,
        OpId::Square,
        OpId::Sinh,
        OpId::Cosh,
        OpId::Tanh,
        OpId::Asin,
        OpId::Acos,
        OpId::Atan,
        OpId::Log10,
        OpId::Log1p,
        OpId::Sqrt,
    ] {
        cur = scope.push_op(*op, vec![cur]);
    }
    scope.mark_output(cur);

    let inputs = vec![f32_input(vec![0.5; 10_000_000])];
    let t0 = std::time::Instant::now();
    let _subgraph = MlxSubgraph::from_fusion_scope(&scope, &inputs).expect("build");
    let elapsed = t0.elapsed();
    // Allow generous headroom; the build is dominated by one mlx_array_from_f32_slice
    // (which copies 10M F32s into MLX-owned memory). Pure graph construction is
    // sub-ms; we assert <500ms to account for the input materialization cost.
    assert!(elapsed.as_millis() < 500, "build took {elapsed:?}");
}

#[test]
fn subgraph_shift_zero_pads() {
    // [1,2,3,4] shift-by-2 → [0,0,1,2]
    let mut scope = FusionScope::new();
    let a = scope.add_input("x", InputDtype::F32);
    let sh = scope.push_op_param(OpId::Shift, vec![a], Some(2));
    scope.mark_output(sh);

    let inputs = vec![f32_input(vec![1.0, 2.0, 3.0, 4.0])];
    let subgraph = MlxSubgraph::from_fusion_scope(&scope, &inputs).expect("build");
    let outputs = subgraph.eval().expect("eval");
    let got = outputs[0].to_f32_vec().unwrap();
    assert_eq!(got, vec![0.0, 0.0, 1.0, 2.0], "got {got:?}");
}
