//! M4 Phase 4 Task 20: random op chain + random F32 input -> MLX eval
//! matches pure-Rust scalar reference within ULP tolerance.
//!
//! Catches dtype/op-id wiring errors and MLX semantic divergences before
//! they slip into the engine. Covers a curated "safe set" of unary ops
//! whose output is meaningful for arbitrary F32 inputs; exotic
//! transcendentals (asin/acos for |x|>1, log/sqrt for negatives) are
//! excluded so the scalar reference is well-defined.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use polars_metal_native::fusion::scope::{FusionScope, InputDtype};
use polars_metal_native::fusion::subgraph::{ColumnBuffer, MlxSubgraph};
use polars_metal_native::fusion::supported_ops::OpId;
use proptest::prelude::*;

fn scalar_apply(op: OpId, x: f32) -> f32 {
    use OpId::*;
    match op {
        Neg => -x,
        Abs => x.abs(),
        Square => x * x,
        Sin => x.sin(),
        Cos => x.cos(),
        Tanh => x.tanh(),
        Exp => x.exp(),
        Floor => x.floor(),
        Ceil => x.ceil(),
        Round => x.round(),
        _ => panic!("unexpected op in safe set: {op:?}"),
    }
}

const SAFE_OPS: &[OpId] = &[
    OpId::Neg,
    OpId::Abs,
    OpId::Square,
    OpId::Sin,
    OpId::Cos,
    OpId::Tanh,
    OpId::Exp,
    OpId::Floor,
    OpId::Ceil,
    OpId::Round,
];

fn ulp_close(actual: f32, expected: f32, tol: f32) -> bool {
    if actual.is_nan() && expected.is_nan() {
        return true;
    }
    if !actual.is_finite() && !expected.is_finite() {
        return actual.is_sign_positive() == expected.is_sign_positive();
    }
    if !actual.is_finite() || !expected.is_finite() {
        return false;
    }
    let abs_diff = (actual - expected).abs();
    let rel_tol = expected.abs().max(1.0) * tol;
    abs_diff <= rel_tol.max(tol)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 32, .. ProptestConfig::default() })]

    #[test]
    fn random_unary_chain_matches_scalar(
        op_indices in prop::collection::vec(0usize..SAFE_OPS.len(), 1..6),
        input_seed in -100.0f32..100.0f32,
        n in 4usize..64usize,
    ) {
        // Bounded random input to keep Exp from overflowing inf in the
        // middle of the chain (e.g. Exp(Square(50)) is inf).
        let input: Vec<f32> = (0..n)
            .map(|i| input_seed * 0.01 + (i as f32) * 0.001)
            .collect();

        let mut scope = FusionScope::new();
        let mut cur = scope.add_input("x", InputDtype::F32);
        let ops: Vec<OpId> = op_indices.iter().map(|&i| SAFE_OPS[i]).collect();
        for op in &ops {
            cur = scope.push_op(*op, vec![cur]);
        }
        scope.mark_output(cur);

        let inputs = vec![ColumnBuffer::from_f32_vec(input.clone())];
        let actual = MlxSubgraph::from_fusion_scope(&scope, &inputs)
            .expect("build")
            .eval()
            .expect("eval")
            .into_iter()
            .next()
            .expect("output")
            .to_f32_vec()
            .expect("readback");

        prop_assert_eq!(actual.len(), input.len());
        for (i, &x) in input.iter().enumerate() {
            let mut expected = x;
            for op in &ops {
                expected = scalar_apply(*op, expected);
            }
            let tol = 1e-4;
            prop_assert!(
                ulp_close(actual[i], expected, tol),
                "mismatch at idx={} ops={:?} input={} expected={} actual={}",
                i, ops, x, expected, actual[i]
            );
        }
    }

    #[test]
    fn binary_add_matches_scalar(
        a in prop::collection::vec(-100.0f32..100.0f32, 1..32),
        b in prop::collection::vec(-100.0f32..100.0f32, 1..32),
    ) {
        let len = a.len().min(b.len());
        let a = a[..len].to_vec();
        let b = b[..len].to_vec();

        let mut scope = FusionScope::new();
        let ai = scope.add_input("a", InputDtype::F32);
        let bi = scope.add_input("b", InputDtype::F32);
        let s = scope.push_op(OpId::Add, vec![ai, bi]);
        scope.mark_output(s);

        let inputs = vec![
            ColumnBuffer::from_f32_vec(a.clone()),
            ColumnBuffer::from_f32_vec(b.clone()),
        ];
        let actual = MlxSubgraph::from_fusion_scope(&scope, &inputs)
            .expect("build")
            .eval()
            .expect("eval")
            .into_iter()
            .next()
            .expect("output")
            .to_f32_vec()
            .expect("readback");

        for i in 0..len {
            prop_assert!(
                ulp_close(actual[i], a[i] + b[i], 1e-5),
                "mismatch at idx={} expected={} actual={}", i, a[i] + b[i], actual[i]
            );
        }
    }
}
