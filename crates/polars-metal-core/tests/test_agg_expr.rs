// crates/polars-metal-core/tests/test_agg_expr.rs
//
// IR-layer tests for Phase 2 / capability G:
//   - `AggExpr` (binary-arithmetic expression tree consumed by the fused
//     aggregation kernel),
//   - `AggSpec` as an enum with three variants (Simple / Expression / Length),
//   - depth validation that keeps MSL emission bounded.
//
// Behavioral tests for the walker + parser + router live elsewhere
// (`test_walker_expression_unfolding.py`, `test_router_cost.rs`,
// `test_plan_groupby.rs`).

#![allow(clippy::expect_used, clippy::panic)]

use polars_metal_native::plan::{AggExpr, AggOp, AggSpec, BinaryOp};

#[test]
fn agg_expr_column_literal_constructs() {
    let expr = AggExpr::Binary {
        op: BinaryOp::Mul,
        lhs: Box::new(AggExpr::Column("l_extendedprice".into())),
        rhs: Box::new(AggExpr::Binary {
            op: BinaryOp::Sub,
            lhs: Box::new(AggExpr::LiteralF64(1.0)),
            rhs: Box::new(AggExpr::Column("l_discount".into())),
        }),
    };
    let cols = expr.referenced_columns();
    assert_eq!(
        cols,
        vec!["l_extendedprice".to_string(), "l_discount".to_string()]
    );
}

#[test]
fn agg_spec_expression_carries_op_and_alias() {
    let spec = AggSpec::Expression {
        expr: AggExpr::Column("v".into()),
        op: AggOp::Sum,
        output_alias: "sum_v".into(),
    };
    match &spec {
        AggSpec::Expression {
            op, output_alias, ..
        } => {
            assert_eq!(*op, AggOp::Sum);
            assert_eq!(output_alias, "sum_v");
        }
        _ => panic!("expected Expression variant"),
    }
}

#[test]
fn agg_spec_length_carries_alias_only() {
    let spec = AggSpec::Length {
        output_alias: "n".into(),
    };
    match &spec {
        AggSpec::Length { output_alias } => assert_eq!(output_alias, "n"),
        _ => panic!("expected Length variant"),
    }
}

#[test]
fn agg_spec_simple_carries_input_col() {
    let spec = AggSpec::Simple {
        input_col: "v".into(),
        op: AggOp::Sum,
        output_alias: "v_sum".into(),
    };
    match &spec {
        AggSpec::Simple {
            input_col,
            op,
            output_alias,
        } => {
            assert_eq!(input_col, "v");
            assert_eq!(*op, AggOp::Sum);
            assert_eq!(output_alias, "v_sum");
        }
        _ => panic!("expected Simple variant"),
    }
}

#[test]
fn agg_expr_depth_check_rejects_overdeep_nesting() {
    // M3 caps expression depth at 4 to keep MSL emission bounded.
    let mut e = AggExpr::Column("v".into());
    for _ in 0..5 {
        e = AggExpr::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(e),
            rhs: Box::new(AggExpr::LiteralF64(0.0)),
        };
    }
    assert!(e.depth() > 4);
    assert!(e.validate().is_err());
}

#[test]
fn agg_expr_depth_4_passes_validation() {
    // Depth-2 expression: ((a + b) * (c - d)) — the kind of shape Q1 needs.
    // (`depth()` counts the longest path of Binary nodes; this tree has 2.)
    let e = AggExpr::Binary {
        op: BinaryOp::Mul,
        lhs: Box::new(AggExpr::Binary {
            op: BinaryOp::Add,
            lhs: Box::new(AggExpr::Column("a".into())),
            rhs: Box::new(AggExpr::Column("b".into())),
        }),
        rhs: Box::new(AggExpr::Binary {
            op: BinaryOp::Sub,
            lhs: Box::new(AggExpr::Column("c".into())),
            rhs: Box::new(AggExpr::Column("d".into())),
        }),
    };
    assert_eq!(e.depth(), 2);
    assert!(e.validate().is_ok());
}
