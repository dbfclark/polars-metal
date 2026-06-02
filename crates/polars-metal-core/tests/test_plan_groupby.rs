// crates/polars-metal-core/tests/test_plan_groupby.rs
//
// Construction sanity tests for the GroupBy IR variant. Confirms the
// types compile, support equality where useful, and round-trip through
// Debug formatting. Behavioral tests for the router's handling of
// GroupBy land in Task 12 (PyO3 wire format).
#![allow(clippy::expect_used, clippy::panic)]

use polars_metal_native::plan::{AggOp, AggSpec, MetalDtype, MetalPlanNode};

fn scan(n_rows: usize) -> MetalPlanNode {
    MetalPlanNode::Scan {
        n_rows,
        columns: vec![("k".into(), MetalDtype::I64), ("v".into(), MetalDtype::F64)],
    }
}

#[test]
fn groupby_variant_constructs() {
    let plan = MetalPlanNode::GroupBy {
        input: Box::new(scan(1_000_000)),
        keys: vec![("k".into(), MetalDtype::I64)],
        aggs: vec![AggSpec::Simple {
            input_col: "v".into(),
            op: AggOp::Sum,
            output_alias: "v_sum".into(),
        }],
    };
    // Smoke test — Debug must format without panicking.
    let _ = format!("{plan:?}");
}

#[test]
fn agg_op_variants_all_present() {
    // Spec § "Aggregations delivered" — six entry points.
    let ops = [
        AggOp::Sum,
        AggOp::Mean,
        AggOp::Count,
        AggOp::Min,
        AggOp::Max,
        AggOp::Len,
    ];
    assert_eq!(ops.len(), 6);
}

#[test]
fn agg_op_equality_distinguishes_variants() {
    assert_eq!(AggOp::Sum, AggOp::Sum);
    assert_ne!(AggOp::Sum, AggOp::Mean);
    assert_ne!(AggOp::Count, AggOp::Len);
}

#[test]
fn agg_spec_carries_all_three_fields() {
    let spec = AggSpec::Simple {
        input_col: "price".into(),
        op: AggOp::Mean,
        output_alias: "avg_price".into(),
    };
    match spec {
        AggSpec::Simple {
            input_col,
            op,
            output_alias,
        } => {
            assert_eq!(input_col, "price");
            assert_eq!(op, AggOp::Mean);
            assert_eq!(output_alias, "avg_price");
        }
        _ => panic!("expected Simple variant"),
    }
}

#[test]
fn groupby_supports_multiple_keys_and_aggs() {
    let plan = MetalPlanNode::GroupBy {
        input: Box::new(scan(10_000_000)),
        keys: vec![
            ("returnflag".into(), MetalDtype::I64),
            ("linestatus".into(), MetalDtype::I64),
        ],
        aggs: vec![
            AggSpec::Simple {
                input_col: "qty".into(),
                op: AggOp::Sum,
                output_alias: "sum_qty".into(),
            },
            AggSpec::Simple {
                input_col: "qty".into(),
                op: AggOp::Mean,
                output_alias: "avg_qty".into(),
            },
            AggSpec::Simple {
                input_col: "price".into(),
                op: AggOp::Sum,
                output_alias: "sum_price".into(),
            },
            AggSpec::Simple {
                input_col: "price".into(),
                op: AggOp::Min,
                output_alias: "min_price".into(),
            },
            AggSpec::Simple {
                input_col: "price".into(),
                op: AggOp::Max,
                output_alias: "max_price".into(),
            },
            AggSpec::Simple {
                input_col: "qty".into(),
                op: AggOp::Count,
                output_alias: "count_qty".into(),
            },
            AggSpec::Length {
                output_alias: "n_rows".into(),
            },
        ],
    };
    match plan {
        MetalPlanNode::GroupBy { keys, aggs, .. } => {
            assert_eq!(keys.len(), 2);
            assert_eq!(aggs.len(), 7);
        }
        _ => panic!("expected GroupBy"),
    }
}
