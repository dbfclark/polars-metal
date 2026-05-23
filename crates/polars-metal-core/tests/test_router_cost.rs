// crates/polars-metal-core/tests/test_router_cost.rs
//
// Cost-model rule tests. Each rule is a small pure function over
// (op_kind, n_rows) → NodeDecision. The thresholds here are M2's
// starting point per the spec § "Routing decisions (cost model)"; PRs
// that re-tune them update both the constants and the tests.
#![allow(clippy::expect_used, clippy::panic)]

use polars_metal_native::plan::{AggExpr, AggOp, AggSpec, BinaryOp, MetalDtype};
use polars_metal_native::router::cost;
use polars_metal_native::router::NodeDecision;

#[test]
fn filter_routes_to_cpu_at_all_sizes() {
    // Spec: "Filter | CPU | always".
    assert_eq!(cost::decide_filter(0), NodeDecision::CpuLeave);
    assert_eq!(cost::decide_filter(1_000), NodeDecision::CpuLeave);
    assert_eq!(cost::decide_filter(100_000_000), NodeDecision::CpuLeave);
}

#[test]
fn groupby_routes_to_gpu_above_100k_rows() {
    // Spec: "GroupBy | GPU iff n_rows > 100_000".
    assert_eq!(cost::decide_groupby(50_000), NodeDecision::CpuLeave);
    assert_eq!(cost::decide_groupby(100_000), NodeDecision::CpuLeave);
    assert_eq!(cost::decide_groupby(100_001), NodeDecision::GpuLift);
    assert_eq!(cost::decide_groupby(10_000_000), NodeDecision::GpuLift);
}

#[test]
fn project_inherits_input_decision() {
    // Spec: "Project / SimpleProjection | follow input".
    assert_eq!(
        cost::decide_project(&NodeDecision::GpuLift),
        NodeDecision::GpuLift
    );
    assert_eq!(
        cost::decide_project(&NodeDecision::CpuLeave),
        NodeDecision::CpuLeave
    );
    // Fallback propagates up.
    assert!(matches!(
        cost::decide_project(&NodeDecision::Fallback("x".into())),
        NodeDecision::Fallback(_)
    ));
}

#[test]
fn scan_inherits_parent_decision() {
    // Spec: "Scan | follow output | inherit parent's decision". Here we
    // pass the parent decision as input; the function's contract matches
    // a top-down second walk if needed. Phase 1 implementation defaults
    // scan to CpuLeave (parent decision arrives in affinity smoothing).
    assert_eq!(cost::decide_scan_initial(), NodeDecision::CpuLeave);
}

#[test]
fn thresholds_are_named_constants_for_pr_tuning() {
    // The threshold MUST live as a named pub constant so PRs that retune
    // it touch exactly one line (per spec § "Routing decisions" — "cost
    // data and the implementation live in the same Rust module so they
    // evolve together by PR").
    assert_eq!(cost::GROUPBY_GPU_MIN_ROWS, 100_000);
}

#[test]
fn groupby_with_composite_key_at_or_below_128_bits_routes_to_gpu() {
    // Spec: Bool (1+1=2) + I64 (1+64=65) = 67 bits, within 128-bit budget.
    let keys = vec![
        ("category".into(), MetalDtype::Bool),
        ("id".into(), MetalDtype::I64),
    ];
    let aggs: Vec<AggSpec> = vec![];
    let d = cost::decide_groupby_with_keys(1_000_000, &keys, &aggs);
    assert_eq!(d, NodeDecision::GpuLift);
}

#[test]
fn groupby_with_oversized_composite_key_falls_back_at_plan_time() {
    // Spec: 3 × I64 = 3 × (1+64) = 195 bits, exceeds 128-bit budget.
    let keys = vec![
        ("a".into(), MetalDtype::I64),
        ("b".into(), MetalDtype::I64),
        ("c".into(), MetalDtype::I64),
    ];
    let aggs: Vec<AggSpec> = vec![];
    let d = cost::decide_groupby_with_keys(1_000_000, &keys, &aggs);
    assert!(matches!(&d, NodeDecision::Fallback(_)));
    if let NodeDecision::Fallback(reason) = d {
        assert!(
            reason.contains("128"),
            "reason should mention 128-bit limit: {reason}"
        );
        assert!(
            reason.contains("195"),
            "reason should mention total bits 195: {reason}"
        );
    }
}

#[test]
fn router_falls_back_when_any_agg_is_expression() {
    // Phase 2 gate (Task 10): any AggSpec::Expression in the agg list
    // must route to CPU until the Phase 3 fused-kernel consumer lands.
    // Row count is well above GROUPBY_GPU_MIN_ROWS, key width is
    // trivially in budget — only the Expression spec triggers fallback.
    let keys = vec![("k".to_string(), MetalDtype::I64)];
    let aggs = vec![
        AggSpec::Simple {
            input_col: "v".into(),
            op: AggOp::Sum,
            output_alias: "v_sum".into(),
        },
        AggSpec::Expression {
            expr: AggExpr::Binary {
                op: BinaryOp::Mul,
                lhs: Box::new(AggExpr::Column("a".into())),
                rhs: Box::new(AggExpr::Column("b".into())),
            },
            op: AggOp::Sum,
            output_alias: "sum_ab".into(),
        },
    ];
    let decision = cost::decide_groupby_with_keys(1_000_000, &keys, &aggs);
    match decision {
        NodeDecision::Fallback(reason) => {
            assert!(
                reason.contains("Expression"),
                "expected Expression reason, got: {reason}"
            );
        }
        other => panic!("expected Fallback, got: {other:?}"),
    }
}

#[test]
fn router_passes_when_only_simple_and_length_aggs() {
    // Counter-test: with no Expression specs, the router still lifts
    // GroupBy to GPU at large row counts (M2-shape queries unchanged).
    let keys = vec![("k".to_string(), MetalDtype::I64)];
    let aggs = vec![
        AggSpec::Simple {
            input_col: "v".into(),
            op: AggOp::Sum,
            output_alias: "v_sum".into(),
        },
        AggSpec::Length {
            output_alias: "n".into(),
        },
    ];
    let decision = cost::decide_groupby_with_keys(1_000_000, &keys, &aggs);
    assert!(matches!(decision, NodeDecision::GpuLift));
}
