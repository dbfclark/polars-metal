// crates/polars-metal-core/tests/test_router_cost.rs
//
// Cost-model rule tests. Each rule is a small pure function over
// (op_kind, n_rows) → NodeDecision. The thresholds here are M2's
// starting point per the spec § "Routing decisions (cost model)"; PRs
// that re-tune them update both the constants and the tests.
#![allow(clippy::expect_used)]

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
