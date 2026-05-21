// crates/polars-metal-core/tests/test_router_types.rs
//
// Sanity tests for the LiftingPlan / NodeDecision types — construction,
// equality, and the trivial "fallback poisons ancestors" predicate.
#![allow(clippy::expect_used, clippy::panic)]

use polars_metal_native::router::{LiftingPlan, NodeDecision, NodeId};

#[test]
fn node_decision_variants_construct() {
    let _ = NodeDecision::GpuLift;
    let _ = NodeDecision::CpuLeave;
    let _ = NodeDecision::Fallback("unsupported IR".to_string());
}

#[test]
fn lifting_plan_records_per_node_decisions() {
    let mut plan = LiftingPlan::new();
    let scan_id = NodeId::new("Scan", 0);
    let filter_id = NodeId::new("Filter", 1);
    plan.set(scan_id.clone(), NodeDecision::CpuLeave);
    plan.set(filter_id.clone(), NodeDecision::CpuLeave);
    assert_eq!(plan.get(&scan_id), Some(&NodeDecision::CpuLeave));
    assert_eq!(plan.get(&filter_id), Some(&NodeDecision::CpuLeave));
    assert_eq!(plan.len(), 2);
}

#[test]
fn fallback_carries_human_readable_reason() {
    let d = NodeDecision::Fallback("composite key > 128 bits".into());
    match d {
        NodeDecision::Fallback(reason) => assert!(reason.contains("128")),
        _ => panic!("expected Fallback"),
    }
}

#[test]
fn node_id_round_trips_kind_and_sequence() {
    let id = NodeId::new("GroupBy", 7);
    assert_eq!(id.kind(), "GroupBy");
    assert_eq!(id.seq(), 7);
}
