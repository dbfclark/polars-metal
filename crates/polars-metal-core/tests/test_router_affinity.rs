// crates/polars-metal-core/tests/test_router_affinity.rs
//
// Affinity smoothing tests. The pass converts close-cost adjacent
// decisions into uniform runs (within a configurable threshold) to
// minimize gratuitous GPU↔CPU transitions. Spec § "Affinity smoothing"
// specifies the initial threshold at 20%.
#![allow(clippy::expect_used)]

use polars_metal_native::router::affinity::{smooth, SmoothingConfig};
use polars_metal_native::router::{LiftingPlan, NodeDecision, NodeId};

fn build(decisions: &[(&str, u32, NodeDecision)]) -> LiftingPlan {
    let mut p = LiftingPlan::new();
    for (kind, seq, d) in decisions {
        p.set(NodeId::new(*kind, *seq), d.clone());
    }
    p
}

#[test]
fn isolated_gpu_in_cpu_run_is_flipped_to_cpu_at_close_cost() {
    // Pattern: Scan(CPU) → GroupBy(GPU) → Filter(CPU). If groupby is
    // close-cost (within 20% of CPU), affinity may flip it to CPU.
    // The cost data driving this pass is queryable; for the unit test
    // we pre-tag the close_cost field.
    let plan = build(&[
        ("Scan", 0, NodeDecision::CpuLeave),
        ("GroupBy", 1, NodeDecision::GpuLift),
        ("Filter", 2, NodeDecision::CpuLeave),
    ]);
    let config = SmoothingConfig {
        window_pct: 20,
        close_cost_node_ids: vec![NodeId::new("GroupBy", 1)],
    };
    let smoothed = smooth(plan, &config);
    assert_eq!(
        smoothed.get(&NodeId::new("GroupBy", 1)),
        Some(&NodeDecision::CpuLeave)
    );
}

#[test]
fn far_cost_decisions_are_preserved() {
    // If GroupBy is decisively GPU (not in close_cost_node_ids), the
    // smoothing pass leaves it.
    let plan = build(&[
        ("Scan", 0, NodeDecision::CpuLeave),
        ("GroupBy", 1, NodeDecision::GpuLift),
        ("Filter", 2, NodeDecision::CpuLeave),
    ]);
    let config = SmoothingConfig {
        window_pct: 20,
        close_cost_node_ids: vec![],
    };
    let smoothed = smooth(plan, &config);
    assert_eq!(
        smoothed.get(&NodeId::new("GroupBy", 1)),
        Some(&NodeDecision::GpuLift)
    );
}

#[test]
fn fallback_is_never_smoothed() {
    let plan = build(&[
        ("Scan", 0, NodeDecision::CpuLeave),
        ("GroupBy", 1, NodeDecision::Fallback("string keys".into())),
        ("Filter", 2, NodeDecision::CpuLeave),
    ]);
    let config = SmoothingConfig {
        window_pct: 20,
        close_cost_node_ids: vec![NodeId::new("GroupBy", 1)],
    };
    let smoothed = smooth(plan, &config);
    assert!(matches!(
        smoothed.get(&NodeId::new("GroupBy", 1)),
        Some(NodeDecision::Fallback(_))
    ));
}
