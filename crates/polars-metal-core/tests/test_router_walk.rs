// crates/polars-metal-core/tests/test_router_walk.rs
//
// End-to-end LiftingPlan-from-MetalPlanNode-tree tests. We construct
// synthetic trees (Scan → Filter → Project, Scan → GroupBy, etc.) and
// assert the per-node decisions match the spec.
#![allow(clippy::expect_used)]

use polars_metal_native::plan::{MetalDtype, MetalPlanNode, PredicateAst};
use polars_metal_native::router::{compute_lifting_plan, NodeDecision, NodeId};

fn scan(n_rows: usize) -> MetalPlanNode {
    MetalPlanNode::Scan {
        n_rows,
        columns: vec![("a".into(), MetalDtype::I64)],
    }
}

#[test]
fn filter_over_scan_is_cpu_leave_throughout() {
    let plan = MetalPlanNode::Filter {
        input: Box::new(scan(1_000_000)),
        predicate: PredicateAst::Column {
            name: "mask".into(),
            dtype: MetalDtype::Bool,
        },
    };
    let lifting = compute_lifting_plan(&plan);
    // Spec § "Routing decisions" — filter always CPU, scan inherits.
    assert_eq!(
        lifting.get(&NodeId::new("Scan", 0)),
        Some(&NodeDecision::CpuLeave)
    );
    assert_eq!(
        lifting.get(&NodeId::new("Filter", 1)),
        Some(&NodeDecision::CpuLeave)
    );
}

#[test]
fn project_over_scan_is_cpu_leave_throughout() {
    let plan = MetalPlanNode::Project {
        input: Box::new(scan(1_000_000)),
        columns: vec!["a".into()],
    };
    let lifting = compute_lifting_plan(&plan);
    assert_eq!(
        lifting.get(&NodeId::new("Scan", 0)),
        Some(&NodeDecision::CpuLeave)
    );
    assert_eq!(
        lifting.get(&NodeId::new("Project", 1)),
        Some(&NodeDecision::CpuLeave)
    );
}

#[test]
fn project_after_filter_inherits_filter_decision() {
    let plan = MetalPlanNode::Project {
        input: Box::new(MetalPlanNode::Filter {
            input: Box::new(scan(1_000_000)),
            predicate: PredicateAst::Column {
                name: "mask".into(),
                dtype: MetalDtype::Bool,
            },
        }),
        columns: vec!["a".into()],
    };
    let lifting = compute_lifting_plan(&plan);
    assert_eq!(
        lifting.get(&NodeId::new("Project", 2)),
        Some(&NodeDecision::CpuLeave)
    );
}

#[test]
fn lifting_plan_has_single_node_for_scan_only() {
    // Construction sanity. The walker walks the entire tree and records
    // a decision for each node.
    let plan = scan(1_000);
    let lifting = compute_lifting_plan(&plan);
    assert_eq!(lifting.len(), 1);
    assert_eq!(
        lifting.get(&NodeId::new("Scan", 0)),
        Some(&NodeDecision::CpuLeave)
    );
}

#[test]
fn node_ids_are_assigned_in_post_order() {
    // Spec implicit: post-order so leaves get the smaller IDs (mirrors
    // the bottom-up walker's natural traversal). Filter-over-Scan:
    // Scan first, Filter second.
    let plan = MetalPlanNode::Filter {
        input: Box::new(scan(100)),
        predicate: PredicateAst::Column {
            name: "mask".into(),
            dtype: MetalDtype::Bool,
        },
    };
    let lifting = compute_lifting_plan(&plan);
    assert!(lifting.get(&NodeId::new("Scan", 0)).is_some());
    assert!(lifting.get(&NodeId::new("Filter", 1)).is_some());
}
