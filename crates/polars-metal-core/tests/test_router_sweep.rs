// crates/polars-metal-core/tests/test_router_sweep.rs
//
// Property test: for any valid MetalPlanNode tree (built from M1's variants),
// `compute_lifting_plan` produces exactly one decision per node and the
// decision is well-formed (not e.g. an internal panic surfacing as a
// missing entry).
#![allow(clippy::expect_used)]

use polars_metal_native::plan::{MetalDtype, MetalPlanNode, PredicateAst};
use polars_metal_native::router::{compute_lifting_plan, NodeDecision};
use proptest::prelude::*;

fn arb_dtype() -> impl Strategy<Value = MetalDtype> {
    prop_oneof![
        Just(MetalDtype::I64),
        Just(MetalDtype::F64),
        Just(MetalDtype::Bool),
    ]
}

fn arb_scan() -> impl Strategy<Value = MetalPlanNode> {
    (0usize..10_000_000, arb_dtype()).prop_map(|(n, dt)| MetalPlanNode::Scan {
        n_rows: n,
        columns: vec![("a".into(), dt)],
    })
}

fn arb_plan() -> impl Strategy<Value = MetalPlanNode> {
    let leaf = arb_scan().boxed();
    leaf.prop_recursive(4, 16, 2, |inner| {
        prop_oneof![
            inner.clone().prop_map(|c| MetalPlanNode::Project {
                input: Box::new(c),
                columns: vec!["a".into()],
            }),
            inner.prop_map(|c| MetalPlanNode::Filter {
                input: Box::new(c),
                predicate: PredicateAst::Column {
                    name: "a".into(),
                    dtype: MetalDtype::Bool,
                },
            }),
        ]
    })
}

fn count_nodes(node: &MetalPlanNode) -> usize {
    match node {
        MetalPlanNode::Scan { .. } => 1,
        MetalPlanNode::Project { input, .. } => 1 + count_nodes(input),
        MetalPlanNode::Filter { input, .. } => 1 + count_nodes(input),
        MetalPlanNode::GroupBy { input, .. } => 1 + count_nodes(input),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]
    #[test]
    fn lifting_plan_has_one_decision_per_node(plan in arb_plan()) {
        let lifting = compute_lifting_plan(&plan);
        prop_assert_eq!(lifting.len(), count_nodes(&plan));
    }

    #[test]
    fn filter_decision_is_always_cpu_leave_under_m2_costs(plan in arb_plan()) {
        let lifting = compute_lifting_plan(&plan);
        for (id, decision) in lifting.iter() {
            if id.kind() == "Filter" {
                prop_assert_eq!(decision, &NodeDecision::CpuLeave);
            }
        }
    }
}
