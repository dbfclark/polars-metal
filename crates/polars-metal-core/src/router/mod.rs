// crates/polars-metal-core/src/router/mod.rs
//! Router: per-op GPU-vs-CPU dispatch decisions for the engine.
//!
//! Given a `MetalPlanNode` tree from the walker, the router produces a
//! `LiftingPlan` — a per-node decision (GpuLift | CpuLeave | Fallback).
//! The walker consumes that decision per node and installs UDFs only
//! for GpuLift subtrees.
//!
//! See `docs/superpowers/specs/2026-05-21-m2-design.md` § "Three-layer
//! flow" for the architectural picture.

use std::collections::HashMap;

pub mod affinity;
pub mod cost;

/// Identifier for an IR node in the MetalPlanNode tree. Tuple of
/// (kind, sequence-number); the walker emits these deterministically so
/// the same node has the same ID across walker → router → walker round
/// trip in one query.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NodeId {
    kind: String,
    seq: u32,
}

impl NodeId {
    pub fn new(kind: impl Into<String>, seq: u32) -> Self {
        Self {
            kind: kind.into(),
            seq,
        }
    }
    pub fn kind(&self) -> &str {
        &self.kind
    }
    pub fn seq(&self) -> u32 {
        self.seq
    }
    /// Wire-format string the Python walker receives: `"Kind#seq"`.
    pub fn to_wire(&self) -> String {
        format!("{}#{}", self.kind, self.seq)
    }
}

/// Per-node routing decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeDecision {
    /// The walker should install a UDF for this subtree; the engine runs
    /// it on the GPU.
    GpuLift,
    /// The walker leaves this subtree alone; Polars' CPU executor runs it.
    CpuLeave,
    /// This subtree (or some descendant) is unrecognized or violates a
    /// plan-time invariant (e.g. composite-key width). Ancestors poison;
    /// the whole query routes to CPU.
    Fallback(String),
}

/// A LiftingPlan: the full per-node decision map for one query.
#[derive(Debug, Default, Clone)]
pub struct LiftingPlan {
    decisions: HashMap<NodeId, NodeDecision>,
}

impl LiftingPlan {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn set(&mut self, id: NodeId, decision: NodeDecision) {
        self.decisions.insert(id, decision);
    }
    pub fn get(&self, id: &NodeId) -> Option<&NodeDecision> {
        self.decisions.get(id)
    }
    pub fn len(&self) -> usize {
        self.decisions.len()
    }
    pub fn is_empty(&self) -> bool {
        self.decisions.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = (&NodeId, &NodeDecision)> {
        self.decisions.iter()
    }
    /// True iff the plan contains any Fallback decision. When true, the
    /// walker drops the entire LiftingPlan and routes to CPU.
    pub fn has_fallback(&self) -> bool {
        self.decisions
            .values()
            .any(|d| matches!(d, NodeDecision::Fallback(_)))
    }
}

use crate::plan::MetalPlanNode;

/// Walk `root` bottom-up assigning per-node IDs in post-order; consult
/// the cost model at each node; return the full LiftingPlan.
///
/// `NodeId::seq` is a monotone counter incremented in post-order. The
/// Python walker uses the same counter (it produces the MetalPlanNode
/// tree in the same shape), so wire-format IDs round-trip.
pub fn compute_lifting_plan(root: &MetalPlanNode) -> LiftingPlan {
    let mut plan = LiftingPlan::new();
    let mut next_seq: u32 = 0;
    let _ = walk(root, &mut plan, &mut next_seq);
    plan
}

/// Recursive worker. Returns the assigned NodeId so the parent can read
/// the child's recorded decision.
fn walk(node: &MetalPlanNode, plan: &mut LiftingPlan, next_seq: &mut u32) -> NodeId {
    match node {
        MetalPlanNode::Scan { .. } => {
            let id = NodeId::new("Scan", *next_seq);
            *next_seq += 1;
            plan.set(id.clone(), cost::decide_scan_initial());
            id
        }
        MetalPlanNode::Project { input, .. } => {
            let child_id = walk(input, plan, next_seq);
            let id = NodeId::new("Project", *next_seq);
            *next_seq += 1;
            let child_decision = plan
                .get(&child_id)
                .cloned()
                .unwrap_or(NodeDecision::Fallback("missing child decision".into()));
            plan.set(id.clone(), cost::decide_project(&child_decision));
            id
        }
        MetalPlanNode::Filter { input, .. } => {
            let _ = walk(input, plan, next_seq);
            let id = NodeId::new("Filter", *next_seq);
            *next_seq += 1;
            plan.set(id.clone(), cost::decide_filter(0));
            id
        }
        MetalPlanNode::GroupBy { input, keys, aggs } => {
            let n_rows = input_row_count(input);
            let input_schema = input_schema_lookup(input);
            let _ = walk(input, plan, next_seq);
            let id = NodeId::new("GroupBy", *next_seq);
            *next_seq += 1;
            // Phase 3 / Task 15: the fused-aggregation kernel only supports
            // 32-bit-or-narrower value-column dtypes. If any Expression agg
            // references a 64-bit-wide column, fall back to CPU at plan
            // time — there is no per-agg twin for Expression specs, so the
            // dispatcher cannot recover.
            let expression_falls_back = aggs.iter().any(|a| {
                if let crate::plan::AggSpec::Expression { expr, .. } = a {
                    expr.referenced_columns().iter().any(|c| {
                        matches!(
                            input_schema.get(c),
                            Some(crate::plan::MetalDtype::F64) | Some(crate::plan::MetalDtype::I64)
                        )
                    })
                } else {
                    false
                }
            });
            let decision = if expression_falls_back {
                NodeDecision::Fallback(
                    "Expression agg references 64-bit-wide column; fused kernel requires \
                     32-bit-or-narrower inputs (Apple Silicon Metal lacks 64-bit atomics)"
                        .into(),
                )
            } else {
                cost::decide_groupby_with_keys(n_rows, keys, aggs)
            };
            plan.set(id.clone(), decision);
            id
        }
    }
}

/// Best-effort row count for cost-model input. Walks past Project and
/// Filter to find the underlying Scan; returns 0 if none. Filter is a
/// notable simplification — we use the *input* row count for the cost
/// estimate (the kernel sees the post-filter count, but at plan time
/// we don't know it). This is the M2 starting heuristic; later PRs may
/// thread an estimated cardinality.
fn input_row_count(node: &MetalPlanNode) -> usize {
    match node {
        MetalPlanNode::Scan { n_rows, .. } => *n_rows,
        MetalPlanNode::Project { input, .. } => input_row_count(input),
        MetalPlanNode::Filter { input, .. } => input_row_count(input),
        MetalPlanNode::GroupBy { .. } => {
            // GroupBy-of-GroupBy: post-grouped row count is unknown at
            // plan time. Conservative default: route the outer GroupBy
            // to CPU by reporting 0 rows.
            0
        }
    }
}

/// Walk past `Project` / `Filter` to find the underlying `Scan`, returning
/// a `name → MetalDtype` lookup over its columns. Used by GroupBy routing
/// to decide whether an Expression agg's referenced columns fit the fused
/// kernel's 32-bit-or-narrower constraint. Returns an empty map for trees
/// that don't terminate in a Scan (e.g. GroupBy-of-GroupBy, where we
/// already route the outer node conservatively via `input_row_count`).
fn input_schema_lookup(
    node: &MetalPlanNode,
) -> std::collections::HashMap<String, crate::plan::MetalDtype> {
    match node {
        MetalPlanNode::Scan { columns, .. } => columns
            .iter()
            .map(|(n, d)| (n.clone(), *d))
            .collect::<std::collections::HashMap<_, _>>(),
        MetalPlanNode::Project { input, .. } => input_schema_lookup(input),
        MetalPlanNode::Filter { input, .. } => input_schema_lookup(input),
        MetalPlanNode::GroupBy { .. } => std::collections::HashMap::new(),
    }
}
