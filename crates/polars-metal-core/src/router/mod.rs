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
            let child_decision = plan.get(&child_id)
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
    }
}
