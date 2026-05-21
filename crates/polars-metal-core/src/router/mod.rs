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
