//! Affinity smoothing — second-pass cleanup of the LiftingPlan.
//!
//! After the bottom-up cost-model pass, this pass examines runs of
//! decisions. If a GpuLift node is surrounded by CpuLeave nodes and its
//! cost was within `window_pct` of the CPU alternative (per the cost
//! data the model recorded), flip it to CpuLeave; conversely for a
//! CpuLeave inside a GpuLift run.
//!
//! Initial implementation: rather than re-running the cost model with
//! both alternatives, the smoother accepts an explicit
//! `close_cost_node_ids` list. The cost model populates this list in
//! Task 6 when it sees a node where both alternatives evaluated within
//! `window_pct`. This keeps the data flow simple and the threshold
//! tunable from one PR.
//!
//! Fallback decisions are never smoothed: a Fallback is a hard
//! invariant violation (unsupported IR, oversized key), not a cost
//! decision.

use super::{LiftingPlan, NodeDecision, NodeId};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct SmoothingConfig {
    /// Cost-window percent that defines "close-cost" for the cost model
    /// upstream. Documented here for reference; smoothing logic uses
    /// the pre-populated `close_cost_node_ids` list.
    pub window_pct: u32,
    /// Nodes the cost model flagged as close-cost — candidates for
    /// transition smoothing.
    pub close_cost_node_ids: Vec<NodeId>,
}

impl Default for SmoothingConfig {
    fn default() -> Self {
        Self {
            window_pct: 20,
            close_cost_node_ids: vec![],
        }
    }
}

/// Apply affinity smoothing to a LiftingPlan, returning a new one.
///
/// Algorithm:
/// 1. Build the set of close-cost candidate IDs.
/// 2. For each candidate, examine its in-tree neighbors (via the IDs'
///    sequence numbers — neighbors are seq-1 and seq+1). If both
///    neighbors agree on a decision opposite the candidate, flip the
///    candidate.
/// 3. Never flip a Fallback.
pub fn smooth(plan: LiftingPlan, config: &SmoothingConfig) -> LiftingPlan {
    let candidates: HashSet<NodeId> = config.close_cost_node_ids.iter().cloned().collect();
    let mut next = plan.clone();
    for id in candidates.iter() {
        let current = match plan.get(id) {
            Some(d) => d,
            None => continue,
        };
        if matches!(current, NodeDecision::Fallback(_)) {
            continue;
        }
        let prev = neighbor_decision(&plan, id, -1);
        let succ = neighbor_decision(&plan, id, 1);
        match (&prev, &succ) {
            (Some(p), Some(s)) => {
                if p == s && p != current && !matches!(p, NodeDecision::Fallback(_)) {
                    next.set(id.clone(), p.clone());
                }
            }
            (Some(p), None) => {
                // No successor (root node). Match parent-side decision.
                if p != current && !matches!(p, NodeDecision::Fallback(_)) {
                    next.set(id.clone(), p.clone());
                }
            }
            _ => {}
        }
    }
    next
}

fn neighbor_decision(plan: &LiftingPlan, id: &NodeId, delta: i64) -> Option<NodeDecision> {
    let target_seq = (id.seq() as i64) + delta;
    if target_seq < 0 {
        return None;
    }
    plan.iter()
        .find(|(other, _)| other.seq() as i64 == target_seq)
        .map(|(_, d)| d.clone())
}
