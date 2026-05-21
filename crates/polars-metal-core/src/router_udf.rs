// crates/polars-metal-core/src/router_udf.rs
//! PyO3 entry point: `_native.compute_lifting_plan(plan_dict) → dict`.
//!
//! Wire format
//! -----------
//! Input: same plan dict the walker will build (see _walker.py). For
//! Phase 1, accepted kinds are "Scan", "Project", "Filter". Phase 2
//! adds "GroupBy" (Task 12).
//!
//! Output: dict[str, str] where the key is "<Kind>#<seq>" and the value
//! is one of "gpu_lift", "cpu_leave", or "fallback:<reason>". The
//! Python walker iterates and applies.
//!
//! Unknown kinds become Fallback at their level; ancestors poison via
//! the cost-model's Fallback-propagation in `cost::decide_project`.

use crate::router::{cost, LiftingPlan, NodeDecision, NodeId};
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction(name = "compute_lifting_plan")]
pub fn compute_lifting_plan_py<'py>(
    py: Python<'py>,
    plan_dict: Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut next_seq: u32 = 0;
    let mut lifting = LiftingPlan::new();
    let _ = parse_and_route(&plan_dict, &mut next_seq, &mut lifting)?;
    // No affinity smoothing in Phase 1 (no close-cost candidates yet).
    let out = PyDict::new_bound(py);
    for (id, decision) in lifting.iter() {
        let value = match decision {
            NodeDecision::GpuLift => "gpu_lift".to_string(),
            NodeDecision::CpuLeave => "cpu_leave".to_string(),
            NodeDecision::Fallback(reason) => format!("fallback:{reason}"),
        };
        out.set_item(id.to_wire(), value)?;
    }
    Ok(out)
}

fn parse_and_route(
    dict: &Bound<PyDict>,
    next_seq: &mut u32,
    lifting: &mut LiftingPlan,
) -> PyResult<NodeId> {
    let kind: String = dict
        .get_item("kind")?
        .ok_or_else(|| PyKeyError::new_err("router: missing 'kind'"))?
        .extract()?;
    match kind.as_str() {
        "Scan" => {
            let id = NodeId::new("Scan", *next_seq);
            *next_seq += 1;
            lifting.set(id.clone(), cost::decide_scan_initial());
            Ok(id)
        }
        "Project" => {
            let input_obj = dict
                .get_item("input")?
                .ok_or_else(|| PyKeyError::new_err("Project: missing input"))?;
            let input_dict: Bound<PyDict> = input_obj.downcast_into()?;
            let child_id = parse_and_route(&input_dict, next_seq, lifting)?;
            let id = NodeId::new("Project", *next_seq);
            *next_seq += 1;
            let child_decision = lifting
                .get(&child_id)
                .cloned()
                .unwrap_or(NodeDecision::Fallback("missing child decision".into()));
            lifting.set(id.clone(), cost::decide_project(&child_decision));
            Ok(id)
        }
        "Filter" => {
            let input_obj = dict
                .get_item("input")?
                .ok_or_else(|| PyKeyError::new_err("Filter: missing input"))?;
            let input_dict: Bound<PyDict> = input_obj.downcast_into()?;
            let _ = parse_and_route(&input_dict, next_seq, lifting)?;
            let id = NodeId::new("Filter", *next_seq);
            *next_seq += 1;
            lifting.set(id.clone(), cost::decide_filter(0));
            Ok(id)
        }
        "GroupBy" => {
            let input_obj = dict
                .get_item("input")?
                .ok_or_else(|| PyKeyError::new_err("GroupBy: missing input"))?;
            let input_dict: Bound<PyDict> = input_obj.downcast_into()?;
            let n_rows = peek_input_row_count(&input_dict)?;
            let _ = parse_and_route(&input_dict, next_seq, lifting)?;
            let id = NodeId::new("GroupBy", *next_seq);
            *next_seq += 1;
            lifting.set(id.clone(), cost::decide_groupby(n_rows));
            Ok(id)
        }
        other => {
            // Walk a single "input" if present so seq numbering matches
            // the walker's post-order traversal.
            if let Ok(Some(input_obj)) = dict.get_item("input") {
                if let Ok(input_dict) = input_obj.downcast_into::<PyDict>() {
                    let _ = parse_and_route(&input_dict, next_seq, lifting)?;
                }
            }
            let id = NodeId::new(other, *next_seq);
            *next_seq += 1;
            lifting.set(
                id.clone(),
                NodeDecision::Fallback(format!("unsupported IR node: {other}")),
            );
            Ok(id)
        }
    }
}

/// Best-effort row count for cost-model input from a plan dict. Walks
/// past Project/Filter/GroupBy `"input"` fields to find the underlying
/// Scan. Mirrors `router::input_row_count`.
fn peek_input_row_count(dict: &Bound<PyDict>) -> PyResult<usize> {
    let kind: String = dict
        .get_item("kind")?
        .ok_or_else(|| PyKeyError::new_err("missing 'kind'"))?
        .extract()?;
    match kind.as_str() {
        "Scan" => {
            let n: usize = dict
                .get_item("n_rows")?
                .ok_or_else(|| PyKeyError::new_err("Scan: missing n_rows"))?
                .extract()?;
            Ok(n)
        }
        "Project" | "Filter" | "GroupBy" => {
            let input_obj = dict
                .get_item("input")?
                .ok_or_else(|| PyKeyError::new_err("missing 'input' in row-count peek"))?;
            let input_dict: Bound<PyDict> = input_obj.downcast_into()?;
            peek_input_row_count(&input_dict)
        }
        _ => Ok(0),
    }
}
