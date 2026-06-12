//! PyO3 entry point for executing a Metal plan against a captured PyDataFrame.
//!
//! The Python walker (`_walker.walk`) produces a serialized
//! [`MetalPlanNode`](crate::plan::MetalPlanNode) tree as a Python dict, then
//! captures the underlying `PyDataFrame` from the `DataFrameScan` IR node.
//! It hands both to this entry point, which interprets the plan and returns
//! a `PyDataFrame` to be re-lifted on the Python side via
//! `pl.DataFrame._from_pydf`.
//!
//! M1 Phase 4 (this task) handles `Scan` (no-op pass-through) and `Project`
//! (column re-selection). `Filter` raises `NotImplementedError`; it lands in
//! Phase 5+ alongside the compaction kernels.
//!
//! Re-entrance safety
//! ------------------
//! Polars' `LazyFrame.collect` is patched on the Python side to re-route into
//! [`crate::execute_plan`] when `engine=MetalEngine()`. If we used
//! `LazyFrame.collect` (or any equivalent that re-enters the engine plugin)
//! to assemble results here, we would recurse infinitely. The fix — applied
//! everywhere a column re-selection is needed — is to call `PyDataFrame.select`
//! directly via PyO3 `call_method1`. `PyDataFrame.select` is a synchronous,
//! in-place column reorder/subset that bypasses the lazy plan entirely.

mod common;
mod compact;
pub use compact::execute_filter_compact;
mod compare;
pub use compare::{cmp_f64_col_col, cmp_f64_col_scalar, cmp_i64_col_col, cmp_i64_col_scalar};
mod dt;
pub use dt::execute_dt;
mod dtw;
pub use dtw::execute_dtw;
mod fused_expr;
pub use fused_expr::execute_fused_expr;
mod groupby;
pub use groupby::{
    execute_groupby, parse_groupby_plan, warmup_common_fused_signatures, GroupByParseError,
    ParsedAgg, ParsedGroupByPlan, ParsedKey,
};
mod logical;
pub use logical::{bool_and_dispatch, bool_or_dispatch};
mod predicate;
use predicate::deserialize_plan;
mod rolling;
pub use rolling::execute_rolling;

use crate::plan::MetalPlanNode;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

/// PyO3 entry point exposed as `polars_metal._native.execute_plan`.
///
/// # Arguments
/// * `df_in` — a Polars `PyDataFrame` (i.e. `pl.DataFrame._df`). The Scan node
///   refers to this frame; project/filter operate on its columns.
/// * `plan_dict` — a dict matching the shape produced by `_walker.walk()`. See
///   [`deserialize_plan`] for the schema.
///
/// # Returns
/// A `PyDataFrame` ready to be re-lifted via `pl.DataFrame._from_pydf`.
#[pyfunction]
pub fn execute_plan<'py>(
    py: Python<'py>,
    df_in: Bound<'py, PyAny>,
    plan_dict: Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let plan = deserialize_plan(&plan_dict)?;
    execute_node(py, df_in, &plan)
}

fn execute_node<'py>(
    py: Python<'py>,
    df: Bound<'py, PyAny>,
    node: &MetalPlanNode,
) -> PyResult<Bound<'py, PyAny>> {
    match node {
        MetalPlanNode::Scan { .. } => {
            // The Scan node's underlying PyDataFrame IS the captured input.
            // The walker may have recorded a projection on the Scan node,
            // but in Task 7 that projection was applied via the Project
            // wrapper above — so here we simply return `df`. (If a future
            // refactor pushes the projection into the Scan, this branch can
            // grow a column-subset step.)
            Ok(df)
        }
        MetalPlanNode::Project { input, columns } => {
            let upstream = execute_node(py, df, input)?;
            // CRITICAL: call PyDataFrame.select directly. Do NOT route through
            // pl.DataFrame.select (which would go via LazyFrame.collect and
            // re-enter MetalEngine, causing infinite recursion).
            let col_list = PyList::new_bound(py, columns.iter().map(|s| s.as_str()));
            upstream.call_method1("select", (col_list,))
        }
        MetalPlanNode::Filter { .. } => {
            // Filter dispatch goes through `execute_filter_compact`, not
            // `execute_plan`. The Python UDF detects the Filter at the plan
            // root and routes through the dedicated entry point because
            // compaction needs raw Arrow buffer bytes that are extracted
            // Python-side (see `_udf.py::build_udf`). Reaching this branch
            // means the walker emitted a Filter that the UDF didn't peel
            // off — surface as a plain NotImplementedError rather than
            // silently producing the wrong result.
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "polars_metal: Filter nodes must be dispatched via execute_filter_compact, \
                 not execute_plan (the Python UDF handles the routing)",
            ))
        }
        MetalPlanNode::GroupBy { .. } => {
            // GroupBy execution lands in Task 28. For now, this code path
            // should not be reached — the Python UDF routes GroupBy through
            // a dedicated entry point (Task 29). If reached, raise a clear
            // error rather than panicking.
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "polars_metal: GroupBy execution not yet implemented (lands in Phase 2)",
            ))
        }
    }
}
