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

use crate::plan::{MetalDtype, MetalPlanNode, PredicateAst};
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
        MetalPlanNode::Filter { .. } => Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "polars_metal: Filter dispatch lands in M1 Phase 5+ (compaction kernels)",
        )),
    }
}

/// Deserialize a Python `dict` into a [`MetalPlanNode`].
///
/// Plan dict schema (mirrors the Python walker output):
/// - `{"kind": "Scan", "n_rows": int, "columns": [(name, dtype_tag), ...]}`
/// - `{"kind": "Project", "input": <plan>, "columns": list[str]}`
/// - `{"kind": "Filter", "input": <plan>, "predicate": <pred>}`
///
/// dtype_tag is one of `"I64"`, `"F64"`, `"Bool"`.
fn deserialize_plan(dict: &Bound<PyDict>) -> PyResult<MetalPlanNode> {
    let kind: String = dict
        .get_item("kind")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("plan dict missing 'kind'"))?
        .extract()?;

    match kind.as_str() {
        "Scan" => {
            // n_rows is informational; Rust does not need it for the no-op
            // pass-through, but we parse it for shape validation / future use.
            let n_rows: usize = dict
                .get_item("n_rows")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Scan missing 'n_rows'"))?
                .extract()?;
            let cols_obj = dict
                .get_item("columns")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Scan missing 'columns'"))?;
            let cols_list: Bound<PyList> = cols_obj.downcast_into()?;
            let mut columns = Vec::with_capacity(cols_list.len());
            for entry in cols_list.iter() {
                // Each column entry is a 2-element (name, dtype_tag) pair. The
                // walker normalizes to tuples, but tests construct lists; we
                // accept both by going through `Vec<String>` and matching on
                // arity. (Pyo3's `extract::<(String, String)>` would reject
                // lists, so we use a Vec and validate length.)
                let pair: Vec<String> = entry.extract().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err(
                        "Scan.columns entries must be (name, dtype) 2-element sequences",
                    )
                })?;
                let [name, dtype_s] = <[String; 2]>::try_from(pair).map_err(|got| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Scan.columns entry expected 2 elements, got {}",
                        got.len()
                    ))
                })?;
                let dtype = parse_dtype(&dtype_s)?;
                columns.push((name, dtype));
            }
            Ok(MetalPlanNode::Scan { n_rows, columns })
        }
        "Project" => {
            let input_obj = dict
                .get_item("input")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Project missing 'input'"))?;
            let input_dict: Bound<PyDict> = input_obj.downcast_into()?;
            let input = Box::new(deserialize_plan(&input_dict)?);
            let cols: Vec<String> = dict
                .get_item("columns")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Project missing 'columns'"))?
                .extract()?;
            Ok(MetalPlanNode::Project {
                input,
                columns: cols,
            })
        }
        "Filter" => {
            // Deserialize for completeness so `execute_node` can raise a clean
            // NotImplementedError on dispatch (rather than panicking during
            // deserialization). The predicate AST is not consumed today.
            let input_obj = dict
                .get_item("input")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Filter missing 'input'"))?;
            let input_dict: Bound<PyDict> = input_obj.downcast_into()?;
            let input = Box::new(deserialize_plan(&input_dict)?);
            let pred_obj = dict.get_item("predicate")?.ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err("Filter missing 'predicate'")
            })?;
            let pred_dict: Bound<PyDict> = pred_obj.downcast_into()?;
            let predicate = deserialize_predicate(&pred_dict)?;
            Ok(MetalPlanNode::Filter { input, predicate })
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "polars_metal: unknown plan kind {other:?}"
        ))),
    }
}

fn deserialize_predicate(dict: &Bound<PyDict>) -> PyResult<PredicateAst> {
    let kind: String = dict
        .get_item("kind")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("predicate dict missing 'kind'"))?
        .extract()?;
    match kind.as_str() {
        "Column" => {
            let name: String = dict
                .get_item("name")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Column missing 'name'"))?
                .extract()?;
            let dtype_s: String = dict
                .get_item("dtype")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Column missing 'dtype'"))?
                .extract()?;
            Ok(PredicateAst::Column {
                name,
                dtype: parse_dtype(&dtype_s)?,
            })
        }
        // Other predicate variants (Compare, And, Or, literals) land in
        // Phases 6/7 when the walker actually produces them. Be strict — any
        // unknown kind today is a bug in the walker, not a fallback case.
        other => Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "polars_metal: predicate kind {other:?} lands in M1 Phase 6+"
        ))),
    }
}

fn parse_dtype(s: &str) -> PyResult<MetalDtype> {
    match s {
        "I64" => Ok(MetalDtype::I64),
        "F64" => Ok(MetalDtype::F64),
        "Bool" => Ok(MetalDtype::Bool),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "polars_metal: unknown MetalDtype tag {other:?}"
        ))),
    }
}
