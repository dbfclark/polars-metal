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

use crate::plan::{AggExpr, AggOp, BinaryOp, MetalDtype, MetalPlanNode, PredicateAst};
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::cmp::{
    dispatch_cmp_f64, dispatch_cmp_f64_scalar, dispatch_cmp_i64, dispatch_cmp_i64_scalar, CompareOp,
};
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::groupby::{
    AggKind, AggRequest, GroupByError, KeyColumn, KeyDtype, ValueColumn,
};
use polars_metal_kernels::logical::{dispatch_bool_and, dispatch_bool_or};
use polars_metal_kernels::pipeline::{
    compact_bool, compact_f64, compact_i64, compute_keep_and_prefix,
};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use std::collections::HashMap;

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

/// PyO3 entry point exposed as `polars_metal._native.execute_filter_compact`.
///
/// Phase 5 — the precomputed-boolean-mask filter path. The Python UDF
/// extracts bit-packed Boolean predicate bytes and per-column data +
/// validity bytes via `Series.to_arrow().buffers()`, then hands them
/// here. Rust runs the three-pass compaction pipeline (predicate
/// evaluation + MLX cumsum + scatter) for each surviving column and
/// returns the compacted bytes to Python, which reassembles a Polars
/// DataFrame via `pa.Array.from_buffers`.
///
/// # Arguments
/// * `pred_data` — bit-packed predicate data buffer, at least
///   `ceil(n_rows / 8)` bytes.
/// * `pred_valid` — bit-packed predicate validity buffer, at least
///   `ceil(n_rows / 8)` bytes. Pass an all-ones buffer if the predicate
///   has no nulls (Arrow's null buffer is `None` in that case — the
///   Python side materialises the all-ones bitmap).
/// * `n_rows` — number of rows in every input.
/// * `columns` — list of `(name, dtype_tag, data_bytes, valid_bytes)`
///   tuples, one per surviving column. `dtype_tag` is `"I64"`, `"F64"`,
///   or `"Bool"`. `valid_bytes` is at least `ceil(n_rows / 8)` bytes
///   (the Python side materialises all-ones for no-null columns).
///
/// # Returns
/// A list of `(data_bytes, valid_bytes, n_out)` tuples, one per input
/// column in the same order. For i64/f64 the `data_bytes` length is
/// `n_out * 8`; for bool it is bit-packed `ceil(n_out / 8)` bytes
/// (potentially padded to 4-byte alignment by the kernel — the Python
/// side trims to `ceil(n_out / 8)` before passing to PyArrow).
#[pyfunction]
pub fn execute_filter_compact<'py>(
    py: Python<'py>,
    pred_data: &Bound<'py, PyBytes>,
    pred_valid: &Bound<'py, PyBytes>,
    n_rows: usize,
    columns: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyList>> {
    let pred_data_bytes: &[u8] = pred_data.as_bytes();
    let pred_valid_bytes: &[u8] = pred_valid.as_bytes();

    let min_pred_bytes = (n_rows + 7) / 8;
    if pred_data_bytes.len() < min_pred_bytes {
        return Err(PyValueError::new_err(format!(
            "polars_metal: pred_data is {got} B, need at least {expected} B for {n} rows",
            got = pred_data_bytes.len(),
            expected = min_pred_bytes,
            n = n_rows,
        )));
    }
    if pred_valid_bytes.len() < min_pred_bytes {
        return Err(PyValueError::new_err(format!(
            "polars_metal: pred_valid is {got} B, need at least {expected} B for {n} rows",
            got = pred_valid_bytes.len(),
            expected = min_pred_bytes,
            n = n_rows,
        )));
    }

    // One device + queue for the whole filter dispatch. Re-creating the
    // queue per-column would force command-buffer serialisation we don't
    // want; sharing the queue lets each column's three-pass pipeline run
    // independently, with explicit waits at the end of each pass.
    let device = MetalDevice::system_default()
        .map_err(|e| crate::engine_err(crate::EngineError::Buffer(e)))?;
    let mut queue = CommandQueue::new(&device)
        .map_err(|e| crate::engine_err(crate::EngineError::Other(format!("command queue: {e}"))))?;

    // Hoist the predicate-to-u8 + MLX cumsum out of the per-column loop:
    // the predicate doesn't depend on the source column, so this work is
    // identical for every column. Running it once and sharing the
    // `(keep, prefix, n_out)` saves `(num_cols - 1) * (predicate + cumsum)`
    // of redundant work per filter dispatch — the dominant cost in the
    // M1 filter path (see Task 30 profiling).
    let (keep, prefix, n_out) = compute_keep_and_prefix(
        &device,
        &mut queue,
        pred_data_bytes,
        pred_valid_bytes,
        n_rows,
    )
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "polars_metal: compute_keep_and_prefix failed: {e}"
        ))
    })?;

    let results = PyList::empty_bound(py);

    for (idx, entry) in columns.iter().enumerate() {
        let tup: Bound<PyTuple> = entry.downcast_into().map_err(|_| {
            PyValueError::new_err(format!("polars_metal: columns[{idx}] is not a tuple"))
        })?;
        if tup.len() != 4 {
            return Err(PyValueError::new_err(format!(
                "polars_metal: columns[{idx}] expected 4 elements (name, dtype, data, valid), got {n}",
                n = tup.len(),
            )));
        }
        let name: String = tup.get_item(0)?.extract()?;
        let dtype_s: String = tup.get_item(1)?.extract()?;
        let data_py: Bound<PyBytes> = tup.get_item(2)?.downcast_into().map_err(|_| {
            PyValueError::new_err(format!(
                "polars_metal: columns[{idx}].data ({name}) must be bytes"
            ))
        })?;
        let valid_py: Bound<PyBytes> = tup.get_item(3)?.downcast_into().map_err(|_| {
            PyValueError::new_err(format!(
                "polars_metal: columns[{idx}].valid ({name}) must be bytes"
            ))
        })?;
        let data_b: &[u8] = data_py.as_bytes();
        let valid_b: &[u8] = valid_py.as_bytes();

        let dtype = parse_dtype(&dtype_s)?;
        let (out_data, out_valid) = compact_one_column(
            &device, &mut queue, dtype, n_rows, data_b, valid_b, &keep, &prefix, n_out, &name,
        )?;

        let tup = PyTuple::new_bound(
            py,
            [
                PyBytes::new_bound(py, &out_data).into_any(),
                PyBytes::new_bound(py, &out_valid).into_any(),
                n_out.into_py(py).into_bound(py),
            ],
        );
        results.append(tup)?;
    }

    Ok(results)
}

/// Run pass 3 (scatter) of the compaction pipeline on a single column,
/// reusing the shared `(keep, prefix, n_out)` produced once per filter
/// dispatch. Per-dtype branching is contained here so
/// `execute_filter_compact`'s loop body stays dtype-agnostic.
///
/// When `n_out == 0` we short-circuit before allocating output buffers
/// or calling Metal — every column's result is `( [], [], 0 )`. The
/// shared n_out is the source of truth across all columns: a single
/// `prefix[n_rows - 1]` read decides whether any column has survivors.
///
/// `column_name` is used only for error messages.
#[allow(clippy::too_many_arguments)]
fn compact_one_column(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    dtype: MetalDtype,
    n_rows: usize,
    src_data: &[u8],
    src_valid: &[u8],
    keep: &[u8],
    prefix: &[u32],
    n_out: usize,
    column_name: &str,
) -> PyResult<(Vec<u8>, Vec<u8>)> {
    let min_valid_bytes = (n_rows + 7) / 8;
    if src_valid.len() < min_valid_bytes {
        return Err(PyValueError::new_err(format!(
            "polars_metal: column {column_name:?} validity buffer is {got} B, need {expected} B for {n} rows",
            got = src_valid.len(),
            expected = min_valid_bytes,
            n = n_rows,
        )));
    }

    // Short-circuit when no rows survive the predicate. Producing an
    // empty `valid` Vec at the kernel's alignment requirement would be
    // lying: there's no data to gate, so an empty `(data, valid)` pair
    // is the honest result. The Python side reassembles a zero-length
    // PyArrow array from these.
    if n_out == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    match dtype {
        MetalDtype::I64 => {
            let expected_data = n_rows * 8;
            if src_data.len() < expected_data {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: column {column_name:?} (I64) data buffer is {got} B, need {expected} B",
                    got = src_data.len(),
                    expected = expected_data,
                )));
            }
            // SAFETY: i64 has no invalid bit patterns; the slice is `expected_data`
            // bytes long, which is exactly `n_rows` i64s. `from_arrow` Arrow
            // buffers are 8-byte aligned (Arrow buffer alignment requirement
            // is 64 bytes for primitive arrays), so the reinterpret is well-aligned.
            let src_typed: &[i64] =
                unsafe { std::slice::from_raw_parts(src_data.as_ptr() as *const i64, n_rows) };
            let result = compact_i64(device, queue, src_typed, src_valid, keep, prefix, n_out)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "polars_metal: compact_i64({column_name:?}) failed: {e}"
                    ))
                })?;
            // SAFETY: `result.data` is a `Vec<i64>` of `result.n_out` elements;
            // reinterpret as bytes for the wire format. i64 has no invalid bit
            // patterns and `from_raw_parts` length is exactly `n_out * 8`.
            let data_bytes: Vec<u8> = {
                let n_bytes = result.n_out * std::mem::size_of::<i64>();
                let mut v = vec![0u8; n_bytes];
                if n_bytes > 0 {
                    let src_slice = unsafe {
                        std::slice::from_raw_parts(result.data.as_ptr() as *const u8, n_bytes)
                    };
                    v.copy_from_slice(src_slice);
                }
                v
            };
            Ok((data_bytes, result.valid))
        }
        MetalDtype::F64 => {
            let expected_data = n_rows * 8;
            if src_data.len() < expected_data {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: column {column_name:?} (F64) data buffer is {got} B, need {expected} B",
                    got = src_data.len(),
                    expected = expected_data,
                )));
            }
            // SAFETY: see I64 branch; f64 has no invalid bit patterns and the
            // slice length is exactly `n_rows` f64s.
            let src_typed: &[f64] =
                unsafe { std::slice::from_raw_parts(src_data.as_ptr() as *const f64, n_rows) };
            let result = compact_f64(device, queue, src_typed, src_valid, keep, prefix, n_out)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "polars_metal: compact_f64({column_name:?}) failed: {e}"
                    ))
                })?;
            // SAFETY: see I64 branch; reinterpret f64 → u8.
            let data_bytes: Vec<u8> = {
                let n_bytes = result.n_out * std::mem::size_of::<f64>();
                let mut v = vec![0u8; n_bytes];
                if n_bytes > 0 {
                    let src_slice = unsafe {
                        std::slice::from_raw_parts(result.data.as_ptr() as *const u8, n_bytes)
                    };
                    v.copy_from_slice(src_slice);
                }
                v
            };
            Ok((data_bytes, result.valid))
        }
        MetalDtype::Bool => {
            // Bool source data is bit-packed: at least `ceil(n_rows / 8)` bytes.
            if src_data.len() < min_valid_bytes {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: column {column_name:?} (Bool) data buffer is {got} B, need {expected} B",
                    got = src_data.len(),
                    expected = min_valid_bytes,
                )));
            }
            let result = compact_bool(
                device, queue, src_data, src_valid, keep, prefix, n_rows, n_out,
            )
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "polars_metal: compact_bool({column_name:?}) failed: {e}"
                ))
            })?;
            // For bool, `result.data` is already bit-packed u8 (padded to
            // 4-byte alignment by the kernel). The Python side will trim to
            // `ceil(n_out / 8)` before calling PyArrow.
            Ok((result.data, result.valid))
        }
        MetalDtype::I32 | MetalDtype::F32 => {
            // I32/F32: not yet supported in the filter compaction path.
            // The walker should fall back for filter on 32-bit columns;
            // reaching here is a logic bug — surface clearly.
            Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "polars_metal: filter compaction for {column_name:?} (I32/F32) not yet implemented"
            )))
        }
        // M3 capability F: small-integer key dtypes. Not supported in the
        // filter compaction path — filter is CPU-routed for these and the
        // walker should never lift a filter over such a column. Surface as
        // PyNotImplementedError so a router bug is visible.
        MetalDtype::I8 | MetalDtype::I16 | MetalDtype::U8 | MetalDtype::U16 | MetalDtype::U32 => {
            Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "polars_metal: filter compaction for {column_name:?} ({dtype:?}) not supported; filter should route CPU"
            )))
        }
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
        "LiteralI64" => {
            let v: i64 = dict
                .get_item("value")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("LiteralI64 missing 'value'"))?
                .extract()?;
            Ok(PredicateAst::LiteralI64(v))
        }
        "LiteralF64" => {
            let v: f64 = dict
                .get_item("value")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("LiteralF64 missing 'value'"))?
                .extract()?;
            Ok(PredicateAst::LiteralF64(v))
        }
        "LiteralBool" => {
            let v: bool = dict
                .get_item("value")?
                .ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err("LiteralBool missing 'value'")
                })?
                .extract()?;
            Ok(PredicateAst::LiteralBool(v))
        }
        "Compare" => {
            let op_s: String = dict
                .get_item("op")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Compare missing 'op'"))?
                .extract()?;
            let op = parse_compare_op(&op_s)?;
            let lhs_obj = dict
                .get_item("lhs")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Compare missing 'lhs'"))?;
            let lhs_dict: Bound<PyDict> = lhs_obj.downcast_into()?;
            let lhs = Box::new(deserialize_predicate(&lhs_dict)?);
            let rhs_obj = dict
                .get_item("rhs")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Compare missing 'rhs'"))?;
            let rhs_dict: Bound<PyDict> = rhs_obj.downcast_into()?;
            let rhs = Box::new(deserialize_predicate(&rhs_dict)?);
            Ok(PredicateAst::Compare {
                op: match op {
                    CompareOp::Eq => crate::plan::CompareOp::Eq,
                    CompareOp::Ne => crate::plan::CompareOp::Ne,
                    CompareOp::Lt => crate::plan::CompareOp::Lt,
                    CompareOp::Le => crate::plan::CompareOp::Le,
                    CompareOp::Gt => crate::plan::CompareOp::Gt,
                    CompareOp::Ge => crate::plan::CompareOp::Ge,
                },
                lhs,
                rhs,
            })
        }
        "And" | "Or" => {
            // Filter dispatch flows entirely through the Python UDF in
            // M1; the Rust `execute_plan` Filter branch raises
            // NotImplementedError before ever reaching the AST. We
            // still accept the AND/OR shape here so the parse step
            // doesn't fail if a Filter plan ever round-trips through
            // `execute_plan` (e.g. a future test or a debug-print
            // path).
            let lhs_obj = dict
                .get_item("lhs")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("And/Or missing 'lhs'"))?;
            let lhs_dict: Bound<PyDict> = lhs_obj.downcast_into()?;
            let lhs = Box::new(deserialize_predicate(&lhs_dict)?);
            let rhs_obj = dict
                .get_item("rhs")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("And/Or missing 'rhs'"))?;
            let rhs_dict: Bound<PyDict> = rhs_obj.downcast_into()?;
            let rhs = Box::new(deserialize_predicate(&rhs_dict)?);
            if kind == "And" {
                Ok(PredicateAst::And(lhs, rhs))
            } else {
                Ok(PredicateAst::Or(lhs, rhs))
            }
        }
        other => Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "polars_metal: predicate kind {other:?} lands in a later phase"
        ))),
    }
}

fn parse_dtype(s: &str) -> PyResult<MetalDtype> {
    MetalDtype::from_wire(s).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "polars_metal: unknown MetalDtype tag {s:?}"
        ))
    })
}

/// Parse the wire-format op tag (matching `CompareOp::Eq/Ne/Lt/Le/Gt/Ge`)
/// into the kernel-side `CompareOp`. Used both at predicate-AST
/// deserialization time and by the `cmp_*` pyfunctions below.
fn parse_compare_op(s: &str) -> PyResult<CompareOp> {
    match s {
        "Eq" => Ok(CompareOp::Eq),
        "Ne" => Ok(CompareOp::Ne),
        "Lt" => Ok(CompareOp::Lt),
        "Le" => Ok(CompareOp::Le),
        "Gt" => Ok(CompareOp::Gt),
        "Ge" => Ok(CompareOp::Ge),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "polars_metal: unknown CompareOp tag {other:?}"
        ))),
    }
}

/// Minimum bytes for a bit-packed output bitmap of `n_rows` rows, mirroring
/// the kernel-side ``out_min_bytes`` (4-byte-aligned for atomic_uint, min 4).
fn cmp_out_min_bytes(n_rows: usize) -> usize {
    let raw = (n_rows + 7) / 8;
    let padded = (raw + 3) & !3;
    padded.max(4)
}

/// PyO3 entry point exposed as `polars_metal._native.cmp_i64_col_scalar`.
///
/// Evaluates a single column-vs-scalar i64 comparison and returns the
/// bit-packed bool predicate `(data, valid)`. The Python UDF calls this
/// when the walker emits a `Compare { lhs: Column(I64), rhs: LiteralI64 }`
/// (and similarly for the other three column-vs-leaf combinations); the
/// resulting predicate bytes feed straight into
/// [`execute_filter_compact`].
///
/// Arguments mirror [`dispatch_cmp_i64_scalar`]; bytes are zero-copied
/// into Metal device buffers inside the dispatcher.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn cmp_i64_col_scalar<'py>(
    py: Python<'py>,
    lhs_data: &Bound<'py, PyBytes>,
    lhs_valid: &Bound<'py, PyBytes>,
    rhs: i64,
    op: &str,
    n_rows: usize,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let op_enum = parse_compare_op(op)?;
    let lhs_data_bytes = lhs_data.as_bytes();
    let lhs_valid_bytes = lhs_valid.as_bytes();
    check_numeric_buffers(lhs_data_bytes, lhs_valid_bytes, n_rows, 8)?;

    // SAFETY: i64 has no invalid bit patterns; `lhs_data_bytes` length is
    // at least `n_rows * 8` (checked above). The Arrow buffer Python hands
    // us is 64-byte-aligned (Arrow alignment requirement), so the reinterpret
    // is well-aligned.
    let lhs_slice: &[i64] =
        unsafe { std::slice::from_raw_parts(lhs_data_bytes.as_ptr() as *const i64, n_rows) };

    let (device, mut queue) = new_device_and_queue()?;
    let min_out = cmp_out_min_bytes(n_rows);
    let mut out_data = vec![0u8; min_out];
    let mut out_valid = vec![0u8; min_out];
    dispatch_cmp_i64_scalar(
        &device,
        &mut queue,
        lhs_slice,
        lhs_valid_bytes,
        rhs,
        n_rows,
        op_enum,
        &mut out_data,
        &mut out_valid,
    )
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "polars_metal: cmp_i64_col_scalar dispatch failed: {e}"
        ))
    })?;
    Ok((
        PyBytes::new_bound(py, &out_data),
        PyBytes::new_bound(py, &out_valid),
    ))
}

/// PyO3 entry point exposed as `polars_metal._native.cmp_i64_col_col`.
///
/// Evaluates a column-vs-column i64 comparison. See [`cmp_i64_col_scalar`]
/// for the bigger picture; this variant just feeds two columns to
/// [`dispatch_cmp_i64`] instead of `(col, scalar)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn cmp_i64_col_col<'py>(
    py: Python<'py>,
    lhs_data: &Bound<'py, PyBytes>,
    lhs_valid: &Bound<'py, PyBytes>,
    rhs_data: &Bound<'py, PyBytes>,
    rhs_valid: &Bound<'py, PyBytes>,
    op: &str,
    n_rows: usize,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let op_enum = parse_compare_op(op)?;
    let lhs_data_bytes = lhs_data.as_bytes();
    let lhs_valid_bytes = lhs_valid.as_bytes();
    let rhs_data_bytes = rhs_data.as_bytes();
    let rhs_valid_bytes = rhs_valid.as_bytes();
    check_numeric_buffers(lhs_data_bytes, lhs_valid_bytes, n_rows, 8)?;
    check_numeric_buffers(rhs_data_bytes, rhs_valid_bytes, n_rows, 8)?;

    // SAFETY: see `cmp_i64_col_scalar`; both slices are at least n_rows*8
    // bytes and 64-byte aligned by Arrow.
    let lhs_slice: &[i64] =
        unsafe { std::slice::from_raw_parts(lhs_data_bytes.as_ptr() as *const i64, n_rows) };
    let rhs_slice: &[i64] =
        unsafe { std::slice::from_raw_parts(rhs_data_bytes.as_ptr() as *const i64, n_rows) };

    let (device, mut queue) = new_device_and_queue()?;
    let min_out = cmp_out_min_bytes(n_rows);
    let mut out_data = vec![0u8; min_out];
    let mut out_valid = vec![0u8; min_out];
    dispatch_cmp_i64(
        &device,
        &mut queue,
        lhs_slice,
        lhs_valid_bytes,
        rhs_slice,
        rhs_valid_bytes,
        n_rows,
        op_enum,
        &mut out_data,
        &mut out_valid,
    )
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "polars_metal: cmp_i64_col_col dispatch failed: {e}"
        ))
    })?;
    Ok((
        PyBytes::new_bound(py, &out_data),
        PyBytes::new_bound(py, &out_valid),
    ))
}

/// PyO3 entry point exposed as `polars_metal._native.cmp_f64_col_scalar`.
///
/// f64 mirror of [`cmp_i64_col_scalar`]. Polars/IEEE 754 NaN semantics
/// are implemented inside the kernel (see `cmp_f64.metal`); the wrapper
/// is dtype-agnostic otherwise.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn cmp_f64_col_scalar<'py>(
    py: Python<'py>,
    lhs_data: &Bound<'py, PyBytes>,
    lhs_valid: &Bound<'py, PyBytes>,
    rhs: f64,
    op: &str,
    n_rows: usize,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let op_enum = parse_compare_op(op)?;
    let lhs_data_bytes = lhs_data.as_bytes();
    let lhs_valid_bytes = lhs_valid.as_bytes();
    check_numeric_buffers(lhs_data_bytes, lhs_valid_bytes, n_rows, 8)?;

    // SAFETY: f64 has no invalid bit patterns (every 8-byte sequence is a
    // legitimate f64 — including NaN payloads). Length and alignment are
    // the same as the i64 path.
    let lhs_slice: &[f64] =
        unsafe { std::slice::from_raw_parts(lhs_data_bytes.as_ptr() as *const f64, n_rows) };

    let (device, mut queue) = new_device_and_queue()?;
    let min_out = cmp_out_min_bytes(n_rows);
    let mut out_data = vec![0u8; min_out];
    let mut out_valid = vec![0u8; min_out];
    dispatch_cmp_f64_scalar(
        &device,
        &mut queue,
        lhs_slice,
        lhs_valid_bytes,
        rhs,
        n_rows,
        op_enum,
        &mut out_data,
        &mut out_valid,
    )
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "polars_metal: cmp_f64_col_scalar dispatch failed: {e}"
        ))
    })?;
    Ok((
        PyBytes::new_bound(py, &out_data),
        PyBytes::new_bound(py, &out_valid),
    ))
}

/// PyO3 entry point exposed as `polars_metal._native.cmp_f64_col_col`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn cmp_f64_col_col<'py>(
    py: Python<'py>,
    lhs_data: &Bound<'py, PyBytes>,
    lhs_valid: &Bound<'py, PyBytes>,
    rhs_data: &Bound<'py, PyBytes>,
    rhs_valid: &Bound<'py, PyBytes>,
    op: &str,
    n_rows: usize,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let op_enum = parse_compare_op(op)?;
    let lhs_data_bytes = lhs_data.as_bytes();
    let lhs_valid_bytes = lhs_valid.as_bytes();
    let rhs_data_bytes = rhs_data.as_bytes();
    let rhs_valid_bytes = rhs_valid.as_bytes();
    check_numeric_buffers(lhs_data_bytes, lhs_valid_bytes, n_rows, 8)?;
    check_numeric_buffers(rhs_data_bytes, rhs_valid_bytes, n_rows, 8)?;

    // SAFETY: see `cmp_f64_col_scalar`.
    let lhs_slice: &[f64] =
        unsafe { std::slice::from_raw_parts(lhs_data_bytes.as_ptr() as *const f64, n_rows) };
    let rhs_slice: &[f64] =
        unsafe { std::slice::from_raw_parts(rhs_data_bytes.as_ptr() as *const f64, n_rows) };

    let (device, mut queue) = new_device_and_queue()?;
    let min_out = cmp_out_min_bytes(n_rows);
    let mut out_data = vec![0u8; min_out];
    let mut out_valid = vec![0u8; min_out];
    dispatch_cmp_f64(
        &device,
        &mut queue,
        lhs_slice,
        lhs_valid_bytes,
        rhs_slice,
        rhs_valid_bytes,
        n_rows,
        op_enum,
        &mut out_data,
        &mut out_valid,
    )
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "polars_metal: cmp_f64_col_col dispatch failed: {e}"
        ))
    })?;
    Ok((
        PyBytes::new_bound(py, &out_data),
        PyBytes::new_bound(py, &out_valid),
    ))
}

/// Build a `(MetalDevice, CommandQueue)` pair for one comparison-kernel
/// dispatch. The kernel dispatcher reuses the queue across its three
/// internal passes (load + compute + readback) so callers don't share
/// queues across pyfunction invocations.
fn new_device_and_queue() -> PyResult<(MetalDevice, CommandQueue)> {
    let device = MetalDevice::system_default()
        .map_err(|e| crate::engine_err(crate::EngineError::Buffer(e)))?;
    let queue = CommandQueue::new(&device)
        .map_err(|e| crate::engine_err(crate::EngineError::Other(format!("command queue: {e}"))))?;
    Ok((device, queue))
}

/// Length-check the data and validity buffers for a numeric column input
/// to a comparison kernel. `data_bytes_per_row` is 8 for i64/f64 (the
/// only widths we support today). Validity is bit-packed.
fn check_numeric_buffers(
    data: &[u8],
    valid: &[u8],
    n_rows: usize,
    data_bytes_per_row: usize,
) -> PyResult<()> {
    let expected_data = n_rows * data_bytes_per_row;
    if data.len() < expected_data {
        return Err(PyValueError::new_err(format!(
            "polars_metal: data buffer is {got} B, need at least {expected} B for {n} rows",
            got = data.len(),
            expected = expected_data,
            n = n_rows,
        )));
    }
    let min_valid = (n_rows + 7) / 8;
    if valid.len() < min_valid {
        return Err(PyValueError::new_err(format!(
            "polars_metal: validity buffer is {got} B, need at least {expected} B for {n} rows",
            got = valid.len(),
            expected = min_valid,
            n = n_rows,
        )));
    }
    Ok(())
}

/// Length-check a bit-packed bool buffer (data or validity) for an
/// `n_rows`-long input to the `bool_and` / `bool_or` kernels.
fn check_bitpacked_buffer(buf: &[u8], n_rows: usize, label: &str) -> PyResult<()> {
    let min_bytes = (n_rows + 7) / 8;
    if buf.len() < min_bytes {
        return Err(PyValueError::new_err(format!(
            "polars_metal: {label} buffer is {got} B, need at least {expected} B for {n} rows",
            label = label,
            got = buf.len(),
            expected = min_bytes,
            n = n_rows,
        )));
    }
    Ok(())
}

// ============================================================================
// GroupBy PyO3 entry point — T28
// ============================================================================

/// Parsed view of a GroupBy plan dict received from the Python UDF.
#[derive(Debug)]
pub struct ParsedGroupByPlan {
    pub keys: Vec<ParsedKey>,
    pub aggs: Vec<ParsedAgg>,
}

/// One key column descriptor from the wire plan.
#[derive(Debug)]
pub struct ParsedKey {
    pub name: String,
    pub dtype: MetalDtype,
}

/// One aggregation descriptor from the wire plan. Mirrors the
/// [`crate::plan::AggSpec`] enum (Simple / Expression / Length).
#[derive(Debug, Clone)]
pub enum ParsedAgg {
    Simple {
        input_col: String,
        op: AggOp,
        output_alias: String,
    },
    Expression {
        expr: AggExpr,
        op: AggOp,
        output_alias: String,
    },
    Length {
        output_alias: String,
    },
}

impl ParsedAgg {
    /// Convenience: the output alias regardless of variant. Every variant
    /// carries one; dispatch reads this for result-column naming.
    pub fn output_alias(&self) -> &str {
        match self {
            ParsedAgg::Simple { output_alias, .. }
            | ParsedAgg::Expression { output_alias, .. }
            | ParsedAgg::Length { output_alias } => output_alias,
        }
    }
}

/// Errors produced while parsing the Python groupby plan dict.
#[derive(Debug, thiserror::Error)]
pub enum GroupByParseError {
    #[error("missing required field: {0}")]
    Missing(&'static str),
    #[error("wrong type for field: {0}")]
    WrongType(&'static str),
    #[error("unknown dtype: {0}")]
    UnknownDtype(String),
    #[error("unknown agg op: {0}")]
    UnknownOp(String),
}

/// Recursively parse one `{"kind": ..., ...}` dict emitted by the Python
/// walker's expression extractor into an [`AggExpr`].
///
/// The accepted shapes mirror `_walk_agg_expr_node` in `_walker.py`:
/// - `{"kind": "Column", "name": str}` → [`AggExpr::Column`]
/// - `{"kind": "LiteralF64", "value": float}` → [`AggExpr::LiteralF64`]
/// - `{"kind": "LiteralI64", "value": int}` → [`AggExpr::LiteralI64`]
/// - `{"kind": "Binary", "op": "Add"|"Sub"|"Mul"|"Div",
///       "lhs": <expr dict>, "rhs": <expr dict>}` → [`AggExpr::Binary`]
///
/// Unknown kinds or unknown binary ops produce
/// [`GroupByParseError::UnknownOp`]; missing or wrongly-typed fields
/// produce [`GroupByParseError::WrongType`].
fn parse_agg_expr_dict(d: &Bound<PyDict>) -> Result<AggExpr, GroupByParseError> {
    let kind: String = d
        .get_item("kind")
        .ok()
        .flatten()
        .and_then(|v| v.extract().ok())
        .ok_or(GroupByParseError::WrongType("expr.kind"))?;
    match kind.as_str() {
        "Column" => {
            let name: String = d
                .get_item("name")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .ok_or(GroupByParseError::WrongType("expr.name"))?;
            Ok(AggExpr::Column(name))
        }
        "LiteralF64" => {
            let v: f64 = d
                .get_item("value")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .ok_or(GroupByParseError::WrongType("expr.value(f64)"))?;
            Ok(AggExpr::LiteralF64(v))
        }
        "LiteralI64" => {
            let v: i64 = d
                .get_item("value")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .ok_or(GroupByParseError::WrongType("expr.value(i64)"))?;
            Ok(AggExpr::LiteralI64(v))
        }
        "Binary" => {
            let op_str: String = d
                .get_item("op")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .ok_or(GroupByParseError::WrongType("expr.op"))?;
            let op = match op_str.as_str() {
                "Add" => BinaryOp::Add,
                "Sub" => BinaryOp::Sub,
                "Mul" => BinaryOp::Mul,
                "Div" => BinaryOp::Div,
                _ => return Err(GroupByParseError::UnknownOp(format!("binary op {op_str}"))),
            };
            let lhs_dict: Bound<PyDict> = d
                .get_item("lhs")
                .ok()
                .flatten()
                .ok_or(GroupByParseError::WrongType("expr.lhs"))?
                .downcast_into()
                .map_err(|_| GroupByParseError::WrongType("expr.lhs(dict)"))?;
            let rhs_dict: Bound<PyDict> = d
                .get_item("rhs")
                .ok()
                .flatten()
                .ok_or(GroupByParseError::WrongType("expr.rhs"))?
                .downcast_into()
                .map_err(|_| GroupByParseError::WrongType("expr.rhs(dict)"))?;
            Ok(AggExpr::Binary {
                op,
                lhs: Box::new(parse_agg_expr_dict(&lhs_dict)?),
                rhs: Box::new(parse_agg_expr_dict(&rhs_dict)?),
            })
        }
        other => Err(GroupByParseError::UnknownOp(format!("expr kind={other}"))),
    }
}

/// Parse the `plan_dict` PyDict emitted by the Python walker into a
/// [`ParsedGroupByPlan`]. No new Python dep required — we read the dict
/// directly via PyO3.
///
/// Expected shape:
/// ```python
/// {
///     "keys": [["col_name", "I64"], ...],
///     "aggs": [{"input_col": "x", "op": "Sum", "output_alias": "x_sum"}, ...],
/// }
/// ```
pub fn parse_groupby_plan(plan: &Bound<PyDict>) -> Result<ParsedGroupByPlan, GroupByParseError> {
    // -- keys ----------------------------------------------------------------
    let keys_obj = plan
        .get_item("keys")
        .ok()
        .flatten()
        .ok_or(GroupByParseError::Missing("keys"))?;
    let keys_list: Bound<PyList> = keys_obj
        .downcast_into()
        .map_err(|_| GroupByParseError::WrongType("keys"))?;
    let mut keys = Vec::with_capacity(keys_list.len());
    for item in keys_list.iter() {
        let entry: Bound<PyList> = item
            .downcast_into()
            .map_err(|_| GroupByParseError::WrongType("key entry"))?;
        if entry.len() < 2 {
            return Err(GroupByParseError::WrongType("key entry"));
        }
        let name: String = entry
            .get_item(0)
            .ok()
            .and_then(|v| v.extract().ok())
            .ok_or(GroupByParseError::WrongType("key name"))?;
        let dtype_str: String = entry
            .get_item(1)
            .ok()
            .and_then(|v| v.extract().ok())
            .ok_or(GroupByParseError::WrongType("key dtype"))?;
        let dtype =
            MetalDtype::from_wire(&dtype_str).ok_or(GroupByParseError::UnknownDtype(dtype_str))?;
        keys.push(ParsedKey { name, dtype });
    }

    // -- aggs ----------------------------------------------------------------
    let aggs_obj = plan
        .get_item("aggs")
        .ok()
        .flatten()
        .ok_or(GroupByParseError::Missing("aggs"))?;
    let aggs_list: Bound<PyList> = aggs_obj
        .downcast_into()
        .map_err(|_| GroupByParseError::WrongType("aggs"))?;
    let mut aggs = Vec::with_capacity(aggs_list.len());
    for item in aggs_list.iter() {
        let entry: Bound<PyDict> = item
            .downcast_into()
            .map_err(|_| GroupByParseError::WrongType("agg entry"))?;

        // Backwards-compatible read: missing "kind" means M2-shape Simple/Length
        // (the existing wire format). Explicit "kind" means M3-shape; the
        // "Expression" arm requires an "expr" sub-dict whose parser lands
        // in Task 9 (Phase 2 Task 8 leaves it as a stub error).
        let kind: String = entry
            .get_item("kind")
            .ok()
            .flatten()
            .and_then(|v| v.extract().ok())
            .unwrap_or_else(|| {
                // Legacy shape: infer Length from op=="Len", Simple otherwise.
                let op_str: String = entry
                    .get_item("op")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or_default();
                if op_str == "Len" {
                    "Length".into()
                } else {
                    "Simple".into()
                }
            });

        let output_alias: String = entry
            .get_item("output_alias")
            .ok()
            .flatten()
            .and_then(|v| v.extract().ok())
            .ok_or(GroupByParseError::WrongType("output_alias"))?;

        let parsed = match kind.as_str() {
            "Length" => ParsedAgg::Length { output_alias },
            "Simple" => {
                // input_col is empty string for Len legacy; default empty if absent.
                let input_col: String = entry
                    .get_item("input_col")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or_default();
                let op_str: String = entry
                    .get_item("op")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .ok_or(GroupByParseError::WrongType("op"))?;
                let op = AggOp::from_wire(&op_str).ok_or(GroupByParseError::UnknownOp(op_str))?;
                ParsedAgg::Simple {
                    input_col,
                    op,
                    output_alias,
                }
            }
            "Expression" => {
                // Capability G (M3 Phase 2): the walker emits a recursive
                // AggExpr sub-tree under `expr` plus the outer reducer (`op`)
                // and alias. Parse the sub-tree, then re-validate the depth
                // cap on the Rust side as defence-in-depth (the walker's
                // `_AGG_EXPR_MAX_DEPTH` is the primary gate).
                let op_str: String = entry
                    .get_item("op")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .ok_or(GroupByParseError::WrongType("op"))?;
                let op = AggOp::from_wire(&op_str).ok_or(GroupByParseError::UnknownOp(op_str))?;
                let expr_dict: Bound<PyDict> = entry
                    .get_item("expr")
                    .ok()
                    .flatten()
                    .ok_or(GroupByParseError::WrongType("expr"))?
                    .downcast_into()
                    .map_err(|_| GroupByParseError::WrongType("expr(dict)"))?;
                let expr = parse_agg_expr_dict(&expr_dict)?;
                expr.validate()
                    .map_err(|_| GroupByParseError::WrongType("expr(too deep)"))?;
                ParsedAgg::Expression {
                    expr,
                    op,
                    output_alias,
                }
            }
            other => {
                return Err(GroupByParseError::UnknownOp(format!("kind={other}")));
            }
        };
        aggs.push(parsed);
    }

    Ok(ParsedGroupByPlan { keys, aggs })
}

/// Map a [`MetalDtype`] (plan layer) to the kernel-layer [`KeyDtype`].
fn metal_dtype_to_key_dtype(d: MetalDtype) -> KeyDtype {
    match d {
        MetalDtype::I64 => KeyDtype::I64,
        MetalDtype::F64 => KeyDtype::F64,
        MetalDtype::Bool => KeyDtype::Bool,
        MetalDtype::I32 => KeyDtype::I32,
        MetalDtype::F32 => KeyDtype::F32,
        MetalDtype::I8 => KeyDtype::I8,
        MetalDtype::I16 => KeyDtype::I16,
        MetalDtype::U8 => KeyDtype::U8,
        MetalDtype::U16 => KeyDtype::U16,
        MetalDtype::U32 => KeyDtype::U32,
    }
}

/// Build the `(AggKind, ValueColumn<'a>)` pair for a single agg request,
/// given the value column's raw byte buffers and dtype tag.
///
/// The `data` slice must be cast to the correct typed slice before being
/// wrapped in `ValueColumn`. We use `unsafe slice::from_raw_parts` here
/// for the same reason as the filter path: no `bytemuck` dep, and Arrow
/// buffers are guaranteed to be at least 8-byte aligned.
fn build_agg_kind_and_vcol<'a>(
    op: AggOp,
    dtype_tag: &str,
    data: &'a [u8],
    valid: &'a [u8],
    n_rows: usize,
) -> PyResult<(AggKind, ValueColumn<'a>)> {
    match (op, dtype_tag) {
        (AggOp::Sum, "I64") => {
            let expected = n_rows * 8;
            if data.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Sum/I64 data buffer too short: {got} < {expected}",
                    got = data.len()
                )));
            }
            // SAFETY: i64 has no invalid bit patterns; data.len() >= n_rows*8
            // and Arrow buffers are 64-byte aligned, so the reinterpret is
            // well-aligned for i64 (8-byte alignment).
            let typed: &[i64] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i64, n_rows) };
            Ok((AggKind::SumI64, ValueColumn::I64 { data: typed, valid }))
        }
        (AggOp::Sum, "F64") => {
            let expected = n_rows * 8;
            if data.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Sum/F64 data buffer too short: {got} < {expected}",
                    got = data.len()
                )));
            }
            // SAFETY: f64 has no invalid bit patterns; same alignment argument
            // as the i64 path.
            let typed: &[f64] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, n_rows) };
            Ok((AggKind::SumF64, ValueColumn::F64 { data: typed, valid }))
        }
        (AggOp::Mean, "I64") => {
            let expected = n_rows * 8;
            if data.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Mean/I64 data buffer too short: {got} < {expected}",
                    got = data.len()
                )));
            }
            // SAFETY: see Sum/I64.
            let typed: &[i64] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i64, n_rows) };
            Ok((AggKind::MeanI64, ValueColumn::I64 { data: typed, valid }))
        }
        (AggOp::Mean, "F64") => {
            let expected = n_rows * 8;
            if data.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Mean/F64 data buffer too short: {got} < {expected}",
                    got = data.len()
                )));
            }
            // SAFETY: see Sum/F64.
            let typed: &[f64] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, n_rows) };
            Ok((AggKind::MeanF64, ValueColumn::F64 { data: typed, valid }))
        }
        (AggOp::Min, "I64") => {
            let expected = n_rows * 8;
            if data.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Min/I64 data buffer too short: {got} < {expected}",
                    got = data.len()
                )));
            }
            // SAFETY: see Sum/I64.
            let typed: &[i64] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i64, n_rows) };
            Ok((AggKind::MinI64, ValueColumn::I64 { data: typed, valid }))
        }
        (AggOp::Min, "F64") => {
            let expected = n_rows * 8;
            if data.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Min/F64 data buffer too short: {got} < {expected}",
                    got = data.len()
                )));
            }
            // SAFETY: see Sum/F64.
            let typed: &[f64] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, n_rows) };
            Ok((AggKind::MinF64, ValueColumn::F64 { data: typed, valid }))
        }
        (AggOp::Max, "I64") => {
            let expected = n_rows * 8;
            if data.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Max/I64 data buffer too short: {got} < {expected}",
                    got = data.len()
                )));
            }
            // SAFETY: see Sum/I64.
            let typed: &[i64] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i64, n_rows) };
            Ok((AggKind::MaxI64, ValueColumn::I64 { data: typed, valid }))
        }
        (AggOp::Max, "F64") => {
            let expected = n_rows * 8;
            if data.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Max/F64 data buffer too short: {got} < {expected}",
                    got = data.len()
                )));
            }
            // SAFETY: see Sum/F64.
            let typed: &[f64] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, n_rows) };
            Ok((AggKind::MaxF64, ValueColumn::F64 { data: typed, valid }))
        }
        (AggOp::Count, "I64") => {
            let expected = n_rows * 8;
            if data.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Count/I64 data buffer too short: {got} < {expected}",
                    got = data.len()
                )));
            }
            // SAFETY: see Sum/I64.
            let typed: &[i64] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i64, n_rows) };
            Ok((AggKind::Count, ValueColumn::I64 { data: typed, valid }))
        }
        (AggOp::Count, "F64") => {
            let expected = n_rows * 8;
            if data.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Count/F64 data buffer too short: {got} < {expected}",
                    got = data.len()
                )));
            }
            // SAFETY: see Sum/F64.
            let typed: &[f64] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, n_rows) };
            Ok((AggKind::Count, ValueColumn::F64 { data: typed, valid }))
        }
        // --- I32 variants ---
        (AggOp::Sum, "I32") => {
            let expected = n_rows * 4;
            if data.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Sum/I32 data buffer too short: {got} < {expected}",
                    got = data.len()
                )));
            }
            // SAFETY: i32 has no invalid bit patterns; data.len() >= n_rows*4
            // and Arrow buffers are 64-byte aligned.
            let typed: &[i32] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i32, n_rows) };
            Ok((AggKind::SumI32, ValueColumn::I32 { data: typed, valid }))
        }
        (AggOp::Mean, "I32") => {
            let expected = n_rows * 4;
            if data.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Mean/I32 data buffer too short: {got} < {expected}",
                    got = data.len()
                )));
            }
            // SAFETY: see Sum/I32.
            let typed: &[i32] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i32, n_rows) };
            Ok((AggKind::MeanI32, ValueColumn::I32 { data: typed, valid }))
        }
        (AggOp::Min, "I32") => {
            let expected = n_rows * 4;
            if data.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Min/I32 data buffer too short: {got} < {expected}",
                    got = data.len()
                )));
            }
            // SAFETY: see Sum/I32.
            let typed: &[i32] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i32, n_rows) };
            Ok((AggKind::MinI32, ValueColumn::I32 { data: typed, valid }))
        }
        (AggOp::Max, "I32") => {
            let expected = n_rows * 4;
            if data.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Max/I32 data buffer too short: {got} < {expected}",
                    got = data.len()
                )));
            }
            // SAFETY: see Sum/I32.
            let typed: &[i32] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i32, n_rows) };
            Ok((AggKind::MaxI32, ValueColumn::I32 { data: typed, valid }))
        }
        (AggOp::Count, "I32") => {
            let expected = n_rows * 4;
            if data.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Count/I32 data buffer too short: {got} < {expected}",
                    got = data.len()
                )));
            }
            // SAFETY: see Sum/I32.
            let typed: &[i32] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i32, n_rows) };
            Ok((AggKind::Count, ValueColumn::I32 { data: typed, valid }))
        }
        // --- F32 variants ---
        (AggOp::Sum, "F32") => {
            let expected = n_rows * 4;
            if data.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Sum/F32 data buffer too short: {got} < {expected}",
                    got = data.len()
                )));
            }
            // SAFETY: f32 has no invalid bit patterns; data.len() >= n_rows*4
            // and Arrow buffers are 64-byte aligned.
            let typed: &[f32] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n_rows) };
            Ok((AggKind::SumF32, ValueColumn::F32 { data: typed, valid }))
        }
        (AggOp::Mean, "F32") => {
            let expected = n_rows * 4;
            if data.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Mean/F32 data buffer too short: {got} < {expected}",
                    got = data.len()
                )));
            }
            // SAFETY: see Sum/F32.
            let typed: &[f32] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n_rows) };
            Ok((AggKind::MeanF32, ValueColumn::F32 { data: typed, valid }))
        }
        (AggOp::Min, "F32") => {
            let expected = n_rows * 4;
            if data.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Min/F32 data buffer too short: {got} < {expected}",
                    got = data.len()
                )));
            }
            // SAFETY: see Sum/F32.
            let typed: &[f32] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n_rows) };
            Ok((AggKind::MinF32, ValueColumn::F32 { data: typed, valid }))
        }
        (AggOp::Max, "F32") => {
            let expected = n_rows * 4;
            if data.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Max/F32 data buffer too short: {got} < {expected}",
                    got = data.len()
                )));
            }
            // SAFETY: see Sum/F32.
            let typed: &[f32] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n_rows) };
            Ok((AggKind::MaxF32, ValueColumn::F32 { data: typed, valid }))
        }
        (AggOp::Count, "F32") => {
            let expected = n_rows * 4;
            if data.len() < expected {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Count/F32 data buffer too short: {got} < {expected}",
                    got = data.len()
                )));
            }
            // SAFETY: see Sum/F32.
            let typed: &[f32] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n_rows) };
            Ok((AggKind::Count, ValueColumn::F32 { data: typed, valid }))
        }
        (op, dtype) => Err(PyValueError::new_err(format!(
            "polars_metal: unsupported (agg_op, dtype) combination: {op:?} / {dtype}"
        ))),
    }
}

/// Pack a `Vec<bool>` validity slice into a 4-byte-aligned little-endian
/// bit-packed validity bitmap, matching Arrow's convention.
fn pack_valid_bitmap(bits: &[bool]) -> Vec<u8> {
    let n_bytes = ((bits.len() + 7) / 8 + 3) & !3;
    let n_bytes = n_bytes.max(4);
    let mut out = vec![0u8; n_bytes];
    for (i, &b) in bits.iter().enumerate() {
        if b {
            out[i >> 3] |= 1 << (i & 7);
        }
    }
    out
}

/// Encode a [`DecodedColumn`] (key output) into `(dtype_tag, data_bytes,
/// valid_bytes)` for the wire return format.
fn encode_decoded_column(
    d: &polars_metal_kernels::groupby::DecodedColumn,
) -> (&'static str, Vec<u8>, Vec<u8>) {
    use polars_metal_kernels::groupby::DecodedColumn;
    match d {
        DecodedColumn::I64 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("I64", data, v)
        }
        DecodedColumn::F64 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("F64", data, v)
        }
        DecodedColumn::Bool { values, valid } => {
            // Bool data is also bit-packed (Arrow convention).
            let data = pack_valid_bitmap(values);
            let v = pack_valid_bitmap(valid);
            ("Bool", data, v)
        }
        DecodedColumn::I32 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("I32", data, v)
        }
        DecodedColumn::F32 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("F32", data, v)
        }
        DecodedColumn::I8 { values, valid } => {
            let data: Vec<u8> = values.iter().map(|v| *v as u8).collect();
            let v = pack_valid_bitmap(valid);
            ("I8", data, v)
        }
        DecodedColumn::I16 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("I16", data, v)
        }
        DecodedColumn::U8 { values, valid } => {
            let data: Vec<u8> = values.to_vec();
            let v = pack_valid_bitmap(valid);
            ("U8", data, v)
        }
        DecodedColumn::U16 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("U16", data, v)
        }
        DecodedColumn::U32 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("U32", data, v)
        }
    }
}

/// Encode an [`AggOutput`] into `(dtype_tag, data_bytes, valid_bytes)`.
///
/// U64 outputs (Count, Len) are cast to u32 — Polars returns u32 for both
/// `pl.col(x).count()` and `pl.len()`. At M2 row counts > 4 billion per
/// group are unrealistic, so this truncation is safe in practice.
fn encode_agg_output(
    o: &polars_metal_kernels::groupby::AggOutput,
) -> (&'static str, Vec<u8>, Vec<u8>) {
    use polars_metal_kernels::groupby::AggOutput;
    match o {
        AggOutput::I64 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("I64", data, v)
        }
        AggOutput::F64 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("F64", data, v)
        }
        AggOutput::U64 { values } => {
            // Cast u64 → u32. Counts / lens that fit in u32 are the common case.
            let data: Vec<u8> = values
                .iter()
                .flat_map(|&v| (v as u32).to_le_bytes())
                .collect();
            let n = values.len();
            // All-ones bitmap: counts/lens are never null.
            let valid_bytes = (((n + 7) / 8 + 3) & !3).max(4);
            let valid = vec![0xFFu8; valid_bytes];
            ("U32", data, valid)
        }
        AggOutput::I32 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("I32", data, v)
        }
        AggOutput::F32 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("F32", data, v)
        }
    }
}

/// Convert the kernel `GroupByError` to a `PyErr`.
fn groupby_err(e: GroupByError) -> PyErr {
    PyValueError::new_err(format!("polars_metal: dispatch_groupby failed: {e}"))
}

/// PyO3 entry point: `polars_metal._native.execute_groupby`.
///
/// # Wire protocol
///
/// `plan_dict` shape:
/// ```python
/// {
///     "keys": [["col_name", "I64"], ...],           # list of [name, dtype_tag]
///     "aggs": [
///         {"input_col": "x", "op": "Sum", "output_alias": "x_sum"},
///         {"input_col": "",  "op": "Len", "output_alias": "n"},
///     ],
/// }
/// ```
///
/// `columns`: one `(name, dtype_tag, data_bytes, valid_bytes)` tuple per
/// column that appears in either `keys` or `aggs.input_col`.
///
/// # Returns
///
/// A Python list of `(col_name: str, dtype_tag: str, data: bytes,
/// valid: bytes)` tuples — first the key columns (in `keys` order), then
/// the agg outputs (in `aggs` order, named by `output_alias`).  The
/// Python UDF (`_udf.py::_build_groupby`) reassembles these into a Polars
/// DataFrame via PyArrow, matching the pattern of `execute_filter_compact`.
///
/// Supported dtype tags on output: `"I64"`, `"F64"`, `"Bool"`, `"U32"`.
#[pyfunction]
pub fn execute_groupby<'py>(
    py: Python<'py>,
    plan_dict: Bound<'py, PyDict>,
    n_rows: usize,
    columns: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyList>> {
    // 1. Parse plan dict → ParsedGroupByPlan.
    let parsed = parse_groupby_plan(&plan_dict)
        .map_err(|e| PyValueError::new_err(format!("polars_metal: plan parse error: {e}")))?;

    // 2. Build lookup: col_name → (dtype_tag, data_bytes, valid_bytes).
    //    We hold references into the PyBytes objects rather than copying.
    //    The PyBytes objects are alive for the duration of this function, so
    //    the byte slice references are safe to use until we return.
    let mut by_name: HashMap<String, (String, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> =
        HashMap::new();
    for (idx, entry) in columns.iter().enumerate() {
        let tup: Bound<PyTuple> = entry.downcast_into().map_err(|_| {
            PyValueError::new_err(format!(
                "polars_metal: execute_groupby columns[{idx}] must be a tuple"
            ))
        })?;
        if tup.len() != 4 {
            return Err(PyValueError::new_err(format!(
                "polars_metal: execute_groupby columns[{idx}] must have 4 elements (name, dtype, data, valid), got {}",
                tup.len()
            )));
        }
        let name: String = tup.get_item(0)?.extract()?;
        let dtype_tag: String = tup.get_item(1)?.extract()?;
        let data_py: Bound<PyBytes> = tup.get_item(2)?.downcast_into().map_err(|_| {
            PyValueError::new_err(format!(
                "polars_metal: execute_groupby columns[{idx}].data must be bytes"
            ))
        })?;
        let valid_py: Bound<PyBytes> = tup.get_item(3)?.downcast_into().map_err(|_| {
            PyValueError::new_err(format!(
                "polars_metal: execute_groupby columns[{idx}].valid must be bytes"
            ))
        })?;
        by_name.insert(name, (dtype_tag, data_py, valid_py));
    }

    // 3. Build KeyColumn slice.
    //    We must keep the &[u8] alive until after dispatch_groupby returns.
    //    We collect (data_bytes, valid_bytes) references before constructing
    //    KeyColumn structs so their lifetimes are tied to the PyBytes in
    //    `by_name`, which outlives the dispatch call.
    let mut key_byte_refs: Vec<(&[u8], &[u8])> = Vec::with_capacity(parsed.keys.len());
    for k in &parsed.keys {
        let (_, data_py, valid_py) = by_name.get(&k.name).ok_or_else(|| {
            PyKeyError::new_err(format!(
                "polars_metal: key column {:?} not found in upstream columns",
                k.name
            ))
        })?;
        key_byte_refs.push((data_py.as_bytes(), valid_py.as_bytes()));
    }
    let key_cols: Vec<KeyColumn<'_>> = parsed
        .keys
        .iter()
        .zip(key_byte_refs.iter())
        .map(|(k, (data, valid))| KeyColumn {
            name: k.name.clone(),
            dtype: metal_dtype_to_key_dtype(k.dtype),
            data,
            valid,
            n_rows,
        })
        .collect();

    // 4. Build (AggRequest, ValueColumn) pairs.
    //    Len needs no value column; we use a zero-length I64 placeholder
    //    because `run_one_agg` for Len ignores the ValueColumn entirely.
    let empty_data: &[u8] = &[];
    let empty_valid: &[u8] = &[];
    // SAFETY: &[] cast to &[i64] is a zero-length slice — no pointer
    // arithmetic occurs. The slice is valid and the pointer is non-null
    // (a valid empty slice). This is the established pattern for
    // zero-length typed slices in this codebase.
    let empty_i64: &[i64] =
        unsafe { std::slice::from_raw_parts(empty_data.as_ptr() as *const i64, 0) };

    // Tuple type alias to keep the Vec type readable for clippy.
    type AggByteRef<'a> = (&'a [u8], &'a [u8], String);

    let mut agg_byte_refs: Vec<Option<AggByteRef<'_>>> = Vec::with_capacity(parsed.aggs.len());
    for agg in &parsed.aggs {
        match agg {
            ParsedAgg::Length { .. } => {
                agg_byte_refs.push(None);
            }
            ParsedAgg::Simple { input_col, .. } => {
                let (dtype_tag, data_py, valid_py) = by_name.get(input_col).ok_or_else(|| {
                    PyKeyError::new_err(format!(
                        "polars_metal: agg input column {input_col:?} not found in upstream columns"
                    ))
                })?;
                agg_byte_refs.push(Some((
                    data_py.as_bytes(),
                    valid_py.as_bytes(),
                    dtype_tag.clone(),
                )));
            }
            ParsedAgg::Expression { .. } => {
                // Phase 3 wires the fused-kernel consumer; the Task 10
                // router gate ensures we never reach here at runtime.
                // Defensively reject if we do.
                return Err(PyValueError::new_err(
                    "polars_metal: AggSpec::Expression dispatch awaits Phase 3 fused-kernel consumer",
                ));
            }
        }
    }

    let mut agg_specs: Vec<(AggRequest, ValueColumn<'_>)> = Vec::with_capacity(parsed.aggs.len());
    for (i, agg) in parsed.aggs.iter().enumerate() {
        match agg {
            ParsedAgg::Length { .. } => {
                agg_specs.push((
                    AggRequest {
                        kind: AggKind::Len,
                        input_col_idx: i,
                    },
                    ValueColumn::I64 {
                        data: empty_i64,
                        valid: empty_valid,
                    },
                ));
            }
            ParsedAgg::Simple { op, .. } => {
                // This entry was populated in the previous loop (Length was already
                // handled via the matching arm), so `None` here is a logic bug
                // rather than a runtime condition — surface it as an internal error.
                let (data, valid, dtype_tag) = agg_byte_refs[i].as_ref().ok_or_else(|| {
                    PyValueError::new_err(
                        "polars_metal: internal error: missing agg byte ref for Simple agg",
                    )
                })?;
                let (kind, vcol) = build_agg_kind_and_vcol(*op, dtype_tag, data, valid, n_rows)?;
                agg_specs.push((
                    AggRequest {
                        kind,
                        input_col_idx: i,
                    },
                    vcol,
                ));
            }
            ParsedAgg::Expression { .. } => {
                // Same defence-in-depth as the byte-ref pass; router gate
                // (Task 10) prevents reaching here at runtime.
                return Err(PyValueError::new_err(
                    "polars_metal: AggSpec::Expression dispatch awaits Phase 3 fused-kernel consumer",
                ));
            }
        }
    }

    // 5. Acquire device + queue.
    let device = MetalDevice::system_default()
        .map_err(|e| crate::engine_err(crate::EngineError::Buffer(e)))?;
    let mut queue = CommandQueue::new(&device)
        .map_err(|e| crate::engine_err(crate::EngineError::Other(format!("command queue: {e}"))))?;

    // 6. Dispatch.
    let result = polars_metal_kernels::groupby::dispatch_groupby(
        &device, &mut queue, &key_cols, &agg_specs, n_rows,
    )
    .map_err(groupby_err)?;

    // 7. Encode result as a list of (name, dtype_tag, data, valid) tuples.
    //    Key columns first, then agg outputs.
    let out_list = PyList::empty_bound(py);

    for (i, key) in parsed.keys.iter().enumerate() {
        let decoded = &result.decoded_keys[i];
        let (dtype_tag, data, valid) = encode_decoded_column(decoded);
        let tup = PyTuple::new_bound(
            py,
            [
                key.name.clone().into_py(py),
                dtype_tag.into_py(py),
                PyBytes::new_bound(py, &data).into_py(py),
                PyBytes::new_bound(py, &valid).into_py(py),
            ],
        );
        out_list.append(tup)?;
    }

    for (i, agg) in parsed.aggs.iter().enumerate() {
        let output = &result.agg_outputs[i];
        let (dtype_tag, data, valid) = encode_agg_output(output);
        let tup = PyTuple::new_bound(
            py,
            [
                agg.output_alias().to_string().into_py(py),
                dtype_tag.into_py(py),
                PyBytes::new_bound(py, &data).into_py(py),
                PyBytes::new_bound(py, &valid).into_py(py),
            ],
        );
        out_list.append(tup)?;
    }

    Ok(out_list)
}

/// PyO3 entry point exposed as `polars_metal._native.bool_and_dispatch`.
///
/// Combines two bit-packed nullable Boolean predicates with Polars'
/// 3-valued AND (false dominates; otherwise null propagates) and
/// returns a fresh `(data, valid)` byte-pair. The Python UDF calls
/// this when the walker emits a `BinaryExpr(Operator.And, ...)` whose
/// operands both resolve to Bool — the recursive ``_evaluate_predicate``
/// in ``_udf.py`` materialises the two sub-predicate bitmaps and hands
/// them here.
///
/// All four input buffers must be at least ``ceil(n_rows / 8)`` bytes;
/// the kernel padding (4-byte alignment for `device atomic_uint`) is
/// handled inside the dispatcher.
#[pyfunction]
pub fn bool_and_dispatch<'py>(
    py: Python<'py>,
    lhs_data: &Bound<'py, PyBytes>,
    lhs_valid: &Bound<'py, PyBytes>,
    rhs_data: &Bound<'py, PyBytes>,
    rhs_valid: &Bound<'py, PyBytes>,
    n_rows: usize,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    dispatch_logical_py(
        py, lhs_data, lhs_valid, rhs_data, rhs_valid, n_rows, /*is_and=*/ true,
    )
}

/// PyO3 entry point exposed as `polars_metal._native.bool_or_dispatch`.
///
/// 3-valued OR mirror of [`bool_and_dispatch`] — true dominates,
/// otherwise null propagates.
#[pyfunction]
pub fn bool_or_dispatch<'py>(
    py: Python<'py>,
    lhs_data: &Bound<'py, PyBytes>,
    lhs_valid: &Bound<'py, PyBytes>,
    rhs_data: &Bound<'py, PyBytes>,
    rhs_valid: &Bound<'py, PyBytes>,
    n_rows: usize,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    dispatch_logical_py(
        py, lhs_data, lhs_valid, rhs_data, rhs_valid, n_rows, /*is_and=*/ false,
    )
}

/// Shared dispatch body for [`bool_and_dispatch`] and [`bool_or_dispatch`].
///
/// The two pyfunctions have identical input/output shapes — they differ
/// only in the kernel called inside `polars_metal_kernels::logical`.
/// Keeping the wrapper monomorphic on a boolean flag (rather than a
/// function pointer) keeps `cargo expand` output readable and gives
/// the rust optimizer a clean inline target.
#[allow(clippy::too_many_arguments)]
fn dispatch_logical_py<'py>(
    py: Python<'py>,
    lhs_data: &Bound<'py, PyBytes>,
    lhs_valid: &Bound<'py, PyBytes>,
    rhs_data: &Bound<'py, PyBytes>,
    rhs_valid: &Bound<'py, PyBytes>,
    n_rows: usize,
    is_and: bool,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let lhs_data_b = lhs_data.as_bytes();
    let lhs_valid_b = lhs_valid.as_bytes();
    let rhs_data_b = rhs_data.as_bytes();
    let rhs_valid_b = rhs_valid.as_bytes();
    check_bitpacked_buffer(lhs_data_b, n_rows, "lhs_data")?;
    check_bitpacked_buffer(lhs_valid_b, n_rows, "lhs_valid")?;
    check_bitpacked_buffer(rhs_data_b, n_rows, "rhs_data")?;
    check_bitpacked_buffer(rhs_valid_b, n_rows, "rhs_valid")?;

    let (device, mut queue) = new_device_and_queue()?;
    let min_out = cmp_out_min_bytes(n_rows);
    let mut out_data = vec![0u8; min_out];
    let mut out_valid = vec![0u8; min_out];

    let kernel_name = if is_and { "bool_and" } else { "bool_or" };
    let result = if is_and {
        dispatch_bool_and(
            &device,
            &mut queue,
            lhs_data_b,
            lhs_valid_b,
            rhs_data_b,
            rhs_valid_b,
            n_rows,
            &mut out_data,
            &mut out_valid,
        )
    } else {
        dispatch_bool_or(
            &device,
            &mut queue,
            lhs_data_b,
            lhs_valid_b,
            rhs_data_b,
            rhs_valid_b,
            n_rows,
            &mut out_data,
            &mut out_valid,
        )
    };
    result.map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "polars_metal: {kernel_name} dispatch failed: {e}"
        ))
    })?;
    Ok((
        PyBytes::new_bound(py, &out_data),
        PyBytes::new_bound(py, &out_valid),
    ))
}
