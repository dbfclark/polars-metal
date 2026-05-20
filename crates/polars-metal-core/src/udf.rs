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
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::pipeline::{compact_bool, compact_f64, compact_i64};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

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
        let (out_data, out_valid, n_out) = compact_one_column(
            &device,
            &mut queue,
            dtype,
            n_rows,
            data_b,
            valid_b,
            pred_data_bytes,
            pred_valid_bytes,
            &name,
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

/// Run the compaction pipeline on a single column and return its
/// `(data_bytes, valid_bytes, n_out)` triple. Per-dtype branching is
/// contained here so `execute_filter_compact`'s loop body stays
/// dtype-agnostic.
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
    pred_data: &[u8],
    pred_valid: &[u8],
    column_name: &str,
) -> PyResult<(Vec<u8>, Vec<u8>, usize)> {
    let min_valid_bytes = (n_rows + 7) / 8;
    if src_valid.len() < min_valid_bytes {
        return Err(PyValueError::new_err(format!(
            "polars_metal: column {column_name:?} validity buffer is {got} B, need {expected} B for {n} rows",
            got = src_valid.len(),
            expected = min_valid_bytes,
            n = n_rows,
        )));
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
            let result = compact_i64(
                device, queue, src_typed, src_valid, pred_data, pred_valid, n_rows,
            )
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
            Ok((data_bytes, result.valid, result.n_out))
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
            let result = compact_f64(
                device, queue, src_typed, src_valid, pred_data, pred_valid, n_rows,
            )
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
            Ok((data_bytes, result.valid, result.n_out))
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
                device, queue, src_data, src_valid, pred_data, pred_valid, n_rows,
            )
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "polars_metal: compact_bool({column_name:?}) failed: {e}"
                ))
            })?;
            // For bool, `result.data` is already bit-packed u8 (padded to
            // 4-byte alignment by the kernel). The Python side will trim to
            // `ceil(n_out / 8)` before calling PyArrow.
            Ok((result.data, result.valid, result.n_out))
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
