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
use polars_metal_kernels::cmp::{
    dispatch_cmp_f64, dispatch_cmp_f64_scalar, dispatch_cmp_i64, dispatch_cmp_i64_scalar, CompareOp,
};
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::logical::{dispatch_bool_and, dispatch_bool_or};
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
    match s {
        "I64" => Ok(MetalDtype::I64),
        "F64" => Ok(MetalDtype::F64),
        "Bool" => Ok(MetalDtype::Bool),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "polars_metal: unknown MetalDtype tag {other:?}"
        ))),
    }
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
