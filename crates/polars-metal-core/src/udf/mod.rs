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
use common::{
    check_bitpacked_buffer, check_numeric_buffers, new_device_and_queue, pack_valid_bitmap,
};
mod compact;
mod compare;
mod dt;
mod dtw;
mod fused_expr;
mod groupby;
mod logical;
mod predicate;
mod rolling;

use crate::plan::{AggExpr, AggOp, BinaryOp, MetalDtype, MetalPlanNode, PredicateAst};
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::aggregate_fused::cache::FusedLibraryCache;
use polars_metal_kernels::aggregate_fused::signature::{
    AggExpr as KAggExpr, AggOp as KAggOp, AggSpec as KAggSpec, BinaryOp as KBinaryOp,
    MetalDtype as KMetalDtype,
};
use polars_metal_kernels::cmp::{
    dispatch_cmp_f64, dispatch_cmp_f64_scalar, dispatch_cmp_i64, dispatch_cmp_i64_scalar, CompareOp,
};
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::groupby::{
    dispatch_groupby_fused, AggKind, AggRequest, GroupByError, KeyColumn, KeyDtype, ValueColumn,
};
use polars_metal_kernels::logical::{dispatch_bool_and, dispatch_bool_or};
use polars_metal_kernels::pipeline::{
    compact_bool, compact_f64, compact_i64, compute_keep_and_prefix,
};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, OnceLock};

// ----------------------------------------------------------------------------
// IR → kernel-layer mirrors (Phase 3 / Task 15)
// ----------------------------------------------------------------------------

/// Convert IR `AggOp` → kernel-layer `AggOp` mirror.
fn convert_agg_op(op: AggOp) -> KAggOp {
    match op {
        AggOp::Sum => KAggOp::Sum,
        AggOp::Mean => KAggOp::Mean,
        AggOp::Count => KAggOp::Count,
        AggOp::Min => KAggOp::Min,
        AggOp::Max => KAggOp::Max,
        AggOp::Len => KAggOp::Len,
    }
}

/// Convert IR `BinaryOp` → kernel-layer mirror.
fn convert_binary_op(op: BinaryOp) -> KBinaryOp {
    match op {
        BinaryOp::Add => KBinaryOp::Add,
        BinaryOp::Sub => KBinaryOp::Sub,
        BinaryOp::Mul => KBinaryOp::Mul,
        BinaryOp::Div => KBinaryOp::Div,
    }
}

/// Convert IR `AggExpr` → kernel-layer mirror (mechanical tree walk).
fn convert_agg_expr(expr: &AggExpr) -> KAggExpr {
    match expr {
        AggExpr::Column(name) => KAggExpr::Column(name.clone()),
        AggExpr::LiteralF64(v) => KAggExpr::LiteralF64(*v),
        AggExpr::LiteralI64(v) => KAggExpr::LiteralI64(*v),
        AggExpr::Binary { op, lhs, rhs } => KAggExpr::Binary {
            op: convert_binary_op(*op),
            lhs: Box::new(convert_agg_expr(lhs)),
            rhs: Box::new(convert_agg_expr(rhs)),
        },
    }
}

/// Wire dtype tag (`"I32"` / `"F32"` / ...) → kernel-layer `MetalDtype`.
fn wire_dtype_tag_to_kernel(tag: &str) -> Option<KMetalDtype> {
    match tag {
        "I64" => Some(KMetalDtype::I64),
        "F64" => Some(KMetalDtype::F64),
        "Bool" => Some(KMetalDtype::Bool),
        "I32" => Some(KMetalDtype::I32),
        "F32" => Some(KMetalDtype::F32),
        "I8" => Some(KMetalDtype::I8),
        "I16" => Some(KMetalDtype::I16),
        "U8" => Some(KMetalDtype::U8),
        "U16" => Some(KMetalDtype::U16),
        "U32" => Some(KMetalDtype::U32),
        // M3 Phase 7: Utf8 is a key dtype only — never a valid agg value
        // column. Returning None here forces `decide_groupby_dispatch` down
        // the PerAgg branch (or further fallback) if a router bug ever lifts
        // a Utf8 column into agg-input position.
        "Utf8" | "String" => None,
        _ => None,
    }
}

/// Routing decision for one groupby query: fused single-kernel dispatch
/// vs M2's per-agg loop. See `decide_groupby_dispatch` for the rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroupByDispatchChoice {
    Fused,
    PerAgg,
}

/// Decide between the fused kernel and M2's per-agg path.
///
/// Rules (Task 15):
///   1. If any agg is Expression → Fused (Expression has no per-agg fallback).
///   2. Otherwise, if every agg is Simple/Length AND there are ≥ 2 aggs
///      AND the signature is fused-supported (all 32-bit-or-narrower inputs)
///      → Fused.
///   3. Otherwise → PerAgg (single-agg queries, F64/I64 inputs, etc.).
///
/// Caller must have already verified the value columns referenced by the
/// aggs are present in the HashMap. The signature is built once here and
/// inspected for support; the same signature is reused at dispatch time.
fn decide_groupby_dispatch(
    parsed: &[ParsedAgg],
    by_name: &HashMap<String, (String, &[u8], &[u8])>,
) -> GroupByDispatchChoice {
    let has_expression = parsed
        .iter()
        .any(|a| matches!(a, ParsedAgg::Expression { .. }));
    let n_simple_or_len = parsed
        .iter()
        .filter(|a| !matches!(a, ParsedAgg::Expression { .. }))
        .count();

    // Check that every referenced column is 32-bit-or-narrower (fused-only
    // supports that). Build a tentative signature inline.
    let mut col_dtypes: BTreeMap<String, KMetalDtype> = BTreeMap::new();
    let mut all_fused_supported = true;
    for a in parsed {
        match a {
            ParsedAgg::Simple { input_col, .. } => {
                let Some((dt_tag, _, _)) = by_name.get(input_col) else {
                    return GroupByDispatchChoice::PerAgg;
                };
                let Some(kdt) = wire_dtype_tag_to_kernel(dt_tag) else {
                    return GroupByDispatchChoice::PerAgg;
                };
                if matches!(kdt, KMetalDtype::F64 | KMetalDtype::I64) {
                    all_fused_supported = false;
                }
                col_dtypes.entry(input_col.clone()).or_insert(kdt);
            }
            ParsedAgg::Expression { expr, .. } => {
                for c in expr.referenced_columns() {
                    let Some((dt_tag, _, _)) = by_name.get(&c) else {
                        return GroupByDispatchChoice::PerAgg;
                    };
                    let Some(kdt) = wire_dtype_tag_to_kernel(dt_tag) else {
                        return GroupByDispatchChoice::PerAgg;
                    };
                    if matches!(kdt, KMetalDtype::F64 | KMetalDtype::I64) {
                        all_fused_supported = false;
                    }
                    col_dtypes.entry(c).or_insert(kdt);
                }
            }
            ParsedAgg::Length { .. } => {}
        }
    }

    if has_expression && all_fused_supported {
        return GroupByDispatchChoice::Fused;
    }
    if !has_expression && n_simple_or_len >= 2 && all_fused_supported {
        return GroupByDispatchChoice::Fused;
    }
    GroupByDispatchChoice::PerAgg
}

/// Process-wide fused-library cache. Constructed lazily on first dispatch
/// when the system default Metal device is acquirable.
static FUSED_CACHE: OnceLock<FusedLibraryCache> = OnceLock::new();

/// Process-global reusable staging buffer for `execute_dt` inputs (B3b).
/// One buffer, grown to the largest input seen; the `Mutex` serializes dt
/// dispatches (Metal command submission serializes anyway). Designed so other
/// kernel bindings can adopt the same pattern with their own pool later.
static DT_STAGING: OnceLock<Mutex<polars_metal_buffer::StagingPool>> = OnceLock::new();

fn get_or_init_fused_cache(device: &MetalDevice) -> &'static FusedLibraryCache {
    FUSED_CACHE.get_or_init(|| FusedLibraryCache::new(device.clone()))
}

/// Pre-compile common fused-agg signatures into the process-wide
/// `FUSED_CACHE` (Task 18). Called from `python/polars_metal/__init__.py`
/// at import time so the first user query of a common shape
/// (single-column F32 Sum, Q1-shape 10-agg, etc.) does not pay the MSL
/// compile cost.
///
/// Best-effort: if the Metal device cannot be acquired (no Metal-capable
/// hardware), or any individual signature fails to compile, the warmup
/// returns the number of signatures actually queued (0 on device failure).
/// The Python wrapper swallows exceptions too — warmup is advisory and
/// must not break engine startup.
///
/// Returns the count of signatures the cache was asked to warm; the
/// Python side uses this for logging and the integration test.
#[pyfunction]
pub fn warmup_common_fused_signatures() -> i32 {
    use polars_metal_kernels::aggregate_fused::cache::common_signatures;

    let Ok(device) = MetalDevice::system_default() else {
        // No Metal device available — running under a non-Metal harness
        // (e.g. CI without a GPU). Warmup is a no-op; skip without error.
        return 0;
    };
    let cache = get_or_init_fused_cache(&device);
    let sigs = common_signatures();
    let count = sigs.len() as i32;
    cache.warmup(&sigs);
    count
}

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
        MetalDtype::I8
        | MetalDtype::I16
        | MetalDtype::U8
        | MetalDtype::U16
        | MetalDtype::U32
        | MetalDtype::U64 => {
            Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "polars_metal: filter compaction for {column_name:?} ({dtype:?}) not supported; filter should route CPU"
            )))
        }
        // M3 Phase 7: Utf8 keys go through the composite-key encoder, but the
        // filter compaction path has no Utf8 support yet. Filter on Utf8 must
        // route CPU; reaching this arm is a router bug.
        MetalDtype::Utf8 => {
            Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
                "polars_metal: filter on Utf8 columns ({column_name:?}): not yet wired"
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
///
/// Returns `Err` for dtypes that have no groupby-key encoding. The only such
/// case today is `U64`: the composite-key encoder has no 64-bit-unsigned
/// `KeyDtype`, and the groupby kernel is conformance-only (Non-goals — not
/// extended). The Python walker already gates U64 keys to CPU fallback, so
/// this arm is defensive: if a router bug ever lifts a U64-key groupby, we
/// surface a clear error rather than mis-encoding or panicking.
fn metal_dtype_to_key_dtype(d: MetalDtype) -> Result<KeyDtype, String> {
    Ok(match d {
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
        MetalDtype::Utf8 => KeyDtype::Utf8,
        MetalDtype::U64 => {
            return Err(
                "groupby key dtype UInt64 has no composite-key encoding (groupby is \
                 conformance-only and not extended); should route CPU at the walker"
                    .to_string(),
            )
        }
    })
}

/// Build the `(AggKind, ValueColumn<'a>)` pair for a single agg request,
/// given the value column's raw byte buffers and dtype tag.
///
/// The `data` slice must be cast to the correct typed slice before being
/// wrapped in `ValueColumn`. We use `unsafe slice::from_raw_parts` here
/// for the same reason as the filter path: no `bytemuck` dep, and Arrow
/// buffers are guaranteed to be at least 8-byte aligned.
/// Construct a typed `ValueColumn` view over raw bytes for the fused
/// groupby dispatcher.
///
/// Unlike [`build_agg_kind_and_vcol`] this does not derive a kernel-side
/// `AggKind` — the fused dispatcher derives output shape from the
/// `AggSignature`. Only the 32-bit-or-narrower dtypes the fused emitter
/// supports are accepted; F64/I64 callers must route through the M2 path.
fn build_value_column<'a>(
    dtype_tag: &str,
    data: &'a [u8],
    valid: &'a [u8],
    n_rows: usize,
) -> Result<ValueColumn<'a>, String> {
    match dtype_tag {
        "I32" => {
            let expected = n_rows * 4;
            if data.len() < expected {
                return Err(format!(
                    "I32 data buffer too short: {got} < {expected}",
                    got = data.len()
                ));
            }
            // SAFETY: i32 has no invalid bit patterns; Arrow buffers are
            // 64-byte aligned so the reinterpret meets the 4-byte alignment.
            let typed: &[i32] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i32, n_rows) };
            Ok(ValueColumn::I32 { data: typed, valid })
        }
        "F32" => {
            let expected = n_rows * 4;
            if data.len() < expected {
                return Err(format!(
                    "F32 data buffer too short: {got} < {expected}",
                    got = data.len()
                ));
            }
            // SAFETY: f32 has no invalid bit patterns; same alignment as I32.
            let typed: &[f32] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n_rows) };
            Ok(ValueColumn::F32 { data: typed, valid })
        }
        other => Err(format!(
            "dtype {other} not supported by fused groupby (only I32/F32 currently)"
        )),
    }
}

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
        DecodedColumn::Utf8 { values, valid } => {
            // Wire format (parsed Python-side in Task 34):
            //   [n_rows: u32 le]
            //   [offsets: (n_rows+1) × i32 le]   Arrow Utf8 offset convention
            //   [concatenated string bytes]
            let n = values.len() as u32;
            let mut data: Vec<u8> = Vec::new();
            data.extend_from_slice(&n.to_le_bytes());
            let mut offsets: Vec<i32> = Vec::with_capacity(values.len() + 1);
            let mut acc: i32 = 0;
            offsets.push(0);
            let mut bytes_blob: Vec<u8> = Vec::new();
            for (s, &is_valid) in values.iter().zip(valid.iter()) {
                if is_valid {
                    bytes_blob.extend_from_slice(s.as_bytes());
                    acc = acc.saturating_add(s.len() as i32);
                }
                offsets.push(acc);
            }
            for o in &offsets {
                data.extend_from_slice(&o.to_le_bytes());
            }
            data.extend_from_slice(&bytes_blob);
            let v = pack_valid_bitmap(valid);
            ("Utf8", data, v)
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
    //
    //    Phase 7 Task 34: Utf8 keys need a server-side preprocessing pass.
    //    The Python walker hands us a packed `[n_rows u32 le | offsets
    //    (n+1)×i32 le | string bytes]` payload; we parse it into Option<&str>
    //    rows, build (dict, codes) via `build_dict_nullable`, then transmute
    //    the Vec<u32> codes to a Vec<u8> that the KeyColumn borrows. Both
    //    the codes bytes AND the dict must outlive the KeyColumn, so we hold
    //    them in `utf8_owned_data` here next to `key_byte_refs`.
    let mut key_byte_refs: Vec<(&[u8], &[u8])> = Vec::with_capacity(parsed.keys.len());
    // Index `parsed.keys` → (codes_bytes, dict). None for non-Utf8 keys.
    let mut utf8_owned_data: Vec<Option<(Vec<u8>, Vec<String>)>> =
        Vec::with_capacity(parsed.keys.len());
    for k in &parsed.keys {
        let (_, data_py, valid_py) = by_name.get(&k.name).ok_or_else(|| {
            PyKeyError::new_err(format!(
                "polars_metal: key column {:?} not found in upstream columns",
                k.name
            ))
        })?;
        let data_bytes: &[u8] = data_py.as_bytes();
        let valid_bytes: &[u8] = valid_py.as_bytes();

        if k.dtype == MetalDtype::Utf8 {
            // Parse the packed wire payload.
            if data_bytes.len() < 4 {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Utf8 column {:?} wire payload too short ({} B; \
                     need >= 4 for header)",
                    k.name,
                    data_bytes.len()
                )));
            }
            let header_n =
                u32::from_le_bytes([data_bytes[0], data_bytes[1], data_bytes[2], data_bytes[3]])
                    as usize;
            if header_n != n_rows {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Utf8 column {:?} wire n_rows={header_n} \
                     disagrees with column n_rows={n_rows}",
                    k.name
                )));
            }
            let offsets_start = 4usize;
            let offsets_end = offsets_start + (n_rows + 1) * 4;
            if data_bytes.len() < offsets_end {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Utf8 column {:?} wire payload truncated in offsets \
                     ({} B; need >= {})",
                    k.name,
                    data_bytes.len(),
                    offsets_end
                )));
            }
            let mut offsets: Vec<i32> = Vec::with_capacity(n_rows + 1);
            for i in 0..=n_rows {
                let off = offsets_start + i * 4;
                offsets.push(i32::from_le_bytes([
                    data_bytes[off],
                    data_bytes[off + 1],
                    data_bytes[off + 2],
                    data_bytes[off + 3],
                ]));
            }
            let string_bytes = &data_bytes[offsets_end..];

            // Minimum validity bitmap length.
            let min_valid = (n_rows + 7) / 8;
            if valid_bytes.len() < min_valid {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Utf8 column {:?} validity buffer is {} B, \
                     need at least {} B for {} rows",
                    k.name,
                    valid_bytes.len(),
                    min_valid,
                    n_rows
                )));
            }

            // Build Option<&str> per row, honoring validity. Null rows get
            // `None` and skip the offset slice entirely (Arrow's null row
            // typically has offset[i] == offset[i+1], but we don't depend on
            // that — we simply ignore offsets for null rows).
            let mut strings_opt: Vec<Option<&str>> = Vec::with_capacity(n_rows);
            for r in 0..n_rows {
                let bit_set = (valid_bytes[r >> 3] >> (r & 7)) & 1 == 1;
                if !bit_set {
                    strings_opt.push(None);
                    continue;
                }
                let s_off = offsets[r];
                let e_off = offsets[r + 1];
                if s_off < 0 || e_off < s_off {
                    return Err(PyValueError::new_err(format!(
                        "polars_metal: Utf8 column {:?} row {r} has invalid offsets \
                         start={s_off} end={e_off}",
                        k.name
                    )));
                }
                let s_idx = s_off as usize;
                let e_idx = e_off as usize;
                if e_idx > string_bytes.len() {
                    return Err(PyValueError::new_err(format!(
                        "polars_metal: Utf8 column {:?} row {r} end offset {e_idx} \
                         exceeds string buffer length {}",
                        k.name,
                        string_bytes.len()
                    )));
                }
                let slice = &string_bytes[s_idx..e_idx];
                let s = std::str::from_utf8(slice).map_err(|e| {
                    PyValueError::new_err(format!(
                        "polars_metal: Utf8 column {:?} row {r} is not valid UTF-8: {e}",
                        k.name
                    ))
                })?;
                strings_opt.push(Some(s));
            }

            let (dict, codes, _valid_again) =
                polars_metal_buffer::dict::build_dict_nullable(&strings_opt);

            // Transmute Vec<u32> codes → Vec<u8> bytes for the wire format
            // that `encode_keys` expects. We can't use a from_raw_parts
            // reinterpret here because the KeyColumn borrows the slice; the
            // Vec<u32> itself must live in `utf8_owned_data` and we want a
            // byte slice over it. Convert via to_le_bytes to keep little-
            // endian semantics consistent across host endianness (M-series
            // is little-endian, but be explicit).
            let mut codes_bytes: Vec<u8> = Vec::with_capacity(codes.len() * 4);
            for c in &codes {
                codes_bytes.extend_from_slice(&c.to_le_bytes());
            }

            utf8_owned_data.push(Some((codes_bytes, dict)));
            // Push placeholder byte-refs; we'll override in the second pass.
            key_byte_refs.push((data_bytes, valid_bytes));
        } else {
            utf8_owned_data.push(None);
            key_byte_refs.push((data_bytes, valid_bytes));
        }
    }
    let key_cols: Vec<KeyColumn<'_>> = parsed
        .keys
        .iter()
        .zip(key_byte_refs.iter())
        .zip(utf8_owned_data.iter())
        .map(|((k, (data, valid)), utf8_opt)| {
            // For Utf8 keys we point `data` at the owned codes bytes and
            // attach the dict. For all other dtypes we keep the original
            // Python-side bytes and a None dict.
            let (data_slice, dict): (&[u8], Option<Vec<String>>) = match utf8_opt {
                Some((codes_bytes, dict)) => (codes_bytes.as_slice(), Some(dict.clone())),
                None => (*data, None),
            };
            Ok(KeyColumn {
                name: k.name.clone(),
                dtype: metal_dtype_to_key_dtype(k.dtype).map_err(|e| {
                    pyo3::exceptions::PyNotImplementedError::new_err(format!("polars_metal: {e}"))
                })?,
                data: data_slice,
                valid,
                n_rows,
                dict,
            })
        })
        .collect::<PyResult<Vec<KeyColumn<'_>>>>()?;

    // 4. Build the routing-input view: each agg's value-column byte/dtype
    //    triple, keyed by column name. The fused path consumes a HashMap of
    //    ValueColumns; the M2 per-agg path consumes (AggRequest, ValueColumn)
    //    pairs. We build the byte-references first; typed slices materialize
    //    after the routing decision so we can specialize both paths
    //    correctly.
    //
    // `name_byte_refs` covers EVERY column referenced by any Simple's
    // `input_col` OR by any Expression's `referenced_columns()`. This is a
    // superset of the M2 byte_refs because Expression-shape aggs can name
    // columns no Simple agg touches.
    let mut name_byte_refs: HashMap<String, (String, &[u8], &[u8])> = HashMap::new();
    for agg in &parsed.aggs {
        let referenced: Vec<String> = match agg {
            ParsedAgg::Length { .. } => Vec::new(),
            ParsedAgg::Simple { input_col, .. } => vec![input_col.clone()],
            ParsedAgg::Expression { expr, .. } => expr.referenced_columns(),
        };
        for col_name in referenced {
            if name_byte_refs.contains_key(&col_name) {
                continue;
            }
            let (dtype_tag, data_py, valid_py) = by_name.get(&col_name).ok_or_else(|| {
                PyKeyError::new_err(format!(
                    "polars_metal: agg input column {col_name:?} not found in upstream columns"
                ))
            })?;
            name_byte_refs.insert(
                col_name,
                (dtype_tag.clone(), data_py.as_bytes(), valid_py.as_bytes()),
            );
        }
    }

    // 5. Acquire device + queue.
    let device = MetalDevice::system_default()
        .map_err(|e| crate::engine_err(crate::EngineError::Buffer(e)))?;
    let mut queue = CommandQueue::new(&device)
        .map_err(|e| crate::engine_err(crate::EngineError::Other(format!("command queue: {e}"))))?;

    // 6. Routing: fused single-kernel vs M2 per-agg.
    //
    // The fused kernel caps `n_groups` at 16 (per-thread register array
    // size in `aggregate_fused::emitter::MAX_GROUPS`). The router can't
    // know n_groups ahead of time (it's a runtime build output), so when
    // Fused is selected and the dispatch returns NgroupsExceedsFusedCap
    // we transparently retry on the per-agg path. Expression aggs can't
    // go through per-agg, so they surface as a hard error.
    let initial_choice = decide_groupby_dispatch(&parsed.aggs, &name_byte_refs);
    let has_expression = parsed
        .aggs
        .iter()
        .any(|a| matches!(a, ParsedAgg::Expression { .. }));

    // First, attempt the chosen path; on NgroupsExceedsFusedCap, fall back.
    let mut fused_attempt: Option<
        Result<
            polars_metal_kernels::groupby::GroupByResult,
            polars_metal_kernels::groupby::FusedDispatchError,
        >,
    > = None;
    if matches!(initial_choice, GroupByDispatchChoice::Fused) {
        // Build kernel-layer specs.
        let kernel_specs: Vec<KAggSpec> = parsed
            .aggs
            .iter()
            .map(|pa| match pa {
                ParsedAgg::Simple {
                    input_col,
                    op,
                    output_alias,
                } => KAggSpec::Simple {
                    input_col: input_col.clone(),
                    op: convert_agg_op(*op),
                    output_alias: output_alias.clone(),
                },
                ParsedAgg::Expression {
                    expr,
                    op,
                    output_alias,
                } => KAggSpec::Expression {
                    expr: convert_agg_expr(expr),
                    op: convert_agg_op(*op),
                    output_alias: output_alias.clone(),
                },
                ParsedAgg::Length { output_alias } => KAggSpec::Length {
                    output_alias: output_alias.clone(),
                },
            })
            .collect();

        // Materialize each referenced column as a typed ValueColumn.
        let mut value_columns: HashMap<String, ValueColumn<'_>> = HashMap::new();
        for (name, (dt_tag, data, valid)) in name_byte_refs.iter() {
            let vcol = build_value_column(dt_tag, data, valid, n_rows).map_err(|e| {
                PyValueError::new_err(format!(
                    "polars_metal: fused groupby — value column {name:?}: {e}"
                ))
            })?;
            value_columns.insert(name.clone(), vcol);
        }

        let cache = get_or_init_fused_cache(&device);
        fused_attempt = Some(dispatch_groupby_fused(
            &device,
            &mut queue,
            cache,
            &key_cols,
            &kernel_specs,
            &value_columns,
            n_rows,
        ));
    }

    // Decide which path's output to use:
    //   - Fused attempt succeeded → use it directly.
    //   - Fused attempt rejected with NgroupsExceedsFusedCap and we can
    //     fall back to per-agg (query has no Expression aggs) → run per-agg.
    //   - Fused attempt failed irrecoverably → surface the error.
    //   - We never attempted fused (initial choice was PerAgg) → run per-agg.
    let early_result: Option<polars_metal_kernels::groupby::GroupByResult> = match fused_attempt {
        Some(Ok(r)) => Some(r),
        Some(Err(polars_metal_kernels::groupby::FusedDispatchError::NgroupsExceedsFusedCap {
            ..
        })) if !has_expression => None,
        Some(Err(e)) => {
            return Err(PyValueError::new_err(format!(
                "polars_metal: fused groupby dispatch: {e}"
            )));
        }
        None => None,
    };

    let result = if let Some(r) = early_result {
        r
    } else {
        // M2's per-agg path. Len uses a zero-length I64 placeholder; Simple
        // looks up its single input column from `name_byte_refs`.
        let empty_data: &[u8] = &[];
        let empty_valid: &[u8] = &[];
        // SAFETY: &[] cast to &[i64] is a zero-length slice — no pointer
        // arithmetic occurs. The slice is valid and the pointer is
        // non-null (a valid empty slice). This is the established
        // pattern for zero-length typed slices in this codebase.
        let empty_i64: &[i64] =
            unsafe { std::slice::from_raw_parts(empty_data.as_ptr() as *const i64, 0) };

        let mut agg_specs: Vec<(AggRequest, ValueColumn<'_>)> =
            Vec::with_capacity(parsed.aggs.len());
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
                ParsedAgg::Simple { input_col, op, .. } => {
                    let (dtype_tag, data, valid) =
                        name_byte_refs.get(input_col).ok_or_else(|| {
                            PyValueError::new_err(
                                "polars_metal: internal error: missing byte ref for Simple agg",
                            )
                        })?;
                    let (kind, vcol) =
                        build_agg_kind_and_vcol(*op, dtype_tag, data, valid, n_rows)?;
                    agg_specs.push((
                        AggRequest {
                            kind,
                            input_col_idx: i,
                        },
                        vcol,
                    ));
                }
                ParsedAgg::Expression { .. } => {
                    // Expression specs should never route here (the
                    // router decides Fused above). Defensive guard.
                    return Err(PyValueError::new_err(
                            "polars_metal: AggSpec::Expression routed to per-agg path; this is a routing bug",
                        ));
                }
            }
        }

        polars_metal_kernels::groupby::dispatch_groupby(
            &device, &mut queue, &key_cols, &agg_specs, n_rows,
        )
        .map_err(groupby_err)?
    };

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

// ── M4 Phase 5 Task 21+22: execute_fused_expr ───────────────────────────────
//
// PyO3 entry point that takes a PyFusionScope plus a list of input column
// byte buffers (each F32-typed) and returns the output as a PyBytes byte
// buffer. The Python wrapper in `_udf.py` converts Polars Series ↔ bytes
// since we don't carry a `pyo3-polars` dep.
//
// Deviation from the plan: the plan called for a `PyMetalPlanNode::FusedExprGraph`
// wrapper sitting on top of the Rust `MetalPlanNode` enum. We don't have a
// PyO3 wrapper for that enum - the Python side talks to Rust via wire-plan
// dicts. We expose the executor directly; the walker stashes the
// PyFusionScope as a side-channel on the binding and the UDF dispatch
// (Task 23) invokes this entry point.

/// Execute a fused MLX subgraph with zero-copy, dtype-polymorphic I/O.
///
/// `inputs` is a list of `(ptr, n_elements, dtype_tag)` triples, one per scope
/// input, where `ptr` is the address of a live, C-contiguous buffer (a numpy
/// array's `__array_interface__` data pointer) holding `n_elements` of the
/// `MlxDtype` named by `dtype_tag`. `out` is `(ptr, capacity_elements,
/// dtype_tag)` for a writable array to receive the single subgraph output;
/// `dtype_tag` is the analyzer's statically-inferred output dtype, so Python
/// pre-allocates the right-width array. Returns the number of elements written
/// (1 for a literal-only / scalar output, else the column length).
///
/// The buffer protocol isn't available under the abi3 limited API, so we pass
/// raw pointers from Python rather than `PyBuffer`. The caller MUST keep every
/// input array and the output array alive for the duration of this (fully
/// synchronous) call; `_udf._dispatch_hstack_fused` does so by holding the
/// arrays in locals across the call.
///
/// After eval, the Rust side asserts the eval'd dtype equals the declared
/// output tag (analyzer mis-inference guard) before any width-aware write — a
/// hard error rather than silently corrupting the caller's bytes.
#[pyfunction]
pub fn execute_fused_expr(
    scope: &crate::fusion::py::PyFusionScope,
    inputs: Vec<(usize, usize, u32)>,
    out: (usize, usize, u32),
) -> PyResult<usize> {
    use polars_metal_mlx_sys::array::MlxDtype;
    use std::sync::Arc;
    let device = MetalDevice::system_default().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "polars_metal: metal device unavailable: {e}"
        ))
    })?;

    // Stage each input buffer into a MetalBuffer at its native width. The
    // byte-level borrow is zero-copy when the pointer is page-aligned
    // (numpy-origin / large allocations), else a single copy — handled inside
    // `from_borrowed_bytes`. The caller keeps the source arrays alive for the
    // whole call, so the borrowed memory outlives every MetalBuffer and the
    // synchronous eval.
    let metal_buffers: Vec<Arc<polars_metal_buffer::MetalBuffer>> = inputs
        .iter()
        .map(|&(ptr, n, tag)| {
            let dtype = MlxDtype::from_tag(tag).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "polars_metal: bad input dtype tag: {e:?}"
                ))
            })?;
            // SAFETY: `ptr` addresses `n` live, contiguous elements of `dtype`
            // (caller contract) for the lifetime of this call; the byte length
            // is `n * element_size`. Page-aligned pointers take bytesNoCopy,
            // others copy. See `from_borrowed_bytes`.
            let buf = unsafe {
                polars_metal_buffer::MetalBuffer::from_borrowed_bytes(
                    &device,
                    ptr as *const u8,
                    n * dtype.element_size(),
                )
            }
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "polars_metal: input buffer staging failed: {e}"
                ))
            })?;
            Ok(Arc::new(buf))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let subgraph = crate::fusion::subgraph::MlxSubgraph::from_fusion_scope_buffers(
        &scope.inner,
        &metal_buffers,
    )
    .map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("polars_metal: subgraph build: {e}"))
    })?;

    // Output-zero-copy: eval writes directly into the caller's `out` array,
    // interpreting it at the analyzer-declared dtype.
    let (out_ptr, out_cap, out_tag) = out;
    let out_dtype = MlxDtype::from_tag(out_tag).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "polars_metal: bad output dtype tag: {e:?}"
        ))
    })?;
    // SAFETY: `out_ptr` addresses `out_cap` writable, contiguous elements of
    // `out_dtype` (caller contract), kept alive for the whole call.
    // `eval_into_typed` blocks on `mlx_array_eval` before copying (no in-flight
    // GPU writes on readback), validates the dtype matches, and bounds-checks
    // against `out_cap`.
    subgraph
        .eval_into_typed(out_ptr, out_cap, out_dtype)
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: subgraph eval: {e}"))
        })
}

// ── M5 Task 3: execute_rolling ───────────────────────────────────────────────
//
// PyO3 entry point that dispatches the Metal rolling-statistics kernels
// (sum / mean / var / std) over a caller-supplied F32 column. The caller
// passes raw pointer + length tuples (from numpy `ctypes.data` + `size`)
// so the buffer protocol isn't needed. The implementation stages both
// buffers via `MetalBuffer::from_borrowed_f32`:
//
//   - Page-aligned pointers (the common case for numpy arrays) → zero-copy
//     `newBufferWithBytesNoCopy`; GPU writes go directly to host memory and
//     are visible after `wait_until_complete`.
//   - Unaligned pointers → `newBufferWithBytes` (copies in). For the output
//     buffer this means the kernel writes to a GPU-private copy that is NOT
//     reflected back to the caller's slice. We detect this case and use the
//     allocate-and-copy-back fallback so correctness is always preserved.
//
// The Python caller must keep both arrays alive for the duration of this
// synchronous call (trivially satisfied for numpy locals).

/// PyO3 entry point exposed as `polars_metal._native.execute_rolling`.
///
/// # Arguments
/// * `inp` — `(ptr, n_elements)`: address and element count of a live,
///   C-contiguous F32 input array (e.g. `x.ctypes.data, x.size`).
/// * `out` — `(ptr, n_elements)`: address and element count of a writable
///   C-contiguous F32 output array of the same length. Overwritten in place.
/// * `w` — window size (1 ≤ w ≤ 4096).
/// * `op` — operation selector: `0` = sum, `1` = mean, `2` = var, `3` = std.
/// * `ddof` — degrees-of-freedom correction (only used for op 2/3; default 1).
///
/// The first `w-1` output elements are zero-filled (structural nulls). The
/// caller is responsible for setting the Arrow validity bitmap accordingly
/// before presenting the result to Polars.
#[pyfunction]
#[pyo3(signature = (inp, out, w, op, ddof=1))]
pub fn execute_rolling(
    inp: (usize, usize),
    out: (usize, usize),
    w: u32,
    op: u32,
    ddof: u32,
) -> PyResult<()> {
    use polars_metal_buffer::is_ptr_page_aligned;
    use polars_metal_kernels::rolling::{
        dispatch_rolling_sum_f32_buf, dispatch_rolling_var_f32_buf,
    };

    let device = MetalDevice::system_default().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "polars_metal: metal device unavailable: {e}"
        ))
    })?;

    let (in_ptr, in_n) = inp;
    let (out_ptr, out_n) = out;

    if in_n != out_n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "polars_metal: rolling input/output length mismatch",
        ));
    }

    // Fix 1: n=0 is a no-op — staging a 0-byte MTLBuffer would error.
    if in_n == 0 {
        return Ok(());
    }

    // Fix 2: guard against silent truncation on pathological inputs.
    let n = u32::try_from(in_n).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(
            "polars_metal: rolling column exceeds u32::MAX rows",
        )
    })?;

    // Stage input buffer. Zero-copy when page-aligned (numpy arrays are),
    // single copy otherwise. The source array is kept alive by the caller
    // for the duration of this synchronous call.
    //
    // SAFETY: `in_ptr` addresses `in_n` live, contiguous f32 values for the
    // whole call. Page-aligned pointers use bytesNoCopy (read-only from the
    // kernel's perspective); others are copied in. `f32` has no invalid bit
    // patterns.
    let inb = unsafe {
        polars_metal_buffer::MetalBuffer::from_borrowed_f32(&device, in_ptr as *const f32, in_n)
    }
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "polars_metal: rolling input staging: {e}"
        ))
    })?;

    // Output staging: two paths.
    //
    // Zero-copy path (page-aligned `out`): `newBufferWithBytesNoCopy` yields a
    // Shared-storage MTLBuffer that aliases the numpy allocation. The kernel
    // writes through the MTLBuffer and `wait_until_complete` (inside the
    // dispatcher) ensures the writes are visible on the host before we return.
    //
    // Copy-back path (unaligned `out`): `newBufferWithBytes` copies the
    // (zero) initial bytes into a GPU-private allocation. The kernel writes
    // to that private buffer; we must read it back via `as_slice()` and copy
    // into the caller's slice before returning.
    let out_ptr_is_aligned = is_ptr_page_aligned(out_ptr);

    // SAFETY: `out_ptr` addresses `out_n` writable, contiguous f32 values
    // for the whole call. Page-aligned: bytesNoCopy shares host memory
    // (writable). Unaligned: bytes are copied in; the MTLBuffer is later
    // read back.
    let outb = unsafe {
        polars_metal_buffer::MetalBuffer::from_borrowed_f32(&device, out_ptr as *const f32, out_n)
    }
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "polars_metal: rolling output staging: {e}"
        ))
    })?;

    // Dispatch the kernel into `outb`. The dispatcher calls
    // `wait_until_complete` internally before returning.
    let res = match op {
        0 => dispatch_rolling_sum_f32_buf(&device, &inb, &outb, n, w, false),
        1 => dispatch_rolling_sum_f32_buf(&device, &inb, &outb, n, w, true),
        2 => dispatch_rolling_var_f32_buf(&device, &inb, &outb, n, w, ddof, false),
        3 => dispatch_rolling_var_f32_buf(&device, &inb, &outb, n, w, ddof, true),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "polars_metal: unknown rolling op {other}"
            )))
        }
    };
    res.map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: rolling dispatch: {e}"))
    })?;

    // Copy-back path: if `out` was not page-aligned, `outb` holds a
    // GPU-private copy of the kernel results. Read it back into the caller's
    // slice now that the GPU is idle.
    if !out_ptr_is_aligned {
        let out_bytes = outb.as_slice();
        // SAFETY: `out_ptr` is valid for `out_n * 4` bytes (caller contract).
        // We just dispatched into `outb` and the GPU is idle; the result bytes
        // are stable. `f32` has no invalid bit patterns.
        let dst: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_n) };
        // SAFETY: `out_ptr` is valid for `out_n * 4` bytes (caller contract).
        // Metal `StorageModeShared` allocations are page-aligned, satisfying
        // f32's 4-byte alignment requirement; `out_n` f32 occupy exactly
        // `out_bytes.len()` bytes.
        let src_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(out_bytes.as_ptr() as *const f32, out_n) };
        dst.copy_from_slice(src_f32);
    }

    Ok(())
}

// ── M6 A4: execute_dtw ───────────────────────────────────────────────────────

/// PyO3 entry exposed as `polars_metal._native.execute_dtw`.
///
/// * `queries` — `(ptr, n_pairs*seq_len)`: pair-major F32 query matrix.
/// * `reference` — `(ptr, seq_len)`: the broadcast reference sequence.
/// * `out` — `(ptr, n_pairs)`: writable F32 output (overwritten in place).
/// * `n_pairs`, `seq_len` — dimensions.
/// * `window` — Sakoe-Chiba radius; negative => unconstrained full DTW.
///
/// Staging mirrors `execute_rolling`: `from_borrowed_f32` (zero-copy when
/// page-aligned; copy-back for an unaligned output). The caller keeps all
/// arrays alive for the synchronous call.
#[pyfunction]
#[pyo3(signature = (queries, reference, out, n_pairs, seq_len, window))]
pub fn execute_dtw(
    queries: (usize, usize),
    reference: (usize, usize),
    out: (usize, usize),
    n_pairs: usize,
    seq_len: usize,
    window: i32,
) -> PyResult<()> {
    use polars_metal_buffer::is_ptr_page_aligned;
    use polars_metal_kernels::dtw::dispatch_dtw_buf;

    let device = MetalDevice::system_default().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "polars_metal: metal device unavailable: {e}"
        ))
    })?;

    let (q_ptr, q_n) = queries;
    let (r_ptr, r_n) = reference;
    let (out_ptr, out_n) = out;

    if r_n != seq_len || out_n != n_pairs || q_n != n_pairs.saturating_mul(seq_len) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "polars_metal: dtw dimension mismatch",
        ));
    }
    if n_pairs == 0 || seq_len == 0 {
        return Ok(());
    }
    let n_u32 = u32::try_from(n_pairs).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("polars_metal: dtw n_pairs exceeds u32::MAX")
    })?;
    let l_u32 = u32::try_from(seq_len).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("polars_metal: dtw seq_len exceeds u32::MAX")
    })?;

    // SAFETY: each ptr addresses its stated count of live, contiguous f32 for
    // the whole call; page-aligned pointers use bytesNoCopy, others copy in.
    let qb = unsafe {
        polars_metal_buffer::MetalBuffer::from_borrowed_f32(&device, q_ptr as *const f32, q_n)
    }
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: dtw query staging: {e}"))
    })?;
    let rb = unsafe {
        polars_metal_buffer::MetalBuffer::from_borrowed_f32(&device, r_ptr as *const f32, r_n)
    }
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: dtw ref staging: {e}"))
    })?;
    let out_aligned = is_ptr_page_aligned(out_ptr);
    // SAFETY: out_ptr addresses out_n writable contiguous f32 for the call.
    let ob = unsafe {
        polars_metal_buffer::MetalBuffer::from_borrowed_f32(&device, out_ptr as *const f32, out_n)
    }
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: dtw out staging: {e}"))
    })?;

    dispatch_dtw_buf(&device, &qb, &rb, &ob, n_u32, l_u32, window).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: dtw dispatch: {e}"))
    })?;

    if !out_aligned {
        let out_bytes = ob.as_slice();
        // SAFETY: out_ptr valid for out_n*4 bytes (caller contract); GPU idle.
        let dst: &mut [f32] = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_n) };
        // SAFETY: Shared allocations are page-aligned (≥4-byte); out_n f32
        // occupy exactly out_bytes.len() bytes; f32 has no invalid patterns.
        let src: &[f32] =
            unsafe { std::slice::from_raw_parts(out_bytes.as_ptr() as *const f32, out_n) };
        dst.copy_from_slice(src);
    }
    Ok(())
}

// ── M6 B3: execute_dt ────────────────────────────────────────────────────────
//
// PyO3 entry point dispatching the gregorian civil-from-days kernel over a
// caller-supplied Int32 days-since-1970 column. Mirrors `execute_rolling`:
// raw (ptr, n) tuples, `from_borrowed_i32` staging (zero-copy when page-
// aligned, copy-back fallback for an unaligned output).

/// PyO3 entry point exposed as `polars_metal._native.execute_dt`.
///
/// # Arguments
/// * `inp` — `(ptr, n)`: address + element count of a live, C-contiguous
///   Int32 days-since-1970 array.
/// * `out` — `(ptr, n)`: address + element count of a writable C-contiguous
///   Int32 array of the same length. Overwritten in place.
/// * `field` — `0` = year, `1` = month, `2` = day.
#[pyfunction]
#[pyo3(signature = (inp, out, field))]
pub fn execute_dt(inp: (usize, usize), out: (usize, usize), field: u32) -> PyResult<()> {
    use polars_metal_buffer::is_ptr_page_aligned;
    use polars_metal_kernels::dt::{dispatch_dt_field_buf, DtField};

    let device = MetalDevice::system_default().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "polars_metal: metal device unavailable: {e}"
        ))
    })?;

    let (in_ptr, in_n) = inp;
    let (out_ptr, out_n) = out;

    if in_n != out_n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "polars_metal: dt input/output length mismatch",
        ));
    }

    if in_n == 0 {
        return Ok(());
    }

    let n = u32::try_from(in_n).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("polars_metal: dt column exceeds u32::MAX rows")
    })?;

    let dt_field = match field {
        0 => DtField::Year,
        1 => DtField::Month,
        2 => DtField::Day,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "polars_metal: unknown dt field {other}"
            )))
        }
    };

    // Output staging: two paths (see execute_rolling for the rationale).
    let out_ptr_is_aligned = is_ptr_page_aligned(out_ptr);

    // SAFETY: `out_ptr` addresses `out_n` writable, contiguous i32 values
    // for the whole call. Page-aligned: bytesNoCopy shares host memory
    // (writable). Unaligned: bytes are copied in; the MTLBuffer is later
    // read back.
    let outb = unsafe {
        polars_metal_buffer::MetalBuffer::from_borrowed_i32(&device, out_ptr as *const i32, out_n)
    }
    .map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: dt output staging: {e}"))
    })?;

    // Input staging: two paths based on pointer alignment.
    //   Aligned (Datetime at large N): numpy mmap buffer is page-aligned →
    //   zero-copy `bytesNoCopy` path via `from_borrowed_i32`, saving ~1ms at
    //   10M rows.
    //   Unaligned (Date / small N): Arrow 64-byte-aligned only → stage
    //   through the reusable pool (one memcpy into a reused Shared buffer,
    //   B3b). The pool avoids the per-call newBufferWithBytes allocation that
    //   dominated the old path (~5x faster at scale).
    // dispatch is called inside each arm to avoid juggling guard lifetimes.
    let in_aligned = is_ptr_page_aligned(in_ptr);
    if in_aligned {
        // SAFETY: `in_ptr` addresses `in_n` live, contiguous i32 values for
        // the whole synchronous call; page-aligned → `bytesNoCopy` shares
        // host memory (read-only here). `from_borrowed_i32` borrows without
        // copying.
        let inb = unsafe {
            polars_metal_buffer::MetalBuffer::from_borrowed_i32(&device, in_ptr as *const i32, in_n)
        }
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "polars_metal: dt input staging: {e}"
            ))
        })?;
        dispatch_dt_field_buf(&device, &inb, &outb, n, dt_field).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: dt dispatch: {e}"))
        })?;
    } else {
        // Recover from a poisoned lock rather than panic (the buffer is
        // scratch; no invariant is corrupted by a prior panic mid-stage).
        let pool = DT_STAGING.get_or_init(|| Mutex::new(polars_metal_buffer::StagingPool::new()));
        let mut staging = pool.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        // SAFETY: `in_ptr` addresses `in_n` live, contiguous i32 values for
        // the whole synchronous call; reinterpreting as `in_n * 4` bytes is
        // sound (`i32` has no invalid bit patterns) and the slice is only
        // read (memcpy source) before this function returns. `in_n * 4`
        // cannot overflow: the `u32::try_from(in_n)` check above bounds
        // `in_n <= u32::MAX`, so `in_n * 4 <= 4 * (2^32 - 1) < usize::MAX`
        // on 64-bit targets.
        let in_bytes: &[u8] = unsafe { std::slice::from_raw_parts(in_ptr as *const u8, in_n * 4) };
        let inb = staging.stage(&device, in_bytes).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "polars_metal: dt input staging: {e}"
            ))
        })?;
        dispatch_dt_field_buf(&device, inb, &outb, n, dt_field).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: dt dispatch: {e}"))
        })?;
    }

    // Copy-back path: if `out` was not page-aligned, `outb` holds a
    // GPU-private copy of the kernel results. Read it back via `as_slice()`
    // and cast to i32, mirroring execute_rolling's f32 read-back.
    if !out_ptr_is_aligned {
        let out_bytes = outb.as_slice();
        // SAFETY: `out_ptr` is valid for `out_n * 4` bytes (caller contract).
        // We just dispatched into `outb` and the GPU is idle; the result bytes
        // are stable. `i32` has no invalid bit patterns.
        let dst: &mut [i32] = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut i32, out_n) };
        // SAFETY: Metal `StorageModeShared` allocations are page-aligned,
        // satisfying i32's 4-byte alignment requirement; `out_n` i32 occupy
        // exactly `out_bytes.len()` bytes.
        let src_i32: &[i32] =
            unsafe { std::slice::from_raw_parts(out_bytes.as_ptr() as *const i32, out_n) };
        dst.copy_from_slice(src_i32);
    }

    Ok(())
}
