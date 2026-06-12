//! Filter-compaction PyO3 entry point: given a precomputed bit-packed boolean
//! predicate and per-column Arrow buffers, runs the three-pass compaction
//! pipeline (predicate eval + MLX cumsum + scatter) per surviving column.

use crate::plan::MetalDtype;
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::pipeline::{
    compact_bool, compact_f64, compact_i64, compute_keep_and_prefix,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyTuple};

use super::predicate::parse_dtype;

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
