//! Comparison-kernel PyO3 entry points: column-vs-scalar and column-vs-column
//! i64/f64 comparisons that return a bit-packed bool predicate `(data, valid)`.
//! The Python walker calls these when it emits a `Compare` node; the predicate
//! bytes feed straight into [`super::compact::execute_filter_compact`].

use polars_metal_kernels::cmp::{
    dispatch_cmp_f64, dispatch_cmp_f64_scalar, dispatch_cmp_i64, dispatch_cmp_i64_scalar,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::common::{check_numeric_buffers, cmp_out_min_bytes, new_device_and_queue};
use super::predicate::parse_compare_op;

/// PyO3 entry point exposed as `polars_metal._native.cmp_i64_col_scalar`.
///
/// Evaluates a single column-vs-scalar i64 comparison and returns the
/// bit-packed bool predicate `(data, valid)`. The Python UDF calls this
/// when the walker emits a `Compare { lhs: Column(I64), rhs: LiteralI64 }`
/// (and similarly for the other three column-vs-leaf combinations); the
/// resulting predicate bytes feed straight into
/// [`super::compact::execute_filter_compact`].
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
