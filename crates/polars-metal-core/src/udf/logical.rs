//! Boolean predicate combinators: Polars 3-valued AND/OR over two bit-packed
//! nullable Boolean predicates, returning a fresh `(data, valid)` byte-pair.
//! The Python walker calls these when it emits a `BinaryExpr(And|Or, ...)`
//! whose operands both resolve to Bool.

use polars_metal_kernels::logical::{dispatch_bool_and, dispatch_bool_or};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::common::{check_bitpacked_buffer, cmp_out_min_bytes, new_device_and_queue};

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
