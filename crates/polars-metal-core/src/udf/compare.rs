//! Comparison-kernel PyO3 entry points: column-vs-scalar and column-vs-column
//! i64/f64 comparisons that return a bit-packed bool predicate `(data, valid)`.
//! The Python walker calls these when it emits a `Compare` node; the predicate
//! bytes feed straight into [`super::compact::execute_filter_compact`].
//!
//! The four entry points are ~95% identical (only the element type, the kernel
//! dispatch fn, and the scalar-vs-column rhs differ), so they are generated
//! from two macros. They stay four distinct `#[pyfunction]`s because the Python
//! side calls each by name. i64/f64 are both 8-byte, so the buffer width is
//! fixed at 8; neither has invalid bit patterns, so the reinterpret is sound.

use polars_metal_kernels::cmp::{
    dispatch_cmp_f64, dispatch_cmp_f64_scalar, dispatch_cmp_i64, dispatch_cmp_i64_scalar,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use super::common::{check_numeric_buffers, cmp_out_min_bytes, new_device_and_queue};
use super::predicate::parse_compare_op;

/// Generate a column-vs-scalar comparison `#[pyfunction]` for element type
/// `$ty` (8-byte, no invalid bit patterns) dispatching through `$dispatch`.
macro_rules! cmp_col_scalar_pyfn {
    ($name:ident, $ty:ty, $dispatch:path) => {
        #[doc = concat!("PyO3 entry point `polars_metal._native.", stringify!($name), "`: column-vs-scalar comparison.")]
        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<'py>(
            py: Python<'py>,
            lhs_data: &Bound<'py, PyBytes>,
            lhs_valid: &Bound<'py, PyBytes>,
            rhs: $ty,
            op: &str,
            n_rows: usize,
        ) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
            let op_enum = parse_compare_op(op)?;
            let lhs_data_bytes = lhs_data.as_bytes();
            let lhs_valid_bytes = lhs_valid.as_bytes();
            check_numeric_buffers(lhs_data_bytes, lhs_valid_bytes, n_rows, 8)?;

            // SAFETY: the element type has no invalid bit patterns;
            // `lhs_data_bytes` is at least `n_rows * 8` bytes (checked above)
            // and Arrow buffers are 64-byte aligned, so the reinterpret is
            // well-aligned.
            let lhs_slice: &[$ty] = unsafe {
                std::slice::from_raw_parts(lhs_data_bytes.as_ptr() as *const $ty, n_rows)
            };

            let (device, mut queue) = new_device_and_queue()?;
            let min_out = cmp_out_min_bytes(n_rows);
            let mut out_data = vec![0u8; min_out];
            let mut out_valid = vec![0u8; min_out];
            $dispatch(
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
                    "polars_metal: {} dispatch failed: {e}",
                    stringify!($name)
                ))
            })?;
            Ok((
                PyBytes::new_bound(py, &out_data),
                PyBytes::new_bound(py, &out_valid),
            ))
        }
    };
}

/// Generate a column-vs-column comparison `#[pyfunction]` for element type
/// `$ty` (8-byte, no invalid bit patterns) dispatching through `$dispatch`.
macro_rules! cmp_col_col_pyfn {
    ($name:ident, $ty:ty, $dispatch:path) => {
        #[doc = concat!("PyO3 entry point `polars_metal._native.", stringify!($name), "`: column-vs-column comparison.")]
        #[pyfunction]
        #[allow(clippy::too_many_arguments)]
        pub fn $name<'py>(
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

            // SAFETY: the element type has no invalid bit patterns; both slices
            // are at least `n_rows * 8` bytes (checked above) and 64-byte
            // aligned by Arrow.
            let lhs_slice: &[$ty] = unsafe {
                std::slice::from_raw_parts(lhs_data_bytes.as_ptr() as *const $ty, n_rows)
            };
            let rhs_slice: &[$ty] = unsafe {
                std::slice::from_raw_parts(rhs_data_bytes.as_ptr() as *const $ty, n_rows)
            };

            let (device, mut queue) = new_device_and_queue()?;
            let min_out = cmp_out_min_bytes(n_rows);
            let mut out_data = vec![0u8; min_out];
            let mut out_valid = vec![0u8; min_out];
            $dispatch(
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
                    "polars_metal: {} dispatch failed: {e}",
                    stringify!($name)
                ))
            })?;
            Ok((
                PyBytes::new_bound(py, &out_data),
                PyBytes::new_bound(py, &out_valid),
            ))
        }
    };
}

cmp_col_scalar_pyfn!(cmp_i64_col_scalar, i64, dispatch_cmp_i64_scalar);
cmp_col_scalar_pyfn!(cmp_f64_col_scalar, f64, dispatch_cmp_f64_scalar);
cmp_col_col_pyfn!(cmp_i64_col_col, i64, dispatch_cmp_i64);
cmp_col_col_pyfn!(cmp_f64_col_col, f64, dispatch_cmp_f64);
