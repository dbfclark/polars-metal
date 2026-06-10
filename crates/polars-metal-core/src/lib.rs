// crates/polars-metal-core/src/lib.rs
//
// PyO3 0.22's `#[pyfunction]` macro generates `IntoPy` wrapper code that
// clippy flags as `useless_conversion` when the return type is `String` or
// `Vec<T>`. The lint is a false positive caused by the macro expansion; allow
// it for this file only.
#![allow(clippy::useless_conversion)]

mod arena;
mod error;
mod fft;
pub mod fusion;
pub mod plan;
pub mod router;
mod router_udf;
mod udf;
mod vector_search;

pub use arena::{BumpArena, ScratchArena, StubArena};
pub use error::EngineError;
pub use udf::{
    bool_and_dispatch, bool_or_dispatch, cmp_f64_col_col, cmp_f64_col_scalar, cmp_i64_col_col,
    cmp_i64_col_scalar, execute_filter_compact, execute_groupby, execute_plan, parse_groupby_plan,
    warmup_common_fused_signatures, GroupByParseError, ParsedAgg, ParsedGroupByPlan, ParsedKey,
};

use polars_metal_buffer::MetalDevice;
use pyo3::prelude::*;

#[pyfunction]
fn version_string() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn device_name() -> PyResult<String> {
    let device = MetalDevice::system_default().map_err(|e| engine_err(EngineError::Buffer(e)))?;
    Ok(device.name())
}

#[pyfunction]
fn add_f32(a: Vec<f32>, b: Vec<f32>) -> PyResult<Vec<f32>> {
    let out = polars_metal_mlx_sys::add_f32(&a, &b).map_err(|e| engine_err(EngineError::Ffi(e)))?;
    Ok(out)
}

/// Convert an [`EngineError`] into a [`PyErr`] at the PyO3 boundary.
///
/// This shim avoids `useless_conversion` lint warnings that arise when using
/// `.map_err(Into::into)` in functions whose return type rustc cannot yet
/// fully infer at the `map_err` call site.
pub(crate) fn engine_err(e: EngineError) -> PyErr {
    e.into()
}

#[pymodule]
#[pyo3(name = "_native")]
fn polars_metal_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version_string, m)?)?;
    m.add_function(wrap_pyfunction!(device_name, m)?)?;
    m.add_function(wrap_pyfunction!(add_f32, m)?)?;
    m.add_function(wrap_pyfunction!(udf::execute_plan, m)?)?;
    m.add_function(wrap_pyfunction!(udf::execute_filter_compact, m)?)?;
    m.add_function(wrap_pyfunction!(udf::cmp_i64_col_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(udf::cmp_i64_col_col, m)?)?;
    m.add_function(wrap_pyfunction!(udf::cmp_f64_col_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(udf::cmp_f64_col_col, m)?)?;
    m.add_function(wrap_pyfunction!(udf::bool_and_dispatch, m)?)?;
    m.add_function(wrap_pyfunction!(udf::bool_or_dispatch, m)?)?;
    m.add_function(wrap_pyfunction!(router_udf::compute_lifting_plan_py, m)?)?;
    m.add_function(wrap_pyfunction!(udf::execute_groupby, m)?)?;
    m.add_function(wrap_pyfunction!(udf::warmup_common_fused_signatures, m)?)?;
    m.add_function(wrap_pyfunction!(udf::execute_fused_expr, m)?)?;
    m.add_function(wrap_pyfunction!(udf::execute_rolling, m)?)?;
    m.add_function(wrap_pyfunction!(udf::execute_dt, m)?)?;
    m.add_function(wrap_pyfunction!(vector_search::execute_vector_search, m)?)?;
    m.add_function(wrap_pyfunction!(fft::execute_fft, m)?)?;
    fusion::py::register(m)?;
    Ok(())
}
