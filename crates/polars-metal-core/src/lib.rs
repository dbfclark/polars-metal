// crates/polars-metal-core/src/lib.rs
//
// PyO3 0.22's `#[pyfunction]` macro generates `IntoPy` wrapper code that
// clippy flags as `useless_conversion` when the return type is `String` or
// `Vec<T>`. The lint is a false positive caused by the macro expansion; allow
// it for this file only.
#![allow(clippy::useless_conversion)]

mod arena;
mod error;
pub mod plan;

pub use arena::{BumpArena, ScratchArena, StubArena};
pub use error::EngineError;

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
fn engine_err(e: EngineError) -> PyErr {
    e.into()
}

#[pymodule]
#[pyo3(name = "_native")]
fn polars_metal_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version_string, m)?)?;
    m.add_function(wrap_pyfunction!(device_name, m)?)?;
    m.add_function(wrap_pyfunction!(add_f32, m)?)?;
    Ok(())
}
