// crates/polars-metal-core/src/lib.rs
//
// PyO3 entry point. Exports the `polars_metal_native` extension module
// loaded by the Python package.

use pyo3::prelude::*;

/// Hello-world function called from the Python integration tests to
/// confirm the extension loaded.
#[pyfunction]
fn version_string() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn polars_metal_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version_string, m)?)?;
    Ok(())
}
