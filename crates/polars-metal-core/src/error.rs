// crates/polars-metal-core/src/error.rs
use polars_metal_buffer::BufferError;
use polars_metal_mlx_sys::FfiError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::PyErr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("buffer: {0}")]
    Buffer(#[from] BufferError),
    #[error("ffi: {0}")]
    Ffi(#[from] FfiError),
    #[error("engine: {0}")]
    Other(String),
}

impl From<EngineError> for PyErr {
    fn from(e: EngineError) -> Self {
        // Surface as `polars.exceptions.ComputeError` so the user sees an
        // exception type indistinguishable from a native Polars failure.
        // Fall back to PyRuntimeError if polars.exceptions can't be loaded
        // (extremely unlikely — our package depends on polars).
        let msg = format!("polars-metal: {e}");
        Python::with_gil(|py| {
            py.import_bound("polars.exceptions")
                .and_then(|m| m.getattr("ComputeError"))
                .and_then(|cls| cls.call1((msg.clone(),)))
                .map(|inst| PyErr::from_value_bound(inst))
                .unwrap_or_else(|_| PyRuntimeError::new_err(msg))
        })
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn buffer_error_converts_into_engine_error() {
        let be = BufferError::AllocationFailed { bytes: 1024 };
        let ee: EngineError = be.into();
        let msg = format!("{ee}");
        assert!(msg.contains("buffer"));
        assert!(msg.contains("1024"));
    }

    #[test]
    fn engine_error_to_pyerr_carries_prefix() {
        pyo3::prepare_freethreaded_python();
        let ee = EngineError::Other("kaboom".into());
        let pe: PyErr = ee.into();
        let msg = pe.to_string();
        assert!(msg.contains("polars-metal:"));
        assert!(msg.contains("kaboom"));
    }

    #[test]
    fn engine_error_to_pyerr_is_polars_compute_error_when_polars_available() {
        // Initialize Python once; pyo3 is already gil-locked by previous test.
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let polars_exc = match py.import_bound("polars.exceptions") {
                Ok(m) => m,
                Err(_) => return, // skip if polars not available in this env
            };
            let compute_error = polars_exc
                .getattr("ComputeError")
                .expect("polars.exceptions.ComputeError should exist");

            let ee = EngineError::Other("boom".into());
            let pe: PyErr = ee.into();
            let value = pe.value_bound(py);
            assert!(
                value
                    .is_instance(&compute_error)
                    .expect("is_instance check"),
                "expected PolarsError::ComputeError but got: {pe:?}"
            );
            assert!(pe.to_string().contains("polars-metal:"));
            assert!(pe.to_string().contains("boom"));
        });
    }
}
