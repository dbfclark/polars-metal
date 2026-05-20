// crates/polars-metal-core/src/error.rs
use polars_metal_buffer::BufferError;
use polars_metal_mlx_sys::FfiError;
use pyo3::exceptions::PyRuntimeError;
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
        // Polars raises ComputeError as a runtime-typed exception on the
        // Python side. We mirror its surface — the user sees
        // "polars.exceptions.ComputeError: polars-metal: <message>" once
        // Polars catches and re-raises this.
        PyRuntimeError::new_err(format!("polars-metal: {e}"))
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
}
