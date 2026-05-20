// crates/polars-metal-mlx-sys/src/error.rs
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FfiError {
    #[error("MLX shape mismatch: lhs {lhs} rhs {rhs}")]
    ShapeMismatch { lhs: usize, rhs: usize },
    #[error("MLX runtime error: {0}")]
    Runtime(String),
}

impl From<cxx::Exception> for FfiError {
    fn from(e: cxx::Exception) -> Self {
        FfiError::Runtime(e.what().to_string())
    }
}
