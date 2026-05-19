// crates/polars-metal-buffer/src/error.rs
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BufferError {
    #[error("alignment violation: pointer {ptr:#x} length {len} required alignment {required}")]
    Alignment {
        ptr: usize,
        len: usize,
        required: usize,
    },
    #[error("validity bitmap shape mismatch: bitmap len {bitmap_len} for {row_count} rows")]
    ValidityShape { bitmap_len: usize, row_count: usize },
    #[error("MTLBuffer allocation failed (requested {bytes} bytes)")]
    AllocationFailed { bytes: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_alignment_error() {
        let err = BufferError::Alignment {
            ptr: 0xdeadbeef,
            len: 1024,
            required: 4096,
        };
        let s = format!("{err}");
        assert!(s.contains("0xdeadbeef"));
        assert!(s.contains("1024"));
        assert!(s.contains("4096"));
    }
}
