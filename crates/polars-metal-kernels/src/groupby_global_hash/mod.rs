//! Capability A3 spike — single-pass global-atomic GPU hash table.
//!
//! Targets the cardinality range where A1 overflows (>~16K groups) and
//! where A2 was outperformed by CPU (every tested cardinality). Apple
//! Silicon's unified memory + single-die architecture may admit a
//! global-atomic design that NVIDIA's cuDF avoided due to cross-die
//! contention. The Phase 5b spike (commit-pending) confirms or denies.
//!
//! See `shaders/groupby_global_hash.metal` for the kernel.

pub mod gpu;

#[derive(Debug, thiserror::Error)]
pub enum GlobalHashError {
    #[error(transparent)]
    Buffer(#[from] polars_metal_buffer::BufferError),
    #[error(transparent)]
    Shader(#[from] crate::shader_lib::ShaderError),
    #[error(transparent)]
    Dispatch(#[from] crate::command::DispatchError),
    #[error("input row count overflows u32")]
    RowOverflow,
    /// At least one row's probe chain exceeded `max_probe`. Caller should
    /// either retry with a larger table or fall back to CPU.
    #[error("A3 hash-table overflow; fallback to CPU")]
    Overflow,
}
