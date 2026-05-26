//! Partitioned-hash build phase (capability A1).
//!
//! Algorithm — per spec § "Algorithm details / A1":
//!   1. Per-row partition_id = (hash(key) >> log2(TGSM_slots)) & (P-1)
//!   2. Scatter rows into partition lanes.
//!   3. Per partition (one threadgroup), build hash table in TGSM with
//!      open addressing + linear probe. Emit (row, local_group_id).
//!   4. CPU: exclusive scan over n_groups_per_partition; offset local
//!      group_ids to produce global row_to_group.
//!   5. CPU: derive first_row_per_group for result reconstruction.
//!
//! See `references/cudf/cpp/src/groupby/hash/groupby.cu` for the source
//! algorithm. Our adaptation: 32-bit atomics only (Apple Silicon
//! constraint); per-threadgroup hash tables (no global atomic-CAS).

pub mod gpu;
pub mod reference;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildOutput {
    /// Per-row group_id (global).
    pub row_to_group: Vec<u32>,
    /// For each group, the index of its first occurrence in the input.
    pub first_row_per_group: Vec<u32>,
    /// Total number of unique groups.
    pub n_groups: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum PartitionedBuildError {
    #[error(transparent)]
    Buffer(#[from] polars_metal_buffer::BufferError),
    #[error(transparent)]
    Shader(#[from] crate::shader_lib::ShaderError),
    #[error(transparent)]
    Dispatch(#[from] crate::command::DispatchError),
    #[error("input row count overflows u32")]
    RowOverflow,
    #[error("A1 TGSM overflow; fallback to A2")]
    Overflow,
}
