//! Sort-then-segment-reduce groupby build (capability A2).
//!
//! High-cardinality fallback when capability A1 (partitioned hash build,
//! see `groupby_build_partitioned`) overflows the per-threadgroup TGSM
//! capacity. The algorithm:
//!
//!   1. Radix-sort the (key, original_row_index) pairs by key.
//!   2. Single segment-boundary scan over sorted keys: each new key
//!      starts a new group.
//!   3. Permute group_ids back to original row order via the saved indices.
//!
//! See `references/cudf/cpp/src/groupby/sort/` for cuDF's analogous
//! implementation. We share the `BuildOutput` struct with A1 so the
//! router (Phase 6) can dispatch either build interchangeably.

pub mod reference;
// `pub mod gpu;` lands in Task 25 alongside the first lane kernel.
