//! Cost-model rules and threshold constants.
//!
//! Each rule is a pure function over the relevant inputs (op kind,
//! row count, input decision). Thresholds are exposed as `pub const`
//! so PR-level tuning touches a single line.
//!
//! Why CPU defaults for Filter
//! ---------------------------
//! M1's perf investigation (see `tests/bench/baseline.json` notes)
//! showed CPU winning at all measured row counts (1K..100M) for the
//! filter operator. Unified memory removes the copy-cost asymmetry
//! that GPUs exploit on discrete-memory systems, and the CPU
//! implementation in Polars is already highly tuned. Future PRs may
//! revisit (e.g. once filter fuses into a larger GPU subtree without
//! a CPU round-trip), but today the default is CpuLeave.
//!
//! Why GPU > 100K rows for GroupBy
//! -------------------------------
//! Atomic-CAS hash-table build and the per-aggregate kernel launches
//! have fixed launch overhead that dominates at small row counts;
//! crossover is empirically near 100K rows on M2 Ultra for low-
//! cardinality keys (Q1's shape). The constant is a starting point;
//! M2's per-kernel benchmarks (Phase 10) inform PR-level tuning.
//!
//! Composite-key width limits
//! --------------------------
//! GroupBy keys are packed into u128 by the encoder (T13). Each key gets
//! 1 bit for the null flag plus its data width. Max supported: 128 bits
//! total. Queries exceeding this fall back to CPU at plan time rather than
//! failing at dispatch.

use super::NodeDecision;
use crate::plan::{AggSpec, MetalDtype};

/// Smallest input row count at which the GroupBy kernel is expected to
/// beat the CPU implementation on M2 Ultra. Tuned by criterion benches
/// per spec § "Risks & open questions — Cost-model threshold tuning".
pub const GROUPBY_GPU_MIN_ROWS: usize = 100_000;

/// Decide for a Filter node. Always CpuLeave under M2's cost model.
pub fn decide_filter(_n_rows: usize) -> NodeDecision {
    NodeDecision::CpuLeave
}

/// Decide for a GroupBy node based on input row count.
pub fn decide_groupby(n_rows: usize) -> NodeDecision {
    if n_rows > GROUPBY_GPU_MIN_ROWS {
        NodeDecision::GpuLift
    } else {
        NodeDecision::CpuLeave
    }
}

/// Decide for a Project / SimpleProjection node. Inherits its input's
/// decision verbatim — projection is metadata-only on our side either
/// way (column re-selection happens after compaction or via the CPU
/// executor's projection).
pub fn decide_project(input: &NodeDecision) -> NodeDecision {
    match input {
        NodeDecision::Fallback(r) => NodeDecision::Fallback(r.clone()),
        NodeDecision::GpuLift => NodeDecision::GpuLift,
        NodeDecision::CpuLeave => NodeDecision::CpuLeave,
    }
}

/// Initial Scan decision (before affinity smoothing applies the parent
/// hint). Default CpuLeave; affinity may upgrade to GpuLift in Task 5.
pub fn decide_scan_initial() -> NodeDecision {
    NodeDecision::CpuLeave
}

/// Per-key width in bits, including the 1-bit null flag the encoder adds.
/// Must match `polars_metal_kernels::groupby::KeyDtype::data_bits` + 1.
fn key_width_bits(dtype: MetalDtype) -> usize {
    match dtype {
        MetalDtype::Bool => 1 + 1,
        MetalDtype::I64 | MetalDtype::F64 => 1 + 64,
    }
}

/// GroupBy decision including the plan-time composite-key width check.
/// Returns `Fallback` when keys would exceed the 128-bit encoder budget,
/// even if the row count would otherwise route to GPU.
pub fn decide_groupby_with_keys(
    n_rows: usize,
    keys: &[(String, MetalDtype)],
    _aggs: &[AggSpec],
) -> NodeDecision {
    let total_bits: usize = keys.iter().map(|(_, d)| key_width_bits(*d)).sum();
    if total_bits > 128 {
        return NodeDecision::Fallback(format!(
            "composite key total {total_bits} bits; M2 supports ≤ 128"
        ));
    }
    decide_groupby(n_rows)
}
