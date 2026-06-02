//! Compute-density routing decision: route a FusionScope to GPU iff its
//! estimated FLOPs and row count both exceed thresholds (5e7 and 1e5 by
//! default). Below threshold, GPU dispatch overhead dominates the work.

use super::scope::FusionScope;

pub const MIN_FLOPS_THRESHOLD: u64 = 50_000_000; // 5e7
pub const MIN_ROWS_THRESHOLD: usize = 100_000; // 1e5

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuReason {
    BelowRowsThreshold,
    BelowFlopsThreshold,
    EmptyScope,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RouteDecision {
    Gpu,
    Cpu(CpuReason),
}

pub fn density_routes_gpu(scope: &FusionScope, n_rows: usize) -> RouteDecision {
    if scope.ops.is_empty() {
        return RouteDecision::Cpu(CpuReason::EmptyScope);
    }
    if n_rows < MIN_ROWS_THRESHOLD {
        return RouteDecision::Cpu(CpuReason::BelowRowsThreshold);
    }
    let flops = scope.est_flops_for(n_rows);
    if flops < MIN_FLOPS_THRESHOLD {
        return RouteDecision::Cpu(CpuReason::BelowFlopsThreshold);
    }
    RouteDecision::Gpu
}
