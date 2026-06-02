use polars_metal_native::fusion::density::*;
use polars_metal_native::fusion::scope::{FusionScope, InputDtype};
use polars_metal_native::fusion::supported_ops::OpId;

#[test]
fn small_workload_routes_cpu() {
    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::F32);
    scope.push_op(OpId::Sqrt, vec![a]);
    let decision = density_routes_gpu(&scope, 1_000);
    assert_eq!(decision, RouteDecision::Cpu(CpuReason::BelowRowsThreshold));
}

#[test]
fn medium_workload_routes_cpu_if_too_few_flops() {
    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::F32);
    scope.push_op(OpId::Sqrt, vec![a]); // 4 FLOPs/row
                                        // 1M rows x 4 = 4e6, below 5e7 threshold
    let decision = density_routes_gpu(&scope, 1_000_000);
    assert_eq!(decision, RouteDecision::Cpu(CpuReason::BelowFlopsThreshold));
}

#[test]
fn black_scholes_at_10m_routes_gpu() {
    let mut scope = FusionScope::new();
    let s = scope.add_input("s", InputDtype::F32);
    let k = scope.add_input("k", InputDtype::F32);
    let t = scope.add_input("t", InputDtype::F32);
    let sk = scope.push_op(OpId::Div, vec![s, k]);
    let log_sk = scope.push_op(OpId::Log, vec![sk]);
    let sqrt_t = scope.push_op(OpId::Sqrt, vec![t]);
    let p1 = scope.push_op(OpId::Add, vec![log_sk, t]);
    let d1 = scope.push_op(OpId::Div, vec![p1, sqrt_t]);
    let tanh_d1 = scope.push_op(OpId::Tanh, vec![d1]);
    scope.mark_output(tanh_d1);
    // 4 + 10 + 4 + 1 + 4 + 10 = 33 FLOPs/row x 10M = 3.3e8, above 5e7
    let decision = density_routes_gpu(&scope, 10_000_000);
    assert_eq!(decision, RouteDecision::Gpu);
}

#[test]
fn empty_scope_routes_cpu() {
    let scope = FusionScope::new();
    assert!(matches!(
        density_routes_gpu(&scope, 10_000_000),
        RouteDecision::Cpu(CpuReason::EmptyScope)
    ));
}

#[test]
fn just_above_flops_threshold_routes_gpu() {
    // Build a scope with 6 FLOPs/row; 10M x 6 = 6e7, above 5e7.
    let mut scope = FusionScope::new();
    let a = scope.add_input("a", InputDtype::F32);
    let b = scope.add_input("b", InputDtype::F32);
    let s = scope.push_op(OpId::Sqrt, vec![a]); // 4
    scope.push_op(OpId::Square, vec![s]); // 1
    scope.push_op(OpId::Add, vec![a, b]); // 1
    assert_eq!(density_routes_gpu(&scope, 10_000_000), RouteDecision::Gpu);
}
