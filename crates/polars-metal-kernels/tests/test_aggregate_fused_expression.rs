#![allow(clippy::expect_used)]
//! Tests for Task 13: MSL emitter handles `AggSpec::Expression`.
//!
//! These tests build kernel-layer mirror types directly (the emitter
//! consumes mirror types, not IR types) and verify that:
//! 1. Every column referenced by an expression has its load and validity
//!    bit emitted exactly once (shared with any other agg referencing
//!    the same column).
//! 2. The inline expression evaluates to a `float` in MSL — binary ops
//!    and literals appear in the emitted source.
//! 3. The emitted source compiles AND the resulting MTLComputePipelineState
//!    builds (PSO creation is where Metal validates atomic-op support on
//!    Apple Silicon).
//!
//! Helpers are duplicated from `test_aggregate_fused_emitter.rs` rather
//! than refactored into a shared module — keeping Task 13 scoped.

use std::collections::BTreeMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLLibrary;
use polars_metal_kernels::aggregate_fused::emitter::emit_msl;
use polars_metal_kernels::aggregate_fused::signature::{
    AggExpr, AggOp, AggSignature, AggSpec, BinaryOp, MetalDtype,
};

// ---------- helpers --------------------------------------------------------

fn col_dtypes(pairs: &[(&str, MetalDtype)]) -> BTreeMap<String, MetalDtype> {
    pairs.iter().map(|(n, d)| ((*n).to_string(), *d)).collect()
}

fn col(name: &str) -> AggExpr {
    AggExpr::Column(name.into())
}

fn bin(op: BinaryOp, lhs: AggExpr, rhs: AggExpr) -> AggExpr {
    AggExpr::Binary {
        op,
        lhs: Box::new(lhs),
        rhs: Box::new(rhs),
    }
}

fn expr_spec(expr: AggExpr, op: AggOp, alias: &str) -> AggSpec {
    AggSpec::Expression {
        expr,
        op,
        output_alias: alias.into(),
    }
}

fn try_compile_msl_to_library(
    device: &polars_metal_buffer::MetalDevice,
    src: &str,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, String> {
    use objc2_foundation::NSString;
    use objc2_metal::{MTLCompileOptions, MTLDevice as _};

    let ns_src = NSString::from_str(src);
    let opts = MTLCompileOptions::new();
    device
        .raw()
        .newLibraryWithSource_options_error(&ns_src, Some(&opts))
        .map_err(|e| format!("{}\n---SRC---\n{src}", e.localizedDescription()))
}

/// Compile MSL *and* build the MTLComputePipelineState. PSO creation is
/// where Metal validates atomic-op support on Apple Silicon; source-only
/// compile is insufficient (the original Task 12 bug compiled OK but
/// failed at PSO creation).
fn build_pso_for_emitted_source(src: &str) {
    use objc2_foundation::NSString;
    use objc2_metal::MTLDevice as _;

    let device = polars_metal_buffer::MetalDevice::system_default()
        .expect("Metal-capable hardware required for this test");
    let lib = try_compile_msl_to_library(&device, src).expect("MSL compile");
    let func_name = NSString::from_str("aggregate_fused");
    let func = lib
        .newFunctionWithName(&func_name)
        .expect("entry `aggregate_fused` exists in emitted library");
    device
        .raw()
        .newComputePipelineStateWithFunction_error(&func)
        .expect("PSO creation must succeed for the emitted Expression kernel");
}

// ---------- tests ----------------------------------------------------------

/// `Sum(a * b)` over F32. Both column loads must appear (the per-column
/// shared-load loop emits both); the multiplication must appear between
/// two value references; and the PSO must build.
#[test]
fn sum_a_mul_b_emits_one_kernel_with_both_loads() {
    let specs = vec![expr_spec(
        bin(BinaryOp::Mul, col("a"), col("b")),
        AggOp::Sum,
        "sum_ab",
    )];
    let dts = col_dtypes(&[("a", MetalDtype::F32), ("b", MetalDtype::F32)]);
    let sig = AggSignature::from_specs(&specs, &dts).expect("signature builds");
    let src = emit_msl(&sig, &specs);

    // Both column loads (one per slot) appear.
    assert!(src.contains("value_0[gid]"), "missing value_0 load:\n{src}");
    assert!(src.contains("value_1[gid]"), "missing value_1 load:\n{src}");

    // The multiplication between the two value references.
    // The emitter casts each Column to `(float)val_<i>` and joins with `*`.
    assert!(
        src.contains("(float)val_0 * (float)val_1")
            || src.contains("((float)val_0 * (float)val_1)"),
        "missing `val_0 * val_1` expression form:\n{src}"
    );

    // PSO must build.
    build_pso_for_emitted_source(&src);
}

/// `Sum(a * (1.0 - b))` — the canonical Q1 `disc_price` shape. The
/// literal `1.0` must appear (as a Metal float literal, suffix `f`),
/// the subtraction must appear, and the PSO must build.
#[test]
fn sum_a_mul_one_minus_b_emits_literal_subtraction() {
    let inner = bin(BinaryOp::Sub, AggExpr::LiteralF64(1.0), col("b"));
    let outer = bin(BinaryOp::Mul, col("a"), inner);
    let specs = vec![expr_spec(outer, AggOp::Sum, "disc_price")];
    let dts = col_dtypes(&[("a", MetalDtype::F32), ("b", MetalDtype::F32)]);
    let sig = AggSignature::from_specs(&specs, &dts).expect("signature builds");
    let src = emit_msl(&sig, &specs);

    // Float-literal must carry the Metal `f` suffix to bind as a float
    // (otherwise it would compile as a double, which the toolchain can't
    // emit code for in compute kernels on this platform).
    assert!(
        src.contains("1f") || src.contains("1.0f") || src.contains("1.000"),
        "missing float literal `1.0f` or similar:\n{src}"
    );

    // The subtraction must appear in the expression body.
    assert!(
        src.contains(" - "),
        "missing subtraction operator in expression body:\n{src}"
    );

    build_pso_for_emitted_source(&src);
}

/// `Sum(a * a)` — same column referenced twice. The shared-load
/// invariant (one load per column slot) must hold: only ONE `value_0[gid]`
/// load appears, and the expression references `val_0` twice.
#[test]
fn expression_columns_dedupe() {
    let specs = vec![expr_spec(
        bin(BinaryOp::Mul, col("a"), col("a")),
        AggOp::Sum,
        "sum_aa",
    )];
    let dts = col_dtypes(&[("a", MetalDtype::F32)]);
    let sig = AggSignature::from_specs(&specs, &dts).expect("signature builds");
    let src = emit_msl(&sig, &specs);

    // Exactly one shared load of value_0[gid] (Task 12 invariant).
    let loads = src.matches("value_0[gid]").count();
    assert_eq!(
        loads, 1,
        "expected exactly 1 shared load of value_0[gid], got {loads}:\n{src}"
    );

    // The expression references val_0 at least twice (lhs and rhs of the
    // multiplication).
    let val0_refs = src.matches("(float)val_0").count();
    assert!(
        val0_refs >= 2,
        "expected expression to reference val_0 at least twice, got {val0_refs}:\n{src}"
    );

    build_pso_for_emitted_source(&src);
}
