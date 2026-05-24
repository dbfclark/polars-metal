#![allow(clippy::expect_used)]
//! Tests for the MSL template emitter for fused multi-aggregation kernels.
//!
//! The emitter consumes kernel-layer mirror types (`AggSpec` from
//! `polars_metal_kernels::aggregate_fused::signature`) plus the
//! pre-built `AggSignature`. The IR-side `AggSpec` is converted by
//! the caller; tests here build the mirror types directly to keep the
//! source small and focused on emitter behavior.

use std::collections::BTreeMap;

use polars_metal_kernels::aggregate_fused::emitter::emit_msl;
use polars_metal_kernels::aggregate_fused::signature::{
    AggOp, AggSignature, AggSpec, MetalDtype,
};

// ---------- helpers --------------------------------------------------------

fn simple(col: &str, op: AggOp, alias: &str) -> AggSpec {
    AggSpec::Simple {
        input_col: col.into(),
        op,
        output_alias: alias.into(),
    }
}

fn col_dtypes(pairs: &[(&str, MetalDtype)]) -> BTreeMap<String, MetalDtype> {
    pairs.iter().map(|(n, d)| ((*n).to_string(), *d)).collect()
}

/// Compile-check the emitted MSL via `MTLDevice::newLibraryWithSource:options:error:`.
///
/// Returns `Ok(())` if the device accepts the source; `Err` carries the
/// localized diagnostic concatenated with the source for easy debugging.
fn try_compile_msl(src: &str) -> Result<(), String> {
    use objc2_foundation::NSString;
    use objc2_metal::{MTLCompileOptions, MTLDevice as _};

    let device = polars_metal_buffer::MetalDevice::system_default()
        .map_err(|e| format!("acquire device: {e:?}"))?;
    let ns_src = NSString::from_str(src);
    let opts = MTLCompileOptions::new();
    device
        .raw()
        .newLibraryWithSource_options_error(&ns_src, Some(&opts))
        .map(|_| ())
        .map_err(|e| format!("{}\n---SRC---\n{src}", e.localizedDescription()))
}

// ---------- tests ----------------------------------------------------------

#[test]
fn emitted_kernel_compiles() {
    let specs = vec![
        simple("a", AggOp::Sum, "sum_a"),
        simple("a", AggOp::Mean, "mean_a"),
    ];
    let dts = col_dtypes(&[("a", MetalDtype::F32)]);
    let sig = AggSignature::from_specs(&specs, &dts).expect("signature builds");
    let src = emit_msl(&sig, &specs);

    assert!(
        src.contains("kernel void aggregate_fused"),
        "missing entry point:\n{src}"
    );
    assert!(
        src.contains("row_to_group [[buffer("),
        "missing row_to_group binding:\n{src}"
    );

    // 1 value column + (sum out, count out for mean) + row_to_group +
    // n_rows + validity_0 = at least 4 buffer bindings.
    let buffer_count = src.matches("[[buffer(").count();
    assert!(
        buffer_count >= 4,
        "expected >= 4 buffer bindings, got {buffer_count}:\n{src}"
    );

    try_compile_msl(&src).expect("emitted MSL must compile on the device");
}

#[test]
fn fused_sum_only_emits_one_atomic_per_row() {
    let specs = vec![simple("v", AggOp::Sum, "sum_v")];
    let dts = col_dtypes(&[("v", MetalDtype::F32)]);
    let sig = AggSignature::from_specs(&specs, &dts).expect("signature builds");
    let src = emit_msl(&sig, &specs);
    let atomics = src.matches("atomic_fetch_add_explicit").count();
    assert_eq!(
        atomics, 1,
        "expected exactly 1 atomic_fetch_add_explicit, got {atomics}:\n{src}"
    );
}

#[test]
fn fused_sum_mean_count_shares_load_once() {
    let specs = vec![
        simple("v", AggOp::Sum, "sum_v"),
        simple("v", AggOp::Mean, "mean_v"),
        simple("v", AggOp::Count, "count_v"),
    ];
    let dts = col_dtypes(&[("v", MetalDtype::F32)]);
    let sig = AggSignature::from_specs(&specs, &dts).expect("signature builds");
    let src = emit_msl(&sig, &specs);
    let loads = src.matches("value_0[gid]").count();
    assert_eq!(
        loads, 1,
        "expected exactly 1 shared load of value_0[gid], got {loads}:\n{src}"
    );
}
