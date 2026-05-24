#![allow(clippy::expect_used)]
//! Tests for the MSL template emitter for fused multi-aggregation kernels.
//!
//! The emitter consumes kernel-layer mirror types (`AggSpec` from
//! `polars_metal_kernels::aggregate_fused::signature`) plus the
//! pre-built `AggSignature`. The IR-side `AggSpec` is converted by
//! the caller; tests here build the mirror types directly to keep the
//! source small and focused on emitter behavior.

use std::collections::BTreeMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLLibrary;
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

/// Compile-check the emitted MSL via `MTLDevice::newLibraryWithSource:options:error:`,
/// returning the resulting library so callers can drive further validation
/// (e.g. `MTLComputePipelineState` creation, which is where atomic-op
/// support is actually validated on Apple Silicon).
///
/// `Err` carries the localized diagnostic concatenated with the source for
/// easy debugging.
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

/// Convenience wrapper for tests that only want a source-compile check.
fn try_compile_msl(src: &str) -> Result<(), String> {
    let device = polars_metal_buffer::MetalDevice::system_default()
        .map_err(|e| format!("acquire device: {e:?}"))?;
    try_compile_msl_to_library(&device, src).map(|_| ())
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
    // For `[Sum F32]`: the kernel must perform exactly one atomic update
    // per row. F32 Sum uses a CAS loop on `atomic_uint` (one
    // `atomic_compare_exchange_weak_explicit` per row of input under no
    // contention; `atomic_fetch_add_explicit` is *not* valid for
    // `atomic_float` at MTLComputePipelineState creation on Apple
    // Silicon Metal 32023.883). Verify the kernel structure has exactly
    // one CAS loop and no fetch-add primitives slipped in.
    let specs = vec![simple("v", AggOp::Sum, "sum_v")];
    let dts = col_dtypes(&[("v", MetalDtype::F32)]);
    let sig = AggSignature::from_specs(&specs, &dts).expect("signature builds");
    let src = emit_msl(&sig, &specs);
    let cas = src.matches("atomic_compare_exchange_weak_explicit").count();
    let fetch_add = src.matches("atomic_fetch_add_explicit").count();
    assert_eq!(cas, 1, "expected 1 CAS loop, got {cas}:\n{src}");
    assert_eq!(
        fetch_add, 0,
        "expected 0 fetch_add (F32 Sum is CAS), got {fetch_add}:\n{src}"
    );
}

#[test]
fn emitted_kernel_creates_pipeline_state() {
    // Runtime-PSO sanity: compile MSL *and* build the
    // MTLComputePipelineState. PSO creation is where Metal validates
    // atomic-op support; `newLibraryWithSource` accepts source that the
    // pipeline-state stage will then reject. This test would have caught
    // the original Task 12 bug (atomic_fetch_add_explicit on
    // atomic_float compiles, fails PSO creation).
    use objc2_foundation::NSString;
    use objc2_metal::MTLDevice as _;

    let specs = vec![
        simple("v", AggOp::Sum, "sum_v"),
        simple("v", AggOp::Mean, "mean_v"),
        simple("v", AggOp::Count, "count_v"),
        simple("v", AggOp::Min, "min_v"),
        simple("v", AggOp::Max, "max_v"),
    ];
    let dts = col_dtypes(&[("v", MetalDtype::F32)]);
    let sig = AggSignature::from_specs(&specs, &dts).expect("signature builds");
    let src = emit_msl(&sig, &specs);

    let device = polars_metal_buffer::MetalDevice::system_default()
        .expect("Metal-capable hardware required for this test");
    let lib = try_compile_msl_to_library(&device, &src).expect("MSL compile");
    let func_name = NSString::from_str("aggregate_fused");
    let func = lib
        .newFunctionWithName(&func_name)
        .expect("entry `aggregate_fused` exists in emitted library");
    device
        .raw()
        .newComputePipelineStateWithFunction_error(&func)
        .expect(
            "PSO creation must succeed — atomic ops in emitted MSL must be runtime-supported",
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
