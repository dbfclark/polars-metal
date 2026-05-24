#![allow(clippy::expect_used)]
//! Tests for `FusedLibraryCache`: compiles fused-agg MSL lazily and caches
//! the resulting compute-pipeline-state by signature hash.

use std::collections::BTreeMap;

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::aggregate_fused::cache::FusedLibraryCache;
use polars_metal_kernels::aggregate_fused::signature::{AggOp, AggSignature, AggSpec, MetalDtype};

// ---------- helpers --------------------------------------------------------

fn dtypes_for(pairs: &[(&str, MetalDtype)]) -> BTreeMap<String, MetalDtype> {
    pairs.iter().map(|(n, d)| ((*n).to_string(), *d)).collect()
}

// ---------- tests ----------------------------------------------------------

#[test]
fn cache_returns_same_pipeline_for_isomorphic_signatures() {
    // Same shape, different column names → cache hit on second call.
    let device = MetalDevice::system_default().expect("Metal-capable hardware required");
    let cache = FusedLibraryCache::new(device);

    // [Sum F32 over "a"] and [Sum F32 over "b"] canonicalize to the same
    // signature (column name is replaced by slot index).
    let specs1 = vec![AggSpec::Simple {
        input_col: "a".into(),
        op: AggOp::Sum,
        output_alias: "x".into(),
    }];
    let specs2 = vec![AggSpec::Simple {
        input_col: "b".into(),
        op: AggOp::Sum,
        output_alias: "x".into(),
    }];
    let sig1 = AggSignature::from_specs(&specs1, &dtypes_for(&[("a", MetalDtype::F32)]))
        .expect("signature 1 builds");
    let sig2 = AggSignature::from_specs(&specs2, &dtypes_for(&[("b", MetalDtype::F32)]))
        .expect("signature 2 builds");
    assert_eq!(sig1, sig2, "isomorphic signatures must compare equal");

    let p1 = cache.get_or_compile(&sig1, &specs1).expect("compile 1");
    let p2 = cache
        .get_or_compile(&sig2, &specs2)
        .expect("compile 2 (cache hit)");
    // Compare PSO identity by underlying Objective-C pointer.
    let p1_ptr: *const _ = &*p1;
    let p2_ptr: *const _ = &*p2;
    assert_eq!(
        p1_ptr, p2_ptr,
        "cache should reuse compiled PSO across isomorphic signatures"
    );
}

#[test]
fn cache_compiles_distinct_for_different_signatures() {
    let device = MetalDevice::system_default().expect("Metal-capable hardware required");
    let cache = FusedLibraryCache::new(device);

    let specs1 = vec![AggSpec::Simple {
        input_col: "v".into(),
        op: AggOp::Sum,
        output_alias: "x".into(),
    }];
    let specs2 = vec![AggSpec::Simple {
        input_col: "v".into(),
        op: AggOp::Mean,
        output_alias: "x".into(),
    }];
    let sig1 = AggSignature::from_specs(&specs1, &dtypes_for(&[("v", MetalDtype::F32)]))
        .expect("signature 1 builds");
    let sig2 = AggSignature::from_specs(&specs2, &dtypes_for(&[("v", MetalDtype::F32)]))
        .expect("signature 2 builds");
    assert_ne!(sig1, sig2, "Sum vs Mean must produce different signatures");

    let p1 = cache.get_or_compile(&sig1, &specs1).expect("compile 1");
    let p2 = cache.get_or_compile(&sig2, &specs2).expect("compile 2");
    let p1_ptr: *const _ = &*p1;
    let p2_ptr: *const _ = &*p2;
    assert_ne!(
        p1_ptr, p2_ptr,
        "different signatures must produce distinct PSOs"
    );
}

#[test]
fn warmup_pre_compiles_then_get_or_compile_is_a_hit() {
    let device = MetalDevice::system_default().expect("Metal-capable hardware required");
    let cache = FusedLibraryCache::new(device);

    let specs = vec![AggSpec::Simple {
        input_col: "v".into(),
        op: AggOp::Sum,
        output_alias: "x".into(),
    }];
    let sig = AggSignature::from_specs(&specs, &dtypes_for(&[("v", MetalDtype::F32)]))
        .expect("signature builds");
    cache.warmup(&[(sig.clone(), specs.clone())]);

    let p_warm = cache
        .get_or_compile(&sig, &specs)
        .expect("post-warmup get_or_compile");
    let p_again = cache
        .get_or_compile(&sig, &specs)
        .expect("second get_or_compile");
    let p_warm_ptr: *const _ = &*p_warm;
    let p_again_ptr: *const _ = &*p_again;
    assert_eq!(
        p_warm_ptr, p_again_ptr,
        "warmup-compiled PSO must be the same instance returned by get_or_compile"
    );
}
