#![allow(clippy::expect_used)]
//! Tests for `AggSignature` — the cache key shared by isomorphic fused
//! aggregation queries.
//!
//! The test source builds IR-side `AggSpec` values (the user-facing shape),
//! then converts them to the kernel-layer mirror types and passes those to
//! `AggSignature::from_specs`. This conversion lives in the test crate
//! because `polars-metal-kernels` cannot depend on `polars-metal-core`
//! directly (that crate already depends on the kernels crate; a regular
//! cycle is rejected by cargo). The dev-dep cycle works because dev-deps
//! are not propagated to downstream crates.

use std::collections::BTreeMap;

// `polars-metal-core` is published under the lib name `polars_metal_native`
// (see its `[lib] name = "polars_metal_native"`).
use polars_metal_kernels::aggregate_fused::signature::{
    AggExpr as KAggExpr, AggOp as KAggOp, AggSignature, AggSpec as KAggSpec, BinaryOp as KBinaryOp,
    MetalDtype as KMetalDtype,
};
use polars_metal_native::plan::{
    AggExpr as IrAggExpr, AggOp as IrAggOp, AggSpec as IrAggSpec, BinaryOp as IrBinaryOp,
    MetalDtype as IrMetalDtype,
};

// ---------- helpers --------------------------------------------------------

fn convert_dtype(d: IrMetalDtype) -> KMetalDtype {
    match d {
        IrMetalDtype::I64 => KMetalDtype::I64,
        IrMetalDtype::F64 => KMetalDtype::F64,
        IrMetalDtype::Bool => KMetalDtype::Bool,
        IrMetalDtype::I32 => KMetalDtype::I32,
        IrMetalDtype::F32 => KMetalDtype::F32,
        IrMetalDtype::I8 => KMetalDtype::I8,
        IrMetalDtype::I16 => KMetalDtype::I16,
        IrMetalDtype::U8 => KMetalDtype::U8,
        IrMetalDtype::U16 => KMetalDtype::U16,
        IrMetalDtype::U32 => KMetalDtype::U32,
    }
}

fn convert_op(o: IrAggOp) -> KAggOp {
    match o {
        IrAggOp::Sum => KAggOp::Sum,
        IrAggOp::Mean => KAggOp::Mean,
        IrAggOp::Count => KAggOp::Count,
        IrAggOp::Min => KAggOp::Min,
        IrAggOp::Max => KAggOp::Max,
        IrAggOp::Len => KAggOp::Len,
    }
}

fn convert_binop(o: IrBinaryOp) -> KBinaryOp {
    match o {
        IrBinaryOp::Add => KBinaryOp::Add,
        IrBinaryOp::Sub => KBinaryOp::Sub,
        IrBinaryOp::Mul => KBinaryOp::Mul,
        IrBinaryOp::Div => KBinaryOp::Div,
    }
}

fn convert_expr(e: &IrAggExpr) -> KAggExpr {
    match e {
        IrAggExpr::Column(name) => KAggExpr::Column(name.clone()),
        IrAggExpr::LiteralF64(v) => KAggExpr::LiteralF64(*v),
        IrAggExpr::LiteralI64(v) => KAggExpr::LiteralI64(*v),
        IrAggExpr::Binary { op, lhs, rhs } => KAggExpr::Binary {
            op: convert_binop(*op),
            lhs: Box::new(convert_expr(lhs)),
            rhs: Box::new(convert_expr(rhs)),
        },
    }
}

fn convert_spec(s: &IrAggSpec) -> KAggSpec {
    match s {
        IrAggSpec::Simple {
            input_col,
            op,
            output_alias,
        } => KAggSpec::Simple {
            input_col: input_col.clone(),
            op: convert_op(*op),
            output_alias: output_alias.clone(),
        },
        IrAggSpec::Expression {
            expr,
            op,
            output_alias,
        } => KAggSpec::Expression {
            expr: convert_expr(expr),
            op: convert_op(*op),
            output_alias: output_alias.clone(),
        },
        IrAggSpec::Length { output_alias } => KAggSpec::Length {
            output_alias: output_alias.clone(),
        },
    }
}

/// Build an IR `AggSpec::Simple` for a single-column agg.
fn simple(col: &str, op: IrAggOp, alias: &str) -> IrAggSpec {
    IrAggSpec::Simple {
        input_col: col.into(),
        op,
        output_alias: alias.into(),
    }
}

/// Convert a slice of IR specs and a dtype map into the kernel-layer
/// arguments expected by `AggSignature::from_specs`.
fn build(ir_specs: &[IrAggSpec], ir_dtypes: &[(&str, IrMetalDtype)]) -> AggSignature {
    let kspecs: Vec<KAggSpec> = ir_specs.iter().map(convert_spec).collect();
    let kdtypes: BTreeMap<String, KMetalDtype> = ir_dtypes
        .iter()
        .map(|(n, d)| ((*n).to_string(), convert_dtype(*d)))
        .collect();
    AggSignature::from_specs(&kspecs, &kdtypes).expect("from_specs ok")
}

// ---------- six required tests ---------------------------------------------

#[test]
fn signature_same_for_isomorphic_specs() {
    // Same shape, different aliases — aliases must NOT affect signature.
    let a = build(
        &[
            simple("v", IrAggOp::Sum, "sum_v"),
            simple("v", IrAggOp::Mean, "mean_v"),
        ],
        &[("v", IrMetalDtype::F64)],
    );
    let b = build(
        &[
            simple("v", IrAggOp::Sum, "anything_else"),
            simple("v", IrAggOp::Mean, "doesnt_matter"),
        ],
        &[("v", IrMetalDtype::F64)],
    );
    assert_eq!(a, b);
    assert_eq!(a.hash64(), b.hash64());
}

#[test]
fn signature_differs_when_op_set_differs() {
    let a = build(
        &[simple("v", IrAggOp::Sum, "s")],
        &[("v", IrMetalDtype::F64)],
    );
    let b = build(
        &[simple("v", IrAggOp::Mean, "m")],
        &[("v", IrMetalDtype::F64)],
    );
    assert_ne!(a, b);
}

#[test]
fn signature_differs_when_dtype_differs() {
    let a = build(
        &[simple("v", IrAggOp::Sum, "s")],
        &[("v", IrMetalDtype::F32)],
    );
    let b = build(
        &[simple("v", IrAggOp::Sum, "s")],
        &[("v", IrMetalDtype::F64)],
    );
    assert_ne!(a, b);
}

#[test]
fn signature_differs_when_column_count_differs() {
    let a = build(
        &[
            simple("a", IrAggOp::Sum, "s"),
            simple("b", IrAggOp::Sum, "t"),
        ],
        &[("a", IrMetalDtype::F64), ("b", IrMetalDtype::F64)],
    );
    let b = build(
        &[simple("a", IrAggOp::Sum, "s")],
        &[("a", IrMetalDtype::F64)],
    );
    assert_ne!(a, b);
    assert_eq!(a.column_count(), 2);
    assert_eq!(b.column_count(), 1);
    assert_eq!(a.agg_count(), 2);
    assert_eq!(b.agg_count(), 1);
}

#[test]
fn signature_collapses_aliases_but_not_column_distinction() {
    // Two aggs over the *same* column should produce a signature that
    // shares the load; two aggs over different columns must differ.
    let same_col = build(
        &[
            simple("a", IrAggOp::Sum, "s"),
            simple("a", IrAggOp::Mean, "m"),
        ],
        &[("a", IrMetalDtype::F64)],
    );
    let diff_col = build(
        &[
            simple("a", IrAggOp::Sum, "s"),
            simple("b", IrAggOp::Mean, "m"),
        ],
        &[("a", IrMetalDtype::F64), ("b", IrMetalDtype::F64)],
    );
    assert_ne!(same_col, diff_col);
    assert_eq!(same_col.column_count(), 1);
    assert_eq!(diff_col.column_count(), 2);
}

#[test]
fn signature_for_expression_includes_expr_shape() {
    // sum(a * b) — single Expression agg.
    let s_inline = build(
        &[IrAggSpec::Expression {
            expr: IrAggExpr::Binary {
                op: IrBinaryOp::Mul,
                lhs: Box::new(IrAggExpr::Column("a".into())),
                rhs: Box::new(IrAggExpr::Column("b".into())),
            },
            op: IrAggOp::Sum,
            output_alias: "sum_ab".into(),
        }],
        &[("a", IrMetalDtype::F64), ("b", IrMetalDtype::F64)],
    );
    // sum(a) — single Simple agg. "Feels similar" but is structurally
    // different and must produce a different signature.
    let s_simple = build(
        &[simple("a", IrAggOp::Sum, "s")],
        &[("a", IrMetalDtype::F64)],
    );
    assert_ne!(s_inline, s_simple);
}

#[test]
fn signature_collapses_different_column_names_same_shape() {
    // Headline canonicalization property: two specs with different
    // column names but identical shape and dtype must hash to the same
    // signature. This is the entire reason slot indices replace column
    // names in the canonical form.
    let a = build(
        &[simple("price", IrAggOp::Sum, "sum_price")],
        &[("price", IrMetalDtype::F64)],
    );
    let b = build(
        &[simple("revenue", IrAggOp::Sum, "sum_revenue")],
        &[("revenue", IrMetalDtype::F64)],
    );
    assert_eq!(a, b);
    assert_eq!(a.hash64(), b.hash64());
}

// ---------- accessor sanity ------------------------------------------------

#[test]
fn column_order_is_first_seen_order() {
    // Reference columns in order b, a — slot 0 must be b, slot 1 must be a.
    let sig = build(
        &[
            simple("b", IrAggOp::Sum, "s"),
            simple("a", IrAggOp::Sum, "t"),
        ],
        &[("a", IrMetalDtype::F64), ("b", IrMetalDtype::F64)],
    );
    assert_eq!(sig.column_order(), &["b".to_string(), "a".to_string()]);
}
