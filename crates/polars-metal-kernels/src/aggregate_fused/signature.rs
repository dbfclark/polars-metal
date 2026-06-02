//! Hashable cache key for fused-aggregation kernel sources.
//!
//! Two query plans with isomorphic agg shapes (same per-column op set,
//! same dtypes, same expression structure — aliases ignored) share one
//! compiled MSL library. The key is canonicalized: column names are
//! replaced with indices in first-seen order to maximize cache hits.
//!
//! ## Why kernel-layer mirror types?
//!
//! `AggSignature` lives in `polars-metal-kernels`. The IR-side
//! `AggSpec` / `AggExpr` / `BinaryOp` / `AggOp` / `MetalDtype` live in
//! `polars-metal-core`, which already depends on `polars-metal-kernels`.
//! A regular cycle is rejected by cargo, so this module defines its own
//! mirror types (`AggSpec`, `AggExpr`, ...). Callers in `polars-metal-core`
//! convert IR → kernel-layer types at the dispatch boundary (see
//! reconciliation note #5 in the M3 plan; the `FusedAggDescriptor`
//! pattern referenced there is exactly the mirror enum here).
//!
//! The mirror types intentionally share field names and variants with
//! the IR. Adding a variant to the IR must be mirrored here too —
//! otherwise the conversion at the dispatch boundary won't compile.
//!
//! ## Per-column dtype derivation for `Expression` aggs
//!
//! `AggSpec::Expression` does not carry an `output_dtype`. The first
//! column referenced by `AggExpr::referenced_columns()` (left-to-right
//! walk) determines the per-slot dtype recorded in the signature. This
//! is a simplifying assumption that matches Q1-shape workloads where
//! every column in a single inline expression shares the same input
//! dtype (e.g. `sum(price * (1 - discount))` over `F64` columns).
//! Mixed-dtype inline expressions are out of scope for M3.

use std::collections::hash_map::DefaultHasher;
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use thiserror::Error;

// ---------- kernel-layer mirrors of the core IR ----------------------------

/// Mirror of `polars_metal_core::plan::MetalDtype`.
// MIRROR of polars_metal_core::plan::MetalDtype — keep in sync
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetalDtype {
    I64,
    F64,
    Bool,
    I32,
    F32,
    I8,
    I16,
    U8,
    U16,
    U32,
    // M3 Phase 7: dictionary-encoded Utf8. Present in this mirror only to
    // stay in sync with the IR enum; Utf8 columns are never agg value cols,
    // so the fused signature will never observe this variant in practice.
    Utf8,
}

/// Mirror of `polars_metal_core::plan::AggOp`.
// MIRROR of polars_metal_core::plan::AggOp — keep in sync
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggOp {
    Sum,
    Mean,
    Count,
    Min,
    Max,
    Len,
}

/// Mirror of `polars_metal_core::plan::BinaryOp`.
// MIRROR of polars_metal_core::plan::BinaryOp — keep in sync
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Mirror of `polars_metal_core::plan::AggExpr`.
// MIRROR of polars_metal_core::plan::AggExpr — keep in sync
#[derive(Debug, Clone, PartialEq)]
pub enum AggExpr {
    Column(String),
    LiteralF64(f64),
    LiteralI64(i64),
    Binary {
        op: BinaryOp,
        lhs: Box<AggExpr>,
        rhs: Box<AggExpr>,
    },
}

impl AggExpr {
    /// All columns referenced anywhere in the tree, in left-to-right
    /// order, deduplicated. Mirrors the IR-side `referenced_columns`.
    pub fn referenced_columns(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        self.walk(&mut |e| {
            if let AggExpr::Column(name) = e {
                if !out.iter().any(|n| n == name) {
                    out.push(name.clone());
                }
            }
        });
        out
    }

    fn walk<F: FnMut(&AggExpr)>(&self, f: &mut F) {
        f(self);
        if let AggExpr::Binary { lhs, rhs, .. } = self {
            lhs.walk(f);
            rhs.walk(f);
        }
    }
}

/// Mirror of `polars_metal_core::plan::AggSpec`. Identical variant and
/// field names — IR callers convert by walking and rebuilding.
// MIRROR of polars_metal_core::plan::AggSpec — keep in sync
#[derive(Debug, Clone, PartialEq)]
pub enum AggSpec {
    Simple {
        input_col: String,
        op: AggOp,
        output_alias: String,
    },
    Expression {
        expr: AggExpr,
        op: AggOp,
        output_alias: String,
    },
    Length {
        output_alias: String,
    },
}

// ---------- error ----------------------------------------------------------

#[derive(Debug, Error, PartialEq, Eq)]
pub enum AggSignatureError {
    #[error("column `{0}` referenced by an agg spec is not present in col_dtypes")]
    UnknownColumn(String),
}

// ---------- canonical (post-canonicalization) shapes -----------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CanonicalAgg {
    Simple {
        col_slot: u16,
        op: AggOp,
        dtype: MetalDtype,
    },
    Expression {
        expr: CanonicalExpr,
        op: AggOp,
        dtype: MetalDtype,
    },
    Length,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CanonicalExpr {
    Column(u16),
    /// `f64` does not satisfy `Eq`; store its bit pattern instead.
    LiteralF64Bits(u64),
    LiteralI64(i64),
    Binary {
        op: BinaryOp,
        lhs: Box<CanonicalExpr>,
        rhs: Box<CanonicalExpr>,
    },
}

// ---------- public type ----------------------------------------------------

/// Canonical signature of a fused-agg query. Identical shape (same per-
/// column op set, same dtypes, same expression structure — aliases
/// ignored) ⇒ identical `AggSignature`, even across different column
/// names.
///
/// Two columns with the same name appearing in different specs share one
/// slot. Slots are assigned in first-seen-walk order over `specs`; for
/// `AggSpec::Expression`, the expression tree is walked left-to-right
/// (via `AggExpr::referenced_columns()`).
///
/// Aliases (`output_alias`) are dropped entirely.
///
/// `PartialEq` / `Eq` / `Hash` deliberately exclude `column_order` — the
/// canonical identity is `(column_dtypes, aggs)`. Two signatures whose
/// columns differ only in name (same shape, same per-slot dtypes, same
/// agg structure) must compare equal and hash equal; otherwise the cache
/// key fragments per query and the whole point of slot canonicalization
/// is lost.
#[derive(Debug, Clone)]
pub struct AggSignature {
    /// Original column names in first-seen order. Slot `i` ↔
    /// `column_order[i]`. The emitter (Task 12) needs the names to keep
    /// alias info available when generating MSL parameter symbols.
    /// NOT included in `PartialEq` / `Hash` — see struct doc comment.
    column_order: Vec<String>,
    /// Per-column-slot dtype, in first-seen order.
    column_dtypes: Vec<MetalDtype>,
    /// Per-agg shape, with column references rewritten as slot indices.
    aggs: Vec<CanonicalAgg>,
}

impl PartialEq for AggSignature {
    fn eq(&self, other: &Self) -> bool {
        // Compare canonical fields only; column_order is metadata for the
        // emitter and must not affect identity.
        self.column_dtypes == other.column_dtypes && self.aggs == other.aggs
    }
}

impl Eq for AggSignature {}

impl Hash for AggSignature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Mirror `PartialEq`: exclude `column_order`.
        self.column_dtypes.hash(state);
        self.aggs.hash(state);
    }
}

impl AggSignature {
    /// Build a signature from kernel-layer specs.
    ///
    /// `col_dtypes` is the caller-supplied map from column name to its
    /// runtime `MetalDtype`. Every column name referenced by any spec
    /// (either as a `Simple::input_col` or transitively inside an
    /// `Expression::expr`) must be present in this map. `Length` aggs
    /// do not require any column lookup.
    ///
    /// For `AggSpec::Expression`, the slot dtype recorded for each
    /// referenced column is the value carried in `col_dtypes`. (Within
    /// a single Q1-shape expression every column shares the same dtype;
    /// mixed-dtype inline expressions are out of scope for M3.)
    pub fn from_specs(
        specs: &[AggSpec],
        col_dtypes: &BTreeMap<String, MetalDtype>,
    ) -> Result<Self, AggSignatureError> {
        let mut slots: BTreeMap<String, u16> = BTreeMap::new();
        let mut order: Vec<String> = Vec::new();
        let mut dtypes: Vec<MetalDtype> = Vec::new();
        let mut canonical_aggs: Vec<CanonicalAgg> = Vec::with_capacity(specs.len());

        for spec in specs {
            let canonical = match spec {
                AggSpec::Simple { input_col, op, .. } => {
                    let dt = lookup_dtype(input_col, col_dtypes)?;
                    let slot = intern(input_col, dt, &mut slots, &mut order, &mut dtypes);
                    CanonicalAgg::Simple {
                        col_slot: slot,
                        op: *op,
                        dtype: dt,
                    }
                }
                AggSpec::Expression { expr, op, .. } => {
                    // Walk once; reuse the result for slot interning and
                    // for output-dtype derivation below.
                    let referenced = expr.referenced_columns();
                    // First-seen order over the expression tree.
                    for col in &referenced {
                        let dt = lookup_dtype(col, col_dtypes)?;
                        intern(col, dt, &mut slots, &mut order, &mut dtypes);
                    }
                    // Output dtype: from the first column referenced in
                    // the expression (see module-level doc comment for
                    // the simplifying assumption).
                    let output_dtype = match referenced.first() {
                        Some(c) => lookup_dtype(c, col_dtypes)?,
                        // Literal-only expressions: fall back to F64. M3
                        // walker never emits this shape, but the kernel
                        // layer should still produce a deterministic key.
                        None => MetalDtype::F64,
                    };
                    let canon = canon_expr(expr, &slots);
                    CanonicalAgg::Expression {
                        expr: canon,
                        op: *op,
                        dtype: output_dtype,
                    }
                }
                AggSpec::Length { .. } => CanonicalAgg::Length,
            };
            canonical_aggs.push(canonical);
        }

        Ok(Self {
            column_order: order,
            column_dtypes: dtypes,
            aggs: canonical_aggs,
        })
    }

    /// Stable 64-bit hash for use as the library-cache key.
    pub fn hash64(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }

    /// Number of distinct column slots referenced by this signature.
    pub fn column_count(&self) -> usize {
        self.column_dtypes.len()
    }

    /// Number of agg outputs this signature produces.
    pub fn agg_count(&self) -> usize {
        self.aggs.len()
    }

    /// First-seen column names, slot index = position in slice. The
    /// emitter (Task 12) reads this to generate MSL parameter symbols
    /// that still carry the original column-name information for debug
    /// output and alias mapping.
    pub fn column_order(&self) -> &[String] {
        &self.column_order
    }

    /// Per-slot value-column dtype, indexed by the slot indices produced by
    /// canonicalization. Task 12's MSL emitter consumes this to emit
    /// `device const <ty>* value_<i>` parameter types.
    pub fn column_dtypes(&self) -> &[MetalDtype] {
        &self.column_dtypes
    }
}

// ---------- internal helpers ----------------------------------------------

fn lookup_dtype(
    col: &str,
    col_dtypes: &BTreeMap<String, MetalDtype>,
) -> Result<MetalDtype, AggSignatureError> {
    col_dtypes
        .get(col)
        .copied()
        .ok_or_else(|| AggSignatureError::UnknownColumn(col.to_string()))
}

fn intern(
    name: &str,
    dtype: MetalDtype,
    slots: &mut BTreeMap<String, u16>,
    order: &mut Vec<String>,
    dtypes: &mut Vec<MetalDtype>,
) -> u16 {
    if let Some(&slot) = slots.get(name) {
        return slot;
    }
    let slot = order.len() as u16;
    slots.insert(name.to_string(), slot);
    order.push(name.to_string());
    dtypes.push(dtype);
    slot
}

/// Rewrite an expression's column references to slot indices. The
/// `slots` map must already contain every column referenced (caller
/// pre-walks `referenced_columns()` to populate).
fn canon_expr(e: &AggExpr, slots: &BTreeMap<String, u16>) -> CanonicalExpr {
    match e {
        AggExpr::Column(name) => {
            // Callers pre-populate `slots` with every column returned by
            // `referenced_columns()`, so this lookup is unreachable
            // unless the contract is violated. A `debug_assert!` documents
            // that intent; release builds fall back to slot 0, which is
            // still deterministic for the signature contract.
            debug_assert!(
                slots.contains_key(name),
                "canon_expr: slots missing column `{name}`; caller must pre-intern via referenced_columns()"
            );
            let slot = slots.get(name).copied().unwrap_or(0);
            CanonicalExpr::Column(slot)
        }
        AggExpr::LiteralF64(v) => CanonicalExpr::LiteralF64Bits(v.to_bits()),
        AggExpr::LiteralI64(v) => CanonicalExpr::LiteralI64(*v),
        AggExpr::Binary { op, lhs, rhs } => CanonicalExpr::Binary {
            op: *op,
            lhs: Box::new(canon_expr(lhs, slots)),
            rhs: Box::new(canon_expr(rhs, slots)),
        },
    }
}
