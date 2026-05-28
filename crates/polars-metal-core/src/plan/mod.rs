//! Intermediate IR between the Polars-IR walker (Python side) and the
//! kernel-dispatch layer. Each accepted IR subtree is lowered into a
//! `MetalPlanNode` tree which the UDF entry point in `udf.rs` interprets.
//!
//! The IR is deliberately small: only the IR shapes M1 actually supports.
//! New IR variants land alongside the kernels that implement them.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalDtype {
    I64,
    F64,
    Bool,
    I32,
    F32,
    // M3 capability F additions: smaller-integer key dtypes.
    // These are accepted as composite-key components only; aggregation
    // value columns of these dtypes are not supported (router falls back).
    I8,
    I16,
    U8,
    U16,
    U32,
    // M3 Phase 7: dictionary-encoded Utf8 keys.
    Utf8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Predicate expression AST. Only shapes in the M1 closed set; the walker
/// returns FallBack for anything else (`is_null`, NOT, casts, arithmetic, etc.)
#[derive(Debug, Clone)]
pub enum PredicateAst {
    Column {
        name: String,
        dtype: MetalDtype,
    },
    LiteralI64(i64),
    LiteralF64(f64),
    LiteralBool(bool),
    Compare {
        op: CompareOp,
        lhs: Box<PredicateAst>,
        rhs: Box<PredicateAst>,
    },
    And(Box<PredicateAst>, Box<PredicateAst>),
    Or(Box<PredicateAst>, Box<PredicateAst>),
}

/// Aggregation operator. Six variants matching spec § "Aggregations
/// delivered". `Len` is `pl.len()` — the row count per group, no input
/// column read. `Count` is `pl.col(x).count()` — the count of non-null
/// values in the input column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggOp {
    Sum,
    Mean,
    Count,
    Min,
    Max,
    Len,
}

impl MetalDtype {
    /// Parse the wire string emitted by the Python walker.
    pub fn from_wire(s: &str) -> Option<Self> {
        match s {
            "I64" => Some(MetalDtype::I64),
            "F64" => Some(MetalDtype::F64),
            "Bool" => Some(MetalDtype::Bool),
            "I32" => Some(MetalDtype::I32),
            "F32" => Some(MetalDtype::F32),
            "I8" => Some(MetalDtype::I8),
            "I16" => Some(MetalDtype::I16),
            "U8" => Some(MetalDtype::U8),
            "U16" => Some(MetalDtype::U16),
            "U32" => Some(MetalDtype::U32),
            "Utf8" | "String" => Some(MetalDtype::Utf8),
            // pl.Date is stored as Int32 days-since-1970 — same physical
            // layout as I32. We alias here rather than add a Date variant
            // because the kernels never need to distinguish at runtime:
            // groupby/encoder/scan all consume the raw i32 buffer. The
            // predicate path widens Date to I64 at the walker level so the
            // existing cmp_i64 kernel handles Date comparisons (see
            // `_walker._PREDICATE_I64_WIDEN`).
            "Date" => Some(MetalDtype::I32),
            _ => None,
        }
    }
}

impl AggOp {
    /// Parse the wire string emitted by the Python walker.
    pub fn from_wire(s: &str) -> Option<Self> {
        match s {
            "Sum" => Some(AggOp::Sum),
            "Mean" => Some(AggOp::Mean),
            "Count" => Some(AggOp::Count),
            "Min" => Some(AggOp::Min),
            "Max" => Some(AggOp::Max),
            "Len" => Some(AggOp::Len),
            _ => None,
        }
    }
}

/// Binary operations supported in inline aggregation expressions.
/// Capability G's scope: arithmetic only; no comparison / boolean / function calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Expression tree consumed by the fused aggregation kernel.
/// Operands are columns or literals; operations are binary arithmetic.
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
    /// All columns referenced anywhere in the tree, in left-to-right order,
    /// deduplicated. Used by the kernel template engine to know which
    /// buffers to bind.
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

    /// Depth of the binary-expression tree. Leaves (columns / literals) have
    /// depth 0; a `Binary { lhs, rhs, .. }` has depth `1 + max(lhs, rhs)`.
    pub fn depth(&self) -> usize {
        match self {
            AggExpr::Column(_) | AggExpr::LiteralF64(_) | AggExpr::LiteralI64(_) => 0,
            AggExpr::Binary { lhs, rhs, .. } => 1 + lhs.depth().max(rhs.depth()),
        }
    }

    /// Apply M3's depth cap (4) to keep MSL emission bounded. Returns a
    /// human-readable error string on violation.
    pub fn validate(&self) -> Result<(), String> {
        if self.depth() > 4 {
            return Err(format!(
                "expression depth {} exceeds M3 cap of 4",
                self.depth()
            ));
        }
        Ok(())
    }

    fn walk<F: FnMut(&AggExpr)>(&self, f: &mut F) {
        f(self);
        if let AggExpr::Binary { lhs, rhs, .. } = self {
            lhs.walk(f);
            rhs.walk(f);
        }
    }
}

/// Aggregation specification. Three variants:
/// - `Simple` — aggregate one input column (M2 shape).
/// - `Expression` — aggregate the value of an inline binary-arithmetic
///   expression (M3, capability G). Phase 2 lands the IR; Phase 3 supplies
///   the fused-kernel consumer.
/// - `Length` — `pl.len()`, counts rows per group; no input column read.
///
/// Output dtype is **not** carried here. The kernel layer derives it from
/// the input column dtype(s) + op semantics at dispatch / signature time.
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

/// Lowered IR — one variant per accepted Polars IR node type.
#[derive(Debug, Clone)]
pub enum MetalPlanNode {
    Scan {
        n_rows: usize,
        columns: Vec<(String, MetalDtype)>,
    },
    Project {
        input: Box<MetalPlanNode>,
        columns: Vec<String>,
    },
    Filter {
        input: Box<MetalPlanNode>,
        predicate: PredicateAst,
    },
    /// Hash groupby with composite keys (≤ 128 bits total) and one or
    /// more aggregation specs. See spec § "Two-pass groupby algorithm"
    /// for the kernel-side flow; this variant only records intent.
    GroupBy {
        input: Box<MetalPlanNode>,
        keys: Vec<(String, MetalDtype)>,
        aggs: Vec<AggSpec>,
    },
}
