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

/// One aggregation expression in a GroupBy. `input_col` is empty for
/// `AggOp::Len` (the kernel doesn't read a value column for row count).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggSpec {
    /// Column the aggregation reads. Empty string for `AggOp::Len`.
    pub input_col: String,
    pub op: AggOp,
    /// Output column name in the result DataFrame. Polars users set this
    /// via `.agg(pl.col(x).sum().alias("foo"))`; if no alias, Polars
    /// synthesises one (e.g. `"v_sum"`). The walker fills it from the
    /// Polars IR.
    pub output_alias: String,
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
