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
}
