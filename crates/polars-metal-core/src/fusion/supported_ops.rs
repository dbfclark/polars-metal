// crates/polars-metal-core/src/fusion/supported_ops.rs
//! Registry of expression ops the M4 fusion analyzer recognizes.
//!
//! Each op has:
//!   - n_args:        how many child expressions it consumes
//!   - flops_per_row: static FLOP estimate per row (0 means dynamic; see
//!                    OpSpec::dynamic_flops)
//!   - input_dtype:   what dtype the op accepts (F32, F32orF64, Bool, ...)
//!   - output_dtype:  what dtype the op produces (same | Bool | F32)
//!
//! The analyzer uses this table to decide whether an expression node is
//! supported and to compute total subtree FLOPs.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OpId {
    // Arithmetic (binary)
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    // Arithmetic (unary)
    Neg,
    Abs,
    Square,
    // Comparison (binary; output Bool)
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    // Logical (binary on Bool)
    LogicalAnd,
    LogicalOr,
    // Logical (unary)
    LogicalNot,
    // Conditional (ternary; cond Bool, then/else same dtype)
    Where,
    // Transcendental (unary, F32)
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Asin,
    Acos,
    Atan,
    Atan2,
    Log,
    Log2,
    Log10,
    Log1p,
    Exp,
    Exp2,
    Sqrt,
    Cbrt,
    // Rounding (unary, F32)
    Floor,
    Ceil,
    Round,
    // Cast
    CastF32,
    CastF64,
    CastI32,
    CastBool,
    // Reduction (unary; output is shape-(1,))
    Sum,
    Mean,
    Min,
    Max,
    Std,
    Var,
    ArgMin,
    ArgMax,
    // Sort/top-k (unary)
    Sort,
    ArgPartition,
    // Cumulative scan (unary)
    CumSum,
    CumProd,
    CumMax,
    CumMin,
    // Matmul (binary)
    MatMul,
    // FFT (unary, F32 -> complex; Phase 11)
    Fft,
    Ifft,
}

#[derive(Clone, Copy, Debug)]
pub enum DtypeReq {
    F32,
    F32OrF64,
    Bool,
    Numeric,
    ListOrArrayF32,
}

#[derive(Clone, Copy, Debug)]
pub enum DtypeOut {
    SameAsInput,
    Bool,
    F32,
    ScalarF32,
    ComplexF32,
    SortedSameAsInput,
    I32,
}

#[derive(Clone, Copy, Debug)]
pub struct OpSpec {
    pub n_args: u32,
    pub flops_per_row: u32,
    pub input_dtype: DtypeReq,
    pub output_dtype: DtypeOut,
    pub dynamic_flops: bool,
    pub allows_null: bool,
}

pub fn op_spec(op: OpId) -> OpSpec {
    use DtypeOut as O;
    use DtypeReq as I;
    use OpId::*;
    match op {
        // Arithmetic
        Add | Sub | Mul => OpSpec {
            n_args: 2,
            flops_per_row: 1,
            input_dtype: I::Numeric,
            output_dtype: O::SameAsInput,
            dynamic_flops: false,
            allows_null: false,
        },
        Div => OpSpec {
            n_args: 2,
            flops_per_row: 4,
            input_dtype: I::Numeric,
            output_dtype: O::SameAsInput,
            dynamic_flops: false,
            allows_null: false,
        },
        Mod => OpSpec {
            n_args: 2,
            flops_per_row: 4,
            input_dtype: I::F32OrF64,
            output_dtype: O::SameAsInput,
            dynamic_flops: false,
            allows_null: false,
        },
        Pow => OpSpec {
            n_args: 2,
            flops_per_row: 12,
            input_dtype: I::F32OrF64,
            output_dtype: O::SameAsInput,
            dynamic_flops: false,
            allows_null: false,
        },
        Neg | Abs => OpSpec {
            n_args: 1,
            flops_per_row: 1,
            input_dtype: I::Numeric,
            output_dtype: O::SameAsInput,
            dynamic_flops: false,
            allows_null: false,
        },
        Square => OpSpec {
            n_args: 1,
            flops_per_row: 1,
            input_dtype: I::Numeric,
            output_dtype: O::SameAsInput,
            dynamic_flops: false,
            allows_null: false,
        },
        // Comparison → Bool
        Eq | Ne | Lt | Le | Gt | Ge => OpSpec {
            n_args: 2,
            flops_per_row: 1,
            input_dtype: I::Numeric,
            output_dtype: O::Bool,
            dynamic_flops: false,
            allows_null: false,
        },
        // Logical
        LogicalAnd | LogicalOr => OpSpec {
            n_args: 2,
            flops_per_row: 1,
            input_dtype: I::Bool,
            output_dtype: O::Bool,
            dynamic_flops: false,
            allows_null: false,
        },
        LogicalNot => OpSpec {
            n_args: 1,
            flops_per_row: 1,
            input_dtype: I::Bool,
            output_dtype: O::Bool,
            dynamic_flops: false,
            allows_null: false,
        },
        // Where (cond, then, else)
        Where => OpSpec {
            n_args: 3,
            flops_per_row: 1,
            input_dtype: I::Numeric,
            output_dtype: O::SameAsInput,
            dynamic_flops: false,
            allows_null: false,
        },
        // Transcendental (unary) — NOTE: Tanh appears once here only (deduped from plan)
        Sin | Cos | Tan | Sinh | Cosh | Tanh | Asin | Acos | Atan | Log | Log2 | Log10 | Log1p
        | Exp | Exp2 => OpSpec {
            n_args: 1,
            flops_per_row: 10,
            input_dtype: I::F32OrF64,
            output_dtype: O::SameAsInput,
            dynamic_flops: false,
            allows_null: false,
        },
        Atan2 => OpSpec {
            n_args: 2,
            flops_per_row: 12,
            input_dtype: I::F32OrF64,
            output_dtype: O::SameAsInput,
            dynamic_flops: false,
            allows_null: false,
        },
        // Roots
        Sqrt | Cbrt => OpSpec {
            n_args: 1,
            flops_per_row: 4,
            input_dtype: I::F32OrF64,
            output_dtype: O::SameAsInput,
            dynamic_flops: false,
            allows_null: false,
        },
        // Rounding
        Floor | Ceil | Round => OpSpec {
            n_args: 1,
            flops_per_row: 1,
            input_dtype: I::F32OrF64,
            output_dtype: O::SameAsInput,
            dynamic_flops: false,
            allows_null: false,
        },
        // Cast — output_dtype is the target dtype, not SameAsInput. CastF32/
        // CastF64/CastI32 all produce numeric output (we use F32 as the tag
        // here since DtypeOut has no F64/I32 variants today; Phase 3's analyzer
        // dispatches the actual concrete dtype from the OpId itself).
        CastF32 | CastF64 | CastI32 => OpSpec {
            n_args: 1,
            flops_per_row: 1,
            input_dtype: I::Numeric,
            output_dtype: O::F32,
            dynamic_flops: false,
            allows_null: false,
        },
        CastBool => OpSpec {
            n_args: 1,
            flops_per_row: 1,
            input_dtype: I::Numeric,
            output_dtype: O::Bool,
            dynamic_flops: false,
            allows_null: false,
        },
        // Reductions
        Sum | Mean | Min | Max => OpSpec {
            n_args: 1,
            flops_per_row: 1,
            input_dtype: I::Numeric,
            output_dtype: O::ScalarF32,
            dynamic_flops: false,
            allows_null: false,
        },
        Std | Var => OpSpec {
            n_args: 1,
            flops_per_row: 3,
            input_dtype: I::Numeric,
            output_dtype: O::ScalarF32,
            dynamic_flops: false,
            allows_null: false,
        },
        ArgMin | ArgMax => OpSpec {
            n_args: 1,
            flops_per_row: 1,
            input_dtype: I::Numeric,
            output_dtype: O::I32,
            dynamic_flops: false,
            allows_null: false,
        },
        // Sort / top-k
        Sort => OpSpec {
            n_args: 1,
            flops_per_row: 0,
            input_dtype: I::Numeric,
            output_dtype: O::SortedSameAsInput,
            dynamic_flops: true,
            allows_null: false,
        }, // log2(n) * n
        ArgPartition => OpSpec {
            n_args: 1,
            flops_per_row: 0,
            input_dtype: I::Numeric,
            output_dtype: O::I32,
            dynamic_flops: true,
            allows_null: false,
        },
        // Cumulative
        CumSum | CumProd => OpSpec {
            n_args: 1,
            flops_per_row: 2,
            input_dtype: I::Numeric,
            output_dtype: O::SameAsInput,
            dynamic_flops: false,
            allows_null: false,
        },
        CumMax | CumMin => OpSpec {
            n_args: 1,
            flops_per_row: 2,
            input_dtype: I::Numeric,
            output_dtype: O::SameAsInput,
            dynamic_flops: false,
            allows_null: false,
        },
        // Matmul — n_rows * 2 * K per output cell
        MatMul => OpSpec {
            n_args: 2,
            flops_per_row: 0,
            input_dtype: I::ListOrArrayF32,
            output_dtype: O::F32,
            dynamic_flops: true,
            allows_null: false,
        },
        // FFT — n * log2(n) per axis
        Fft | Ifft => OpSpec {
            n_args: 1,
            flops_per_row: 0,
            input_dtype: I::F32,
            output_dtype: O::ComplexF32,
            dynamic_flops: true,
            allows_null: false,
        },
    }
}

pub fn all_op_ids() -> impl Iterator<Item = OpId> {
    use OpId::*;
    [
        Add,
        Sub,
        Mul,
        Div,
        Mod,
        Pow,
        Neg,
        Abs,
        Square,
        Eq,
        Ne,
        Lt,
        Le,
        Gt,
        Ge,
        LogicalAnd,
        LogicalOr,
        LogicalNot,
        Where,
        Sin,
        Cos,
        Tan,
        Sinh,
        Cosh,
        Tanh,
        Asin,
        Acos,
        Atan,
        Atan2,
        Log,
        Log2,
        Log10,
        Log1p,
        Exp,
        Exp2,
        Sqrt,
        Cbrt,
        Floor,
        Ceil,
        Round,
        CastF32,
        CastF64,
        CastI32,
        CastBool,
        Sum,
        Mean,
        Min,
        Max,
        Std,
        Var,
        ArgMin,
        ArgMax,
        Sort,
        ArgPartition,
        CumSum,
        CumProd,
        CumMax,
        CumMin,
        MatMul,
        Fft,
        Ifft,
    ]
    .into_iter()
}
