// crates/polars-metal-mlx-sys/src/elementwise.rs
//! Elementwise op bindings: thin wrappers over the cxx::bridge functions.
//!
//! Each function returns a fresh `MlxArrayHandle` representing the graph node
//! for the op. Nothing executes until [`crate::array::mlx_array_eval`] is called
//! on the result (or any downstream handle).
//!
//! # Keep-alive semantics
//! Binary ops inherit `_input_refs` from both arguments so that any
//! `MetalBuffer` backing a zero-copy input stays alive until the op result
//! is dropped or evaluated. Unary ops inherit from the single argument.
//!
//! # Errors
//! - Shape mismatch (broadcast-incompatible) → `FfiError::Runtime`
//! - Dtype mismatch → `FfiError::Runtime`

use crate::array::MlxArrayHandle;
use crate::error::FfiError;
use crate::ffi;

macro_rules! binop {
    ($rs:ident, $cpp:ident) => {
        /// Elementwise binary op. Inherits `_input_refs` from both args.
        pub fn $rs(a: &MlxArrayHandle, b: &MlxArrayHandle) -> Result<MlxArrayHandle, FfiError> {
            // Inherit input refs from both args so the result keeps inputs alive.
            let mut refs = a._input_refs.clone();
            refs.extend(b._input_refs.iter().cloned());
            let ptr = ffi::$cpp(&a.ptr, &b.ptr).map_err(FfiError::from)?;
            Ok(MlxArrayHandle {
                ptr,
                _input_refs: refs,
            })
        }
    };
}

macro_rules! unop {
    ($rs:ident, $cpp:ident) => {
        /// Elementwise unary op. Inherits `_input_refs` from the argument.
        pub fn $rs(a: &MlxArrayHandle) -> Result<MlxArrayHandle, FfiError> {
            let ptr = ffi::$cpp(&a.ptr).map_err(FfiError::from)?;
            Ok(MlxArrayHandle {
                ptr,
                _input_refs: a._input_refs.clone(),
            })
        }
    };
}

// ── Binary arithmetic ─────────────────────────────────────────────────────────
binop!(mlx_add, mlx_op_add);
binop!(mlx_sub, mlx_op_sub);
binop!(mlx_mul, mlx_op_mul);
binop!(mlx_div, mlx_op_div);
// mlx_mod_ wraps `mlx::core::remainder`; trailing underscore avoids `mod` keyword.
binop!(mlx_mod_, mlx_op_mod);
binop!(mlx_pow, mlx_op_pow);

// ── Binary comparison (Bool output) ──────────────────────────────────────────
binop!(mlx_eq, mlx_op_eq);
binop!(mlx_ne, mlx_op_ne);
binop!(mlx_lt, mlx_op_lt);
binop!(mlx_le, mlx_op_le);
binop!(mlx_gt, mlx_op_gt);
binop!(mlx_ge, mlx_op_ge);

// ── Binary logical ────────────────────────────────────────────────────────────
binop!(mlx_logical_and, mlx_op_logical_and);
binop!(mlx_logical_or, mlx_op_logical_or);

// ── Unary ─────────────────────────────────────────────────────────────────────
// Note: `neg` wraps `mlx::core::negative`; `logical_not` wraps `mlx::core::logical_not`.
unop!(mlx_logical_not, mlx_op_logical_not);
unop!(mlx_neg, mlx_op_neg);
unop!(mlx_abs, mlx_op_abs);
unop!(mlx_square, mlx_op_square);

// ── Ternary ───────────────────────────────────────────────────────────────────

/// Element-wise selection: `cond ? then_v : else_v`.
///
/// `cond` must be a Bool array; `then_v` and `else_v` must have the same dtype.
/// Inherits `_input_refs` from all three arguments.
///
/// # Errors
/// Returns `FfiError::Runtime` on dtype/shape mismatch (propagated from MLX).
pub fn mlx_where(
    cond: &MlxArrayHandle,
    then_v: &MlxArrayHandle,
    else_v: &MlxArrayHandle,
) -> Result<MlxArrayHandle, FfiError> {
    let mut refs = cond._input_refs.clone();
    refs.extend(then_v._input_refs.iter().cloned());
    refs.extend(else_v._input_refs.iter().cloned());
    let ptr = ffi::mlx_op_where(&cond.ptr, &then_v.ptr, &else_v.ptr).map_err(FfiError::from)?;
    Ok(MlxArrayHandle {
        ptr,
        _input_refs: refs,
    })
}
