//! Build an MLX expression graph from a FusionScope; eval; fold back.
//!
//! This is the Phase 4 bridge between the analyzer's declarative scope
//! (Phase 2 + 3) and the MLX FFI surface from Phase 1. The subgraph builder
//! walks the scope's op list, maps each OpId to the matching MLX FFI call,
//! and produces an evaluable handle graph.
//!
//! Architectural note: the plan placed this in `polars-metal-kernels`, but
//! that creates a circular dependency on `polars-metal-core::fusion`. The
//! subgraph builder lives in `polars-metal-core` as a continuation of the
//! fusion module; `polars-metal-kernels` continues to own the custom-MSL
//! kernel work.

use polars_metal_buffer::{MetalBuffer, MetalDevice};
use polars_metal_mlx_sys::array::{
    mlx_array_eval, mlx_array_from_f32_slice, mlx_array_to_f32_vec, mlx_array_view_metal_buffer,
    MlxArrayHandle, MlxDtype,
};
use polars_metal_mlx_sys::elementwise::{
    mlx_abs, mlx_acos, mlx_add, mlx_asin, mlx_atan, mlx_atan2, mlx_cbrt, mlx_ceil, mlx_cos,
    mlx_cosh, mlx_div, mlx_eq, mlx_exp, mlx_exp2, mlx_floor, mlx_ge, mlx_gt, mlx_le, mlx_log,
    mlx_log10, mlx_log1p, mlx_log2, mlx_logical_and, mlx_logical_not, mlx_logical_or, mlx_lt,
    mlx_mod_, mlx_mul, mlx_ne, mlx_neg, mlx_pow, mlx_round, mlx_sin, mlx_sinh, mlx_sqrt,
    mlx_square, mlx_sub, mlx_tan, mlx_tanh, mlx_where,
};
use polars_metal_mlx_sys::fft::{mlx_fft, mlx_ifft};
use polars_metal_mlx_sys::matmul::mlx_matmul;
use polars_metal_mlx_sys::reduce::{
    mlx_argmax, mlx_argmin, mlx_max, mlx_mean, mlx_min, mlx_std, mlx_sum, mlx_var,
};
use polars_metal_mlx_sys::scan::{mlx_cummax, mlx_cummin, mlx_cumprod, mlx_cumsum};
use polars_metal_mlx_sys::sort::mlx_sort;
use thiserror::Error;

use super::scope::{FusionScope, OpNode};
use super::supported_ops::OpId;

#[derive(Debug, Error)]
pub enum BuildError {
    #[error("scope expected {expected} inputs, got {actual}")]
    InputCountMismatch { expected: usize, actual: usize },
    #[error("scope references undefined node index {0}")]
    UndefinedNode(u32),
    #[error("MLX FFI error: {0}")]
    MlxError(String),
    #[error("op {0:?} not yet supported by the subgraph builder")]
    UnsupportedOp(OpId),
    #[error("op {op:?} expected {expected} arg(s), got {actual}")]
    ArgCountMismatch {
        op: OpId,
        expected: usize,
        actual: usize,
    },
}

/// Stand-in for `polars_metal_buffer::MetalBuffer` used by tests. The
/// production zero-copy path goes through `from_fusion_scope_buffers`
/// (Task 19); this Vec<f32>-backed type keeps Task 18 tests self-contained.
pub struct ColumnBuffer {
    data: Vec<f32>,
}

impl ColumnBuffer {
    pub fn from_f32_vec(data: Vec<f32>) -> Self {
        Self { data }
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>, BuildError> {
        Ok(self.data.clone())
    }

    pub fn as_handle(&self) -> Result<MlxArrayHandle, BuildError> {
        mlx_array_from_f32_slice(&self.data).map_err(|e| BuildError::MlxError(format!("{e:?}")))
    }
}

/// An evaluable MLX expression graph built from a `FusionScope`. Handles
/// are kept alive in a single flat vector indexed by `NodeIdx`: inputs come
/// first, then op-results in scope-order.
pub struct MlxSubgraph {
    /// Used for refcount keep-alive only; not read after construction.
    #[allow(dead_code)]
    handles: Vec<MlxArrayHandle>,
    outputs: Vec<MlxArrayHandle>,
}

impl MlxSubgraph {
    pub fn from_fusion_scope(
        scope: &FusionScope,
        inputs: &[ColumnBuffer],
    ) -> Result<Self, BuildError> {
        if inputs.len() != scope.inputs.len() {
            return Err(BuildError::InputCountMismatch {
                expected: scope.inputs.len(),
                actual: inputs.len(),
            });
        }

        let mut handles: Vec<MlxArrayHandle> = inputs
            .iter()
            .map(|b| b.as_handle())
            .collect::<Result<Vec<_>, _>>()?;

        for op_node in &scope.ops {
            let handle = build_op(op_node, &handles)?;
            handles.push(handle);
        }

        let outputs: Vec<MlxArrayHandle> = scope
            .outputs
            .iter()
            .map(|idx| {
                handles
                    .get(idx.0 as usize)
                    .cloned()
                    .ok_or(BuildError::UndefinedNode(idx.0))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self { handles, outputs })
    }

    pub fn eval(&self) -> Result<Vec<ColumnBuffer>, BuildError> {
        mlx_array_eval(&self.outputs).map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
        let mut outs = Vec::with_capacity(self.outputs.len());
        for h in &self.outputs {
            let data =
                mlx_array_to_f32_vec(h).map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
            outs.push(ColumnBuffer { data });
        }
        Ok(outs)
    }

    /// Production-path constructor: build the subgraph over zero-copy views
    /// of existing `MetalBuffer`s. Input arrays are constructed via
    /// `mlx_array_view_metal_buffer` so MLX reads directly from the caller's
    /// memory without an additional copy. Lifetime safety is enforced by
    /// `_input_refs` in the MlxArrayHandle.
    pub fn from_fusion_scope_buffers(
        scope: &FusionScope,
        inputs: &[std::sync::Arc<MetalBuffer>],
    ) -> Result<Self, BuildError> {
        if inputs.len() != scope.inputs.len() {
            return Err(BuildError::InputCountMismatch {
                expected: scope.inputs.len(),
                actual: inputs.len(),
            });
        }
        let mut handles: Vec<MlxArrayHandle> = inputs
            .iter()
            .map(|buf| {
                // Derive 1-D shape from buffer byte length (F32 = 4 bytes each).
                let n_elements = (buf.len() / 4) as i64;
                let shape = [n_elements];
                mlx_array_view_metal_buffer(buf.clone(), &shape, MlxDtype::F32)
                    .map_err(|e| BuildError::MlxError(format!("{e:?}")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for op_node in &scope.ops {
            let handle = build_op(op_node, &handles)?;
            handles.push(handle);
        }
        let outputs: Vec<MlxArrayHandle> = scope
            .outputs
            .iter()
            .map(|idx| {
                handles
                    .get(idx.0 as usize)
                    .cloned()
                    .ok_or(BuildError::UndefinedNode(idx.0))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { handles, outputs })
    }

    /// Eval the subgraph and copy each output back into a freshly allocated
    /// `MetalBuffer`.
    ///
    /// Phase 4 implementation note: MLX-side allocations live behind the
    /// allocator API; extracting the underlying MTL::Buffer\* and wrapping
    /// it as a `MetalBuffer` without copying requires exposing the MLX
    /// allocator surface through the FFI bridge. For Phase 4 we take the
    /// pragmatic path - read the output as Vec<f32> and stage a fresh
    /// `MetalBuffer`. This adds one F32 copy at the output, which doesn't
    /// affect the input zero-copy path (the dominant cost for large
    /// transcendental chains). Full output-zero-copy is a future
    /// optimization tracked alongside the MLX allocator-surface work.
    pub fn eval_to_metal_buffers(
        &self,
        device: &MetalDevice,
    ) -> Result<Vec<MetalBuffer>, BuildError> {
        mlx_array_eval(&self.outputs).map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
        let mut outs = Vec::with_capacity(self.outputs.len());
        for h in &self.outputs {
            let data =
                mlx_array_to_f32_vec(h).map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
            let buf = MetalBuffer::from_f32_slice(device, &data)
                .map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
            outs.push(buf);
        }
        Ok(outs)
    }
}

fn ffi<T>(r: Result<T, polars_metal_mlx_sys::FfiError>) -> Result<T, BuildError> {
    r.map_err(|e| BuildError::MlxError(format!("{e:?}")))
}

fn arg_count(op: OpId, expected: usize, args: &[&MlxArrayHandle]) -> Result<(), BuildError> {
    if args.len() != expected {
        return Err(BuildError::ArgCountMismatch {
            op,
            expected,
            actual: args.len(),
        });
    }
    Ok(())
}

fn build_op(node: &OpNode, handles: &[MlxArrayHandle]) -> Result<MlxArrayHandle, BuildError> {
    use OpId::*;

    let args: Vec<&MlxArrayHandle> = node
        .args
        .iter()
        .map(|idx| {
            handles
                .get(idx.0 as usize)
                .ok_or(BuildError::UndefinedNode(idx.0))
        })
        .collect::<Result<Vec<_>, _>>()?;

    match node.op {
        // Arithmetic
        Add => {
            arg_count(node.op, 2, &args)?;
            ffi(mlx_add(args[0], args[1]))
        }
        Sub => {
            arg_count(node.op, 2, &args)?;
            ffi(mlx_sub(args[0], args[1]))
        }
        Mul => {
            arg_count(node.op, 2, &args)?;
            ffi(mlx_mul(args[0], args[1]))
        }
        Div => {
            arg_count(node.op, 2, &args)?;
            ffi(mlx_div(args[0], args[1]))
        }
        Mod => {
            arg_count(node.op, 2, &args)?;
            ffi(mlx_mod_(args[0], args[1]))
        }
        Pow => {
            arg_count(node.op, 2, &args)?;
            ffi(mlx_pow(args[0], args[1]))
        }
        Neg => {
            arg_count(node.op, 1, &args)?;
            ffi(mlx_neg(args[0]))
        }
        Abs => {
            arg_count(node.op, 1, &args)?;
            ffi(mlx_abs(args[0]))
        }
        Square => {
            arg_count(node.op, 1, &args)?;
            ffi(mlx_square(args[0]))
        }

        // Comparison
        Eq => ffi(mlx_eq(args[0], args[1])),
        Ne => ffi(mlx_ne(args[0], args[1])),
        Lt => ffi(mlx_lt(args[0], args[1])),
        Le => ffi(mlx_le(args[0], args[1])),
        Gt => ffi(mlx_gt(args[0], args[1])),
        Ge => ffi(mlx_ge(args[0], args[1])),

        // Logical
        LogicalAnd => ffi(mlx_logical_and(args[0], args[1])),
        LogicalOr => ffi(mlx_logical_or(args[0], args[1])),
        LogicalNot => ffi(mlx_logical_not(args[0])),

        // Where
        Where => {
            arg_count(node.op, 3, &args)?;
            ffi(mlx_where(args[0], args[1], args[2]))
        }

        // Transcendentals
        Sin => ffi(mlx_sin(args[0])),
        Cos => ffi(mlx_cos(args[0])),
        Tan => ffi(mlx_tan(args[0])),
        Sinh => ffi(mlx_sinh(args[0])),
        Cosh => ffi(mlx_cosh(args[0])),
        Tanh => ffi(mlx_tanh(args[0])),
        Asin => ffi(mlx_asin(args[0])),
        Acos => ffi(mlx_acos(args[0])),
        Atan => ffi(mlx_atan(args[0])),
        Atan2 => ffi(mlx_atan2(args[0], args[1])),
        Log => ffi(mlx_log(args[0])),
        Log2 => ffi(mlx_log2(args[0])),
        Log10 => ffi(mlx_log10(args[0])),
        Log1p => ffi(mlx_log1p(args[0])),
        Exp => ffi(mlx_exp(args[0])),
        Exp2 => ffi(mlx_exp2(args[0])),
        Sqrt => ffi(mlx_sqrt(args[0])),
        Cbrt => ffi(mlx_cbrt(args[0])),
        Floor => ffi(mlx_floor(args[0])),
        Ceil => ffi(mlx_ceil(args[0])),
        Round => ffi(mlx_round(args[0])),

        // Cast - the subgraph builder uses our existing mlx_cast which takes
        // an MlxDtype. CastF64 will throw at runtime since MLX 0.22.0 lacks F64.
        CastF32 => ffi(polars_metal_mlx_sys::elementwise::mlx_cast(
            args[0],
            polars_metal_mlx_sys::array::MlxDtype::F32,
        )),
        CastF64 => ffi(polars_metal_mlx_sys::elementwise::mlx_cast(
            args[0],
            polars_metal_mlx_sys::array::MlxDtype::F64,
        )),
        CastI32 => ffi(polars_metal_mlx_sys::elementwise::mlx_cast(
            args[0],
            polars_metal_mlx_sys::array::MlxDtype::I32,
        )),
        CastBool => ffi(polars_metal_mlx_sys::elementwise::mlx_cast(
            args[0],
            polars_metal_mlx_sys::array::MlxDtype::Bool,
        )),

        // Reductions (global; the scope::has_terminator tag identifies these).
        Sum => ffi(mlx_sum(args[0])),
        Mean => ffi(mlx_mean(args[0])),
        Min => ffi(mlx_min(args[0])),
        Max => ffi(mlx_max(args[0])),
        Std => ffi(mlx_std(args[0])),
        Var => ffi(mlx_var(args[0])),
        ArgMin => ffi(mlx_argmin(args[0])),
        ArgMax => ffi(mlx_argmax(args[0])),

        // Sort / top-k
        Sort => ffi(mlx_sort(args[0])),
        ArgPartition => {
            // ArgPartition needs a `kth` parameter that the FusionScope doesn't
            // carry yet. Phase 10 (vector search) adds it as scope metadata.
            Err(BuildError::UnsupportedOp(node.op))
        }

        // Cumulative scans need an axis argument. For 1-D inputs (the only
        // shape produced by the analyzer in this chunk) axis=0 is the only
        // valid choice; multi-dim cumulative scans land in a later phase.
        CumSum => ffi(mlx_cumsum(args[0], 0)),
        CumProd => ffi(mlx_cumprod(args[0], 0)),
        CumMax => ffi(mlx_cummax(args[0], 0)),
        CumMin => ffi(mlx_cummin(args[0], 0)),

        // Matmul - inputs must be 2-D; Phase 10 (List/Array dot) sets that up.
        MatMul => ffi(mlx_matmul(args[0], args[1])),

        // FFT (Phase 11 surface; needs the metal.fft() namespace before it's
        // reachable from the analyzer, but the dispatch is wired now).
        Fft => ffi(mlx_fft(args[0])),
        Ifft => ffi(mlx_ifft(args[0])),
    }
}
