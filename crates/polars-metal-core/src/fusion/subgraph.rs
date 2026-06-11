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
    mlx_array_eval, mlx_array_from_f32_slice, mlx_array_to_f32_vec, mlx_array_to_i16_vec,
    mlx_array_to_i32_vec, mlx_array_to_i64_vec, mlx_array_to_i8_vec, mlx_array_to_u16_vec,
    mlx_array_to_u32_vec, mlx_array_to_u64_vec, mlx_array_to_u8_vec, mlx_array_view_metal_buffer,
    MlxArrayHandle, MlxDtype,
};
use polars_metal_mlx_sys::elementwise::{
    mlx_abs, mlx_acos, mlx_add, mlx_asin, mlx_atan, mlx_atan2, mlx_cbrt, mlx_ceil, mlx_cos,
    mlx_cosh, mlx_div, mlx_eq, mlx_exp, mlx_exp2, mlx_floor, mlx_ge, mlx_gt, mlx_le, mlx_log,
    mlx_log10, mlx_log1p, mlx_log2, mlx_logical_and, mlx_logical_not, mlx_logical_or, mlx_lt,
    mlx_mod_, mlx_mul, mlx_ne, mlx_neg, mlx_pow, mlx_round, mlx_sin, mlx_sinh, mlx_sqrt,
    mlx_square, mlx_sub, mlx_tan, mlx_tanh, mlx_where,
};
use polars_metal_mlx_sys::matmul::mlx_matmul;
use polars_metal_mlx_sys::reduce::{
    mlx_argmax, mlx_argmin, mlx_max, mlx_mean, mlx_min, mlx_std, mlx_sum, mlx_var,
};
use polars_metal_mlx_sys::scan::{
    mlx_cummax, mlx_cummin, mlx_cumprod, mlx_cumsum, mlx_iota_f32, mlx_shift,
};
use polars_metal_mlx_sys::sort::mlx_sort;
use thiserror::Error;

use super::scope::{FusionScope, InputDtype, OpNode};
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
    #[error("unsupported input dtype for buffer path: {0}")]
    UnsupportedInputDtype(String),
}

/// Map a fused-scope `InputDtype` to the `MlxDtype` used to wrap its buffer.
/// Composite / unsupported dtypes (`ArrayF32`/`ListF32`/`F64`) are not flat
/// 1-D numeric columns the buffer path ingests → `UnsupportedInputDtype`.
fn input_dtype_to_mlx(dtype: InputDtype) -> Result<MlxDtype, BuildError> {
    Ok(match dtype {
        InputDtype::F32 => MlxDtype::F32,
        InputDtype::I32 => MlxDtype::I32,
        InputDtype::Bool => MlxDtype::Bool,
        InputDtype::I8 => MlxDtype::I8,
        InputDtype::I16 => MlxDtype::I16,
        InputDtype::I64 => MlxDtype::I64,
        InputDtype::U8 => MlxDtype::U8,
        InputDtype::U16 => MlxDtype::U16,
        InputDtype::U32 => MlxDtype::U32,
        InputDtype::U64 => MlxDtype::U64,
        InputDtype::F64 | InputDtype::ArrayF32(_) | InputDtype::ListF32 => {
            return Err(BuildError::UnsupportedInputDtype(format!("{dtype:?}")))
        }
    })
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

    /// Consume self and return the underlying `Vec<f32>`. Use this from the
    /// PyO3 dispatch path where the ColumnBuffer is dropped right after
    /// readback — saves a 40MB+ clone on each call.
    pub fn into_f32_vec(self) -> Vec<f32> {
        self.data
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

    /// Eval the single-output subgraph and write the output into the caller's
    /// raw buffer at `out_ptr` (output-zero-copy), interpreting it as
    /// `out_dtype`. Returns the element count written.
    ///
    /// The caller pre-allocates an output array of the statically-inferred width
    /// (the analyzer reports it), and this writer copies the eval'd MLX array
    /// into it width-aware. It hard-errors if the eval'd dtype != `out_dtype`
    /// (an analyzer mis-inference guard — refuse before corrupting the caller's
    /// bytes) or if the output doesn't fit `out_cap`.
    ///
    /// # Safety contract (caller)
    /// `out_ptr` addresses `out_cap` writable, contiguous elements of
    /// `out_dtype`, alive for the whole call.
    pub fn eval_into_typed(
        &self,
        out_ptr: usize,
        out_cap: usize,
        out_dtype: MlxDtype,
    ) -> Result<usize, BuildError> {
        if self.outputs.len() != 1 {
            return Err(BuildError::InputCountMismatch {
                expected: 1,
                actual: self.outputs.len(),
            });
        }
        mlx_array_eval(&self.outputs).map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
        let h = &self.outputs[0];
        let got = h
            .dtype()
            .map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
        if got != out_dtype {
            return Err(BuildError::MlxError(format!(
                "output dtype mismatch: declared {out_dtype:?}, eval'd {got:?}"
            )));
        }
        let n: usize = h.shape().iter().product();
        if n > out_cap {
            return Err(BuildError::MlxError(format!(
                "output too large: {n} > capacity {out_cap}"
            )));
        }
        if n == 0 {
            return Ok(0);
        }
        // Local macro to DRY the per-width arms. This is the SECOND
        // exhaustive `match out_dtype` site (the first is
        // `eval_to_metal_buffers`); both are exhaustive over `MlxDtype`, so a
        // new dtype forces a compile error in BOTH — they can't silently drift.
        macro_rules! write_back {
            ($to_vec:path, $t:ty) => {{
                let data = $to_vec(h).map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
                // SAFETY: `out_ptr` addresses `out_cap >= n` writable, contiguous
                // elements of `out_dtype` (caller contract); we copy exactly `n`.
                let dst = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut $t, n) };
                dst.copy_from_slice(&data);
            }};
        }
        match out_dtype {
            MlxDtype::F32 => write_back!(mlx_array_to_f32_vec, f32),
            MlxDtype::I32 => write_back!(mlx_array_to_i32_vec, i32),
            MlxDtype::I8 => write_back!(mlx_array_to_i8_vec, i8),
            MlxDtype::I16 => write_back!(mlx_array_to_i16_vec, i16),
            MlxDtype::I64 => write_back!(mlx_array_to_i64_vec, i64),
            MlxDtype::U8 => write_back!(mlx_array_to_u8_vec, u8),
            MlxDtype::U16 => write_back!(mlx_array_to_u16_vec, u16),
            MlxDtype::U32 => write_back!(mlx_array_to_u32_vec, u32),
            MlxDtype::U64 => write_back!(mlx_array_to_u64_vec, u64),
            MlxDtype::Bool | MlxDtype::F64 => {
                return Err(BuildError::UnsupportedInputDtype(format!(
                    "output {out_dtype:?}"
                )))
            }
        }
        Ok(n)
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
            .zip(scope.inputs.iter())
            .map(|(buf, input_ref)| {
                // Derive 1-D shape from buffer byte length and the input's
                // declared dtype (element width is dtype-dependent).
                let mlx_dtype = input_dtype_to_mlx(input_ref.dtype)?;
                let n_elements = (buf.len() / mlx_dtype.element_size()) as i64;
                let shape = [n_elements];
                mlx_array_view_metal_buffer(buf.clone(), &shape, mlx_dtype)
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
    /// pragmatic path - read the output as a typed Vec matching the eval'd
    /// dtype (`f32` / `i*` / `u*`) and stage a fresh `MetalBuffer` of that
    /// dtype. This adds one copy at the output, which doesn't
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
            let dtype = h
                .dtype()
                .map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
            let buf = match dtype {
                MlxDtype::F32 => {
                    let data = mlx_array_to_f32_vec(h)
                        .map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
                    MetalBuffer::from_f32_slice(device, &data)
                }
                MlxDtype::I32 => {
                    let data = mlx_array_to_i32_vec(h)
                        .map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
                    MetalBuffer::from_i32_slice(device, &data)
                }
                MlxDtype::I8 => {
                    let data = mlx_array_to_i8_vec(h)
                        .map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
                    MetalBuffer::from_i8_slice(device, &data)
                }
                MlxDtype::I16 => {
                    let data = mlx_array_to_i16_vec(h)
                        .map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
                    MetalBuffer::from_i16_slice(device, &data)
                }
                MlxDtype::I64 => {
                    let data = mlx_array_to_i64_vec(h)
                        .map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
                    MetalBuffer::from_i64_slice(device, &data)
                }
                MlxDtype::U8 => {
                    let data = mlx_array_to_u8_vec(h)
                        .map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
                    MetalBuffer::from_u8_slice(device, &data)
                }
                MlxDtype::U16 => {
                    let data = mlx_array_to_u16_vec(h)
                        .map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
                    MetalBuffer::from_u16_slice(device, &data)
                }
                MlxDtype::U32 => {
                    let data = mlx_array_to_u32_vec(h)
                        .map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
                    MetalBuffer::from_u32_slice(device, &data)
                }
                MlxDtype::U64 => {
                    let data = mlx_array_to_u64_vec(h)
                        .map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
                    MetalBuffer::from_u64_slice(device, &data)
                }
                // B1 ships integer + F32 output only. Bool output would need
                // repacking to Arrow's bit-packed validity-style layout, and
                // F64 is unsupported by the MLX-on-Metal view path; both are
                // out of B1 scope and surface as an error (caller falls back to
                // CPU). Revisit if a later milestone needs Bool/F64 output.
                MlxDtype::Bool | MlxDtype::F64 => {
                    return Err(BuildError::UnsupportedInputDtype(format!(
                        "output {dtype:?}"
                    )))
                }
            }
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

        // FFT. UNREACHABLE from this fused-walker path: `fft` is not a NodeTraverser-viewable
        // expression, so the analyzer never builds this arm, and folding a complex FFT result
        // back into an F32 Series here would be wrong anyway. The LIVE FFT path is the `.metal`
        // namespace backed by the hand-rolled MSL kernel (`shaders/fft.metal` /
        // `polars_metal_kernels::fft`), which handles all sizes on-GPU. The fused subgraph does
        // not support FFT, hence `UnsupportedOp`. Do not rely on / extend these arms.
        Fft | Ifft => Err(BuildError::UnsupportedOp(node.op)),

        // Shift (M5 rolling): forward shift with zero-fill; requires a scalar
        // param carrying the shift amount.
        Shift => {
            arg_count(node.op, 1, &args)?;
            let w = node.param.ok_or(BuildError::UnsupportedOp(node.op))?;
            ffi(mlx_shift(args[0], w))
        }

        // RowIndex (M5 rolling): 0-arg iota [0,1,…,n_rows-1] as F32. n_rows is
        // taken from the first input handle — inputs are always bound into
        // `handles` before the op loop, so handles[0] is the first scope input
        // and its length is the row count shared by all inputs.
        // The `.ok_or(UnsupportedOp)` below fires only if `handles` is empty,
        // i.e. a RowIndex op with zero scope inputs — a caller-contract violation
        // that the FusionScope analyzer never produces for a valid rolling scope.
        // This path is effectively unreachable; `UnsupportedOp` is reused purely
        // as a defensive fallback rather than adding a bespoke error variant for
        // an unreachable condition.
        RowIndex => {
            arg_count(node.op, 0, &args)?;
            let n_rows = handles
                .first()
                .and_then(|h| h.shape().first().copied())
                .ok_or(BuildError::UnsupportedOp(node.op))?;
            ffi(mlx_iota_f32(n_rows as i64))
        }
    }
}
