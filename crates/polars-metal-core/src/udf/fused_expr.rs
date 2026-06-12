//! `execute_fused_expr` PyO3 entry point: evaluates a built MLX subgraph
//! (`PyFusionScope`) over a list of zero-copy, dtype-polymorphic input column
//! buffers and writes the result into the caller's output buffer in place.

use polars_metal_buffer::MetalDevice;
use pyo3::prelude::*;

// ── M4 Phase 5 Task 21+22: execute_fused_expr ───────────────────────────────
//
// PyO3 entry point that takes a PyFusionScope plus a list of input column
// byte buffers (each F32-typed) and returns the output as a PyBytes byte
// buffer. The Python wrapper in `_udf.py` converts Polars Series ↔ bytes
// since we don't carry a `pyo3-polars` dep.
//
// Deviation from the plan: the plan called for a `PyMetalPlanNode::FusedExprGraph`
// wrapper sitting on top of the Rust `MetalPlanNode` enum. We don't have a
// PyO3 wrapper for that enum - the Python side talks to Rust via wire-plan
// dicts. We expose the executor directly; the walker stashes the
// PyFusionScope as a side-channel on the binding and the UDF dispatch
// (Task 23) invokes this entry point.

/// Execute a fused MLX subgraph with zero-copy, dtype-polymorphic I/O.
///
/// `inputs` is a list of `(ptr, n_elements, dtype_tag)` triples, one per scope
/// input, where `ptr` is the address of a live, C-contiguous buffer (a numpy
/// array's `__array_interface__` data pointer) holding `n_elements` of the
/// `MlxDtype` named by `dtype_tag`. `out` is `(ptr, capacity_elements,
/// dtype_tag)` for a writable array to receive the single subgraph output;
/// `dtype_tag` is the analyzer's statically-inferred output dtype, so Python
/// pre-allocates the right-width array. Returns the number of elements written
/// (1 for a literal-only / scalar output, else the column length).
///
/// The buffer protocol isn't available under the abi3 limited API, so we pass
/// raw pointers from Python rather than `PyBuffer`. The caller MUST keep every
/// input array and the output array alive for the duration of this (fully
/// synchronous) call; `_udf._dispatch_hstack_fused` does so by holding the
/// arrays in locals across the call.
///
/// After eval, the Rust side asserts the eval'd dtype equals the declared
/// output tag (analyzer mis-inference guard) before any width-aware write — a
/// hard error rather than silently corrupting the caller's bytes.
#[pyfunction]
pub fn execute_fused_expr(
    scope: &crate::fusion::py::PyFusionScope,
    inputs: Vec<(usize, usize, u32)>,
    out: (usize, usize, u32),
) -> PyResult<usize> {
    use polars_metal_mlx_sys::array::MlxDtype;
    use std::sync::Arc;
    let device = MetalDevice::system_default().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "polars_metal: metal device unavailable: {e}"
        ))
    })?;

    // Stage each input buffer into a MetalBuffer at its native width. The
    // byte-level borrow is zero-copy when the pointer is page-aligned
    // (numpy-origin / large allocations), else a single copy — handled inside
    // `from_borrowed_bytes`. The caller keeps the source arrays alive for the
    // whole call, so the borrowed memory outlives every MetalBuffer and the
    // synchronous eval.
    let metal_buffers: Vec<Arc<polars_metal_buffer::MetalBuffer>> = inputs
        .iter()
        .map(|&(ptr, n, tag)| {
            let dtype = MlxDtype::from_tag(tag).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "polars_metal: bad input dtype tag: {e:?}"
                ))
            })?;
            // SAFETY: `ptr` addresses `n` live, contiguous elements of `dtype`
            // (caller contract) for the lifetime of this call; the byte length
            // is `n * element_size`. Page-aligned pointers take bytesNoCopy,
            // others copy. See `from_borrowed_bytes`.
            let buf = unsafe {
                polars_metal_buffer::MetalBuffer::from_borrowed_bytes(
                    &device,
                    ptr as *const u8,
                    n * dtype.element_size(),
                )
            }
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "polars_metal: input buffer staging failed: {e}"
                ))
            })?;
            Ok(Arc::new(buf))
        })
        .collect::<PyResult<Vec<_>>>()?;

    let subgraph = crate::fusion::subgraph::MlxSubgraph::from_fusion_scope_buffers(
        &scope.inner,
        &metal_buffers,
    )
    .map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("polars_metal: subgraph build: {e}"))
    })?;

    // Output-zero-copy: eval writes directly into the caller's `out` array,
    // interpreting it at the analyzer-declared dtype.
    let (out_ptr, out_cap, out_tag) = out;
    let out_dtype = MlxDtype::from_tag(out_tag).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "polars_metal: bad output dtype tag: {e:?}"
        ))
    })?;
    // SAFETY: `out_ptr` addresses `out_cap` writable, contiguous elements of
    // `out_dtype` (caller contract), kept alive for the whole call.
    // `eval_into_typed` blocks on `mlx_array_eval` before copying (no in-flight
    // GPU writes on readback), validates the dtype matches, and bounds-checks
    // against `out_cap`.
    subgraph
        .eval_into_typed(out_ptr, out_cap, out_dtype)
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("polars_metal: subgraph eval: {e}"))
        })
}
