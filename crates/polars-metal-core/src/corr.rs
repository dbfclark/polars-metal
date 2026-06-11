//! M6 corr: GPU Pearson correlation matrix via one MLX subgraph.
//! C = Znᵀ·Zn where Zn = per-column-L2-normalized centered columns (the
//! (N−1) normalization cancels). Mirrors `vector_search.rs` for FFI idioms.

use std::sync::Arc;

use polars_metal_buffer::{MetalBuffer, MetalDevice};
use polars_metal_mlx_sys::array::{
    mlx_array_eval, mlx_array_to_f32_vec, mlx_array_view_metal_buffer, MlxArrayHandle, MlxDtype,
};
use polars_metal_mlx_sys::elementwise::{mlx_div, mlx_mul, mlx_sqrt, mlx_sub};
use polars_metal_mlx_sys::matmul::mlx_matmul;
use polars_metal_mlx_sys::reduce::{mlx_mean_axis, mlx_sum_axis};
use polars_metal_mlx_sys::shape::{mlx_reshape, mlx_transpose};
use polars_metal_mlx_sys::FfiError;

/// View a row-major (n, p) F32 slice as an MLX array. Borrows; the caller keeps
/// `data` alive until after `mlx_array_eval`. Mirrors vector_search::view2d.
fn view2d(data: &[f32], rows: i64, cols: i64) -> Result<MlxArrayHandle, FfiError> {
    let device = MetalDevice::system_default()
        .map_err(|e| FfiError::Runtime(format!("metal device unavailable: {e}")))?;
    // SAFETY: `data` outlives every use of the returned handle within this fn's callers,
    // which eval and read back before returning. MetalBuffer borrows, does not own.
    let buf = unsafe { MetalBuffer::from_borrowed_f32(&device, data.as_ptr(), data.len()) }
        .map(Arc::new)
        .map_err(|e| FfiError::Runtime(format!("corr staging: {e}")))?;
    mlx_array_view_metal_buffer(buf, &[rows, cols], MlxDtype::F32)
}

/// Pearson correlation matrix of a row-major (n, p) F32 matrix → p*p row-major F32.
///
/// Columns with zero variance produce NaN (norm = 0 → division by zero in the
/// per-column normalize), which matches Polars `df.corr()`. Callers that need to
/// avoid NaN must validate upstream; the engine routes null/degenerate inputs to
/// the CPU fallback (see the Python dispatch layer).
pub fn corr_matrix(data: &[f32], n: i64, p: i64) -> Result<Vec<f32>, FfiError> {
    // MLX reshape dims are i32; guard the (physically unreachable but type-unchecked)
    // p > i32::MAX case so a wide p can never silently wrap to a negative dim.
    if p > i64::from(i32::MAX) {
        return Err(FfiError::Runtime(format!(
            "corr: column count {p} exceeds i32::MAX"
        )));
    }
    let x = view2d(data, n, p)?; // (N,p)
    let mean = mlx_reshape(&mlx_mean_axis(&x, 0)?, &[1, p as i32])?; // (1,p)
    let xc = mlx_sub(&x, &mean)?; // (N,p) centered columns
    let colss = mlx_reshape(&mlx_sum_axis(&mlx_mul(&xc, &xc)?, 0)?, &[1, p as i32])?; // (1,p)
    let norm = mlx_sqrt(&colss)?; // (1,p) column L2 norms
    let zn = mlx_div(&xc, &norm)?; // (N,p) unit-norm columns
    let zt = mlx_transpose(&zn, &[1, 0])?; // (p,N)
    let c = mlx_matmul(&zt, &zn)?; // (p,p)
    mlx_array_eval(&[c.clone()])?;
    mlx_array_to_f32_vec(&c)
}

use pyo3::prelude::*;

/// PyO3 entry: (ptr,len) row-major (n,p) F32 → flat p*p F32 correlation matrix.
/// Mirrors vector_search::execute_vector_search's (ptr,len) ABI.
#[pyfunction]
pub fn execute_corr(data: (usize, usize), n: i64, p: i64) -> PyResult<Vec<f32>> {
    let (ptr, len) = data;
    if n < 0 || p < 0 || (n as usize).saturating_mul(p as usize) != len {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "polars_metal: corr dimension mismatch (n*p != len)",
        ));
    }
    // SAFETY: Python guarantees ptr addresses `len` contiguous live F32 (numpy
    // array kept alive across the call); read-only, no invalid f32 patterns.
    let slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, len) };
    corr_matrix(slice, n, p)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("corr: {e}")))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn corr_2x2_known() {
        // X (N=4, p=2), row-major: cols [1,2,3,4] and [2,1,4,3].
        // centered c0=[-1.5,-0.5,0.5,1.5], c1=[-0.5,-1.5,1.5,0.5];
        // dot=3.0, ||c0||=||c1||=sqrt(5); corr = 3/5 = 0.6.
        let data: Vec<f32> = vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0];
        let c = corr_matrix(&data, 4, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert!((c[0] - 1.0).abs() < 1e-5, "C[0,0]={}", c[0]);
        assert!((c[1] - 0.6).abs() < 1e-5, "C[0,1]={}", c[1]);
        assert!((c[2] - 0.6).abs() < 1e-5, "C[1,0]={}", c[2]);
        assert!((c[3] - 1.0).abs() < 1e-5, "C[1,1]={}", c[3]);
    }
}
