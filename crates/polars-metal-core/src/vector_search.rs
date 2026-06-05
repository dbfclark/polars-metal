//! M6 vector search: MLX-composition top-k over a query×corpus GEMM.
use std::sync::Arc;

use polars_metal_buffer::{MetalBuffer, MetalDevice};
use polars_metal_mlx_sys::array::{
    mlx_array_eval, mlx_array_to_f32_vec, mlx_array_to_i32_vec, mlx_array_view_metal_buffer,
    MlxArrayHandle, MlxDtype,
};
use polars_metal_mlx_sys::elementwise::{mlx_add, mlx_cast, mlx_div, mlx_mul, mlx_neg, mlx_sqrt, mlx_sub};
use polars_metal_mlx_sys::matmul::mlx_matmul;
use polars_metal_mlx_sys::reduce::mlx_sum_axis;
use polars_metal_mlx_sys::shape::{mlx_reshape, mlx_slice, mlx_take_along_axis, mlx_transpose};
use polars_metal_mlx_sys::sort::mlx_argpartition_axis;
use polars_metal_mlx_sys::FfiError;

pub const OP_COSINE: u32 = 0;
pub const OP_KNN_L2: u32 = 1;

/// View a row-major host F32 slice as a 2-D `(rows, cols)` MLX array.
fn view2d(data: &[f32], rows: i64, cols: i64) -> Result<MlxArrayHandle, FfiError> {
    let device = MetalDevice::system_default()
        .map_err(|e| FfiError::Runtime(format!("metal device unavailable: {e}")))?;
    // SAFETY: `data` outlives every use of the returned handle within this fn's callers,
    // which eval and read back before returning. MetalBuffer borrows, does not own.
    let buf = unsafe { MetalBuffer::from_borrowed_f32(&device, data.as_ptr(), data.len()) }
        .map(Arc::new)
        .map_err(|e| FfiError::Runtime(format!("metal buffer staging: {e}")))?;
    mlx_array_view_metal_buffer(buf, &[rows, cols], MlxDtype::F32)
}

/// L2-normalize rows of `(rows, d)`: `x / sqrt(sum(x^2, axis=1))` (keepdim broadcast).
fn l2_normalize_rows(x: &MlxArrayHandle, rows: i32, _d: i32) -> Result<MlxArrayHandle, FfiError> {
    let sq = mlx_mul(x, x)?;
    let ss = mlx_sum_axis(&sq, 1)?; // (rows,)
    let ss = mlx_reshape(&ss, &[rows, 1])?; // (rows,1)
    let norm = mlx_sqrt(&ss)?;
    mlx_div(x, &norm) // (rows,d) / (rows,1) broadcasts
}

/// Compute unordered top-k. Returns `(indices (Q*k, i32), values (Q*k, f32))` row-major.
pub fn vector_search_topk(
    query: &[f32],
    q_rows: i64,
    corpus: &[f32],
    n_rows: i64,
    d: i64,
    k: i64,
    op: u32,
) -> Result<(Vec<i32>, Vec<f32>), FfiError> {
    let q = view2d(query, q_rows, d)?;
    let c = view2d(corpus, n_rows, d)?;

    // metric: (Q,N) similarity (cosine) or squared distance (knn).
    let (metric, partition_on) = match op {
        OP_COSINE => {
            let qn = l2_normalize_rows(&q, q_rows as i32, d as i32)?;
            let cn = l2_normalize_rows(&c, n_rows as i32, d as i32)?;
            let ct = mlx_transpose(&cn, &[1, 0])?; // (D,N)
            let sims = mlx_matmul(&qn, &ct)?; // (Q,N)
            let neg = mlx_neg(&sims)?; // argpartition picks SMALLEST → largest sims
            (sims, neg)
        }
        OP_KNN_L2 => {
            // d2 = q2 + c2 - 2 q·cᵀ  (broadcast (Q,1)+(1,N))
            let q2 = mlx_reshape(&mlx_sum_axis(&mlx_mul(&q, &q)?, 1)?, &[q_rows as i32, 1])?;
            let c2 = mlx_reshape(&mlx_sum_axis(&mlx_mul(&c, &c)?, 1)?, &[1, n_rows as i32])?;
            let ct = mlx_transpose(&c, &[1, 0])?; // (D,N)
            let cross = mlx_matmul(&q, &ct)?; // (Q,N)
            let two_cross = mlx_add(&cross, &cross)?;
            let d2 = mlx_sub(&mlx_add(&q2, &c2)?, &two_cross)?;
            (d2.clone(), d2) // knn partitions on the distance directly (smallest)
        }
        _ => return Err(FfiError::Runtime("unknown vector-search op".to_string())),
    };

    // argpartition along LAST axis (axis=-1) → (Q,N) indices; take first k columns.
    // NOTE: must use the axis-aware wrapper; the bare mlx_argpartition flattens to 1-D (Task 0).
    let part = mlx_argpartition_axis(&partition_on, (k - 1) as i32, -1)?;
    let idx_k = mlx_slice(&part, &[0, 0], &[q_rows as i32, k as i32], &[1, 1])?; // (Q,k)
    let idx_k_i = mlx_cast(&idx_k, MlxDtype::I32)?;
    // gather the metric values at those indices.
    let val_k = mlx_take_along_axis(&metric, &idx_k_i, 1)?; // (Q,k)

    mlx_array_eval(&[idx_k_i.clone(), val_k.clone()])?;
    let indices = mlx_array_to_i32_vec(&idx_k_i)?;
    let values = mlx_array_to_f32_vec(&val_k)?;
    Ok((indices, values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_topk_small() {
        // D=2. corpus rows: e0=[1,0], e1=[0,1], e2=[1,1]. query=[1,0].
        // cosine(query,·): e0=1.0, e1=0.0, e2=0.707. top-2 = {e0, e2}.
        let q = [1.0f32, 0.0]; // (1,2)
        let c = [1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0]; // (3,2)
        let (idx, score) = vector_search_topk(&q, 1, &c, 3, 2, /*k=*/ 2, OP_COSINE).unwrap();
        assert_eq!(idx.len(), 2);
        assert_eq!(score.len(), 2);
        // Sort the (unordered) result by score desc for a stable assertion.
        let mut pairs: Vec<(i32, f32)> = idx.iter().copied().zip(score.iter().copied()).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        assert_eq!(pairs[0].0, 0);
        assert!((pairs[0].1 - 1.0).abs() < 1e-5);
        assert_eq!(pairs[1].0, 2);
        assert!((pairs[1].1 - 0.70710677).abs() < 1e-5);
    }

    #[test]
    fn knn_l2_small() {
        // D=2. corpus e0=[0,0], e1=[3,4], e2=[1,0]. query=[0,0].
        // squared dists: e0=0, e1=25, e2=1. top-2 nearest = {e0, e2}.
        let q = [0.0f32, 0.0];
        let c = [0.0f32, 0.0, 3.0, 4.0, 1.0, 0.0];
        let (idx, d2) = vector_search_topk(&q, 1, &c, 3, 2, 2, OP_KNN_L2).unwrap();
        let mut pairs: Vec<(i32, f32)> = idx.iter().copied().zip(d2.iter().copied()).collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap()); // ascending (nearest)
        assert_eq!(pairs[0].0, 0);
        assert!(pairs[0].1.abs() < 1e-4); // squared distance 0
        assert_eq!(pairs[1].0, 2);
        assert!((pairs[1].1 - 1.0).abs() < 1e-4); // squared distance 1 (sqrt applied later, host)
    }
}
