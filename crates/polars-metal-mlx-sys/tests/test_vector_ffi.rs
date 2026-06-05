//! M6 vector-search FFI building blocks + characterization.
//!
//! Task 0: pin MLX `argpartition` 2-D semantics. The whole vector-search
//! top-k design rests on (1) which axis argpartition partitions on a 2-D
//! array, and (2) the index dtype it returns. This test characterizes both.
//!
//! ## OBSERVED BEHAVIOR (MLX 0.22.0, this bridge) — READ BEFORE BUILDING TOP-K
//!
//! The current `mlx_argpartition` wrapper calls the C++ overload
//! `mlx::core::argpartition(const array& a, int kth)` (mlx/ops.h:704), which
//! **FLATTENS the input to 1-D before partitioning**. So on a 2-D (rows, cols)
//! array it does NOT partition each row along the last axis — it returns a
//! flat (rows*cols,) array of indices into the *flattened* (row-major) input,
//! partitioned globally.
//!
//! For the per-row top-k that vector search needs, later tasks MUST expose the
//! axis-aware overload `argpartition(const array& a, int kth, int axis)`
//! (mlx/ops.h:710) in the cxx bridge, e.g. `mlx_op_argpartition_axis(a, kth,
//! axis)` with `axis = -1` (last axis). The flattening overload is unusable for
//! per-query top-k.
//!
//! Index dtype: the result casts cleanly to F32 via `mlx_cast(.., F32)` (MLX
//! argpartition returns integer indices — uint32 in MLX 0.22.0); the I32→F32
//! cast in this test is exact for these small indices.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Arc;

use polars_metal_buffer::{MetalBuffer, MetalDevice};
use polars_metal_mlx_sys::array::{
    mlx_array_eval, mlx_array_to_f32_vec, mlx_array_view_metal_buffer, MlxArrayHandle, MlxDtype,
};
use polars_metal_mlx_sys::elementwise::mlx_cast;
use polars_metal_mlx_sys::sort::mlx_argpartition;

/// Build a 2-D (rows, cols) F32 MLX array from a row-major host slice.
fn arr2d(data: &[f32], rows: i64, cols: i64) -> MlxArrayHandle {
    let device = MetalDevice::system_default().expect("metal device");
    // SAFETY: `data` outlives the borrowed buffer within this call — the buffer
    // is consumed by the MLX view and fully eval'd before `data` (a caller-owned
    // slice that lives for the whole test) goes out of scope. On the non-page-
    // aligned copy path the borrow ends when `from_borrowed_f32` returns.
    let buf = unsafe { MetalBuffer::from_borrowed_f32(&device, data.as_ptr(), data.len()) }
        .map(Arc::new)
        .expect("metal buffer");
    mlx_array_view_metal_buffer(buf, &[rows, cols], MlxDtype::F32).expect("2d view")
}

/// Characterize how the CURRENT `mlx_argpartition` wrapper behaves on a 2-D
/// input. This PINS the observed behavior so later tasks know exactly what they
/// are building on (and that they must add an axis-aware overload for per-row
/// top-k — see the module doc).
#[test]
fn argpartition_2d_flattens_not_last_axis() {
    // Two rows; smallest value per row sits at a different column.
    // Row-major flat layout: [3, 5, 1, 2, 8, 9].
    //   Row 0 = [3, 5, 1] (min at col 2, flat idx 2)
    //   Row 1 = [2, 8, 9] (min at col 0, flat idx 3)
    // Global min over the flattened array is 1.0 at flat index 2.
    let data = [3.0f32, 5.0, 1.0, 2.0, 8.0, 9.0];
    let a = arr2d(&data, 2, 3);

    let idx = mlx_argpartition(&a, 0).expect("argpartition");
    let idx_f = mlx_cast(&idx, MlxDtype::F32).expect("cast");
    mlx_array_eval(&[idx_f.clone()]).expect("eval");

    // OBSERVED: the result is FLAT (shape [6]), NOT same-shape [2, 3]. The
    // current bridge overload flattens before partitioning.
    let shape = idx_f.shape();
    assert_eq!(
        shape,
        vec![6],
        "OBSERVED: argpartition flattens 2-D input to 1-D (shape [rows*cols]); \
         it does NOT partition along the last axis"
    );

    let v = mlx_array_to_f32_vec(&idx_f).expect("readback");
    assert_eq!(v.len(), 6);

    // kth=0 → position 0 holds the index (into the FLATTENED array) of the
    // single global minimum, i.e. flat index 2 (value 1.0). This is global
    // argmin, NOT per-row argmin.
    assert_eq!(
        v[0] as i32, 2,
        "OBSERVED: position 0 holds the GLOBAL argmin (flat index 2 = value 1.0), \
         not a per-row argmin"
    );
}

#[test]
fn transpose_2x3_to_3x2() {
    use polars_metal_mlx_sys::shape::mlx_transpose;

    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // (2,3) row-major
    let a = arr2d(&data, 2, 3);
    let t = mlx_transpose(&a, &[1, 0]).expect("transpose");
    mlx_array_eval(&[t.clone()]).expect("eval");
    assert_eq!(t.shape(), vec![3, 2]);
    // (3,2) row-major = columns of original = [1,4, 2,5, 3,6]
    assert_eq!(
        mlx_array_to_f32_vec(&t).unwrap(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

#[test]
fn reshape_6_to_3x2_and_keepdim() {
    use polars_metal_mlx_sys::shape::mlx_reshape;

    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = arr2d(&data, 1, 6);
    let r = mlx_reshape(&a, &[3, 2]).expect("reshape");
    mlx_array_eval(&[r.clone()]).expect("eval");
    assert_eq!(r.shape(), vec![3, 2]);
    // (N,) -> (N,1) keepdim case used by norm broadcasting:
    let n = arr2d(&[7.0, 8.0, 9.0], 1, 3);
    let col = mlx_reshape(&n, &[3, 1]).expect("reshape col");
    mlx_array_eval(&[col.clone()]).expect("eval");
    assert_eq!(col.shape(), vec![3, 1]);
}

#[test]
fn slice_first_2_cols_of_2x3() {
    use polars_metal_mlx_sys::shape::mlx_slice;

    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // (2,3)
    let a = arr2d(&data, 2, 3);
    // start=[0,0], stop=[2,2], strides=[1,1] -> (2,2) first two columns.
    let s = mlx_slice(&a, &[0, 0], &[2, 2], &[1, 1]).expect("slice");
    mlx_array_eval(&[s.clone()]).expect("eval");
    assert_eq!(s.shape(), vec![2, 2]);
    assert_eq!(mlx_array_to_f32_vec(&s).unwrap(), vec![1.0, 2.0, 4.0, 5.0]);
}
