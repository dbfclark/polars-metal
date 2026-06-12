//! M2's per-agg groupby path (conformance-only): the legacy fallback used when
//! the router picks `PerAgg` or the fused path rejects with
//! `NgroupsExceedsFusedCap`. Builds one `(AggRequest, ValueColumn)` per agg and
//! dispatches the per-agg kernel. Not extended — the fused path in
//! `super::execute_groupby` is the live one.

use std::collections::HashMap;

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::groupby::{AggKind, AggRequest, GroupByError, KeyColumn, ValueColumn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::plan::AggOp;

use super::{ParsedAgg, ParsedGroupByPlan};

/// Map `(op, dtype_tag)` → the kernel `AggKind`. `Count` maps to
/// `AggKind::Count` for every supported value dtype; the other ops carry the
/// dtype in their variant. Errors on an unsupported `(op, dtype)` pair.
fn agg_kind_for(op: AggOp, dtype_tag: &str) -> PyResult<AggKind> {
    let kind = match (op, dtype_tag) {
        (AggOp::Count, "I64" | "F64" | "I32" | "F32") => AggKind::Count,
        (AggOp::Sum, "I64") => AggKind::SumI64,
        (AggOp::Mean, "I64") => AggKind::MeanI64,
        (AggOp::Min, "I64") => AggKind::MinI64,
        (AggOp::Max, "I64") => AggKind::MaxI64,
        (AggOp::Sum, "F64") => AggKind::SumF64,
        (AggOp::Mean, "F64") => AggKind::MeanF64,
        (AggOp::Min, "F64") => AggKind::MinF64,
        (AggOp::Max, "F64") => AggKind::MaxF64,
        (AggOp::Sum, "I32") => AggKind::SumI32,
        (AggOp::Mean, "I32") => AggKind::MeanI32,
        (AggOp::Min, "I32") => AggKind::MinI32,
        (AggOp::Max, "I32") => AggKind::MaxI32,
        (AggOp::Sum, "F32") => AggKind::SumF32,
        (AggOp::Mean, "F32") => AggKind::MeanF32,
        (AggOp::Min, "F32") => AggKind::MinF32,
        (AggOp::Max, "F32") => AggKind::MaxF32,
        (op, dtype) => {
            return Err(PyValueError::new_err(format!(
                "polars_metal: unsupported (agg_op, dtype) combination: {op:?} / {dtype}"
            )))
        }
    };
    Ok(kind)
}

/// Buffer-check `data` (≥ `n_rows * size_of::<T>()` bytes) and reinterpret it
/// as `&[T]`. `T` must be a plain numeric scalar with no invalid bit patterns
/// (i32/i64/f32/f64); the byte buffer is an Arrow value buffer (64-byte
/// aligned). `tag` names the dtype for the error message.
fn checked_reinterpret<'a, T>(data: &'a [u8], n_rows: usize, tag: &str) -> PyResult<&'a [T]> {
    let expected = n_rows * std::mem::size_of::<T>();
    if data.len() < expected {
        return Err(PyValueError::new_err(format!(
            "polars_metal: {tag} value buffer too short: {got} < {expected}",
            got = data.len()
        )));
    }
    // SAFETY: `T` has no invalid bit patterns (numeric scalar); `data.len() >=
    // n_rows * size_of::<T>()` (checked above) and Arrow value buffers are
    // 64-byte aligned, so the reinterpret is well-aligned and in-bounds.
    Ok(unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, n_rows) })
}

/// Build the `(AggKind, ValueColumn)` for one aggregation over a value column.
/// Folds the former 21-arm `(op, dtype)` match into an `AggKind` lookup plus a
/// per-dtype reinterpret (M7 B-2). Supported value dtypes: I64/F64/I32/F32 —
/// the project does not extend groupby past these (conformance-only).
fn build_agg_kind_and_vcol<'a>(
    op: AggOp,
    dtype_tag: &str,
    data: &'a [u8],
    valid: &'a [u8],
    n_rows: usize,
) -> PyResult<(AggKind, ValueColumn<'a>)> {
    let kind = agg_kind_for(op, dtype_tag)?;
    // `agg_kind_for` already validated `(op, dtype_tag)`; the dtype is one of
    // I64/F64/I32/F32 here. The per-dtype arm picks the width + ValueColumn.
    let vc = match dtype_tag {
        "I64" => ValueColumn::I64 {
            data: checked_reinterpret::<i64>(data, n_rows, dtype_tag)?,
            valid,
        },
        "F64" => ValueColumn::F64 {
            data: checked_reinterpret::<f64>(data, n_rows, dtype_tag)?,
            valid,
        },
        "I32" => ValueColumn::I32 {
            data: checked_reinterpret::<i32>(data, n_rows, dtype_tag)?,
            valid,
        },
        "F32" => ValueColumn::F32 {
            data: checked_reinterpret::<f32>(data, n_rows, dtype_tag)?,
            valid,
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "polars_metal: unsupported value dtype: {other}"
            )))
        }
    };
    Ok((kind, vc))
}

/// M2's per-agg groupby path (conformance-only): builds one
/// `(AggRequest, ValueColumn)` per agg and dispatches the per-agg kernel.
/// Invoked when the router chose `PerAgg`, or when the fused path rejected with
/// `NgroupsExceedsFusedCap` on a query with no `Expression` aggs. The fused
/// path lives inline in `execute_groupby`; this is the legacy fallback.
pub(super) fn execute_peragg(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    key_cols: &[KeyColumn],
    parsed: &ParsedGroupByPlan,
    name_byte_refs: &HashMap<String, (String, &[u8], &[u8])>,
    n_rows: usize,
) -> PyResult<polars_metal_kernels::groupby::GroupByResult> {
    // Len uses a zero-length I64 placeholder; Simple looks up its single input
    // column from `name_byte_refs`.
    let empty_data: &[u8] = &[];
    let empty_valid: &[u8] = &[];
    // SAFETY: &[] cast to &[i64] is a zero-length slice — no pointer arithmetic
    // occurs and the pointer is non-null (a valid empty slice). Established
    // pattern for zero-length typed slices in this codebase.
    let empty_i64: &[i64] =
        unsafe { std::slice::from_raw_parts(empty_data.as_ptr() as *const i64, 0) };

    let mut agg_specs: Vec<(AggRequest, ValueColumn<'_>)> = Vec::with_capacity(parsed.aggs.len());
    for (i, agg) in parsed.aggs.iter().enumerate() {
        match agg {
            ParsedAgg::Length { .. } => {
                agg_specs.push((
                    AggRequest {
                        kind: AggKind::Len,
                        input_col_idx: i,
                    },
                    ValueColumn::I64 {
                        data: empty_i64,
                        valid: empty_valid,
                    },
                ));
            }
            ParsedAgg::Simple { input_col, op, .. } => {
                let (dtype_tag, data, valid) = name_byte_refs.get(input_col).ok_or_else(|| {
                    PyValueError::new_err(
                        "polars_metal: internal error: missing byte ref for Simple agg",
                    )
                })?;
                let (kind, vcol) = build_agg_kind_and_vcol(*op, dtype_tag, data, valid, n_rows)?;
                agg_specs.push((
                    AggRequest {
                        kind,
                        input_col_idx: i,
                    },
                    vcol,
                ));
            }
            ParsedAgg::Expression { .. } => {
                // Expression specs should never route here (the router decides
                // Fused above). Defensive guard.
                return Err(PyValueError::new_err(
                    "polars_metal: AggSpec::Expression routed to per-agg path; this is a routing bug",
                ));
            }
        }
    }

    polars_metal_kernels::groupby::dispatch_groupby(device, queue, key_cols, &agg_specs, n_rows)
        .map_err(groupby_err)
}

/// Convert the kernel `GroupByError` to a `PyErr`.
fn groupby_err(e: GroupByError) -> PyErr {
    PyValueError::new_err(format!("polars_metal: dispatch_groupby failed: {e}"))
}

#[cfg(test)]
mod build_agg_tests {
    //! M7 B-2 characterization: pin the full (op, dtype) → (AggKind, ValueColumn)
    //! matrix of `build_agg_kind_and_vcol` before the 21-arm fold, so the fold
    //! cannot silently break a dtype/agg arm. Asserts variant identity (not data).
    use super::*;

    /// Build zero buffers of the right width and assert the returned AggKind /
    /// ValueColumn variants for a supported (op, dtype) pair.
    macro_rules! check_ok {
        ($op:expr, $tag:expr, $width:expr, $kpat:pat, $vpat:pat) => {{
            let n = 4usize;
            let data = vec![0u8; n * $width];
            let valid = vec![0u8; n.div_ceil(8)];
            let (k, vc) = build_agg_kind_and_vcol($op, $tag, &data, &valid, n)
                .unwrap_or_else(|e| panic!("{:?}/{} should be supported: {e}", $op, $tag));
            assert!(matches!(k, $kpat), "{:?}/{}: wrong AggKind", $op, $tag);
            assert!(matches!(vc, $vpat), "{:?}/{}: wrong ValueColumn", $op, $tag);
        }};
    }

    #[test]
    fn dtype_agg_matrix_is_pinned() {
        // I64 (width 8)
        check_ok!(
            AggOp::Sum,
            "I64",
            8,
            AggKind::SumI64,
            ValueColumn::I64 { .. }
        );
        check_ok!(
            AggOp::Mean,
            "I64",
            8,
            AggKind::MeanI64,
            ValueColumn::I64 { .. }
        );
        check_ok!(
            AggOp::Min,
            "I64",
            8,
            AggKind::MinI64,
            ValueColumn::I64 { .. }
        );
        check_ok!(
            AggOp::Max,
            "I64",
            8,
            AggKind::MaxI64,
            ValueColumn::I64 { .. }
        );
        check_ok!(
            AggOp::Count,
            "I64",
            8,
            AggKind::Count,
            ValueColumn::I64 { .. }
        );
        // F64 (width 8)
        check_ok!(
            AggOp::Sum,
            "F64",
            8,
            AggKind::SumF64,
            ValueColumn::F64 { .. }
        );
        check_ok!(
            AggOp::Mean,
            "F64",
            8,
            AggKind::MeanF64,
            ValueColumn::F64 { .. }
        );
        check_ok!(
            AggOp::Min,
            "F64",
            8,
            AggKind::MinF64,
            ValueColumn::F64 { .. }
        );
        check_ok!(
            AggOp::Max,
            "F64",
            8,
            AggKind::MaxF64,
            ValueColumn::F64 { .. }
        );
        check_ok!(
            AggOp::Count,
            "F64",
            8,
            AggKind::Count,
            ValueColumn::F64 { .. }
        );
        // I32 (width 4)
        check_ok!(
            AggOp::Sum,
            "I32",
            4,
            AggKind::SumI32,
            ValueColumn::I32 { .. }
        );
        check_ok!(
            AggOp::Mean,
            "I32",
            4,
            AggKind::MeanI32,
            ValueColumn::I32 { .. }
        );
        check_ok!(
            AggOp::Min,
            "I32",
            4,
            AggKind::MinI32,
            ValueColumn::I32 { .. }
        );
        check_ok!(
            AggOp::Max,
            "I32",
            4,
            AggKind::MaxI32,
            ValueColumn::I32 { .. }
        );
        check_ok!(
            AggOp::Count,
            "I32",
            4,
            AggKind::Count,
            ValueColumn::I32 { .. }
        );
        // F32 (width 4)
        check_ok!(
            AggOp::Sum,
            "F32",
            4,
            AggKind::SumF32,
            ValueColumn::F32 { .. }
        );
        check_ok!(
            AggOp::Mean,
            "F32",
            4,
            AggKind::MeanF32,
            ValueColumn::F32 { .. }
        );
        check_ok!(
            AggOp::Min,
            "F32",
            4,
            AggKind::MinF32,
            ValueColumn::F32 { .. }
        );
        check_ok!(
            AggOp::Max,
            "F32",
            4,
            AggKind::MaxF32,
            ValueColumn::F32 { .. }
        );
        check_ok!(
            AggOp::Count,
            "F32",
            4,
            AggKind::Count,
            ValueColumn::F32 { .. }
        );
    }

    #[test]
    fn buffer_too_short_errors() {
        // n=4 i64 needs 32 bytes; give 8.
        let short = vec![0u8; 8];
        let valid = vec![0u8; 1];
        assert!(build_agg_kind_and_vcol(AggOp::Sum, "I64", &short, &valid, 4).is_err());
    }

    #[test]
    fn unsupported_dtype_errors() {
        let data = vec![0u8; 8];
        let valid = vec![0u8; 1];
        // Bool is not a supported agg value dtype.
        assert!(build_agg_kind_and_vcol(AggOp::Sum, "Bool", &data, &valid, 1).is_err());
    }
}
