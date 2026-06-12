//! GroupBy execution (conformance-only per the Mission non-goals): plan
//! parsing, fused-vs-per-agg routing, the fused-library cache + warmup, key /
//! value-column building, output encoding, and the `execute_groupby` PyO3
//! entry point. Maintained for correctness on groupby-shaped plans; not
//! extended. The core/legacy split + `build_agg_kind_and_vcol` fold land in B-2.

use crate::plan::{AggExpr, AggOp, BinaryOp, MetalDtype};
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::aggregate_fused::cache::FusedLibraryCache;
use polars_metal_kernels::aggregate_fused::signature::{
    AggExpr as KAggExpr, AggOp as KAggOp, AggSpec as KAggSpec, BinaryOp as KBinaryOp,
    MetalDtype as KMetalDtype,
};
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::groupby::{dispatch_groupby_fused, KeyColumn, KeyDtype, ValueColumn};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

use super::common::pack_valid_bitmap;

mod legacy;

// ----------------------------------------------------------------------------
// IR → kernel-layer mirrors (Phase 3 / Task 15)
// ----------------------------------------------------------------------------

/// Convert IR `AggOp` → kernel-layer `AggOp` mirror.
fn convert_agg_op(op: AggOp) -> KAggOp {
    match op {
        AggOp::Sum => KAggOp::Sum,
        AggOp::Mean => KAggOp::Mean,
        AggOp::Count => KAggOp::Count,
        AggOp::Min => KAggOp::Min,
        AggOp::Max => KAggOp::Max,
        AggOp::Len => KAggOp::Len,
    }
}

/// Convert IR `BinaryOp` → kernel-layer mirror.
fn convert_binary_op(op: BinaryOp) -> KBinaryOp {
    match op {
        BinaryOp::Add => KBinaryOp::Add,
        BinaryOp::Sub => KBinaryOp::Sub,
        BinaryOp::Mul => KBinaryOp::Mul,
        BinaryOp::Div => KBinaryOp::Div,
    }
}

/// Convert IR `AggExpr` → kernel-layer mirror (mechanical tree walk).
fn convert_agg_expr(expr: &AggExpr) -> KAggExpr {
    match expr {
        AggExpr::Column(name) => KAggExpr::Column(name.clone()),
        AggExpr::LiteralF64(v) => KAggExpr::LiteralF64(*v),
        AggExpr::LiteralI64(v) => KAggExpr::LiteralI64(*v),
        AggExpr::Binary { op, lhs, rhs } => KAggExpr::Binary {
            op: convert_binary_op(*op),
            lhs: Box::new(convert_agg_expr(lhs)),
            rhs: Box::new(convert_agg_expr(rhs)),
        },
    }
}

/// Wire dtype tag (`"I32"` / `"F32"` / ...) → kernel-layer `MetalDtype`.
fn wire_dtype_tag_to_kernel(tag: &str) -> Option<KMetalDtype> {
    match tag {
        "I64" => Some(KMetalDtype::I64),
        "F64" => Some(KMetalDtype::F64),
        "Bool" => Some(KMetalDtype::Bool),
        "I32" => Some(KMetalDtype::I32),
        "F32" => Some(KMetalDtype::F32),
        "I8" => Some(KMetalDtype::I8),
        "I16" => Some(KMetalDtype::I16),
        "U8" => Some(KMetalDtype::U8),
        "U16" => Some(KMetalDtype::U16),
        "U32" => Some(KMetalDtype::U32),
        // M3 Phase 7: Utf8 is a key dtype only — never a valid agg value
        // column. Returning None here forces `decide_groupby_dispatch` down
        // the PerAgg branch (or further fallback) if a router bug ever lifts
        // a Utf8 column into agg-input position.
        "Utf8" | "String" => None,
        _ => None,
    }
}

/// Routing decision for one groupby query: fused single-kernel dispatch
/// vs M2's per-agg loop. See `decide_groupby_dispatch` for the rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroupByDispatchChoice {
    Fused,
    PerAgg,
}

/// Decide between the fused kernel and M2's per-agg path.
///
/// Rules (Task 15):
///   1. If any agg is Expression → Fused (Expression has no per-agg fallback).
///   2. Otherwise, if every agg is Simple/Length AND there are ≥ 2 aggs
///      AND the signature is fused-supported (all 32-bit-or-narrower inputs)
///      → Fused.
///   3. Otherwise → PerAgg (single-agg queries, F64/I64 inputs, etc.).
///
/// Caller must have already verified the value columns referenced by the
/// aggs are present in the HashMap. The signature is built once here and
/// inspected for support; the same signature is reused at dispatch time.
fn decide_groupby_dispatch(
    parsed: &[ParsedAgg],
    by_name: &HashMap<String, (String, &[u8], &[u8])>,
) -> GroupByDispatchChoice {
    let has_expression = parsed
        .iter()
        .any(|a| matches!(a, ParsedAgg::Expression { .. }));
    let n_simple_or_len = parsed
        .iter()
        .filter(|a| !matches!(a, ParsedAgg::Expression { .. }))
        .count();

    // Check that every referenced column is 32-bit-or-narrower (fused-only
    // supports that). Build a tentative signature inline.
    let mut col_dtypes: BTreeMap<String, KMetalDtype> = BTreeMap::new();
    let mut all_fused_supported = true;
    for a in parsed {
        match a {
            ParsedAgg::Simple { input_col, .. } => {
                let Some((dt_tag, _, _)) = by_name.get(input_col) else {
                    return GroupByDispatchChoice::PerAgg;
                };
                let Some(kdt) = wire_dtype_tag_to_kernel(dt_tag) else {
                    return GroupByDispatchChoice::PerAgg;
                };
                if matches!(kdt, KMetalDtype::F64 | KMetalDtype::I64) {
                    all_fused_supported = false;
                }
                col_dtypes.entry(input_col.clone()).or_insert(kdt);
            }
            ParsedAgg::Expression { expr, .. } => {
                for c in expr.referenced_columns() {
                    let Some((dt_tag, _, _)) = by_name.get(&c) else {
                        return GroupByDispatchChoice::PerAgg;
                    };
                    let Some(kdt) = wire_dtype_tag_to_kernel(dt_tag) else {
                        return GroupByDispatchChoice::PerAgg;
                    };
                    if matches!(kdt, KMetalDtype::F64 | KMetalDtype::I64) {
                        all_fused_supported = false;
                    }
                    col_dtypes.entry(c).or_insert(kdt);
                }
            }
            ParsedAgg::Length { .. } => {}
        }
    }

    if has_expression && all_fused_supported {
        return GroupByDispatchChoice::Fused;
    }
    if !has_expression && n_simple_or_len >= 2 && all_fused_supported {
        return GroupByDispatchChoice::Fused;
    }
    GroupByDispatchChoice::PerAgg
}

/// Process-wide fused-library cache. Constructed lazily on first dispatch
/// when the system default Metal device is acquirable.
static FUSED_CACHE: OnceLock<FusedLibraryCache> = OnceLock::new();

fn get_or_init_fused_cache(device: &MetalDevice) -> &'static FusedLibraryCache {
    FUSED_CACHE.get_or_init(|| FusedLibraryCache::new(device.clone()))
}

/// Pre-compile common fused-agg signatures into the process-wide
/// `FUSED_CACHE` (Task 18). Called from `python/polars_metal/__init__.py`
/// at import time so the first user query of a common shape
/// (single-column F32 Sum, Q1-shape 10-agg, etc.) does not pay the MSL
/// compile cost.
///
/// Best-effort: if the Metal device cannot be acquired (no Metal-capable
/// hardware), or any individual signature fails to compile, the warmup
/// returns the number of signatures actually queued (0 on device failure).
/// The Python wrapper swallows exceptions too — warmup is advisory and
/// must not break engine startup.
///
/// Returns the count of signatures the cache was asked to warm; the
/// Python side uses this for logging and the integration test.
#[pyfunction]
pub fn warmup_common_fused_signatures() -> i32 {
    use polars_metal_kernels::aggregate_fused::cache::common_signatures;

    let Ok(device) = MetalDevice::system_default() else {
        // No Metal device available — running under a non-Metal harness
        // (e.g. CI without a GPU). Warmup is a no-op; skip without error.
        return 0;
    };
    let cache = get_or_init_fused_cache(&device);
    let sigs = common_signatures();
    let count = sigs.len() as i32;
    cache.warmup(&sigs);
    count
}

// ============================================================================
// GroupBy PyO3 entry point — T28
// ============================================================================

/// Parsed view of a GroupBy plan dict received from the Python UDF.
#[derive(Debug)]
pub struct ParsedGroupByPlan {
    pub keys: Vec<ParsedKey>,
    pub aggs: Vec<ParsedAgg>,
}

/// One key column descriptor from the wire plan.
#[derive(Debug)]
pub struct ParsedKey {
    pub name: String,
    pub dtype: MetalDtype,
}

/// One aggregation descriptor from the wire plan. Mirrors the
/// [`crate::plan::AggSpec`] enum (Simple / Expression / Length).
#[derive(Debug, Clone)]
pub enum ParsedAgg {
    Simple {
        input_col: String,
        op: AggOp,
        output_alias: String,
    },
    Expression {
        expr: AggExpr,
        op: AggOp,
        output_alias: String,
    },
    Length {
        output_alias: String,
    },
}

impl ParsedAgg {
    /// Convenience: the output alias regardless of variant. Every variant
    /// carries one; dispatch reads this for result-column naming.
    pub fn output_alias(&self) -> &str {
        match self {
            ParsedAgg::Simple { output_alias, .. }
            | ParsedAgg::Expression { output_alias, .. }
            | ParsedAgg::Length { output_alias } => output_alias,
        }
    }
}

/// Errors produced while parsing the Python groupby plan dict.
#[derive(Debug, thiserror::Error)]
pub enum GroupByParseError {
    #[error("missing required field: {0}")]
    Missing(&'static str),
    #[error("wrong type for field: {0}")]
    WrongType(&'static str),
    #[error("unknown dtype: {0}")]
    UnknownDtype(String),
    #[error("unknown agg op: {0}")]
    UnknownOp(String),
}

/// Recursively parse one `{"kind": ..., ...}` dict emitted by the Python
/// walker's expression extractor into an [`AggExpr`].
///
/// The accepted shapes mirror `_walk_agg_expr_node` in `_walker.py`:
/// - `{"kind": "Column", "name": str}` → [`AggExpr::Column`]
/// - `{"kind": "LiteralF64", "value": float}` → [`AggExpr::LiteralF64`]
/// - `{"kind": "LiteralI64", "value": int}` → [`AggExpr::LiteralI64`]
/// - `{"kind": "Binary", "op": "Add"|"Sub"|"Mul"|"Div",
///       "lhs": <expr dict>, "rhs": <expr dict>}` → [`AggExpr::Binary`]
///
/// Unknown kinds or unknown binary ops produce
/// [`GroupByParseError::UnknownOp`]; missing or wrongly-typed fields
/// produce [`GroupByParseError::WrongType`].
fn parse_agg_expr_dict(d: &Bound<PyDict>) -> Result<AggExpr, GroupByParseError> {
    let kind: String = d
        .get_item("kind")
        .ok()
        .flatten()
        .and_then(|v| v.extract().ok())
        .ok_or(GroupByParseError::WrongType("expr.kind"))?;
    match kind.as_str() {
        "Column" => {
            let name: String = d
                .get_item("name")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .ok_or(GroupByParseError::WrongType("expr.name"))?;
            Ok(AggExpr::Column(name))
        }
        "LiteralF64" => {
            let v: f64 = d
                .get_item("value")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .ok_or(GroupByParseError::WrongType("expr.value(f64)"))?;
            Ok(AggExpr::LiteralF64(v))
        }
        "LiteralI64" => {
            let v: i64 = d
                .get_item("value")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .ok_or(GroupByParseError::WrongType("expr.value(i64)"))?;
            Ok(AggExpr::LiteralI64(v))
        }
        "Binary" => {
            let op_str: String = d
                .get_item("op")
                .ok()
                .flatten()
                .and_then(|v| v.extract().ok())
                .ok_or(GroupByParseError::WrongType("expr.op"))?;
            let op = match op_str.as_str() {
                "Add" => BinaryOp::Add,
                "Sub" => BinaryOp::Sub,
                "Mul" => BinaryOp::Mul,
                "Div" => BinaryOp::Div,
                _ => return Err(GroupByParseError::UnknownOp(format!("binary op {op_str}"))),
            };
            let lhs_dict: Bound<PyDict> = d
                .get_item("lhs")
                .ok()
                .flatten()
                .ok_or(GroupByParseError::WrongType("expr.lhs"))?
                .downcast_into()
                .map_err(|_| GroupByParseError::WrongType("expr.lhs(dict)"))?;
            let rhs_dict: Bound<PyDict> = d
                .get_item("rhs")
                .ok()
                .flatten()
                .ok_or(GroupByParseError::WrongType("expr.rhs"))?
                .downcast_into()
                .map_err(|_| GroupByParseError::WrongType("expr.rhs(dict)"))?;
            Ok(AggExpr::Binary {
                op,
                lhs: Box::new(parse_agg_expr_dict(&lhs_dict)?),
                rhs: Box::new(parse_agg_expr_dict(&rhs_dict)?),
            })
        }
        other => Err(GroupByParseError::UnknownOp(format!("expr kind={other}"))),
    }
}

/// Parse the `plan_dict` PyDict emitted by the Python walker into a
/// [`ParsedGroupByPlan`]. No new Python dep required — we read the dict
/// directly via PyO3.
///
/// Expected shape:
/// ```python
/// {
///     "keys": [["col_name", "I64"], ...],
///     "aggs": [{"input_col": "x", "op": "Sum", "output_alias": "x_sum"}, ...],
/// }
/// ```
pub fn parse_groupby_plan(plan: &Bound<PyDict>) -> Result<ParsedGroupByPlan, GroupByParseError> {
    // -- keys ----------------------------------------------------------------
    let keys_obj = plan
        .get_item("keys")
        .ok()
        .flatten()
        .ok_or(GroupByParseError::Missing("keys"))?;
    let keys_list: Bound<PyList> = keys_obj
        .downcast_into()
        .map_err(|_| GroupByParseError::WrongType("keys"))?;
    let mut keys = Vec::with_capacity(keys_list.len());
    for item in keys_list.iter() {
        let entry: Bound<PyList> = item
            .downcast_into()
            .map_err(|_| GroupByParseError::WrongType("key entry"))?;
        if entry.len() < 2 {
            return Err(GroupByParseError::WrongType("key entry"));
        }
        let name: String = entry
            .get_item(0)
            .ok()
            .and_then(|v| v.extract().ok())
            .ok_or(GroupByParseError::WrongType("key name"))?;
        let dtype_str: String = entry
            .get_item(1)
            .ok()
            .and_then(|v| v.extract().ok())
            .ok_or(GroupByParseError::WrongType("key dtype"))?;
        let dtype =
            MetalDtype::from_wire(&dtype_str).ok_or(GroupByParseError::UnknownDtype(dtype_str))?;
        keys.push(ParsedKey { name, dtype });
    }

    // -- aggs ----------------------------------------------------------------
    let aggs_obj = plan
        .get_item("aggs")
        .ok()
        .flatten()
        .ok_or(GroupByParseError::Missing("aggs"))?;
    let aggs_list: Bound<PyList> = aggs_obj
        .downcast_into()
        .map_err(|_| GroupByParseError::WrongType("aggs"))?;
    let mut aggs = Vec::with_capacity(aggs_list.len());
    for item in aggs_list.iter() {
        let entry: Bound<PyDict> = item
            .downcast_into()
            .map_err(|_| GroupByParseError::WrongType("agg entry"))?;

        // Backwards-compatible read: missing "kind" means M2-shape Simple/Length
        // (the existing wire format). Explicit "kind" means M3-shape; the
        // "Expression" arm requires an "expr" sub-dict whose parser lands
        // in Task 9 (Phase 2 Task 8 leaves it as a stub error).
        let kind: String = entry
            .get_item("kind")
            .ok()
            .flatten()
            .and_then(|v| v.extract().ok())
            .unwrap_or_else(|| {
                // Legacy shape: infer Length from op=="Len", Simple otherwise.
                let op_str: String = entry
                    .get_item("op")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or_default();
                if op_str == "Len" {
                    "Length".into()
                } else {
                    "Simple".into()
                }
            });

        let output_alias: String = entry
            .get_item("output_alias")
            .ok()
            .flatten()
            .and_then(|v| v.extract().ok())
            .ok_or(GroupByParseError::WrongType("output_alias"))?;

        let parsed = match kind.as_str() {
            "Length" => ParsedAgg::Length { output_alias },
            "Simple" => {
                // input_col is empty string for Len legacy; default empty if absent.
                let input_col: String = entry
                    .get_item("input_col")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .unwrap_or_default();
                let op_str: String = entry
                    .get_item("op")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .ok_or(GroupByParseError::WrongType("op"))?;
                let op = AggOp::from_wire(&op_str).ok_or(GroupByParseError::UnknownOp(op_str))?;
                ParsedAgg::Simple {
                    input_col,
                    op,
                    output_alias,
                }
            }
            "Expression" => {
                // Capability G (M3 Phase 2): the walker emits a recursive
                // AggExpr sub-tree under `expr` plus the outer reducer (`op`)
                // and alias. Parse the sub-tree, then re-validate the depth
                // cap on the Rust side as defence-in-depth (the walker's
                // `_AGG_EXPR_MAX_DEPTH` is the primary gate).
                let op_str: String = entry
                    .get_item("op")
                    .ok()
                    .flatten()
                    .and_then(|v| v.extract().ok())
                    .ok_or(GroupByParseError::WrongType("op"))?;
                let op = AggOp::from_wire(&op_str).ok_or(GroupByParseError::UnknownOp(op_str))?;
                let expr_dict: Bound<PyDict> = entry
                    .get_item("expr")
                    .ok()
                    .flatten()
                    .ok_or(GroupByParseError::WrongType("expr"))?
                    .downcast_into()
                    .map_err(|_| GroupByParseError::WrongType("expr(dict)"))?;
                let expr = parse_agg_expr_dict(&expr_dict)?;
                expr.validate()
                    .map_err(|_| GroupByParseError::WrongType("expr(too deep)"))?;
                ParsedAgg::Expression {
                    expr,
                    op,
                    output_alias,
                }
            }
            other => {
                return Err(GroupByParseError::UnknownOp(format!("kind={other}")));
            }
        };
        aggs.push(parsed);
    }

    Ok(ParsedGroupByPlan { keys, aggs })
}

/// Map a [`MetalDtype`] (plan layer) to the kernel-layer [`KeyDtype`].
///
/// Returns `Err` for dtypes that have no groupby-key encoding. The only such
/// case today is `U64`: the composite-key encoder has no 64-bit-unsigned
/// `KeyDtype`, and the groupby kernel is conformance-only (Non-goals — not
/// extended). The Python walker already gates U64 keys to CPU fallback, so
/// this arm is defensive: if a router bug ever lifts a U64-key groupby, we
/// surface a clear error rather than mis-encoding or panicking.
fn metal_dtype_to_key_dtype(d: MetalDtype) -> Result<KeyDtype, String> {
    Ok(match d {
        MetalDtype::I64 => KeyDtype::I64,
        MetalDtype::F64 => KeyDtype::F64,
        MetalDtype::Bool => KeyDtype::Bool,
        MetalDtype::I32 => KeyDtype::I32,
        MetalDtype::F32 => KeyDtype::F32,
        MetalDtype::I8 => KeyDtype::I8,
        MetalDtype::I16 => KeyDtype::I16,
        MetalDtype::U8 => KeyDtype::U8,
        MetalDtype::U16 => KeyDtype::U16,
        MetalDtype::U32 => KeyDtype::U32,
        MetalDtype::Utf8 => KeyDtype::Utf8,
        MetalDtype::U64 => {
            return Err(
                "groupby key dtype UInt64 has no composite-key encoding (groupby is \
                 conformance-only and not extended); should route CPU at the walker"
                    .to_string(),
            )
        }
    })
}

/// Build the `(AggKind, ValueColumn<'a>)` pair for a single agg request,
/// given the value column's raw byte buffers and dtype tag.
///
/// The `data` slice must be cast to the correct typed slice before being
/// wrapped in `ValueColumn`. We use `unsafe slice::from_raw_parts` here
/// for the same reason as the filter path: no `bytemuck` dep, and Arrow
/// buffers are guaranteed to be at least 8-byte aligned.
/// Construct a typed `ValueColumn` view over raw bytes for the fused
/// groupby dispatcher.
///
/// Unlike [`build_agg_kind_and_vcol`] this does not derive a kernel-side
/// `AggKind` — the fused dispatcher derives output shape from the
/// `AggSignature`. Only the 32-bit-or-narrower dtypes the fused emitter
/// supports are accepted; F64/I64 callers must route through the M2 path.
fn build_value_column<'a>(
    dtype_tag: &str,
    data: &'a [u8],
    valid: &'a [u8],
    n_rows: usize,
) -> Result<ValueColumn<'a>, String> {
    match dtype_tag {
        "I32" => {
            let expected = n_rows * 4;
            if data.len() < expected {
                return Err(format!(
                    "I32 data buffer too short: {got} < {expected}",
                    got = data.len()
                ));
            }
            // SAFETY: i32 has no invalid bit patterns; Arrow buffers are
            // 64-byte aligned so the reinterpret meets the 4-byte alignment.
            let typed: &[i32] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const i32, n_rows) };
            Ok(ValueColumn::I32 { data: typed, valid })
        }
        "F32" => {
            let expected = n_rows * 4;
            if data.len() < expected {
                return Err(format!(
                    "F32 data buffer too short: {got} < {expected}",
                    got = data.len()
                ));
            }
            // SAFETY: f32 has no invalid bit patterns; same alignment as I32.
            let typed: &[f32] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n_rows) };
            Ok(ValueColumn::F32 { data: typed, valid })
        }
        other => Err(format!(
            "dtype {other} not supported by fused groupby (only I32/F32 currently)"
        )),
    }
}

/// Encode a [`DecodedColumn`] (key output) into `(dtype_tag, data_bytes,
/// valid_bytes)` for the wire return format.
fn encode_decoded_column(
    d: &polars_metal_kernels::groupby::DecodedColumn,
) -> (&'static str, Vec<u8>, Vec<u8>) {
    use polars_metal_kernels::groupby::DecodedColumn;
    match d {
        DecodedColumn::I64 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("I64", data, v)
        }
        DecodedColumn::F64 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("F64", data, v)
        }
        DecodedColumn::Bool { values, valid } => {
            // Bool data is also bit-packed (Arrow convention).
            let data = pack_valid_bitmap(values);
            let v = pack_valid_bitmap(valid);
            ("Bool", data, v)
        }
        DecodedColumn::I32 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("I32", data, v)
        }
        DecodedColumn::F32 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("F32", data, v)
        }
        DecodedColumn::I8 { values, valid } => {
            let data: Vec<u8> = values.iter().map(|v| *v as u8).collect();
            let v = pack_valid_bitmap(valid);
            ("I8", data, v)
        }
        DecodedColumn::I16 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("I16", data, v)
        }
        DecodedColumn::U8 { values, valid } => {
            let data: Vec<u8> = values.to_vec();
            let v = pack_valid_bitmap(valid);
            ("U8", data, v)
        }
        DecodedColumn::U16 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("U16", data, v)
        }
        DecodedColumn::U32 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("U32", data, v)
        }
        DecodedColumn::Utf8 { values, valid } => {
            // Wire format (parsed Python-side in Task 34):
            //   [n_rows: u32 le]
            //   [offsets: (n_rows+1) × i32 le]   Arrow Utf8 offset convention
            //   [concatenated string bytes]
            let n = values.len() as u32;
            let mut data: Vec<u8> = Vec::new();
            data.extend_from_slice(&n.to_le_bytes());
            let mut offsets: Vec<i32> = Vec::with_capacity(values.len() + 1);
            let mut acc: i32 = 0;
            offsets.push(0);
            let mut bytes_blob: Vec<u8> = Vec::new();
            for (s, &is_valid) in values.iter().zip(valid.iter()) {
                if is_valid {
                    bytes_blob.extend_from_slice(s.as_bytes());
                    acc = acc.saturating_add(s.len() as i32);
                }
                offsets.push(acc);
            }
            for o in &offsets {
                data.extend_from_slice(&o.to_le_bytes());
            }
            data.extend_from_slice(&bytes_blob);
            let v = pack_valid_bitmap(valid);
            ("Utf8", data, v)
        }
    }
}

/// Encode an [`AggOutput`] into `(dtype_tag, data_bytes, valid_bytes)`.
///
/// U64 outputs (Count, Len) are cast to u32 — Polars returns u32 for both
/// `pl.col(x).count()` and `pl.len()`. At M2 row counts > 4 billion per
/// group are unrealistic, so this truncation is safe in practice.
fn encode_agg_output(
    o: &polars_metal_kernels::groupby::AggOutput,
) -> (&'static str, Vec<u8>, Vec<u8>) {
    use polars_metal_kernels::groupby::AggOutput;
    match o {
        AggOutput::I64 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("I64", data, v)
        }
        AggOutput::F64 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("F64", data, v)
        }
        AggOutput::U64 { values } => {
            // Cast u64 → u32. Counts / lens that fit in u32 are the common case.
            let data: Vec<u8> = values
                .iter()
                .flat_map(|&v| (v as u32).to_le_bytes())
                .collect();
            let n = values.len();
            // All-ones bitmap: counts/lens are never null.
            let valid_bytes = (((n + 7) / 8 + 3) & !3).max(4);
            let valid = vec![0xFFu8; valid_bytes];
            ("U32", data, valid)
        }
        AggOutput::I32 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("I32", data, v)
        }
        AggOutput::F32 { values, valid } => {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let v = pack_valid_bitmap(valid);
            ("F32", data, v)
        }
    }
}

/// PyO3 entry point: `polars_metal._native.execute_groupby`.
///
/// # Wire protocol
///
/// `plan_dict` shape:
/// ```python
/// {
///     "keys": [["col_name", "I64"], ...],           # list of [name, dtype_tag]
///     "aggs": [
///         {"input_col": "x", "op": "Sum", "output_alias": "x_sum"},
///         {"input_col": "",  "op": "Len", "output_alias": "n"},
///     ],
/// }
/// ```
///
/// `columns`: one `(name, dtype_tag, data_bytes, valid_bytes)` tuple per
/// column that appears in either `keys` or `aggs.input_col`.
///
/// # Returns
///
/// A Python list of `(col_name: str, dtype_tag: str, data: bytes,
/// valid: bytes)` tuples — first the key columns (in `keys` order), then
/// the agg outputs (in `aggs` order, named by `output_alias`).  The
/// Python UDF (`_udf.py::_build_groupby`) reassembles these into a Polars
/// DataFrame via PyArrow, matching the pattern of `execute_filter_compact`.
///
/// Supported dtype tags on output: `"I64"`, `"F64"`, `"Bool"`, `"U32"`.
#[pyfunction]
pub fn execute_groupby<'py>(
    py: Python<'py>,
    plan_dict: Bound<'py, PyDict>,
    n_rows: usize,
    columns: &Bound<'py, PyList>,
) -> PyResult<Bound<'py, PyList>> {
    // 1. Parse plan dict → ParsedGroupByPlan.
    let parsed = parse_groupby_plan(&plan_dict)
        .map_err(|e| PyValueError::new_err(format!("polars_metal: plan parse error: {e}")))?;

    // 2. Build lookup: col_name → (dtype_tag, data_bytes, valid_bytes).
    //    We hold references into the PyBytes objects rather than copying.
    //    The PyBytes objects are alive for the duration of this function, so
    //    the byte slice references are safe to use until we return.
    let mut by_name: HashMap<String, (String, Bound<'py, PyBytes>, Bound<'py, PyBytes>)> =
        HashMap::new();
    for (idx, entry) in columns.iter().enumerate() {
        let tup: Bound<PyTuple> = entry.downcast_into().map_err(|_| {
            PyValueError::new_err(format!(
                "polars_metal: execute_groupby columns[{idx}] must be a tuple"
            ))
        })?;
        if tup.len() != 4 {
            return Err(PyValueError::new_err(format!(
                "polars_metal: execute_groupby columns[{idx}] must have 4 elements (name, dtype, data, valid), got {}",
                tup.len()
            )));
        }
        let name: String = tup.get_item(0)?.extract()?;
        let dtype_tag: String = tup.get_item(1)?.extract()?;
        let data_py: Bound<PyBytes> = tup.get_item(2)?.downcast_into().map_err(|_| {
            PyValueError::new_err(format!(
                "polars_metal: execute_groupby columns[{idx}].data must be bytes"
            ))
        })?;
        let valid_py: Bound<PyBytes> = tup.get_item(3)?.downcast_into().map_err(|_| {
            PyValueError::new_err(format!(
                "polars_metal: execute_groupby columns[{idx}].valid must be bytes"
            ))
        })?;
        by_name.insert(name, (dtype_tag, data_py, valid_py));
    }

    // 3. Build KeyColumn slice.
    //    We must keep the &[u8] alive until after dispatch_groupby returns.
    //    We collect (data_bytes, valid_bytes) references before constructing
    //    KeyColumn structs so their lifetimes are tied to the PyBytes in
    //    `by_name`, which outlives the dispatch call.
    //
    //    Phase 7 Task 34: Utf8 keys need a server-side preprocessing pass.
    //    The Python walker hands us a packed `[n_rows u32 le | offsets
    //    (n+1)×i32 le | string bytes]` payload; we parse it into Option<&str>
    //    rows, build (dict, codes) via `build_dict_nullable`, then transmute
    //    the Vec<u32> codes to a Vec<u8> that the KeyColumn borrows. Both
    //    the codes bytes AND the dict must outlive the KeyColumn, so we hold
    //    them in `utf8_owned_data` here next to `key_byte_refs`.
    let mut key_byte_refs: Vec<(&[u8], &[u8])> = Vec::with_capacity(parsed.keys.len());
    // Index `parsed.keys` → (codes_bytes, dict). None for non-Utf8 keys.
    let mut utf8_owned_data: Vec<Option<(Vec<u8>, Vec<String>)>> =
        Vec::with_capacity(parsed.keys.len());
    for k in &parsed.keys {
        let (_, data_py, valid_py) = by_name.get(&k.name).ok_or_else(|| {
            PyKeyError::new_err(format!(
                "polars_metal: key column {:?} not found in upstream columns",
                k.name
            ))
        })?;
        let data_bytes: &[u8] = data_py.as_bytes();
        let valid_bytes: &[u8] = valid_py.as_bytes();

        if k.dtype == MetalDtype::Utf8 {
            // Parse the packed wire payload.
            if data_bytes.len() < 4 {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Utf8 column {:?} wire payload too short ({} B; \
                     need >= 4 for header)",
                    k.name,
                    data_bytes.len()
                )));
            }
            let header_n =
                u32::from_le_bytes([data_bytes[0], data_bytes[1], data_bytes[2], data_bytes[3]])
                    as usize;
            if header_n != n_rows {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Utf8 column {:?} wire n_rows={header_n} \
                     disagrees with column n_rows={n_rows}",
                    k.name
                )));
            }
            let offsets_start = 4usize;
            let offsets_end = offsets_start + (n_rows + 1) * 4;
            if data_bytes.len() < offsets_end {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Utf8 column {:?} wire payload truncated in offsets \
                     ({} B; need >= {})",
                    k.name,
                    data_bytes.len(),
                    offsets_end
                )));
            }
            let mut offsets: Vec<i32> = Vec::with_capacity(n_rows + 1);
            for i in 0..=n_rows {
                let off = offsets_start + i * 4;
                offsets.push(i32::from_le_bytes([
                    data_bytes[off],
                    data_bytes[off + 1],
                    data_bytes[off + 2],
                    data_bytes[off + 3],
                ]));
            }
            let string_bytes = &data_bytes[offsets_end..];

            // Minimum validity bitmap length.
            let min_valid = (n_rows + 7) / 8;
            if valid_bytes.len() < min_valid {
                return Err(PyValueError::new_err(format!(
                    "polars_metal: Utf8 column {:?} validity buffer is {} B, \
                     need at least {} B for {} rows",
                    k.name,
                    valid_bytes.len(),
                    min_valid,
                    n_rows
                )));
            }

            // Build Option<&str> per row, honoring validity. Null rows get
            // `None` and skip the offset slice entirely (Arrow's null row
            // typically has offset[i] == offset[i+1], but we don't depend on
            // that — we simply ignore offsets for null rows).
            let mut strings_opt: Vec<Option<&str>> = Vec::with_capacity(n_rows);
            for r in 0..n_rows {
                let bit_set = (valid_bytes[r >> 3] >> (r & 7)) & 1 == 1;
                if !bit_set {
                    strings_opt.push(None);
                    continue;
                }
                let s_off = offsets[r];
                let e_off = offsets[r + 1];
                if s_off < 0 || e_off < s_off {
                    return Err(PyValueError::new_err(format!(
                        "polars_metal: Utf8 column {:?} row {r} has invalid offsets \
                         start={s_off} end={e_off}",
                        k.name
                    )));
                }
                let s_idx = s_off as usize;
                let e_idx = e_off as usize;
                if e_idx > string_bytes.len() {
                    return Err(PyValueError::new_err(format!(
                        "polars_metal: Utf8 column {:?} row {r} end offset {e_idx} \
                         exceeds string buffer length {}",
                        k.name,
                        string_bytes.len()
                    )));
                }
                let slice = &string_bytes[s_idx..e_idx];
                let s = std::str::from_utf8(slice).map_err(|e| {
                    PyValueError::new_err(format!(
                        "polars_metal: Utf8 column {:?} row {r} is not valid UTF-8: {e}",
                        k.name
                    ))
                })?;
                strings_opt.push(Some(s));
            }

            let (dict, codes, _valid_again) =
                polars_metal_buffer::dict::build_dict_nullable(&strings_opt);

            // Transmute Vec<u32> codes → Vec<u8> bytes for the wire format
            // that `encode_keys` expects. We can't use a from_raw_parts
            // reinterpret here because the KeyColumn borrows the slice; the
            // Vec<u32> itself must live in `utf8_owned_data` and we want a
            // byte slice over it. Convert via to_le_bytes to keep little-
            // endian semantics consistent across host endianness (M-series
            // is little-endian, but be explicit).
            let mut codes_bytes: Vec<u8> = Vec::with_capacity(codes.len() * 4);
            for c in &codes {
                codes_bytes.extend_from_slice(&c.to_le_bytes());
            }

            utf8_owned_data.push(Some((codes_bytes, dict)));
            // Push placeholder byte-refs; we'll override in the second pass.
            key_byte_refs.push((data_bytes, valid_bytes));
        } else {
            utf8_owned_data.push(None);
            key_byte_refs.push((data_bytes, valid_bytes));
        }
    }
    let key_cols: Vec<KeyColumn<'_>> = parsed
        .keys
        .iter()
        .zip(key_byte_refs.iter())
        .zip(utf8_owned_data.iter())
        .map(|((k, (data, valid)), utf8_opt)| {
            // For Utf8 keys we point `data` at the owned codes bytes and
            // attach the dict. For all other dtypes we keep the original
            // Python-side bytes and a None dict.
            let (data_slice, dict): (&[u8], Option<Vec<String>>) = match utf8_opt {
                Some((codes_bytes, dict)) => (codes_bytes.as_slice(), Some(dict.clone())),
                None => (*data, None),
            };
            Ok(KeyColumn {
                name: k.name.clone(),
                dtype: metal_dtype_to_key_dtype(k.dtype).map_err(|e| {
                    pyo3::exceptions::PyNotImplementedError::new_err(format!("polars_metal: {e}"))
                })?,
                data: data_slice,
                valid,
                n_rows,
                dict,
            })
        })
        .collect::<PyResult<Vec<KeyColumn<'_>>>>()?;

    // 4. Build the routing-input view: each agg's value-column byte/dtype
    //    triple, keyed by column name. The fused path consumes a HashMap of
    //    ValueColumns; the M2 per-agg path consumes (AggRequest, ValueColumn)
    //    pairs. We build the byte-references first; typed slices materialize
    //    after the routing decision so we can specialize both paths
    //    correctly.
    //
    // `name_byte_refs` covers EVERY column referenced by any Simple's
    // `input_col` OR by any Expression's `referenced_columns()`. This is a
    // superset of the M2 byte_refs because Expression-shape aggs can name
    // columns no Simple agg touches.
    let mut name_byte_refs: HashMap<String, (String, &[u8], &[u8])> = HashMap::new();
    for agg in &parsed.aggs {
        let referenced: Vec<String> = match agg {
            ParsedAgg::Length { .. } => Vec::new(),
            ParsedAgg::Simple { input_col, .. } => vec![input_col.clone()],
            ParsedAgg::Expression { expr, .. } => expr.referenced_columns(),
        };
        for col_name in referenced {
            if name_byte_refs.contains_key(&col_name) {
                continue;
            }
            let (dtype_tag, data_py, valid_py) = by_name.get(&col_name).ok_or_else(|| {
                PyKeyError::new_err(format!(
                    "polars_metal: agg input column {col_name:?} not found in upstream columns"
                ))
            })?;
            name_byte_refs.insert(
                col_name,
                (dtype_tag.clone(), data_py.as_bytes(), valid_py.as_bytes()),
            );
        }
    }

    // 5. Acquire device + queue.
    let device = MetalDevice::system_default()
        .map_err(|e| crate::engine_err(crate::EngineError::Buffer(e)))?;
    let mut queue = CommandQueue::new(&device)
        .map_err(|e| crate::engine_err(crate::EngineError::Other(format!("command queue: {e}"))))?;

    // 6. Routing: fused single-kernel vs M2 per-agg.
    //
    // The fused kernel caps `n_groups` at 16 (per-thread register array
    // size in `aggregate_fused::emitter::MAX_GROUPS`). The router can't
    // know n_groups ahead of time (it's a runtime build output), so when
    // Fused is selected and the dispatch returns NgroupsExceedsFusedCap
    // we transparently retry on the per-agg path. Expression aggs can't
    // go through per-agg, so they surface as a hard error.
    let initial_choice = decide_groupby_dispatch(&parsed.aggs, &name_byte_refs);
    let has_expression = parsed
        .aggs
        .iter()
        .any(|a| matches!(a, ParsedAgg::Expression { .. }));

    // First, attempt the chosen path; on NgroupsExceedsFusedCap, fall back.
    let mut fused_attempt: Option<
        Result<
            polars_metal_kernels::groupby::GroupByResult,
            polars_metal_kernels::groupby::FusedDispatchError,
        >,
    > = None;
    if matches!(initial_choice, GroupByDispatchChoice::Fused) {
        // Build kernel-layer specs.
        let kernel_specs: Vec<KAggSpec> = parsed
            .aggs
            .iter()
            .map(|pa| match pa {
                ParsedAgg::Simple {
                    input_col,
                    op,
                    output_alias,
                } => KAggSpec::Simple {
                    input_col: input_col.clone(),
                    op: convert_agg_op(*op),
                    output_alias: output_alias.clone(),
                },
                ParsedAgg::Expression {
                    expr,
                    op,
                    output_alias,
                } => KAggSpec::Expression {
                    expr: convert_agg_expr(expr),
                    op: convert_agg_op(*op),
                    output_alias: output_alias.clone(),
                },
                ParsedAgg::Length { output_alias } => KAggSpec::Length {
                    output_alias: output_alias.clone(),
                },
            })
            .collect();

        // Materialize each referenced column as a typed ValueColumn.
        let mut value_columns: HashMap<String, ValueColumn<'_>> = HashMap::new();
        for (name, (dt_tag, data, valid)) in name_byte_refs.iter() {
            let vcol = build_value_column(dt_tag, data, valid, n_rows).map_err(|e| {
                PyValueError::new_err(format!(
                    "polars_metal: fused groupby — value column {name:?}: {e}"
                ))
            })?;
            value_columns.insert(name.clone(), vcol);
        }

        let cache = get_or_init_fused_cache(&device);
        fused_attempt = Some(dispatch_groupby_fused(
            &device,
            &mut queue,
            cache,
            &key_cols,
            &kernel_specs,
            &value_columns,
            n_rows,
        ));
    }

    // Decide which path's output to use:
    //   - Fused attempt succeeded → use it directly.
    //   - Fused attempt rejected with NgroupsExceedsFusedCap and we can
    //     fall back to per-agg (query has no Expression aggs) → run per-agg.
    //   - Fused attempt failed irrecoverably → surface the error.
    //   - We never attempted fused (initial choice was PerAgg) → run per-agg.
    let early_result: Option<polars_metal_kernels::groupby::GroupByResult> = match fused_attempt {
        Some(Ok(r)) => Some(r),
        Some(Err(polars_metal_kernels::groupby::FusedDispatchError::NgroupsExceedsFusedCap {
            ..
        })) if !has_expression => None,
        Some(Err(e)) => {
            return Err(PyValueError::new_err(format!(
                "polars_metal: fused groupby dispatch: {e}"
            )));
        }
        None => None,
    };

    let result = match early_result {
        Some(r) => r,
        // M2's per-agg path (conformance-only) — see `execute_peragg`.
        None => legacy::execute_peragg(
            &device,
            &mut queue,
            &key_cols,
            &parsed,
            &name_byte_refs,
            n_rows,
        )?,
    };

    // 7. Encode result as a list of (name, dtype_tag, data, valid) tuples.
    //    Key columns first, then agg outputs.
    let out_list = PyList::empty_bound(py);

    for (i, key) in parsed.keys.iter().enumerate() {
        let decoded = &result.decoded_keys[i];
        let (dtype_tag, data, valid) = encode_decoded_column(decoded);
        let tup = PyTuple::new_bound(
            py,
            [
                key.name.clone().into_py(py),
                dtype_tag.into_py(py),
                PyBytes::new_bound(py, &data).into_py(py),
                PyBytes::new_bound(py, &valid).into_py(py),
            ],
        );
        out_list.append(tup)?;
    }

    for (i, agg) in parsed.aggs.iter().enumerate() {
        let output = &result.agg_outputs[i];
        let (dtype_tag, data, valid) = encode_agg_output(output);
        let tup = PyTuple::new_bound(
            py,
            [
                agg.output_alias().to_string().into_py(py),
                dtype_tag.into_py(py),
                PyBytes::new_bound(py, &data).into_py(py),
                PyBytes::new_bound(py, &valid).into_py(py),
            ],
        );
        out_list.append(tup)?;
    }

    Ok(out_list)
}
