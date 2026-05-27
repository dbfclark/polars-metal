//! MSL template emitter for fused multi-aggregation kernels.
//!
//! Given an [`AggSignature`] and the kernel-layer [`AggSpec`] slice the
//! signature was built from, produces an MSL source string with one
//! `aggregate_fused` entry point. The kernel:
//!
//! 1. Loads each value column exactly once per row (shared across every
//!    Simple agg referencing that column).
//! 2. Updates each per-group accumulator via 32-bit atomics.
//! 3. Emits a separate trailing `Length` section (no value column).
//!
//! ## Scope (Task 12)
//!
//! Only **Simple** aggs over **32-bit-input** columns are emitted; this
//! covers Sum / Mean / Min / Max over F32, I32, U32 inputs plus Count
//! and Length over any column. F64 / I64 / U64 input columns are
//! rejected by [`signature_supported_by_fused`]; the dispatcher (Task 15)
//! routes those to the M2 per-agg fallback, which loads as their 32-bit
//! equivalent and finalizes the widened sum on CPU (`aggregate_sum_f64_cpu`
//! pattern).
//!
//! Expression aggs (Task 13) are emitted **after** the per-column-load
//! loop in their own stanza: each Expression evaluates inline as a
//! `float`, guarded by the AND of its referenced columns' validity bits,
//! and feeds the same per-op accumulator template as Simple. Shared
//! value-column loads (Task 12 invariant) are preserved — Expression aggs
//! re-use the `val_<i>` / `val_<i>_valid` locals already in scope.
//!
//! ## Apple Silicon atomic constraints
//!
//! Toolchain 32023.883 lacks 64-bit atomics. It *also* rejects
//! `atomic_fetch_add_explicit` on `atomic_float` at
//! `MTLComputePipelineState` creation, even though the source compiles
//! via `newLibraryWithSource`. For float Sum/Mean the emitter therefore
//! uses a CAS-loop over `atomic_uint` (the bit-pattern container) —
//! matching the M2 `agg_sum_f32` shader in `shaders/aggregate.metal`.
//! Float Min/Max uses the same CAS-loop pattern because Metal lacks
//! `atomic_fetch_min/max` on floats. Integer Sum/Min/Max use native
//! `atomic_fetch_{add,min,max}_explicit` on `atomic_int` / `atomic_uint`.
//! Count is `atomic_fetch_add_explicit` on `atomic_uint`. Mean is
//! `(sum, count)` — CPU finalizes the division. The Mean *count*
//! companion is `atomic_uint` with `atomic_fetch_add_explicit` (the
//! count of valid rows).
//!
//! ## Slot layout
//!
//! | range                | role                              |
//! |----------------------|-----------------------------------|
//! | 0                    | `row_to_group` (uint per row)     |
//! | 1                    | `n_rows` (1-element uint buffer)  |
//! | `2 .. 2+C`           | `value_<i>` per column slot       |
//! | `2+C .. 2+2C`        | `validity_<i>` per column slot    |
//! | `2+2C ..`            | output buffers, one per agg slot  |
//!
//! Mean reserves two output slots (sum, count); every other agg reserves one.

use std::fmt::Write as _;

use super::signature::{AggExpr, AggOp, AggSignature, AggSpec, BinaryOp, MetalDtype};

/// Public entry point: emit MSL source for the given signature + specs.
///
/// `sig` is the canonical key; `specs` carries the original aliases and
/// is consulted in slot order so output bindings match dispatch order.
pub fn emit_msl(sig: &AggSignature, specs: &[AggSpec]) -> String {
    let n_cols = sig.column_count();
    let column_order = sig.column_order();
    let column_dtypes = sig.column_dtypes();

    let mut slot = 0usize;
    let row_to_group_slot = slot;
    slot += 1;
    let n_rows_slot = slot;
    slot += 1;

    let mut value_slots = Vec::with_capacity(n_cols);
    for _ in 0..n_cols {
        value_slots.push(slot);
        slot += 1;
    }
    let mut validity_slots = Vec::with_capacity(n_cols);
    for _ in 0..n_cols {
        validity_slots.push(slot);
        slot += 1;
    }

    // Outputs: one slot per agg, two for Mean (sum + count).
    let mut output_slots: Vec<Vec<usize>> = Vec::with_capacity(specs.len());
    for spec in specs {
        let mut outs = vec![slot];
        slot += 1;
        let is_mean = matches!(
            spec,
            AggSpec::Simple {
                op: AggOp::Mean,
                ..
            } | AggSpec::Expression {
                op: AggOp::Mean,
                ..
            }
        );
        if is_mean {
            outs.push(slot);
            slot += 1;
        }
        output_slots.push(outs);
    }

    let mut s = String::new();
    s.push_str("#include <metal_stdlib>\n");
    s.push_str("#include <metal_atomic>\n");
    s.push_str("using namespace metal;\n\n");

    // ---- kernel signature ----
    s.push_str("kernel void aggregate_fused(\n");
    let _ = writeln!(
        s,
        "  device const uint*  row_to_group [[buffer({row_to_group_slot})]],"
    );
    let _ = writeln!(
        s,
        "  device const uint*  n_rows       [[buffer({n_rows_slot})]],"
    );
    for (i, sl) in value_slots.iter().enumerate() {
        let ty = msl_value_load_type(column_dtypes[i]);
        let _ = writeln!(s, "  device const {ty}*  value_{i}    [[buffer({sl})]],");
    }
    for (i, sl) in validity_slots.iter().enumerate() {
        let _ = writeln!(s, "  device const uchar* validity_{i} [[buffer({sl})]],");
    }
    for (a, outs) in output_slots.iter().enumerate() {
        for (j, sl) in outs.iter().enumerate() {
            let ty = msl_output_atomic_type(&specs[a], j, column_dtypes, sig);
            let _ = writeln!(s, "  device {ty}*  out_{a}_{j}     [[buffer({sl})]],");
        }
    }
    s.push_str("  uint gid [[thread_position_in_grid]])\n{\n");

    // ---- bounds + group lookup ----
    s.push_str("  if (gid >= n_rows[0]) return;\n");
    s.push_str("  uint g = row_to_group[gid];\n\n");

    // ---- per-column: validity + shared load + per-agg updates ----
    for (col_idx, name) in column_order.iter().enumerate() {
        let _ = writeln!(s, "  // --- column slot {col_idx} ({name}) ---");
        let _ = writeln!(
            s,
            "  uchar val_{col_idx}_valid = (validity_{col_idx}[gid >> 3] >> (gid & 7)) & 1u;"
        );
        let _ = writeln!(s, "  auto val_{col_idx} = value_{col_idx}[gid];");
        for (a, spec) in specs.iter().enumerate() {
            if spec_references_simple_col(spec, name) {
                emit_simple_agg_update(&mut s, spec, a, col_idx, column_dtypes[col_idx]);
            }
        }
        s.push('\n');
    }

    // ---- expression aggs (Task 13) ----
    // Emitted after the per-column-load loop because an Expression may
    // reference multiple column slots. The `val_<i>` / `val_<i>_valid`
    // locals are already in scope.
    for (a, spec) in specs.iter().enumerate() {
        if let AggSpec::Expression { expr, op, .. } = spec {
            emit_expression_agg_update(&mut s, expr, *op, a, column_order);
        }
    }

    // ---- length aggs (no value column) ----
    for (a, spec) in specs.iter().enumerate() {
        if matches!(spec, AggSpec::Length { .. }) {
            let _ = writeln!(
                s,
                "  atomic_fetch_add_explicit((device atomic_uint*)&out_{a}_0[g], 1u, memory_order_relaxed);"
            );
        }
    }

    s.push_str("}\n");
    s
}

/// Cheap predicate: is every column dtype in `sig` representable by a
/// 32-bit (or smaller) MSL value load + 32-bit atomic accumulator?
///
/// Task 15's dispatcher consults this to decide whether to route the
/// query through the fused kernel or fall back to M2's per-agg path.
/// F64 / I64 input columns return `false`; their accumulation goes
/// through M2's CPU-finalize path.
pub fn signature_supported_by_fused(sig: &AggSignature) -> bool {
    sig.column_dtypes().iter().all(|d| match d {
        MetalDtype::F64 | MetalDtype::I64 => false,
        // Utf8 is never an agg value column (it's a key dtype only), but the
        // mirror enum carries it. Report unsupported so a bug that lifts a
        // Utf8 column into the fused path falls back rather than panics.
        MetalDtype::Utf8 => false,
        // List every other variant so adding one to the mirror is a
        // compile error here (force the author to think about it).
        MetalDtype::F32
        | MetalDtype::I32
        | MetalDtype::U32
        | MetalDtype::I16
        | MetalDtype::U16
        | MetalDtype::I8
        | MetalDtype::U8
        | MetalDtype::Bool => true,
    })
}

// ---------- helpers --------------------------------------------------------

fn msl_value_load_type(dt: MetalDtype) -> &'static str {
    match dt {
        MetalDtype::F32 | MetalDtype::F64 => "float",
        MetalDtype::I32 | MetalDtype::I16 | MetalDtype::I8 | MetalDtype::I64 => "int",
        MetalDtype::U32 | MetalDtype::U16 | MetalDtype::U8 => "uint",
        MetalDtype::Bool => "uchar",
        // Utf8 is never a value-column dtype reaching the MSL emitter; the
        // fused-supported predicate above filters it out. Map to a no-op
        // placeholder so the panic path is unreachable in practice but the
        // match remains exhaustive.
        MetalDtype::Utf8 => "uint",
    }
}

/// MSL atomic output-pointer type for the `j`th output of `spec`.
fn msl_output_atomic_type(
    spec: &AggSpec,
    j: usize,
    column_dtypes: &[MetalDtype],
    sig: &AggSignature,
) -> &'static str {
    match spec {
        AggSpec::Simple { op, input_col, .. } => match (op, j) {
            (AggOp::Count, _) => "atomic_uint",
            (AggOp::Len, _) => "atomic_uint",
            (AggOp::Mean, 1) => "atomic_uint",
            // Sum / Mean (sum slot) / Min / Max: type depends on input
            // dtype + op. Float Sum/Mean *and* Min/Max use `atomic_uint`
            // as a bit-pattern container — Apple Silicon Metal toolchain
            // 32023.883 rejects atomic_fetch_add/min/max on `atomic_float`
            // at MTLComputePipelineState creation. See
            // `shaders/aggregate.metal:64-91` and
            // `docs/kernel-authoring.md` § "Apple Silicon Metal atomic
            // ops constraint".
            _ => {
                let dt = column_dtype_for(sig, column_dtypes, input_col).unwrap_or(MetalDtype::F32);
                match (dt, op) {
                    (MetalDtype::F32 | MetalDtype::F64, _) => "atomic_uint",
                    (MetalDtype::I32 | MetalDtype::I16 | MetalDtype::I8 | MetalDtype::I64, _) => {
                        "atomic_int"
                    }
                    _ => "atomic_uint",
                }
            }
        },
        AggSpec::Expression { op, .. } => match (op, j) {
            // Count and Mean's count companion → uint atomic.
            (AggOp::Count, _) | (AggOp::Mean, 1) | (AggOp::Len, _) => "atomic_uint",
            // Sum / Mean (sum slot) / Min / Max: expression evaluates to
            // float (per `emit_expr_msl`), so the accumulator is always
            // `atomic_uint` used as a float bit-pattern container (same
            // CAS-loop pattern as Simple-float Sum/Min/Max — Apple Silicon
            // Metal 32023.883 rejects `atomic_fetch_add` on `atomic_float`
            // at MTLComputePipelineState creation).
            _ => "atomic_uint",
        },
        AggSpec::Length { .. } => "atomic_uint",
    }
}

fn column_dtype_for(
    sig: &AggSignature,
    column_dtypes: &[MetalDtype],
    name: &str,
) -> Option<MetalDtype> {
    sig.column_order()
        .iter()
        .position(|c| c == name)
        .map(|i| column_dtypes[i])
}

fn spec_references_simple_col(spec: &AggSpec, name: &str) -> bool {
    matches!(spec, AggSpec::Simple { input_col, .. } if input_col == name)
}

/// Emit the per-row update for a Simple agg over column slot `col_idx`.
/// The shared `val_<col_idx>` and `val_<col_idx>_valid` are in scope.
fn emit_simple_agg_update(
    out: &mut String,
    spec: &AggSpec,
    agg_idx: usize,
    col_idx: usize,
    col_dtype: MetalDtype,
) {
    let AggSpec::Simple { op, .. } = spec else {
        return;
    };
    let is_float = matches!(col_dtype, MetalDtype::F32 | MetalDtype::F64);
    let is_signed = matches!(
        col_dtype,
        MetalDtype::I32 | MetalDtype::I16 | MetalDtype::I8 | MetalDtype::I64
    );

    match op {
        AggOp::Count => {
            let _ = writeln!(out, "  if (val_{col_idx}_valid) {{");
            let _ = writeln!(
                out,
                "    atomic_fetch_add_explicit((device atomic_uint*)&out_{agg_idx}_0[g], 1u, memory_order_relaxed);"
            );
            out.push_str("  }\n");
        }
        AggOp::Sum => emit_sum(out, agg_idx, col_idx, is_float, is_signed),
        AggOp::Mean => {
            emit_sum(out, agg_idx, col_idx, is_float, is_signed);
            // Companion count update.
            let _ = writeln!(out, "  if (val_{col_idx}_valid) {{");
            let _ = writeln!(
                out,
                "    atomic_fetch_add_explicit((device atomic_uint*)&out_{agg_idx}_1[g], 1u, memory_order_relaxed);"
            );
            out.push_str("  }\n");
        }
        AggOp::Min => emit_min_max(out, agg_idx, col_idx, true, is_float, is_signed),
        AggOp::Max => emit_min_max(out, agg_idx, col_idx, false, is_float, is_signed),
        AggOp::Len => {
            // `Len` normally appears as `AggSpec::Length`; tolerate it
            // here defensively as an unconditional fetch-add.
            let _ = writeln!(
                out,
                "  atomic_fetch_add_explicit((device atomic_uint*)&out_{agg_idx}_0[g], 1u, memory_order_relaxed);"
            );
        }
    }
}

fn emit_sum(out: &mut String, agg_idx: usize, col_idx: usize, is_float: bool, is_signed: bool) {
    let _ = writeln!(out, "  if (val_{col_idx}_valid) {{");
    if is_float {
        // Float Sum via CAS-loop on `atomic_uint` (bit-pattern container).
        // `atomic_fetch_add_explicit` on `atomic_float` fails at
        // MTLComputePipelineState creation on Apple Silicon Metal
        // toolchain 32023.883 even though source compiles. Mirrors
        // `agg_sum_f32` in `shaders/aggregate.metal`.
        let _ = writeln!(
            out,
            "    uint old_bits_a = atomic_load_explicit((device atomic_uint*)&out_{agg_idx}_0[g], memory_order_relaxed);"
        );
        out.push_str("    while (true) {\n");
        out.push_str("      float cur = as_type<float>(old_bits_a);\n");
        let _ = writeln!(
            out,
            "      uint next_bits = as_type<uint>(cur + (float)val_{col_idx});"
        );
        let _ = writeln!(
            out,
            "      if (atomic_compare_exchange_weak_explicit((device atomic_uint*)&out_{agg_idx}_0[g], &old_bits_a, next_bits, memory_order_relaxed, memory_order_relaxed)) {{ break; }}"
        );
        out.push_str("    }\n");
    } else if is_signed {
        let _ = writeln!(
            out,
            "    atomic_fetch_add_explicit((device atomic_int*)&out_{agg_idx}_0[g], (int)val_{col_idx}, memory_order_relaxed);"
        );
    } else {
        let _ = writeln!(
            out,
            "    atomic_fetch_add_explicit((device atomic_uint*)&out_{agg_idx}_0[g], (uint)val_{col_idx}, memory_order_relaxed);"
        );
    }
    out.push_str("  }\n");
}

fn emit_min_max(
    out: &mut String,
    agg_idx: usize,
    col_idx: usize,
    is_min: bool,
    is_float: bool,
    is_signed: bool,
) {
    if is_float {
        let break_cond = if is_min {
            "curf <= valf"
        } else {
            "curf >= valf"
        };
        let _ = writeln!(out, "  if (val_{col_idx}_valid) {{");
        let _ = writeln!(out, "    float valf = (float)val_{col_idx};");
        let _ = writeln!(
            out,
            "    uint old_bits = atomic_load_explicit((device atomic_uint*)&out_{agg_idx}_0[g], memory_order_relaxed);"
        );
        out.push_str("    while (true) {\n");
        out.push_str("      float curf = as_type<float>(old_bits);\n");
        let _ = writeln!(out, "      if ({break_cond}) {{ break; }}");
        out.push_str("      uint next_bits = as_type<uint>(valf);\n");
        let _ = writeln!(
            out,
            "      if (atomic_compare_exchange_weak_explicit((device atomic_uint*)&out_{agg_idx}_0[g], &old_bits, next_bits, memory_order_relaxed, memory_order_relaxed)) {{ break; }}"
        );
        out.push_str("    }\n");
        out.push_str("  }\n");
    } else {
        let fn_name = if is_min {
            "atomic_fetch_min_explicit"
        } else {
            "atomic_fetch_max_explicit"
        };
        let _ = writeln!(out, "  if (val_{col_idx}_valid) {{");
        if is_signed {
            let _ = writeln!(
                out,
                "    {fn_name}((device atomic_int*)&out_{agg_idx}_0[g], (int)val_{col_idx}, memory_order_relaxed);"
            );
        } else {
            let _ = writeln!(
                out,
                "    {fn_name}((device atomic_uint*)&out_{agg_idx}_0[g], (uint)val_{col_idx}, memory_order_relaxed);"
            );
        }
        out.push_str("  }\n");
    }
}

// ---------- Expression-agg emission (Task 13) ------------------------------

/// Convert a kernel-layer `AggExpr` to an MSL `float`-typed expression.
///
/// `col_order` is the first-seen column order from `AggSignature` — the
/// position of a column name in this slice IS its `val_<i>` slot index.
/// Callers must have pre-interned every column referenced by `expr`
/// (`AggSignature::from_specs` does this).
///
/// Literals are emitted with explicit `(float)` casts (and the `f`
/// suffix for finite floats) to keep the result type `float` throughout
/// the tree — Apple Silicon Metal can't run `double` arithmetic in
/// compute kernels under toolchain 32023.883.
fn emit_expr_msl(expr: &AggExpr, col_order: &[String]) -> String {
    match expr {
        AggExpr::Column(name) => {
            // `from_specs` interns every referenced column, so position()
            // is always Some in a valid signature. Fall back to slot 0
            // in release builds to keep behavior deterministic if the
            // contract is violated; debug_assert documents the invariant.
            debug_assert!(
                col_order.iter().any(|c| c == name),
                "emit_expr_msl: column `{name}` missing from col_order; \
                 caller must build the signature with from_specs() first"
            );
            let idx = col_order.iter().position(|c| c == name).unwrap_or(0);
            format!("(float)val_{idx}")
        }
        AggExpr::LiteralF64(v) => {
            if v.is_nan() {
                "(float)NAN".into()
            } else if v.is_infinite() {
                if *v > 0.0 {
                    "(float)INFINITY".into()
                } else {
                    "-(float)INFINITY".into()
                }
            } else {
                // Stringify with enough precision to round-trip an f32
                // and append the Metal `f` suffix so the literal binds
                // as `float`, not `double`.
                format!("{v:?}f")
            }
        }
        AggExpr::LiteralI64(v) => format!("(float){v}"),
        AggExpr::Binary { op, lhs, rhs } => {
            let l = emit_expr_msl(lhs, col_order);
            let r = emit_expr_msl(rhs, col_order);
            let s = match op {
                BinaryOp::Add => "+",
                BinaryOp::Sub => "-",
                BinaryOp::Mul => "*",
                BinaryOp::Div => "/",
            };
            format!("({l} {s} {r})")
        }
    }
}

/// AND of all referenced columns' validity bits. Empty (literal-only)
/// expression returns the constant `1u` — the M3 walker doesn't emit
/// literal-only Expression aggs, but the helper stays deterministic.
fn emit_expr_validity_check(expr: &AggExpr, col_order: &[String]) -> String {
    let cols = expr.referenced_columns();
    if cols.is_empty() {
        return "1u".into();
    }
    cols.iter()
        .map(|c| {
            debug_assert!(
                col_order.iter().any(|x| x == c),
                "emit_expr_validity_check: column `{c}` missing from col_order"
            );
            let idx = col_order.iter().position(|x| x == c).unwrap_or(0);
            format!("val_{idx}_valid")
        })
        .collect::<Vec<_>>()
        .join(" & ")
}

/// Emit the per-row update for one `AggSpec::Expression` agg. The
/// `val_<i>` / `val_<i>_valid` locals are already in scope (the
/// per-column-load loop ran first). The expression evaluates to a
/// `float`; the accumulator pattern matches Simple-float for Sum/Min/Max
/// (CAS loop on `atomic_uint` as a bit-pattern container) and
/// `atomic_fetch_add_explicit` on `atomic_uint` for Count and Mean's
/// count companion.
fn emit_expression_agg_update(
    out: &mut String,
    expr: &AggExpr,
    op: AggOp,
    agg_idx: usize,
    col_order: &[String],
) {
    let _ = writeln!(out, "  // --- expression agg slot {agg_idx} ---");
    let guard = emit_expr_validity_check(expr, col_order);
    let value = emit_expr_msl(expr, col_order);

    match op {
        AggOp::Count => {
            let _ = writeln!(out, "  if ({guard}) {{");
            let _ = writeln!(
                out,
                "    atomic_fetch_add_explicit((device atomic_uint*)&out_{agg_idx}_0[g], 1u, memory_order_relaxed);"
            );
            out.push_str("  }\n");
        }
        AggOp::Len => {
            // Len normally appears as AggSpec::Length; tolerate it here
            // defensively — semantics match an unconditional row count.
            let _ = writeln!(
                out,
                "  atomic_fetch_add_explicit((device atomic_uint*)&out_{agg_idx}_0[g], 1u, memory_order_relaxed);"
            );
        }
        AggOp::Sum => emit_expression_sum(out, agg_idx, &guard, &value),
        AggOp::Mean => {
            emit_expression_sum(out, agg_idx, &guard, &value);
            // Companion count update — increments per valid-row.
            let _ = writeln!(out, "  if ({guard}) {{");
            let _ = writeln!(
                out,
                "    atomic_fetch_add_explicit((device atomic_uint*)&out_{agg_idx}_1[g], 1u, memory_order_relaxed);"
            );
            out.push_str("  }\n");
        }
        AggOp::Min => emit_expression_min_max(out, agg_idx, &guard, &value, true),
        AggOp::Max => emit_expression_min_max(out, agg_idx, &guard, &value, false),
    }
}

fn emit_expression_sum(out: &mut String, agg_idx: usize, guard: &str, value: &str) {
    // Float Sum via CAS loop on `atomic_uint` (bit-pattern container),
    // matching the Simple-float Sum pattern.
    let _ = writeln!(out, "  if ({guard}) {{");
    let _ = writeln!(out, "    float ex_{agg_idx} = (float)({value});");
    let _ = writeln!(
        out,
        "    uint old_bits_e{agg_idx} = atomic_load_explicit((device atomic_uint*)&out_{agg_idx}_0[g], memory_order_relaxed);"
    );
    out.push_str("    while (true) {\n");
    let _ = writeln!(
        out,
        "      float cur_e{agg_idx} = as_type<float>(old_bits_e{agg_idx});"
    );
    let _ = writeln!(
        out,
        "      uint next_bits_e{agg_idx} = as_type<uint>(cur_e{agg_idx} + ex_{agg_idx});"
    );
    let _ = writeln!(
        out,
        "      if (atomic_compare_exchange_weak_explicit((device atomic_uint*)&out_{agg_idx}_0[g], &old_bits_e{agg_idx}, next_bits_e{agg_idx}, memory_order_relaxed, memory_order_relaxed)) {{ break; }}"
    );
    out.push_str("    }\n");
    out.push_str("  }\n");
}

fn emit_expression_min_max(
    out: &mut String,
    agg_idx: usize,
    guard: &str,
    value: &str,
    is_min: bool,
) {
    let break_cond = if is_min {
        format!("curf_e{agg_idx} <= valf_e{agg_idx}")
    } else {
        format!("curf_e{agg_idx} >= valf_e{agg_idx}")
    };
    let _ = writeln!(out, "  if ({guard}) {{");
    let _ = writeln!(out, "    float valf_e{agg_idx} = (float)({value});");
    let _ = writeln!(
        out,
        "    uint old_bits_e{agg_idx} = atomic_load_explicit((device atomic_uint*)&out_{agg_idx}_0[g], memory_order_relaxed);"
    );
    out.push_str("    while (true) {\n");
    let _ = writeln!(
        out,
        "      float curf_e{agg_idx} = as_type<float>(old_bits_e{agg_idx});"
    );
    let _ = writeln!(out, "      if ({break_cond}) {{ break; }}");
    let _ = writeln!(
        out,
        "      uint next_bits_e{agg_idx} = as_type<uint>(valf_e{agg_idx});"
    );
    let _ = writeln!(
        out,
        "      if (atomic_compare_exchange_weak_explicit((device atomic_uint*)&out_{agg_idx}_0[g], &old_bits_e{agg_idx}, next_bits_e{agg_idx}, memory_order_relaxed, memory_order_relaxed)) {{ break; }}"
    );
    out.push_str("    }\n");
    out.push_str("  }\n");
}
