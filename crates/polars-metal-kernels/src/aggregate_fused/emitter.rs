//! MSL template emitter for fused multi-aggregation kernels.
//!
//! Given an [`AggSignature`] and the kernel-layer [`AggSpec`] slice the
//! signature was built from, produces an MSL source string with one
//! `aggregate_fused` entry point. The kernel:
//!
//! 1. Strides each thread over `(row += grid_size)` rows, loading each
//!    value column exactly once per row (shared across every Simple agg
//!    referencing that column).
//! 2. Accumulates per-row updates into **per-thread register arrays**
//!    sized `[MAX_GROUPS]`. Nothing touches device memory in the inner
//!    loop.
//! 3. After the row loop, performs a **simdgroup reduce** of every agg
//!    accumulator (one per (agg, group)), stages the per-simdgroup
//!    partial in threadgroup shared memory, threadgroup-syncs, and has
//!    thread 0 sum the simdgroups and emit a single CAS / fetch-add per
//!    (TG, group) to the device output buffer.
//!
//! ## Why pre-reduce?
//!
//! Before this rewrite, every row issued an atomic update directly on
//! `out[g]`. F32 accumulators use a CAS loop on `atomic_uint` (a bit-
//! pattern container — Apple Silicon's `atomic_float` rejects
//! `atomic_fetch_add` at `MTLComputePipelineState` creation), so under
//! low n_groups every CAS retries against contending threads. At 10M
//! rows × 4 groups the retry chain is O(N²/2) and trips Metal's GPU
//! watchdog (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`).
//! The pre-reduce structure collapses N atomic-CAS attempts to
//! (n_threadgroups × n_groups), which is ≤ a few thousand on a Q1-shape
//! query.
//!
//! Integer aggs (Sum/Min/Max over I32/U32) don't have the contention
//! pathology — `atomic_fetch_add/min/max` on `atomic_int`/`atomic_uint`
//! is a hardware op with no retry — but they share the same pre-reduce
//! structure here for uniformity (and to amortize the per-row group
//! lookup and bounds check).
//!
//! ## Scope (Task 12 + Phase-13-bugfix)
//!
//! Only **Simple** aggs over **32-bit-input** columns are emitted; this
//! covers Sum / Mean / Min / Max over F32, I32, U32 inputs plus Count
//! and Length over any column. F64 / I64 / U64 input columns are
//! rejected by [`signature_supported_by_fused`]; the dispatcher routes
//! those to the M2 per-agg fallback, which loads as their 32-bit
//! equivalent and finalizes the widened sum on CPU.
//!
//! Expression aggs evaluate to `float`, guarded by the AND of their
//! referenced columns' validity bits. They share the same per-thread
//! accumulator + reduce + flush pattern as Simple F32 aggs.
//!
//! ## `MAX_GROUPS = 16` cap
//!
//! The per-thread register arrays are sized `[MAX_GROUPS]`. Higher
//! cardinality must be routed via the per-agg path (the dispatcher
//! checks `n_groups <= F32_AGG_MAX_GROUPS` before calling this kernel
//! and falls back otherwise). Apple Silicon registers comfortably hold
//! ~16 floats per agg per thread; pushing higher would spill to
//! threadgroup memory and erase the win.
//!
//! ## Apple Silicon atomic constraints
//!
//! Toolchain 32023.883 lacks 64-bit atomics. It *also* rejects
//! `atomic_fetch_add_explicit` on `atomic_float` at
//! `MTLComputePipelineState` creation, even though the source compiles
//! via `newLibraryWithSource`. For float Sum/Mean the final flush uses
//! a CAS loop over `atomic_uint` (the bit-pattern container). Float
//! Min/Max uses the same CAS-loop pattern (Metal lacks
//! `atomic_fetch_min/max` on floats). Integer Sum/Min/Max use native
//! `atomic_fetch_{add,min,max}_explicit` on `atomic_int` /
//! `atomic_uint` at the flush step. Count is
//! `atomic_fetch_add_explicit` on `atomic_uint`. Mean is `(sum, count)`
//! — CPU finalizes the division.
//!
//! ## Slot layout
//!
//! | range                | role                              |
//! |----------------------|-----------------------------------|
//! | 0                    | `row_to_group` (uint per row)     |
//! | 1                    | `n_rows` (1-element uint buffer)  |
//! | 2                    | `n_groups` (1-element uint buffer)|
//! | `3 .. 3+C`           | `value_<i>` per column slot       |
//! | `3+C .. 3+2C`        | `validity_<i>` per column slot    |
//! | `3+2C ..`            | output buffers, one per agg slot  |
//!
//! Mean reserves two output slots (sum, count); every other agg reserves one.

use std::fmt::Write as _;

use super::signature::{AggExpr, AggOp, AggSignature, AggSpec, BinaryOp, MetalDtype};

/// Per-thread register-array capacity for per-group accumulators. Mirrors
/// `MAX_GROUPS` in `shaders/aggregate.metal` and the
/// `F32_AGG_MAX_GROUPS` cap in `groupby.rs`. Callers must guarantee
/// `n_groups <= 16` before dispatching the fused kernel.
pub const MAX_GROUPS: usize = 16;

/// Threadgroup width assumed by the emitted kernel. The MSL declares
/// `tg_partial[MAX_SIMDS_PER_TG * MAX_GROUPS]` with a fixed
/// `MAX_SIMDS_PER_TG = 8`, which corresponds to a 256-wide threadgroup
/// on Apple Silicon (32-lane simdgroup). Dispatcher MUST match.
pub const FUSED_TG_WIDTH: usize = 256;

/// Reduction kind for one accumulator slot. Drives kind-specific source
/// in `init` / `simd reduce` / `flush` stanzas.
#[derive(Debug, Clone, Copy)]
enum ReduceKind {
    /// `float`, simd_sum, CAS-add on `atomic_uint` (bit pattern).
    F32Sum,
    /// `float`, simd_min, CAS-min on `atomic_uint` (bit pattern).
    F32Min,
    /// `float`, simd_max, CAS-max on `atomic_uint` (bit pattern).
    F32Max,
    /// `int`, simd_sum, `atomic_fetch_add` on `atomic_int`.
    I32Sum,
    /// `int`, simd_min, `atomic_fetch_min` on `atomic_int`.
    I32Min,
    /// `int`, simd_max, `atomic_fetch_max` on `atomic_int`.
    I32Max,
    /// `uint`, simd_sum, `atomic_fetch_add` on `atomic_uint`.
    U32Sum,
    /// `uint`, simd_min, `atomic_fetch_min` on `atomic_uint`.
    U32Min,
    /// `uint`, simd_max, `atomic_fetch_max` on `atomic_uint`.
    U32Max,
    /// `uint`, simd_sum, `atomic_fetch_add` on `atomic_uint`. Used for
    /// Count / Length / Mean's count companion.
    Counter,
}

impl ReduceKind {
    fn cpp_scalar(self) -> &'static str {
        match self {
            ReduceKind::F32Sum | ReduceKind::F32Min | ReduceKind::F32Max => "float",
            ReduceKind::I32Sum | ReduceKind::I32Min | ReduceKind::I32Max => "int",
            ReduceKind::U32Sum | ReduceKind::U32Min | ReduceKind::U32Max | ReduceKind::Counter => {
                "uint"
            }
        }
    }

    /// MSL initial value at the start of the kernel.
    fn identity(self) -> &'static str {
        match self {
            ReduceKind::F32Sum => "0.0f",
            ReduceKind::F32Min => "INFINITY",
            ReduceKind::F32Max => "-INFINITY",
            ReduceKind::I32Sum | ReduceKind::U32Sum | ReduceKind::Counter => "0",
            ReduceKind::I32Min => "INT_MAX",
            ReduceKind::I32Max => "INT_MIN",
            ReduceKind::U32Min => "UINT_MAX",
            ReduceKind::U32Max => "0u",
        }
    }

    /// `simd_*` reduction over the lane-local value.
    fn simd_reduce_call(self, expr: &str) -> String {
        match self {
            ReduceKind::F32Sum | ReduceKind::I32Sum | ReduceKind::U32Sum | ReduceKind::Counter => {
                format!("simd_sum({expr})")
            }
            ReduceKind::F32Min | ReduceKind::I32Min | ReduceKind::U32Min => {
                format!("simd_min({expr})")
            }
            ReduceKind::F32Max | ReduceKind::I32Max | ReduceKind::U32Max => {
                format!("simd_max({expr})")
            }
        }
    }

    /// Output atomic pointer type for the device buffer.
    fn atomic_ptr(self) -> &'static str {
        match self {
            ReduceKind::F32Sum
            | ReduceKind::F32Min
            | ReduceKind::F32Max
            | ReduceKind::U32Sum
            | ReduceKind::U32Min
            | ReduceKind::U32Max
            | ReduceKind::Counter => "atomic_uint",
            ReduceKind::I32Sum | ReduceKind::I32Min | ReduceKind::I32Max => "atomic_int",
        }
    }
}

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
    let n_groups_slot = slot;
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

    // Output slots: one per agg, two for Mean (sum + count).
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

    // Build the per-agg reduce-slot list. Each agg owns one or two
    // (output_slot_index, reduce_kind) pairs.
    //
    // The accumulator variables (`acc_<a>_<j>`) live in private memory;
    // `j` distinguishes Mean's sum and count slots.
    let agg_slots = build_agg_slots(specs, column_dtypes, sig);

    let mut s = String::new();
    s.push_str("#include <metal_stdlib>\n");
    s.push_str("#include <metal_atomic>\n");
    s.push_str("#include <metal_simdgroup>\n");
    s.push_str("using namespace metal;\n\n");
    let _ = writeln!(s, "#define MAX_GROUPS {MAX_GROUPS}u");
    s.push_str("#define MAX_SIMDS_PER_TG 8u\n\n");

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
    let _ = writeln!(
        s,
        "  device const uint*  n_groups     [[buffer({n_groups_slot})]],"
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
            let kind = agg_slots[a][j];
            let _ = writeln!(
                s,
                "  device {ptr}*  out_{a}_{j}     [[buffer({sl})]],",
                ptr = kind.atomic_ptr()
            );
        }
    }
    s.push_str("  uint gid              [[thread_position_in_grid]],\n");
    s.push_str("  uint grid_size        [[threads_per_grid]],\n");
    s.push_str("  uint tid_in_tg        [[thread_index_in_threadgroup]],\n");
    s.push_str("  uint sg_index         [[simdgroup_index_in_threadgroup]],\n");
    s.push_str("  uint lane             [[thread_index_in_simdgroup]],\n");
    s.push_str("  uint n_simdgroups     [[simdgroups_per_threadgroup]])\n{\n");

    s.push_str("  uint nr = n_rows[0];\n");
    s.push_str("  uint ng = n_groups[0];\n\n");

    // ---- per-thread accumulator declarations + init -------------------------
    for (a, slots) in agg_slots.iter().enumerate() {
        for (j, kind) in slots.iter().enumerate() {
            let _ = writeln!(s, "  {ty} acc_{a}_{j}[MAX_GROUPS];", ty = kind.cpp_scalar());
            let _ = writeln!(
                s,
                "  for (uint i = 0u; i < MAX_GROUPS; ++i) {{ acc_{a}_{j}[i] = {id}; }}",
                id = kind.identity()
            );
        }
    }
    s.push('\n');

    // ---- strided main row loop ---------------------------------------------
    s.push_str("  for (uint row = gid; row < nr; row += grid_size) {\n");
    s.push_str("    uint g = row_to_group[row];\n");
    s.push_str("    if (g >= ng) continue;\n\n");

    // Per-column shared load + per-agg update bodies.
    for (col_idx, _name) in column_order.iter().enumerate() {
        let _ = writeln!(
            s,
            "    uchar val_{col_idx}_valid = (validity_{col_idx}[row >> 3] >> (row & 7)) & 1u;"
        );
        let _ = writeln!(s, "    auto val_{col_idx} = value_{col_idx}[row];");
    }
    s.push('\n');

    for (a, spec) in specs.iter().enumerate() {
        emit_agg_row_update(&mut s, spec, a, column_order, column_dtypes, sig);
    }

    // Length aggs increment unconditionally — emitted last so they share the
    // same per-row loop body.
    for (a, spec) in specs.iter().enumerate() {
        if matches!(spec, AggSpec::Length { .. }) {
            let _ = writeln!(s, "    acc_{a}_0[g] += 1u;");
        }
    }

    s.push_str("  }\n\n");

    // ---- simdgroup reduce + TGSM staging + per-TG final flush --------------
    //
    // Each reduce slot uses its own MAX_SIMDS_PER_TG * MAX_GROUPS slab of
    // threadgroup memory. We declare slabs of the slot's scalar type so
    // simd reductions can stage without bit-tricking.
    s.push('\n');
    for (a, slots) in agg_slots.iter().enumerate() {
        for (j, kind) in slots.iter().enumerate() {
            let _ = writeln!(
                s,
                "  threadgroup {ty} tg_part_{a}_{j}[MAX_SIMDS_PER_TG * MAX_GROUPS];",
                ty = kind.cpp_scalar()
            );
        }
    }
    s.push('\n');

    // simd reduce: each (a, j) reduces its per-thread accumulator slot
    // across the simdgroup and stages the result.
    s.push_str("  for (uint gi = 0u; gi < MAX_GROUPS; ++gi) {\n");
    s.push_str("    if (gi >= ng) break;\n");
    for (a, slots) in agg_slots.iter().enumerate() {
        for (j, kind) in slots.iter().enumerate() {
            let reduce = kind.simd_reduce_call(&format!("acc_{a}_{j}[gi]"));
            let _ = writeln!(
                s,
                "    {ty} sgv_{a}_{j} = {reduce};",
                ty = kind.cpp_scalar()
            );
            let _ = writeln!(
                s,
                "    if (lane == 0u) tg_part_{a}_{j}[sg_index * MAX_GROUPS + gi] = sgv_{a}_{j};"
            );
        }
    }
    s.push_str("  }\n");
    s.push_str("  threadgroup_barrier(mem_flags::mem_threadgroup);\n\n");

    // Thread 0 of the TG sums simdgroup partials and flushes to device.
    s.push_str("  if (tid_in_tg == 0u) {\n");
    s.push_str("    for (uint gi = 0u; gi < MAX_GROUPS; ++gi) {\n");
    s.push_str("      if (gi >= ng) break;\n");
    for (a, slots) in agg_slots.iter().enumerate() {
        for (j, kind) in slots.iter().enumerate() {
            emit_final_flush(&mut s, a, j, *kind);
        }
    }
    s.push_str("    }\n");
    s.push_str("  }\n");

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

/// Map each agg in `specs` to its list of `(output_slot_index, kind)`
/// reduce slots. Returns `agg_slots[a][j] = kind` such that the `j`th
/// device output buffer of agg `a` has reduction kind `kind`.
fn build_agg_slots(
    specs: &[AggSpec],
    column_dtypes: &[MetalDtype],
    sig: &AggSignature,
) -> Vec<Vec<ReduceKind>> {
    let mut out: Vec<Vec<ReduceKind>> = Vec::with_capacity(specs.len());
    for spec in specs {
        let mut slots: Vec<ReduceKind> = Vec::new();
        match spec {
            AggSpec::Simple { op, input_col, .. } => {
                let dt = column_dtype_for(sig, column_dtypes, input_col).unwrap_or(MetalDtype::F32);
                match op {
                    AggOp::Count => slots.push(ReduceKind::Counter),
                    AggOp::Len => slots.push(ReduceKind::Counter),
                    AggOp::Sum => slots.push(sum_kind_for(dt)),
                    AggOp::Min => slots.push(min_kind_for(dt)),
                    AggOp::Max => slots.push(max_kind_for(dt)),
                    AggOp::Mean => {
                        slots.push(sum_kind_for(dt));
                        slots.push(ReduceKind::Counter);
                    }
                }
            }
            AggSpec::Expression { op, .. } => {
                // Expressions always evaluate to float.
                match op {
                    AggOp::Count => slots.push(ReduceKind::Counter),
                    AggOp::Len => slots.push(ReduceKind::Counter),
                    AggOp::Sum => slots.push(ReduceKind::F32Sum),
                    AggOp::Min => slots.push(ReduceKind::F32Min),
                    AggOp::Max => slots.push(ReduceKind::F32Max),
                    AggOp::Mean => {
                        slots.push(ReduceKind::F32Sum);
                        slots.push(ReduceKind::Counter);
                    }
                }
            }
            AggSpec::Length { .. } => slots.push(ReduceKind::Counter),
        }
        out.push(slots);
    }
    out
}

fn sum_kind_for(dt: MetalDtype) -> ReduceKind {
    match dt {
        MetalDtype::F32 | MetalDtype::F64 => ReduceKind::F32Sum,
        MetalDtype::I32 | MetalDtype::I16 | MetalDtype::I8 | MetalDtype::I64 => ReduceKind::I32Sum,
        _ => ReduceKind::U32Sum,
    }
}

fn min_kind_for(dt: MetalDtype) -> ReduceKind {
    match dt {
        MetalDtype::F32 | MetalDtype::F64 => ReduceKind::F32Min,
        MetalDtype::I32 | MetalDtype::I16 | MetalDtype::I8 | MetalDtype::I64 => ReduceKind::I32Min,
        _ => ReduceKind::U32Min,
    }
}

fn max_kind_for(dt: MetalDtype) -> ReduceKind {
    match dt {
        MetalDtype::F32 | MetalDtype::F64 => ReduceKind::F32Max,
        MetalDtype::I32 | MetalDtype::I16 | MetalDtype::I8 | MetalDtype::I64 => ReduceKind::I32Max,
        _ => ReduceKind::U32Max,
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

/// Emit the per-row update body for one agg, writing into the per-thread
/// `acc_<a>_<j>[g]` accumulator. Wrapped by the per-row strided loop in
/// `emit_msl`.
fn emit_agg_row_update(
    out: &mut String,
    spec: &AggSpec,
    agg_idx: usize,
    column_order: &[String],
    column_dtypes: &[MetalDtype],
    _sig: &AggSignature,
) {
    match spec {
        AggSpec::Simple { op, input_col, .. } => {
            let Some(col_idx) = column_order.iter().position(|c| c == input_col) else {
                return;
            };
            let col_dtype = column_dtypes[col_idx];
            let is_float = matches!(col_dtype, MetalDtype::F32 | MetalDtype::F64);
            let is_signed = matches!(
                col_dtype,
                MetalDtype::I32 | MetalDtype::I16 | MetalDtype::I8 | MetalDtype::I64
            );
            let cast = if is_float {
                "(float)"
            } else if is_signed {
                "(int)"
            } else {
                "(uint)"
            };
            match op {
                AggOp::Count => {
                    let _ = writeln!(
                        out,
                        "    if (val_{col_idx}_valid) acc_{agg_idx}_0[g] += 1u;"
                    );
                }
                AggOp::Len => {
                    // Defensive: tolerate Len as Simple (M3 walker emits
                    // it as AggSpec::Length instead). Unconditional row
                    // count.
                    let _ = writeln!(out, "    acc_{agg_idx}_0[g] += 1u;");
                }
                AggOp::Sum => {
                    let _ = writeln!(
                        out,
                        "    if (val_{col_idx}_valid) acc_{agg_idx}_0[g] += {cast}val_{col_idx};"
                    );
                }
                AggOp::Min => {
                    let _ = writeln!(
                        out,
                        "    if (val_{col_idx}_valid) acc_{agg_idx}_0[g] = min(acc_{agg_idx}_0[g], {cast}val_{col_idx});"
                    );
                }
                AggOp::Max => {
                    let _ = writeln!(
                        out,
                        "    if (val_{col_idx}_valid) acc_{agg_idx}_0[g] = max(acc_{agg_idx}_0[g], {cast}val_{col_idx});"
                    );
                }
                AggOp::Mean => {
                    let _ = writeln!(out, "    if (val_{col_idx}_valid) {{");
                    let _ = writeln!(out, "      acc_{agg_idx}_0[g] += {cast}val_{col_idx};");
                    let _ = writeln!(out, "      acc_{agg_idx}_1[g] += 1u;");
                    let _ = writeln!(out, "    }}");
                }
            }
        }
        AggSpec::Expression { expr, op, .. } => {
            let guard = emit_expr_validity_check(expr, column_order);
            let value = emit_expr_msl(expr, column_order);
            match op {
                AggOp::Count => {
                    let _ = writeln!(out, "    if ({guard}) acc_{agg_idx}_0[g] += 1u;");
                }
                AggOp::Len => {
                    let _ = writeln!(out, "    acc_{agg_idx}_0[g] += 1u;");
                }
                AggOp::Sum => {
                    let _ = writeln!(
                        out,
                        "    if ({guard}) {{ float ex = (float)({value}); acc_{agg_idx}_0[g] += ex; }}"
                    );
                }
                AggOp::Min => {
                    let _ = writeln!(
                        out,
                        "    if ({guard}) {{ float ex = (float)({value}); acc_{agg_idx}_0[g] = min(acc_{agg_idx}_0[g], ex); }}"
                    );
                }
                AggOp::Max => {
                    let _ = writeln!(
                        out,
                        "    if ({guard}) {{ float ex = (float)({value}); acc_{agg_idx}_0[g] = max(acc_{agg_idx}_0[g], ex); }}"
                    );
                }
                AggOp::Mean => {
                    let _ = writeln!(
                        out,
                        "    if ({guard}) {{ float ex = (float)({value}); acc_{agg_idx}_0[g] += ex; acc_{agg_idx}_1[g] += 1u; }}"
                    );
                }
            }
        }
        AggSpec::Length { .. } => {
            // Emitted by the dedicated loop below the per-column section.
        }
    }
}

/// Emit the per-TG, per-group final flush for one reduce slot. Runs
/// inside `if (tid_in_tg == 0u) { for (gi …) { … } }`.
fn emit_final_flush(out: &mut String, agg_idx: usize, j: usize, kind: ReduceKind) {
    let ty = kind.cpp_scalar();
    // Combine the per-simdgroup partials.
    let combine = match kind {
        ReduceKind::F32Sum | ReduceKind::I32Sum | ReduceKind::U32Sum | ReduceKind::Counter => "+=",
        ReduceKind::F32Min | ReduceKind::I32Min | ReduceKind::U32Min => "=min",
        ReduceKind::F32Max | ReduceKind::I32Max | ReduceKind::U32Max => "=max",
    };
    let _ = writeln!(
        out,
        "      {ty} total_{agg_idx}_{j} = {id};",
        id = kind.identity()
    );
    let _ = writeln!(out, "      for (uint s = 0u; s < n_simdgroups; ++s) {{");
    match combine {
        "+=" => {
            let _ = writeln!(
                out,
                "        total_{agg_idx}_{j} += tg_part_{agg_idx}_{j}[s * MAX_GROUPS + gi];"
            );
        }
        "=min" => {
            let _ = writeln!(
                out,
                "        total_{agg_idx}_{j} = min(total_{agg_idx}_{j}, tg_part_{agg_idx}_{j}[s * MAX_GROUPS + gi]);"
            );
        }
        "=max" => {
            let _ = writeln!(
                out,
                "        total_{agg_idx}_{j} = max(total_{agg_idx}_{j}, tg_part_{agg_idx}_{j}[s * MAX_GROUPS + gi]);"
            );
        }
        _ => unreachable!(),
    }
    let _ = writeln!(out, "      }}");

    // Flush to device atomic. Skip when the combined value is still the
    // identity — saves a CAS on TGs that saw no rows for this group.
    match kind {
        ReduceKind::F32Sum => {
            let _ = writeln!(out, "      if (total_{agg_idx}_{j} != 0.0f) {{");
            let _ = writeln!(
                out,
                "        uint old_bits = atomic_load_explicit(&out_{agg_idx}_{j}[gi], memory_order_relaxed);"
            );
            let _ = writeln!(out, "        while (true) {{");
            let _ = writeln!(out, "          float cur = as_type<float>(old_bits);");
            let _ = writeln!(
                out,
                "          uint next_bits = as_type<uint>(cur + total_{agg_idx}_{j});"
            );
            let _ = writeln!(
                out,
                "          if (atomic_compare_exchange_weak_explicit(&out_{agg_idx}_{j}[gi], &old_bits, next_bits, memory_order_relaxed, memory_order_relaxed)) break;"
            );
            let _ = writeln!(out, "        }}");
            let _ = writeln!(out, "      }}");
        }
        ReduceKind::F32Min => {
            // Skip if total is still +INF.
            let _ = writeln!(
                out,
                "      if (!(isinf(total_{agg_idx}_{j}) && total_{agg_idx}_{j} > 0.0f)) {{"
            );
            let _ = writeln!(
                out,
                "        uint old_bits = atomic_load_explicit(&out_{agg_idx}_{j}[gi], memory_order_relaxed);"
            );
            let _ = writeln!(out, "        while (true) {{");
            let _ = writeln!(out, "          float cur = as_type<float>(old_bits);");
            let _ = writeln!(out, "          if (!(total_{agg_idx}_{j} < cur)) break;");
            let _ = writeln!(
                out,
                "          uint next_bits = as_type<uint>(total_{agg_idx}_{j});"
            );
            let _ = writeln!(
                out,
                "          if (atomic_compare_exchange_weak_explicit(&out_{agg_idx}_{j}[gi], &old_bits, next_bits, memory_order_relaxed, memory_order_relaxed)) break;"
            );
            let _ = writeln!(out, "        }}");
            let _ = writeln!(out, "      }}");
        }
        ReduceKind::F32Max => {
            // Skip if total is still -INF.
            let _ = writeln!(
                out,
                "      if (!(isinf(total_{agg_idx}_{j}) && total_{agg_idx}_{j} < 0.0f)) {{"
            );
            let _ = writeln!(
                out,
                "        uint old_bits = atomic_load_explicit(&out_{agg_idx}_{j}[gi], memory_order_relaxed);"
            );
            let _ = writeln!(out, "        while (true) {{");
            let _ = writeln!(out, "          float cur = as_type<float>(old_bits);");
            let _ = writeln!(out, "          if (!(total_{agg_idx}_{j} > cur)) break;");
            let _ = writeln!(
                out,
                "          uint next_bits = as_type<uint>(total_{agg_idx}_{j});"
            );
            let _ = writeln!(
                out,
                "          if (atomic_compare_exchange_weak_explicit(&out_{agg_idx}_{j}[gi], &old_bits, next_bits, memory_order_relaxed, memory_order_relaxed)) break;"
            );
            let _ = writeln!(out, "        }}");
            let _ = writeln!(out, "      }}");
        }
        ReduceKind::I32Sum | ReduceKind::U32Sum | ReduceKind::Counter => {
            let _ = writeln!(out, "      if (total_{agg_idx}_{j} != 0) {{");
            let _ = writeln!(
                out,
                "        atomic_fetch_add_explicit(&out_{agg_idx}_{j}[gi], total_{agg_idx}_{j}, memory_order_relaxed);"
            );
            let _ = writeln!(out, "      }}");
        }
        ReduceKind::I32Min => {
            let _ = writeln!(out, "      if (total_{agg_idx}_{j} != INT_MAX) {{");
            let _ = writeln!(
                out,
                "        atomic_fetch_min_explicit(&out_{agg_idx}_{j}[gi], total_{agg_idx}_{j}, memory_order_relaxed);"
            );
            let _ = writeln!(out, "      }}");
        }
        ReduceKind::I32Max => {
            let _ = writeln!(out, "      if (total_{agg_idx}_{j} != INT_MIN) {{");
            let _ = writeln!(
                out,
                "        atomic_fetch_max_explicit(&out_{agg_idx}_{j}[gi], total_{agg_idx}_{j}, memory_order_relaxed);"
            );
            let _ = writeln!(out, "      }}");
        }
        ReduceKind::U32Min => {
            let _ = writeln!(out, "      if (total_{agg_idx}_{j} != UINT_MAX) {{");
            let _ = writeln!(
                out,
                "        atomic_fetch_min_explicit(&out_{agg_idx}_{j}[gi], total_{agg_idx}_{j}, memory_order_relaxed);"
            );
            let _ = writeln!(out, "      }}");
        }
        ReduceKind::U32Max => {
            let _ = writeln!(out, "      if (total_{agg_idx}_{j} != 0u) {{");
            let _ = writeln!(
                out,
                "        atomic_fetch_max_explicit(&out_{agg_idx}_{j}[gi], total_{agg_idx}_{j}, memory_order_relaxed);"
            );
            let _ = writeln!(out, "      }}");
        }
    }
}

// ---------- Expression-agg helpers (Task 13) -------------------------------

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
