"""Walk a Polars expression IR and identify the maximal fused subtree.

Public API:
    analyze_expression(expr: pl.Expr, schema: Schema) -> PyFusionScope | None

Returns a constructed PyFusionScope if the entire expression maps to a
supported chain of compute ops. Returns None on the first unsupported node
(no partial fusion in this chunk).

Implementation note: traverses the JSON form of `expr.meta.serialize()`.
The JSON vocabulary is stable in py-1.40.1. When bumping Polars, this file
is the first to verify against `expr.meta.serialize(format="json")`.
"""

from __future__ import annotations

import json
from typing import Any

import polars as pl

from polars_metal._native import PyFusionScope

# ── Operator-name maps ─────────────────────────────────────────────────────

# BinaryExpr.op string -> our OpId string.
_BINOP_MAP: dict[str, str] = {
    "Plus": "Add",
    "Minus": "Sub",
    "Multiply": "Mul",
    "Divide": "Div",
    "TrueDivide": "Div",
    "Modulus": "Mod",
    "Eq": "Eq",
    "NotEq": "Ne",
    "Lt": "Lt",
    "LtEq": "Le",
    "Gt": "Gt",
    "GtEq": "Ge",
    "And": "LogicalAnd",
    "Or": "LogicalOr",
}

# Plain-string Function names -> OpId string. (Functions whose `function` field
# is a bare string rather than a namespaced object.)
_FUNCTION_PLAIN_MAP: dict[str, str] = {
    "Abs": "Abs",
    "Negate": "Neg",
    "Exp": "Exp",
    # Log is special-cased - it's 2-arg (input + base) and we map based on the
    # base literal value to Log / Log2 / Log10 / Log1p.
    "Atan2": "Atan2",
}

# {"Trigonometry": <name>} -> OpId string.
_TRIG_MAP: dict[str, str] = {
    "Sin": "Sin",
    "Cos": "Cos",
    "Tan": "Tan",
    "Sinh": "Sinh",
    "Cosh": "Cosh",
    "Tanh": "Tanh",
    "ArcSin": "Asin",
    "ArcCos": "Acos",
    "ArcTan": "Atan",
}

# {"Pow": <name>} -> OpId. "Sqrt" is unary; "Cbrt" is unary; "Generic" is the
# binary `a**b` form (our Pow). `Square` shows up as Pow("Generic") with rhs=2.
_POW_MAP: dict[str, str] = {
    "Sqrt": "Sqrt",
    "Cbrt": "Cbrt",
}

# {"Boolean": <name>} -> OpId.
_BOOLEAN_MAP: dict[str, str] = {
    "Not": "LogicalNot",
}

# Agg dict key -> OpId.
_AGG_MAP: dict[str, str] = {
    "Sum": "Sum",
    "Mean": "Mean",
    "Min": "Min",
    "Max": "Max",
    "Std": "Std",
    "Var": "Var",
    "ArgMin": "ArgMin",
    "ArgMax": "ArgMax",
}

# Cast.dtype string ({"Literal": "Float32"} etc.) -> OpId.
_CAST_DTYPE_MAP: dict[str, str] = {
    "Float32": "CastF32",
    "Float64": "CastF64",
    "Int32": "CastI32",
    "Boolean": "CastBool",
}


class _Aborted(Exception):
    """Raised inside the visitor when an unsupported node is encountered.

    `analyze_expression` catches this and returns None.
    """


def analyze_expression(expr: pl.Expr, schema: dict[str, Any]) -> PyFusionScope | None:
    """Walk `expr` and build a PyFusionScope.

    Returns None if any node is unsupported (no partial fusion this chunk).
    """
    try:
        raw = expr.meta.serialize(format="json")
        tree = json.loads(raw)
    except Exception:
        return None

    try:
        scope = PyFusionScope()
        idx = _visit(tree, schema, scope)
        scope.mark_output(idx)
        return scope
    except _Aborted:
        return None


def _visit(node: Any, schema: dict[str, Any], scope: PyFusionScope) -> int:
    """Recursive descent over a JSON expression node. Returns the NodeIdx."""
    if not isinstance(node, dict) or len(node) != 1:
        raise _Aborted

    (kind, body) = next(iter(node.items()))

    if kind == "Column":
        name = body
        dtype = schema.get(name)
        if dtype is None:
            raise _Aborted
        return scope.add_input(name, _dtype_to_input_str(dtype))

    if kind == "Literal":
        val = _extract_literal_value(body)
        # Encode the literal as a synthetic F32 input. The subgraph builder
        # (Phase 4) will materialize the constant when building the MLX graph.
        return scope.add_input(f"__lit_{val}", "F32")

    if kind == "BinaryExpr":
        op_str = body.get("op")
        op_id = _BINOP_MAP.get(op_str)
        if op_id is None:
            raise _Aborted
        left = _visit(body["left"], schema, scope)
        right = _visit(body["right"], schema, scope)
        return scope.push_op(op_id, [left, right])

    if kind == "Cast":
        dtype_field = body.get("dtype")
        # dtype is wrapped as {"Literal": "Float32"} or similar.
        if not isinstance(dtype_field, dict):
            raise _Aborted
        dtype_name = dtype_field.get("Literal")
        op_id = _CAST_DTYPE_MAP.get(dtype_name)
        if op_id is None:
            raise _Aborted
        child = _visit(body["expr"], schema, scope)
        return scope.push_op(op_id, [child])

    if kind == "Function":
        return _visit_function(body, schema, scope)

    if kind == "Agg":
        return _visit_agg(body, schema, scope)

    if kind == "Ternary":
        cond = _visit(body["predicate"], schema, scope)
        then_v = _visit(body["truthy"], schema, scope)
        else_v = _visit(body["falsy"], schema, scope)
        return scope.push_op("Where", [cond, then_v, else_v])

    raise _Aborted


def _visit_function(body: dict[str, Any], schema: dict[str, Any], scope: PyFusionScope) -> int:
    """Handle a {"Function": {"input": [...], "function": <variant>}} node."""
    fn = body.get("function")
    inputs = body.get("input", [])

    # function: plain string (e.g. "Abs", "Negate", "Exp", "Log", "Atan2").
    if isinstance(fn, str):
        if fn == "Log":
            return _visit_log(inputs, schema, scope)
        op_id = _FUNCTION_PLAIN_MAP.get(fn)
        if op_id is None:
            raise _Aborted
        child_idxs = [_visit(child, schema, scope) for child in inputs]
        return scope.push_op(op_id, child_idxs)

    # function: namespaced dict (e.g. {"Trigonometry": "Sin"}).
    if isinstance(fn, dict) and len(fn) == 1:
        (ns, name) = next(iter(fn.items()))
        if ns == "Trigonometry":
            op_id = _TRIG_MAP.get(name)
            if op_id is None:
                raise _Aborted
            if len(inputs) != 1:
                raise _Aborted
            child = _visit(inputs[0], schema, scope)
            return scope.push_op(op_id, [child])
        if ns == "Pow":
            # Pow:Sqrt / Pow:Cbrt are unary; Pow:Generic is binary (x ** y).
            # Generic with a literal y=2 is `square` - we emit Pow rather than
            # special-casing Square (the subgraph builder can do that).
            if name == "Generic":
                if len(inputs) != 2:
                    raise _Aborted
                left = _visit(inputs[0], schema, scope)
                right = _visit(inputs[1], schema, scope)
                return scope.push_op("Pow", [left, right])
            op_id = _POW_MAP.get(name)
            if op_id is None:
                raise _Aborted
            if len(inputs) != 1:
                raise _Aborted
            child = _visit(inputs[0], schema, scope)
            return scope.push_op(op_id, [child])
        if ns == "Boolean":
            op_id = _BOOLEAN_MAP.get(name)
            if op_id is None:
                raise _Aborted
            if len(inputs) != 1:
                raise _Aborted
            child = _visit(inputs[0], schema, scope)
            return scope.push_op(op_id, [child])

    # StringExpr, ListExpr, TemporalExpr, etc. are all unsupported in M4.
    raise _Aborted


def _visit_log(inputs: list, schema: dict[str, Any], scope: PyFusionScope) -> int:
    """Polars represents log() as Log(x, base). Map to Log/Log2/Log10 based
    on the base literal, treating the natural-log base (e) as our Log."""
    if len(inputs) != 2:
        raise _Aborted
    base = inputs[1]
    base_val = None
    if isinstance(base, dict) and "Literal" in base:
        base_val = _extract_literal_value(base["Literal"])
    if base_val is None:
        raise _Aborted
    if abs(base_val - 2.718281828459045) < 1e-9:
        op_id = "Log"
    elif base_val == 2 or base_val == 2.0:
        op_id = "Log2"
    elif base_val == 10 or base_val == 10.0:
        op_id = "Log10"
    else:
        # Arbitrary base - not in our op set this chunk.
        raise _Aborted
    child = _visit(inputs[0], schema, scope)
    return scope.push_op(op_id, [child])


def _visit_agg(body: dict[str, Any], schema: dict[str, Any], scope: PyFusionScope) -> int:
    """{"Agg": {"Sum": <child>}} or {"Agg": {"Std": [<child>, ddof]}}."""
    if not isinstance(body, dict) or len(body) != 1:
        raise _Aborted
    (name, payload) = next(iter(body.items()))
    op_id = _AGG_MAP.get(name)
    if op_id is None:
        raise _Aborted
    # Std/Var carry a ddof param; we accept any ddof and let the engine handle
    # the Bessel correction (Polars default is ddof=1, MLX default is ddof=0).
    if isinstance(payload, list):
        if not payload:
            raise _Aborted
        child_node = payload[0]
    else:
        child_node = payload
    child = _visit(child_node, schema, scope)
    return scope.push_op(op_id, [child])


def _extract_literal_value(payload: Any) -> Any:
    """Pull the scalar value out of a Polars literal JSON payload.

    Literals come in several wrappings; we unwrap until we hit a primitive.
    """
    while isinstance(payload, dict) and len(payload) == 1:
        (_, inner) = next(iter(payload.items()))
        if isinstance(inner, (int, float, bool, str)):
            return inner
        payload = inner
    if isinstance(payload, (int, float, bool, str)):
        return payload
    raise _Aborted


# Polars dtype -> PyFusionScope input-dtype string, for the integer family.
# Used by both `_dtype_to_input_str` (leaf staging) and the monomorphic
# output-dtype inference (`_scan_binding_col_dtype`).
_INT_DTYPE_TO_STR: dict[Any, str] = {
    pl.Int8: "I8",
    pl.Int16: "I16",
    pl.Int32: "I32",
    pl.Int64: "I64",
    pl.UInt8: "U8",
    pl.UInt16: "U16",
    pl.UInt32: "U32",
    pl.UInt64: "U64",
}


def _dtype_to_input_str(dtype: Any) -> str:
    """Map a Polars dtype to the input-dtype string the PyFusionScope accepts."""
    if dtype == pl.Float32:
        return "F32"
    if dtype == pl.Float64:
        return "F64"
    if dtype == pl.Boolean:
        return "Bool"
    for pl_dtype, s in _INT_DTYPE_TO_STR.items():
        if dtype == pl_dtype:
            return s
    s = str(dtype)
    if s.startswith("Array(Float32"):
        inner_d = _extract_array_dim(s)
        return f"ArrayF32({inner_d})"
    if s == "List(Float32)":
        return "ListF32"
    raise _Aborted


def _extract_array_dim(s: str) -> int:
    """`Array(Float32, 768)` -> 768."""
    import re

    m = re.search(r",\s*(\d+)\)$", s)
    if m:
        return int(m.group(1))
    raise _Aborted


# ── IR-arena analyzer (for walker integration) ─────────────────────────────
#
# When the walker is invoked, the Polars expression has already been lowered
# from `pl.Expr` to IR-arena nodes accessed via `nt.view_expression(node_id)`.
# These don't expose `.meta.serialize()` — we walk the arena directly.
#
# The IR-arena vocabulary maps onto the same OpId set, with slight encoding
# differences from the JSON form:
#   - Function.function_data is a flat tuple like ('sin',) - no namespacing.
#   - Cast.dtype is a Polars dtype object directly.
#   - BinaryExpr.op is a pl.Operator enum member.

# Function-name string (from function_data[0]) -> OpId string.
# Polars flattens namespaced fns into bare names at the IR layer.
_IR_FUNCTION_MAP: dict[str, str] = {
    "sin": "Sin",
    "cos": "Cos",
    "tan": "Tan",
    "sinh": "Sinh",
    "cosh": "Cosh",
    "tanh": "Tanh",
    "arcsin": "Asin",
    "arccos": "Acos",
    "arctan": "Atan",
    "arctan2": "Atan2",
    "exp": "Exp",
    "sqrt": "Sqrt",
    "cbrt": "Cbrt",
    "abs": "Abs",
    "negate": "Neg",
    "floor": "Floor",
    "ceil": "Ceil",
    "round": "Round",
    # Cumulative scans (Phase 7 Task 28 + cum_prod/max/min). Polars encodes
    # each as a Function whose function_data is (name, reverse: bool). The MLX
    # bindings (CumSum/CumProd/CumMax/CumMin, wired in fusion/subgraph.rs) are
    # forward-only, so reverse=True falls back to CPU (see the guards in
    # `_gather_leaves_ir` / `_visit_ir_ops`). Null-bearing scans also fall back
    # (their null propagation isn't the elementwise AND rule — see `null_mode_ir`).
    "cum_sum": "CumSum",
    "cum_prod": "CumProd",
    "cum_max": "CumMax",
    "cum_min": "CumMin",
}

# OpId strings for the cumulative scans (forward-only MLX bindings; null
# propagation isn't the elementwise AND rule).
_CUMULATIVE_OPS: frozenset[str] = frozenset({"CumSum", "CumProd", "CumMax", "CumMin"})


def _is_reverse_cumulative(fn_name: str, fd: tuple) -> bool:
    """True for a reverse cumulative scan, which has no MLX forward-only
    binding and must fall back to CPU. `fd` is the Function.function_data
    tuple, e.g. ('cum_sum', True)."""
    return _IR_FUNCTION_MAP.get(fn_name) in _CUMULATIVE_OPS and len(fd) > 1 and bool(fd[1])


# ── shift / int_range special-cased recognition (M5 rolling Task 5) ─────────
#
# Empirically pinned IR shapes (py-1.40.1; verified via nt.view_expression):
#
#   shift:  Function function_data=('shift',) with
#           input=[value_child, Literal(offset: Int32)].  The offset is a
#           *child Literal*, not a kwarg / not in function_data. A forward
#           shift fills the leading `offset` rows with null (structural).
#           `shift_and_fill` is a DISTINCT function name (3-input: value,
#           offset, fill) — it is not in any map and aborts on its own, so
#           plain `shift` never carries a fill_value to reject here.
#
#   int_range(len()):  OPAQUE. nt.view_expression raises
#           NotImplementedError('range'), so the walkers cannot recognize it
#           at all (they hit `except Exception: raise _Aborted`). The RowIndex
#           code paths below exist for faithfulness / future unlock but are
#           currently unreachable through the walker. RowIndex is a 0-input
#           generator (iota); it never contributes a null.
#
# Neither `shift` nor `int_range` is added to _IR_FUNCTION_MAP: the generic
# path there would (wrongly) recurse the offset literal as a value input and
# push with no scalar param.

_SHIFT_FN = "shift"
_INT_RANGE_FN = "int_range"


def _shift_offset(nt: Any, fn_inputs: list) -> int:
    """Extract and validate a forward `shift`'s integer offset.

    Aborts (CPU fallback) unless the offset is a compile-time, non-negative
    integer Literal — the MLX Shift binding is forward-only and a non-literal
    / negative / expression offset can't be reproduced. A `shift` node has
    exactly 2 inputs: [value_child, offset_literal]; a 3-input node is
    `shift_and_fill` (different function name) and never reaches here."""
    if len(fn_inputs) != 2:
        raise _Aborted
    try:
        off_node = nt.view_expression(fn_inputs[1])
    except Exception as e:
        raise _Aborted from e
    if type(off_node).__name__ != "Literal":
        raise _Aborted
    val = getattr(off_node, "value", None)
    if not isinstance(val, int) or isinstance(val, bool):
        raise _Aborted
    if val < 0:
        # Forward-only binding; negative (look-ahead) shift must go to CPU.
        raise _Aborted
    return val


def has_structural_null_op(nt: Any, node_id: int, schema: dict[str, Any]) -> bool:
    """True iff the IR subtree at ``node_id`` contains a ``shift`` — a
    structural-null op whose leading rows are null EVEN on null-free input.

    The walker uses this to decide whether to build a validity subgraph for an
    otherwise-null-free binding (the fast path skips the null subgraph when
    inputs are null-free, but Shift's head-nulls are present regardless). Best
    effort: any unviewable node (opaque IR) simply reports no structural op for
    that branch — such a binding falls back to CPU elsewhere anyway."""
    try:
        node = nt.view_expression(node_id)
    except Exception:
        return False
    cls = type(node).__name__
    if cls == "Function":
        fd = getattr(node, "function_data", ())
        if fd and str(fd[0]).lower() == _SHIFT_FN:
            return True
    for attr in ("left", "right", "expr", "predicate", "truthy", "falsy"):
        cid = getattr(node, attr, None)
        if isinstance(cid, int) and has_structural_null_op(nt, cid, schema):
            return True
    return any(has_structural_null_op(nt, cid, schema) for cid in list(getattr(node, "input", [])))


def analyze_ir_expression(nt: Any, node_id: int, schema: dict[str, Any]) -> PyFusionScope | None:
    """Walk a Polars IR expression by arena ID and build a PyFusionScope.

    Used by the walker after Polars has optimized the LazyFrame into arena
    form. `nt` is a NodeTraverser; `node_id` is the int arena ID of the
    expression root (typically from `binding.node` in an HStack/Select).

    Returns None on the first unsupported node.
    """
    result = analyze_ir_with_columns(nt, node_id, schema)
    return None if result is None else result[0]


# Input-dtype strings the B1 monomorphic-int fused path can produce as output.
# Float32 is handled by the legacy F32-only path; B1 extends recognition to a
# chain whose every column leaf shares one of these integer dtypes. A mixed
# int/F32 chain (or a chain with no columns) aborts -> CPU fallback (B2 owns
# generalization), so B1 never returns a wrong answer on a shape it can't
# statically infer the output dtype for.
_MONOMORPHIC_OUT_DTYPES: frozenset[str] = frozenset(
    {"I8", "I16", "I32", "I64", "U8", "U16", "U32", "U64"}
)


def _scan_binding_col_dtype(nt: Any, node_id: int, schema: dict[str, Any]) -> str | None:
    """Walk the expr subtree and return the single common column-leaf
    InputDtype string, or ``None`` if the chain has zero columns, mixes
    multiple column dtypes, or contains a node we don't recognize.

    This drives B1's monomorphic-output inference: for a chain like
    ``col + 1`` where every Column leaf shares one dtype, the fused output
    dtype equals that dtype. Literals are skipped (they're staged at the
    column width, not their own). A heterogeneous chain (mixed int/F32, or an
    explicit non-F32 Cast) returns ``None`` so the caller falls back to CPU —
    B2 will generalize promotion semantics.

    Returns the wire dtype string (e.g. ``"I64"``) or ``None``."""
    found: set[str] = set()

    def walk(nid: int) -> bool:
        """Return False to signal an unrecognized node (caller aborts)."""
        try:
            node = nt.view_expression(nid)
        except Exception:
            return False
        cls = type(node).__name__
        if cls == "Column":
            name = getattr(node, "name", None)
            if name is None:
                return False
            dtype = schema.get(str(name))
            if dtype is None:
                return False
            try:
                found.add(_dtype_to_input_str(dtype))
            except _Aborted:
                return False
            return True
        if cls == "Literal":
            return True
        if cls == "BinaryExpr":
            return walk(node.left) and walk(node.right)
        if cls == "Cast":
            # Only an F32 Cast is honored by `_visit_ir_ops`; a non-F32 Cast
            # already aborts there, so we don't need to model it. An F32 Cast
            # makes the output F32 (handled by the legacy path), so a chain
            # containing one is not monomorphic-int — signal "unrecognized"
            # for this inference, which routes it via the F32 path / CPU.
            return False
        if cls == "Ternary":
            return walk(node.predicate) and walk(node.truthy) and walk(node.falsy)
        if cls == "Agg":
            args = list(getattr(node, "arguments", []))
            return bool(args) and all(walk(a) for a in args)
        if cls == "Function":
            return all(walk(cid) for cid in list(getattr(node, "input", [])))
        return False

    if not walk(node_id):
        return None
    if len(found) != 1:
        # Zero columns (literal-only) or mixed column dtypes: not monomorphic.
        return None
    return next(iter(found))


def analyze_ir_with_columns(
    nt: Any, node_id: int, schema: dict[str, Any]
) -> tuple[PyFusionScope, list[tuple[str, str | float]], str] | None:
    """Like `analyze_ir_expression`, but also returns the ordered list of
    input descriptors AND the statically-inferred output dtype string. Each
    descriptor is `("col", column_name)` for real columns or `("lit", value)`
    for literal scalars. Order matches the scope's input order, so the executor
    can construct input buffers in the correct sequence.

    The third return element ``out_dtype_str`` is the wire dtype the fused
    output will carry. For a monomorphic-int chain (every column leaf shares
    one integer dtype, e.g. ``col + 1`` over Int64) it is that dtype (``"I64"``)
    and the integer literal is staged at the column width (so MLX doesn't
    promote ``int + f32_literal`` to f32). Otherwise it is ``"F32"`` — the
    legacy float path, where literals stage at F32 as before.

    Two-pass: pass 1 collects every leaf (Column/Literal) into the scope in
    DFS order so input NodeIdxs are contiguous from 0. Pass 2 walks the tree
    again and pushes ops; on a leaf it returns the precomputed input idx.
    Single-pass interleaving doesn't work because the executor assumes a
    flat `[inputs..., ops...]` layout (see fusion/subgraph.rs build_op)."""
    # Statically infer the output dtype. A monomorphic-int chain reports its
    # column dtype (and stages literals at that width); everything else
    # (F32 chains, mixed, F32-cast) stays on the F32 path with F32 literals.
    col_dtype = _scan_binding_col_dtype(nt, node_id, schema)
    if col_dtype in _MONOMORPHIC_OUT_DTYPES:
        out_dtype_str = col_dtype
        lit_dtype_str = col_dtype
    else:
        out_dtype_str = "F32"
        lit_dtype_str = "F32"
    try:
        scope = PyFusionScope()
        descriptors: list[tuple[str, str | float]] = []
        leaf_idx: dict[int, int] = {}
        # Dedup tables — semantically equal leaves share a scope input slot.
        # Without this, a repeated literal like `d2r` in haversine (used 3x)
        # would produce three separate F32 inputs, each materialized as an
        # n_rows-wide broadcast buffer. Same column referenced N times
        # would similarly stage N copies through the FFI boundary.
        col_dedup: dict[str, int] = {}
        lit_dedup: dict[float, int] = {}
        _gather_leaves_ir(
            nt,
            node_id,
            schema,
            scope,
            descriptors,
            leaf_idx,
            col_dedup,
            lit_dedup,
            lit_dtype_str,
        )
        idx = _visit_ir_ops(nt, node_id, schema, scope, leaf_idx)
        scope.mark_output(idx)
        return scope, descriptors, out_dtype_str
    except _Aborted:
        return None


# Lowercase Agg.name (IR form) -> reduction OpId string. Used only by the
# select-reduction path (`analyze_ir_reduction`). argmin/argmax are excluded:
# they return integer indices, not an F32 scalar.
_REDUCTION_OP: dict[str, str] = {
    "sum": "Sum",
    "mean": "Mean",
    "min": "Min",
    "max": "Max",
    "std": "Std",
    "var": "Var",
}


# B2: GPU-admissible integer reductions. The fused reduction output dtype must
# equal the Polars output dtype with NO fold-back cast (there is no CastI64 /
# CastU64 op), so we admit exactly the (op, dtype) pairs where MLX's native
# reduction dtype already matches Polars:
#   - sum: MLX keeps the input width; Polars keeps it ONLY for the 32/64-bit
#     widths (Int32→Int32, Int64→Int64, UInt32→UInt32, UInt64→UInt64). Polars
#     upcasts narrow-int sum to Int64/UInt64 while MLX widens to int32/uint32 →
#     mismatch → those stay CPU.
#   - min/max: both MLX and Polars preserve the input width for all 8 widths.
#   - mean/std/var: MLX→float32, Polars→Float64 → mismatch → CPU.
_SUM_ADMIT: dict[Any, str] = {
    pl.Int32: "I32",
    pl.Int64: "I64",
    pl.UInt32: "U32",
    pl.UInt64: "U64",
}
_MINMAX_ADMIT: dict[Any, str] = {
    pl.Int8: "I8",
    pl.Int16: "I16",
    pl.Int32: "I32",
    pl.Int64: "I64",
    pl.UInt8: "U8",
    pl.UInt16: "U16",
    pl.UInt32: "U32",
    pl.UInt64: "U64",
}


def _int_reduction_out_dtype(agg_kind: str, col_dtype: Any) -> str | None:
    """Wire output-dtype string for a GPU-admissible integer reduction, or
    ``None`` when the (op, dtype) pair is not GPU-admissible (→ CPU fallback).

    Admits int ``sum`` for {Int32, Int64, UInt32, UInt64} and int ``min``/
    ``max`` for all 8 integer widths — exactly the pairs where MLX's native
    reduction dtype equals the Polars output dtype (no fold-back cast). All
    other int reductions (narrow ``sum``, ``mean``/``std``/``var``) return
    ``None`` so the caller aborts to CPU.
    """
    if agg_kind == "sum":
        return _SUM_ADMIT.get(col_dtype)
    if agg_kind in ("min", "max"):
        return _MINMAX_ADMIT.get(col_dtype)
    return None


def analyze_ir_reduction(
    nt: Any, agg_node_id: int, schema: dict[str, Any]
) -> tuple[PyFusionScope, list[tuple[str, str | float]], str, bool, int, str] | None:
    """Analyze a full-column reduction `agg(expr)` (the terminus of a
    `select(pl.col(...).std())`-shaped node) into a fused scope whose single
    output is the scalar reduction.

    Returns `(scope, descriptors, agg_kind, is_chain, arg_id, out_dtype_str)`
    or None if not
    fusion-eligible. `arg_id` is the reduction argument's arena id, so the
    walker can classify the chain's null mode (a null-bearing *elementwise*
    chain reduces on the GPU after `drop_nulls` — positions don't matter for a
    reduction). `agg_kind` (lowercase) lets the dispatch apply the Bessel
    correction — MLX uses population variance (ddof=0); Polars defaults to
    sample (ddof=1). `is_chain` is True when the reduction's argument is a
    compute chain (≥1 op), False for a bare column.

    The agg's *argument* subtree is analyzed with the shared leaf/op walkers
    (which never recurse into Agg nodes), then the reduction op is pushed
    explicitly as the terminus. Two shapes:
      - bare column (`is_chain=False`): the routing layer only sends compute-
        bound ops (std/var) to GPU on their own; the dispatch handles nulls /
        <2-row inputs by replaying the reduction on the source column.
      - compute chain (`is_chain=True`): the chain amortizes the dispatch floor,
        so any reduction op is worth fusing. The chain's null propagation can't
        be replayed on the source, so the walker only routes a chain whose
        input columns are null-free (else CPU).

    Admit rule (B2): every column input must share ONE dtype that is either
    Float32 (the legacy float path → ``out_dtype_str == "F32"``) or a
    GPU-admissible integer reduction (see ``_int_reduction_out_dtype``: int
    ``sum`` for {Int32, Int64, UInt32, UInt64}, int ``min``/``max`` for all 8
    widths → ``out_dtype_str`` is the matching int tag, no fold-back cast).
    Anything else (narrow-int ``sum``, int ``mean``/``std``/``var``, F64, a
    mixed-dtype chain) falls back to CPU. ``out_dtype_str`` is the 6th return
    element so the dispatch pre-allocates the right-width output array.
    """
    try:
        agg_node = nt.view_expression(agg_node_id)
    except Exception:
        return None
    if type(agg_node).__name__ != "Agg":
        return None
    kind = str(getattr(agg_node, "name", "")).lower()
    op_id = _REDUCTION_OP.get(kind)
    if op_id is None:
        return None
    args = list(getattr(agg_node, "arguments", []))
    if len(args) != 1:
        return None
    arg_id = args[0]
    try:
        scope = PyFusionScope()
        descriptors: list[tuple[str, str | float]] = []
        leaf_idx: dict[int, int] = {}
        col_dedup: dict[str, int] = {}
        lit_dedup: dict[float, int] = {}
        _gather_leaves_ir(nt, arg_id, schema, scope, descriptors, leaf_idx, col_dedup, lit_dedup)
        inner_idx = _visit_ir_ops(nt, arg_id, schema, scope, leaf_idx)
        is_chain = scope.n_ops() > 0
        # Need at least one real column (literal-only reductions are degenerate).
        col_descriptors = [(k, p) for k, p in descriptors if k == "col"]
        if not col_descriptors:
            raise _Aborted

        # Resolve the binding's single column dtype. B2 admits only monomorphic
        # reductions (every column leaf shares one dtype); a mixed-dtype chain
        # falls back so we never mis-infer the output width.
        col_dtypes = {schema.get(p) for _, p in col_descriptors}
        if len(col_dtypes) != 1:
            raise _Aborted
        col_dtype = next(iter(col_dtypes))

        # Output-dtype inference + GPU-admit gate:
        #   - Float32 column → "F32" (the existing float path, unchanged).
        #   - GPU-admissible int (op, dtype) → its wire int dtype (no cast).
        #   - anything else (narrow int sum, int mean/std/var, F64, …) → CPU.
        if col_dtype == pl.Float32:
            out_dtype_str = "F32"
        else:
            out_dtype_str = _int_reduction_out_dtype(kind, col_dtype)
            if out_dtype_str is None:
                raise _Aborted

        if not is_chain and (len(descriptors) != 1 or descriptors[0][0] != "col"):
            # Bare reduction must be a single column.
            raise _Aborted
        red_idx = scope.push_op(op_id, [inner_idx])
        scope.mark_output(red_idx)
        return scope, descriptors, kind, is_chain, arg_id, out_dtype_str
    except _Aborted:
        return None


def build_sort_scope(col_name: str) -> PyFusionScope:
    """Build a one-op fused scope that sorts an F32 column ascending via MLX.

    The dispatch reverses (for descending) and slices (for top_k) on the host;
    MLX `Sort` is the only GPU op. Used by the walker's Sort path (Task 27).
    """
    scope = PyFusionScope()
    leaf = scope.add_input(col_name, "F32")
    scope.mark_output(scope.push_op("Sort", [leaf]))
    return scope


def null_mode_ir(nt: Any, node_id: int, schema: dict[str, Any]) -> str | None:
    """Classify how nulls propagate through a fused HStack expression, so the
    walker can keep null-bearing inputs on the GPU instead of falling back.

    Returns:
      - ``"elementwise"`` — output is null iff *any* input column is null at
        that row (arithmetic / transcendental / cast / comparison chains). The
        dispatch combines the input columns' null masks and attaches them to
        the output; the value compute stays on the GPU.
      - ``"where"`` — the expression contains a ``Ternary``/``Where`` whose
        null mask is data-dependent (``cond_null or (cond ? then_null :
        else_null)``); handled by a validity subgraph at dispatch.
      - ``None`` — null semantics we don't reproduce on the fused path:
        Kleene 3-valued ``And``/``Or``, null-skipping reductions (``Agg``), or
        cumulative scans (``CumSum``). The walker falls these back to CPU when
        an input has nulls.

    Recognizes exactly the node set ``_visit_ir_ops`` accepts, so a scope the
    value-graph builder admitted always classifies here (never desyncs).
    """
    state = {"where": False}
    try:
        _classify_null_ir(nt, node_id, schema, state)
    except _Aborted:
        return None
    return "where" if state["where"] else "elementwise"


def _classify_null_ir(nt: Any, node_id: int, schema: dict[str, Any], state: dict) -> None:
    """DFS the IR recording whether a data-dependent ``Where`` appears; abort
    on any op whose null semantics the fused path can't reproduce."""
    try:
        node = nt.view_expression(node_id)
    except Exception as e:
        raise _Aborted from e
    cls = type(node).__name__

    if cls in ("Column", "Literal"):
        return

    if cls == "BinaryExpr":
        op = getattr(node, "op", None)
        op_name = getattr(op, "name", None) or str(op).rsplit(".", 1)[-1]
        op_id = _BINOP_MAP.get(op_name)
        if op_id is None or op_id in ("LogicalAnd", "LogicalOr"):
            # LogicalAnd/Or use Kleene 3-valued logic (false&null=false), which
            # is not the AND-of-validity rule; refuse rather than mis-null.
            raise _Aborted
        _classify_null_ir(nt, node.left, schema, state)
        _classify_null_ir(nt, node.right, schema, state)
        return

    if cls == "Cast":
        if getattr(node, "dtype", None) != pl.Float32:
            raise _Aborted
        _classify_null_ir(nt, node.expr, schema, state)
        return

    if cls == "Function":
        fd = getattr(node, "function_data", ())
        if not fd:
            raise _Aborted
        fn_name = str(fd[0]).lower()
        if fn_name == _SHIFT_FN:
            # Shift has STRUCTURAL leading nulls (the front `w` rows become
            # null even on null-free input). Force the validity-subgraph path
            # so the dispatch reproduces them. Recurse only the value child;
            # the offset literal carries no null.
            fn_inputs = list(getattr(node, "input", []))
            _shift_offset(nt, fn_inputs)  # validate (negative/expr offset -> CPU)
            state["where"] = True
            _classify_null_ir(nt, fn_inputs[0], schema, state)
            return
        if fn_name == _INT_RANGE_FN:
            # Terminal iota generator — never null, no children.
            return
        if _is_reverse_cumulative(fn_name, fd):
            raise _Aborted
        if fn_name not in ("log", "pow") and fn_name not in _IR_FUNCTION_MAP:
            raise _Aborted
        if _IR_FUNCTION_MAP.get(fn_name) in _CUMULATIVE_OPS:
            # Scan: Polars cum_sum null propagation isn't the AND rule.
            raise _Aborted
        for cid in list(getattr(node, "input", [])):
            _classify_null_ir(nt, cid, schema, state)
        return

    if cls == "Agg":
        # Reductions skip nulls; MLX over NaN-filled inputs would not.
        raise _Aborted

    if cls == "Ternary":
        state["where"] = True
        # The fused Where reproduces Polars' "null cond -> else" only when the
        # cond is a comparison whose NaN-collapse is `false` (Eq/Lt/Le/Gt/Ge).
        # A top-level `Ne` (NaN != k -> true), bare bool column, or logical cond
        # would select the wrong branch on a null row — refuse those.
        if not _is_nan_safe_predicate(nt, node.predicate):
            raise _Aborted
        _classify_null_ir(nt, node.truthy, schema, state)
        _classify_null_ir(nt, node.falsy, schema, state)
        return

    raise _Aborted


# Comparison ops whose result is `false` when any operand is NaN — so an MLX
# value graph collapses a null `when` cond to the else branch, matching Polars.
# `Ne` is excluded: `NaN != k` is `true`, which would select the wrong branch.
_NAN_FALSE_COMPARISONS: frozenset[str] = frozenset({"Eq", "Lt", "Le", "Gt", "Ge"})


def _is_nan_safe_predicate(nt: Any, pred_id: int) -> bool:
    """True iff the Where predicate's top op is a NaN-collapses-to-false
    comparison, so a null operand drives the cond false (== Polars null cond)."""
    try:
        node = nt.view_expression(pred_id)
    except Exception:
        return False
    if type(node).__name__ != "BinaryExpr":
        return False
    op = getattr(node, "op", None)
    op_name = getattr(op, "name", None) or str(op).rsplit(".", 1)[-1]
    return _BINOP_MAP.get(op_name) in _NAN_FALSE_COMPARISONS


def analyze_ir_validity(
    nt: Any, node_id: int, schema: dict[str, Any], structural_only: bool = False
) -> tuple[PyFusionScope, list[tuple[str, str | float]]] | None:
    """Build a PyFusionScope whose single output is the row null mask (F32:
    1.0 = valid, 0.0 = null) for a fused HStack expression whose null
    propagation is data-dependent (contains a ``Where``) OR structural
    (contains a ``Shift``, whose leading ``w`` rows are null even on a
    null-free input).

    The validity transform ``V(node)`` (1.0 valid / 0.0 null):
      - ``V(col)``            = the column's is-valid (a per-row F32 input)
      - ``V(lit)``            = 1.0 (the shared constant)
      - ``V(f(args...))``     = AND of the args' validity (= product; a unary
                                op is the identity on validity)
      - ``V(shift(c, w))``    = ``Shift(V(c), w)`` — the front ``w`` validity
                                positions become 0.0 (Shift zero-fills),
                                reproducing Polars' structural head-nulls for
                                ANY input. This holds regardless of
                                ``structural_only``.
      - ``V(a <op> b)``       = ``V(a) * V(b)``  (arithmetic / comparison)
      - ``V(when c then t else e)`` = ``V(c) * where(value(c), V(t), V(e))``

    ``value(c)`` reuses the value-graph builder (`_visit_ir_ops`) so branch
    selection is computed with the SAME ops as the output graph — the null
    mask agrees with which branch the value dispatch actually took.

    ``structural_only`` (default False): set True when the caller has
    confirmed every input column is null-free, so the ONLY nulls are
    structural ones introduced by ``Shift``. In that regime cumulative scans
    (``cum_sum`` etc.) are validity-preserving (a null-free scan input yields
    no nulls), so they pass as validity-identity instead of aborting. When
    False (general input-null case) the conservative behavior is kept: scans
    abort, because Polars' cum_sum null propagation isn't the AND rule.

    Returns ``(scope, descriptors)`` or ``None`` if the expression is not
    validity-computable. Descriptor kinds: ``("valid", col)`` (pass the
    column's is-valid as F32), ``("col", col)`` / ``("lit", v)`` (column
    values / a scalar, for the ``value(c)`` sub-graphs). Inputs are added in
    two passes BEFORE any op, per the PyFusionScope synthesis invariant.
    """
    try:
        scope = PyFusionScope()
        descriptors: list[tuple[str, str | float]] = []
        valid_idx: dict[str, int] = {}
        val_leaf_idx: dict[int, int] = {}
        col_dedup: dict[str, int] = {}
        lit_dedup: dict[float, int] = {}
        # Shared constant 1.0 — used for V(lit) and as the validity-AND identity.
        one_idx = scope.add_input("__lit_1.0", "F32")
        descriptors.append(("lit", 1.0))
        lit_dedup[1.0] = one_idx
        # Pass 1a: a validity input per column leaf (output null if any leaf null).
        _gather_valid_leaves(nt, node_id, schema, scope, descriptors, valid_idx, structural_only)
        # Pass 1b: value inputs for the columns/literals inside Where predicates.
        _gather_cond_value_leaves(
            nt,
            node_id,
            schema,
            scope,
            descriptors,
            val_leaf_idx,
            col_dedup,
            lit_dedup,
            structural_only,
        )
        # Pass 2: build the validity op graph.
        out_idx = _visit_validity(
            nt, node_id, schema, scope, valid_idx, val_leaf_idx, one_idx, structural_only
        )
        scope.mark_output(out_idx)
        return scope, descriptors
    except _Aborted:
        return None


def _gather_valid_leaves(
    nt: Any,
    node_id: int,
    schema: dict[str, Any],
    scope: PyFusionScope,
    descriptors: list[tuple[str, str | float]],
    valid_idx: dict[str, int],
    structural_only: bool = False,
) -> None:
    """Pass 1a: add one ``("valid", col)`` input per distinct column leaf."""
    try:
        node = nt.view_expression(node_id)
    except Exception as e:
        raise _Aborted from e
    cls = type(node).__name__

    if cls == "Column":
        name = getattr(node, "name", None)
        if name is None:
            raise _Aborted
        name_s = str(name)
        if name_s not in valid_idx:
            idx = scope.add_input(f"__valid_{name_s}", "F32")
            descriptors.append(("valid", name_s))
            valid_idx[name_s] = idx
        return
    if cls == "Literal":
        return
    if cls == "BinaryExpr":
        _gather_valid_leaves(nt, node.left, schema, scope, descriptors, valid_idx, structural_only)
        _gather_valid_leaves(nt, node.right, schema, scope, descriptors, valid_idx, structural_only)
        return
    if cls == "Cast":
        _gather_valid_leaves(nt, node.expr, schema, scope, descriptors, valid_idx, structural_only)
        return
    if cls == "Function":
        fd = getattr(node, "function_data", ())
        fn_name = str(fd[0]).lower() if fd else ""
        if fn_name == _SHIFT_FN:
            # Recurse only the value child (the offset literal has no leaf).
            fn_inputs = list(getattr(node, "input", []))
            _gather_valid_leaves(
                nt, fn_inputs[0], schema, scope, descriptors, valid_idx, structural_only
            )
            return
        if fn_name == _INT_RANGE_FN:
            return  # iota generator — no leaves, never null
        for cid in list(getattr(node, "input", [])):
            _gather_valid_leaves(nt, cid, schema, scope, descriptors, valid_idx, structural_only)
        return
    if cls == "Ternary":
        _gather_valid_leaves(
            nt, node.predicate, schema, scope, descriptors, valid_idx, structural_only
        )
        _gather_valid_leaves(
            nt, node.truthy, schema, scope, descriptors, valid_idx, structural_only
        )
        _gather_valid_leaves(nt, node.falsy, schema, scope, descriptors, valid_idx, structural_only)
        return
    raise _Aborted


def _gather_cond_value_leaves(
    nt: Any,
    node_id: int,
    schema: dict[str, Any],
    scope: PyFusionScope,
    descriptors: list[tuple[str, str | float]],
    val_leaf_idx: dict[int, int],
    col_dedup: dict[str, int],
    lit_dedup: dict[float, int],
    structural_only: bool = False,
) -> None:
    """Pass 1b: add value inputs for leaves reachable inside a Where predicate
    (needed to recompute ``value(c)`` for branch selection). Non-predicate
    leaves contribute no value input."""
    try:
        node = nt.view_expression(node_id)
    except Exception as e:
        raise _Aborted from e
    cls = type(node).__name__

    if cls in ("Column", "Literal"):
        return
    if cls == "BinaryExpr":
        _gather_cond_value_leaves(
            nt,
            node.left,
            schema,
            scope,
            descriptors,
            val_leaf_idx,
            col_dedup,
            lit_dedup,
            structural_only,
        )
        _gather_cond_value_leaves(
            nt,
            node.right,
            schema,
            scope,
            descriptors,
            val_leaf_idx,
            col_dedup,
            lit_dedup,
            structural_only,
        )
        return
    if cls == "Cast":
        _gather_cond_value_leaves(
            nt,
            node.expr,
            schema,
            scope,
            descriptors,
            val_leaf_idx,
            col_dedup,
            lit_dedup,
            structural_only,
        )
        return
    if cls == "Function":
        fd = getattr(node, "function_data", ())
        fn_name = str(fd[0]).lower() if fd else ""
        if fn_name == _SHIFT_FN:
            # Recurse only the value child; the offset literal contributes no
            # cond value leaf.
            fn_inputs = list(getattr(node, "input", []))
            _gather_cond_value_leaves(
                nt,
                fn_inputs[0],
                schema,
                scope,
                descriptors,
                val_leaf_idx,
                col_dedup,
                lit_dedup,
                structural_only,
            )
            return
        if fn_name == _INT_RANGE_FN:
            return  # iota generator — no leaves
        for cid in list(getattr(node, "input", [])):
            _gather_cond_value_leaves(
                nt,
                cid,
                schema,
                scope,
                descriptors,
                val_leaf_idx,
                col_dedup,
                lit_dedup,
                structural_only,
            )
        return
    if cls == "Ternary":
        # The predicate's values drive branch selection — gather its leaves.
        _gather_leaves_ir(
            nt, node.predicate, schema, scope, descriptors, val_leaf_idx, col_dedup, lit_dedup
        )
        _gather_cond_value_leaves(
            nt,
            node.truthy,
            schema,
            scope,
            descriptors,
            val_leaf_idx,
            col_dedup,
            lit_dedup,
            structural_only,
        )
        _gather_cond_value_leaves(
            nt,
            node.falsy,
            schema,
            scope,
            descriptors,
            val_leaf_idx,
            col_dedup,
            lit_dedup,
            structural_only,
        )
        return
    raise _Aborted


def _visit_validity(
    nt: Any,
    node_id: int,
    schema: dict[str, Any],
    scope: PyFusionScope,
    valid_idx: dict[str, int],
    val_leaf_idx: dict[int, int],
    one_idx: int,
    structural_only: bool = False,
) -> int:
    """Pass 2: push the validity (null-mask) op graph; return its NodeIdx."""
    try:
        node = nt.view_expression(node_id)
    except Exception as e:
        raise _Aborted from e
    cls = type(node).__name__

    def recurse(cid: int) -> int:
        return _visit_validity(
            nt, cid, schema, scope, valid_idx, val_leaf_idx, one_idx, structural_only
        )

    if cls == "Column":
        return valid_idx[str(node.name)]
    if cls == "Literal":
        return one_idx
    if cls == "BinaryExpr":
        op = getattr(node, "op", None)
        op_name = getattr(op, "name", None) or str(op).rsplit(".", 1)[-1]
        op_id = _BINOP_MAP.get(op_name)
        if op_id is None or op_id in ("LogicalAnd", "LogicalOr"):
            raise _Aborted
        left = recurse(node.left)
        right = recurse(node.right)
        return _validity_and(scope, [left, right], one_idx)
    if cls == "Cast":
        if getattr(node, "dtype", None) != pl.Float32:
            raise _Aborted
        return recurse(node.expr)
    if cls == "Function":
        fd = getattr(node, "function_data", ())
        if not fd:
            raise _Aborted
        fn_name = str(fd[0]).lower()
        fn_inputs = list(getattr(node, "input", []))
        if fn_name == _SHIFT_FN:
            # V(shift(c, w)) = Shift(V(c), w): the front `w` validity slots
            # become 0.0 (Shift zero-fills), so the output is structurally
            # null there. Faithful for ANY input — applies even when
            # structural_only is False. The offset literal is the scalar param.
            w = _shift_offset(nt, fn_inputs)
            child_v = recurse(fn_inputs[0])
            return scope.push_op("Shift", [child_v], w)
        if fn_name == _INT_RANGE_FN:
            # iota — never contributes a null (always-valid identity).
            return one_idx
        if _is_reverse_cumulative(fn_name, fd):
            raise _Aborted
        if _IR_FUNCTION_MAP.get(fn_name) in _CUMULATIVE_OPS:
            # Cumulative scans don't follow the elementwise AND rule under
            # arbitrary input nulls, so abort in the general case. But when
            # structural_only (inputs confirmed null-free), a scan introduces
            # no nulls and is validity-preserving — treat as validity-identity.
            if not structural_only:
                raise _Aborted
            child_vs = [recurse(cid) for cid in fn_inputs]
            return _validity_and(scope, child_vs, one_idx)
        if fn_name not in ("log", "pow") and fn_name not in _IR_FUNCTION_MAP:
            raise _Aborted
        child_vs = [recurse(cid) for cid in fn_inputs]
        return _validity_and(scope, child_vs, one_idx)
    if cls == "Agg":
        raise _Aborted
    if cls == "Ternary":
        # Polars treats a null `when` condition as FALSE (the row takes the
        # else / next branch), so the result is null iff the *selected* branch
        # is null: V = where(value(cond), V(then), V(else)). No cond-validity
        # factor — `null_mode_ir` only admits NaN-safe comparison conds
        # (Eq/Lt/Le/Gt/Ge), where MLX's `NaN <op> k -> false` collapses a null
        # cond to the else branch exactly like Polars.
        cond_val = _visit_ir_ops(nt, node.predicate, schema, scope, val_leaf_idx)
        then_v = recurse(node.truthy)
        else_v = recurse(node.falsy)
        return scope.push_op("Where", [cond_val, then_v, else_v])
    raise _Aborted


def _validity_and(scope: PyFusionScope, idxs: list[int], one_idx: int) -> int:
    """AND together validity NodeIdxs via ``Mul`` (operands are 0.0/1.0). The
    identity input (``one_idx``) is dropped; an empty product is the identity."""
    operands = [i for i in idxs if i != one_idx]
    if not operands:
        return one_idx
    acc = operands[0]
    for nxt in operands[1:]:
        acc = scope.push_op("Mul", [acc, nxt])
    return acc


def _gather_leaves_ir(
    nt: Any,
    node_id: int,
    schema: dict[str, Any],
    scope: PyFusionScope,
    descriptors: list[tuple[str, str | float]],
    leaf_idx: dict[int, int],
    col_dedup: dict[str, int],
    lit_dedup: dict[float, int],
    lit_dtype_str: str = "F32",
) -> None:
    """Pass 1: DFS-walk the IR tree and `add_input` every Column/Literal,
    recording arena-id -> input-NodeIdx in `leaf_idx`.

    Dedups by value: repeat references to the same column name or the same
    literal value share a single scope input slot. This is critical for
    perf — a literal like `d2r` appearing 3x in haversine would otherwise
    stage three independent F32 broadcast buffers (3 * n_rows * 4 bytes)
    through the FFI for each call. `leaf_idx[node_id]` still maps every
    arena id to its (possibly shared) input idx so pass 2 can resolve.

    `lit_dtype_str` is the width at which literal leaves are staged. It is
    ``"F32"`` for the legacy float path (and the reduction / validity callers,
    which keep their existing behavior). For a B1 monomorphic-int chain the
    HStack caller passes the column dtype (e.g. ``"I64"``) so the integer
    literal in ``col + 1`` is staged at the column width — otherwise MLX would
    promote ``int_col + f32_literal`` to f32, producing the wrong output dtype.

    Aborts on the same unsupported-node set as `_visit_ir_ops`."""
    try:
        node = nt.view_expression(node_id)
    except Exception as e:
        raise _Aborted from e
    cls = type(node).__name__

    if cls == "Column":
        name = getattr(node, "name", None)
        if name is None:
            raise _Aborted
        dtype = schema.get(str(name))
        if dtype is None:
            raise _Aborted
        name_s = str(name)
        existing = col_dedup.get(name_s)
        if existing is None:
            idx = scope.add_input(name_s, _dtype_to_input_str(dtype))
            descriptors.append(("col", name_s))
            col_dedup[name_s] = idx
        else:
            idx = existing
        leaf_idx[node_id] = idx
        return

    if cls == "Literal":
        val = getattr(node, "value", None)
        if val is None:
            raise _Aborted
        val_f = float(val)
        existing = lit_dedup.get(val_f)
        if existing is None:
            # Stage the literal at `lit_dtype_str` (the column width for a
            # monomorphic-int chain, else F32). Keeps `int_col + lit` integer
            # rather than letting MLX promote it to f32. The descriptor carries
            # the value; the dispatch stages it at the binding's output dtype.
            idx = scope.add_input(f"__lit_{val_f}", lit_dtype_str)
            descriptors.append(("lit", val_f))
            lit_dedup[val_f] = idx
        else:
            idx = existing
        leaf_idx[node_id] = idx
        return

    if cls == "BinaryExpr":
        _gather_leaves_ir(
            nt, node.left, schema, scope, descriptors, leaf_idx, col_dedup, lit_dedup, lit_dtype_str
        )
        _gather_leaves_ir(
            nt,
            node.right,
            schema,
            scope,
            descriptors,
            leaf_idx,
            col_dedup,
            lit_dedup,
            lit_dtype_str,
        )
        return

    if cls == "Cast":
        # Mirror the pass-2 restriction: only CastF32 is honored downstream
        # (see `_visit_ir_ops` Cast branch). Abort here so we don't add
        # leaves for a tree that pass 2 will reject.
        if getattr(node, "dtype", None) != pl.Float32:
            raise _Aborted
        _gather_leaves_ir(
            nt, node.expr, schema, scope, descriptors, leaf_idx, col_dedup, lit_dedup, lit_dtype_str
        )
        return

    if cls == "Function":
        fd = getattr(node, "function_data", ())
        if not fd:
            raise _Aborted
        fn_name = str(fd[0]).lower()
        fn_inputs = list(getattr(node, "input", []))
        if fn_name == _SHIFT_FN:
            # Recurse leaves of ONLY the value child; the offset Literal is a
            # compile-time param (validated in pass 2), not an input leaf.
            _shift_offset(nt, fn_inputs)  # validate now so pass 1/2 agree
            _gather_leaves_ir(
                nt,
                fn_inputs[0],
                schema,
                scope,
                descriptors,
                leaf_idx,
                col_dedup,
                lit_dedup,
                lit_dtype_str,
            )
            return
        if fn_name == _INT_RANGE_FN:
            # Terminal generator (RowIndex, 0 inputs): adds NO leaves. (Note:
            # currently unreachable — int_range is opaque to view_expression.)
            return
        if fn_name != "log" and fn_name != "pow" and fn_name not in _IR_FUNCTION_MAP:
            raise _Aborted
        if _is_reverse_cumulative(fn_name, fd):
            raise _Aborted
        for cid in fn_inputs:
            _gather_leaves_ir(
                nt, cid, schema, scope, descriptors, leaf_idx, col_dedup, lit_dedup, lit_dtype_str
            )
        return

    if cls == "Agg":
        agg_name = getattr(node, "name", None)
        if _AGG_MAP.get(str(agg_name)) is None:
            raise _Aborted
        args = list(getattr(node, "arguments", []))
        if not args:
            raise _Aborted
        _gather_leaves_ir(
            nt, args[0], schema, scope, descriptors, leaf_idx, col_dedup, lit_dedup, lit_dtype_str
        )
        return

    if cls == "Ternary":
        _gather_leaves_ir(
            nt,
            node.predicate,
            schema,
            scope,
            descriptors,
            leaf_idx,
            col_dedup,
            lit_dedup,
            lit_dtype_str,
        )
        _gather_leaves_ir(
            nt,
            node.truthy,
            schema,
            scope,
            descriptors,
            leaf_idx,
            col_dedup,
            lit_dedup,
            lit_dtype_str,
        )
        _gather_leaves_ir(
            nt,
            node.falsy,
            schema,
            scope,
            descriptors,
            leaf_idx,
            col_dedup,
            lit_dedup,
            lit_dtype_str,
        )
        return

    raise _Aborted


def _visit_ir_ops(
    nt: Any,
    node_id: int,
    schema: dict[str, Any],
    scope: PyFusionScope,
    leaf_idx: dict[int, int],
) -> int:
    """Pass 2: DFS-walk the IR and push ops. Leaves return their precomputed
    input NodeIdx from `leaf_idx`."""
    try:
        node = nt.view_expression(node_id)
    except Exception as e:
        raise _Aborted from e
    cls = type(node).__name__

    if cls in ("Column", "Literal"):
        return leaf_idx[node_id]

    if cls == "BinaryExpr":
        op = getattr(node, "op", None)
        op_name = getattr(op, "name", None) or str(op).rsplit(".", 1)[-1]
        op_id = _BINOP_MAP.get(op_name)
        if op_id is None:
            raise _Aborted
        left = _visit_ir_ops(nt, node.left, schema, scope, leaf_idx)
        right = _visit_ir_ops(nt, node.right, schema, scope, leaf_idx)
        return scope.push_op(op_id, [left, right])

    if cls == "Cast":
        target = getattr(node, "dtype", None)
        # MLX 0.22.0 has no F64 (Apple Silicon ignores it at runtime). Our
        # executor only round-trips F32 buffers — accept CastF32 only and
        # reject everything else. A Cast we can't honor poisons the whole
        # expression, which is correct: we cannot faithfully emulate F64
        # or Bool semantics through an F32-only kernel chain.
        if target != pl.Float32:
            raise _Aborted
        child = _visit_ir_ops(nt, node.expr, schema, scope, leaf_idx)
        return scope.push_op("CastF32", [child])

    if cls == "Function":
        fd = getattr(node, "function_data", ())
        if not fd:
            raise _Aborted
        fn_name = str(fd[0]).lower()
        fn_inputs = list(getattr(node, "input", []))
        if _is_reverse_cumulative(fn_name, fd):
            raise _Aborted
        if fn_name == _SHIFT_FN:
            # Forward shift with a non-negative integer offset param. Recurse
            # ONLY the value child; the offset Literal is the scalar param.
            w = _shift_offset(nt, fn_inputs)
            child = _visit_ir_ops(nt, fn_inputs[0], schema, scope, leaf_idx)
            return scope.push_op("Shift", [child], w)
        if fn_name == _INT_RANGE_FN:
            # 0-input iota generator (currently unreachable; opaque IR).
            return scope.push_op("RowIndex", [])
        if fn_name == "log":
            return _visit_ir_log_ops(nt, fn_inputs, schema, scope, leaf_idx)
        if fn_name == "pow":
            if len(fn_inputs) != 2:
                raise _Aborted
            left = _visit_ir_ops(nt, fn_inputs[0], schema, scope, leaf_idx)
            right = _visit_ir_ops(nt, fn_inputs[1], schema, scope, leaf_idx)
            return scope.push_op("Pow", [left, right])
        op_id = _IR_FUNCTION_MAP.get(fn_name)
        if op_id is None:
            raise _Aborted
        child_idxs = [_visit_ir_ops(nt, cid, schema, scope, leaf_idx) for cid in fn_inputs]
        return scope.push_op(op_id, child_idxs)

    if cls == "Agg":
        agg_name = getattr(node, "name", None)
        op_id = _AGG_MAP.get(str(agg_name))
        if op_id is None:
            raise _Aborted
        args = list(getattr(node, "arguments", []))
        if not args:
            raise _Aborted
        child = _visit_ir_ops(nt, args[0], schema, scope, leaf_idx)
        return scope.push_op(op_id, [child])

    if cls == "Ternary":
        cond = _visit_ir_ops(nt, node.predicate, schema, scope, leaf_idx)
        then_v = _visit_ir_ops(nt, node.truthy, schema, scope, leaf_idx)
        else_v = _visit_ir_ops(nt, node.falsy, schema, scope, leaf_idx)
        return scope.push_op("Where", [cond, then_v, else_v])

    raise _Aborted


def _visit_ir_log_ops(
    nt: Any,
    fn_inputs: list,
    schema: dict[str, Any],
    scope: PyFusionScope,
    leaf_idx: dict[int, int],
) -> int:
    """log() is 2-arg (x, base) in the IR too; map base to Log/Log2/Log10."""
    if len(fn_inputs) != 2:
        raise _Aborted
    base_node = nt.view_expression(fn_inputs[1])
    if type(base_node).__name__ != "Literal":
        raise _Aborted
    base_val = getattr(base_node, "value", None)
    if base_val is None:
        raise _Aborted
    if abs(base_val - 2.718281828459045) < 1e-9:
        op_id = "Log"
    elif base_val == 2 or base_val == 2.0:
        op_id = "Log2"
    elif base_val == 10 or base_val == 10.0:
        op_id = "Log10"
    else:
        raise _Aborted
    child = _visit_ir_ops(nt, fn_inputs[0], schema, scope, leaf_idx)
    return scope.push_op(op_id, [child])
