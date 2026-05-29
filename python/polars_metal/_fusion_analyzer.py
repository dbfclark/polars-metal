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


def _dtype_to_input_str(dtype: Any) -> str:
    """Map a Polars dtype to the input-dtype string the PyFusionScope accepts."""
    if dtype == pl.Float32:
        return "F32"
    if dtype == pl.Float64:
        return "F64"
    if dtype == pl.Int32:
        return "I32"
    if dtype == pl.Boolean:
        return "Bool"
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
}


def analyze_ir_expression(nt: Any, node_id: int, schema: dict[str, Any]) -> PyFusionScope | None:
    """Walk a Polars IR expression by arena ID and build a PyFusionScope.

    Used by the walker after Polars has optimized the LazyFrame into arena
    form. `nt` is a NodeTraverser; `node_id` is the int arena ID of the
    expression root (typically from `binding.node` in an HStack/Select).

    Returns None on the first unsupported node.
    """
    result = analyze_ir_with_columns(nt, node_id, schema)
    return None if result is None else result[0]


def analyze_ir_with_columns(
    nt: Any, node_id: int, schema: dict[str, Any]
) -> tuple[PyFusionScope, list[str]] | None:
    """Like `analyze_ir_expression`, but also returns the ordered list of
    real input column names (excluding synthetic literal inputs) so the
    executor knows which Polars Series to pass to `execute_fused_expr`."""
    try:
        scope = PyFusionScope()
        columns: list[str] = []
        idx = _visit_ir(nt, node_id, schema, scope, columns)
        scope.mark_output(idx)
        return scope, columns
    except _Aborted:
        return None


def _visit_ir(
    nt: Any,
    node_id: int,
    schema: dict[str, Any],
    scope: PyFusionScope,
    columns: list[str] | None = None,
) -> int:
    """Recursive descent over the Polars IR arena.

    `columns`, when given, is appended with each real input column name (in
    scope-add order) so the caller can pass only the user-facing series to
    `execute_fused_expr`. Synthetic literal inputs (`__lit_*`) are excluded.
    """
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
        if columns is not None:
            columns.append(str(name))
        return scope.add_input(str(name), _dtype_to_input_str(dtype))

    if cls == "Literal":
        val = getattr(node, "value", None)
        if val is None:
            raise _Aborted
        return scope.add_input(f"__lit_{val}", "F32")

    if cls == "BinaryExpr":
        op = getattr(node, "op", None)
        op_name = getattr(op, "name", None) or str(op).rsplit(".", 1)[-1]
        op_id = _BINOP_MAP.get(op_name)
        if op_id is None:
            raise _Aborted
        left = _visit_ir(nt, node.left, schema, scope, columns)
        right = _visit_ir(nt, node.right, schema, scope, columns)
        return scope.push_op(op_id, [left, right])

    if cls == "Cast":
        target = getattr(node, "dtype", None)
        op_id = None
        if target == pl.Float32:
            op_id = "CastF32"
        elif target == pl.Float64:
            op_id = "CastF64"
        elif target == pl.Int32:
            op_id = "CastI32"
        elif target == pl.Boolean:
            op_id = "CastBool"
        if op_id is None:
            raise _Aborted
        child = _visit_ir(nt, node.expr, schema, scope, columns)
        return scope.push_op(op_id, [child])

    if cls == "Function":
        fd = getattr(node, "function_data", ())
        if not fd:
            raise _Aborted
        fn_name = str(fd[0]).lower()
        inputs = list(getattr(node, "input", []))
        if fn_name == "log":
            return _visit_ir_log(nt, inputs, schema, scope, columns)
        if fn_name == "pow":
            # `square` desugars to pow(x, 2); we emit Pow.
            if len(inputs) != 2:
                raise _Aborted
            left = _visit_ir(nt, inputs[0], schema, scope, columns)
            right = _visit_ir(nt, inputs[1], schema, scope, columns)
            return scope.push_op("Pow", [left, right])
        op_id = _IR_FUNCTION_MAP.get(fn_name)
        if op_id is None:
            raise _Aborted
        child_idxs = [_visit_ir(nt, cid, schema, scope, columns) for cid in inputs]
        return scope.push_op(op_id, child_idxs)

    if cls == "Agg":
        agg_name = getattr(node, "name", None)
        op_id = _AGG_MAP.get(str(agg_name))
        if op_id is None:
            raise _Aborted
        args = list(getattr(node, "arguments", []))
        if not args:
            raise _Aborted
        child = _visit_ir(nt, args[0], schema, scope, columns)
        return scope.push_op(op_id, [child])

    if cls == "Ternary":
        cond = _visit_ir(nt, node.predicate, schema, scope, columns)
        then_v = _visit_ir(nt, node.truthy, schema, scope, columns)
        else_v = _visit_ir(nt, node.falsy, schema, scope, columns)
        return scope.push_op("Where", [cond, then_v, else_v])

    raise _Aborted


def _visit_ir_log(
    nt: Any,
    inputs: list,
    schema: dict[str, Any],
    scope: PyFusionScope,
    columns: list[str] | None = None,
) -> int:
    """log() is 2-arg (x, base) in the IR too; map base to Log/Log2/Log10."""
    if len(inputs) != 2:
        raise _Aborted
    base_node = nt.view_expression(inputs[1])
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
    child = _visit_ir(nt, inputs[0], schema, scope, columns)
    return scope.push_op(op_id, [child])
