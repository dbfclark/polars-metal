"""Bottom-up walk of the Polars IR via NodeTraverser.

For each node, the walker returns either:
- ``Handled(plan_dict)`` — the subtree can be handled by the Metal engine;
  ``plan_dict`` is a serialized MetalPlanNode tree (see
  ``crates/polars-metal-core/src/plan/mod.rs``).
- ``FallBack(reason)`` — at least one descendant or the node itself is not
  supported. Any FallBack poisons all ancestors; in M1 we never lift partial
  subtrees to GPU.

API discovery
-------------
The Polars Python IR API (verified against pinned ``py-1.40.1`` and the
generated stub at ``polars._plr.NodeTraverser``):

- ``nt.view_current_node()`` returns the IR node Python object — class names
  like ``DataFrameScan``, ``SimpleProjection``, ``Select``, ``Filter`` etc.
  The classes live in the unnamed ``builtins`` module so we identify them by
  ``type(node).__name__``.
- ``nt.get_inputs()`` returns the input node IDs.
- ``nt.get_node()`` / ``nt.set_node(int)`` move the visitor's "current" node;
  ``set_udf(fn)`` replaces *that* current node with a PythonScan wrapping fn.
- ``nt.get_schema()`` returns the schema of the **current** node, in output
  order. For ``SimpleProjection`` this is the project's column order; for
  ``DataFrameScan`` with a non-None ``.projection`` it is the projected order.

IR-node shapes for the nodes we touch in Phase 4 (from
``polars-python/src/lazyframe/visitor/nodes.rs`` at py-1.40.1):

- ``DataFrameScan``: ``.df`` (PyDataFrame), ``.projection`` (list[str] | None),
  ``.selection`` (Option[PyExprIR]).
- ``SimpleProjection``: ``.input`` (int). Output columns come from
  ``nt.get_schema()`` while the node is current.
- ``Select``: ``.input`` (int), ``.expr`` (list of PyExprIR), ``.should_broadcast``.
- ``Filter``: ``.input`` (int), ``.predicate`` (PyExprIR). M1 Phase 5
  (Task 15) accepts predicates that resolve to a single ``Column`` whose
  dtype is Boolean. Phase 6 (Task 18) additionally accepts a
  ``BinaryExpr`` whose op is one of ``Operator.{Eq,NotEq,Lt,LtEq,Gt,GtEq}``
  and whose operands are each either a ``Column(I64|F64)`` or a
  matching-dtype ``Literal``. Compound predicates (AND/OR) and arithmetic
  inside the predicate land in later tasks.

PyExprIR shape (from ``polars-python/src/lazyframe/visit.rs``): ``.node``
(usize into the expression arena) and ``.output_name`` (str). To inspect
the underlying expression class we call ``nt.view_expression(node_id)``;
to resolve its dtype against the current IR root we call
``nt.get_dtype(node_id)``.

Expression-node shapes used in Phase 6 (from
``polars-python/src/lazyframe/visitor/expr_nodes.rs``):

- ``Column``: ``.name`` (str).
- ``Literal``: ``.value`` (Python int/float/bool) and ``.dtype`` (Polars dtype).
- ``BinaryExpr``: ``.left`` (int node id), ``.op`` (``builtins.Operator``
  enum — comparable via ``str(op) == "Operator.Gt"``), ``.right`` (int node id).

Note: under default optimization, ``df.lazy().select([...])`` is collapsed
into ``DataFrameScan.projection`` and no separate Projection node appears.
SimpleProjection/Select still arise from pipelines that the optimizer can't
fold (and when optimization is disabled). We support all three shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Handled:
    """Wraps a serialized MetalPlanNode dict produced by the walker."""

    plan: dict


@dataclass(frozen=True)
class FallBack:
    """Indicates the current subtree cannot be lifted to the Metal engine."""

    reason: str


WalkResult = Handled | FallBack


def walk(nt: Any) -> WalkResult:
    """Walk the IR rooted at ``nt``'s current node.

    Leaves ``nt`` with the same current-node value it had on entry, so the
    caller can hand the traverser to ``nt.set_udf(...)`` after a Handled
    result.
    """
    saved_root = nt.get_node()
    try:
        return _walk_at_current(nt)
    finally:
        # Restore on every path — exceptions, FallBacks, or Handled.
        nt.set_node(saved_root)


def _walk_at_current(nt: Any) -> WalkResult:
    node = nt.view_current_node()
    cls = type(node).__name__

    if cls == "DataFrameScan":
        return _walk_dataframe_scan(nt, node)
    if cls == "SimpleProjection":
        return _walk_simple_projection(nt, node)
    if cls == "Select":
        return _walk_select(nt, node)
    if cls == "Filter":
        return _walk_filter(nt, node)
    return FallBack(reason=f"unsupported IR node: {cls}")


def _walk_dataframe_scan(nt: Any, node: Any) -> WalkResult:
    """Lower a DataFrameScan into a Metal Scan plan node.

    The schema we report is the post-projection schema of the scan (what
    Polars actually emits), captured from ``nt.get_schema()``. We also hold
    on to the raw ``node.df`` and the projection list so the UDF can
    reconstruct the resulting DataFrame.
    """
    # `selection` is an optional predicate the optimizer pushed into the scan.
    # We don't handle predicates yet — fall back if one is present.
    if getattr(node, "selection", None) is not None:
        return FallBack(reason="DataFrameScan with pushed-down predicate")

    schema = dict(nt.get_schema())
    columns: list[tuple[str, str]] = []
    for name, dtype in schema.items():
        mapped = _map_dtype(dtype)
        if mapped is None:
            return FallBack(reason=f"unsupported dtype {dtype!s} on column {name!r}")
        columns.append((str(name), mapped))

    projection = getattr(node, "projection", None)
    projection_list: list[str] | None
    if projection is None:
        projection_list = None
    else:
        try:
            projection_list = [str(c) for c in projection]
        except Exception:
            return FallBack(reason="DataFrameScan.projection not iterable")

    return Handled(
        plan={
            "kind": "Scan",
            "columns": columns,
            # The raw underlying PyDataFrame — used by the UDF to reconstruct
            # the result. Captured here as a Python object reference; not
            # serialized to Rust in Task 7.
            "df": node.df,
            "projection": projection_list,
        }
    )


def _walk_simple_projection(nt: Any, node: Any) -> WalkResult:
    """Lower a SimpleProjection: column re-selection by name.

    Polars stores only ``.input`` on the node; the projected column names
    (in order) come from ``nt.get_schema()`` while the projection is the
    current node.
    """
    out_schema = dict(nt.get_schema())
    columns: list[str] = []
    for name, dtype in out_schema.items():
        if _map_dtype(dtype) is None:
            return FallBack(reason=f"unsupported dtype {dtype!s} on column {name!r}")
        columns.append(str(name))

    inputs = nt.get_inputs()
    if len(inputs) != 1:
        return FallBack(reason=f"SimpleProjection expected 1 input, got {len(inputs)}")

    nt.set_node(inputs[0])
    inner = _walk_at_current(nt)
    # _walk_at_current does NOT restore; we leave the restore to the outer
    # `walk()`. But we also need the child's view-current-node calls below
    # to see *its* node, so we don't restore here.

    if isinstance(inner, FallBack):
        return inner

    return Handled(plan={"kind": "Project", "input": inner.plan, "columns": columns})


def _walk_select(nt: Any, node: Any) -> WalkResult:
    """Lower a Select node iff every expression is a plain ``Column(name)``.

    Anything else — arithmetic, aliasing, casts, literals — falls back to
    CPU in Phase 4.
    """
    exprs = getattr(node, "expr", None)
    if exprs is None:
        return FallBack(reason="Select node missing .expr")

    columns: list[str] = []
    for e in exprs:
        # PyExprIR has .node (int) and .output_name (str). To inspect the
        # underlying shape, view_expression(node_id) returns the actual
        # expression object (Column, BinaryExpr, etc.).
        try:
            inner_node = nt.view_expression(e.node)
        except Exception as ex:
            return FallBack(reason=f"could not view expression: {ex!r}")
        inner_cls = type(inner_node).__name__
        if inner_cls != "Column":
            return FallBack(reason=f"Select expression {inner_cls} not supported in Phase 4")
        # When the output_name differs from the column name, the projection
        # involves an alias — also a Phase 4 fallback.
        col_name = getattr(inner_node, "name", None)
        if col_name is None:
            return FallBack(reason="Column expression missing .name")
        if str(col_name) != str(e.output_name):
            return FallBack(reason="aliased Column projection")
        columns.append(str(col_name))

    # Validate the output schema dtypes too — guards against e.g. struct
    # columns sneaking through (we should fall back even on a plain Column
    # if its dtype isn't in our closed set).
    out_schema = dict(nt.get_schema())
    for name, dtype in out_schema.items():
        if _map_dtype(dtype) is None:
            return FallBack(reason=f"unsupported dtype {dtype!s} on column {name!r}")

    inputs = nt.get_inputs()
    if len(inputs) != 1:
        return FallBack(reason=f"Select expected 1 input, got {len(inputs)}")
    nt.set_node(inputs[0])
    inner = _walk_at_current(nt)
    if isinstance(inner, FallBack):
        return inner

    return Handled(plan={"kind": "Project", "input": inner.plan, "columns": columns})


def _walk_filter(nt: Any, node: Any) -> WalkResult:
    """Lower a Filter node iff its predicate is in the M1 closed set.

    Accepted shapes:
    - **Phase 5** — a single ``Column`` of Boolean dtype:
      ``df.filter(pl.col("mask"))``.
    - **Phase 6** — a ``BinaryExpr`` whose op is one of the six comparison
      operators and whose operands are each a ``Column(I64|F64)`` or a
      same-dtype ``Literal``: ``df.filter(pl.col("a") > 0)``,
      ``df.filter(pl.col("x") < pl.col("y"))``, etc.

    Everything else (AND/OR — Task 20; arithmetic in the predicate — M2+;
    casts, function calls, etc.) FallBacks so the CPU executor produces
    the correct result.

    The plan we emit serialises the predicate AST; the Rust UDF walks it
    at dispatch time.
    """
    pred_expr_ir = getattr(node, "predicate", None)
    if pred_expr_ir is None:
        return FallBack(reason="Filter node missing .predicate")

    pred_node_id = getattr(pred_expr_ir, "node", None)
    if pred_node_id is None:
        return FallBack(reason="Filter predicate has no .node id")

    # IMPORTANT: dtypes must be resolved against the *input* schema (the
    # Filter's child), not the post-Filter schema. The current node is
    # still the Filter itself here, and Polars' get_dtype uses
    # `self.root.schema()` which for a Filter is the input schema —
    # confirmed in polars-python's visit.rs:get_dtype implementation.
    pred_dict = _walk_predicate(nt, pred_node_id)
    if pred_dict is None:
        return FallBack(reason="Filter predicate not in M1 closed set")

    # Predicates must resolve to a Boolean column for `df.filter` to make
    # sense. For a Column leaf this means the column itself is Bool; for
    # a Compare it's implicit (every comparison produces a Bool). Reject
    # bare-Column leaves of non-Bool dtype the way Phase 5 always has.
    if pred_dict["kind"] == "Column" and pred_dict["dtype"] != "Bool":
        return FallBack(
            reason=f"Filter predicate is Column of dtype {pred_dict['dtype']}; M1 requires Boolean"
        )
    # A bare Literal at the filter root isn't meaningful as a predicate
    # (Polars would broadcast it). Fall back to keep the closed set tight.
    if pred_dict["kind"] in ("LiteralI64", "LiteralF64", "LiteralBool"):
        return FallBack(reason="Filter predicate is a bare literal")

    # Walk the input subtree. We re-validate the *output* schema dtypes
    # for the surviving columns (the post-Filter schema is the same as
    # the input schema — Filter doesn't drop columns).
    out_schema = dict(nt.get_schema())
    for name, dtype in out_schema.items():
        if _map_dtype(dtype) is None:
            return FallBack(reason=f"unsupported dtype {dtype!s} on column {name!r}")

    inputs = nt.get_inputs()
    if len(inputs) != 1:
        return FallBack(reason=f"Filter expected 1 input, got {len(inputs)}")
    nt.set_node(inputs[0])
    inner = _walk_at_current(nt)
    if isinstance(inner, FallBack):
        return inner

    return Handled(
        plan={
            "kind": "Filter",
            "input": inner.plan,
            "predicate": pred_dict,
        }
    )


_CMP_OP_NAMES: dict[str, str] = {
    # Polars' Operator enum's `str(op)` form → our MetalDtype-side op tag.
    "Operator.Eq": "Eq",
    "Operator.NotEq": "Ne",
    "Operator.Lt": "Lt",
    "Operator.LtEq": "Le",
    "Operator.Gt": "Gt",
    "Operator.GtEq": "Ge",
}


def _walk_predicate(nt: Any, node_id: int) -> dict | None:
    """Walk an expression-node id into a predicate-AST dict, or None if rejected.

    The dict shape mirrors ``crates/polars-metal-core/src/plan/mod.rs::PredicateAst``:

    - ``{"kind": "Column", "name": str, "dtype": "I64"|"F64"|"Bool"}``
    - ``{"kind": "LiteralI64", "value": int}``
    - ``{"kind": "LiteralF64", "value": float}``
    - ``{"kind": "LiteralBool", "value": bool}``
    - ``{"kind": "Compare", "op": "Eq|Ne|Lt|Le|Gt|Ge",
         "lhs": <predicate>, "rhs": <predicate>, "dtype": "I64"|"F64"}``

    The dtype tag on ``Compare`` is the *operand* dtype (the dtype that
    drives kernel selection on the Rust side); both lhs and rhs must
    resolve to the same dtype.
    """
    try:
        expr = nt.view_expression(node_id)
    except Exception:
        return None
    cls = type(expr).__name__

    if cls == "Column":
        name = getattr(expr, "name", None)
        if name is None:
            return None
        try:
            dtype = nt.get_dtype(node_id)
        except Exception:
            return None
        m1_dtype = _map_dtype(dtype)
        if m1_dtype is None:
            return None
        return {"kind": "Column", "name": str(name), "dtype": m1_dtype}

    if cls == "Literal":
        value = getattr(expr, "value", None)
        if value is None:
            # Polars' typed-null literal — we don't have a representation
            # for "null literal" in our closed AST yet; fall back.
            return None
        # `bool` is a subclass of `int`; check bool first.
        if isinstance(value, bool):
            return {"kind": "LiteralBool", "value": bool(value)}
        if isinstance(value, int):
            return {"kind": "LiteralI64", "value": int(value)}
        if isinstance(value, float):
            return {"kind": "LiteralF64", "value": float(value)}
        return None

    if cls == "BinaryExpr":
        op_tag = _CMP_OP_NAMES.get(str(expr.op))
        if op_tag is None:
            # Operator.And/Or/Plus/etc. — not in this task's closed set.
            return None
        lhs = _walk_predicate(nt, expr.left)
        rhs = _walk_predicate(nt, expr.right)
        if lhs is None or rhs is None:
            return None
        # Nested comparisons (Compare-inside-Compare) aren't a Phase-6
        # shape — those would be 3-valued bool combinators handled by
        # And/Or in Phase 7. Reject defensively.
        if lhs["kind"] == "Compare" or rhs["kind"] == "Compare":
            return None
        lhs_dt = _leaf_dtype(lhs)
        rhs_dt = _leaf_dtype(rhs)
        # The cmp kernels operate on a single numeric dtype; both sides
        # must agree. Polars inserts an explicit Cast when types differ
        # (we'd see a Cast node and fail to walk it).
        if lhs_dt is None or rhs_dt is None or lhs_dt != rhs_dt:
            return None
        # Only numeric (I64/F64) operand dtypes are wired today; Bool-Bool
        # comparison isn't a Phase-6 shape (use the bool column directly).
        if lhs_dt not in ("I64", "F64"):
            return None
        # Both leaves can't be literals — that's a constant-folded predicate
        # the optimizer should have collapsed; reject to keep the dispatcher
        # contract (at least one Column reference) clean.
        if lhs["kind"].startswith("Literal") and rhs["kind"].startswith("Literal"):
            return None
        return {
            "kind": "Compare",
            "op": op_tag,
            "lhs": lhs,
            "rhs": rhs,
            "dtype": lhs_dt,
        }

    return None


def _leaf_dtype(pred: dict) -> str | None:
    """Return the operand-side dtype tag for a leaf predicate dict.

    For ``Column`` it's the column dtype; for the literal variants we
    map the Python value class onto the matching dtype tag.
    """
    k = pred["kind"]
    if k == "Column":
        return pred["dtype"]
    if k == "LiteralI64":
        return "I64"
    if k == "LiteralF64":
        return "F64"
    if k == "LiteralBool":
        return "Bool"
    return None


def _map_dtype(dt: Any) -> str | None:
    """Map a Polars DataType to a MetalDtype tag, or None if unsupported.

    The current closed set: Int64 (i64), Float64 (f64), Boolean. Everything
    else (String, Categorical, List, Struct, smaller ints, etc.) is a
    fallback for M1.
    """
    s = str(dt)
    if s == "Int64":
        return "I64"
    if s == "Float64":
        return "F64"
    if s == "Boolean":
        return "Bool"
    return None
