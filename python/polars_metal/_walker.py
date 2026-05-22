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
- ``GroupBy``: ``.keys`` (list[PyExprIR]), ``.aggs`` (list[PyExprIR]),
  ``.maintain_order`` (bool), ``.options`` (GroupbyOptions). The ``PyExprIR``
  objects each have ``.node`` (int arena index) and ``.output_name`` (str).
  The alias, if any, is conveyed via ``agg_expr.output_name`` — there is no
  separate ``Alias`` wrapper node for aggregations. Expression-arena nodes
  reachable from ``GroupBy.aggs`` are either:
  - ``Agg``: ``.name`` (lowercase str, e.g. ``'sum'``), ``.arguments``
    (list[int] — raw arena indices, **not** PyExprIR objects).
  - ``Len``: no attributes (empty ``dir()``).
  For ``Agg``, each element of ``.arguments`` is a plain ``int`` used as the
  arena index for the argument expression (typically a ``Column`` node).

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
    if cls == "GroupBy":
        return _walk_group_by(nt, node)
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

    # Multi-chunk Series are not yet supported. Convert the PyDataFrame to a
    # pl.DataFrame and check each column's chunk count.
    df = getattr(node, "df", None)
    if df is not None:
        try:
            import polars as pl

            polars_df = pl.DataFrame._from_pydf(df)
            for col_name in polars_df.columns:
                try:
                    n_chunks = polars_df[col_name].n_chunks()
                    if n_chunks > 1:
                        return FallBack(
                            reason=f"multi-chunk Series not yet supported (column {col_name!r} has {n_chunks} chunks)"
                        )
                except Exception:
                    # If n_chunks() raises (shouldn't happen), assume single chunk
                    pass
        except Exception:
            # If we can't convert or inspect the DataFrame, just proceed; the
            # kernel will handle it or the UDF fallback will catch it.
            pass

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
    - **Phase 7** — a ``BinaryExpr`` whose op is ``Operator.And`` or
      ``Operator.Or`` and whose operands are each themselves any of the
      shapes above (recursively): ``df.filter((pl.col("a") > 0) &
      (pl.col("b") < pl.col("c")))``, mixed AND/OR, nested combinations,
      etc.

    Everything else (NOT via ``.not_()``, arithmetic in the predicate,
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


_AGG_NAME_TO_OP: dict[str, str] = {
    # Polars Agg node's .name (lowercase) → MetalAggOp tag used in the plan dict.
    "sum": "Sum",
    "mean": "Mean",
    "min": "Min",
    "max": "Max",
    "count": "Count",
}


def _walk_group_by(nt: Any, node: Any) -> WalkResult:
    """Lower a Polars GroupBy IR node iff every key is a bare Column of an
    accepted dtype and every aggregation is in the M2 closed set.

    Verified IR shape (py-1.40.1):
    - ``node.keys``: list[PyExprIR] — each has ``.node`` (int arena index)
      and ``.output_name`` (str). The arena expression is a ``Column`` node.
    - ``node.aggs``: list[PyExprIR] — each has ``.node`` and ``.output_name``
      (carries the alias; there is no separate Alias wrapper). The arena
      expression is either ``Agg`` or ``Len``.
    - ``Agg``: ``.name`` (lowercase str), ``.arguments`` (list[int] — raw
      arena indices, not PyExprIR objects).
    - ``Len``: no attributes (empty dir).

    For key dtype resolution, ``nt.get_dtype(key_node_id)`` works from the
    GroupBy level because key columns are present in the output schema.
    For agg argument columns (e.g. the ``v`` in ``pl.col("v").sum()``),
    ``nt.get_dtype`` raises ``ColumnNotFoundError`` because those columns may
    not appear in the GroupBy output; we look them up in the input schema
    fetched by navigating to the child node.
    """
    keys_expr = getattr(node, "keys", None)
    aggs_expr = getattr(node, "aggs", None)
    if keys_expr is None or aggs_expr is None:
        return FallBack(reason="GroupBy node missing .keys or .aggs")

    inputs = nt.get_inputs()
    if len(inputs) != 1:
        return FallBack(reason=f"GroupBy expected 1 input, got {len(inputs)}")

    # Fetch the input schema once: needed for agg-argument dtype validation.
    parent_id = nt.get_node()
    nt.set_node(inputs[0])
    try:
        in_schema = dict(nt.get_schema())
    finally:
        nt.set_node(parent_id)

    keys: list[list[str]] = []
    for key_expr in keys_expr:
        key_node_id = getattr(key_expr, "node", None)
        if key_node_id is None:
            return FallBack(reason="GroupBy key expression has no .node id")
        try:
            key_inner = nt.view_expression(key_node_id)
        except Exception as ex:
            return FallBack(reason=f"could not view key expression: {ex!r}")
        key_cls = type(key_inner).__name__
        if key_cls != "Column":
            return FallBack(reason=f"GroupBy key expression {key_cls} not supported")
        key_name = getattr(key_inner, "name", None)
        if key_name is None:
            return FallBack(reason="GroupBy key Column missing .name")
        dtype = in_schema.get(str(key_name))
        if dtype is None:
            return FallBack(reason=f"GroupBy key {key_name!r} not in input schema")
        mapped = _map_dtype(dtype)
        if mapped is None:
            return FallBack(reason=f"unsupported dtype {dtype!s} on key {key_name!r}")
        keys.append([str(key_name), mapped])

    aggs: list[dict[str, str]] = []
    for agg_expr in aggs_expr:
        agg_dict = _walk_agg_expression(nt, agg_expr, in_schema)
        if agg_dict is None:
            return FallBack(reason="GroupBy agg expression not in M2 closed set")
        aggs.append(agg_dict)

    nt.set_node(inputs[0])
    inner = _walk_at_current(nt)
    if isinstance(inner, FallBack):
        return inner

    return Handled(
        plan={
            "kind": "GroupBy",
            "input": inner.plan,
            "keys": keys,
            "aggs": aggs,
        }
    )


def _walk_agg_expression(
    nt: Any, agg_expr: Any, in_schema: dict[str, Any]
) -> dict[str, str] | None:
    """Lower one aggregation PyExprIR to ``{input_col, op, output_alias}``.

    The alias is read from ``agg_expr.output_name`` (Polars carries it on the
    PyExprIR wrapper, not as a separate Alias expression node).

    ``in_schema`` is the GroupBy node's input schema (pre-aggregation), used to
    validate that agg argument columns are present and have supported dtypes.
    We cannot use ``nt.get_dtype()`` for agg-argument columns from the GroupBy
    level — they may not appear in the GroupBy output schema, causing
    ``ColumnNotFoundError``.

    Accepted shapes:
    - ``Agg(name=<op>, arguments=[col_arena_id])`` where the argument resolves
      to a bare ``Column`` of a supported dtype, and ``name`` is one of the
      entries in ``_AGG_NAME_TO_OP``.
    - ``Len`` (no arguments) — maps to op ``"Len"`` with ``input_col=""``.

    Everything else (e.g. ``Agg`` with a ``BinaryExpr`` argument, unknown agg
    names, multi-arg aggs) returns ``None`` and causes the GroupBy to fall back.
    """
    node_id = getattr(agg_expr, "node", None)
    if node_id is None:
        return None
    output_alias = str(getattr(agg_expr, "output_name", "") or "")

    try:
        inner = nt.view_expression(node_id)
    except Exception:
        return None

    inner_cls = type(inner).__name__

    if inner_cls == "Len":
        return {
            "input_col": "",
            "op": "Len",
            "output_alias": output_alias or "len",
        }

    if inner_cls != "Agg":
        return None

    agg_name = getattr(inner, "name", None)
    if agg_name is None:
        return None
    op = _AGG_NAME_TO_OP.get(str(agg_name))
    if op is None:
        return None

    # .arguments is a list of raw int arena indices (not PyExprIR objects).
    args = getattr(inner, "arguments", None)
    if not args or len(args) != 1:
        return None
    arg_id = args[0]
    if not isinstance(arg_id, int):
        return None

    try:
        col_expr = nt.view_expression(arg_id)
    except Exception:
        return None
    if type(col_expr).__name__ != "Column":
        return None
    col_name = getattr(col_expr, "name", None)
    if col_name is None:
        return None

    # Look up the column dtype in the input schema rather than via get_dtype,
    # because agg argument columns are not in the GroupBy output schema.
    dtype = in_schema.get(str(col_name))
    if dtype is None:
        return None
    if _map_dtype(dtype) is None:
        return None

    return {
        "input_col": str(col_name),
        "op": op,
        "output_alias": output_alias or f"{col_name}_{op.lower()}",
    }


_CMP_OP_NAMES: dict[str, str] = {
    # Polars' Operator enum's `str(op)` form → our MetalDtype-side op tag.
    "Operator.Eq": "Eq",
    "Operator.NotEq": "Ne",
    "Operator.Lt": "Lt",
    "Operator.LtEq": "Le",
    "Operator.Gt": "Gt",
    "Operator.GtEq": "Ge",
}

# Bool combinators: Polars' `Operator.{And, Or}` (the Python-level `&`/`|`
# operators on Bool expressions). `LogicalAnd`/`LogicalOr` are the
# short-circuiting variants used for non-Boolean lhs operands; we don't
# encounter them in predicate position because the predicate must
# evaluate to Boolean.
_LOGICAL_OP_NAMES: dict[str, str] = {
    "Operator.And": "And",
    "Operator.Or": "Or",
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
    - ``{"kind": "And"|"Or", "lhs": <predicate>, "rhs": <predicate>}`` —
      Phase 7 (Task 20). Both sides must resolve to Boolean output
      (a ``Column(Bool)``, a ``Compare``, or another ``And``/``Or``).

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
        op_str = str(expr.op)
        if op_str in _LOGICAL_OP_NAMES:
            return _walk_logical(nt, expr, _LOGICAL_OP_NAMES[op_str])
        op_tag = _CMP_OP_NAMES.get(op_str)
        if op_tag is None:
            # Operator.Plus/Minus/Xor/etc. — not in this task's closed set.
            return None
        lhs = _walk_predicate(nt, expr.left)
        rhs = _walk_predicate(nt, expr.right)
        if lhs is None or rhs is None:
            return None
        # A Compare always returns Bool, so a Compare-inside-Compare
        # ("(a > 0) > 1") would be a type error in Polars itself; reject
        # defensively rather than try to coerce. AND/OR combine Compares;
        # they're handled by the _LOGICAL_OP_NAMES branch above.
        if lhs["kind"] == "Compare" or rhs["kind"] == "Compare":
            return None
        if lhs["kind"] in ("And", "Or") or rhs["kind"] in ("And", "Or"):
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


def _walk_logical(nt: Any, expr: Any, op_tag: str) -> dict | None:
    """Walk a ``BinaryExpr(And|Or, lhs, rhs)`` into an AND/OR predicate dict.

    Both sides must resolve to a Boolean-typed predicate — i.e. one of:
    ``Column(Bool)``, ``Compare`` (always returns Bool), ``And``, or ``Or``.
    ``LiteralBool`` is rejected for the same reason a bare literal is
    rejected at the Filter root: Polars would broadcast a literal mask,
    and we keep the closed set tight to avoid surprising shapes hitting
    the kernel.
    """
    lhs = _walk_predicate(nt, expr.left)
    rhs = _walk_predicate(nt, expr.right)
    if lhs is None or rhs is None:
        return None
    if _result_dtype(lhs) != "Bool" or _result_dtype(rhs) != "Bool":
        return None
    # LiteralBool is a Bool-typed predicate but degenerate (constant mask);
    # the optimizer should have collapsed it. Match the Filter-root rule
    # rather than asking the kernel to broadcast a scalar bit-packed mask.
    if lhs["kind"] == "LiteralBool" or rhs["kind"] == "LiteralBool":
        return None
    return {"kind": op_tag, "lhs": lhs, "rhs": rhs}


def _result_dtype(pred: dict) -> str | None:
    """Return the dtype tag of the *result* a predicate produces.

    For leaves this is the column/literal dtype; for combinators
    (``Compare``, ``And``, ``Or``) it is always ``"Bool"``.
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
    if k in ("Compare", "And", "Or"):
        return "Bool"
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

    The current closed set: Int64 (i64), Float64 (f64), Boolean, Int32 (i32),
    Float32 (f32). Everything else (String, Categorical, List, Struct, etc.)
    is a fallback.
    """
    s = str(dt)
    if s == "Int64":
        return "I64"
    if s == "Float64":
        return "F64"
    if s == "Boolean":
        return "Bool"
    if s == "Int32":
        return "I32"
    if s == "Float32":
        return "F32"
    return None
