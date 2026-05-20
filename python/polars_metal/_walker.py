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
  dtype is Boolean; everything else falls back.

PyExprIR shape (from ``polars-python/src/lazyframe/visit.rs``): ``.node``
(usize into the expression arena) and ``.output_name`` (str). To inspect
the underlying expression class we call ``nt.view_expression(node_id)``;
to resolve its dtype against the current IR root we call
``nt.get_dtype(node_id)``.

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
    """Lower a Filter node iff its predicate is a single ``Column`` of Boolean dtype.

    Phase 5 supports only the precomputed-mask shape:
    ``df.filter(pl.col("mask"))``. Arithmetic / comparison / compound
    predicates land in Phases 6+ (Tasks 16-20); for now they FallBack so
    the CPU executor produces the correct result.

    The plan we emit captures the predicate column name so the Rust UDF
    can locate it in the upstream DataFrame at dispatch time.
    """
    pred_expr_ir = getattr(node, "predicate", None)
    if pred_expr_ir is None:
        return FallBack(reason="Filter node missing .predicate")

    pred_node_id = getattr(pred_expr_ir, "node", None)
    if pred_node_id is None:
        return FallBack(reason="Filter predicate has no .node id")

    # IMPORTANT: dtype must be resolved against the *input* schema (the
    # Filter's child), not the post-Filter schema. The current node is
    # still the Filter itself here, and Polars' get_dtype uses
    # `self.root.schema()` which for a Filter is the input schema —
    # confirmed in polars-python's visit.rs:get_dtype implementation.
    try:
        inner_expr = nt.view_expression(pred_node_id)
    except Exception as ex:
        return FallBack(reason=f"could not view filter predicate expression: {ex!r}")

    inner_cls = type(inner_expr).__name__
    if inner_cls != "Column":
        return FallBack(
            reason=f"Filter predicate is {inner_cls}; Phase 5 supports only Column(bool)"
        )

    pred_col_name = getattr(inner_expr, "name", None)
    if pred_col_name is None:
        return FallBack(reason="Column predicate missing .name")

    # Resolve the column's dtype. Must be Boolean.
    try:
        pred_dtype = nt.get_dtype(pred_node_id)
    except Exception as ex:
        return FallBack(reason=f"could not resolve filter predicate dtype: {ex!r}")
    if _map_dtype(pred_dtype) != "Bool":
        return FallBack(
            reason=f"Filter predicate dtype is {pred_dtype!s}; Phase 5 requires Boolean"
        )

    # Walk the input subtree. We re-validate the *output* schema dtypes
    # for the surviving columns (the post-Filter schema is the same as
    # the input schema, minus nothing — Filter doesn't drop columns).
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
            "predicate": {
                "kind": "Column",
                "name": str(pred_col_name),
                "dtype": "Bool",
            },
        }
    )


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
