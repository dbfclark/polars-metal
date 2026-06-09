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

import logging
from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime as _datetime
from typing import Any

import polars as pl

from polars_metal._fusion_analyzer import (
    analyze_ir_reduction,
    analyze_ir_validity,
    analyze_ir_with_columns,
    has_structural_null_op,
    null_mode_ir,
)
from polars_metal._fusion_analyzer import (
    build_sort_scope as _build_sort_scope,
)

_fusion_log = logging.getLogger("polars_metal.fusion")

# B1: wire integer dtype tag -> Polars dtype string (as `nt.get_dtype(...)`
# stringifies it). Used to admit a monomorphic-int fused HStack binding whose
# statically-inferred output tag matches the real output dtype.
_INT_TAG_TO_POLARS: dict[str, str] = {
    "I8": "Int8",
    "I16": "Int16",
    "I32": "Int32",
    "I64": "Int64",
    "U8": "UInt8",
    "U16": "UInt16",
    "U32": "UInt32",
    "U64": "UInt64",
}

# Dtype tags the predicate path widens to I64 (the cmp_i64 kernel covers
# all of them; the runtime evaluator casts the underlying buffer). Narrow
# unsigned integers fit losslessly in i64; Date is stored as i32 days
# since 1970-01-01 and widens to the literal's days-since-epoch encoding.
_PREDICATE_I64_WIDEN: set[str] = {"I8", "I16", "I32", "U8", "U16", "U32", "Date"}

# F32 columns widen to F64 for the cmp_f64 kernel — M2 only emitted F64
# / I64 cmp shaders, and F32 → F64 casts are exact, so this is a lossless
# detour. Without this, any F32 predicate falls back at the walker.
_PREDICATE_F64_WIDEN: set[str] = {"F32"}

_EPOCH = _date(1970, 1, 1)


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
    if cls == "Sort":
        return _walk_sort(nt, node)
    if cls == "HStack":
        return _walk_hstack(nt, node)
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

    # Multi-chunk Series are handled transparently by `_materialize_arrow`,
    # which calls `combine_chunks()` before extracting buffer bytes. M2
    # used to fall back here as a defensive guard; Phase 9 removes that —
    # concatenated DataFrames and parquet row-group reads commonly land
    # multi-chunk and we shouldn't punt the whole subtree just for that.
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
    """Lower a Select node. Two recognized shapes:

    - **Projection**: every expression is a plain ``Column(name)`` — emits a
      ``Project`` plan node (M2 path).
    - **Reduction** (Phase 13): every expression is an ``Agg`` or ``Len``
      wrapping a column or expression — emits a ``GroupBy`` plan node with
      empty keys. This is the IR shape TPC-H Q6 takes after optimization:
      ``SELECT [(a*b).sum().alias(...)]`` over ``Filter/SimpleProjection``.
      The kernel layer treats empty keys as a single group covering all
      input rows.

    A Select that mixes column projections and aggregations is not a valid
    Polars shape at this level (the optimizer rewrites that into a
    Select-over-GroupBy), so we don't try to handle the mix.
    """
    exprs = getattr(node, "expr", None)
    if exprs is None:
        return FallBack(reason="Select node missing .expr")
    if len(exprs) == 0:
        return FallBack(reason="Select with zero expressions")

    classes: list[tuple[Any, str]] = []
    for e in exprs:
        try:
            inner_node = nt.view_expression(e.node)
        except Exception as ex:
            return FallBack(reason=f"could not view expression: {ex!r}")
        classes.append((inner_node, type(inner_node).__name__))

    all_columns = all(cls == "Column" for _, cls in classes)
    all_aggs = all(cls in ("Agg", "Len") for _, cls in classes)

    if all_columns:
        return _walk_select_projection(nt, exprs, classes)
    if all_aggs:
        return _walk_select_reduction(nt, exprs)
    return FallBack(reason="Select mixes projections and aggregations")


def _walk_select_projection(
    nt: Any, exprs: list[Any], classes: list[tuple[Any, str]]
) -> WalkResult:
    """Existing M2 path: column-only Select becomes a Project plan node."""
    columns: list[str] = []
    for e, (inner_node, _cls) in zip(exprs, classes, strict=False):
        col_name = getattr(inner_node, "name", None)
        if col_name is None:
            return FallBack(reason="Column expression missing .name")
        if str(col_name) != str(e.output_name):
            return FallBack(reason="aliased Column projection")
        columns.append(str(col_name))

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


def _walk_select_reduction(nt: Any, exprs: list[Any]) -> WalkResult:
    """Phase 13: Select(agg(...)) → empty-key GroupBy plan node.

    The aggregation argument columns live in the *input* schema (not the
    Select's output schema), so we mirror ``_walk_group_by``: navigate to
    the input child, fetch its schema, then walk each agg expression
    against that schema before recursing into the inner subtree.
    """
    inputs = nt.get_inputs()
    if len(inputs) != 1:
        return FallBack(reason=f"Select expected 1 input, got {len(inputs)}")

    parent_id = nt.get_node()
    nt.set_node(inputs[0])
    try:
        in_schema = dict(nt.get_schema())
    finally:
        nt.set_node(parent_id)

    # M4 Phase 7 (Task 26): if every aggregation is a fusion-eligible F32
    # reduction (sum/mean/min/max/std/var of one column or a compute chain),
    # route the whole Select through the fused MLX path instead of the
    # empty-key GroupBy conformance kernel. All-or-nothing: a single
    # non-eligible agg falls through to the GroupBy path below.
    fused_aggs = _try_fused_select_reduction(nt, exprs, in_schema)
    if fused_aggs is not None:
        nt.set_node(inputs[0])
        inner = _walk_at_current(nt)
        if not isinstance(inner, FallBack) and _route_fused_reduction(fused_aggs, inner.plan):
            return Handled(
                plan={
                    "kind": "GroupBy",
                    "input": inner.plan,
                    "keys": [],
                    # Empty aggs: the router parses this as a no-op empty-key
                    # GroupBy (non-fallback); the `_fused_aggs` side-channel
                    # drives `_build_select_reduction_fused` on the dispatch
                    # side, bypassing the GroupBy kernel entirely.
                    "aggs": [],
                    "_fused_aggs": fused_aggs,
                }
            )
        # inner fell back, or a chain reduction reads a possibly-null column
        # (its null-skip can't be replayed) — fall through to the GroupBy/CPU
        # path below, which preserves Polars semantics exactly.

    aggs: list[dict[str, str]] = []
    for agg_expr in exprs:
        agg_dict = _walk_agg_expression(nt, agg_expr, in_schema)
        if agg_dict is None:
            return FallBack(reason="Select reduction expression not in closed set")
        aggs.append(agg_dict)

    nt.set_node(inputs[0])
    inner = _walk_at_current(nt)
    if isinstance(inner, FallBack):
        return inner

    return Handled(
        plan={
            "kind": "GroupBy",
            "input": inner.plan,
            "keys": [],
            "aggs": aggs,
        }
    )


# Bare (plain-column) reductions worth routing to MLX on their own: their CPU
# cost clears the ~1 ms fused-dispatch floor. sum/min/max/mean are bandwidth-
# bound (a ~0.35 ms memory scan on CPU) and lose ~3x on Metal, so a *lone* one
# stays on CPU — it only rides to GPU alongside a compute-worthy reduction in
# the same select (CLAUDE.md principle #3: route on compute intensity, not op
# identity). Chain-terminated reductions are worthy regardless of op (the chain
# amortizes the floor) — that lands in a follow-up increment.
_BARE_GPU_WORTHY_REDUCTIONS: frozenset[str] = frozenset({"std", "var"})


def _try_fused_select_reduction(
    nt: Any, exprs: list[Any], in_schema: dict[str, Any]
) -> list[dict] | None:
    """Return a list of fused-reduction binding dicts if every expression in
    `exprs` is a fusion-eligible F32 reduction AND at least one is GPU-worthy,
    else None.

    Each binding carries the analyzed scope, its ordered input descriptors,
    the output column name, and the lowercase agg kind (for the dispatch-side
    Bessel correction on std/var).
    """
    bindings: list[dict] = []
    for agg_expr in exprs:
        node_id = getattr(agg_expr, "node", None)
        output_name = getattr(agg_expr, "output_name", None)
        if node_id is None or output_name is None:
            return None
        result = analyze_ir_reduction(nt, node_id, in_schema)
        if result is None:
            return None
        scope, columns, agg_kind, is_chain, arg_id, out_dtype_str = result
        _fusion_log.info(
            "FusedReduction candidate column=%r kind=%s n_inputs=%d n_ops=%d chain=%s",
            output_name,
            agg_kind,
            scope.n_inputs(),
            scope.n_ops(),
            is_chain,
        )
        bindings.append(
            {
                "name": output_name,
                "_fused_scope": scope,
                "_fused_columns": columns,
                "_agg_kind": agg_kind,
                "_is_chain": is_chain,
                # Statically-inferred wire output dtype ("F32" for the float
                # path, or an int tag like "I64" for a GPU-admissible int
                # reduction). The dispatch pre-allocates the right-width output;
                # the full int_fused_ok routing guard lands in Task 3.
                "_fused_out_dtype": out_dtype_str,
                # Null mode of the chain argument (None for bare). An
                # "elementwise" chain over a null column can reduce on the GPU
                # after dropping nulls (positions don't matter for a reduction);
                # "where" can't (a null cond keeps the else branch valid).
                "_null_mode": null_mode_ir(nt, arg_id, in_schema) if is_chain else None,
            }
        )
    if not bindings:
        return None
    # Compute-intensity gate: route to MLX only if at least one reduction clears
    # the dispatch floor — a compute-bound bare op (std/var) or any reduction
    # over a compute chain (the chain amortizes the floor). A select of only
    # bandwidth-bound bare sum/min/max/mean stays on CPU (~3x faster there).
    if not any(b["_agg_kind"] in _BARE_GPU_WORTHY_REDUCTIONS or b["_is_chain"] for b in bindings):
        return None
    return bindings


def _route_fused_reduction(fused_aggs: list[dict], inner_plan: dict) -> bool:
    """Decide each chain reduction's null strategy; return False to send the
    whole select to CPU.

    A bare reduction handles nulls in the dispatch (replaying the reduction on
    the source column). For a *chain* reduction whose inputs may have nulls:
      - **elementwise** chain → stamp ``_drop_nulls``: drop the null rows in
        Polars (native, ~one SIMD pass), then reduce the dense survivors on the
        GPU. Lossless because a reduction skips nulls and doesn't care about row
        positions — there's nothing to rejoin (the output is a scalar).
      - **where** chain (or any other null mode) → CPU: a null cond keeps the
        else branch *valid*, so dropping the row is wrong, and the chain can't
        be replayed from the wire plan.
    Confirmed-null-free chains keep the zero-`drop_nulls` fast path.
    """
    for agg in fused_aggs:
        if not agg.get("_is_chain"):
            continue
        cols = {name for kind, name in agg["_fused_columns"] if kind == "col"}
        if _fused_inputs_null_free(inner_plan, cols):
            agg["_drop_nulls"] = False
        elif agg.get("_null_mode") == "elementwise":
            agg["_drop_nulls"] = True
        else:
            return False
    return True


def _probe_fusion_analyzer(
    nt: Any, node_id: int, in_schema: dict[str, Any], output_name: str
) -> tuple[Any, list[str], str] | None:
    """Run the M4 fusion analyzer on an HStack binding, log the decision,
    and return ``(scope, input_descriptors, out_dtype_str)`` when the analyzer
    accepts the expression. Returns ``None`` on rejection.

    ``out_dtype_str`` is the statically-inferred wire output dtype (``"F32"``
    for the legacy float path, or an integer tag like ``"I64"`` for a B1
    monomorphic-int chain). The walker stamps it on the binding so the
    dispatch pre-allocates the right-width output array.

    The returned scope is stashed on the binding's wire-plan dict as a
    side-channel for Phase 5 dispatch.
    """
    try:
        result = analyze_ir_with_columns(nt, node_id, in_schema)
    except Exception as e:
        _fusion_log.debug("analyzer raised for %r: %r", output_name, e)
        return None
    if result is None:
        _fusion_log.debug("analyzer rejected expr for column %r (unsupported op)", output_name)
        return None
    scope, columns, out_dtype_str = result
    decision = scope.route_decision(10_000_000)
    _fusion_log.info(
        "FusedExprGraph candidate column=%r n_inputs=%d n_ops=%d out_dtype=%s decision=%s",
        output_name,
        scope.n_inputs(),
        scope.n_ops(),
        out_dtype_str,
        decision,
    )
    return scope, columns, out_dtype_str


def _walk_hstack(nt: Any, node: Any) -> WalkResult:
    """Lower an HStack (``with_columns``) IR node — appends one or more
    derived columns to the input frame.

    Polars' CSE optimizer hoists shared sub-expressions into a synthetic
    HStack above GroupBy. The canonical example is Q1: the
    ``(l_extendedprice * (1 - l_discount))`` sub-tree appears in two
    aggregations, so the optimizer materializes it as
    ``__POLARS_CSER_<hash>`` and rewrites the two aggs to reference the
    new column.

    Walker contract for HStack: every appended expression must be in the
    M3 capability-G closed set (BinaryExpr/Column/Literal, max depth
    ``_AGG_EXPR_MAX_DEPTH``). The dispatch path evaluates each expression
    via Polars on the upstream DataFrame and stitches the new columns in
    before downstream nodes consume.
    """
    exprs = getattr(node, "exprs", None)
    if exprs is None:
        return FallBack(reason="HStack node missing .exprs")
    if len(exprs) == 0:
        return FallBack(reason="HStack with zero expressions")

    inputs = nt.get_inputs()
    if len(inputs) != 1:
        return FallBack(reason=f"HStack expected 1 input, got {len(inputs)}")

    # Resolve expression sub-trees against the *input* schema (the
    # appended columns aren't visible until after HStack runs).
    parent_id = nt.get_node()
    nt.set_node(inputs[0])
    try:
        in_schema = dict(nt.get_schema())
    finally:
        nt.set_node(parent_id)

    out_exprs: list[dict] = []
    # Per fused binding: ``(binding, input_col_names, null_mode)``. The fused
    # MLX path ingests each input via `series.to_numpy()`, which turns nulls
    # into NaN — but Polars elementwise ops propagate nulls. After the inner
    # walk we check (against the Scan leaf) whether each binding's inputs can
    # have nulls; if so the dispatch reproduces Polars' null semantics for the
    # binding's `null_mode` (e.g. "elementwise" = union of input null masks),
    # or — for modes we can't reproduce — the whole subtree falls back to CPU.
    fused_records: list[tuple[dict, set[str], str | None, int]] = []
    for e in exprs:
        node_id = getattr(e, "node", None)
        if node_id is None:
            return FallBack(reason="HStack expression has no .node id")

        # M4 Phase 3+5: probe the fusion analyzer; if it accepts the binding
        # we route it via the fused MLX subgraph path and DO NOT require the
        # M3 closed-set check (it's a superset of what M3 supports). We emit
        # a placeholder expr_dict pointing at the first input column so the
        # Rust router still sees a valid HStack wire-plan; the Python
        # _dispatch path intercepts all-fused HStacks before any Rust
        # expression eval happens.
        output_name = str(getattr(e, "output_name", "") or "")

        fused = _probe_fusion_analyzer(nt, node_id, in_schema, output_name)

        # The M3 HStack dispatch path produces Float32 output buffers, and the
        # fused F32 path likewise. If this binding's correct output dtype is
        # anything else the GPU result would be wrong-typed/downcast — UNLESS
        # the fused analyzer statically inferred a *monomorphic-int* output
        # dtype that matches the real output dtype (B1). In that case the typed
        # FFI path produces the exact integer dtype and we let it through.
        # Everything else (Float64, mixed-int chains, Boolean comparison
        # outputs, …) still falls back so Polars' dtype is preserved exactly.
        # (An F64/int input the chain explicitly casts to F32 has an F32 output
        # and still fuses on the F32 path.)
        # Placed AFTER the fusion probe so the analyzer still runs (and logs its
        # accept/reject) for non-F32 expressions.
        try:
            out_dtype = str(nt.get_dtype(node_id))
        except Exception:
            out_dtype = "Float32"  # undeterminable — let the normal paths decide

        # The analyzer's inferred wire output tag (None when not fused).
        fused_out_tag = fused[2] if fused is not None else None
        # A monomorphic-int fused binding is allowed through iff the inferred
        # integer tag corresponds to the real Polars output dtype.
        int_fused_ok = (
            fused is not None
            and fused_out_tag in _INT_TAG_TO_POLARS
            and out_dtype == _INT_TAG_TO_POLARS[fused_out_tag]
        )
        if out_dtype != "Float32" and not int_fused_ok:
            return FallBack(
                reason=f"HStack binding output dtype is {out_dtype}, not Float32 "
                "and not a recognized monomorphic-int fused output; the GPU path "
                "can't produce it — CPU preserves Polars' dtype"
            )

        if fused is not None:
            scope, descriptors, fused_out_dtype = fused
            # Find the first real column descriptor for the placeholder; if the
            # expression is all literals (rare but possible), grab any column
            # name from the input schema.
            real_col = next(
                (name for kind, name in descriptors if kind == "col"),
                next(iter(in_schema), ""),
            )
            binding: dict = {
                "name": output_name,
                "expr": {"kind": "Column", "name": real_col},
                "_fused_scope": scope,
                "_fused_columns": descriptors,
                "_fused_out_dtype": fused_out_dtype,
            }
            binding_cols = {name for kind, name in descriptors if kind == "col"}
            fused_records.append(
                (binding, binding_cols, null_mode_ir(nt, node_id, in_schema), node_id)
            )
        else:
            expr_dict = _walk_agg_expr_node(nt, node_id, in_schema, _AGG_EXPR_MAX_DEPTH)
            if expr_dict is None:
                return FallBack(reason="HStack expression not in closed set")
            binding = {
                "name": output_name,
                "expr": expr_dict,
            }
        out_exprs.append(binding)

    nt.set_node(inputs[0])
    inner = _walk_at_current(nt)
    if isinstance(inner, FallBack):
        return inner

    # Null handling for the fused path. A binding whose inputs are all confirmed
    # null-free runs the fast path with no per-row mask. When an input may have
    # nulls, the dispatch reproduces Polars' null semantics for the binding's
    # mode (stamped as `_fused_null_mode`); modes we can't reproduce
    # (Kleene And/Or, null-skipping reductions, scans) fall the subtree to CPU.
    for binding, binding_cols, null_mode, node_id in fused_records:
        inputs_null_free = bool(binding_cols) and _fused_inputs_null_free(inner.plan, binding_cols)
        # A binding with no real column inputs, or whose inputs are confirmed
        # null-free, normally skips the null path entirely. BUT a structural-
        # null op (Shift) introduces leading nulls even on null-free input, so
        # such a binding must still build a validity subgraph.
        has_structural = has_structural_null_op(nt, node_id, in_schema)
        inputs_clean = inputs_null_free or not binding_cols
        if inputs_clean and not has_structural:
            continue
        if inputs_clean and has_structural:
            # Inputs are null-free, but a Shift introduces structural head-
            # nulls — build the validity subgraph in the structural_only
            # regime (validity-preserving ops like cum_sum pass instead of
            # aborting; the only nulls come from Shift). `null_mode_ir` may
            # return None here (e.g. a cum_sum upstream of the shift makes the
            # general classifier abort), so we drive the validity path off
            # `has_structural`, not `null_mode`.
            validity = analyze_ir_validity(nt, node_id, in_schema, structural_only=True)
            if validity is None:
                return FallBack(
                    reason="fused HStack has a structural-null op (shift) whose "
                    "validity graph is not constructible; CPU preserves nulls"
                )
            v_scope, v_columns = validity
            binding["_fused_null_mode"] = "where"
            binding["_fused_validity_scope"] = v_scope
            binding["_fused_validity_columns"] = v_columns
        elif null_mode == "elementwise":
            # Output null iff any input column null — dispatch ORs the input
            # null masks. No extra graph.
            binding["_fused_null_mode"] = "elementwise"
        elif null_mode == "where":
            # Data-dependent and/or structural null mask — build a validity
            # subgraph the dispatch evaluates (one extra MLX dispatch) to
            # derive the null mask. Inputs may have nulls here, so use the
            # general (conservative) regime: structural_only=False.
            validity = analyze_ir_validity(nt, node_id, in_schema, structural_only=False)
            if validity is None:
                return FallBack(
                    reason="fused HStack Where has nulls but its validity graph "
                    "is not constructible; CPU preserves Polars null semantics"
                )
            v_scope, v_columns = validity
            binding["_fused_null_mode"] = "where"
            binding["_fused_validity_scope"] = v_scope
            binding["_fused_validity_columns"] = v_columns
        else:
            return FallBack(
                reason=f"fused HStack input has nulls and null_mode={null_mode!r} "
                "is not reproducible on the fused path; CPU preserves Polars null semantics"
            )

    # The only executable HStack dispatch path is the all-fused fast path
    # (`_dispatch_hstack_fused`). A non-fused binding (an M3 closed-set expr the
    # F32 fusion analyzer rejected) — or a mix of fused and non-fused bindings —
    # would be handed to `_native.execute_plan`, which has no HStack handler and
    # raises `unknown plan kind "HStack"`. Fall back to CPU at plan time so the
    # whole query runs on Polars and produces the correct result. (Mixed/partial
    # fused dispatch was never supported; this converts a crash into a fallback.)
    if not all("_fused_scope" in b for b in out_exprs):
        return FallBack(
            reason="HStack has non-fused binding(s); only all-fused HStacks are "
            "executable on the GPU path — CPU produces the correct result"
        )

    return Handled(
        plan={
            "kind": "HStack",
            "input": inner.plan,
            "exprs": out_exprs,
        }
    )


def _find_scan_df(plan: dict) -> Any:
    """Return the captured ``PyDataFrame`` from the (unique) Scan leaf of a
    walker plan tree, or ``None`` if no Scan leaf carries one."""
    if plan.get("kind") == "Scan":
        return plan.get("df")
    inner = plan.get("input")
    if isinstance(inner, dict):
        return _find_scan_df(inner)
    return None


def _fused_inputs_null_free(inner_plan: dict, cols: set[str]) -> bool:
    """True iff every column in ``cols`` is a confirmed null-free column of the
    Scan leaf under ``inner_plan``.

    A column not present in the Scan leaf (one computed by an upstream node) is
    treated as *not confirmed* — we conservatively report False so the caller
    falls back to CPU. The headline fused shapes (haversine / Black-Scholes)
    read raw Scan columns directly, so they hit the fast O(1) ``null_count``
    check (Arrow tracks null_count; clean columns short-circuit). Chained
    ``with_columns`` over a computed column is rare and stays correct on CPU.
    """
    scan_df = _find_scan_df(inner_plan)
    if scan_df is None:
        return False
    df = pl.DataFrame._from_pydf(scan_df)
    available = set(df.columns)
    for name in cols:
        if name not in available:
            return False
        if df.get_column(name).null_count() > 0:
            return False
    return True


def _walk_sort(nt: Any, node: Any) -> WalkResult:
    """Lower a Sort node. Two shapes are handled:

    - **CPU conformance path** (M3): every ``by_column`` is a bare Column and
      no slice is set — the Sort runs CPU via Polars in the UDF, so lifting it
      lets the inner GroupBy/Filter route to Metal (TPC-H Sort-over-GroupBy).
    - **Fused single-column path** (M4 Task 27): a single F32 by-column over a
      single-column input — the sort routes to one MLX ``Sort`` op (the
      dispatch reverses for descending and slices for top_k). ``df.top_k`` /
      ``bottom_k`` lower to a Sort with ``slice=(0, k)``, handled here too.

    ``sort_options`` is ``(maintain_order, nulls_last, descending)`` (each of
    the latter a per-key list).
    """
    by_column = getattr(node, "by_column", None)
    if by_column is None:
        return FallBack(reason="Sort node missing .by_column")
    sort_slice = getattr(node, "slice", None)
    sort_options = getattr(node, "sort_options", None)
    if sort_options is None:
        return FallBack(reason="Sort missing .sort_options")
    try:
        _maintain_order, nulls_last, descending = sort_options
    except (TypeError, ValueError):
        return FallBack(reason="Sort .sort_options has unexpected shape")

    by_columns: list[str] = []
    for expr in by_column:
        node_id = getattr(expr, "node", None)
        if node_id is None:
            return FallBack(reason="Sort by_column expression has no .node id")
        try:
            inner_expr = nt.view_expression(node_id)
        except Exception as ex:
            return FallBack(reason=f"could not view sort expression: {ex!r}")
        if type(inner_expr).__name__ != "Column":
            return FallBack(
                reason=f"Sort by complex expression {type(inner_expr).__name__} not supported"
            )
        col_name = getattr(inner_expr, "name", None)
        if col_name is None:
            return FallBack(reason="Sort Column missing .name")
        by_columns.append(str(col_name))

    inputs = nt.get_inputs()
    if len(inputs) != 1:
        return FallBack(reason=f"Sort expected 1 input, got {len(inputs)}")

    # Resolve the input schema (the Sort's child) to test fused eligibility.
    parent_id = nt.get_node()
    nt.set_node(inputs[0])
    try:
        in_schema = dict(nt.get_schema())
    finally:
        nt.set_node(parent_id)

    # Fused single-column F32 sort: one MLX Sort op. Eligible iff there's a
    # single F32 by-column and the input frame has only that column (so the
    # sorted column IS the whole result — no other columns to gather, which
    # would be a bandwidth-bound permutation we deliberately leave to CPU).
    fused_sort: dict | None = None
    if (
        len(by_columns) == 1
        and len(in_schema) == 1
        and by_columns[0] in in_schema
        and _map_dtype(in_schema[by_columns[0]]) == "F32"
    ):
        col = by_columns[0]
        fused_sort = {
            "scope": _build_sort_scope(col),
            "column": col,
            "descending": bool(descending[0]) if descending else False,
            "nulls_last": bool(nulls_last[0]) if nulls_last else False,
            "slice": (int(sort_slice[0]), int(sort_slice[1])) if sort_slice is not None else None,
        }
    elif sort_slice is not None:
        # The CPU conformance path doesn't apply a slice (top_k/head shapes);
        # only the fused path does. Fall back so Polars produces them.
        return FallBack(reason="Sort with slice not supported on the CPU path")

    nt.set_node(inputs[0])
    inner = _walk_at_current(nt)
    if isinstance(inner, FallBack) and fused_sort is not None and fused_sort["slice"] is not None:
        # top_k / bottom_k lower to Sort(slice=(0,k)) over a *dynamic-predicate*
        # Filter that drops nulls — the predicate can't be viewed as a normal
        # expression (NodeTraverser raises), so the inner walk above fell back.
        # Bypass that one filter: source from its input and drop nulls in the
        # dispatch (matching top_k's null semantics). A real user filter has a
        # viewable predicate, so this never skips genuine row selection.
        bypassed = _bypass_dynamic_null_filter(nt, inputs[0])
        if bypassed is not None:
            inner = bypassed
            fused_sort["drop_nulls"] = True
    if isinstance(inner, FallBack):
        return inner

    if fused_sort is not None:
        fused_sort.setdefault("drop_nulls", False)

    plan = {
        "kind": "Sort",
        "input": inner.plan,
        "by_columns": by_columns,
        "descending": list(descending),
        "nulls_last": list(nulls_last),
    }
    if fused_sort is not None:
        plan["_fused_sort"] = fused_sort
    return Handled(plan=plan)


def _bypass_dynamic_null_filter(nt: Any, filter_node_id: int) -> Handled | None:
    """If ``filter_node_id`` is a Filter whose predicate is a *dynamic* one the
    NodeTraverser can't view (top_k/bottom_k's internal null-drop), walk and
    return its single input instead. Returns ``None`` if the node isn't such a
    filter or its input doesn't lift — the caller then keeps its FallBack."""
    nt.set_node(filter_node_id)
    node = nt.view_current_node()
    if type(node).__name__ != "Filter":
        return None
    pred = getattr(node, "predicate", None)
    pred_node = getattr(pred, "node", None)
    if pred_node is None:
        return None
    try:
        nt.view_expression(pred_node)
    except Exception:
        # Unviewable (dynamic) predicate — the top_k/bottom_k null-drop. Bypass.
        finputs = nt.get_inputs()
        if len(finputs) != 1:
            return None
        nt.set_node(finputs[0])
        inner = _walk_at_current(nt)
        return inner if isinstance(inner, Handled) else None
    # Viewable predicate => a genuine user filter; do not bypass it.
    return None


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
    #
    # Utf8 is allowed even though `compact_one_column` (M2 GPU compaction)
    # can't materialize strings: Phase 10's `_filter_via_polars` route
    # (used when a GroupBy sits above the Filter) handles strings fine
    # through Polars CPU. Top-level Filter dispatch detects Utf8 and
    # falls back to the same CPU path in `_udf._dispatch_filter`.
    out_schema = dict(nt.get_schema())
    for name, dtype in out_schema.items():
        mapped = _map_dtype(dtype)
        if mapped is None:
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


# Polars' BinaryExpr op names — verified against py-1.40.1 (probed
# 2026-05-23 from a real BinaryExpr inside an Agg). M2's predicate path
# uses the same `str(op)` discovery against the `Operator` enum.
_AGG_BINARY_OP_NAMES: dict[str, str] = {
    "Operator.Multiply": "Mul",
    "Operator.Plus": "Add",
    "Operator.Minus": "Sub",
    "Operator.TrueDivide": "Div",  # `/` on float columns
}

_AGG_EXPR_MAX_DEPTH = 4


def _walk_agg_expr_node(
    nt: Any, node_id: Any, in_schema: dict[str, Any], depth: int
) -> dict | None:
    """Recursively lower one Polars expression sub-node to an AggExpr dict.

    Returns the dict on success; returns None if the shape is outside
    capability G (function calls, comparisons, deeper than
    ``_AGG_EXPR_MAX_DEPTH``, unsupported binary ops, unknown literal types,
    columns not in the input schema). Returning None causes the caller to
    fall back the whole agg.
    """
    if depth < 0:
        return None
    try:
        node = nt.view_expression(node_id)
    except Exception:
        return None
    cls = type(node).__name__

    if cls == "Column":
        col_name = getattr(node, "name", None)
        if col_name is None:
            return None
        # Validate the column exists in input schema (same check as Simple path).
        col_dtype = in_schema.get(str(col_name))
        if col_dtype is None:
            return None
        # Utf8 is a key-only dtype; arithmetic agg expressions over strings
        # have no kernel-side support.
        if _map_dtype(col_dtype) == "Utf8":
            return None
        return {"kind": "Column", "name": str(col_name)}

    if cls == "Literal":
        val = getattr(node, "value", None)
        # `bool` is a subclass of `int`; reject bool explicitly before the
        # int branch. (Matches the predicate walker's literal handling.)
        if isinstance(val, bool):
            return None
        if isinstance(val, float):
            return {"kind": "LiteralF64", "value": float(val)}
        if isinstance(val, int):
            return {"kind": "LiteralI64", "value": int(val)}
        return None

    if cls == "BinaryExpr":
        op = getattr(node, "op", None)
        op_key = str(op) if op is not None else ""
        op_tag = _AGG_BINARY_OP_NAMES.get(op_key)
        if op_tag is None:
            return None
        left_id = getattr(node, "left", None)
        right_id = getattr(node, "right", None)
        if left_id is None or right_id is None:
            return None
        lhs = _walk_agg_expr_node(nt, left_id, in_schema, depth - 1)
        if lhs is None:
            return None
        rhs = _walk_agg_expr_node(nt, right_id, in_schema, depth - 1)
        if rhs is None:
            return None
        return {"kind": "Binary", "op": op_tag, "lhs": lhs, "rhs": rhs}

    return None


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
      entries in ``_AGG_NAME_TO_OP`` — emits ``"kind": "Simple"``.
    - ``Agg(name=<op>, arguments=[binary_expr_arena_id])`` where the argument
      resolves to a ``BinaryExpr`` over Add/Sub/Mul/Div whose operand tree
      bottoms out at supported ``Column``/``Literal`` leaves within the
      depth cap — emits ``"kind": "Expression"`` with a recursive ``expr``
      sub-dict (capability G, M3 Phase 2).
    - ``Len`` (no arguments) — emits ``"kind": "Length"``.

    Everything else (unknown agg names, multi-arg aggs, function calls or
    comparisons inside the argument, columns absent from the input schema)
    returns ``None`` and causes the GroupBy to fall back.
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
            "kind": "Length",
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
    inner_arg_cls = type(col_expr).__name__

    if inner_arg_cls == "BinaryExpr":
        # Capability G (M3 Phase 2): aggregate over an inline binary
        # expression like ``(pl.col("a") * pl.col("b")).sum()``. Lower the
        # sub-tree recursively; if any part of it is outside the supported
        # closed set (function calls, comparisons, depth > cap, unknown
        # literals, missing columns) the extractor returns None and we fall
        # back the whole agg.
        expr_dict = _walk_agg_expr_node(nt, arg_id, in_schema, _AGG_EXPR_MAX_DEPTH)
        if expr_dict is None:
            return None
        return {
            "kind": "Expression",
            "expr": expr_dict,
            "op": op,
            "output_alias": output_alias or f"expr_{op.lower()}",
        }

    if inner_arg_cls != "Column":
        return None
    col_name = getattr(col_expr, "name", None)
    if col_name is None:
        return None

    # Look up the column dtype in the input schema rather than via get_dtype,
    # because agg argument columns are not in the GroupBy output schema.
    dtype = in_schema.get(str(col_name))
    if dtype is None:
        return None
    mapped = _map_dtype(dtype)
    if mapped is None:
        return None
    # Utf8 is a key-only dtype: sum/mean/min/max over strings have no
    # kernel-side implementation. Reject so the whole GroupBy falls
    # back rather than dispatching a broken kernel.
    if mapped == "Utf8":
        return None

    return {
        "kind": "Simple",
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
        # Predicate-side widening: narrow integer dtypes and Date are
        # stored internally as i8/i16/i32; the cmp_i64 kernel covers all
        # i64 comparisons, so we widen here and the evaluator
        # (`_udf._evaluate_compare`) casts the Polars Series to Int64
        # before materializing the buffer. F32 columns analogously widen
        # to F64 for the cmp_f64 kernel.
        if m1_dtype in _PREDICATE_I64_WIDEN:
            m1_dtype = "I64"
        elif m1_dtype in _PREDICATE_F64_WIDEN:
            m1_dtype = "F64"
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
        # `datetime.date` (but not `datetime.datetime`, which is a Date
        # subclass with finer resolution). Convert to days-since-1970 so
        # the value matches the widened Date column buffer.
        if isinstance(value, _date) and not isinstance(value, _datetime):
            return {"kind": "LiteralI64", "value": (value - _EPOCH).days}
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

    M2 set: Int64, Float64, Boolean, Int32, Float32.
    M3 adds: Int8, Int16, UInt8, UInt16, UInt32 (capability F).
    M3 Phase 7 (Task 34) adds: String (pl.Utf8) — dictionary-encoded as
    a key column. py-1.40.1 reports ``str(pl.Utf8) == "String"``; older
    Polars reported ``"Utf8"`` — accept both for forward/back compat.
    Everything else (Categorical, List, Struct, etc.) is a fallback.
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
    if s == "Int8":
        return "I8"
    if s == "Int16":
        return "I16"
    if s == "UInt8":
        return "U8"
    if s == "UInt16":
        return "U16"
    if s == "UInt32":
        return "U32"
    if s in ("String", "Utf8"):
        return "Utf8"
    if s == "Date":
        # pl.Date is stored as Int32 days-since-1970-01-01. The predicate
        # path widens Date comparisons to the cmp_i64 kernel via a runtime
        # cast (see `_udf._evaluate_compare`); other paths that pass
        # through `_map_dtype` will today reject Date downstream (no
        # groupby-on-Date, no sum-of-Date), which is the correct behavior
        # for M3.
        return "Date"
    return None
