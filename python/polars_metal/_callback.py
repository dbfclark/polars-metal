"""Polars engine callback. Walks the IR; dispatches handled subtrees via UDF.

In M0 every node fell back to CPU. M1 recognized DataFrameScan + SimpleProjection
/ Select + Filter. M2 moved routing decisions out of the walker into the Rust
router (see `crates/polars-metal-core/src/router/`). The walker still walks the
IR and produces a plan dict; the router decides which subtrees route to GPU.

For Phase 1 the cost model defaults filter→CPU and project→inherits, so all
M1 queries route to CPU. GPU paths come online with hash groupby (Phase 5+).
"""

from __future__ import annotations

import logging
from typing import Any

from polars_metal import _native
from polars_metal._engine import MetalEngine
from polars_metal._udf import build_udf
from polars_metal._walker import FallBack, Handled, walk

log = logging.getLogger("polars_metal")


def execute_with_metal(nt: Any, duration_since_start: int | None, *, config: MetalEngine) -> None:
    """Polars post-optimization callback entry point.

    Walks the IR. If the walker handles every node, builds a wire-form plan,
    asks the Rust router for a lifting plan, and only installs the UDF if at
    least one node was lifted to GPU. Otherwise leaves ``nt`` untouched.
    """
    if config.debug:
        log.debug("polars_metal: execute_with_metal invoked")

    try:
        result = walk(nt)
    except Exception as e:
        if config.debug:
            log.debug("polars_metal: walker raised %r; falling back", e)
        return

    if isinstance(result, FallBack):
        if config.debug:
            log.debug("polars_metal: walker fallback: %s", result.reason)
        return

    assert isinstance(result, Handled)
    plan = result.plan

    # M10: a Join-bearing plan can't cross the Rust router (`compute_lifting_plan`
    # and `_strip_side_channels` don't model a Join node). Bypass the router and
    # build the whole-plan scan-source UDF directly from the Join node (which
    # carries both scan dfs + a `_parent_chain` back-ref to the chain above it).
    # The UDF reproduces the full HStack output (join + chain).
    if _plan_has_join(plan):
        join_plan = _find_join_plan(plan)
        if not config.force_fusion and not _join_routes_gpu(plan, join_plan):
            if config.debug:
                log.debug("polars_metal: join below density gate; routing CPU")
            return  # full CPU
        try:
            udf = build_udf(join_plan)
        except Exception as e:
            if config.debug:
                log.debug("polars_metal: join UDF build failed %r; falling back", e)
            return
        nt.set_udf(udf)
        if config.debug:
            log.debug("polars_metal: installed join UDF (CPU-lookup branch)")
        return

    wire_plan = _strip_side_channels(plan)
    try:
        lifting = _native.compute_lifting_plan(wire_plan)
    except Exception as e:
        if config.debug:
            log.debug("polars_metal: router raised %r; falling back", e)
        return

    if config.debug:
        log.debug("router decisions: %r", dict(lifting))

    if any(v.startswith("fallback:") for v in lifting.values()):
        if config.debug:
            log.debug("polars_metal: router fallback: %s", lifting)
        return

    # M4 Phase 5: the Rust router doesn't yet know about FusedExprGraph
    # bindings; it leaves HStacks on CPU by default. If the walker has
    # attached a fused scope to an HStack binding, we override the router
    # and install the UDF (the Python `_dispatch_hstack_fused` path
    # intercepts before any Rust expression eval).
    has_fused_binding = _plan_has_fused_binding(plan)

    if not has_fused_binding and not any(v == "gpu_lift" for v in lifting.values()):
        if config.debug:
            log.debug("polars_metal: router routes entire query to CPU")
        return

    try:
        udf = build_udf(plan)
    except NotImplementedError as e:
        if config.debug:
            log.debug(
                "polars_metal: UDF builder not ready for plan kind=%s (%r); falling back",
                plan["kind"],
                e,
            )
        return
    nt.set_udf(udf)
    if config.debug:
        log.debug(
            "polars_metal: installed UDF for plan kind=%s (lifting=%s)",
            plan["kind"],
            lifting,
        )


def _plan_has_fused_binding(plan: dict) -> bool:
    """Recurse into the plan tree looking for any HStack binding that the M4
    fusion analyzer accepted (carries a `_fused_scope` side-channel)."""
    if plan.get("kind") == "HStack":
        for binding in plan.get("exprs", []):
            if "_fused_scope" in binding:
                return True
    # M4 Phase 7: empty-key GroupBy carrying fused reduction bindings.
    if plan.get("kind") == "GroupBy" and plan.get("_fused_aggs"):
        return True
    # M4 Phase 7 (Task 27): single-column F32 Sort routed to MLX.
    if plan.get("kind") == "Sort" and plan.get("_fused_sort"):
        return True
    inner = plan.get("input")
    if isinstance(inner, dict):
        return _plan_has_fused_binding(inner)
    return False


def _plan_has_join(plan: dict) -> bool:
    if plan.get("kind") == "Join":
        return True
    inner = plan.get("input")
    return _plan_has_join(inner) if isinstance(inner, dict) else False


def _find_join_plan(plan: dict) -> dict | None:
    if plan.get("kind") == "Join":
        return plan
    inner = plan.get("input")
    return _find_join_plan(inner) if isinstance(inner, dict) else None


def _is_gpu_decision(decision: Any) -> bool:
    """True iff a ``PyFusionScope.route_decision`` result indicates GPU routing.

    The Rust side stringifies the decision as ``"Gpu"`` or ``"Cpu(<reason>)"``
    (see `fusion/py.rs` / `fusion/density.rs`). Match defensively on the repr."""
    s = str(decision).lower()
    return "gpu" in s and "cpu" not in s


def _join_routes_gpu(plan: dict, join_plan: dict | None) -> bool:
    """Density+size gate for the M10 join path. Returns True iff the fused chain
    above the join clears the GPU routing threshold (FLOPs>=5e7 AND rows>=1e5).

    Defensive: if the fused scope or the fact df can't be located, return True so
    a join path we can't assess isn't wrongly blocked (the M10 join path always
    carries a fused binding, so a miss is unexpected)."""
    if join_plan is None:
        return True
    parent_chain = join_plan.get("_parent_chain")
    if not isinstance(parent_chain, dict):
        return True
    exprs = parent_chain.get("exprs") or []
    if not exprs:
        return True
    scope = exprs[0].get("_fused_scope")
    if scope is None:
        return True

    fact_df = join_plan.get("left", {}).get("df")
    if fact_df is None:
        return True
    try:
        n_rows = fact_df.height()
    except Exception:
        try:
            import polars as pl

            n_rows = pl.DataFrame._from_pydf(fact_df).height
        except Exception:
            return True

    try:
        decision = scope.route_decision(n_rows)
    except Exception:
        return True
    return _is_gpu_decision(decision)


def _strip_side_channels(plan: dict) -> dict:
    """Remove walker-only keys (df, projection) before crossing to Rust.

    Recurses into the `input` of each non-leaf node.
    """
    out: dict = {"kind": plan["kind"]}
    if plan["kind"] == "Scan":
        df = plan.get("df")
        try:
            out["n_rows"] = df.height() if df is not None else 0
        except Exception:
            out["n_rows"] = 0
        out["columns"] = plan.get("columns", [])
    elif plan["kind"] in ("Project", "Filter"):
        out["input"] = _strip_side_channels(plan["input"])
        if plan["kind"] == "Project":
            out["columns"] = plan.get("columns", [])
        else:
            out["predicate"] = plan.get("predicate")
    elif plan["kind"] == "GroupBy":
        out["input"] = _strip_side_channels(plan["input"])
        out["keys"] = plan.get("keys", [])
        out["aggs"] = plan.get("aggs", [])
    elif plan["kind"] == "Sort":
        out["input"] = _strip_side_channels(plan["input"])
        out["by_columns"] = plan.get("by_columns", [])
        out["descending"] = plan.get("descending", [])
        out["nulls_last"] = plan.get("nulls_last", [])
    elif plan["kind"] == "HStack":
        out["input"] = _strip_side_channels(plan["input"])
        out["exprs"] = plan.get("exprs", [])
    return out
