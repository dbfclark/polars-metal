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
    if not any(v == "gpu_lift" for v in lifting.values()):
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
    return out
