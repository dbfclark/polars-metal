"""Polars engine callback. Walks the IR; dispatches handled subtrees via UDF.

In M0 every node fell back to CPU. M1 Phase 4 (Task 7) recognizes
DataFrameScan + SimpleProjection / Select; when every node in the
subtree is supported we replace the root with a Python UDF that
reconstructs the requested DataFrame.

The walker (`_walker.walk`) returns ``Handled`` or ``FallBack``; only on
``Handled`` do we call ``nt.set_udf``. Any failure inside the walker
(including unexpected exceptions) falls back rather than escaping to the
user's query.
"""

from __future__ import annotations

import logging
from typing import Any

from polars_metal._engine import MetalEngine
from polars_metal._udf import build_udf
from polars_metal._walker import FallBack, Handled, walk

log = logging.getLogger("polars_metal")


def execute_with_metal(nt: Any, duration_since_start: int | None, *, config: MetalEngine) -> None:
    """Polars post-optimization callback entry point.

    ``nt`` is the Polars NodeTraverser exposing the optimized IR plus the
    expression/lp arenas. We walk it bottom-up: if every node we visit is
    in our supported set, we install a UDF that produces the resulting
    DataFrame. Otherwise we return without mutating ``nt``, leaving the
    CPU executor to run the query.
    """
    if config.debug:
        log.debug("polars_metal: execute_with_metal invoked")

    try:
        result = walk(nt)
    except Exception as e:
        # Defensive: a bug in the walker itself must never take down a
        # user's query. Fall back to CPU and log loudly if debug=True.
        if config.debug:
            log.debug("polars_metal: walker raised %r; falling back", e)
        return

    if isinstance(result, FallBack):
        if config.debug:
            log.debug("polars_metal: falling back: %s", result.reason)
        return

    assert isinstance(result, Handled)
    nt.set_udf(build_udf(result.plan))
    if config.debug:
        log.debug("polars_metal: installed UDF for plan kind=%s", result.plan["kind"])
