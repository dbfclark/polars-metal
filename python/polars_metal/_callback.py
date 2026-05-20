"""Polars engine callback. Walks the IR; in M0 falls back to CPU on every node."""

from __future__ import annotations

import logging
from functools import singledispatch
from typing import Any

from polars_metal._engine import MetalEngine

log = logging.getLogger("polars_metal")


@singledispatch
def _handle_node(node: Any, nt: Any, config: MetalEngine) -> bool:
    """Returns True if `node` can be handled on GPU; False to fall back.

    M0 has no handlers, so the default arm returns False for every node.
    M1 registers handlers via `@_handle_node.register(...)`.
    """
    if config.debug:
        log.debug("polars_metal: falling back for node type %s", type(node).__name__)
    return False


def execute_with_metal(nt: Any, duration_since_start: int | None, *, config: MetalEngine) -> None:
    """Polars engine entry point.

    `nt` is a polars NodeTraverser carrying the optimized IR and arenas.
    We walk the tree; if every node says "can handle," we replace the
    plan via `nt.set_udf(...)`. In M0 nothing is handled, so we return
    without calling `set_udf` — Polars' CPU executor takes over.
    """
    # M0: no walk needed — nothing is supported. The architecture supports
    # walking via `nt.view_current_node()` / `nt.get_inputs()` / etc. once
    # M1 registers real handlers.
    if config.debug:
        log.debug("polars_metal: execute_with_metal invoked, falling back")
    return
