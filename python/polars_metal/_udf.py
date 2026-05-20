"""Polars UDF entry point. Polars invokes this with
``(with_columns, predicate, n_rows, should_time)`` once the optimized plan
has decided our subtree is responsible for producing a DataFrame.

In Task 8 (Phase 4) the UDF body delegates to Rust's
``_native.execute_plan``, which interprets the plan and assembles the
result via direct ``PyDataFrame.select`` calls. The Python side only
performs the wire-format adaptation (separating the captured DataFrame
from the rest of the plan dict, applying the optional ``projection`` as
a synthetic ``Project`` wrapper) and the Polars-mandated UDF signature
shim (``should_time``, ``n_rows`` slice).

The plan-dict shape produced by ``_walker.walk()`` mirrors
``MetalPlanNode`` in ``crates/polars-metal-core/src/plan/mod.rs`` *plus*
walker-only side-channel keys on the Scan leaf:

- ``{"kind": "Scan", "columns": [(name, dtype_tag), ...], "df": <PyDataFrame>,
   "projection": list[str] | None}``
- ``{"kind": "Project", "input": <plan>, "columns": list[str]}``
- ``{"kind": "Filter", "input": <plan>, ...}``  (Phase 5+)

``df`` and ``projection`` are walker-only side channels; we strip them
before handing the plan to Rust. ``execute_plan`` expects the
``MetalPlanNode``-shaped wire format (``{"kind": "Scan", "n_rows": int,
"columns": [...]}``) — no DataFrame embedded.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from polars_metal import _native


def build_udf(plan: dict) -> Any:
    """Return a callable suitable for ``nt.set_udf(...)``.

    The returned function matches the polars-mem-engine PythonScanSource::Cuda
    signature: ``(with_columns, predicate, n_rows, should_time)``. We ignore
    ``predicate`` (the optimizer hasn't pushed predicates into our subtree —
    we returned FallBack on DataFrameScan.selection) and ``with_columns``
    arrives as ``None`` because we never opt into column-projection pushdown;
    the walker handles projection internally.

    When ``should_time`` is true Polars expects a ``(df, timings)`` tuple.
    We don't measure kernel timings yet; emit an empty timing list.
    """
    df_pydf, wire_plan = _extract_scan_df_and_wire_plan(plan)

    def udf(
        with_columns: list[str] | None,
        predicate: Any,
        n_rows: int | None,
        should_time: bool,
    ) -> Any:
        result_pydf = _native.execute_plan(df_pydf, wire_plan)
        df = pl.DataFrame._from_pydf(result_pydf)
        # Apply Polars-requested slice if any. Defensive: in Phase 4 the
        # optimizer should not push a slice into us, but if it does we honor
        # it rather than silently producing a too-large frame.
        if n_rows is not None:
            df = df.slice(0, n_rows)
        if should_time:
            return df, []
        return df

    return udf


def _extract_scan_df_and_wire_plan(plan: dict) -> tuple[Any, dict]:
    """Walk the plan dict, extract the captured ``PyDataFrame`` from its
    (unique) Scan leaf, and rewrite the tree into the Rust wire format.

    The walker stores ``df`` and ``projection`` on the Scan as side channels;
    Rust's ``MetalPlanNode::Scan`` wants ``{kind, n_rows, columns}`` only. If a
    ``projection`` is present we lift it into a synthetic ``Project`` wrapper
    above the Scan so the Rust dispatch handles it uniformly.

    Returns ``(df_pydf, wire_plan)``.

    Multiple-Scan plans are not expected in M1 (no joins/unions yet); if one
    appears we raise — better to surface than to silently mis-route.
    """
    captured: list[Any] = []

    def rewrite(node: dict) -> dict:
        kind = node["kind"]
        if kind == "Scan":
            df = node["df"]
            projection = node.get("projection")
            captured.append(df)
            cols = node["columns"]
            n_rows = df.height()
            # When PyDataFrame has no `height` attribute (older Polars), fall
            # back to len(); but py-1.40.1 exposes `.height()`.
            scan_wire = {
                "kind": "Scan",
                "n_rows": n_rows,
                # Rust extracts each entry as a 2-tuple (name, dtype); a list
                # of [name, dtype] also extract()s as a 2-tuple, but we
                # normalize to tuples for clarity.
                "columns": [(str(name), str(dtype)) for name, dtype in cols],
            }
            if projection is None:
                return scan_wire
            return {
                "kind": "Project",
                "columns": list(projection),
                "input": scan_wire,
            }
        if kind == "Project":
            return {
                "kind": "Project",
                "columns": list(node["columns"]),
                "input": rewrite(node["input"]),
            }
        if kind == "Filter":
            # Pass through unchanged for now; Rust raises NotImplementedError.
            return {
                "kind": "Filter",
                "predicate": node["predicate"],
                "input": rewrite(node["input"]),
            }
        raise ValueError(f"unknown plan kind: {kind!r}")

    wire = rewrite(plan)
    if len(captured) != 1:
        raise RuntimeError(
            f"polars_metal: expected exactly one Scan leaf in plan, got {len(captured)}. "
            "Multi-scan plans (joins/unions) are not supported in M1."
        )
    return captured[0], wire
