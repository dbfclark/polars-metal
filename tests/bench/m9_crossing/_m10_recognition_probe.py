"""M10 recognition spike (throwaway): does the post-opt NodeTraverser expose a
gather/take expression, or is it opaque like list/array/corr were in M4?

We probe several candidate Polars shapes for "gather a per-key value, feed an F32
compute chain" and, for every expression node the walker would see, record whether
`nt.view_expression(node_id)` succeeds (viewable -> class name) or raises (opaque).

Run with raw polars (NOT engine='metal') so we get the unpatched NodeTraverser.
"""

from __future__ import annotations

import traceback
from typing import Any

import numpy as np
import polars as pl


_EXPR_CHILD_ATTRS = (
    "expr",
    "input",
    "left",
    "right",
    "by",
    "idx",
    "indices",
    "truthy",
    "falsy",
    "predicate",
)


def _child_node_ids(inner: Any) -> list[int]:
    """Pull child expression node-ids out of a viewed expression object."""
    ids: list[int] = []
    for attr in _EXPR_CHILD_ATTRS:
        v = getattr(inner, attr, None)
        if v is None:
            continue
        items = v if isinstance(v, (list, tuple)) else [v]
        for it in items:
            nid = getattr(it, "node", None)
            if isinstance(nid, int):
                ids.append(nid)
    return ids


def _probe_callback(report: dict[str, Any]):
    def cb(nt: Any, _duration: int | None = None) -> None:
        seen_nodes: list[str] = []
        expr_results: list[str] = []

        def walk_expr(node_id: int, depth: int) -> None:
            pad = "  " * depth
            try:
                inner = nt.view_expression(node_id)
            except Exception as ex:  # noqa: BLE001
                expr_results.append(f"{pad}[{node_id}] RAISES -> {ex}")
                return
            label = type(inner).__name__
            # surface a function-name hint when present (Gather/Take live here)
            fdata = getattr(inner, "function_data", None) or getattr(
                inner, "function", None
            )
            extra = f" function={fdata!r}" if fdata is not None else ""
            expr_results.append(f"{pad}[{node_id}] {label} OK{extra}")
            for cid in _child_node_ids(inner):
                walk_expr(cid, depth + 1)

        def walk(node_id: int) -> None:
            nt.set_node(node_id)
            node = nt.view_current_node()
            seen_nodes.append(type(node).__name__)
            for attr in ("expr", "exprs", "predicate", "left_on", "right_on"):
                exprs = getattr(node, attr, None)
                if exprs is None:
                    continue
                if not isinstance(exprs, (list, tuple)):
                    exprs = [exprs]
                for e in exprs:
                    nid = getattr(e, "node", None)
                    if isinstance(nid, int):
                        expr_results.append(f"  ({attr}) root:")
                        walk_expr(nid, 2)
            for inp in nt.get_inputs():
                walk(inp)

        try:
            walk(nt.get_node())
        except Exception as ex:  # noqa: BLE001
            report["walk_error"] = f"{ex}\n{traceback.format_exc()}"
        report["nodes"] = seen_nodes
        report["exprs"] = expr_results

    return cb


def run_shape(name: str, lf: pl.LazyFrame) -> None:
    report: dict[str, Any] = {}
    try:
        lf.collect(engine="cpu", post_opt_callback=_probe_callback(report))
    except Exception as ex:  # noqa: BLE001
        report["collect_error"] = str(ex)
    print(f"\n===== {name} =====")
    if "collect_error" in report:
        print(f"  COLLECT ERROR: {report['collect_error']}")
    if "walk_error" in report:
        print(f"  WALK ERROR: {report['walk_error']}")
    print(f"  IR nodes: {report.get('nodes')}")
    for line in report.get("exprs", []):
        print(f"    expr {line}")


def main() -> None:
    n = 10_000
    dim_n = 2_000
    rng = np.random.default_rng(0)
    fact = pl.DataFrame(
        {
            "id": rng.integers(0, dim_n, size=n).astype(np.int64),
            "value": rng.uniform(50, 150, size=n).astype(np.float32),
        }
    )
    dim = pl.DataFrame(
        {
            "id": np.arange(dim_n, dtype=np.int64),
            "vol": rng.uniform(0.1, 0.5, size=dim_n).astype(np.float32),
        }
    )
    dim_vol = pl.Series("vol", rng.uniform(0.1, 0.5, size=dim_n).astype(np.float32))

    # Shape A: explicit gather expr — gather a dim column by an id column, then chain.
    shape_a = fact.lazy().select(
        (
            pl.lit(dim_vol).gather(pl.col("id")).tanh() * pl.col("value")
        ).alias("out")
    )

    # Shape B: join then with_columns compute chain (the natural "fact->dim lookup").
    shape_b = (
        fact.lazy()
        .join(dim.lazy(), on="id", how="left")
        .with_columns((pl.col("vol").tanh() * pl.col("value")).alias("out"))
    )

    # Shape C: gather on a column already in the frame (self-gather by index).
    shape_c = fact.lazy().select(
        pl.col("value").gather(pl.col("id") % n).sqrt().alias("out")
    )

    # Shape D: Expr.take alias (older name) if present.
    try:
        shape_d = fact.lazy().select(
            pl.col("value").take(pl.col("id") % n).alias("out")  # type: ignore[attr-defined]
        )
    except Exception:  # noqa: BLE001
        shape_d = None

    run_shape("A: lit(dim).gather(id) -> tanh*value", shape_a)
    run_shape("B: join(dim) -> with_columns(tanh*value)", shape_b)
    run_shape("C: col.gather(id%n) -> sqrt", shape_c)
    if shape_d is not None:
        run_shape("D: col.take(id%n)", shape_d)


if __name__ == "__main__":
    main()
