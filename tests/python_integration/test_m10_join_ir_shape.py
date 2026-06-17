"""Characterization: the IR shape a `join -> F32 chain` produces at the
post-optimization NodeTraverser. Pins what `_walk_join` must handle."""

from __future__ import annotations

import numpy as np
import polars as pl


def _capture_ir(lf: pl.LazyFrame) -> dict:
    report: dict = {}

    def cb(nt, _d=None):
        def visit(nid, depth):
            nt.set_node(nid)
            node = nt.view_current_node()
            report.setdefault("nodes", []).append((depth, type(node).__name__))
            if type(node).__name__ == "Join":
                report["join_attrs"] = sorted(a for a in dir(node) if not a.startswith("_"))
                report["join_how"] = repr(getattr(node, "options", None))
                report["left_on"] = [
                    type(nt.view_expression(e.node)).__name__ for e in getattr(node, "left_on", [])
                ]
                report["right_on"] = [
                    type(nt.view_expression(e.node)).__name__ for e in getattr(node, "right_on", [])
                ]
            for inp in nt.get_inputs():
                visit(inp, depth + 1)

        visit(nt.get_node(), 0)

    lf.collect(engine="cpu", post_opt_callback=cb)
    return report


def test_join_then_chain_ir_shape():
    rng = np.random.default_rng(0)
    fact = pl.DataFrame(
        {
            "id": rng.integers(0, 500, 2000).astype(np.int64),
            "value": rng.uniform(50, 150, 2000).astype(np.float32),
        }
    )
    dim = pl.DataFrame(
        {
            "id": np.arange(500, dtype=np.int64),
            "vol": rng.uniform(0.1, 0.5, 500).astype(np.float32),
        }
    )
    lf = (
        fact.lazy()
        .join(dim.lazy(), on="id", how="left")
        .with_columns((pl.col("vol").tanh() * pl.col("value")).alias("out"))
    )
    rep = _capture_ir(lf)
    # The compute chain sits in an HStack above a Join above two scans.
    kinds = [k for _, k in rep["nodes"]]
    assert "Join" in kinds, rep
    assert "HStack" in kinds or "Select" in kinds, rep
    assert rep["left_on"] == ["Column"] and rep["right_on"] == ["Column"], rep
    print("M10 IR shape:", rep)  # captured for _walk_join authoring
