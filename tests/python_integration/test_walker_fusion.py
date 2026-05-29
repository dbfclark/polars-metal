"""M4 Phase 3 Task 17: walker integration of the fusion analyzer.

Verifies that when the walker processes an HStack/with_columns node, it
probes the analyzer and emits a diagnostic log line capturing the
FusedExprGraph candidate decision.

Phase 5 will replace the diagnostic log with actual MetalPlanNode::
FusedExprGraph emission; until then the log line is the contract.
"""

from __future__ import annotations

import logging

import polars as pl

import polars_metal


def test_walker_logs_fused_expr_for_transcendental_chain(caplog):
    """A 4-op transcendental chain on F32 columns should produce an analyzer
    decision log line citing the candidate column and op count."""
    n = 100
    df = pl.DataFrame(
        {
            "a": pl.Series([float(i) * 0.01 for i in range(n)], dtype=pl.Float32),
            "b": pl.Series([float(i) * 0.02 for i in range(n)], dtype=pl.Float32),
        }
    )
    engine = polars_metal.MetalEngine(debug=True)
    expr = (pl.col("a").sin() * pl.col("b").cos()).sqrt()

    with caplog.at_level(logging.INFO, logger="polars_metal.fusion"):
        df.lazy().with_columns(y=expr).collect(engine=engine)

    msgs = [r.getMessage() for r in caplog.records]
    fused = [m for m in msgs if "FusedExprGraph candidate" in m]
    assert fused, f"expected a FusedExprGraph log entry; got: {msgs!r}"
    # 4 compute ops: Sin, Cos, Mul, Sqrt.
    assert any("n_ops=4" in m for m in fused), f"expected n_ops=4; got: {fused!r}"
    assert any("column='y'" in m for m in fused)


def test_walker_rejects_string_expression(caplog):
    """A string-typed expression should not produce a FusedExprGraph candidate
    log (analyzer rejects it)."""
    df = pl.DataFrame({"s": pl.Series(["alpha", "beta", "gamma"], dtype=pl.Utf8)})
    engine = polars_metal.MetalEngine(debug=True)

    with caplog.at_level(logging.DEBUG, logger="polars_metal.fusion"):
        df.lazy().with_columns(y=pl.col("s").str.len_chars()).collect(engine=engine)

    msgs = [r.getMessage() for r in caplog.records]
    candidate = [m for m in msgs if "FusedExprGraph candidate" in m]
    rejected = [m for m in msgs if "analyzer rejected" in m]
    assert not candidate, "string expr should not produce a candidate log"
    assert rejected, f"expected analyzer-rejected log; got: {msgs!r}"
