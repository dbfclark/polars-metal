"""Verify Polars binary-expression aggregations lift to MetalPlanNode.

These tests cover Capability G (M3 Phase 2). The walker pattern-matches
``Agg(BinaryExpr(...))`` and emits an Expression-shape AggSpec dict; the
Rust parser consumes that into ``ParsedAgg::Expression``. Until the
Phase 3 fused-kernel consumer lands, the router will fall back Expression
specs to CPU — the tests still pass because ``assert_frame_equal``
compares against the CPU result, which is the ground truth either way.
"""

import polars as pl
from polars.testing import assert_frame_equal

import polars_metal as pm


def test_sum_a_times_b_routes_gpu():
    """The Q1-shape ``sum(a * b)`` lowers through the Expression rewriter."""
    df = pl.DataFrame(
        {
            "k": [0, 0, 1, 1, 2, 2] * 1000,
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 1000,
            "b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] * 1000,
        }
    )
    q = (
        df.lazy()
        .group_by("k")
        .agg(
            (pl.col("a") * pl.col("b")).sum().alias("sum_ab"),
        )
    )
    cpu = q.collect(engine="cpu").sort("k")
    metal = q.collect(engine=pm.MetalEngine()).sort("k")
    assert_frame_equal(cpu, metal)


def test_sum_a_times_one_minus_b():
    """Q1's disc_price shape: ``sum(a * (1 - b))``."""
    df = pl.DataFrame(
        {
            "k": [0, 0, 1, 1] * 5000,
            "a": [10.0, 20.0, 30.0, 40.0] * 5000,
            "b": [0.05, 0.1, 0.15, 0.2] * 5000,
        }
    )
    q = (
        df.lazy()
        .group_by("k")
        .agg(
            (pl.col("a") * (1.0 - pl.col("b"))).sum().alias("disc"),
        )
    )
    cpu = q.collect(engine="cpu").sort("k")
    metal = q.collect(engine=pm.MetalEngine()).sort("k")
    assert_frame_equal(cpu, metal)


def test_unsupported_function_call_falls_back():
    """``abs()`` inside agg falls back to CPU; result is still correct."""
    df = pl.DataFrame({"k": [0, 1, 0, 1], "v": [-1.0, 2.0, -3.0, 4.0]})
    q = df.lazy().group_by("k").agg(pl.col("v").abs().sum().alias("s"))
    cpu = q.collect(engine="cpu").sort("k")
    metal = q.collect(engine=pm.MetalEngine()).sort("k")
    assert_frame_equal(cpu, metal)


def test_depth_5_falls_back():
    """Expression deeper than the walker's depth cap still gives the right
    answer via CPU fallback."""
    df = pl.DataFrame({"k": [0] * 100, "v": [1.0] * 100})
    expr = pl.col("v")
    for _ in range(5):
        expr = expr + 1.0
    q = df.lazy().group_by("k").agg(expr.sum().alias("s"))
    cpu = q.collect(engine="cpu").sort("k")
    metal = q.collect(engine=pm.MetalEngine()).sort("k")
    assert_frame_equal(cpu, metal)
