"""M7 C1: plan-level differential properties — collect(engine=metal) matches CPU.

Plan surface only (scan / filter / predicate / projection / fused F32 chains).
Kernel-level numerics are covered Rust-side by proptest (Tasks 1-2); this slice
exists for the irreducibly-Python engine-plugin + routing surface.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

import polars_metal
from tests.diff.strategies import (
    m1_null_density_dataframe,
    null_heavy_frame,
    numeric_frame,
)

_ENG = polars_metal.MetalEngine()


@given(numeric_frame())
@settings(max_examples=100, deadline=None)
def test_numeric_collect_matches_cpu(lf) -> None:  # type: ignore[no-untyped-def]
    assert lf.collect(engine=_ENG).equals(lf.collect())


@given(null_heavy_frame())
@settings(max_examples=100, deadline=None)
def test_null_heavy_collect_matches_cpu(lf) -> None:  # type: ignore[no-untyped-def]
    assert lf.collect(engine=_ENG).equals(lf.collect())


@given(m1_null_density_dataframe())
@settings(max_examples=100, deadline=None)
def test_m1_frame_scan_matches_cpu(df) -> None:  # type: ignore[no-untyped-def]
    # Strategy returns a DataFrame (predates the LazyFrame-returning strategies), so wrap it.
    lf = df.lazy()
    assert lf.collect(engine=_ENG).equals(lf.collect())


# --- M7 extension: random F32 fused compute chains vs CPU --------------------


# A "safe" op set whose F32 output is finite for bounded inputs, so engine and
# CPU agree to tolerance (no NaN/Inf divergence). Restricted to
# magnitude-non-expanding ops (no square/exp/log/sqrt/div): chained application
# stays finite AND keeps transcendental arguments small enough that CPU-vs-Metal
# range reduction agrees. The multiplicative square/mul ops are covered
# separately by the Rust proptest_subgraph net (Task 1), so excluding them here
# loses no coverage of the kernel.
def _apply(expr: pl.Expr, op: str) -> pl.Expr:
    if op == "neg":
        return -expr
    if op == "abs":
        return expr.abs()
    if op == "sin":
        return expr.sin()
    if op == "cos":
        return expr.cos()
    if op == "tanh":
        return expr.tanh()
    raise ValueError(f"unknown op {op!r}")


_OPS = ("neg", "abs", "sin", "cos", "tanh")


@given(
    n=st.integers(min_value=1, max_value=2000),
    ops=st.lists(st.sampled_from(_OPS), min_size=1, max_size=5),
    reducer=st.sampled_from((None, "sum", "mean", "std", "var")),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=60, deadline=None)
def test_fused_f32_chain_matches_cpu(n, ops, reducer, seed) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(n) * 3.0).astype(np.float32)
    df = pl.DataFrame({"x": x}, schema={"x": pl.Float32})

    expr = pl.col("x")
    for op in ops:
        expr = _apply(expr, op)
    if reducer is not None:
        if reducer in ("std", "var") and n < 2:
            return  # ddof=1 undefined for n<2; CPU and engine both skip
        expr = getattr(expr, reducer)()

    lf = df.lazy().select(r=expr)
    assert_frame_equal(
        lf.collect(engine=_ENG),
        lf.collect(),
        check_exact=False,
        rel_tol=1e-3,
        abs_tol=1e-4,
    )
