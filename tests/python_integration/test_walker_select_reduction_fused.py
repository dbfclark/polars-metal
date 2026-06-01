"""M4 Phase 7 (Task 26): full-column F32 reductions route through fused MLX.

`select(pl.col("x").std())` is an empty-key reduction. Before M4 it went to the
empty-key GroupBy conformance kernel; now a fusion-eligible reduction (sum/mean/
min/max/std/var of a bare Float32 column) routes through the fused MLX subgraph.
std/var carry a Bessel correction at the dispatch boundary because MLX uses
population variance (ddof=0) while Polars defaults to sample (ddof=1). Null
columns and degenerate (<2-row) inputs fall back to a Polars reduction on the
source column (MLX over `to_numpy()` would turn nulls into NaN).
"""

import math

import numpy as np
import polars as pl

import polars_metal
from polars_metal import _native


def _make_floats(n, seed=0xFA57):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"x": rng.standard_normal(n).astype(np.float32)})


def _count_dispatches(monkeypatch):
    count = {"n": 0}
    orig = _native.execute_fused_expr

    def counting(scope, inputs, out):
        count["n"] += 1
        return orig(scope=scope, inputs=inputs, out=out)

    monkeypatch.setattr(_native, "execute_fused_expr", counting)
    return count


def _check(monkeypatch, expr_fn, expected_dispatches, abs_tol=1e-2, rel_tol=1e-4):
    df = _make_floats(200_000)
    count = _count_dispatches(monkeypatch)
    cpu = df.lazy().select(expr_fn()).collect()
    metal = df.lazy().select(expr_fn()).collect(engine=polars_metal.MetalEngine())
    assert count["n"] == expected_dispatches, (
        f"expected {expected_dispatches} fused dispatch(es), got {count['n']}"
    )
    assert cpu.columns == metal.columns
    assert cpu.dtypes == metal.dtypes, f"dtype mismatch: {cpu.dtypes} vs {metal.dtypes}"
    for col in cpu.columns:
        a = float(cpu[col][0])
        b = float(metal[col][0])
        assert math.isclose(a, b, abs_tol=abs_tol, rel_tol=rel_tol), f"{col}: cpu={a} metal={b}"


def test_std_matches_cpu(monkeypatch):
    _check(monkeypatch, lambda: pl.col("x").std().alias("s"), expected_dispatches=1)


def test_var_matches_cpu(monkeypatch):
    _check(monkeypatch, lambda: pl.col("x").var().alias("v"), expected_dispatches=1)


def test_sum_matches_cpu(monkeypatch):
    _check(monkeypatch, lambda: pl.col("x").sum().alias("s"), expected_dispatches=1)


def test_mean_matches_cpu(monkeypatch):
    _check(monkeypatch, lambda: pl.col("x").mean().alias("m"), expected_dispatches=1)


def test_std_and_var_together(monkeypatch):
    # Two aggregations in one Select -> two fused scalar dispatches.
    _check(
        monkeypatch,
        lambda: [pl.col("x").std().alias("s"), pl.col("x").var().alias("v")],
        expected_dispatches=2,
    )


def test_compute_chain_reduction_falls_back(monkeypatch):
    # Compute-chain reductions ((x*2).std()) are NOT fused (a chain's null
    # propagation can't be replayed for the null fallback) — they stay on the
    # CPU/GroupBy path. Still correct, just not routed.
    _check(monkeypatch, lambda: ((pl.col("x") * 2.0) + 1.0).std().alias("s"), expected_dispatches=0)


def test_column_with_nulls_falls_back_to_polars(monkeypatch):
    # A null-containing F32 column reduces via Polars (null-skipping) rather
    # than the MLX path (which would turn nulls into NaN). The walker still
    # routes it (the analyzer can't see nulls), but the dispatch detects the
    # null and computes on the host — so zero MLX dispatches, exact result.
    from polars.testing import assert_frame_equal

    x = np.random.default_rng(1).standard_normal(10_000).astype(np.float32)
    df = pl.DataFrame({"x": x})
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) % 7 == 0).then(None).otherwise(pl.col("x")).alias("x")
    )
    assert df["x"].null_count() > 0

    count = _count_dispatches(monkeypatch)
    exprs = [
        pl.col("x").sum().alias("sum"),
        pl.col("x").std().alias("std"),
        pl.col("x").mean().alias("mean"),
    ]
    cpu = df.lazy().select(exprs).collect()
    metal = df.lazy().select(exprs).collect(engine=polars_metal.MetalEngine())
    assert count["n"] == 0, "null column must skip the MLX path"
    assert_frame_equal(cpu, metal, check_exact=False, abs_tol=1e-3, rel_tol=1e-5)


def _check_edges(df, exprs):
    cpu = df.lazy().select(exprs).collect()
    metal = df.lazy().select(exprs).collect(engine=polars_metal.MetalEngine())
    # Exact match including null position and dtype for the degenerate cases.
    from polars.testing import assert_frame_equal

    assert_frame_equal(cpu, metal)


def test_single_row_reductions_match_cpu():
    # n=1: sum/mean/min/max = the value; sample std/var are null.
    df = pl.DataFrame({"x": pl.Series([3.0], dtype=pl.Float32)})
    _check_edges(
        df,
        [
            pl.col("x").sum().alias("sum"),
            pl.col("x").mean().alias("mean"),
            pl.col("x").min().alias("min"),
            pl.col("x").max().alias("max"),
            pl.col("x").std().alias("std"),
            pl.col("x").var().alias("var"),
        ],
    )


def test_empty_frame_reductions_match_cpu():
    # n=0: sum is 0.0, every other reduction is null.
    df = pl.DataFrame({"x": pl.Series([], dtype=pl.Float32)})
    _check_edges(
        df,
        [
            pl.col("x").sum().alias("sum"),
            pl.col("x").mean().alias("mean"),
            pl.col("x").std().alias("std"),
            pl.col("x").var().alias("var"),
        ],
    )
