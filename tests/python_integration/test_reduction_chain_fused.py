"""M4 Phase 7 (increment 2): compute-chain-terminated reductions fuse to MLX.

A reduction over a compute chain — `(x.log().exp()).sum()`, `(x*y).std()` — is
GPU-worthy regardless of the reduction op: the chain amortizes the dispatch
floor and runs the whole chain+reduce as one MLX dispatch. Null inputs fall
back to CPU (the chain's null-propagation + reduction null-skip can't be
replayed). Null-free F32 only.
"""

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

import polars_metal
from polars_metal import _native


def _dispatches(lf, eng):
    n = {"c": 0}
    orig = _native.execute_fused_expr

    def cnt(scope, inputs, out):
        n["c"] += 1
        return orig(scope=scope, inputs=inputs, out=out)

    _native.execute_fused_expr = cnt
    try:
        lf.collect(engine=eng)
    finally:
        _native.execute_fused_expr = orig
    return n["c"]


def _df(n=50_000):
    rng = np.random.default_rng(0xC4)
    return pl.DataFrame(
        {
            "x": (np.abs(rng.standard_normal(n)) + 0.1).astype(np.float32),
            "y": rng.standard_normal(n).astype(np.float32),
        }
    )


def test_transcendental_chain_sum_uses_gpu():
    eng = polars_metal.MetalEngine()
    lf = _df().lazy().select(pl.col("x").log().exp().sum().alias("r"))
    assert _dispatches(lf, eng) == 1, "compute-chain sum should fuse to MLX"
    assert_frame_equal(lf.collect(engine=eng), lf.collect(), check_exact=False, rel_tol=1e-3)


def test_arith_chain_min_uses_gpu():
    eng = polars_metal.MetalEngine()
    lf = _df().lazy().select((pl.col("x") * pl.col("y")).min().alias("r"))
    assert _dispatches(lf, eng) == 1, "compute-chain min should fuse to MLX"
    assert_frame_equal(lf.collect(engine=eng), lf.collect(), check_exact=False, abs_tol=1e-3)


def test_chain_std_uses_gpu():
    eng = polars_metal.MetalEngine()
    lf = _df().lazy().select((pl.col("x") * 2.0 + 1.0).std().alias("r"))
    assert _dispatches(lf, eng) == 1
    assert_frame_equal(lf.collect(engine=eng), lf.collect(), check_exact=False, rel_tol=1e-3)


def _null_df(n=5_000):
    rng = np.random.default_rng(0xC4)
    x = (np.abs(rng.standard_normal(n)) + 0.1).astype(np.float32)
    df = pl.DataFrame({"x": x})
    return df.with_columns(
        pl.when(pl.int_range(pl.len()) % 9 == 0).then(None).otherwise(pl.col("x")).alias("x")
    )


def test_null_elementwise_chain_uses_gpu():
    """An elementwise chain over a null column reduces on the GPU: Polars drops
    the null rows (native), the GPU reduces the dense survivors — positions
    don't matter for a reduction, so there's nothing to rejoin. One dispatch,
    matches CPU."""
    eng = polars_metal.MetalEngine()
    df = _null_df()
    assert df["x"].null_count() > 0
    lf = df.lazy().select(pl.col("x").log().exp().sum().alias("r"))
    assert _dispatches(lf, eng) == 1, "elementwise null chain should reduce on the GPU"
    assert_frame_equal(lf.collect(engine=eng), lf.collect(), check_exact=False, rel_tol=1e-3)


def test_null_chain_std_and_min_match_cpu():
    eng = polars_metal.MetalEngine()
    df = _null_df()
    for expr in ((pl.col("x") * 2.0 + 1.0).std(), (pl.col("x") * 3.0).min()):
        lf = df.lazy().select(expr.alias("r"))
        assert _dispatches(lf, eng) == 1
        assert_frame_equal(lf.collect(engine=eng), lf.collect(), check_exact=False, rel_tol=1e-3)


def test_null_multicol_chain_matches_cpu():
    """Drop rows where ANY input column is null (Polars elementwise null rule)."""
    eng = polars_metal.MetalEngine()
    rng = np.random.default_rng(7)
    a = (np.abs(rng.standard_normal(5_000)) + 0.1).astype(np.float32)
    b = rng.standard_normal(5_000).astype(np.float32)
    df = pl.DataFrame({"a": a, "b": b}).with_columns(
        pl.when(pl.int_range(pl.len()) % 5 == 0).then(None).otherwise(pl.col("a")).alias("a"),
        pl.when(pl.int_range(pl.len()) % 7 == 0).then(None).otherwise(pl.col("b")).alias("b"),
    )
    lf = df.lazy().select((pl.col("a") * pl.col("b")).sum().alias("r"))
    assert _dispatches(lf, eng) == 1
    assert_frame_equal(lf.collect(engine=eng), lf.collect(), check_exact=False, rel_tol=1e-3)


def test_null_where_chain_stays_on_cpu():
    """A `where` chain over a null column stays on CPU: a null cond keeps the
    else branch valid, so dropping the null row would be wrong."""
    eng = polars_metal.MetalEngine()
    lf = (
        _null_df()
        .lazy()
        .select(pl.when(pl.col("x") > 1.0).then(pl.col("x").sqrt()).otherwise(0.0).sum().alias("r"))
    )
    assert _dispatches(lf, eng) == 0, "where chain over a null column must fall back to CPU"
    assert_frame_equal(lf.collect(engine=eng), lf.collect(), check_exact=False, rel_tol=1e-3)


def test_null_chain_degenerate_n_matches_cpu():
    """Null chain reducing to 0 or 1 survivor: empty -> sum 0.0 / else null;
    <2 survivors -> std/var null."""
    eng = polars_metal.MetalEngine()
    for data in ([None, None], [None, 4.0]):
        df = pl.DataFrame({"x": pl.Series(data, dtype=pl.Float32)})
        lf = df.lazy().select(
            (pl.col("x") * 2.0).sum().alias("s"),
            (pl.col("x") * 2.0).std().alias("d"),
        )
        assert_frame_equal(lf.collect(engine=eng), lf.collect(), check_exact=False, rel_tol=1e-3)


def test_tiny_chain_matches_cpu():
    """Degenerate n on a null-free chain: empty / single-row reductions match
    Polars (sum=0.0 on empty; std/var of <2 rows are null)."""
    eng = polars_metal.MetalEngine()
    for data in ([], [2.0], [3.0, 4.0]):
        df = pl.DataFrame({"x": pl.Series(data, dtype=pl.Float32)})
        lf = df.lazy().select(
            (pl.col("x") * 2.0).sum().alias("s"),
            (pl.col("x") * 2.0).std().alias("d"),
        )
        assert_frame_equal(lf.collect(engine=eng), lf.collect(), check_exact=False, rel_tol=1e-3)
