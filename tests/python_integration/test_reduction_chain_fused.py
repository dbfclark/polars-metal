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


def test_chain_with_nulls_falls_back():
    eng = polars_metal.MetalEngine()
    df = pl.DataFrame({"x": pl.Series([1.0, None, 3.0, 4.0, 5.0], dtype=pl.Float32)})
    lf = df.lazy().select(pl.col("x").log().exp().sum().alias("r"))
    assert _dispatches(lf, eng) == 0, "chain over a null column must fall back to CPU"
    assert_frame_equal(lf.collect(engine=eng), lf.collect(), check_exact=False, rel_tol=1e-3)
