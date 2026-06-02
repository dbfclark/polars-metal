"""M4 Phase 7: cum_prod / cum_max / cum_min route to MLX like cum_sum (Task 28).

These are viewable Function nodes whose MLX kernels (CumProd/CumMax/CumMin) are
already wired in fusion/subgraph.rs; only the IR analyzer needs to recognize
them. Forward only (the MLX bindings are forward-only, like cum_sum); reverse
and null-bearing columns fall back to CPU.
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


def _df(n=4096):
    rng = np.random.default_rng(0xCABB)
    # bounded so cum_prod doesn't overflow F32
    return pl.DataFrame({"x": (rng.uniform(0.95, 1.05, n)).astype(np.float32)})


def test_cum_prod_uses_mlx_and_matches():
    eng = polars_metal.MetalEngine()
    lf = _df().lazy().with_columns(r=pl.col("x").cum_prod())
    assert _dispatches(lf, eng) == 1
    assert_frame_equal(lf.collect(engine=eng), lf.collect(), check_exact=False, rel_tol=1e-3)


def test_cum_max_matches():
    eng = polars_metal.MetalEngine()
    lf = _df().lazy().with_columns(r=pl.col("x").cum_max())
    assert _dispatches(lf, eng) == 1
    assert_frame_equal(lf.collect(engine=eng), lf.collect(), check_exact=False, rel_tol=1e-4)


def test_cum_min_matches():
    eng = polars_metal.MetalEngine()
    lf = _df().lazy().with_columns(r=pl.col("x").cum_min())
    assert _dispatches(lf, eng) == 1
    assert_frame_equal(lf.collect(engine=eng), lf.collect(), check_exact=False, rel_tol=1e-4)


def test_cum_prod_reverse_falls_back():
    eng = polars_metal.MetalEngine()
    lf = _df().lazy().with_columns(r=pl.col("x").cum_prod(reverse=True))
    assert _dispatches(lf, eng) == 0, "reverse scan is forward-only on MLX; CPU"
    assert_frame_equal(lf.collect(engine=eng), lf.collect(), check_exact=False, rel_tol=1e-3)


def test_cum_max_null_falls_back():
    eng = polars_metal.MetalEngine()
    df = pl.DataFrame({"x": pl.Series([1.0, None, 3.0, 2.0, 5.0], dtype=pl.Float32)})
    lf = df.lazy().with_columns(r=pl.col("x").cum_max())
    assert _dispatches(lf, eng) == 0, "scan null propagation not the AND rule; CPU"
    assert_frame_equal(lf.collect(engine=eng), lf.collect())
