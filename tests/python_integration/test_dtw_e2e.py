"""Engine-level DTW differential vs dtaidistance (M6 A4)."""

import numpy as np
import polars as pl
import pytest
from dtaidistance import dtw as _dtw

import polars_metal
from polars_metal import _native


def _ref_distances(Q, r, window=None):
    # our window w <-> dtaidistance window=w+1 (equal length); None => unconstrained
    kw = {} if window is None else {"window": window + 1}
    return np.array(
        [_dtw.distance(Q[i].astype(np.float64), r.astype(np.float64), **kw) for i in range(len(Q))],
        dtype=np.float64,
    )


def _frame(Q):
    L = Q.shape[1]
    return pl.DataFrame({"seq": [list(row) for row in Q]}, schema={"seq": pl.Array(pl.Float32, L)})


@pytest.mark.parametrize("L", [16, 64])
@pytest.mark.parametrize("window", [None, 0, 2, 8])
def test_dtw_matches_dtaidistance(L, window):
    eng = polars_metal.MetalEngine()
    rng = np.random.default_rng(0xD7 + L + (window or 0))
    N = 40
    r = rng.standard_normal(L).astype(np.float32)
    Q = rng.standard_normal((N, L)).astype(np.float32)
    lf = _frame(Q).lazy().with_columns(pl.col("seq").metal.dtw(r, window=window).alias("d"))
    got = lf.collect(engine=eng).get_column("d").to_numpy()
    exp = _ref_distances(Q, r, window)
    np.testing.assert_allclose(got, exp, atol=1e-3, rtol=1e-3)


def test_dtw_uses_gpu_path():
    """Prove the GPU dispatch actually fires (B2 lesson: assert dispatch, not just equals-oracle)."""
    eng = polars_metal.MetalEngine()
    rng = np.random.default_rng(1)
    r = rng.standard_normal(16).astype(np.float32)
    Q = rng.standard_normal((10, 16)).astype(np.float32)
    n = {"c": 0}
    orig = _native.execute_dtw

    def cnt(*a, **k):
        n["c"] += 1
        return orig(*a, **k)

    _native.execute_dtw = cnt
    try:
        _frame(Q).lazy().with_columns(pl.col("seq").metal.dtw(r).alias("d")).collect(engine=eng)
    finally:
        _native.execute_dtw = orig
    assert n["c"] == 1


def test_dtw_identical_is_zero():
    eng = polars_metal.MetalEngine()
    Q = np.tile(np.arange(8, dtype=np.float32), (3, 1))
    r = np.arange(8, dtype=np.float32)
    got = _frame(Q).lazy().with_columns(pl.col("seq").metal.dtw(r).alias("d")).collect(engine=eng)
    assert np.allclose(got.get_column("d").to_numpy(), 0.0, atol=1e-3)


def test_dtw_nulls_restored_positionally():
    eng = polars_metal.MetalEngine()
    rng = np.random.default_rng(2)
    r = rng.standard_normal(8).astype(np.float32)
    rows = [
        list(rng.standard_normal(8).astype(np.float32)),
        None,
        list(rng.standard_normal(8).astype(np.float32)),
    ]
    df = pl.DataFrame({"seq": rows}, schema={"seq": pl.Array(pl.Float32, 8)})
    got = df.lazy().with_columns(pl.col("seq").metal.dtw(r).alias("d")).collect(engine=eng)
    d = got.get_column("d")
    assert d[1] is None
    assert d[0] is not None and d[2] is not None


def test_dtw_nan_in_non_null_row_raises():
    eng = polars_metal.MetalEngine()
    r = np.arange(8, dtype=np.float32)
    rows = [list(np.arange(8, dtype=np.float32))]
    rows[0][3] = float("nan")  # NaN cell in a non-null row
    df = pl.DataFrame({"seq": rows}, schema={"seq": pl.Array(pl.Float32, 8)})
    with pytest.raises(ValueError, match="NaN"):
        df.lazy().with_columns(pl.col("seq").metal.dtw(r).alias("d")).collect(engine=eng)
