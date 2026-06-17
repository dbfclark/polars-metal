"""M11 Task 2.1: .metal namespace verbs must be detected under LazyFrame.select,
not only LazyFrame.with_columns. The capture monkey-patch now also wraps
`.select`, and the slow-path fragment parser also recognizes the Select node's
'"expr":[' key. corr is a frame-level verb (lf.metal.corr()), not an expr in a
projection, so it is intentionally not covered here."""

from __future__ import annotations

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from polars_metal import MetalEngine


def test_rolling_under_select():
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"x": rng.standard_normal(50_000).astype(np.float32)})
    lf = df.lazy().select(pl.col("x").rolling_mean(window_size=100).alias("rm"))
    assert_frame_equal(
        lf.collect(),
        lf.collect(engine=MetalEngine()),
        check_dtypes=True,
        rel_tol=1e-3,
        abs_tol=1e-3,
    )


def test_dt_under_select():
    d = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), interval="1d", eager=True)
    df = pl.DataFrame({"d": d})
    lf = df.lazy().select(pl.col("d").dt.year().alias("yr"))
    assert_frame_equal(lf.collect(), lf.collect(engine=MetalEngine()), check_dtypes=True)


def test_cosine_topk_under_select():
    rng = np.random.default_rng(1)
    N, D, Q, k = 2000, 64, 30, 5
    corpus = pl.DataFrame(
        {"emb": [list(map(float, r)) for r in rng.standard_normal((N, D)).astype(np.float32)]},
        schema={"emb": pl.Array(pl.Float32, D)},
    )
    q = pl.DataFrame(
        {"emb": [list(map(float, r)) for r in rng.standard_normal((Q, D)).astype(np.float32)]},
        schema={"emb": pl.Array(pl.Float32, D)},
    )
    res = (
        q.lazy()
        .select(pl.col("emb").metal.cosine_topk(corpus, k).alias("hit"))
        .collect(engine=MetalEngine())
    )
    assert res.columns == ["hit"]
    assert res["hit"].dtype == pl.Struct(
        {"indices": pl.List(pl.UInt32), "scores": pl.List(pl.Float32)}
    )
    assert res.height == Q


def test_knn_under_select():
    rng = np.random.default_rng(2)
    N, D, Q, k = 2000, 64, 30, 5
    corpus = pl.DataFrame(
        {"emb": [list(map(float, r)) for r in rng.standard_normal((N, D)).astype(np.float32)]},
        schema={"emb": pl.Array(pl.Float32, D)},
    )
    q = pl.DataFrame(
        {"emb": [list(map(float, r)) for r in rng.standard_normal((Q, D)).astype(np.float32)]},
        schema={"emb": pl.Array(pl.Float32, D)},
    )
    res = (
        q.lazy()
        .select(pl.col("emb").metal.knn(corpus, k).alias("hit"))
        .collect(engine=MetalEngine())
    )
    assert res.columns == ["hit"]
    assert res.height == Q


def test_fft_under_select():
    rng = np.random.default_rng(3)
    sig = rng.standard_normal(64).astype(np.float32)
    df = pl.DataFrame({"sig": sig}, schema={"sig": pl.Float32})
    out = df.lazy().select(pl.col("sig").metal.fft().alias("spec")).collect(engine=MetalEngine())
    spec = out.get_column("spec")
    got_re = np.asarray(spec.struct.field("real").to_numpy(), dtype=np.float32)
    got_im = np.asarray(spec.struct.field("imag").to_numpy(), dtype=np.float32)
    exp = np.fft.fft(sig.astype(np.float32))
    assert out.columns == ["spec"]
    assert np.allclose(got_re, exp.real, rtol=1e-3, atol=1e-3)
    assert np.allclose(got_im, exp.imag, rtol=1e-3, atol=1e-3)


def test_dtw_under_select():
    rng = np.random.default_rng(4)
    N, L = 40, 64
    r = rng.standard_normal(L).astype(np.float32)
    Q = rng.standard_normal((N, L)).astype(np.float32)
    df = pl.DataFrame({"seq": [list(row) for row in Q]}, schema={"seq": pl.Array(pl.Float32, L)})
    eng = MetalEngine()
    got = df.lazy().select(pl.col("seq").metal.dtw(r).alias("d")).collect(engine=eng)
    assert got.columns == ["d"]
    assert got.height == N
    # Reference: same verb via with_columns must agree.
    ref = (
        df.lazy()
        .with_columns(pl.col("seq").metal.dtw(r).alias("d"))
        .collect(engine=eng)
        .select("d")
    )
    np.testing.assert_allclose(
        got.get_column("d").to_numpy(), ref.get_column("d").to_numpy(), atol=1e-3, rtol=1e-3
    )
