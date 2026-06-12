"""Characterization tests pinning the .metal namespace verb contracts on
three axes: null handling, boundary error type, and streaming. These lock
CURRENT behavior before the M7-A consolidation refactor. Intentional
divergences (corr CPU-fallback vs vector raise; rolling silent-CPU vs vector
raise-on-streaming) are pinned here as deliberate, not bugs.

See docs/metal-namespace-contracts.md for the documented contract.
"""

from __future__ import annotations

import datetime

import polars as pl
import pytest

import polars_metal  # noqa: F401  (registers the .metal namespace + collect patch)
from polars_metal import MetalEngine

ComputeError = pl.exceptions.ComputeError


# ---------- null handling ----------


def test_vector_nulls_raise():
    corpus = pl.DataFrame({"emb": [[1.0, 0.0], [0.0, 1.0]]}).select(
        pl.col("emb").cast(pl.Array(pl.Float32, 2))
    )
    q = pl.LazyFrame({"emb": [[1.0, 0.0], None]}).select(
        pl.col("emb").cast(pl.Array(pl.Float32, 2))
    )
    with pytest.raises((ValueError, ComputeError)):
        # alias is required for detection; without it the sentinel falls through
        # to the CPU _raise_cpu guard (RuntimeError). The contract being pinned is
        # "null query rows raise" in the dispatch path — alias ensures detection fires.
        q.with_columns(
            pl.col("emb").metal.cosine_topk(corpus, k=1, corpus_col="emb").alias("hits")
        ).collect(engine=MetalEngine())


def test_fft_nulls_raise():
    lf = pl.LazyFrame({"x": [1.0, 2.0, None, 4.0]}).select(pl.col("x").cast(pl.Float32))
    with pytest.raises((ValueError, ComputeError)):
        lf.with_columns(pl.col("x").metal.fft().alias("f")).collect(engine=MetalEngine())


def test_corr_nulls_fall_back_to_cpu():
    # corr tolerates nulls by routing to CPU — no raise, finite result.
    lf = pl.LazyFrame({"a": [1.0, 2.0, 3.0, None], "b": [2.0, 4.0, 6.0, 8.0]})
    out = lf.metal.corr().collect(engine=MetalEngine())
    assert out.shape == (2, 2)


def test_rolling_nulls_fall_back_to_cpu():
    lf = pl.LazyFrame({"x": [1.0, None, 3.0, 4.0, 5.0]}).select(pl.col("x").cast(pl.Float32))
    out = lf.with_columns(pl.col("x").rolling_mean(window_size=2).alias("r")).collect(
        engine=MetalEngine()
    )
    cpu = lf.with_columns(pl.col("x").rolling_mean(window_size=2).alias("r")).collect(engine="cpu")
    assert out.equals(cpu)


def test_dt_nulls_restored():
    dates = [datetime.date(2021, 1, 1), None]
    # explicit Series keeps the Date dtype unambiguous (Polars infers Date from date objects)
    lf = pl.DataFrame({"d": pl.Series("d", dates)}).lazy()
    out = lf.with_columns(pl.col("d").dt.year().alias("y")).collect(engine=MetalEngine())
    cpu = lf.with_columns(pl.col("d").dt.year().alias("y")).collect(engine="cpu")
    assert out.equals(cpu)


def test_dtw_nulls_restored():
    import numpy as np

    rng = np.random.default_rng(2)
    r = rng.standard_normal(8).astype(np.float32)
    rows = [
        list(rng.standard_normal(8).astype(np.float32)),
        None,
        list(rng.standard_normal(8).astype(np.float32)),
    ]
    df = pl.DataFrame({"seq": rows}, schema={"seq": pl.Array(pl.Float32, 8)})
    out = (
        df.lazy()
        .with_columns(pl.col("seq").metal.dtw(r).alias("dist"))
        .collect(engine=MetalEngine())
    )
    d = out.get_column("dist")
    # null row produces null output, non-null rows produce finite distances
    assert d.is_null().to_list() == [False, True, False]
    assert d[0] is not None and d[2] is not None


# ---------- streaming ----------


def test_vector_streaming_raises():
    corpus = pl.DataFrame({"emb": [[1.0, 0.0]]}).select(pl.col("emb").cast(pl.Array(pl.Float32, 2)))
    lf = pl.LazyFrame({"emb": [[1.0, 0.0]]}).select(pl.col("emb").cast(pl.Array(pl.Float32, 2)))
    with pytest.raises(ComputeError):
        lf.with_columns(pl.col("emb").metal.cosine_topk(corpus, k=1, corpus_col="emb")).collect(
            engine=MetalEngine(), streaming=True
        )


def test_fft_streaming_raises():
    lf = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0]}).select(pl.col("x").cast(pl.Float32))
    with pytest.raises(ComputeError):
        lf.with_columns(pl.col("x").metal.fft().alias("f")).collect(
            engine=MetalEngine(), streaming=True
        )


def test_corr_streaming_raises():
    lf = pl.LazyFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    with pytest.raises(ComputeError):
        lf.metal.corr().collect(engine=MetalEngine(), streaming=True)


def test_dtw_streaming_raises():
    import numpy as np

    r = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    lf = pl.LazyFrame({"seq": [[1.0, 2.0, 3.0, 4.0]]}).select(
        pl.col("seq").cast(pl.Array(pl.Float32, 4))
    )
    with pytest.raises(ComputeError):
        lf.with_columns(pl.col("seq").metal.dtw(r).alias("d")).collect(
            engine=MetalEngine(), streaming=True
        )


def test_rolling_streaming_silent_cpu_fallback():
    # rolling HAS a CPU implementation, so streaming silently runs on CPU
    # (no raise). This divergence from vector/fft/dtw/corr is intentional.
    lf = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}).select(pl.col("x").cast(pl.Float32))
    out = lf.with_columns(pl.col("x").rolling_mean(window_size=2).alias("r")).collect(
        engine=MetalEngine(), streaming=True
    )
    cpu = lf.with_columns(pl.col("x").rolling_mean(window_size=2).alias("r")).collect(engine="cpu")
    assert out.equals(cpu)


def test_dt_streaming_silent_cpu_fallback():
    dates = [datetime.date(2021, 1, 1), datetime.date(2022, 6, 15)]
    lf = pl.DataFrame({"d": pl.Series("d", dates)}).lazy()
    out = lf.with_columns(pl.col("d").dt.year().alias("y")).collect(
        engine=MetalEngine(), streaming=True
    )
    cpu = lf.with_columns(pl.col("d").dt.year().alias("y")).collect(engine="cpu")
    assert out.equals(cpu)


# ---------- fft has no capture cache (repeated collect must work) ----------


def test_fft_repeated_collect_no_eviction():
    # fft inlines its op code in the sentinel literal (no handle, no cache),
    # so repeated collects of the same lf must both succeed. Pins the
    # "fft has no handle-evicted guard, by design" contract.
    lf = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0]}).select(pl.col("x").cast(pl.Float32))
    built = lf.with_columns(pl.col("x").metal.fft().alias("f"))
    first = built.collect(engine=MetalEngine())
    second = built.collect(engine=MetalEngine())
    assert first.equals(second)


# ---------- handle-missing -> ComputeError (A1 correction) ----------


def test_vector_handle_missing_raises_compute_error():
    """Simulate the race where the captured corpus spec was GC'd/evicted before dispatch.
    After the A1 correction, this must raise ComputeError (not RuntimeError)."""
    import polars_metal._vector_namespace as ns

    corpus = pl.DataFrame({"emb": [[1.0, 0.0]]}).select(pl.col("emb").cast(pl.Array(pl.Float32, 2)))
    lf = pl.LazyFrame({"emb": [[1.0, 0.0]]}).select(pl.col("emb").cast(pl.Array(pl.Float32, 2)))
    built = lf.with_columns(
        pl.col("emb").metal.cosine_topk(corpus, k=1, corpus_col="emb").alias("hits")
    )
    # Evict all cached corpus specs to trigger the handle-missing path.
    # Reaches into _CACHE._specs (CaptureCache); updated from _CORPUS_CACHE in M7 A-2.
    for h in list(ns._CACHE._specs.keys()):
        ns.evict_capture(h)
    with pytest.raises(ComputeError):
        built.collect(engine=MetalEngine())


def test_corr_handle_missing_raises_compute_error():
    """Simulate the race where the captured corr spec was GC'd/evicted before dispatch.
    After the A1 correction, this must raise ComputeError (not RuntimeError)."""
    import polars_metal._corr_namespace as ns

    lf = pl.LazyFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    built = lf.metal.corr()
    # Evict all cached corr specs to trigger the handle-missing path.
    # Reaches into the current _CORR_CACHE global; update if cache is refactored.
    for h in list(ns._CORR_CACHE.keys()):
        ns.evict_capture(h)
    with pytest.raises(ComputeError):
        built.collect(engine=MetalEngine())
