import numpy as np
import polars as pl
import pytest

from polars_metal import MetalEngine


def _build(corpus, weight=None):
    _N, D = corpus.shape
    schema = {"emb": pl.Array(pl.Float32, D)}
    return pl.DataFrame({"emb": [list(map(float, r)) for r in corpus]}, schema=schema)


def test_cosine_topk_rerank_matches_numpy():
    rng = np.random.default_rng(30)
    N, D, Q, k = 2000, 64, 50, 10
    corpus = rng.standard_normal((N, D)).astype(np.float32)
    weight = rng.uniform(0, 1, N).astype(np.float32)
    queries = rng.standard_normal((Q, D)).astype(np.float32)
    qdf = pl.DataFrame({"emb": [list(map(float, r)) for r in queries]},
                       schema={"emb": pl.Array(pl.Float32, D)})
    cdf = _build(corpus)
    res = (qdf.lazy().with_columns(
        pl.col("emb").metal.cosine_topk(cdf, k, rerank_weight=pl.Series(weight), rerank="exp_decay").alias("hit"))
        .collect(engine=MetalEngine()))
    # numpy oracle: cosine sims -> top-k by sim -> rerank sim*exp(-w[hit]) -> sort desc by reranked
    qn = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    cn = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    sims = qn @ cn.T
    for qi in range(Q):
        top = np.argpartition(-sims[qi], k - 1)[:k]       # top-k by similarity
        rer = sims[qi, top] * np.exp(-weight[top])
        hit = res["hit"][qi]
        got_idx = [int(x) for x in hit["indices"]]
        got_scores = [float(x) for x in hit["scores"]]
        assert set(got_idx) == set(int(x) for x in top), f"q{qi} index set"
        # scores must be the reranked values, sorted desc
        np.testing.assert_allclose(sorted(got_scores, reverse=True),
                                   sorted(rer.tolist(), reverse=True), rtol=1e-3, atol=1e-3)
        # and the returned order must be by descending reranked score
        assert got_scores == sorted(got_scores, reverse=True), f"q{qi} order"


def test_rerank_weight_length_mismatch_raises():
    rng = np.random.default_rng(31)
    N, D = 100, 8
    corpus = rng.standard_normal((N, D)).astype(np.float32)
    cdf = pl.DataFrame({"emb": [list(map(float, r)) for r in corpus]},
                       schema={"emb": pl.Array(pl.Float32, D)})
    q = pl.DataFrame({"emb": [[0.0] * D]}, schema={"emb": pl.Array(pl.Float32, D)})
    with pytest.raises(ValueError, match="rerank_weight length"):
        (q.lazy().with_columns(
            pl.col("emb").metal.cosine_topk(cdf, 5, rerank_weight=pl.Series(np.ones(N - 1, np.float32)), rerank="exp_decay").alias("h"))
         .collect(engine=MetalEngine()))


def test_no_rerank_unchanged():
    # rerank=None (default) -> raw cosine sims, unchanged from before.
    rng = np.random.default_rng(32)
    N, D, k = 500, 16, 5
    corpus = rng.standard_normal((N, D)).astype(np.float32)
    cdf = pl.DataFrame({"emb": [list(map(float, r)) for r in corpus]},
                       schema={"emb": pl.Array(pl.Float32, D)})
    qv = rng.standard_normal(D).astype(np.float32)
    q = pl.DataFrame({"emb": [list(map(float, qv))]}, schema={"emb": pl.Array(pl.Float32, D)})
    res = (q.lazy().with_columns(pl.col("emb").metal.cosine_topk(cdf, k).alias("h")).collect(engine=MetalEngine()))
    qn = qv / np.linalg.norm(qv)
    cn = corpus / np.linalg.norm(corpus, axis=1, keepdims=True)
    sims = cn @ qn
    top = np.argpartition(-sims, k - 1)[:k]
    got = res["h"][0]
    assert set(int(x) for x in got["indices"]) == set(int(x) for x in top)
    np.testing.assert_allclose(sorted([float(x) for x in got["scores"]], reverse=True),
                               sorted(sims[top].tolist(), reverse=True), rtol=1e-3, atol=1e-3)
