"""M6 vector search: corpus capture + sentinel + .metal namespace tests."""

from __future__ import annotations

import polars as pl
import pytest

from polars_metal import _vector_namespace as vns


def test_capture_assigns_unique_handles_and_stores_corpus():
    corpus = pl.DataFrame(
        {"emb": [[1.0, 0.0], [0.0, 1.0]]},
        schema={"emb": pl.Array(pl.Float32, 2)},
    ).lazy()
    h1 = vns._capture_corpus(corpus, "emb", k=5, metric="cosine")
    h2 = vns._capture_corpus(corpus, "emb", k=5, metric="cosine")
    assert h1 != h2
    spec = vns._peek_capture(h1)  # non-destructive peek for the test
    assert spec.corpus_col == "emb"
    assert spec.k == 5 and spec.metric == "cosine"


def test_namespace_methods_build_sentinel_structs():
    corpus = pl.DataFrame(
        {"emb": [[1.0, 0.0]]},
        schema={"emb": pl.Array(pl.Float32, 2)},
    ).lazy()
    e1 = pl.col("emb").metal.cosine_topk(corpus, k=3)
    e2 = pl.col("emb").metal.knn(corpus, k=3)
    # Both serialize to a struct ("as_struct") carrying our tag.
    assert vns.SENTINEL_TAG in e1.meta.serialize(format="json")
    assert vns.SENTINEL_TAG in e2.meta.serialize(format="json")


def test_sentinel_raises_on_plain_cpu():
    df = pl.DataFrame({"emb": [[1.0, 0.0]]}, schema={"emb": pl.Array(pl.Float32, 2)})
    corpus = df.lazy()
    expr = pl.col("emb").metal.cosine_topk(corpus, k=1)
    # The opaque map_batches sentinel field fires on a plain CPU collect (no
    # engine="metal") and raises our RuntimeError carrying the engine hint.
    with pytest.raises(RuntimeError, match="engine='metal'"):
        df.lazy().with_columns(expr.alias("hits")).collect()  # no engine="metal"
