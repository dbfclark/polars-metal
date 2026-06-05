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


def test_detect_finds_sentinel_binding():
    from polars_metal import _vector_detect as vdet

    df = pl.DataFrame(
        {"id": [0, 1], "emb": [[1.0, 0.0], [0.0, 1.0]]},
        schema={"id": pl.Int64, "emb": pl.Array(pl.Float32, 2)},
    )
    corpus = df.lazy()
    lf = df.lazy().with_columns(
        pl.col("emb").metal.cosine_topk(corpus, k=1).alias("hits")
    )
    bindings = vdet.find_vector_bindings(lf)
    assert len(bindings) == 1
    b = bindings[0]
    assert b.out_name == "hits"
    assert b.query_col == "emb"
    assert b.handle in vns._CORPUS_CACHE  # not yet popped


def test_dispatch_builds_struct_column_cosine():
    from polars_metal import _vector_detect as vdet
    from polars_metal import _vector_dispatch as vdisp

    corpus = pl.DataFrame(
        {"emb": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
        schema={"emb": pl.Array(pl.Float32, 2)},
    ).lazy()
    qframe = pl.DataFrame(
        {"id": [0], "emb": [[1.0, 0.0]]},
        schema={"id": pl.Int64, "emb": pl.Array(pl.Float32, 2)},
    )
    lf = qframe.lazy().with_columns(
        pl.col("emb").metal.cosine_topk(corpus, k=2).alias("hits")
    )
    bindings = vdet.find_vector_bindings(lf)
    df = vdisp.apply_vector_search(lf, bindings, collect_fn=lambda rest: rest.collect())
    assert df.columns == ["id", "emb", "hits"]
    hits = df.get_column("hits")
    assert hits.dtype == pl.Struct(
        {"indices": pl.List(pl.UInt32), "scores": pl.List(pl.Float32)}
    )
    row = hits[0]
    assert list(row["indices"]) == [0, 2]  # cosine: e0=1.0 then e2=0.707, desc
    assert abs(row["scores"][0] - 1.0) < 1e-5


def test_end_to_end_cosine_topk_via_engine():
    from polars_metal import MetalEngine

    corpus = pl.DataFrame(
        {"emb": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.9, 0.1]]},
        schema={"emb": pl.Array(pl.Float32, 2)},
    ).lazy()
    qframe = pl.DataFrame(
        {"id": [10, 20], "emb": [[1.0, 0.0], [0.0, 1.0]]},
        schema={"id": pl.Int64, "emb": pl.Array(pl.Float32, 2)},
    )
    out = (
        qframe.lazy()
        .with_columns(pl.col("emb").metal.cosine_topk(corpus, k=2).alias("hits"))
        .collect(engine=MetalEngine())
    )

    assert out.columns == ["id", "emb", "hits"]
    assert out.get_column("hits").dtype == pl.Struct(
        {"indices": pl.List(pl.UInt32), "scores": pl.List(pl.Float32)}
    )
    # query 0 = [1,0]: nearest is corpus[0]; query 1 = [0,1]: nearest is corpus[1].
    assert out["hits"][0]["indices"][0] == 0
    assert out["hits"][1]["indices"][0] == 1
