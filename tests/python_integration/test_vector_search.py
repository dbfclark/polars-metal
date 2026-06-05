"""M6 vector search: corpus capture + sentinel + .metal namespace tests."""

from __future__ import annotations

import numpy as np
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


# ─────────────────────────────────────────────────────────────────────────────
# Task 15 — differential tests vs a numpy oracle (the real correctness bar).
# ─────────────────────────────────────────────────────────────────────────────


def _oracle_cosine_topk(q: np.ndarray, c: np.ndarray, k: int):
    """Reference cosine top-k: similarity desc, tie-break (-score, index)."""
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    cn = c / np.linalg.norm(c, axis=1, keepdims=True)
    sims = qn @ cn.T  # (Q, N)
    out_idx, out_sc = [], []
    for row in sims:
        order = sorted(range(len(row)), key=lambda j: (-row[j], j))[:k]
        out_idx.append(order)
        out_sc.append([float(row[j]) for j in order])
    return out_idx, out_sc


def _oracle_knn(q: np.ndarray, c: np.ndarray, k: int):
    """Reference knn: TRUE L2 distance asc, tie-break (dist, index)."""
    d = np.sqrt(((q[:, None, :] - c[None, :, :]) ** 2).sum(-1))  # (Q, N) true L2
    out_idx, out_sc = [], []
    for row in d:
        order = sorted(range(len(row)), key=lambda j: (row[j], j))[:k]
        out_idx.append(order)
        out_sc.append([float(row[j]) for j in order])
    return out_idx, out_sc


@pytest.mark.parametrize("metric", ["cosine", "knn"])
@pytest.mark.parametrize(
    "Q,N,D,k",
    [
        (1, 5, 3, 2),
        (4, 50, 8, 5),
        (3, 17, 4, 17),  # k == N
        (2, 8, 1, 12),  # k > N (clamps), D == 1
        (5, 200, 16, 10),
    ],
)
def test_matches_numpy_oracle(metric, Q, N, D, k):
    from polars_metal import MetalEngine

    # Fixed seed; the +0.1 offset keeps norms away from zero. Random F32 in
    # general position makes exact ties (which would make index order ambiguous)
    # vanishingly unlikely, so exact index match is expected.
    rng = np.random.default_rng(0)
    qv = rng.standard_normal((Q, D)).astype(np.float32) + 0.1
    cv = rng.standard_normal((N, D)).astype(np.float32) + 0.1
    corpus = pl.DataFrame(
        {"emb": list(cv)}, schema={"emb": pl.Array(pl.Float32, D)}
    ).lazy()
    qframe = pl.DataFrame({"emb": list(qv)}, schema={"emb": pl.Array(pl.Float32, D)})
    verb = "cosine_topk" if metric == "cosine" else "knn"
    out = (
        qframe.lazy()
        .with_columns(getattr(pl.col("emb").metal, verb)(corpus, k=k).alias("hits"))
        .collect(engine=MetalEngine())
    )

    oracle = _oracle_cosine_topk if metric == "cosine" else _oracle_knn
    oi, osc = oracle(qv, cv, min(k, N))
    for qi in range(Q):
        got_idx = list(out["hits"][qi]["indices"])
        got_sc = list(out["hits"][qi]["scores"])
        # Clamp: a query against N rows can return at most N neighbours.
        assert len(got_idx) == min(k, N)
        assert got_idx == oi[qi], f"q{qi} idx {got_idx} != oracle {oi[qi]}"
        # F32 GEMM vs F64 numpy oracle: ~1e-4 tolerance.
        np.testing.assert_allclose(got_sc, osc[qi], rtol=1e-4, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Task 16 — mismatch / raise tests + k>N clamp + empty-corpus semantics.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["dtype", "dmismatch", "ragged"])
def test_raises_on_bad_inputs(bad):
    from polars_metal import MetalEngine

    if bad == "dtype":  # F64 instead of F32
        corpus = pl.DataFrame(
            {"emb": [[1.0, 0.0]]}, schema={"emb": pl.Array(pl.Float64, 2)}
        ).lazy()
        qframe = pl.DataFrame(
            {"emb": [[1.0, 0.0]]}, schema={"emb": pl.Array(pl.Float64, 2)}
        )
    elif bad == "dmismatch":  # query D != corpus D
        corpus = pl.DataFrame(
            {"emb": [[1.0, 0.0, 0.0]]}, schema={"emb": pl.Array(pl.Float32, 3)}
        ).lazy()
        qframe = pl.DataFrame(
            {"emb": [[1.0, 0.0]]}, schema={"emb": pl.Array(pl.Float32, 2)}
        )
    else:  # ragged List, not fixed-width Array
        corpus = pl.DataFrame({"emb": [[1.0, 0.0], [1.0]]}).lazy()
        qframe = pl.DataFrame({"emb": [[1.0, 0.0]]})
    # All three rejections surface as ValueError from the dispatch validators
    # (Array(F32, D) requirement / D-mismatch check) before any FFI call.
    with pytest.raises(ValueError):
        qframe.lazy().with_columns(
            pl.col("emb").metal.cosine_topk(corpus, k=1).alias("hits")
        ).collect(engine=MetalEngine())


def test_k_greater_than_n_clamps():
    from polars_metal import MetalEngine

    corpus = pl.DataFrame(
        {"emb": [[1.0, 0.0], [0.0, 1.0]]}, schema={"emb": pl.Array(pl.Float32, 2)}
    ).lazy()
    qframe = pl.DataFrame(
        {"emb": [[1.0, 0.0]]}, schema={"emb": pl.Array(pl.Float32, 2)}
    )
    out = (
        qframe.lazy()
        .with_columns(pl.col("emb").metal.cosine_topk(corpus, k=10).alias("hits"))
        .collect(engine=MetalEngine())
    )
    assert len(out["hits"][0]["indices"]) == 2  # clamped to N


def test_empty_corpus_returns_empty_hits():
    """An empty corpus (N=0) has zero neighbours: each query gets empty lists,
    NOT a low-level MLX allocation error. Dtype stays Struct{List[u32], List[f32]}."""
    from polars_metal import MetalEngine

    corpus = pl.DataFrame(
        {"emb": []}, schema={"emb": pl.Array(pl.Float32, 2)}
    ).lazy()
    qframe = pl.DataFrame(
        {"id": [0, 1], "emb": [[1.0, 0.0], [0.0, 1.0]]},
        schema={"id": pl.Int64, "emb": pl.Array(pl.Float32, 2)},
    )
    out = (
        qframe.lazy()
        .with_columns(pl.col("emb").metal.cosine_topk(corpus, k=5).alias("hits"))
        .collect(engine=MetalEngine())
    )
    assert out.get_column("hits").dtype == pl.Struct(
        {"indices": pl.List(pl.UInt32), "scores": pl.List(pl.Float32)}
    )
    assert out.height == 2
    for qi in range(2):
        assert len(out["hits"][qi]["indices"]) == 0
        assert len(out["hits"][qi]["scores"]) == 0
