import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from polars_metal import MetalEngine, _udf

# Phase 1 (cosine_topk) is a `.metal` verb with no CPU implementation, so it
# always runs on the metal engine -- even for the all-"CPU" reference, whose CPU
# half is the explode + metadata-join rerank (phases 2-3). Only `engine` (the
# rerank engine) varies between the resident-GPU run and the CPU baseline.
_TOPK_ENGINE = MetalEngine()


def _retrieval(queries, corpus, metadata, k, engine):
    hits = (
        queries.lazy()
        .with_columns(hit=pl.col("emb").metal.cosine_topk(corpus, k))
        .collect(engine=_TOPK_ENGINE)
    )
    fact = (
        hits.lazy()
        .with_columns(
            idx=pl.col("hit").struct.field("indices"), sc=pl.col("hit").struct.field("scores")
        )
        .explode(["idx", "sc"])
        .with_columns(idx=pl.col("idx").cast(pl.Int64))
        .collect()
    )  # eager -> fact is a Scan
    return (
        fact.lazy()
        .join(metadata.lazy(), left_on="idx", right_on="id", how="left")
        .with_columns(rr=(pl.col("sc") * pl.col("price").exp() * pl.col("rating").log()))
        .collect(engine=engine)
    )


def test_retrieval_pipeline_resident_matches_cpu():
    rng = np.random.default_rng(3)
    Q, N, D, k = 4000, 5000, 64, 10
    corpus = pl.DataFrame(
        {"emb": [list(map(float, r)) for r in rng.standard_normal((N, D)).astype(np.float32)]},
        schema={"emb": pl.Array(pl.Float32, D)},
    )
    queries = pl.DataFrame(
        {"emb": [list(map(float, r)) for r in rng.standard_normal((Q, D)).astype(np.float32)]},
        schema={"emb": pl.Array(pl.Float32, D)},
    )
    metadata = pl.DataFrame(
        {
            "id": rng.permutation(N).astype(np.int64),
            "price": rng.uniform(0.1, 2.0, N).astype(np.float32),
            "rating": rng.uniform(1, 5, N).astype(np.float32),
        }
    )
    _udf._M10_DENSE_GATHERS = 0
    gpu = _retrieval(queries, corpus, metadata, k, MetalEngine(force_fusion=True))
    assert _udf._M10_DENSE_GATHERS == 1, "metadata gather did not run resident"
    cpu = _retrieval(queries, corpus, metadata, k, "cpu")
    assert_frame_equal(
        cpu.sort(["idx", "rr"]),
        gpu.sort(["idx", "rr"]),
        check_dtypes=True,
        rel_tol=1e-3,
        abs_tol=1e-3,
    )
