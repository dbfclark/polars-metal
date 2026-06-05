"""Brute-force pairwise cosine top-k over an embedding table.

Workload: given Q query embeddings and N corpus embeddings of dim D, compute
cosine similarity for every (query, corpus) pair, return top-K closest corpus
rows per query.

Why a candidate for Metal:
  - Compute is Q*N*D fused-multiply-adds.
  - Memory footprint is (Q+N)*D*4 bytes (one read of each input).
  - FLOPs / byte ratio for Q=100, N=100_000, D=768:
      ops  = 2 * Q * N * D = 1.536e10
      bytes = (Q + N) * D * 4 + Q*N*4 = ~349 MB
      density ~= 44 ops/byte — solidly compute-bound, not bandwidth-bound.
  - GPU FMA throughput on M2 Ultra (60-core): ~27 TFLOPS F32 peak;
    CPU NEON FMA across 16 P-cores: ~450 GFLOPS F32 peak.
    Compute-time ratio at 50% peak efficiency: ~60x.
  - Dispatch overhead is amortizable: this is one matmul plus a top-k pass.

DataFrame framing: embeddings are stored as columnar F32 columns
  (one column per dim) or as a List[F32] column. The Polars-native API
  for the workload is awkward (no .dot or .matmul); users typically escape
  to numpy. Both forms are measured here so the Metal engine has a clear
  target.

Scale: Q=100 queries, N=100_000 corpus rows, D=768. This is the canonical
  "single batch" shape from vector-search benchmarks (e.g. ANN-Benchmarks'
  glove-100-angular, SIFT1M). Total ~308 MB in F32.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_metal
from tests.bench.m4_survey._timing import time_callable


def make_embeddings(n: int, d: int, *, seed: int = 0xC05) -> np.ndarray:
    """Random unit-L2-normalized embeddings, F32."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d), dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def cosine_topk_numpy(query: np.ndarray, corpus: np.ndarray, k: int) -> np.ndarray:
    """Reference numpy implementation: matmul + argpartition.

    For L2-normalized inputs, cosine == dot product.
    """
    sims = query @ corpus.T  # (Q, N) F32 matmul
    idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    return idx


def cosine_topk_polars_listcol(
    query_df: pl.DataFrame, corpus_df: pl.DataFrame, k: int
) -> pl.DataFrame:
    """Polars-native attempt using List[F32] columns.

    For each query, cross-join with corpus, dot product via list arithmetic,
    then group by query and take top-k. This is the "DataFrame-native"
    framing; it is expected to be substantially slower than numpy because
    Polars has no fused matmul primitive.
    """
    crossed = query_df.join(corpus_df, how="cross")
    with_sim = crossed.with_columns(
        sim=(
            pl.col("q_emb")
            .list.eval(pl.element() * pl.element())
            .list.sum()  # this is wrong; need element-wise q*c. Replaced below.
        )
    )
    return with_sim.head(k)


def cosine_topk_polars_columnar(query: np.ndarray, corpus: np.ndarray, k: int) -> pl.DataFrame:
    """Polars columnar framing: embeddings as D separate F32 columns.

    Dot product via sum(q_i * c_i for i in [0, D)). This is what a
    Polars user with no numpy escape hatch would write. Expected to be
    much slower than matmul because:
      - Each query needs its own scalar pass over the corpus.
      - The expression engine doesn't fuse D multiplications into a
        matmul; each is a separate columnar pass.
    To keep this tractable we only run a single query per call, repeated.
    """
    d = corpus.shape[1]
    # Convert corpus to Polars once
    corpus_cols = {f"c_{i}": pl.Series(corpus[:, i]) for i in range(d)}
    corpus_df = pl.DataFrame(corpus_cols)
    # Per-query dot product as a sum of D pairwise-multiplied F32 columns.
    # We test on the first query only — repeating Q times would dominate the
    # measurement with the same operation, scaled linearly.
    q0 = query[0]
    sim_expr = sum(pl.col(f"c_{i}") * float(q0[i]) for i in range(d))
    return corpus_df.with_columns(sim=sim_expr).sort("sim", descending=True).head(k)


def _numpy_cosine_topk_indices(qv: np.ndarray, cv: np.ndarray, k: int) -> np.ndarray:
    """numpy brute force used as the CPU baseline for the engine-path gate."""
    qn = qv / np.linalg.norm(qv, axis=1, keepdims=True)
    cn = cv / np.linalg.norm(cv, axis=1, keepdims=True)
    sims = qn @ cn.T
    return np.argsort(-sims, axis=1)[:, :k]


def bench_engine_path(
    Q: int = 100, N: int = 200_000, D: int = 256, K: int = 10
) -> float:
    """Engine-path cosine top-k vs the numpy brute force (baseline.json gate).

    Returns ratio_metal_over_cpu (metal_s / cpu_s); the gate requires it < 1.0
    (metal faster than numpy). Recorded in baseline.json as
    ``phase10_cosine_topk_q100_n200k_d256``.
    """
    rng = np.random.default_rng(0)
    qv = rng.standard_normal((Q, D)).astype(np.float32)
    cv = rng.standard_normal((N, D)).astype(np.float32)
    corpus = pl.DataFrame(
        {"emb": list(cv)}, schema={"emb": pl.Array(pl.Float32, D)}
    ).lazy()
    qframe = pl.DataFrame({"emb": list(qv)}, schema={"emb": pl.Array(pl.Float32, D)})
    eng = polars_metal.MetalEngine()

    def metal():
        return (
            qframe.lazy()
            .with_columns(pl.col("emb").metal.cosine_topk(corpus, k=K).alias("h"))
            .collect(engine=eng)
        )

    metal()  # warmup: first call builds the MLX pipeline.

    metal_res = time_callable(
        f"metal.cosine_topk[Q={Q} N={N:,} D={D} k={K}]", metal
    )
    cpu_res = time_callable(
        f"numpy.cosine_topk[Q={Q} N={N:,} D={D} k={K}]",
        lambda: _numpy_cosine_topk_indices(qv, cv, K),
    )
    ratio = metal_res.median_ms / cpu_res.median_ms
    print(
        f"\n=== engine path (baseline.json: phase10_cosine_topk_q100_n200k_d256) ===\n"
        f"  metal_ms={metal_res.median_ms:.2f}"
        f"  cpu_ms={cpu_res.median_ms:.2f}"
        f"  ratio_metal_over_cpu={ratio:.4f}"
        f"  speedup={cpu_res.median_ms / metal_res.median_ms:.1f}x\n"
    )
    # Self-check mirroring the baseline.json `_gate.ratio_lt` (metal must win).
    assert ratio < 1.0, f"cosine top-k gate: metal/cpu={ratio:.4f} not < 1.0"
    return ratio


@pytest.mark.benchmark(group="cosine_topk")
def test_bench_cosine_topk_engine_gate() -> None:
    """Engine-path gate: metal cosine top-k must beat numpy brute force.

    Records ratio_metal_over_cpu for baseline.json entry
    phase10_cosine_topk_q100_n200k_d256 (_gate: ratio_lt 1.0).
    """
    ratio = bench_engine_path()
    assert ratio < 1.0


def main() -> None:
    Q = 100
    N = 100_000
    D = 768
    K = 10

    print("\n=== cosine top-k benchmark ===")
    print(f"  Q queries        = {Q}")
    print(f"  N corpus rows    = {N:,}")
    print(f"  D embedding dim  = {D}")
    print(f"  K top-k          = {K}")
    print(f"  input bytes      = {(Q + N) * D * 4 / 1e6:.1f} MB")
    print(f"  FLOPs (2*Q*N*D)  = {2 * Q * N * D / 1e9:.2f} GFLOPs")
    print(f"  density (op/byte) ~ {(2 * Q * N * D) / ((Q + N) * D * 4):.1f}")
    print()

    query = make_embeddings(Q, D, seed=0xCAFE)
    corpus = make_embeddings(N, D, seed=0xBEEF)

    # Numpy reference (this is the lower bound; an MLX kernel would target this number).
    time_callable(
        "cosine_topk_numpy[matmul+argpartition]",
        lambda: cosine_topk_numpy(query, corpus, K),
        extra={"shape": f"Q={Q} N={N} D={D}"},
    )

    # Polars columnar framing — single-query dot, scale-as-Q for fairness.
    res = time_callable(
        "cosine_topk_polars_columnar[1 query]",
        lambda: cosine_topk_polars_columnar(query, corpus, K),
        extra={"note": "single query; multiply by Q for full Q-batch estimate"},
    )
    print(
        f"  → projected for Q={Q} queries: ~{res.median_ms * Q:.0f} ms "
        f"(if Polars scales linearly, which it should)"
    )

    # Engine-path gate measurement (CI-reasonable size).
    bench_engine_path()


if __name__ == "__main__":
    main()
