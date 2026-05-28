"""Pairwise L2 distance / k-NN brute force on M2 Ultra.

A second flavor of the pairwise-similarity family. Where cosine assumes
unit-normalized vectors, L2 distance is the canonical metric for k-NN
classifiers, clustering, and many nearest-neighbor benchmarks.

The compute density here is even higher than cosine because for each
(query, corpus) pair we compute (q - c)^T (q - c) — 3 ops per dim
(subtract, multiply, accumulate). With Q=100 queries, N=100_000 corpus
rows, D=128 dims (SIFT1M-like):
  FLOPs = 3 * Q * N * D = 3.84e9
  bytes = (Q + N) * D * 4 = 51.3 MB
  density ~= 75 ops/byte — solidly compute-bound.

Same conclusion as cosine: MLX matmul-based formulation
  ||q - c||^2 = ||q||^2 + ||c||^2 - 2 q . c
is dispatched as a matmul and easily wins.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from tests.bench.m4_survey._timing import time_callable


def make_vectors(n: int, d: int, *, seed: int = 0xC05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d), dtype=np.float32)


def knn_l2_numpy(query: np.ndarray, corpus: np.ndarray, k: int) -> np.ndarray:
    q2 = (query * query).sum(axis=1, keepdims=True)  # (Q, 1)
    c2 = (corpus * corpus).sum(axis=1)  # (N,)
    dist2 = q2 + c2 - 2.0 * (query @ corpus.T)  # (Q, N)
    return np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]


def knn_l2_mlx(q_mx: mx.array, c_mx: mx.array, k: int) -> mx.array:
    q2 = (q_mx * q_mx).sum(axis=1, keepdims=True)
    c2 = (c_mx * c_mx).sum(axis=1)
    dist2 = q2 + c2 - 2.0 * (q_mx @ c_mx.T)
    idx = mx.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
    mx.eval(idx)
    return idx


def main() -> None:
    Q = 100
    N = 100_000
    D = 128
    K = 10

    print(f"\n=== pairwise L2 / k-NN brute force ===")
    print(f"  Q={Q} N={N:,} D={D} K={K}")
    print(f"  FLOPs (3*Q*N*D) = {3 * Q * N * D / 1e9:.2f} GFLOPs")
    print(f"  density (op/byte) ~ {3 * Q * N * D / ((Q + N) * D * 4):.1f}")
    print()

    query = make_vectors(Q, D, seed=0xCAFE)
    corpus = make_vectors(N, D, seed=0xBEEF)

    time_callable(
        "knn_l2_numpy",
        lambda: knn_l2_numpy(query, corpus, K),
    )

    q_mx, c_mx = mx.array(query), mx.array(corpus)
    mx.eval(q_mx, c_mx)
    time_callable(
        "knn_l2_mlx",
        lambda: knn_l2_mlx(q_mx, c_mx, K),
    )

    # SIFT1M scale: N=1M, D=128
    N2 = 1_000_000
    corpus2 = make_vectors(N2, D, seed=0xBEEF)
    c2_mx = mx.array(corpus2)
    mx.eval(c2_mx)
    print(f"\n  SIFT1M-scale: Q={Q} N={N2:,} D={D}")
    time_callable("knn_l2_numpy[N=1M]", lambda: knn_l2_numpy(query, corpus2, K))
    time_callable("knn_l2_mlx[N=1M]", lambda: knn_l2_mlx(q_mx, c2_mx, K))


if __name__ == "__main__":
    main()
