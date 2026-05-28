"""GPU ceiling for cosine top-k via MLX matmul on M2 Ultra.

This is NOT a polars-metal engine benchmark. MLX is the practical upper
bound for what a custom Metal kernel could achieve on this workload —
it's a hand-tuned matmul + reduction pipeline that already exists. If
MLX wins, an engine that routes the Polars expression tree to MLX (or
to a custom kernel matching MLX's throughput) plausibly wins too.

The numpy reference is on CPU via Accelerate (AMX). The MLX reference
is on GPU via Metal Performance Shaders / hand-tuned MSL.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from tests.bench.m4_survey._timing import time_callable


def make_embeddings(n: int, d: int, *, seed: int = 0xC05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d), dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def cosine_topk_mlx(q_mx: mx.array, c_mx: mx.array, k: int) -> mx.array:
    sims = q_mx @ c_mx.T
    idx = mx.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    mx.eval(idx)
    return idx


def main() -> None:
    Q = 100
    N = 100_000
    D = 768
    K = 10

    print(f"\n=== MLX cosine top-k (GPU ceiling) ===")
    print(f"  Q={Q} N={N:,} D={D} K={K}")
    print(f"  FLOPs (2*Q*N*D) = {2 * Q * N * D / 1e9:.2f} GFLOPs")
    print()

    query = make_embeddings(Q, D, seed=0xCAFE)
    corpus = make_embeddings(N, D, seed=0xBEEF)

    q_mx = mx.array(query)
    c_mx = mx.array(corpus)
    mx.eval(q_mx, c_mx)  # ensure transfer/materialization is out of timing

    time_callable(
        "cosine_topk_mlx_matmul",
        lambda: cosine_topk_mlx(q_mx, c_mx, K),
        extra={"backend": "MLX GPU"},
    )

    # Larger scale: N=1M
    N2 = 1_000_000
    corpus2 = make_embeddings(N2, D, seed=0xBEEF)
    c2_mx = mx.array(corpus2)
    mx.eval(c2_mx)
    print(f"\n  larger: Q={Q} N={N2:,} D={D}")
    time_callable(
        "cosine_topk_mlx_matmul[N=1M]",
        lambda: cosine_topk_mlx(q_mx, c2_mx, K),
        extra={"backend": "MLX GPU"},
    )

    # numpy equivalent on N=1M for reference
    def numpy_1m():
        sims = query @ corpus2.T
        idx = np.argpartition(-sims, kth=K - 1, axis=1)[:, :K]
        return idx

    time_callable(
        "cosine_topk_numpy[N=1M]",
        numpy_1m,
        extra={"backend": "numpy/Accelerate"},
    )


if __name__ == "__main__":
    main()
