"""Spike: where is the column-count crossover for GPU correlation matrix?

corr matrix of an (N x p) F32 matrix = standardize columns (subtract mean, /std)
-> Z (NxP), then C = Z^T Z / (N-1) -> pxp. That's a GEMM: compute ~ p^2 * N,
ingest ~ N * p, so compute/ingest ~ p. Small p (e.g. a single pair, p=2) is
bandwidth-bound (B4 loser); large p is compute-bound (the survey's ~7.8x GEMM
win). This finds the crossover so routing can switch GPU/CPU on column count.

Baseline = Polars CPU df.corr() (the engine's real alternative + the oracle).
GPU = MLX standardize+GEMM+eval+readback (proxy for the engine path, like the
A4/B4 spikes). Honest: the engine pays the same MLX dispatch; a custom kernel
could do better but this is the conservative floor.
"""

import time

import mlx.core as mx
import numpy as np
import polars as pl


def med(fn, it):
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def gpu_corr(Xmx, n):
    mu = mx.mean(Xmx, axis=0, keepdims=True)
    xc = Xmx - mu
    # population std with ddof=1 to match Pearson normalization in the GEMM
    var = mx.sum(xc * xc, axis=0, keepdims=True) / (n - 1)
    std = mx.sqrt(var)
    z = xc / std
    c = mx.matmul(z.T, z) / (n - 1)
    mx.eval(c)
    return c


def main():
    rng = np.random.default_rng(0xC0)
    print(f"{'N':>9} {'p':>5} {'cpu_ms':>9} {'gpu_ms':>9} {'speedup':>8}  {'win':>4}")
    print("-" * 52)
    for n in (10_000, 100_000, 1_000_000):
        for p in (2, 5, 10, 25, 50, 100, 200):
            X = rng.standard_normal((n, p)).astype(np.float32)
            df = pl.DataFrame(X, schema=[f"c{i}" for i in range(p)])
            Xmx = mx.array(X)
            mx.eval(Xmx)
            # warmup
            df.corr()
            gpu_corr(Xmx, n)
            it = 7 if n <= 100_000 else 4
            cpu = med(lambda df=df: df.corr(), it)
            gpu = med(lambda Xmx=Xmx, n=n: gpu_corr(Xmx, n), it)
            sp = cpu / gpu
            print(
                f"{n:>9,} {p:>5} {cpu * 1e3:>9.3f} {gpu * 1e3:>9.3f} {sp:>7.2f}x  "
                f"{'GPU' if sp > 1 else 'cpu':>4}"
            )


if __name__ == "__main__":
    main()
