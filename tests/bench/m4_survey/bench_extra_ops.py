"""Extended candidate sweep: ops I dismissed too quickly in the first pass.

The first survey pass declared sort, statistical reductions, cumulative
ops, conditional chains, and binning "probably bandwidth-shaped" by
analogy to TPC-H — without actually measuring the MLX ceiling for each.
This script does the missing measurements.

Each block reports: Polars CPU time, NumPy time (where applicable),
MLX time, and the win ratio. Same harness as the other m4_survey
scripts.

The goal is to either confirm the dismissal or surface additional
existence proofs. Scale: N=10M for single-column work, smaller for
matrix-shaped work.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import polars as pl

from tests.bench.m4_survey._timing import time_callable


def report_block(name: str, n: int) -> None:
    print(f"\n=== {name} ===  N={n:,}")
    print()


def main() -> None:
    rng = np.random.default_rng(0xCAFE)
    N = 10_000_000

    # =============== SORT ===============
    report_block("sort F32 column", N)
    arr_f32 = rng.standard_normal(N).astype(np.float32)
    s_pl = pl.Series("x", arr_f32)
    a_mx = mx.array(arr_f32)
    mx.eval(a_mx)

    time_callable("polars.sort[F32]", lambda: s_pl.sort())
    time_callable("numpy.sort[F32]", lambda: np.sort(arr_f32))

    def mlx_sort():
        s = mx.sort(a_mx)
        mx.eval(s)
        return s

    time_callable("mlx.sort[F32]", mlx_sort)

    # =============== ARGSORT + TAKE (top-k via sort) ===============
    report_block("argsort top-k F32", N)
    K = 100
    time_callable(
        "polars.top_k[F32, k=100]",
        lambda: pl.DataFrame({"x": arr_f32}).top_k(K, by="x"),
    )
    time_callable(
        "numpy.argpartition_top_k[F32]",
        lambda: arr_f32[np.argpartition(-arr_f32, K)[:K]],
    )

    def mlx_topk():
        idx = mx.argpartition(-a_mx, kth=K - 1)[:K]
        mx.eval(idx)
        return idx

    time_callable("mlx.argpartition_top_k[F32]", mlx_topk)

    # =============== STATISTICAL REDUCTIONS ===============
    report_block("std / var / quantile global reductions", N)

    time_callable("polars.std[F32]", lambda: s_pl.std())
    time_callable("numpy.std[F32]", lambda: arr_f32.std())

    def mlx_std():
        v = mx.std(a_mx)
        mx.eval(v)
        return v

    time_callable("mlx.std[F32]", mlx_std)

    time_callable("polars.var[F32]", lambda: s_pl.var())

    def mlx_var():
        v = mx.var(a_mx)
        mx.eval(v)
        return v

    time_callable("mlx.var[F32]", mlx_var)

    time_callable(
        "polars.quantile[F32, p=0.5]",
        lambda: s_pl.quantile(0.5),
    )
    time_callable("numpy.quantile[F32, p=0.5]", lambda: np.quantile(arr_f32, 0.5))

    # MLX doesn't have quantile directly; via partial sort
    def mlx_quantile_p50():
        # midpoint of sorted; not interpolated, but close enough for timing
        s = mx.sort(a_mx)
        v = s[N // 2]
        mx.eval(v)
        return v

    time_callable("mlx.quantile_via_sort[F32, p=0.5]", mlx_quantile_p50)

    # =============== CUMULATIVE SUM ===============
    report_block("cumsum F32", N)
    time_callable("polars.cum_sum[F32]", lambda: s_pl.cum_sum())
    time_callable("numpy.cumsum[F32]", lambda: np.cumsum(arr_f32))

    def mlx_cumsum():
        c = mx.cumsum(a_mx)
        mx.eval(c)
        return c

    time_callable("mlx.cumsum[F32]", mlx_cumsum)

    # =============== EXP / LOG / POW CHAINS (Black-Scholes-like) ===============
    report_block("Black-Scholes-shaped: 3 columns -> 4 outputs", N)
    # Classic option-pricing kernel: lots of exp/log/sqrt/erf-like ops.
    spot = rng.uniform(50.0, 150.0, size=N).astype(np.float32)
    strike = rng.uniform(50.0, 150.0, size=N).astype(np.float32)
    time_to_exp = rng.uniform(0.1, 2.0, size=N).astype(np.float32)
    df = pl.DataFrame({"s": spot, "k": strike, "t": time_to_exp})

    sigma = 0.2
    r = 0.05

    def polars_bs() -> pl.DataFrame:
        # d1 = (log(s/k) + (r + sigma^2/2) * t) / (sigma * sqrt(t))
        # d2 = d1 - sigma * sqrt(t)
        # price ~ s * cdf(d1) - k * exp(-rt) * cdf(d2)   (cdf via erf approximation)
        # For benchmarking purposes we use a polynomial erf approximation
        # (Abramowitz & Stegun 7.1.26) since Polars doesn't have erf.
        s = pl.col("s")
        k = pl.col("k")
        t = pl.col("t")
        sigma_sqrt_t = float(sigma) * t.sqrt()
        d1 = ((s / k).log() + (float(r) + 0.5 * sigma * sigma) * t) / sigma_sqrt_t
        d2 = d1 - sigma_sqrt_t

        # erf approximation as polynomial in d/(d+1)
        def cdf(x: pl.Expr) -> pl.Expr:
            # Φ(x) ≈ 0.5 * (1 + tanh(x * 0.7978845608))  (a fast approx)
            return 0.5 * (1.0 + (x * 0.7978845608).tanh())

        call = s * cdf(d1) - k * (-float(r) * t).exp() * cdf(d2)
        return df.with_columns(call=call)

    time_callable("polars.black_scholes_call", polars_bs)

    # numpy reference
    def numpy_bs():
        s, k, t = spot, strike, time_to_exp
        sst = sigma * np.sqrt(t)
        d1 = (np.log(s / k) + (r + 0.5 * sigma * sigma) * t) / sst
        d2 = d1 - sst

        def cdf(x):
            return 0.5 * (1.0 + np.tanh(0.7978845608 * x))

        return s * cdf(d1) - k * np.exp(-r * t) * cdf(d2)

    time_callable("numpy.black_scholes_call", numpy_bs)

    s_mx = mx.array(spot)
    k_mx = mx.array(strike)
    t_mx = mx.array(time_to_exp)
    mx.eval(s_mx, k_mx, t_mx)

    def mlx_bs():
        sst = sigma * mx.sqrt(t_mx)
        d1 = (mx.log(s_mx / k_mx) + (r + 0.5 * sigma * sigma) * t_mx) / sst
        d2 = d1 - sst

        def cdf(x):
            return 0.5 * (1.0 + mx.tanh(0.7978845608 * x))

        out = s_mx * cdf(d1) - k_mx * mx.exp(-r * t_mx) * cdf(d2)
        mx.eval(out)
        return out

    time_callable("mlx.black_scholes_call", mlx_bs)

    # =============== HISTOGRAM / BINNING ===============
    report_block("histogram / value binning", N)
    BINS = 256
    time_callable(
        "polars.cut_into_bins[F32, 256]",
        lambda: s_pl.cut(np.linspace(-4, 4, BINS - 1).tolist()).value_counts(),
    )
    time_callable(
        "numpy.histogram[F32, 256 bins]",
        lambda: np.histogram(arr_f32, bins=BINS, range=(-4, 4)),
    )
    # MLX has no histogram; implement via floor + bincount substitute.
    # We approximate using clipped index + counted sum-per-bin via segment_sum,
    # but MLX doesn't have segment_sum either — so we skip the MLX leg here.
    print("  (mlx: no native histogram — would need custom MSL kernel)")

    # =============== CONDITIONAL CHAINS ===============
    report_block("conditional chain (5-tier when/then/otherwise)", N)
    cond = rng.standard_normal(N).astype(np.float32)

    def polars_when():
        x = pl.col("x")
        return pl.DataFrame({"x": cond}).with_columns(
            bucket=pl.when(x < -2.0)
            .then(0)
            .when(x < -1.0)
            .then(1)
            .when(x < 0.0)
            .then(2)
            .when(x < 1.0)
            .then(3)
            .when(x < 2.0)
            .then(4)
            .otherwise(5)
        )

    time_callable("polars.when_chain", polars_when)

    def numpy_when():
        return np.select(
            [cond < -2.0, cond < -1.0, cond < 0.0, cond < 1.0, cond < 2.0],
            [0, 1, 2, 3, 4],
            default=5,
        )

    time_callable("numpy.select_chain", numpy_when)

    c_mx = mx.array(cond)
    mx.eval(c_mx)

    def mlx_when():
        # cascade: count how many thresholds < x is below
        b = (
            (c_mx >= -2.0).astype(mx.int32)
            + (c_mx >= -1.0).astype(mx.int32)
            + (c_mx >= 0.0).astype(mx.int32)
            + (c_mx >= 1.0).astype(mx.int32)
            + (c_mx >= 2.0).astype(mx.int32)
        )
        mx.eval(b)
        return b

    time_callable("mlx.cascade_threshold", mlx_when)

    # =============== FFT (signal processing) ===============
    report_block("FFT 1D F32", N)
    fft_in = arr_f32[: 1 << 23]  # ~8M points, power of 2
    time_callable(
        "numpy.fft.fft[N=8M F32]",
        lambda: np.fft.fft(fft_in),
    )
    fft_mx = mx.array(fft_in)
    mx.eval(fft_mx)

    def mlx_fft():
        out = mx.fft.fft(fft_mx)
        mx.eval(out)
        return out

    time_callable("mlx.fft.fft[N=8M F32]", mlx_fft)

    # =============== CORRELATION MATRIX OVER K COLUMNS ===============
    report_block("correlation matrix K columns x N rows", 200_000)
    K_COLS = 200
    N_ROWS = 200_000
    mat = rng.standard_normal((N_ROWS, K_COLS)).astype(np.float32)
    df_corr = pl.from_numpy(mat, schema={f"c{i}": pl.Float32 for i in range(K_COLS)})

    time_callable(
        "polars.corr_matrix[200x200000]",
        lambda: df_corr.corr(),
    )
    time_callable("numpy.corrcoef[200x200000]", lambda: np.corrcoef(mat.T))

    mat_mx = mx.array(mat)
    mx.eval(mat_mx)

    def mlx_corr():
        # Standardize, then matmul
        mean = mat_mx.mean(axis=0, keepdims=True)
        centered = mat_mx - mean
        std = mx.sqrt((centered * centered).mean(axis=0, keepdims=True))
        normed = centered / std
        c = normed.T @ normed / float(N_ROWS)
        mx.eval(c)
        return c

    time_callable("mlx.corr_matmul[200x200000]", mlx_corr)


if __name__ == "__main__":
    main()
