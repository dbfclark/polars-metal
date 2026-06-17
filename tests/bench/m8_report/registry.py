"""The op registry: pure data + callables, no timing/formatting logic.

Each BenchEntry carries up to three callables:
  - engine_fn: full engine="metal" wall-clock (ingest+compute+fold-back). REQUIRED.
  - cpu_fn:    the mission baseline -- Polars CPU where a native expr exists,
               else the idiomatic CPU tool (numpy/scipy/dtaidistance). REQUIRED.
  - ceiling_fn: raw MLX/numpy with no engine overhead. OPTIONAL (None where no
               meaningful raw form exists).
  - check:     optional correctness comparator (engine_out, cpu_out) -> None,
               raises on mismatch. None => default numeric-allclose on result.

Fixtures are imported from existing m4_* benches, not rebuilt.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

import mlx.core as mx
import numpy as np
import polars as pl

import polars_metal as pm
from tests.bench._canonical_q1_fixture_f32 import make_canonical_q1_fixture_f32
from tests.bench._q6_fixture_f32 import make_q6_fixture_f32
from tests.bench.m4_engine.bench_haversine_e2e import _haversine_expr, _make_taxi

_ENGINE = pm.MetalEngine()

# ---------------------------------------------------------------------------
# Ceiling memoization: host->mx transfer excluded from the timed region.
#
# measure() discards warmup; the first warmup call populates the cache so
# all measured calls reuse already-resident mx.arrays (pure GPU compute).
# The data object is held alive by the caller throughout measure(), so
# id(data) is stable.
# ---------------------------------------------------------------------------
_CEIL_CACHE: dict[int, Any] = {}


def _ceil_inputs(data: Any, builder: Callable[[Any], Any]) -> Any:
    """Return memoized mx.array(s) for *data*, building once on first call."""
    key = id(data)
    if key not in _CEIL_CACHE:
        arrs = builder(data)
        if isinstance(arrs, tuple):
            mx.eval(*arrs)
        else:
            mx.eval(arrs)
        _CEIL_CACHE[key] = arrs
    return _CEIL_CACHE[key]


@dataclass
class BenchEntry:
    name: str
    category: str
    sizes: list[int]
    make_input: Callable[[int], Any]
    engine_fn: Callable[[Any], Any]
    cpu_fn: Callable[[Any], Any]
    ceiling_fn: Callable[[Any], Any] | None = None
    check: Callable[[Any, Any], None] | None = None


# ---- helpers -------------------------------------------------------------


def _black_scholes_expr() -> pl.Expr:
    # F32 transcendental chain on a single price column.
    s = pl.col("s")
    k, r, t, vol = 100.0, 0.02, 1.0, 0.3
    d1 = ((s / k).log() + (r + 0.5 * vol * vol) * t) / (vol * (t**0.5))
    d2 = d1 - vol * (t**0.5)

    # crude normal-CDF proxy via tanh approx -- identical on both paths.
    def ncdf(x: pl.Expr) -> pl.Expr:
        return 0.5 * (1.0 + (x * 0.7978845608).tanh())

    # discount factor is a scalar constant -- compute in Python, not as a Polars expr.
    discount = math.exp(-r * t)
    return s * ncdf(d1) - k * discount * ncdf(d2)


# ---- haversine ceiling ----------------------------------------------------


def _ceil_haversine(df: pl.DataFrame) -> mx.array:
    """Raw MLX haversine: same formula as _haversine_expr, cached mx inputs."""
    R = 6371.0
    deg2rad = float(np.pi / 180.0)

    def _build(data: pl.DataFrame) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        return (
            mx.array(data["pickup_lat"].to_numpy()),
            mx.array(data["pickup_lon"].to_numpy()),
            mx.array(data["drop_lat"].to_numpy()),
            mx.array(data["drop_lon"].to_numpy()),
        )

    p_lat, p_lon, d_lat, d_lon = _ceil_inputs(df, _build)
    p_lat_r = p_lat * deg2rad
    d_lat_r = d_lat * deg2rad
    dlat = (d_lat_r - p_lat_r) / 2.0
    dlon = (d_lon - p_lon) * deg2rad / 2.0
    a = mx.sin(dlat) ** 2 + mx.cos(p_lat_r) * mx.cos(d_lat_r) * mx.sin(dlon) ** 2
    out = 2.0 * R * mx.arcsin(mx.sqrt(a))
    mx.eval(out)
    return out


# ---- black_scholes ceiling ------------------------------------------------

_BS_K = 100.0
_BS_R = 0.02
_BS_T = 1.0
_BS_VOL = 0.3
_BS_DISCOUNT = math.exp(-_BS_R * _BS_T)


def _ceil_black_scholes(df: pl.DataFrame) -> mx.array:
    """Raw MLX Black-Scholes: same tanh-approx formula as _black_scholes_expr."""

    def _build(data: pl.DataFrame) -> mx.array:
        return mx.array(data["s"].to_numpy())

    s = _ceil_inputs(df, _build)
    d1 = (mx.log(s / _BS_K) + (_BS_R + 0.5 * _BS_VOL * _BS_VOL) * _BS_T) / (_BS_VOL * (_BS_T**0.5))
    d2 = d1 - _BS_VOL * (_BS_T**0.5)

    def ncdf_mx(x: mx.array) -> mx.array:
        return 0.5 * (1.0 + mx.tanh(x * 0.7978845608))

    out = s * ncdf_mx(d1) - _BS_K * _BS_DISCOUNT * ncdf_mx(d2)
    mx.eval(out)
    return out


def _make_prices(n: int, seed: int = 0xB5) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"s": rng.uniform(50, 150, size=n).astype(np.float32)})


def _frame_allclose(
    engine_out: pl.DataFrame,
    cpu_out: pl.DataFrame,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> None:
    """Default check: every numeric column close between engine and CPU output."""
    assert engine_out.columns == cpu_out.columns, (engine_out.columns, cpu_out.columns)
    for col in engine_out.columns:
        a = engine_out[col].to_numpy()
        b = cpu_out[col].to_numpy()
        if np.issubdtype(a.dtype, np.number):
            np.testing.assert_allclose(
                a, b, rtol=rtol, atol=atol, equal_nan=True, err_msg=f"col {col}"
            )


def _make_signal_1col(n: int, seed: int = 0x501) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"x": rng.standard_normal(n).astype(np.float32)})


def _rolling_entry(stat: str, window: int) -> BenchEntry:
    expr = getattr(pl.col("x"), f"rolling_{stat}")(window_size=window)
    return BenchEntry(
        name=f"rolling_{stat}_w{window}",
        category="rolling",
        sizes=[1_000_000, 10_000_000],
        make_input=_make_signal_1col,
        engine_fn=lambda df, e=expr: df.lazy().with_columns(r=e).collect(engine=_ENGINE),
        cpu_fn=lambda df, e=expr: df.lazy().with_columns(r=e).collect(),
        ceiling_fn=None,
        check=_frame_allclose,
    )


# ---- registry ------------------------------------------------------------

ENTRIES: list[BenchEntry] = [
    BenchEntry(
        name="haversine",
        category="fusion-chain",
        sizes=[1_000_000, 10_000_000, 100_000_000],
        make_input=_make_taxi,
        engine_fn=lambda df: df.lazy().with_columns(d=_haversine_expr()).collect(engine=_ENGINE),
        cpu_fn=lambda df: df.lazy().with_columns(d=_haversine_expr()).collect(),
        ceiling_fn=_ceil_haversine,
        check=_frame_allclose,
    ),
    BenchEntry(
        name="black_scholes",
        category="fusion-chain",
        sizes=[1_000_000, 10_000_000, 100_000_000],
        make_input=_make_prices,
        engine_fn=lambda df: (
            df.lazy().with_columns(c=_black_scholes_expr()).collect(engine=_ENGINE)
        ),
        cpu_fn=lambda df: df.lazy().with_columns(c=_black_scholes_expr()).collect(),
        ceiling_fn=_ceil_black_scholes,
        check=_frame_allclose,
    ),
]

ENTRIES += [
    _rolling_entry("mean", 1000),
    _rolling_entry("sum", 1000),
    _rolling_entry("var", 1000),
    _rolling_entry("std", 1000),
]

# ---- vector-search helpers -----------------------------------------------

_VEC_D = 768
_VEC_K = 10
_VEC_CORPUS_N = 50_000
_VEC_CPU_CHUNK = 64  # query rows per block for chunked CPU baselines


def _make_queries(n: int, seed: int = 0x7EC) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n, _VEC_D)).astype(np.float32)
    return pl.DataFrame({"emb": emb.tolist()}, schema={"emb": pl.Array(pl.Float32, _VEC_D)})


def _vec_corpus(seed: int = 0xC0A) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((_VEC_CORPUS_N, _VEC_D)).astype(np.float32)


_CORPUS = _vec_corpus()
# Precompute normalized corpus for cosine once.
_CORPUS_NORMS = np.linalg.norm(_CORPUS, axis=1, keepdims=True)
_CORPUS_N = _CORPUS / np.maximum(_CORPUS_NORMS, 1e-12)

# Corpus mx.arrays are module-level constants (corpus never changes across
# benchmark sizes).  Transfer them once at import time so they are always
# resident on GPU when ceiling_fn is called.
_CORPUS_N_MX: mx.array = mx.array(_CORPUS_N)
_CORPUS_MX: mx.array = mx.array(_CORPUS)
mx.eval(_CORPUS_N_MX, _CORPUS_MX)


# ---- cosine_topk ceiling --------------------------------------------------


def _ceil_cosine_topk(df: pl.DataFrame) -> mx.array:
    """Raw MLX cosine top-k: normalize queries on GPU, matmul with pre-resident
    normalized corpus, argpartition.  Host->mx transfer for queries is memoized
    (excluded from timed region); corpus is already resident above."""

    def _build(data: pl.DataFrame) -> mx.array:
        q_np = np.asarray(data["emb"].to_list(), dtype=np.float32)
        q_norms = np.linalg.norm(q_np, axis=1, keepdims=True)
        qn = q_np / np.maximum(q_norms, 1e-12)
        return mx.array(qn)

    q_mx = _ceil_inputs(df, _build)
    sims = q_mx @ _CORPUS_N_MX.T
    idx = mx.argpartition(-sims, kth=_VEC_K - 1, axis=1)[:, :_VEC_K]
    mx.eval(idx)
    return idx


# ---- knn ceiling ----------------------------------------------------------


def _ceil_knn(df: pl.DataFrame) -> mx.array:
    """Raw MLX L2 k-NN: squared-distance identity via matmul, argpartition.
    Query host->mx transfer is memoized; corpus is already resident above."""

    def _build(data: pl.DataFrame) -> mx.array:
        q_np = np.asarray(data["emb"].to_list(), dtype=np.float32)
        return mx.array(q_np)

    q_mx = _ceil_inputs(df, _build)
    q2 = (q_mx * q_mx).sum(axis=1, keepdims=True)
    c2 = (_CORPUS_MX * _CORPUS_MX).sum(axis=1)
    dist2 = q2 + c2 - 2.0 * (q_mx @ _CORPUS_MX.T)
    idx = mx.argpartition(dist2, kth=_VEC_K - 1, axis=1)[:, :_VEC_K]
    mx.eval(idx)
    return idx


def _cpu_cosine_topk(df: pl.DataFrame) -> np.ndarray:
    q = np.asarray(df["emb"].to_list(), dtype=np.float32)
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    # Chunk over queries to cap memory at _VEC_CPU_CHUNK * N floats (~200 MB/chunk).
    rows: list[np.ndarray] = []
    for start in range(0, len(qn), _VEC_CPU_CHUNK):
        block = qn[start : start + _VEC_CPU_CHUNK]
        sims = block @ _CORPUS_N.T
        idx = np.argpartition(-sims, kth=_VEC_K - 1, axis=1)[:, :_VEC_K]
        rows.append(np.sort(idx, axis=1))
    return np.concatenate(rows, axis=0)


def _engine_cosine_topk(df: pl.DataFrame) -> np.ndarray:
    out = (
        df.lazy()
        .with_columns(pl.col("emb").metal.cosine_topk(_CORPUS, k=_VEC_K).alias("h"))
        .collect(engine=_ENGINE)
    )
    # Output: Struct({'indices': List(UInt32), 'scores': List(Float32)})
    hits = np.asarray(out["h"].struct.field("indices").to_list(), dtype=np.int64)
    return np.sort(hits, axis=1)


def _cpu_knn(df: pl.DataFrame) -> np.ndarray:
    # HONEST baseline: the squared-distance identity ‖q-c‖² = ‖q‖² + ‖c‖² - 2·q·cᵀ
    # turns the inner loop into a BLAS matmul (q @ cᵀ) — what a competent CPU knn
    # does. (A naive (Q,N,D) broadcast is ~30x slower and would inflate the win.)
    q = np.asarray(df["emb"].to_list(), dtype=np.float32)
    c2 = np.einsum("ij,ij->i", _CORPUS, _CORPUS)[None, :]  # (1, N) ‖c‖²
    rows: list[np.ndarray] = []
    for start in range(0, len(q), _VEC_CPU_CHUNK):
        block = q[start : start + _VEC_CPU_CHUNK]  # (chunk, D)
        q2 = np.einsum("ij,ij->i", block, block)[:, None]  # (chunk, 1) ‖q‖²
        d2 = q2 + c2 - 2.0 * (block @ _CORPUS.T)  # (chunk, N) via BLAS
        idx = np.argpartition(d2, kth=_VEC_K - 1, axis=1)[:, :_VEC_K]
        rows.append(np.sort(idx, axis=1))
    return np.concatenate(rows, axis=0)


def _engine_knn(df: pl.DataFrame) -> np.ndarray:
    out = (
        df.lazy()
        .with_columns(pl.col("emb").metal.knn(_CORPUS, k=_VEC_K).alias("h"))
        .collect(engine=_ENGINE)
    )
    # Output: Struct({'indices': List(UInt32), 'scores': List(Float32)})
    hits = np.asarray(out["h"].struct.field("indices").to_list(), dtype=np.int64)
    return np.sort(hits, axis=1)


def _check_topk(engine_out: np.ndarray, cpu_out: np.ndarray) -> None:
    # Set-compare per row (top-k order is not defined). Assumes no score ties at
    # the k-boundary, which holds for random standard-normal float32 embeddings
    # (exact ties are measure-zero); a quantized/integer fixture could surface a
    # spurious mismatch here and would need a score-tolerant comparison.
    assert engine_out.shape == cpu_out.shape, (engine_out.shape, cpu_out.shape)
    for i in range(engine_out.shape[0]):
        assert set(engine_out[i].tolist()) == set(cpu_out[i].tolist()), (
            f"row {i}: engine={engine_out[i].tolist()} cpu={cpu_out[i].tolist()}"
        )


ENTRIES += [
    BenchEntry(
        name="cosine_topk",
        category="vector-search",
        # 10k (not 100k): the engine materializes the full Qxcorpus score
        # matrix on-GPU; at Q=100k/corpus=50k/D=768 that's ~20GB, which exceeds
        # Metal's maxBufferLength and OOMs. 10k-by-50k (~2GB) is the largest that
        # fits. The Q=100k OOM cliff is recorded in the report verdict.
        sizes=[1_000, 10_000],
        make_input=_make_queries,
        engine_fn=_engine_cosine_topk,
        cpu_fn=_cpu_cosine_topk,
        ceiling_fn=_ceil_cosine_topk,
        check=_check_topk,
    ),
    BenchEntry(
        name="knn",
        category="vector-search",
        sizes=[1_000, 10_000],  # see cosine_topk note: 100k OOMs the GPU score matrix
        make_input=_make_queries,
        engine_fn=_engine_knn,
        cpu_fn=_cpu_knn,
        ceiling_fn=_ceil_knn,
        check=_check_topk,
    ),
]

# ---- fft helpers -------------------------------------------------------------


def _make_fft_signal(n: int, seed: int = 0xFF7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"sig": rng.standard_normal(n).astype(np.float32)})


def _engine_fft(df: pl.DataFrame) -> pl.DataFrame:
    return df.lazy().with_columns(pl.col("sig").metal.fft().alias("spec")).collect(engine=_ENGINE)


def _cpu_fft(df: pl.DataFrame) -> pl.DataFrame:
    spec = np.fft.fft(df["sig"].to_numpy().astype(np.float64))
    return pl.DataFrame({"spec_re": spec.real, "spec_im": spec.imag})


def _engine_fft_to_complex(out: pl.DataFrame) -> np.ndarray:
    # Engine output: Struct({'real': Float32, 'imag': Float32}) — one struct per row.
    spec = out["spec"]
    re = spec.struct.field("real").to_numpy()
    im = spec.struct.field("imag").to_numpy()
    return re + 1j * im


def _check_fft(engine_out: pl.DataFrame, cpu_out: pl.DataFrame) -> None:
    ev = _engine_fft_to_complex(engine_out)
    cv = cpu_out["spec_re"].to_numpy() + 1j * cpu_out["spec_im"].to_numpy()
    np.testing.assert_allclose(np.abs(ev), np.abs(cv), rtol=1e-2, atol=1e-1)


ENTRIES += [
    BenchEntry(
        name="fft",
        category="fft",
        sizes=[1 << 20, 1 << 23, 1 << 25],
        make_input=_make_fft_signal,
        engine_fn=_engine_fft,
        cpu_fn=_cpu_fft,
        ceiling_fn=None,  # numpy IS the bar; raw MLX fft broken >2^20
        check=_check_fft,
    ),
]

# ---- dtw helpers -------------------------------------------------------------

_DTW_L = 256
_DTW_W = 16
_DTW_REF = np.random.default_rng(0xD7).standard_normal(_DTW_L).astype(np.float32)


def _make_dtw_seqs(n: int, seed: int = 0xD75) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    seqs = rng.standard_normal((n, _DTW_L)).astype(np.float32)
    return pl.DataFrame({"seq": seqs.tolist()}, schema={"seq": pl.Array(pl.Float32, _DTW_L)})


def _engine_dtw(df: pl.DataFrame) -> np.ndarray:
    # Engine output is a plain Float32 column named "d".
    out = (
        df.lazy()
        .with_columns(pl.col("seq").metal.dtw(_DTW_REF, window=_DTW_W).alias("d"))
        .collect(engine=_ENGINE)
    )
    return out["d"].to_numpy()


def _cpu_dtw(df: pl.DataFrame) -> np.ndarray:
    from dtaidistance import dtw

    # HONEST baseline: distance_fast is dtaidistance's C implementation — what a
    # user reaching for the library actually gets. The pure-Python `distance` is
    # ~100x slower and would massively inflate the win.
    seqs = np.ascontiguousarray(np.asarray(df["seq"].to_list(), dtype=np.float64))
    ref = np.ascontiguousarray(_DTW_REF.astype(np.float64))
    # engine window=W  <->  dtaidistance window=W+1  (confirmed in test_dtw_e2e.py)
    return np.array([dtw.distance_fast(s, ref, window=_DTW_W + 1) for s in seqs])


ENTRIES += [
    BenchEntry(
        name="dtw",
        category="dtw",
        sizes=[1_000, 50_000],
        make_input=_make_dtw_seqs,
        engine_fn=_engine_dtw,
        cpu_fn=_cpu_dtw,
        ceiling_fn=None,
        check=lambda e, c: np.testing.assert_allclose(e, c, rtol=1e-2, atol=1e-2),
    ),
]

# ---- corr helpers -------------------------------------------------------------


def _make_corr_df(n: int, p: int, seed: int = 0xC1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p)).astype(np.float32)
    return pl.DataFrame(x, schema=[f"c{i}" for i in range(p)])


def _ceil_corr(df: pl.DataFrame) -> mx.array:
    """Raw MLX correlation matrix: standardize(X) @ standardize(X).T / (n-1).

    Replicates the spike_corr_crossover.py gpu_corr() formula, which matches
    df.corr() (Pearson, ddof=1) byte-for-byte within F32 tolerance.
    Input layout: (n, p) row-major (same as what the engine ingests).
    Host->mx transfer is memoized; only the standardize+GEMM+eval is timed."""
    n = len(df)

    def _build(data: pl.DataFrame) -> mx.array:
        # Stack columns into (n, p) contiguous F32 array; mx.array takes it directly.
        np_mat = np.column_stack([data[c].to_numpy().astype(np.float32) for c in data.columns])
        return mx.array(np_mat)

    Xmx = _ceil_inputs(df, _build)
    mu = mx.mean(Xmx, axis=0, keepdims=True)
    xc = Xmx - mu
    # ddof=1 variance to match Pearson normalization in the GEMM
    var = mx.sum(xc * xc, axis=0, keepdims=True) / (n - 1)
    std = mx.sqrt(var)
    z = xc / std
    c = mx.matmul(z.T, z) / (n - 1)
    mx.eval(c)
    return c


def _corr_entry(p: int) -> BenchEntry:
    return BenchEntry(
        name=f"corr_p{p}",
        category="corr",
        sizes=[100_000, 1_000_000],
        make_input=lambda n, p=p: _make_corr_df(n, p),
        engine_fn=lambda df: df.lazy().metal.corr(force_gpu=True).collect(engine=_ENGINE),
        cpu_fn=lambda df: df.corr(),
        ceiling_fn=_ceil_corr,
        check=_frame_allclose,
    )


ENTRIES += [_corr_entry(10), _corr_entry(50)]

# ---- temporal-int helpers -----------------------------------------------


def _make_datetimes(n: int, seed: int = 0xD7E) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    ms = rng.integers(0, 1_262_304_000_000, size=n)  # epoch-ms over ~40 years
    return pl.DataFrame({"ts": ms}).with_columns(ts=pl.col("ts").cast(pl.Datetime(time_unit="ms")))


def _make_ints(n: int, seed: int = 0x1A7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"v": rng.integers(-1_000_000, 1_000_000, size=n).astype(np.int32)})


ENTRIES += [
    BenchEntry(
        name="dt_year",
        category="temporal-int",
        sizes=[1_000_000, 10_000_000, 50_000_000],
        make_input=_make_datetimes,
        engine_fn=lambda df: (
            df.lazy().with_columns(y=pl.col("ts").dt.year()).collect(engine=_ENGINE)
        ),
        cpu_fn=lambda df: df.lazy().with_columns(y=pl.col("ts").dt.year()).collect(),
        ceiling_fn=None,
        check=_frame_allclose,
    ),
    BenchEntry(
        name="int_sum",
        category="temporal-int",
        sizes=[1_000_000, 10_000_000, 100_000_000],
        make_input=_make_ints,
        engine_fn=lambda df: df.lazy().select(s=pl.col("v").sum()).collect(engine=_ENGINE),
        cpu_fn=lambda df: df.lazy().select(s=pl.col("v").sum()).collect(),
        ceiling_fn=None,
        check=_frame_allclose,
    ),
]

# ---- conformance-loser helpers -----------------------------------------------

_Q1_THRESHOLD = date(1998, 9, 2)


def _apply_q1(df: pl.DataFrame, engine) -> pl.DataFrame:
    return (
        df.lazy()
        .filter(pl.col("l_shipdate") <= _Q1_THRESHOLD)
        .group_by("l_returnflag", "l_linestatus")
        .agg(
            pl.col("l_quantity").sum().alias("sum_qty"),
            pl.col("l_extendedprice").sum().alias("sum_base_price"),
            (pl.col("l_extendedprice") * (1.0 - pl.col("l_discount")))
            .sum()
            .alias("sum_disc_price"),
            (pl.col("l_extendedprice") * (1.0 - pl.col("l_discount")) * (1.0 + pl.col("l_tax")))
            .sum()
            .alias("sum_charge"),
            pl.col("l_quantity").mean().alias("avg_qty"),
            pl.col("l_extendedprice").mean().alias("avg_price"),
            pl.col("l_discount").mean().alias("avg_disc"),
            pl.len().alias("count_order"),
        )
        .sort("l_returnflag", "l_linestatus")
        .collect(engine=engine)
    )


def _apply_q6(df: pl.DataFrame, engine) -> pl.DataFrame:
    return (
        df.lazy()
        .filter(
            (pl.col("l_shipdate") >= date(1994, 1, 1))
            & (pl.col("l_shipdate") < date(1995, 1, 1))
            & (pl.col("l_discount") >= 0.05)
            & (pl.col("l_discount") <= 0.07)
            & (pl.col("l_quantity") < 24)
        )
        .select((pl.col("l_extendedprice") * pl.col("l_discount")).sum().alias("revenue"))
        .collect(engine=engine)
    )


ENTRIES += [
    BenchEntry(
        name="tpch_q1",
        category="conformance-loser",
        sizes=[10_000_000],
        make_input=lambda n: make_canonical_q1_fixture_f32(n_rows=n),
        engine_fn=lambda df: _apply_q1(df, _ENGINE),
        cpu_fn=lambda df: _apply_q1(df, "cpu"),
        ceiling_fn=None,
        # Q1 output is sorted by (l_returnflag, l_linestatus) — deterministic order.
        check=_frame_allclose,
    ),
    BenchEntry(
        name="tpch_q6",
        category="conformance-loser",
        sizes=[10_000_000],
        make_input=lambda n: make_q6_fixture_f32(n_rows=n),
        engine_fn=lambda df: _apply_q6(df, _ENGINE),
        cpu_fn=lambda df: _apply_q6(df, "cpu"),
        ceiling_fn=None,
        check=_frame_allclose,
    ),
    BenchEntry(
        name="bare_sum_f32",
        category="conformance-loser",
        sizes=[1_000_000, 100_000_000],
        make_input=lambda n: pl.DataFrame(
            {"x": np.random.default_rng(0xBA5).standard_normal(n).astype(np.float32)}
        ),
        engine_fn=lambda df: df.lazy().select(s=pl.col("x").sum()).collect(engine=_ENGINE),
        cpu_fn=lambda df: df.lazy().select(s=pl.col("x").sum()).collect(),
        ceiling_fn=None,
        # bare F32 sum at 1e8 magnitude diverges in low digits (known: prop_gpu_sum_f32
        # 1e11 flake). Relative tolerance scaled to the sum magnitude.
        check=lambda e, c: _frame_allclose(e, c, rtol=1e-2, atol=1.0),
    ),
]

# ---- M10 join->gather helpers -----------------------------------------------
#
# These cases measure the engine's join->gather->compute pipeline:
#   fact.join(dim, on="id").with_columns(<Black-Scholes chain>)
#
# Two variants:
#   - dense: dim keys = shuffled 0..dim_n-1 (contiguous) -> resident GPU path.
#     engine_fn uses force_fusion=True so the GPU branch runs at all sizes
#     (the density gate routes rows<~2.1M to CPU by default; force bypasses it).
#   - nondense: dim keys = sparse subset of a wide range -> CPU-lookup + GPU chain.

_M10_DIM_N = 20_000  # dim table size (both variants)
_M10_CHAIN_EXPR = (
    pl.col("value") * 0.5 * (1.0 + (0.7978845608 * pl.col("vol").log()).tanh())
).alias("out")
_M10_ENGINE_FORCE = pm.MetalEngine(force_fusion=True)


def _m10_make_dense(n: int, seed: int = 0xA1) -> dict[str, pl.DataFrame]:
    """fact (n rows, dense keys 0.._M10_DIM_N-1) + dim (_M10_DIM_N rows, shuffled key)."""
    rng = np.random.default_rng(seed)
    fact = pl.DataFrame(
        {
            "id": rng.integers(0, _M10_DIM_N, size=n).astype(np.int64),
            "value": rng.uniform(50, 150, size=n).astype(np.float32),
        }
    )
    dim = pl.DataFrame(
        {
            "id": rng.permutation(_M10_DIM_N).astype(np.int64),
            "vol": rng.uniform(0.1, 0.5, size=_M10_DIM_N).astype(np.float32),
        }
    )
    return {"fact": fact, "dim": dim}


def _m10_make_nondense(n: int, seed: int = 0xA2) -> dict[str, pl.DataFrame]:
    """fact (n rows) + dim with sparse keys (subset of a wide key range)."""
    rng = np.random.default_rng(seed)
    # Wide key range makes the dim keys non-contiguous -> nondense branch.
    wide = max(_M10_DIM_N * 5, 100_000)
    dim_keys = rng.choice(wide, _M10_DIM_N, replace=False).astype(np.int64)
    fact = pl.DataFrame(
        {
            "id": rng.choice(dim_keys, size=n).astype(np.int64),
            "value": rng.uniform(50, 150, size=n).astype(np.float32),
        }
    )
    dim = pl.DataFrame(
        {"id": dim_keys, "vol": rng.uniform(0.1, 0.5, size=_M10_DIM_N).astype(np.float32)}
    )
    return {"fact": fact, "dim": dim}


def _m10_lf(inp: dict[str, pl.DataFrame]) -> pl.LazyFrame:
    return inp["fact"].lazy().join(inp["dim"].lazy(), on="id").with_columns(_M10_CHAIN_EXPR)


def _ceil_m10_dense(inp: dict[str, pl.DataFrame]) -> mx.array:
    """Raw MLX resident gather + BS chain (ceiling for dense path, no engine overhead)."""

    def _build(data: dict[str, pl.DataFrame]) -> tuple[mx.array, mx.array, mx.array]:
        f, d = data["fact"], data["dim"]
        # Sort dim by key to build a dense 0-indexed lookup (key == position after sort).
        d_sorted = d.sort("id")
        dim_vol = mx.array(d_sorted["vol"].to_numpy())
        fact_id = mx.array(f["id"].to_numpy())
        fact_val = mx.array(f["value"].to_numpy())
        return dim_vol, fact_id, fact_val

    dim_vol, fact_id, fact_val = _ceil_inputs(inp, _build)
    gvol = mx.take(dim_vol, fact_id, axis=0)  # resident gather
    out = fact_val * 0.5 * (1.0 + mx.tanh(0.7978845608 * mx.log(gvol)))
    mx.eval(out)
    return out


def _ceil_m10_nondense(inp: dict[str, pl.DataFrame]) -> mx.array:
    """CPU gather + raw MLX chain (ceiling for nondense: gather is cheap CPU indexing)."""

    def _build(data: dict[str, pl.DataFrame]) -> tuple[mx.array, mx.array]:
        # Join on CPU (what the engine's CPU-lookup branch does), then GPU chain.
        joined = data["fact"].join(data["dim"], on="id")
        gs = mx.array(joined["value"].to_numpy())
        gvol = mx.array(joined["vol"].to_numpy())
        return gs, gvol

    gs, gvol = _ceil_inputs(inp, _build)
    out = gs * 0.5 * (1.0 + mx.tanh(0.7978845608 * mx.log(gvol)))
    mx.eval(out)
    return out


def _check_m10(engine_out: pl.DataFrame, cpu_out: pl.DataFrame) -> None:
    # Join output row order may differ between engine and CPU paths; sort by (id, value)
    # before comparing so the numeric check is stable.
    sort_cols = ["id", "value"]
    e = engine_out.sort(sort_cols)
    c = cpu_out.sort(sort_cols)
    _frame_allclose(e, c, rtol=1e-3, atol=1e-3)


ENTRIES += [
    BenchEntry(
        name="m10_join_gather_dense",
        category="join-gather",
        # Headline: 10M rows. Smoke runs at 1_000 (force_fusion bypasses density gate).
        sizes=[1_000, 10_000_000],
        make_input=_m10_make_dense,
        # force_fusion so the GPU resident-gather branch runs at all sizes; default routing
        # gates rows < ~2.1M to CPU (the FLOPs floor). Comment kept for report readers.
        engine_fn=lambda inp: _m10_lf(inp).collect(engine=_M10_ENGINE_FORCE),
        cpu_fn=lambda inp: _m10_lf(inp).collect(),
        ceiling_fn=_ceil_m10_dense,
        check=_check_m10,
    ),
    BenchEntry(
        name="m10_join_gather_nondense",
        category="join-gather",
        # Nondense: sparse keys -> CPU-lookup + GPU chain. No force needed.
        sizes=[1_000, 10_000_000],
        make_input=_m10_make_nondense,
        engine_fn=lambda inp: _m10_lf(inp).collect(engine=_ENGINE),
        cpu_fn=lambda inp: _m10_lf(inp).collect(),
        ceiling_fn=_ceil_m10_nondense,
        check=_check_m10,
    ),
]

# ---- M10 vector rerank helpers -----------------------------------------------
#
# Measures the resident vector rerank path: cosine top-k retrieval followed by
# exp_decay reranking, all on GPU with no fold-back between the two phases.
#
# Corpus is fixed at _RERANK_N rows (module-level constant, pre-resident on GPU
# for the ceiling). `sizes` drives Q (query count); corpus size is always _RERANK_N.

_RERANK_D = 256
_RERANK_K = 10
_RERANK_N = 200_000  # corpus size (fixed; benchmark scale lever is Q)


def _rerank_corpus_arr(seed: int = 0xB1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((_RERANK_N, _RERANK_D)).astype(np.float32)


def _rerank_weights_arr(seed: int = 0xB2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, _RERANK_N).astype(np.float32)


_RERANK_CORPUS = _rerank_corpus_arr()
_RERANK_WEIGHTS = _rerank_weights_arr()
_RERANK_CORPUS_N = _RERANK_CORPUS / np.maximum(
    np.linalg.norm(_RERANK_CORPUS, axis=1, keepdims=True), 1e-12
)

# Pre-build the Polars corpus DataFrame (needed by cosine_topk engine_fn).
_RERANK_CORPUS_DF = pl.DataFrame(
    {"emb": _RERANK_CORPUS.tolist()},
    schema={"emb": pl.Array(pl.Float32, _RERANK_D)},
)
_RERANK_WEIGHTS_SERIES = pl.Series(_RERANK_WEIGHTS)

# Pre-resident MLX arrays for the ceiling (transferred once at import).
_RERANK_CORPUS_N_MX: mx.array = mx.array(_RERANK_CORPUS_N)
mx.eval(_RERANK_CORPUS_N_MX)


def _make_rerank_queries(q: int, seed: int = 0xB3) -> pl.DataFrame:
    """Return a DataFrame of q query embeddings (Array[Float32, D])."""
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((q, _RERANK_D)).astype(np.float32)
    return pl.DataFrame({"emb": emb.tolist()}, schema={"emb": pl.Array(pl.Float32, _RERANK_D)})


def _engine_rerank(qdf: pl.DataFrame) -> pl.DataFrame:
    return (
        qdf.lazy()
        .with_columns(
            pl.col("emb")
            .metal.cosine_topk(
                _RERANK_CORPUS_DF,
                _RERANK_K,
                rerank_weight=_RERANK_WEIGHTS_SERIES,
                rerank="exp_decay",
            )
            .alias("hit")
        )
        .collect(engine=_ENGINE)
    )


def _cpu_rerank(qdf: pl.DataFrame) -> pl.DataFrame:
    """Numpy oracle: cosine top-k by similarity, then rerank sim*exp(-weight[hit])."""
    q_np = np.asarray(qdf["emb"].to_list(), dtype=np.float32)
    qn = q_np / np.maximum(np.linalg.norm(q_np, axis=1, keepdims=True), 1e-12)
    # Chunked to cap peak memory at ~64 * N floats per block.
    chunk = 64
    rows_idx: list[np.ndarray] = []
    rows_scores: list[np.ndarray] = []
    for start in range(0, len(qn), chunk):
        block = qn[start : start + chunk]
        sims = block @ _RERANK_CORPUS_N.T  # (chunk, N)
        hits = np.argpartition(-sims, kth=_RERANK_K - 1, axis=1)[:, :_RERANK_K]
        hit_sims = np.take_along_axis(sims, hits, axis=1)
        reranked = hit_sims * np.exp(-_RERANK_WEIGHTS[hits])
        rows_idx.append(hits)
        rows_scores.append(reranked)
    idx_all = np.concatenate(rows_idx, axis=0)
    scores_all = np.concatenate(rows_scores, axis=0)
    # Return as a DataFrame parallel to the engine's "hit" struct output.
    return pl.DataFrame(
        {
            "indices": [row.tolist() for row in idx_all],
            "scores": [row.tolist() for row in scores_all],
        }
    )


def _ceil_rerank(qdf: pl.DataFrame) -> mx.array:
    """Raw MLX rerank: normalize queries on GPU, matmul w/ pre-resident corpus,
    argpartition top-k, gather weights, exp_decay. Query transfer is memoized."""

    def _build(data: pl.DataFrame) -> mx.array:
        q_np = np.asarray(data["emb"].to_list(), dtype=np.float32)
        qn = q_np / np.maximum(np.linalg.norm(q_np, axis=1, keepdims=True), 1e-12)
        return mx.array(qn)

    w_mx = mx.array(_RERANK_WEIGHTS)
    mx.eval(w_mx)
    q_mx = _ceil_inputs(qdf, _build)
    sims = q_mx @ _RERANK_CORPUS_N_MX.T  # (Q, N)
    hits = mx.argpartition(-sims, kth=_RERANK_K - 1, axis=1)[:, :_RERANK_K]
    hit_sims = mx.take_along_axis(sims, hits, axis=1)
    feat = mx.take(w_mx, hits.reshape(-1), axis=0).reshape(hits.shape)
    reranked = hit_sims * mx.exp(-feat)
    mx.eval(hits, reranked)
    return hits  # ceiling = how fast top-k indices land on GPU


def _check_rerank(engine_out: pl.DataFrame, cpu_out: pl.DataFrame) -> None:
    """Set-compare top-k indices per row; allclose on reranked scores (sorted desc)."""
    hit_col = engine_out["hit"]
    for i in range(len(hit_col)):
        eng_idx = {int(x) for x in hit_col[i]["indices"]}
        cpu_idx = {int(x) for x in cpu_out["indices"][i]}
        assert eng_idx == cpu_idx, f"row {i}: engine={sorted(eng_idx)} cpu={sorted(cpu_idx)}"
    eng_scores = np.array(
        [
            sorted((float(x) for x in hit_col[i]["scores"]), reverse=True)
            for i in range(len(hit_col))
        ]
    )
    cpu_scores = np.array(
        [
            sorted((float(x) for x in cpu_out["scores"][i]), reverse=True)
            for i in range(len(hit_col))
        ]
    )
    np.testing.assert_allclose(eng_scores, cpu_scores, rtol=1e-3, atol=1e-3)


ENTRIES += [
    BenchEntry(
        name="m10_vector_rerank",
        category="join-gather",
        # Q=100 headline. Smoke runs at Q=10. Corpus is always _RERANK_N=200_000.
        sizes=[10, 100],
        make_input=_make_rerank_queries,
        engine_fn=_engine_rerank,
        cpu_fn=_cpu_rerank,
        ceiling_fn=_ceil_rerank,
        check=_check_rerank,
    ),
]
