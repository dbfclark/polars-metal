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

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl

import polars_metal as pm
from tests.bench._canonical_q1_fixture_f32 import make_canonical_q1_fixture_f32
from tests.bench._q6_fixture_f32 import make_q6_fixture_f32
from tests.bench.m4_engine.bench_haversine_e2e import _haversine_expr, _make_taxi

_ENGINE = pm.MetalEngine()


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
    import math

    s = pl.col("s")
    k, r, t, vol = 100.0, 0.02, 1.0, 0.3
    d1 = ((s / k).log() + (r + 0.5 * vol * vol) * t) / (vol * (t**0.5))
    d2 = d1 - vol * (t**0.5)

    # crude normal-CDF proxy via tanh approx -- identical on both paths.
    def ncdf(x: pl.Expr) -> pl.Expr:
        return 0.5 * (1.0 + (x * 0.7978845608).tanh())

    # discount factor is a scalar constant — compute in Python, not as a Polars expr.
    discount = math.exp(-r * t)
    return s * ncdf(d1) - k * discount * ncdf(d2)


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
        ceiling_fn=None,
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
        ceiling_fn=None,
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
        ceiling_fn=None,
        check=_check_topk,
    ),
    BenchEntry(
        name="knn",
        category="vector-search",
        sizes=[1_000, 10_000],  # see cosine_topk note: 100k OOMs the GPU score matrix
        make_input=_make_queries,
        engine_fn=_engine_knn,
        cpu_fn=_cpu_knn,
        ceiling_fn=None,
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


def _corr_entry(p: int) -> BenchEntry:
    return BenchEntry(
        name=f"corr_p{p}",
        category="corr",
        sizes=[100_000, 1_000_000],
        make_input=lambda n, p=p: _make_corr_df(n, p),
        engine_fn=lambda df: df.lazy().metal.corr(force_gpu=True).collect(engine=_ENGINE),
        cpu_fn=lambda df: df.corr(),
        ceiling_fn=None,
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
