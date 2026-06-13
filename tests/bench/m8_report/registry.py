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
from typing import Any

import numpy as np
import polars as pl

import polars_metal as pm
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
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=f"col {col}")


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
