# tests/bench/test_tpch_q1.py
"""Modified TPC-H Q1 benchmark.

Spec § "Workload validated":
  - Boolean l_returnflag / l_linestatus (not the spec-proposed i64, due
    to encoder's 128-bit composite-key budget; see _lineitem_fixture.py)
  - disc_price and charge pre-projected into the input
  - Otherwise identical to TPC-H Q1: filter on shipdate threshold,
    group_by(returnflag, linestatus), 7 aggregations + count, sort by keys.

Two benchmarks: tpch_q1_cpu and tpch_q1_metal. The timed region includes
the filter (CPU-routed under M2), the groupby (GPU-routed), and the sort
(CPU-routed) — that's the full Q1 wall-clock the user observes.

baseline.json records cpu_ms / metal_ms / ratio_metal_over_cpu after T39.
M2 ships iff ratio < 1.0.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

import polars_metal
from tests.bench._lineitem_fixture import make_lineitem

_THRESHOLD = (date(1998, 9, 2) - date(1970, 1, 1)).days


def _q1(lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        lf.filter(pl.col("l_shipdate") <= _THRESHOLD)
        .group_by("l_returnflag", "l_linestatus")
        .agg(
            pl.col("l_quantity").sum().alias("sum_qty"),
            pl.col("l_extendedprice").sum().alias("sum_base_price"),
            pl.col("disc_price").sum().alias("sum_disc_price"),
            pl.col("charge").sum().alias("sum_charge"),
            pl.col("l_quantity").mean().alias("avg_qty"),
            pl.col("l_extendedprice").mean().alias("avg_price"),
            pl.col("l_discount").mean().alias("avg_disc"),
            pl.len().alias("count_order"),
        )
        .sort("l_returnflag", "l_linestatus")
    )


@pytest.fixture(scope="module")
def lineitem_10m() -> pl.DataFrame:
    """10M-row lineitem fixture, built once per test module."""
    return make_lineitem(n_rows=10_000_000, seed=0xC0FFEE)


@pytest.mark.benchmark(group="tpch_q1")
def test_bench_tpch_q1_cpu(benchmark, lineitem_10m: pl.DataFrame) -> None:
    """Baseline: pure-CPU Polars on the modified Q1."""

    def run() -> pl.DataFrame:
        return _q1(lineitem_10m.lazy()).collect(engine="cpu")

    out = benchmark(run)
    assert out.height == 4, f"expected 4 (returnflag, linestatus) groups, got {out.height}"


@pytest.mark.benchmark(group="tpch_q1")
def test_bench_tpch_q1_metal(benchmark, lineitem_10m: pl.DataFrame) -> None:
    """Metal engine: filter on CPU, groupby on GPU, sort on CPU."""
    engine = polars_metal.MetalEngine()

    def run() -> pl.DataFrame:
        return _q1(lineitem_10m.lazy()).collect(engine=engine)

    out = benchmark(run)
    assert out.height == 4, f"expected 4 (returnflag, linestatus) groups, got {out.height}"


def test_q1_correctness(lineitem_10m: pl.DataFrame) -> None:
    """Sanity: both engines produce the same result for the modified Q1."""
    from polars.testing import assert_frame_equal

    cpu = _q1(lineitem_10m.lazy()).collect(engine="cpu")
    metal = _q1(lineitem_10m.lazy()).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)
