"""100M-row scale of TPC-H Q1. Validates whether M3's engine-level
perf scales — at 10M rows fixed-overhead (encode, dispatch_build,
FFI marshalling) is ~50% of Metal's wall-clock; at 100M rows the
fixed-overhead becomes a much smaller fraction and any per-row
benefit (A1 build, fused agg) should show more clearly.

Skipped on small machines (the fixture is ~6GB).
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

import polars_metal
from tests.bench._canonical_q1_fixture import make_canonical_q1_fixture
from tests.bench._lineitem_fixture import make_lineitem

_THRESHOLD_DAYS = (date(1998, 9, 2) - date(1970, 1, 1)).days
_THRESHOLD_DATE = date(1998, 9, 2)


def _q1_bool(df: pl.DataFrame, engine):
    """Bool-keyed Q1 (M2 schema; 4 groups, mixed i64/f64 sum/mean)."""
    return (
        df.lazy()
        .filter(pl.col("l_shipdate") <= _THRESHOLD_DAYS)
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
        .collect(engine=engine)
    )


def _q1_canonical(df: pl.DataFrame, engine):
    """Canonical-schema Q1 (Utf8 keys, expression aggs, fused path)."""
    return (
        df.lazy()
        .filter(pl.col("l_shipdate") <= _THRESHOLD_DATE)
        .group_by("l_returnflag", "l_linestatus")
        .agg(
            pl.col("l_quantity").sum().alias("sum_qty"),
            pl.col("l_extendedprice").sum().alias("sum_base_price"),
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount"))).sum().alias("sum_disc_price"),
            pl.col("l_quantity").mean().alias("avg_qty"),
            pl.len().alias("count_order"),
        )
        .sort("l_returnflag", "l_linestatus")
        .collect(engine=engine)
    )


@pytest.fixture(scope="module")
def lineitem_100m_bool() -> pl.DataFrame:
    """M2-schema 100M-row fixture (Bool keys)."""
    return make_lineitem(n_rows=100_000_000)


@pytest.fixture(scope="module")
def lineitem_100m_canonical() -> pl.DataFrame:
    """Canonical-schema 100M-row fixture (Utf8 keys)."""
    return make_canonical_q1_fixture(n_rows=100_000_000)


@pytest.mark.benchmark(group="tpch_q1_100m")
def test_bench_100m_q1_bool_cpu(benchmark, lineitem_100m_bool) -> None:
    result = benchmark(lambda: _q1_bool(lineitem_100m_bool, "cpu"))
    assert result.height >= 1


@pytest.mark.benchmark(group="tpch_q1_100m")
def test_bench_100m_q1_bool_metal(benchmark, lineitem_100m_bool) -> None:
    engine = polars_metal.MetalEngine()
    result = benchmark(lambda: _q1_bool(lineitem_100m_bool, engine))
    assert result.height >= 1


@pytest.mark.benchmark(group="tpch_q1_100m_canonical")
def test_bench_100m_q1_canonical_cpu(benchmark, lineitem_100m_canonical) -> None:
    result = benchmark(lambda: _q1_canonical(lineitem_100m_canonical, "cpu"))
    assert result.height >= 1


@pytest.mark.benchmark(group="tpch_q1_100m_canonical")
def test_bench_100m_q1_canonical_metal(benchmark, lineitem_100m_canonical) -> None:
    engine = polars_metal.MetalEngine()
    result = benchmark(lambda: _q1_canonical(lineitem_100m_canonical, engine))
    assert result.height >= 1
