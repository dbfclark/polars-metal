"""Canonical TPC-H Q1 bench: Utf8 keys (l_returnflag, l_linestatus) +
raw arithmetic expressions in the aggs. The M3 Phase 7 (Utf8 dtype)
work unlocks this fixture as the actual TPC-H Q1 shape — prior M2/M3
benches used Bool keys as an encoder shortcut.

CPU vs Metal: correctness first, then median wall-clock recorded.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import polars_metal
import pytest
from polars.testing import assert_frame_equal

from tests.bench._canonical_q1_fixture import make_canonical_q1_fixture

_THRESHOLD = date(1998, 9, 2)


def _query(df: pl.DataFrame, engine):
    return (
        df.lazy()
        .filter(pl.col("l_shipdate") <= _THRESHOLD)
        .group_by("l_returnflag", "l_linestatus")
        .agg(
            pl.col("l_quantity").sum().alias("sum_qty"),
            pl.col("l_extendedprice").sum().alias("sum_base_price"),
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount")))
            .sum()
            .alias("sum_disc_price"),
            (pl.col("l_extendedprice") * (1 - pl.col("l_discount")) * (1 + pl.col("l_tax")))
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


@pytest.fixture(scope="module")
def lineitem_canonical() -> pl.DataFrame:
    return make_canonical_q1_fixture()


def test_canonical_q1_correctness(lineitem_canonical: pl.DataFrame) -> None:
    """CPU and Metal must produce byte-equal results."""
    cpu = _query(lineitem_canonical, "cpu")
    metal = _query(lineitem_canonical, polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


@pytest.mark.benchmark(group="tpch_canonical_q1")
def test_bench_canonical_q1_cpu(benchmark, lineitem_canonical: pl.DataFrame) -> None:
    result = benchmark(lambda: _query(lineitem_canonical, "cpu"))
    assert result.height >= 1


@pytest.mark.benchmark(group="tpch_canonical_q1")
def test_bench_canonical_q1_metal(benchmark, lineitem_canonical: pl.DataFrame) -> None:
    engine = polars_metal.MetalEngine()
    result = benchmark(lambda: _query(lineitem_canonical, engine))
    assert result.height >= 1
