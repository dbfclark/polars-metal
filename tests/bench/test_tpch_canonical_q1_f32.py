"""Canonical-shape TPC-H Q1 bench with F32 numerics. See
``_canonical_q1_fixture_f32`` for the architectural framing — F32 is the
"what the chip can do with the right primitives" reading; F64 is a
chip-limitation reading (no 64-bit atomic_float on this toolchain).
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_metal
from tests.bench._canonical_q1_fixture_f32 import make_canonical_q1_fixture_f32

_THRESHOLD = date(1998, 9, 2)


def _query(df: pl.DataFrame, engine):
    return (
        df.lazy()
        .filter(pl.col("l_shipdate") <= _THRESHOLD)
        .group_by("l_returnflag", "l_linestatus")
        .agg(
            pl.col("l_quantity").sum().alias("sum_qty"),
            pl.col("l_extendedprice").sum().alias("sum_base_price"),
            (pl.col("l_extendedprice") * (1.0 - pl.col("l_discount"))).sum().alias("sum_disc_price"),
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


@pytest.fixture(scope="module")
def lineitem_canonical_f32() -> pl.DataFrame:
    return make_canonical_q1_fixture_f32()


def test_canonical_q1_f32_correctness(lineitem_canonical_f32: pl.DataFrame) -> None:
    """CPU and Metal must produce equal results (F32 rounding only)."""
    cpu = _query(lineitem_canonical_f32, "cpu")
    metal = _query(lineitem_canonical_f32, polars_metal.MetalEngine())
    # F32 arithmetic accumulates differently between Polars CPU and the
    # fused Metal kernel; allow ~1ulp per value via assert_frame_equal's
    # default float-tolerance helpers.
    assert_frame_equal(cpu, metal, check_dtypes=False, rtol=1e-5, atol=1e-2)


@pytest.mark.benchmark(group="tpch_canonical_q1_f32")
def test_bench_canonical_q1_f32_cpu(benchmark, lineitem_canonical_f32: pl.DataFrame) -> None:
    result = benchmark(lambda: _query(lineitem_canonical_f32, "cpu"))
    assert result.height >= 1


@pytest.mark.benchmark(group="tpch_canonical_q1_f32")
def test_bench_canonical_q1_f32_metal(benchmark, lineitem_canonical_f32: pl.DataFrame) -> None:
    engine = polars_metal.MetalEngine()
    result = benchmark(lambda: _query(lineitem_canonical_f32, engine))
    assert result.height >= 1
