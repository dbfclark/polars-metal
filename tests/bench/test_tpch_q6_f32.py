"""F32 variant of the Q6 bench. See ``_q6_fixture_f32`` for the F32 rationale."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_metal
from tests.bench._q6_fixture_f32 import make_q6_fixture_f32


def _query(df: pl.DataFrame, engine):
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


@pytest.fixture(scope="module")
def lineitem_q6_f32() -> pl.DataFrame:
    return make_q6_fixture_f32()


def test_q6_f32_correctness(lineitem_q6_f32: pl.DataFrame) -> None:
    cpu = _query(lineitem_q6_f32, "cpu")
    metal = _query(lineitem_q6_f32, polars_metal.MetalEngine())
    # F32 rounding tolerance — same as canonical Q1 F32.
    assert_frame_equal(cpu, metal, check_dtypes=False, rtol=1e-5, atol=1e-2)


@pytest.mark.benchmark(group="tpch_q6_f32")
def test_bench_q6_f32_cpu(benchmark, lineitem_q6_f32: pl.DataFrame) -> None:
    result = benchmark(lambda: _query(lineitem_q6_f32, "cpu"))
    assert result.height == 1


@pytest.mark.benchmark(group="tpch_q6_f32")
def test_bench_q6_f32_metal(benchmark, lineitem_q6_f32: pl.DataFrame) -> None:
    engine = polars_metal.MetalEngine()
    result = benchmark(lambda: _query(lineitem_q6_f32, engine))
    assert result.height == 1
