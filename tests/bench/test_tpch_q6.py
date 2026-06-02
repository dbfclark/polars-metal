"""TPC-H Q6 bench: filter-heavy single-group reduction. Capability C
end-to-end. Phase 13 unlocks this shape by:

  - Walker recognizes ``SELECT(agg(expr))`` and emits an empty-key
    ``GroupBy`` plan node (`_walker._walk_select_reduction`).
  - Kernel layer's ``dispatch_groupby`` / ``dispatch_groupby_fused``
    short-circuit empty keys with a synthetic single-group BuildOutput.
  - Predicate walker widens Date + narrow-integer columns to I64 so the
    existing cmp_i64 kernel covers Q6's 4-predicate AND.
  - Empty-keys F64-Expression aggs are materialized via Polars on the
    (post-filter) upstream and rewritten as Simple-Sum specs.

CPU vs Metal: correctness first, then median wall-clock recorded.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_metal
from tests.bench._q6_fixture import make_q6_fixture


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
def lineitem_q6() -> pl.DataFrame:
    return make_q6_fixture()


def test_q6_correctness(lineitem_q6: pl.DataFrame) -> None:
    """CPU and Metal must produce byte-equal results."""
    cpu = _query(lineitem_q6, "cpu")
    metal = _query(lineitem_q6, polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


@pytest.mark.benchmark(group="tpch_q6")
def test_bench_q6_cpu(benchmark, lineitem_q6: pl.DataFrame) -> None:
    result = benchmark(lambda: _query(lineitem_q6, "cpu"))
    assert result.height == 1


@pytest.mark.benchmark(group="tpch_q6")
def test_bench_q6_metal(benchmark, lineitem_q6: pl.DataFrame) -> None:
    engine = polars_metal.MetalEngine()
    result = benchmark(lambda: _query(lineitem_q6, engine))
    assert result.height == 1
