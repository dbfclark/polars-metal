"""End-to-end pytest-benchmark queries for M1 filter coverage.

Five queries, each run on both CPU and Metal engines (10 tests total). Per the
M1 spec § Layer 4: 10M-row Int64 columns; column ``a`` has ~50% null density to
exercise the null-aware kernel path.

Selectivity for the ``high_selectivity`` / ``low_selectivity`` variants is
controlled via column ``b`` (null-free, values in ``[0, 99]``) so the target
ratios (~1% / ~99%) are actually reachable. Predicates on ``a`` cannot exceed
50% selectivity because of the null density, which would invalidate the
``low_selectivity`` variant.
"""

from __future__ import annotations

import polars as pl
import pytest

import polars_metal

ROWS = 10_000_000


@pytest.fixture(scope="module")
def big_frame() -> pl.DataFrame:
    n = ROWS
    idx = pl.int_range(0, n, dtype=pl.Int64, eager=True)
    return pl.select(
        pl.when(idx % 2 == 0).then(None).otherwise((idx % 200) - 100).cast(pl.Int64).alias("a"),
        (idx % 100).cast(pl.Int64).alias("b"),
        ((idx * 7) % 100).cast(pl.Int64).alias("c"),
    )


@pytest.fixture(scope="module")
def metal_engine() -> polars_metal.MetalEngine:
    # Warm up pipeline-state-object construction / shader compilation so the
    # first measured Metal run isn't paying one-time setup cost.
    engine = polars_metal.MetalEngine()
    warmup = pl.DataFrame({"x": pl.Series("x", [1, 2, 3], dtype=pl.Int64)})
    warmup.lazy().filter(pl.col("x") > 0).collect(engine=engine)
    return engine


# ---------- bench_filter_simple ----------
# pl.col("a") > 0 — ~50% of non-null rows match (~25% of total).


@pytest.mark.benchmark(group="filter_simple_cpu")
def test_bench_filter_simple_cpu(benchmark, big_frame: pl.DataFrame) -> None:
    benchmark(lambda: big_frame.lazy().filter(pl.col("a") > 0).collect())


@pytest.mark.benchmark(group="filter_simple_metal")
def test_bench_filter_simple_metal(
    benchmark,
    big_frame: pl.DataFrame,
    metal_engine: polars_metal.MetalEngine,
) -> None:
    benchmark(lambda: big_frame.lazy().filter(pl.col("a") > 0).collect(engine=metal_engine))


# ---------- bench_filter_compound ----------
# (a > 0) AND (b < c) — exercises 3-valued AND with nulls in a.


@pytest.mark.benchmark(group="filter_compound_cpu")
def test_bench_filter_compound_cpu(benchmark, big_frame: pl.DataFrame) -> None:
    benchmark(
        lambda: big_frame.lazy().filter((pl.col("a") > 0) & (pl.col("b") < pl.col("c"))).collect()
    )


@pytest.mark.benchmark(group="filter_compound_metal")
def test_bench_filter_compound_metal(
    benchmark,
    big_frame: pl.DataFrame,
    metal_engine: polars_metal.MetalEngine,
) -> None:
    benchmark(
        lambda: (
            big_frame.lazy()
            .filter((pl.col("a") > 0) & (pl.col("b") < pl.col("c")))
            .collect(engine=metal_engine)
        )
    )


# ---------- bench_filter_then_project ----------
# Compound filter + select(a, b).


@pytest.mark.benchmark(group="filter_then_project_cpu")
def test_bench_filter_then_project_cpu(benchmark, big_frame: pl.DataFrame) -> None:
    benchmark(
        lambda: (
            big_frame.lazy()
            .filter((pl.col("a") > 0) & (pl.col("b") < pl.col("c")))
            .select("a", "b")
            .collect()
        )
    )


@pytest.mark.benchmark(group="filter_then_project_metal")
def test_bench_filter_then_project_metal(
    benchmark,
    big_frame: pl.DataFrame,
    metal_engine: polars_metal.MetalEngine,
) -> None:
    benchmark(
        lambda: (
            big_frame.lazy()
            .filter((pl.col("a") > 0) & (pl.col("b") < pl.col("c")))
            .select("a", "b")
            .collect(engine=metal_engine)
        )
    )


# ---------- bench_filter_then_project_high_selectivity ----------
# Predicate matches ~1% of rows. Filters on b (null-free, uniform in [0,99]).


@pytest.mark.benchmark(group="filter_then_project_high_selectivity_cpu")
def test_bench_filter_then_project_high_selectivity_cpu(
    benchmark,
    big_frame: pl.DataFrame,
) -> None:
    benchmark(
        lambda: big_frame.lazy().filter(pl.col("b") < 1).select("a", "b").collect(),
    )


@pytest.mark.benchmark(group="filter_then_project_high_selectivity_metal")
def test_bench_filter_then_project_high_selectivity_metal(
    benchmark,
    big_frame: pl.DataFrame,
    metal_engine: polars_metal.MetalEngine,
) -> None:
    benchmark(
        lambda: (
            big_frame.lazy().filter(pl.col("b") < 1).select("a", "b").collect(engine=metal_engine)
        ),
    )


# ---------- bench_filter_then_project_low_selectivity ----------
# Predicate matches ~99% of rows.


@pytest.mark.benchmark(group="filter_then_project_low_selectivity_cpu")
def test_bench_filter_then_project_low_selectivity_cpu(
    benchmark,
    big_frame: pl.DataFrame,
) -> None:
    benchmark(
        lambda: big_frame.lazy().filter(pl.col("b") < 99).select("a", "b").collect(),
    )


@pytest.mark.benchmark(group="filter_then_project_low_selectivity_metal")
def test_bench_filter_then_project_low_selectivity_metal(
    benchmark,
    big_frame: pl.DataFrame,
    metal_engine: polars_metal.MetalEngine,
) -> None:
    benchmark(
        lambda: (
            big_frame.lazy().filter(pl.col("b") < 99).select("a", "b").collect(engine=metal_engine)
        ),
    )
