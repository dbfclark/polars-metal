"""df.collect(engine=MetalEngine()) returns CPU-equivalent results in M0."""

import polars as pl

import polars_metal


def test_collect_with_metal_engine_returns_cpu_result() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    cpu = lf.collect()
    metal = lf.collect(engine=polars_metal.MetalEngine())
    assert metal.equals(cpu)


def test_collect_with_metal_engine_filter() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    cpu = lf.filter(pl.col("a") > 2).collect()
    metal = lf.filter(pl.col("a") > 2).collect(engine=polars_metal.MetalEngine())
    assert metal.equals(cpu)


def test_collect_with_metal_engine_groupby() -> None:
    lf = pl.LazyFrame({"k": ["a", "a", "b"], "v": [1, 2, 3]})
    cpu = lf.group_by("k").agg(pl.col("v").sum()).sort("k").collect()
    metal = (
        lf.group_by("k").agg(pl.col("v").sum()).sort("k").collect(engine=polars_metal.MetalEngine())
    )
    assert metal.equals(cpu)
