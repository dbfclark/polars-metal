"""Representative Polars operations under both engines. Output must match."""

import polars as pl
from polars.testing import assert_frame_equal


def test_select(engine) -> None:  # type: ignore[no-untyped-def]
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    df = lf.select("a", (pl.col("b") + 1).alias("b1")).collect(engine=engine)
    assert df.columns == ["a", "b1"]
    assert df["b1"].to_list() == [11, 21, 31]


def test_filter(engine) -> None:  # type: ignore[no-untyped-def]
    lf = pl.LazyFrame({"a": [1, 2, 3, 4]})
    df = lf.filter(pl.col("a") > 2).collect(engine=engine)
    assert df["a"].to_list() == [3, 4]


def test_group_by_sum(engine) -> None:  # type: ignore[no-untyped-def]
    lf = pl.LazyFrame({"k": ["a", "a", "b", "b"], "v": [1, 2, 3, 4]})
    df = lf.group_by("k").agg(pl.col("v").sum()).sort("k").collect(engine=engine)
    assert df["k"].to_list() == ["a", "b"]
    assert df["v"].to_list() == [3, 7]


def test_join(engine) -> None:  # type: ignore[no-untyped-def]
    left = pl.LazyFrame({"k": [1, 2, 3], "lv": ["a", "b", "c"]})
    right = pl.LazyFrame({"k": [2, 3, 4], "rv": ["x", "y", "z"]})
    df = left.join(right, on="k").sort("k").collect(engine=engine)
    assert df["k"].to_list() == [2, 3]
    assert df["lv"].to_list() == ["b", "c"]
    assert df["rv"].to_list() == ["x", "y"]


def test_sort_with_nulls(engine) -> None:  # type: ignore[no-untyped-def]
    lf = pl.LazyFrame({"a": [3, None, 1, 2]})
    df = lf.sort("a").collect(engine=engine)
    assert df["a"].to_list() == [None, 1, 2, 3]


def test_with_columns_arithmetic(engine) -> None:  # type: ignore[no-untyped-def]
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    df = lf.with_columns((pl.col("a") * 2).alias("doubled")).collect(engine=engine)
    assert df["doubled"].to_list() == [2, 4, 6]


def test_two_engines_match(engine) -> None:  # type: ignore[no-untyped-def]
    """Sanity: an arbitrary query produces identical output on cpu and metal."""
    lf = pl.LazyFrame({"a": [1, 2, 3, None, 5], "b": [10.0, None, 30.0, 40.0, 50.0]})
    query = lf.filter(pl.col("a").is_not_null()).group_by("a").agg(pl.col("b").sum()).sort("a")
    cpu = query.collect(engine="cpu")
    metal = query.collect(engine=engine)
    assert_frame_equal(metal, cpu)
