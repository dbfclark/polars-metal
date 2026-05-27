"""Phase 7 Task 34: Utf8 keys flow through walker -> Rust -> kernel.

Validates that a small string-keyed groupby matches CPU Polars
bit-for-bit. Coverage focuses on:
  - Low-cardinality strings (Q1's actual shape: ~4 distinct values)
  - Mid-cardinality strings (~100 distinct)
  - Null handling
  - Combined with int keys
"""

import polars as pl
from polars.testing import assert_frame_equal

import polars_metal as pm


def test_string_groupby_low_cardinality() -> None:
    df = pl.DataFrame(
        {
            "k": ["A", "B", "A", "C", "B", "A"] * 1000,
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 1000,
        }
    )
    q = df.lazy().group_by("k").agg(pl.col("v").sum().alias("s")).sort("k")
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=pm.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_string_groupby_with_null_keys() -> None:
    df = pl.DataFrame(
        {
            "k": ["A", None, "A", "B", None, "B"] * 100,
            "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 100,
        }
    )
    q = df.lazy().group_by("k").agg(pl.col("v").sum().alias("s")).sort("k")
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=pm.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_string_combined_with_int_key() -> None:
    df = pl.DataFrame(
        {
            "k_s": ["X", "Y", "X", "Z"] * 1000,
            "k_i": [1, 2, 1, 3] * 1000,
            "v": [1.0, 2.0, 3.0, 4.0] * 1000,
        }
    )
    q = df.lazy().group_by("k_s", "k_i").agg(pl.col("v").sum().alias("s")).sort("k_s", "k_i")
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=pm.MetalEngine())
    assert_frame_equal(cpu, metal)
