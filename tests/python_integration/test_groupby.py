# tests/python_integration/test_groupby.py
"""Engine-boundary tests for GroupBy.

Each test asserts byte-exact equality between engine="cpu" and
engine=MetalEngine(). Property-based coverage lives in
crates/polars-metal-kernels/tests/test_groupby_pipeline.rs.

Row counts are above the GROUPBY_GPU_MIN_ROWS threshold (100K) so the
router takes the GPU path. Composite-key shapes are chosen to fit
within the 128-bit encoder budget (1 i64 = 65 bits including null,
so 2 i64 keys = 130 bits don't fit; Bool + i64 = 67 bits fits; 2 Bool
= 4 bits fits trivially).
"""

from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_metal


def _engine() -> polars_metal.MetalEngine:
    return polars_metal.MetalEngine()


def test_groupby_single_i64_key_sum_matches_cpu() -> None:
    n = 200_000
    df = pl.DataFrame(
        {
            "k": [(i % 7) for i in range(n)],
            "v": list(range(n)),
        }
    ).with_columns([pl.col("k").cast(pl.Int64), pl.col("v").cast(pl.Int64)])
    q = df.lazy().group_by("k").agg(pl.col("v").sum().alias("s")).sort("k")
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=_engine())
    assert_frame_equal(cpu, metal)


def test_groupby_single_i64_key_all_aggs_matches_cpu() -> None:
    n = 200_000
    df = pl.DataFrame(
        {
            "k": [(i % 5) for i in range(n)],
            "v": [float(i % 100) for i in range(n)],
        }
    ).with_columns(pl.col("k").cast(pl.Int64))
    q = (
        df.lazy()
        .group_by("k")
        .agg(
            pl.col("v").sum().alias("s"),
            pl.col("v").mean().alias("m"),
            pl.col("v").min().alias("mn"),
            pl.col("v").max().alias("mx"),
            pl.col("v").count().alias("c"),
            pl.len().alias("n"),
        )
        .sort("k")
    )
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=_engine())
    assert_frame_equal(cpu, metal)


def test_groupby_two_bool_keys_matches_cpu() -> None:
    """Q1-shape composite key: two Bool keys (each {0, 1})."""
    n = 200_000
    df = pl.DataFrame(
        {
            "returnflag": [bool(i % 2) for i in range(n)],
            "linestatus": [bool((i // 2) % 2) for i in range(n)],
            "qty": list(range(n)),
        }
    ).with_columns(pl.col("qty").cast(pl.Int64))
    q = (
        df.lazy()
        .group_by("returnflag", "linestatus")
        .agg(
            pl.col("qty").sum().alias("sum_qty"),
        )
        .sort("returnflag", "linestatus")
    )
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=_engine())
    assert_frame_equal(cpu, metal)


def test_groupby_bool_plus_i64_key_matches_cpu() -> None:
    """Composite key: Bool + I64 = 67 bits, fits in 128-bit budget."""
    n = 200_000
    df = pl.DataFrame(
        {
            "category": [bool(i % 2) for i in range(n)],
            "subkey": [(i % 13) for i in range(n)],
            "v": [float(i) for i in range(n)],
        }
    ).with_columns(pl.col("subkey").cast(pl.Int64))
    q = (
        df.lazy()
        .group_by("category", "subkey")
        .agg(
            pl.col("v").sum().alias("sum_v"),
        )
        .sort("category", "subkey")
    )
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=_engine())
    assert_frame_equal(cpu, metal)


def test_groupby_null_in_key_becomes_its_own_group() -> None:
    n = 200_000
    df = pl.DataFrame(
        {
            "k": [1 if i % 3 != 0 else None for i in range(n)],
            "v": list(range(n)),
        }
    ).with_columns([pl.col("k").cast(pl.Int64), pl.col("v").cast(pl.Int64)])
    q = df.lazy().group_by("k").agg(pl.col("v").sum().alias("s")).sort("k")
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=_engine())
    assert_frame_equal(cpu, metal)


def test_groupby_null_in_value_skipped_by_agg_ops() -> None:
    n = 200_000
    df = pl.DataFrame(
        {
            "k": [(i % 4) for i in range(n)],
            "v": [None if i % 5 == 0 else float(i) for i in range(n)],
        }
    ).with_columns(pl.col("k").cast(pl.Int64))
    q = (
        df.lazy()
        .group_by("k")
        .agg(
            pl.col("v").sum().alias("s"),
            pl.col("v").mean().alias("m"),
            pl.col("v").count().alias("c"),
            pl.len().alias("n"),
        )
        .sort("k")
    )
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=_engine())
    assert_frame_equal(cpu, metal)


def test_groupby_all_unique_keys() -> None:
    n = 200_000
    df = pl.DataFrame(
        {
            "k": list(range(n)),
            "v": [1] * n,
        }
    ).with_columns([pl.col("k").cast(pl.Int64), pl.col("v").cast(pl.Int64)])
    q = df.lazy().group_by("k").agg(pl.col("v").sum().alias("s")).sort("k")
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=_engine())
    assert_frame_equal(cpu, metal)


def test_groupby_all_same_key() -> None:
    n = 200_000
    df = pl.DataFrame(
        {
            "k": [7] * n,
            "v": list(range(n)),
        }
    ).with_columns([pl.col("k").cast(pl.Int64), pl.col("v").cast(pl.Int64)])
    q = (
        df.lazy()
        .group_by("k")
        .agg(
            pl.col("v").sum().alias("s"),
            pl.col("v").count().alias("c"),
        )
        .sort("k")
    )
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=_engine())
    assert_frame_equal(cpu, metal)


@pytest.mark.xfail(
    reason="Accepted conformance deferral: mean() of an integer column returns Float32 "
    "on the Metal groupby path vs Float64 on Polars CPU (see m3-conformance-deferrals). "
    "The groupby kernel is conformance-only and not extended (per CLAUDE.md non-goals).",
    strict=True,
)
def test_groupby_i32_keys_i32_values_matches_cpu() -> None:
    """I32 key + I32 values — exercises the 32-bit GPU dispatcher path."""
    n = 200_000
    df = pl.DataFrame(
        {
            "k": [(i % 7) for i in range(n)],
            "v": [(i % 100) for i in range(n)],
        }
    ).with_columns([pl.col("k").cast(pl.Int32), pl.col("v").cast(pl.Int32)])
    q = (
        df.lazy()
        .group_by("k")
        .agg(
            pl.col("v").sum().alias("s"),
            pl.col("v").mean().alias("m"),
            pl.col("v").min().alias("mn"),
            pl.col("v").max().alias("mx"),
            pl.col("v").count().alias("c"),
            pl.len().alias("n"),
        )
        .sort("k")
    )
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=_engine())
    assert_frame_equal(cpu, metal)


def test_groupby_f32_values_matches_cpu() -> None:
    """I32 key + F32 values — exercises the 32-bit float GPU dispatcher path."""
    n = 200_000
    df = pl.DataFrame(
        {
            "k": [(i % 5) for i in range(n)],
            "v": [float(i % 100) for i in range(n)],
        }
    ).with_columns([pl.col("k").cast(pl.Int32), pl.col("v").cast(pl.Float32)])
    q = (
        df.lazy()
        .group_by("k")
        .agg(
            pl.col("v").sum().alias("s"),
            pl.col("v").mean().alias("m"),
        )
        .sort("k")
    )
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=_engine())
    assert_frame_equal(cpu, metal)
