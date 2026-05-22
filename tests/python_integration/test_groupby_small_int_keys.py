"""Verify groupby on i8/i16/i32/u8/u16/u32 key columns matches CPU byte-exact."""

import polars as pl
from polars.testing import assert_frame_equal

import polars_metal as pm


def _make_df(n=10_000, n_groups=4, dtype=pl.Int8):
    keys = pl.Series("k", [(i % n_groups) for i in range(n)], dtype=dtype)
    vals = pl.Series("v", [i * 1.5 for i in range(n)], dtype=pl.Float64)
    return pl.DataFrame([keys, vals])


def _check(df):
    q = df.lazy().group_by("k").agg(pl.col("v").sum(), pl.len())
    cpu = q.collect(engine="cpu").sort("k")
    metal = q.collect(engine=pm.MetalEngine()).sort("k")
    assert_frame_equal(cpu, metal)


def test_groupby_i8_key():
    _check(_make_df(dtype=pl.Int8))


def test_groupby_i16_key():
    _check(_make_df(dtype=pl.Int16))


def test_groupby_i32_key():
    # M2 already supports I32, but include here for regression coverage.
    _check(_make_df(dtype=pl.Int32))


def test_groupby_u8_key():
    _check(_make_df(dtype=pl.UInt8))


def test_groupby_u16_u32_mixed():
    """Multi-key composite with smaller integers fits in 128 bits."""
    n = 10_000
    k1 = pl.Series("k1", [(i % 4) for i in range(n)], dtype=pl.UInt16)
    k2 = pl.Series("k2", [(i % 8) for i in range(n)], dtype=pl.UInt32)
    v = pl.Series("v", [i * 1.5 for i in range(n)], dtype=pl.Float64)
    df = pl.DataFrame([k1, k2, v])
    q = df.lazy().group_by("k1", "k2").agg(pl.col("v").sum())
    assert_frame_equal(
        q.collect(engine="cpu").sort(["k1", "k2"]),
        q.collect(engine=pm.MetalEngine()).sort(["k1", "k2"]),
    )
