"""B1 integer-foundation: a monomorphic-int fused chain (`col + 1` over an
integer column) must round-trip end-to-end through ``engine="metal"`` with the
exact Polars output dtype.

This is the B1 exit-bar smoke test: the dtype threads through the full PyO3 +
Python boundary (analyzer infers the output dtype statically, the literal is
staged at the column width so MLX doesn't promote to f32, and the typed FFI
writes the right-width output). The full Int8/Int32/Int64/UInt64 x nulls matrix
is Task 7; here we only assert the one Int64 path and rely on the existing F32
suites for no-regression.
"""

from __future__ import annotations

import warnings

import polars as pl

from polars_metal import MetalEngine


def test_int64_add_one_roundtrips():
    df = pl.DataFrame({"x": pl.Series([1, 2, 3, 1_000_000_000_000], dtype=pl.Int64)})
    got = df.lazy().with_columns((pl.col("x") + 1).alias("y")).collect(engine=MetalEngine())
    want = df.lazy().with_columns((pl.col("x") + 1).alias("y")).collect()
    assert got.equals(want)
    assert got["y"].dtype == pl.Int64


def test_nullable_int64_add_one_preserves_nulls_no_warning():
    """B1 T6 regression: a nullable Int64 fused chain must (1) restore nulls in
    the output and (2) NOT emit a NaN->int cast RuntimeWarning during staging
    (the conformance suite runs `filterwarnings = error`, which made that
    warning a hard failure in 4 `operations_group_by` tests).
    """
    df = pl.DataFrame({"x": pl.Series([1, None, 3, None, 5], dtype=pl.Int64)})
    expr = (pl.col("x") + 1).alias("y")

    # Any RuntimeWarning ("invalid value encountered in cast") becomes an error,
    # so a regression surfaces here rather than silently.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        got = df.lazy().with_columns(expr).collect(engine=MetalEngine())

    want = df.lazy().with_columns(expr).collect()
    assert got.equals(want)  # [2, None, 4, None, 6], nulls preserved
    assert got["y"].dtype == pl.Int64
    assert got["y"].to_list() == [2, None, 4, None, 6]
