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

import polars as pl

from polars_metal import MetalEngine


def test_int64_add_one_roundtrips():
    df = pl.DataFrame({"x": pl.Series([1, 2, 3, 1_000_000_000_000], dtype=pl.Int64)})
    got = df.lazy().with_columns((pl.col("x") + 1).alias("y")).collect(engine=MetalEngine())
    want = df.lazy().with_columns((pl.col("x") + 1).alias("y")).collect()
    assert got.equals(want)
    assert got["y"].dtype == pl.Int64
