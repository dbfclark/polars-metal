"""Phase 9: multi-chunk Series no longer cause walker fallback.

`_materialize_arrow` calls `combine_chunks()` before extracting buffer
bytes, so the kernel-layer sees a contiguous Arrow array regardless of
how many chunks the upstream Series has. The walker used to defensively
fall back on `n_chunks() > 1`; that guard is removed in Phase 9.

These tests cover the two common multi-chunk shapes:
  - `pl.concat([df_a, df_b], rechunk=False)` — two-chunk fixture
  - 1000-piece concat — many-chunk pathological case
"""

import polars as pl
from polars.testing import assert_frame_equal

import polars_metal


def test_concat_two_dataframes_groupby() -> None:
    """LazyFrame from `pl.concat(..., rechunk=False)` has multi-chunk Series."""
    a = pl.DataFrame({"k": [0, 0, 1, 1], "v": [1.0, 2.0, 3.0, 4.0]})
    b = pl.DataFrame({"k": [0, 1, 0, 1], "v": [5.0, 6.0, 7.0, 8.0]})
    combined = pl.concat([a, b], rechunk=False)
    assert combined["k"].n_chunks() > 1, "fixture should be multi-chunk"

    q = combined.lazy().group_by("k").agg(pl.col("v").sum().alias("s"))
    cpu = q.collect(engine="cpu").sort("k")
    metal = q.collect(engine=polars_metal.MetalEngine()).sort("k")
    assert_frame_equal(cpu, metal)


def test_many_small_chunks_groupby() -> None:
    """1000-piece concat exercises the many-chunk path through combine_chunks."""
    parts = [pl.DataFrame({"k": [i % 4], "v": [float(i)]}) for i in range(1000)]
    combined = pl.concat(parts, rechunk=False)
    assert combined["k"].n_chunks() > 100, "fixture should have many chunks"

    q = combined.lazy().group_by("k").agg(pl.col("v").sum().alias("s"))
    cpu = q.collect(engine="cpu").sort("k")
    metal = q.collect(engine=polars_metal.MetalEngine()).sort("k")
    assert_frame_equal(cpu, metal)


def test_multichunk_string_keys_groupby() -> None:
    """Phase 7's Utf8 path + Phase 9's multi-chunk handling combined."""
    a = pl.DataFrame({"k": ["A", "A", "B"], "v": [1.0, 2.0, 3.0]})
    b = pl.DataFrame({"k": ["B", "C", "A"], "v": [4.0, 5.0, 6.0]})
    combined = pl.concat([a, b], rechunk=False)
    assert combined["k"].n_chunks() > 1

    q = combined.lazy().group_by("k").agg(pl.col("v").sum().alias("s"))
    cpu = q.collect(engine="cpu").sort("k")
    metal = q.collect(engine=polars_metal.MetalEngine()).sort("k")
    assert_frame_equal(cpu, metal)


def test_multichunk_with_filter() -> None:
    """Filter + groupby on multi-chunk source."""
    a = pl.DataFrame({"k": [0, 1, 2, 0], "v": [10.0, 20.0, 30.0, 40.0]})
    b = pl.DataFrame({"k": [1, 2, 0, 1], "v": [50.0, 60.0, 70.0, 80.0]})
    combined = pl.concat([a, b], rechunk=False)
    assert combined["k"].n_chunks() > 1

    q = combined.lazy().filter(pl.col("v") > 25.0).group_by("k").agg(pl.col("v").sum().alias("s"))
    cpu = q.collect(engine="cpu").sort("k")
    metal = q.collect(engine=polars_metal.MetalEngine()).sort("k")
    assert_frame_equal(cpu, metal)
