"""When the walker encounters a multi-chunk Series in the input frame, it
returns a clean FallBack reason. The query still produces the correct result
via Polars' CPU executor — this test asserts both behaviors.

T31: Defensive fallback for multi-chunk Series. The Polars optimizer can
sometimes produce multi-chunk DataFrames (e.g. when concat with rechunk=False
precedes a lazy operation). M2 does not yet handle this, so the walker detects
it at plan time and returns Fallback. The entire query routes to CPU, avoiding
dispatch-time errors.
"""

from __future__ import annotations

import logging

import polars as pl
from polars.testing import assert_frame_equal

import polars_metal


def _make_multichunk_df() -> pl.DataFrame:
    """Create a multi-chunk DataFrame by concatenating without rechunking."""
    a = pl.DataFrame({"k": [1, 2, 3], "v": [10, 20, 30]})
    b = pl.DataFrame({"k": [4, 5], "v": [40, 50]})
    df = pl.concat([a, b], rechunk=False)
    # Sanity: confirm chunks > 1 on at least one column.
    assert df["v"].n_chunks() > 1
    return df


def test_multichunk_groupby_falls_back_cleanly() -> None:
    """Multi-chunk input to GroupBy routes to CPU and produces correct result."""
    df = _make_multichunk_df()
    q = df.lazy().group_by("k").agg(pl.col("v").sum().alias("s")).sort("k")
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_multichunk_filter_falls_back_cleanly() -> None:
    """Multi-chunk input to Filter routes to CPU and produces correct result."""
    df = _make_multichunk_df()
    q = df.lazy().filter(pl.col("v") > 25).sort("k")
    cpu = q.collect(engine="cpu")
    metal = q.collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_multichunk_fallback_reason_in_debug_log(caplog) -> None:
    """The walker logs the multi-chunk fallback reason at debug level."""
    caplog.set_level(logging.DEBUG, logger="polars_metal")
    df = _make_multichunk_df()
    q = df.lazy().group_by("k").agg(pl.col("v").sum())
    q.collect(engine=polars_metal.MetalEngine(debug=True))
    log_text = " ".join(r.getMessage() for r in caplog.records if r.name == "polars_metal")
    assert "multi-chunk" in log_text or "chunk" in log_text, (
        f"expected multi-chunk fallback reason in debug log; got:\n{log_text}"
    )
