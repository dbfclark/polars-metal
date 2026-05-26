"""Verify import-time warmup runs and first query is fast.

The warmup should pre-compile common fused-agg signatures, so the
first user query with a common shape doesn't pay MSL compile (~100-300ms).
This test verifies warmup runs without error; precise timing is in
the bench suite.
"""

import time

import polars as pl

import polars_metal as pm


def test_warmup_runs_without_error() -> None:
    # If we got here, import succeeded — warmup ran.
    from polars_metal._native import warmup_common_fused_signatures

    count = warmup_common_fused_signatures()
    # Re-running is idempotent and cheap (cache hits).
    assert count >= 3, f"expected >=3 signatures warmed, got {count}"


def test_first_query_after_warmup_is_fast() -> None:
    # Common shape: Sum over F32. Warmup should have pre-compiled this.
    df = pl.DataFrame(
        {
            "k": pl.Series([0, 0, 1, 1, 2, 2] * 1000, dtype=pl.Int32),
            "v": pl.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 1000, dtype=pl.Float32),
        }
    )
    t0 = time.perf_counter()
    result = df.lazy().group_by("k").agg(pl.col("v").sum()).collect(engine=pm.MetalEngine())
    dt_ms = (time.perf_counter() - t0) * 1000
    # Without warmup, first F32 Sum query pays ~100-300ms MSL compile.
    # With warmup, expected <50ms (full query including encode/dispatch/finalize).
    # Allow generous headroom; we're only testing the warmup ran, not perf.
    assert dt_ms < 250, f"first query took {dt_ms:.1f}ms — likely MSL compile not amortized"
    assert result.height == 3
