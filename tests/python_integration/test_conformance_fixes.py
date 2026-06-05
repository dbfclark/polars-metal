"""Regression tests for the M6 conformance fixes (surgical CPU fallbacks).

These cover two pre-existing engine defects surfaced by the Polars-own-suite
conformance run and fixed surgically:

1. A non-fused ``HStack`` (a ``with_columns`` whose appended columns the F32
   fusion analyzer rejects, e.g. integer arithmetic) crashed with
   ``ValueError: unknown plan kind "HStack"`` because Rust's ``execute_plan``
   has no HStack handler and the only executable HStack path is the all-fused
   fast path. Fix: the walker falls back to CPU at plan time when any binding
   is non-fused.

2. Group-by ``mean``/``median``/``quantile`` over an integer column returned
   ``Float32`` where Polars returns ``Float64`` (the documented F32→F64
   divergence). Fix: cast those agg outputs to F64 to match Polars dtype.
"""

from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

from polars_metal import MetalEngine


def test_nonfused_hstack_falls_back_to_cpu() -> None:
    # i64 arithmetic in with_columns is not F32-fusible → non-fused HStack.
    # Must fall back to CPU (correct result), not raise "unknown plan kind HStack".
    df = pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    lf = df.lazy().with_columns((pl.col("a") * 2).alias("c"))
    got = lf.collect(engine=MetalEngine())
    expected = pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30], "c": [2, 4, 6]})
    assert_frame_equal(got, expected)


def test_mixed_fused_and_nonfused_hstack_falls_back() -> None:
    # One F32-fusible column + one integer (non-fusible) column in the same
    # with_columns → mixed HStack, which the all-or-nothing fused dispatch
    # cannot run. Must fall back to CPU, not crash.
    df = pl.DataFrame(
        {"x": [1.0, 2.0, 3.0], "k": [1, 2, 3]}, schema={"x": pl.Float32, "k": pl.Int64}
    )
    lf = df.lazy().with_columns(
        (pl.col("x") * 2.0).alias("x2"),  # F32-fusible
        (pl.col("k") + 1).alias("k1"),  # integer, non-fusible
    )
    got = lf.collect(engine=MetalEngine())
    expected = lf.collect(engine="cpu")
    assert_frame_equal(got, expected)


def test_f64_compute_not_downcast_to_f32() -> None:
    # The fused path computes in F32 (no GPU F64). An F64 column must NOT be
    # silently downcast — the analyzer rejects F64 inputs so the chain falls
    # back to CPU, preserving Polars' Float64 dtype and precision.
    df = pl.DataFrame({"foo": [0.5, 1.7, 3.2]}, schema={"foo": pl.Float64})
    got = df.lazy().with_columns((pl.col("foo") * 2.0).alias("bar")).collect(engine=MetalEngine())
    assert got.schema["bar"] == pl.Float64
    assert got.schema["foo"] == pl.Float64
    assert got["bar"].to_list() == [1.0, 3.4, 6.4]
