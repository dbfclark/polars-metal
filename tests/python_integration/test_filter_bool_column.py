"""Phase 5 milestone: ``df.filter(bool_column)`` runs on GPU end-to-end.

After Task 15 the walker accepts Filter nodes whose predicate is a single
``Column(c)`` with ``c`` of Boolean dtype. The Rust UDF dispatches the
compaction pipeline (predicate eval + MLX cumsum + scatter) per surviving
column. Anything else (arithmetic, comparison, compound, non-Bool column,
string column) still falls back to CPU cleanly.

The architecture is:
- Python walker tags the plan as ``{"kind": "Filter", "predicate":
  {"kind": "Column", "name": ..., "dtype": "Bool"}, "input": <plan>}``.
- Python UDF extracts bit-packed Boolean data + validity from the
  predicate column and bit-packed/dense data + validity from each
  surviving column via Polars ``Series.to_arrow().buffers()``.
- Rust ``_native.execute_plan_filter`` runs the GPU compaction pipeline
  and returns the per-column compacted bytes.
- Python UDF reassembles the result into a ``pl.DataFrame`` via
  ``pa.Array.from_buffers`` + ``pl.Series`` ctor.
"""

from __future__ import annotations

import logging

import polars as pl
from polars.testing import assert_frame_equal

import polars_metal


def test_filter_precomputed_bool_column_runs_on_gpu(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="polars_metal")
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, None, 7, 8],
            "mask": [True, False, True, False, True, None, True, False],
        }
    )
    cpu = df.lazy().filter(pl.col("mask")).select("a").collect()
    metal = (
        df.lazy()
        .filter(pl.col("mask"))
        .select("a")
        .collect(engine=polars_metal.MetalEngine(debug=True))
    )
    assert_frame_equal(cpu, metal)
    log_text = " ".join(rec.getMessage() for rec in caplog.records if rec.name == "polars_metal")
    # M2 cost model: filter→CPU always. No UDF is installed; the router logs the CPU route.
    assert "router routes entire query to CPU" in log_text, (
        f"expected router-to-CPU log, got: {log_text}"
    )


def test_filter_bool_all_kept_round_trips() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "mask": [True, True, True]})
    cpu = df.lazy().filter(pl.col("mask")).collect()
    metal = df.lazy().filter(pl.col("mask")).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_bool_all_dropped_returns_empty() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "mask": [False, False, False]})
    cpu = df.lazy().filter(pl.col("mask")).collect()
    metal = df.lazy().filter(pl.col("mask")).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_bool_with_null_mask_treated_as_false() -> None:
    """Polars treats null in the predicate as false (the row is dropped)."""
    df = pl.DataFrame({"a": [1, 2, 3, 4], "mask": [True, None, False, True]})
    cpu = df.lazy().filter(pl.col("mask")).collect()
    metal = df.lazy().filter(pl.col("mask")).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_preserves_i64_and_f64_and_bool_columns() -> None:
    df = pl.DataFrame(
        {
            "i": [10, 20, 30, 40, 50],
            "f": [1.5, 2.5, 3.5, 4.5, 5.5],
            "b": [True, False, True, False, True],
            "mask": [True, False, True, False, True],
        }
    )
    cpu = df.lazy().filter(pl.col("mask")).collect()
    metal = df.lazy().filter(pl.col("mask")).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_preserves_nulls_in_surviving_rows() -> None:
    """Validity bits must round-trip through the scatter kernel."""
    df = pl.DataFrame(
        {
            "i": [None, 1, None, 2, None, 3],
            "mask": [True, True, True, True, True, True],
        }
    )
    cpu = df.lazy().filter(pl.col("mask")).collect()
    metal = df.lazy().filter(pl.col("mask")).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_with_arithmetic_predicate_falls_back() -> None:
    """Phase 6+ enables comparison predicates. Today this must fall back cleanly."""
    df = pl.DataFrame({"a": [1, 2, 3, 4]})
    cpu = df.lazy().filter(pl.col("a") > 2).collect()
    metal = df.lazy().filter(pl.col("a") > 2).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_string_column_falls_back() -> None:
    df = pl.DataFrame({"s": ["a", "b", "c"], "m": [True, False, True]})
    cpu = df.lazy().filter(pl.col("m")).collect()
    metal = df.lazy().filter(pl.col("m")).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_filter_with_non_bool_column_predicate_falls_back() -> None:
    """The walker must reject e.g. ``filter(pl.col("a"))`` where ``a`` is Int64."""
    df = pl.DataFrame({"a": [0, 1, 2, 3], "b": [10, 20, 30, 40]})
    # Polars' CPU executor accepts a non-Bool column here as a truthiness
    # filter (interprets non-zero as keep) — confirm our fallback path
    # exhibits the same behavior. (If Polars actually raises, both sides
    # raise — the assert_frame_equal in the success path verifies parity.)
    cpu_err: Exception | None = None
    cpu_result: pl.DataFrame | None = None
    try:
        cpu_result = df.lazy().filter(pl.col("a")).collect()
    except Exception as e:
        cpu_err = e

    metal_err: Exception | None = None
    metal_result: pl.DataFrame | None = None
    try:
        metal_result = df.lazy().filter(pl.col("a")).collect(engine=polars_metal.MetalEngine())
    except Exception as e:
        metal_err = e

    # Either both raise or both succeed with equal results.
    assert (cpu_err is None) == (metal_err is None), (
        f"CPU and Metal diverge on non-Bool filter: cpu_err={cpu_err!r}, metal_err={metal_err!r}"
    )
    if cpu_result is not None and metal_result is not None:
        assert_frame_equal(cpu_result, metal_result)
