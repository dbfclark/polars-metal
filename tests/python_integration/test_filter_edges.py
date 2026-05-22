"""Filter edge-case tests, migrated from `tests/diff/test_filter_edges.py`.

Lives in `tests/python_integration/` per the M2 testing taxonomy: explicit
Python cases for engine-boundary correctness live here; property-based
testing lives in `crates/polars-metal-kernels/tests/`.

See `docs/superpowers/specs/2026-05-21-m2-design.md` § "Testing strategy"
for the full taxonomy.
"""

from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal

import polars_metal


def _engine() -> polars_metal.MetalEngine:
    return polars_metal.MetalEngine()


def test_empty_dataframe() -> None:
    """Zero rows with a numeric predicate. Schema and row count both must round-trip."""
    df = pl.DataFrame({"a": pl.Series("a", [], dtype=pl.Int64)})
    pred = pl.col("a") > 0
    cpu = df.lazy().filter(pred).collect()
    metal = df.lazy().filter(pred).collect(engine=_engine())
    assert_frame_equal(cpu, metal)


def test_single_row_predicate_true() -> None:
    df = pl.DataFrame({"a": [5], "mask": [True]})
    cpu = df.lazy().filter(pl.col("mask")).collect()
    metal = df.lazy().filter(pl.col("mask")).collect(engine=_engine())
    assert_frame_equal(cpu, metal)


def test_single_row_predicate_false() -> None:
    df = pl.DataFrame({"a": [5], "mask": [False]})
    cpu = df.lazy().filter(pl.col("mask")).collect()
    metal = df.lazy().filter(pl.col("mask")).collect(engine=_engine())
    assert_frame_equal(cpu, metal)


def test_single_row_predicate_null() -> None:
    """Polars treats null in the predicate as 'do not keep'."""
    df = pl.DataFrame({"a": [5], "mask": pl.Series("mask", [None], dtype=pl.Boolean)})
    cpu = df.lazy().filter(pl.col("mask")).collect()
    metal = df.lazy().filter(pl.col("mask")).collect(engine=_engine())
    assert_frame_equal(cpu, metal)


def test_all_null_column() -> None:
    """Filtering on ``col > 0`` when ``col`` is all-null yields 0 rows."""
    df = pl.DataFrame({"a": pl.Series("a", [None, None, None, None], dtype=pl.Int64)})
    pred = pl.col("a") > 0
    cpu = df.lazy().filter(pred).collect()
    metal = df.lazy().filter(pred).collect(engine=_engine())
    assert_frame_equal(cpu, metal)


def test_predicate_all_true() -> None:
    """Predicate is always True (via a tautological comparison). Result == input."""
    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    # ``a == a`` is True for every non-null row; here we have no nulls.
    pred = pl.col("a") == pl.col("a")
    cpu = df.lazy().filter(pred).collect()
    metal = df.lazy().filter(pred).collect(engine=_engine())
    assert_frame_equal(cpu, metal)


def test_predicate_all_false() -> None:
    """Predicate is always False (via an impossible comparison). Result is empty with same schema."""
    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    pred = pl.col("a") != pl.col("a")  # always false on non-null inputs
    cpu = df.lazy().filter(pred).collect()
    metal = df.lazy().filter(pred).collect(engine=_engine())
    assert_frame_equal(cpu, metal)


def test_predicate_all_null() -> None:
    """Bool column of all nulls: Polars treats null-in-predicate as drop → 0 rows."""
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "mask": pl.Series("mask", [None, None, None, None], dtype=pl.Boolean),
        }
    )
    cpu = df.lazy().filter(pl.col("mask")).collect()
    metal = df.lazy().filter(pl.col("mask")).collect(engine=_engine())
    assert_frame_equal(cpu, metal)


def test_predicate_yields_empty_subset() -> None:
    """Non-degenerate predicate that happens to match no rows."""
    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    pred = pl.col("a") > 1000  # no row qualifies
    cpu = df.lazy().filter(pred).collect()
    metal = df.lazy().filter(pred).collect(engine=_engine())
    assert_frame_equal(cpu, metal)


def test_finite_f64_predicate_works() -> None:
    """Sanity: finite f64 comparisons round-trip cleanly (the strategy stays here)."""
    df = pl.DataFrame({"x": [-1.5, 0.0, 1.5, 2.5]})
    pred = pl.col("x") > 0.0
    cpu = df.lazy().filter(pred).collect()
    metal = df.lazy().filter(pred).collect(engine=_engine())
    assert_frame_equal(cpu, metal)
