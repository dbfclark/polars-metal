"""B1 integer-foundation: a monomorphic-int fused chain (`col + 1` over an
integer column) must round-trip end-to-end through ``engine="metal"`` with the
exact Polars output dtype.

This is the B1 exit bar: the dtype threads through the full PyO3 + Python
boundary (analyzer infers the output dtype statically, the literal is staged at
the column width so MLX doesn't promote to f32, and the typed FFI writes the
right-width output). Task 7 generalizes the original Int64 smoke test to the
full dtype x null differential matrix and adds a drift-guard asserting the three
Python dtype-tag tables agree with the canonical Rust ``MlxDtype`` discriminants.
"""

from __future__ import annotations

import warnings

import polars as pl
import pytest

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


# ── B1 exit-bar differential matrix ─────────────────────────────────────────
#
# `col + 1` over each integer family, against the Polars CPU oracle, byte-exact.
# Values are chosen so `+ 1` stays in range (no overflow) and so the engine path
# is exercised at boundary-ish magnitudes (UInt64 beyond i64 range, Int8 near but
# not at the 127 ceiling). Overflow-domain semantics are B2 — not tested here.
_CASES = [
    (pl.Int8, [1, 2, 3, 100, -5, 126]),  # +1 stays in range (127 max)
    (pl.Int32, [-7, 0, 1, 100, 2_000_000_000]),
    (pl.Int64, [1, 2, 3, 3_000_000_000, -2_000_000_000]),
    (pl.UInt64, [0, 1, 5, 10_000_000_000_000_000_000]),  # beyond i64 range
]


def _run(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    lf = df.lazy().with_columns((pl.col("x") + 1).alias("y"))
    return lf.collect(engine=MetalEngine()), lf.collect()


@pytest.mark.parametrize("dtype,values", _CASES)
def test_add_one_roundtrip_no_nulls(dtype, values):
    df = pl.DataFrame({"x": pl.Series(values, dtype=dtype)})
    got, want = _run(df)
    assert got.equals(want), f"{dtype}: {got} != {want}"
    assert got["y"].dtype == dtype


@pytest.mark.parametrize("dtype,values", _CASES)
def test_add_one_roundtrip_with_nulls(dtype, values):
    values_with_null = [*values[:1], None, *values[1:]]
    df = pl.DataFrame({"x": pl.Series(values_with_null, dtype=dtype)})
    got, want = _run(df)
    assert got.equals(want), f"{dtype} (nulls): {got} != {want}"
    assert got["y"].dtype == dtype
    assert got["y"].is_null().to_list() == want["y"].is_null().to_list()


# ── dtype-tag drift guard ───────────────────────────────────────────────────


def test_dtype_tag_tables_match_canonical_mlx_dtype():
    """Guard against drift between the Python dtype-tag tables and the canonical
    Rust ``MlxDtype`` discriminants (``crates/polars-metal-mlx-sys/src/array.rs``:
    F32=0, F64=1, I32=2, Bool=3, I8=4, I16=5, I64=6, U8=7, U16=8, U32=9, U64=10).

    Three Python tables duplicate slices of that mapping, each keyed differently:
      - ``_udf._DTYPE_STR_TO_NP_AND_TAG`` (and its ``_np_dtype_and_tag`` helper):
        wire str -> (numpy dtype name, u32 tag). Carries the actual tag values.
      - ``_fusion_analyzer._INT_DTYPE_TO_STR``: Polars dtype -> wire str.
      - ``_walker._INT_TAG_TO_POLARS``: wire str -> Polars dtype string.

    If a dtype is added/renumbered in Rust, these tables must be updated to
    match — this test makes that loud, and also asserts the three tables are
    self-consistent (so an edit to one without the others fails here).
    """
    from polars_metal._fusion_analyzer import _INT_DTYPE_TO_STR
    from polars_metal._udf import _DTYPE_STR_TO_NP_AND_TAG, _np_dtype_and_tag
    from polars_metal._walker import _INT_TAG_TO_POLARS

    # Canonical MlxDtype u32 tags (array.rs). Integer dtypes only (what B1 added).
    canonical_int_tags = {
        "I8": 4,
        "I16": 5,
        "I32": 2,
        "I64": 6,
        "U8": 7,
        "U16": 8,
        "U32": 9,
        "U64": 10,
    }
    # The Polars dtype each wire string denotes (for cross-table consistency).
    wire_to_pl_dtype = {
        "I8": pl.Int8,
        "I16": pl.Int16,
        "I32": pl.Int32,
        "I64": pl.Int64,
        "U8": pl.UInt8,
        "U16": pl.UInt16,
        "U32": pl.UInt32,
        "U64": pl.UInt64,
    }

    # 1. _udf: the str->tag table (via the public helper) must equal the canon.
    for wire, tag in canonical_int_tags.items():
        _np, got_tag = _np_dtype_and_tag(wire)
        assert got_tag == tag, f"_np_dtype_and_tag({wire!r}) tag {got_tag} != canonical {tag}"
        # And the raw table entry agrees (helper is a thin wrapper, but assert
        # both so a future helper change can't silently mask a table desync).
        assert _DTYPE_STR_TO_NP_AND_TAG[wire][1] == tag

    # 2. _fusion_analyzer: Polars-dtype -> wire-str must cover every int family
    #    and map to the canonical wire string.
    analyzer_pl_to_wire = {pl_dt: wire for pl_dt, wire in _INT_DTYPE_TO_STR.items()}
    for wire, pl_dt in wire_to_pl_dtype.items():
        assert analyzer_pl_to_wire.get(pl_dt) == wire, (
            f"_INT_DTYPE_TO_STR[{pl_dt}] = {analyzer_pl_to_wire.get(pl_dt)!r} != {wire!r}"
        )
    # No extra/missing entries vs the canonical int set.
    assert set(_INT_DTYPE_TO_STR.values()) == set(canonical_int_tags)

    # 3. _walker: wire-str -> Polars-dtype-string must round-trip to the same
    #    Polars dtype each wire string denotes.
    for wire, pl_dt in wire_to_pl_dtype.items():
        assert _INT_TAG_TO_POLARS[wire] == str(pl_dt), (
            f"_INT_TAG_TO_POLARS[{wire!r}] = {_INT_TAG_TO_POLARS[wire]!r} != {str(pl_dt)!r}"
        )
    # Same key set as the canonical int tags (no drift in coverage).
    assert set(_INT_TAG_TO_POLARS) == set(canonical_int_tags)
