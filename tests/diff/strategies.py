"""Hypothesis strategies that produce small Polars LazyFrames for differential testing."""

from __future__ import annotations

import polars as pl
from hypothesis import strategies as st


def _ints(min_size: int = 0, max_size: int = 32) -> st.SearchStrategy[list[int]]:
    return st.lists(
        st.integers(min_value=-1_000, max_value=1_000),
        min_size=min_size,
        max_size=max_size,
    )


def _strs(min_size: int = 0, max_size: int = 32) -> st.SearchStrategy[list[str]]:
    return st.lists(
        st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=8),
        min_size=min_size,
        max_size=max_size,
    )


def _ints_with_nulls(max_size: int = 32) -> st.SearchStrategy[list[int | None]]:
    return st.lists(
        st.one_of(st.none(), st.integers(min_value=-1_000, max_value=1_000)),
        max_size=max_size,
    )


@st.composite
def numeric_frame(draw):  # type: ignore[no-untyped-def]
    n = draw(st.integers(min_value=0, max_value=32))
    a = draw(_ints(min_size=n, max_size=n))
    b = draw(_ints(min_size=n, max_size=n))
    return pl.LazyFrame({"a": a, "b": b})


@st.composite
def string_frame(draw):  # type: ignore[no-untyped-def]
    n = draw(st.integers(min_value=0, max_value=16))
    a = draw(_strs(min_size=n, max_size=n))
    return pl.LazyFrame({"a": a})


@st.composite
def null_heavy_frame(draw):  # type: ignore[no-untyped-def]
    a = draw(_ints_with_nulls(max_size=32))
    return pl.LazyFrame({"a": a})


# ---------------------------------------------------------------------------
# M1 strategies — DataFrames + predicates + projections in the closed M1 set.
# See ``python/polars_metal/_walker.py`` for the closed predicate set this
# mirrors. Important: NaN is intentionally excluded from f64 columns and
# literals because cmp_f64 is IEEE 754 while Polars CPU is TotalOrd; the
# divergence is tracked in ``docs/open-questions.md`` and pinned by the
# xfail-strict in ``tests/python_integration/test_filter_comparison.py``.
# ---------------------------------------------------------------------------


_M1_DTYPES = ("i64", "f64", "bool")
# Bias toward 0% nulls (no validity bitmap at all) and include 100% nulls
# (all-null bitmap) — both stress edge cases on the bit-packing path.
_NULL_DENSITIES = (0.0, 0.0, 0.3, 0.7, 1.0)


def _maybe_null(draw, value_strategy, null_density: float):
    """Draw a value or ``None`` with probability ~``null_density``.

    Density 0.0 and 1.0 short-circuit so callers get the "no validity
    bitmap" and "all-null bitmap" edge cases they rely on. The mixed
    branch routes the non-null path through ``st.one_of(st.none(),
    value_strategy)`` so Hypothesis sees ``None`` as a first-class shrink
    target on every row, not just rows that hit the density threshold.
    """
    if null_density >= 1.0:
        return None
    if null_density <= 0.0:
        return draw(value_strategy)
    if draw(st.floats(min_value=0.0, max_value=1.0)) < null_density:
        return None
    return draw(st.one_of(st.none(), value_strategy))


def _gen_i64_column(draw, n: int, null_density: float) -> list[int | None]:
    value_strategy = st.integers(min_value=-1_000, max_value=1_000)
    return [_maybe_null(draw, value_strategy, null_density) for _ in range(n)]


def _gen_f64_column(draw, n: int, null_density: float) -> list[float | None]:
    value_strategy = st.floats(
        min_value=-1_000.0,
        max_value=1_000.0,
        allow_nan=False,
        allow_infinity=False,
    )
    return [_maybe_null(draw, value_strategy, null_density) for _ in range(n)]


def _gen_bool_column(draw, n: int, null_density: float) -> list[bool | None]:
    return [_maybe_null(draw, st.booleans(), null_density) for _ in range(n)]


@st.composite
def m1_null_density_dataframe(draw):  # type: ignore[no-untyped-def]
    """DataFrame with i64/f64/bool columns; null density biased toward
    0% and 100% (most likely to expose bit-packing bugs).

    f64 columns contain only finite values (no NaN, no Inf) — see the
    cmp_f64 IEEE-754-vs-TotalOrd entry in docs/open-questions.md.
    """
    n_rows = draw(st.integers(min_value=0, max_value=1000))
    n_cols = draw(st.integers(min_value=1, max_value=4))
    cols: dict[str, pl.Series] = {}
    for i in range(n_cols):
        name = f"c{i}"
        dtype = draw(st.sampled_from(_M1_DTYPES))
        null_density = draw(st.sampled_from(_NULL_DENSITIES))
        if dtype == "i64":
            values = _gen_i64_column(draw, n_rows, null_density)
            cols[name] = pl.Series(name, values, dtype=pl.Int64)
        elif dtype == "f64":
            values = _gen_f64_column(draw, n_rows, null_density)
            cols[name] = pl.Series(name, values, dtype=pl.Float64)
        else:  # bool
            values = _gen_bool_column(draw, n_rows, null_density)
            cols[name] = pl.Series(name, values, dtype=pl.Boolean)
    return pl.DataFrame(cols)


def _cols_of(schema: dict[str, pl.DataType], *dtypes: pl.DataType) -> list[str]:
    return [name for name, dt in schema.items() if dt in dtypes]


@st.composite
def m1_predicate_expr(draw, schema):  # type: ignore[no-untyped-def]
    """Generate predicates from the closed M1 set, depth <= 3.

    Shapes produced:
      - ``pl.col(b)`` for a Boolean column ``b``.
      - ``pl.col(c) <op> <lit>`` where ``<op>`` in {==,!=,<,<=,>,>=} and
        the column/literal are the same numeric dtype (i64/f64).
      - ``pl.col(x) <op> pl.col(y)`` for two same-dtype numeric columns.
      - ``lhs & rhs`` / ``lhs | rhs`` where lhs/rhs are any of the above
        (recursive, depth-limited).
    """
    bool_cols = _cols_of(schema, pl.Boolean)
    i64_cols = _cols_of(schema, pl.Int64)
    f64_cols = _cols_of(schema, pl.Float64)
    expr = draw(_m1_predicate_at_depth(bool_cols, i64_cols, f64_cols, depth=3))
    return expr


@st.composite
def _m1_predicate_at_depth(  # type: ignore[no-untyped-def]
    draw,
    bool_cols: list[str],
    i64_cols: list[str],
    f64_cols: list[str],
    depth: int,
):
    """Recursive predicate generator. ``depth`` is the remaining nesting
    budget for AND/OR combinators; leaves have ``depth=0``.
    """
    leaf_choices: list[str] = []
    if bool_cols:
        leaf_choices.append("col_bool")
    if i64_cols or f64_cols:
        leaf_choices.append("compare")

    if depth == 0:
        kind = draw(st.sampled_from(leaf_choices))
    else:
        choices = [*leaf_choices, "and", "or"]
        kind = draw(st.sampled_from(choices))

    if kind == "col_bool":
        return pl.col(draw(st.sampled_from(bool_cols)))

    if kind == "compare":
        return draw(_m1_compare(i64_cols, f64_cols))

    # AND/OR
    lhs = draw(_m1_predicate_at_depth(bool_cols, i64_cols, f64_cols, depth - 1))
    rhs = draw(_m1_predicate_at_depth(bool_cols, i64_cols, f64_cols, depth - 1))
    if kind == "and":
        return lhs & rhs
    return lhs | rhs


@st.composite
def _m1_compare(draw, i64_cols: list[str], f64_cols: list[str]):  # type: ignore[no-untyped-def]
    """Generate a comparison expression: ``col <op> lit``, ``lit <op> col``,
    or ``col <op> col`` (operands always same dtype).

    Per walker contract: both sides same dtype, at least one side a Column,
    op in {==,!=,<,<=,>,>=}. The col/lit ordering is randomised to cover the
    walker's symmetric handling of lhs/rhs.
    """
    dtype_choices: list[str] = []
    if i64_cols:
        dtype_choices.append("i64")
    if f64_cols:
        dtype_choices.append("f64")
    dtype = draw(st.sampled_from(dtype_choices))

    op = draw(st.sampled_from(("eq", "ne", "lt", "le", "gt", "ge")))

    cols = i64_cols if dtype == "i64" else f64_cols
    # Operand shapes: "col_lit" (col on left), "lit_col" (lit on left),
    # "col_col" (two cols — only possible if we have >=2 of this dtype).
    shape_choices = ["col_lit", "lit_col"]
    if len(cols) >= 2:
        shape_choices.append("col_col")
    shape = draw(st.sampled_from(shape_choices))

    def gen_lit():
        if dtype == "i64":
            return pl.lit(draw(st.integers(min_value=-100, max_value=100)))
        return pl.lit(
            draw(
                st.floats(
                    min_value=-100.0,
                    max_value=100.0,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
        )

    if shape == "col_lit":
        lhs_expr = pl.col(draw(st.sampled_from(cols)))
        rhs_expr = gen_lit()
    elif shape == "lit_col":
        lhs_expr = gen_lit()
        rhs_expr = pl.col(draw(st.sampled_from(cols)))
    else:  # col_col
        lhs_expr = pl.col(draw(st.sampled_from(cols)))
        rhs_expr = pl.col(draw(st.sampled_from(cols)))

    return _apply_cmp(op, lhs_expr, rhs_expr)


def _apply_cmp(op: str, lhs, rhs):
    if op == "eq":
        return lhs == rhs
    if op == "ne":
        return lhs != rhs
    if op == "lt":
        return lhs < rhs
    if op == "le":
        return lhs <= rhs
    if op == "gt":
        return lhs > rhs
    if op == "ge":
        return lhs >= rhs
    raise AssertionError(f"unknown op {op!r}")


@st.composite
def m1_projection_subset(draw, schema):  # type: ignore[no-untyped-def]
    """Generate a random non-empty subset of columns in a random order."""
    names = list(schema.keys())
    k = draw(st.integers(min_value=1, max_value=len(names)))
    perm = draw(st.permutations(names))
    return perm[:k]
