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
