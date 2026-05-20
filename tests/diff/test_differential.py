"""Differential properties: df.collect(engine=MetalEngine()) matches CPU."""

from hypothesis import given, settings

import polars_metal
from tests.diff.strategies import null_heavy_frame, numeric_frame, string_frame


@given(numeric_frame())
@settings(max_examples=100, deadline=None)
def test_numeric_collect_matches_cpu(lf) -> None:  # type: ignore[no-untyped-def]
    cpu = lf.collect()
    metal = lf.collect(engine=polars_metal.MetalEngine())
    assert metal.equals(cpu)


@given(string_frame())
@settings(max_examples=50, deadline=None)
def test_string_collect_matches_cpu(lf) -> None:  # type: ignore[no-untyped-def]
    cpu = lf.collect()
    metal = lf.collect(engine=polars_metal.MetalEngine())
    assert metal.equals(cpu)


@given(null_heavy_frame())
@settings(max_examples=100, deadline=None)
def test_null_heavy_collect_matches_cpu(lf) -> None:  # type: ignore[no-untyped-def]
    cpu = lf.collect()
    metal = lf.collect(engine=polars_metal.MetalEngine())
    assert metal.equals(cpu)
