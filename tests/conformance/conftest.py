"""Conformance harness: parametrize across cpu and metal engines."""

import pytest

import polars_metal


@pytest.fixture(params=["cpu", "metal"])
def engine(request):  # type: ignore[no-untyped-def]
    if request.param == "cpu":
        return "cpu"
    return polars_metal.MetalEngine()
