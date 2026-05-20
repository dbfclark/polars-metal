"""Pytest plugin that forces engine=MetalEngine() on every LazyFrame.collect().

Activate by passing `-p polars_metal._pytest_plugin` to pytest. Used by the
conformance harness (`tests/conformance/test_polars_suite.py`) to re-run
Polars' own test suite through our engine and assert we add no new failures
beyond the baseline.

The plugin only intercepts when the caller passes `engine="auto"` (the
default). Explicit `engine="cpu"` or `engine=MetalEngine(...)` is respected
as-is.
"""

from __future__ import annotations

from typing import Any

import polars as pl

import polars_metal


def pytest_configure(config: Any) -> None:
    """Replace LazyFrame.collect to inject MetalEngine on default-engine calls."""
    real_collect = pl.LazyFrame.collect

    def metal_collect(self: Any, *, engine: Any = "auto", **kwargs: Any) -> Any:
        if engine == "auto":
            engine = polars_metal.MetalEngine()
        return real_collect(self, engine=engine, **kwargs)

    pl.LazyFrame.collect = metal_collect  # type: ignore[method-assign]
