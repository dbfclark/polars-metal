"""MetalEngine configuration object — the user-facing engine handle."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetalEngine:
    """Configuration for the polars-metal engine.

    Mirrors Polars' GPUEngine pattern. Pass an instance to df.collect():

        import polars as pl
        import polars_metal
        df = pl.LazyFrame({"a": [1, 2, 3]}).collect(engine=polars_metal.MetalEngine())

    In M0 every IR node falls back to CPU; the engine entry point exists
    primarily to validate the registration mechanism.
    """

    device: int | None = None
    """Index into available Metal devices. None = system default."""

    debug: bool = False
    """If True, emit verbose dispatch logging via Python's logging module."""
