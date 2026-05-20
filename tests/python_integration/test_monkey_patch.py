"""The Polars callback monkey-patch is installed and idempotent."""

import importlib

import polars.lazyframe.frame as plf
import pytest

import polars_metal


def test_patch_installed() -> None:
    assert hasattr(plf, "_polars_metal_original_gpu_engine_callback")
    assert hasattr(plf.LazyFrame, "_polars_metal_original_collect")


def test_patch_is_idempotent() -> None:
    importlib.reload(polars_metal)
    # Patch attributes still present; wrapper is not the original.
    original_cb = plf._polars_metal_original_gpu_engine_callback
    wrapper_cb = plf._gpu_engine_callback
    assert wrapper_cb is not original_cb


def test_patch_passes_through_non_metal_engines() -> None:
    # An unknown engine config should pass through to the original
    # callback. The original raises ValueError for unknown strings.
    with pytest.raises(ValueError):
        plf._gpu_engine_callback(
            engine="nonsense",
            streaming=False,
            background=False,
            new_streaming=False,
            _eager=False,
        )
