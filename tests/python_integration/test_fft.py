"""M6 A3: FFT sentinel builder tests."""

from __future__ import annotations

import json

import polars as pl
import pytest

from polars_metal import _fft_namespace as fns


def test_fft_sentinel_carries_col_and_op():
    expr = fns.build_fft_sentinel(pl.col("sig"), "sig", fns.OP_FFT)
    j = json.loads(expr.meta.serialize(format="json"))
    s = json.dumps(j)
    assert fns.FFT_SENTINEL_TAG in s
    assert "sig" in s


def test_fft_verb_builds_sentinel_and_raises_on_cpu():
    df = pl.DataFrame({"sig": [1.0, 2.0, 3.0, 4.0]}, schema={"sig": pl.Float32})
    expr = pl.col("sig").metal.fft()
    j = json.loads(expr.meta.serialize(format="json"))
    assert fns.FFT_SENTINEL_TAG in json.dumps(j)
    with pytest.raises(Exception):
        df.lazy().with_columns(expr.alias("spec")).collect()  # plain CPU → raises
