"""M6 A3: FFT sentinel builder tests."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

import polars_metal  # noqa: F401  (registers engine + .metal namespace)
from polars_metal import MetalEngine, _fft_detect
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
    with pytest.raises(RuntimeError, match="engine='metal'"):
        df.lazy().with_columns(expr.alias("spec")).collect()  # plain CPU → raises


def test_find_fft_bindings_recovers_col_and_op():
    df = pl.DataFrame({"sig": [1.0, 2.0, 3.0, 4.0]}, schema={"sig": pl.Float32})
    lf = df.lazy().with_columns(pl.col("sig").metal.fft().alias("spec"))
    bindings = _fft_detect.find_fft_bindings(lf)
    assert len(bindings) == 1
    assert bindings[0].out_name == "spec"
    assert bindings[0].input_col == "sig"
    assert bindings[0].op == fns.OP_FFT


def test_fft_matches_numpy_end_to_end():
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(64).astype(np.float32)
    df = pl.DataFrame({"sig": sig}, schema={"sig": pl.Float32})
    out = df.lazy().with_columns(pl.col("sig").metal.fft().alias("spec")).collect(
        engine=MetalEngine()
    )
    spec = out.get_column("spec")
    got_re = np.asarray(spec.struct.field("real").to_numpy(), dtype=np.float32)
    got_im = np.asarray(spec.struct.field("imag").to_numpy(), dtype=np.float32)
    exp = np.fft.fft(sig.astype(np.float32))
    assert np.allclose(got_re, exp.real, rtol=1e-3, atol=1e-3)
    assert np.allclose(got_im, exp.imag, rtol=1e-3, atol=1e-3)
