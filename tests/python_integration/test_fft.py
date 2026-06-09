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
    out = (
        df.lazy()
        .with_columns(pl.col("sig").metal.fft().alias("spec"))
        .collect(engine=MetalEngine())
    )
    spec = out.get_column("spec")
    got_re = np.asarray(spec.struct.field("real").to_numpy(), dtype=np.float32)
    got_im = np.asarray(spec.struct.field("imag").to_numpy(), dtype=np.float32)
    exp = np.fft.fft(sig.astype(np.float32))
    assert np.allclose(got_re, exp.real, rtol=1e-3, atol=1e-3)
    assert np.allclose(got_im, exp.imag, rtol=1e-3, atol=1e-3)


def test_ifft_matches_numpy_real_input():
    rng = np.random.default_rng(1)
    sig = rng.standard_normal(128).astype(np.float32)
    df = pl.DataFrame({"sig": sig}, schema={"sig": pl.Float32})
    out = (
        df.lazy()
        .with_columns(pl.col("sig").metal.ifft().alias("out"))
        .collect(engine=MetalEngine())
    )
    spec = out.get_column("out")
    got = np.asarray(spec.struct.field("real").to_numpy(), np.float32) + 1j * np.asarray(
        spec.struct.field("imag").to_numpy(), np.float32
    )
    exp = np.fft.ifft(sig.astype(np.float32))
    assert np.allclose(got.real, exp.real, rtol=1e-3, atol=1e-3)
    assert np.allclose(got.imag, exp.imag, rtol=1e-3, atol=1e-3)


def test_fft_then_ifft_round_trip_struct_input():
    rng = np.random.default_rng(2)
    sig = rng.standard_normal(256).astype(np.float32)
    df = pl.DataFrame({"sig": sig}, schema={"sig": pl.Float32})
    spec_df = (
        df.lazy()
        .with_columns(pl.col("sig").metal.fft().alias("spec"))
        .collect(engine=MetalEngine())
    )
    rec = (
        spec_df.lazy()
        .with_columns(pl.col("spec").metal.ifft().alias("rec"))
        .collect(engine=MetalEngine())
    )
    got = np.asarray(rec.get_column("rec").struct.field("real").to_numpy(), np.float32)
    assert np.allclose(got, sig, rtol=1e-3, atol=1e-3)


def test_fft_non_f32_raises():
    df = pl.DataFrame({"sig": [1, 2, 3, 4]}, schema={"sig": pl.Int64})
    with pytest.raises(ValueError):
        df.lazy().with_columns(pl.col("sig").metal.fft().alias("o")).collect(engine=MetalEngine())


def test_fft_nulls_raise():
    df = pl.DataFrame({"sig": [1.0, None, 3.0, 4.0]}, schema={"sig": pl.Float32})
    with pytest.raises(ValueError):
        df.lazy().with_columns(pl.col("sig").metal.fft().alias("o")).collect(engine=MetalEngine())


def test_fft_large_n_correct_on_gpu():
    # N = 3,000,000 is non-pow2 and > 1024 → the hand-rolled MSL kernel routes it through
    # Bluestein (M = next_pow2(2N-1) = 2^23) and computes it CORRECTLY on-GPU. (Previously this
    # size fell back to CPU because MLX's Metal FFT was broken above 2^20, ml-explore/mlx#1800.)
    n = 3_000_000
    sig = np.random.default_rng(7).standard_normal(n).astype(np.float32)
    df = pl.DataFrame({"sig": sig}, schema={"sig": pl.Float32})
    out = (
        df.lazy()
        .with_columns(pl.col("sig").metal.fft().alias("spec"))
        .collect(engine=MetalEngine())
    )
    spec = out.get_column("spec")
    got = np.asarray(spec.struct.field("real").to_numpy(), np.float64) + 1j * np.asarray(
        spec.struct.field("imag").to_numpy(), np.float64
    )
    exp = np.fft.fft(sig.astype(np.float64))
    l2 = np.linalg.norm(got - exp) / np.linalg.norm(exp)
    assert l2 < 1e-3, f"large-N FFT not correct on GPU: L2={l2}"


def test_fft_empty_column():
    df = pl.DataFrame({"sig": []}, schema={"sig": pl.Float32})
    out = df.lazy().with_columns(pl.col("sig").metal.fft().alias("o")).collect(engine=MetalEngine())
    assert out.get_column("o").len() == 0
    assert isinstance(out.get_column("o").dtype, pl.Struct)
