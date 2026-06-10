"""Detect-layer tests for .metal.dtw (M6 A4): no engine call."""

import json
import warnings

import numpy as np
import polars as pl

from polars_metal import _dtw_namespace


def _serialize(expr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return json.loads(expr.meta.serialize(format="json"))


def test_dtw_builds_tagged_sentinel():
    ref = np.arange(4, dtype=np.float32)
    expr = pl.col("seq").metal.dtw(ref, window=2)
    s = json.dumps(_serialize(expr))
    assert _dtw_namespace.DTW_SENTINEL_TAG in s


def test_dtw_captures_spec():
    ref = np.arange(4, dtype=np.float32)
    n_before = len(_dtw_namespace._DTW_CACHE)
    pl.col("seq").metal.dtw(ref, window=2, allow_cpu_fallback=True)
    assert len(_dtw_namespace._DTW_CACHE) == n_before + 1
    spec = list(_dtw_namespace._DTW_CACHE.values())[-1]
    assert spec.window == 2
    assert spec.allow_cpu_fallback is True
    assert spec.query_col == "seq"
    np.testing.assert_array_equal(np.asarray(spec.reference, dtype=np.float32), ref)
