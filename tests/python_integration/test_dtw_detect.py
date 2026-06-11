"""Detect-layer tests for .metal.dtw (M6 A4): no engine call."""

import json
import warnings

import numpy as np
import polars as pl

from polars_metal import _dtw_detect, _dtw_namespace


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


def test_find_dtw_bindings_recognizes_sentinel():
    ref = np.arange(4, dtype=np.float32)
    df = pl.DataFrame(
        {"seq": [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]},
        schema={"seq": pl.Array(pl.Float32, 4)},
    )
    lf = df.lazy().with_columns(pl.col("seq").metal.dtw(ref, window=1).alias("d"))
    bindings = _dtw_detect.find_dtw_bindings(lf)
    assert len(bindings) == 1
    assert bindings[0].out_name == "d"
    assert bindings[0].query_col == "seq"


def test_find_dtw_bindings_ignores_plain_exprs():
    df = pl.DataFrame({"x": [1.0, 2.0]})
    lf = df.lazy().with_columns((pl.col("x") * 2).alias("y"))
    assert _dtw_detect.find_dtw_bindings(lf) == []
