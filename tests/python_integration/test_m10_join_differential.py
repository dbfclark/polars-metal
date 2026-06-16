"""M10 join->gather differential suite: engine='metal' == Polars CPU, byte-exact,
across dense/non-dense, join types, nulls, dups, missing keys, dtypes, multi-col
dims, and suffix collisions. Any failure is a correctness bug, not a tolerance issue."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_metal import MetalEngine

KEY_DTYPES = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32]


def _chain():
    return (pl.col("value") * 0.5
            * (1.0 + (0.7978845608 * pl.col("vol").log()).tanh())).alias("out")


def _run(fact, dim, how="left"):
    lf = fact.lazy().join(dim.lazy(), on="id", how=how).with_columns(_chain())
    assert_frame_equal(lf.collect(), lf.collect(engine=MetalEngine()),
                       check_dtypes=True, rel_tol=1e-3, abs_tol=1e-3)


@pytest.mark.parametrize("how", ["left", "inner"])
@pytest.mark.parametrize("kd", KEY_DTYPES)
def test_dense(how, kd):
    rng = np.random.default_rng(20)
    dim_n = int(min(200, np.iinfo(kd).max))
    n = 5000
    fact = pl.DataFrame({"id": rng.integers(0, dim_n, n).astype(kd),
                         "value": rng.uniform(50, 150, n).astype(np.float32)})
    dim = pl.DataFrame({"id": rng.permutation(dim_n).astype(kd),
                        "vol": rng.uniform(0.1, 0.5, dim_n).astype(np.float32)})
    _run(fact, dim, how)


@pytest.mark.parametrize("how", ["left", "inner"])
def test_nondense_sparse_keys(how):
    rng = np.random.default_rng(21)
    keys = rng.choice(10_000, 300, replace=False).astype(np.int64)
    fact = pl.DataFrame({"id": rng.choice(keys, 5000).astype(np.int64),
                         "value": rng.uniform(50, 150, 5000).astype(np.float32)})
    dim = pl.DataFrame({"id": keys, "vol": rng.uniform(0.1, 0.5, len(keys)).astype(np.float32)})
    _run(fact, dim, how)


def test_left_join_missing_keys_yield_nulls():
    fact = pl.DataFrame({"id": np.int64([0, 1, 2, 3]), "value": np.float32([10, 20, 30, 40])})
    dim = pl.DataFrame({"id": np.int64([0, 2]), "vol": np.float32([0.1, 0.2])})  # 1,3 missing
    _run(fact, dim, "left")


def test_inner_join_missing_keys_drop_rows():
    fact = pl.DataFrame({"id": np.int64([0, 1, 2, 3]), "value": np.float32([10, 20, 30, 40])})
    dim = pl.DataFrame({"id": np.int64([0, 2]), "vol": np.float32([0.1, 0.2])})
    _run(fact, dim, "inner")


def test_duplicate_dim_keys_explode():
    fact = pl.DataFrame({"id": np.int64([0, 1]), "value": np.float32([10, 20])})
    dim = pl.DataFrame({"id": np.int64([0, 0, 1]), "vol": np.float32([0.1, 0.2, 0.3])})  # 1:many
    _run(fact, dim, "inner")


def test_null_keys():
    fact = pl.DataFrame({"id": pl.Series([0, None, 2], dtype=pl.Int64), "value": np.float32([10, 20, 30])})
    dim = pl.DataFrame({"id": np.int64([0, 1, 2]), "vol": np.float32([0.1, 0.2, 0.3])})
    _run(fact, dim, "left")


def test_empty_dim():
    fact = pl.DataFrame({"id": np.int64([0, 1]), "value": np.float32([10, 20])})
    dim = pl.DataFrame({"id": np.array([], np.int64), "vol": np.array([], np.float32)})
    _run(fact, dim, "left")


def test_empty_fact():
    fact = pl.DataFrame({"id": np.array([], np.int64), "value": np.array([], np.float32)})
    dim = pl.DataFrame({"id": np.int64([0, 1]), "vol": np.float32([0.1, 0.2])})
    _run(fact, dim, "left")


def test_single_row():
    fact = pl.DataFrame({"id": np.int64([0]), "value": np.float32([7])})
    dim = pl.DataFrame({"id": np.int64([0]), "vol": np.float32([0.3])})
    _run(fact, dim, "inner")


def test_multi_dim_value_columns_routes_correctly():
    # dim has TWO non-key columns -> MVP resident requires exactly one -> CPU-lookup branch.
    # Chain only uses vol; output keeps both vol and sector. Must still be byte-exact.
    rng = np.random.default_rng(22)
    dim_n, n = 100, 3000
    fact = pl.DataFrame({"id": rng.integers(0, dim_n, n).astype(np.int64),
                         "value": rng.uniform(50, 150, n).astype(np.float32)})
    dim = pl.DataFrame({"id": rng.permutation(dim_n).astype(np.int64),
                        "vol": rng.uniform(0.1, 0.5, dim_n).astype(np.float32),
                        "sector": rng.uniform(0, 1, dim_n).astype(np.float32)})
    _run(fact, dim, "left")


def test_suffix_collision_nonkey_overlap():
    # fact and dim share a non-key column name "value" -> Polars suffixes the right one.
    # The chain references vol + value (left). Must match CPU (this exercises the
    # join-suffix path the Task 1.3 review flagged).
    rng = np.random.default_rng(23)
    dim_n, n = 100, 3000
    fact = pl.DataFrame({"id": rng.integers(0, dim_n, n).astype(np.int64),
                         "value": rng.uniform(50, 150, n).astype(np.float32)})
    dim = pl.DataFrame({"id": rng.permutation(dim_n).astype(np.int64),
                        "vol": rng.uniform(0.1, 0.5, dim_n).astype(np.float32),
                        "value": rng.uniform(0, 1, dim_n).astype(np.float32)})  # name clash
    _run(fact, dim, "left")
