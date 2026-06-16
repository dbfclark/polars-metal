"""End-to-end: engine='metal' on join->chain == Polars CPU, byte-exact.
Phase 1 = CPU-lookup branch (join on CPU, fused chain on GPU)."""
from __future__ import annotations

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from polars_metal import MetalEngine


def _pipeline(fact, dim, how="left"):
    return (fact.lazy().join(dim.lazy(), on="id", how=how)
            .with_columns(
                (pl.col("value") * 0.5
                 * (1.0 + (0.7978845608 * (pl.col("vol").log())).tanh())).alias("out")))


def test_join_chain_matches_cpu_dense_key():
    rng = np.random.default_rng(10)
    n, dim_n = 1_000_000, 20_000
    fact = pl.DataFrame({"id": rng.integers(0, dim_n, n).astype(np.int64),
                         "value": rng.uniform(50, 150, n).astype(np.float32)})
    dim = pl.DataFrame({"id": np.arange(dim_n, dtype=np.int64),
                        "vol": rng.uniform(0.1, 0.5, dim_n).astype(np.float32)})
    lf = _pipeline(fact, dim)
    cpu = lf.collect()

    from polars_metal import _udf
    _udf._M10_JOIN_DISPATCHES = 0
    gpu = lf.collect(engine=MetalEngine())
    assert _udf._M10_JOIN_DISPATCHES == 1, "GPU join path did not run"
    assert_frame_equal(cpu, gpu, check_dtypes=True, rel_tol=1e-3, abs_tol=1e-3)
