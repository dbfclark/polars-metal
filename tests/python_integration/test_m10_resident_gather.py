import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from polars_metal import MetalEngine, _udf


def _pipeline(fact, dim, how="left"):
    return (
        fact.lazy()
        .join(dim.lazy(), on="id", how=how)
        .with_columns(
            (pl.col("value") * 0.5 * (1.0 + (0.7978845608 * pl.col("vol").log()).tanh())).alias(
                "out"
            )
        )
    )


def test_dense_resident_gather_matches_cpu():
    rng = np.random.default_rng(11)
    # 2.5M rows comfortably clears the join->gather density gate (1e7 FLOPs / 1e5
    # rows): the ~24-FLOPs/row chain clears the 1e7 floor from ~420k rows up.
    n, dim_n = 2_500_000, 20_000
    fact = pl.DataFrame(
        {
            "id": rng.integers(0, dim_n, n).astype(np.int64),
            "value": rng.uniform(50, 150, n).astype(np.float32),
        }
    )
    dim = pl.DataFrame(
        {
            "id": rng.permutation(dim_n).astype(np.int64),
            "vol": rng.uniform(0.1, 0.5, dim_n).astype(np.float32),
        }
    )
    lf = _pipeline(fact, dim)
    _udf._M10_DENSE_GATHERS = 0
    gpu = lf.collect(engine=MetalEngine())
    assert _udf._M10_DENSE_GATHERS == 1, "dense resident branch did not run"
    assert_frame_equal(lf.collect(), gpu, check_dtypes=True, rel_tol=1e-3, abs_tol=1e-3)


def test_nondense_falls_back_to_cpu_lookup_correct():
    rng = np.random.default_rng(12)
    keys = rng.choice(100_000, 2000, replace=False).astype(np.int64)
    fact = pl.DataFrame(
        {
            "id": rng.choice(keys, 50_000).astype(np.int64),
            "value": rng.uniform(50, 150, 50_000).astype(np.float32),
        }
    )
    dim = pl.DataFrame({"id": keys, "vol": rng.uniform(0.1, 0.5, len(keys)).astype(np.float32)})
    lf = _pipeline(fact, dim)
    _udf._M10_DENSE_GATHERS = 0
    gpu = lf.collect(engine=MetalEngine())
    assert _udf._M10_DENSE_GATHERS == 0, "should NOT use resident branch for sparse keys"
    assert_frame_equal(lf.collect(), gpu, check_dtypes=True, rel_tol=1e-3, abs_tol=1e-3)
