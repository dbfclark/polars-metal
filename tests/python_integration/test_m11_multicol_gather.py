import numpy as np, polars as pl
from polars_metal import MetalEngine
from polars_metal import _udf
from polars.testing import assert_frame_equal


def _pipeline(fact, dim, how="left"):
    return (
        fact.lazy()
        .join(dim.lazy(), on="id", how=how)
        .with_columns((pl.col("sc") * pl.col("price").exp() * pl.col("rating").log()).alias("rr"))
    )


def test_multicol_resident_gather_matches_cpu():
    rng = np.random.default_rng(11)
    n, dim_n = 1_000_000, 20_000
    fact = pl.DataFrame(
        {
            "id": rng.integers(0, dim_n, n).astype(np.int64),
            "sc": rng.uniform(0, 1, n).astype(np.float32),
        }
    )
    dim = pl.DataFrame(
        {
            "id": rng.permutation(dim_n).astype(np.int64),
            "price": rng.uniform(0.1, 2.0, dim_n).astype(np.float32),
            "rating": rng.uniform(1, 5, dim_n).astype(np.float32),
        }
    )
    lf = _pipeline(fact, dim)
    _udf._M10_DENSE_GATHERS = 0
    gpu = lf.collect(engine=MetalEngine(force_fusion=True))
    assert _udf._M10_DENSE_GATHERS == 1, "multi-col resident gather did not run"
    assert_frame_equal(lf.collect(), gpu, check_dtypes=True, rel_tol=1e-3, abs_tol=1e-3)
