import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from polars_metal import MetalEngine, _udf


def _small_dense_pipeline():
    rng = np.random.default_rng(40)
    n, dim_n = 2000, 100  # n < MIN_ROWS_THRESHOLD (1e5) -> below density gate
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
    return (
        fact.lazy()
        .join(dim.lazy(), on="id")
        .with_columns((pl.col("value") * (pl.col("vol").exp())).alias("out"))
    )


def test_below_threshold_defaults_cpu():
    lf = _small_dense_pipeline()
    _udf._M10_DENSE_GATHERS = 0
    out = lf.collect(engine=MetalEngine())
    assert _udf._M10_DENSE_GATHERS == 0, "below-threshold join should route CPU by default"
    assert_frame_equal(lf.collect(), out, check_dtypes=True, rel_tol=1e-3, abs_tol=1e-3)


def test_force_fusion_overrides():
    lf = _small_dense_pipeline()
    _udf._M10_DENSE_GATHERS = 0
    out = lf.collect(engine=MetalEngine(force_fusion=True))
    assert _udf._M10_DENSE_GATHERS == 1, "force_fusion should run the GPU resident branch"
    assert_frame_equal(lf.collect(), out, check_dtypes=True, rel_tol=1e-3, abs_tol=1e-3)


def test_large_dense_routes_gpu_by_default():
    # n above threshold + dense + compute chain -> GPU by default (no force needed).
    # 2.5M rows clears BOTH gates: rows>=1e5 and FLOPs>=5e7 (the 2-transcendental
    # chain is ~24 FLOPs/row, so it needs ~2.1M rows to clear the 5e7 FLOPs floor).
    rng = np.random.default_rng(41)
    n, dim_n = 2_500_000, 5000
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
    lf = (
        fact.lazy()
        .join(dim.lazy(), on="id")
        .with_columns(
            (pl.col("value") * 0.5 * (1.0 + (0.7978845608 * pl.col("vol").log()).tanh())).alias(
                "out"
            )
        )
    )
    _udf._M10_DENSE_GATHERS = 0
    out = lf.collect(engine=MetalEngine())
    assert _udf._M10_DENSE_GATHERS == 1, "above-threshold dense compute chain should route GPU"
    assert_frame_equal(lf.collect(), out, check_dtypes=True, rel_tol=1e-3, abs_tol=1e-3)
