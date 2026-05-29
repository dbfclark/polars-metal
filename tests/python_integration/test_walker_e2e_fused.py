"""M4 Phase 5 Task 23: end-to-end fused subgraph execution.

`df.with_columns(...).collect(engine="metal")` runs through the fused MLX
subgraph and returns a result equal to `engine="cpu"`.
"""

import polars as pl
from polars.testing import assert_frame_equal

import polars_metal


def test_sqrt_chain_e2e():
    n = 1024
    df = pl.DataFrame(
        {
            "a": pl.Series([float(i % 100) for i in range(n)], dtype=pl.Float32),
            "b": pl.Series([float((i * 7) % 256) for i in range(n)], dtype=pl.Float32),
        }
    )
    expr = (pl.col("a").sqrt() + pl.col("b").sqrt()).cos()
    cpu_result = df.lazy().with_columns(y=expr).collect()
    metal_result = df.lazy().with_columns(y=expr).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu_result, metal_result, check_exact=False, abs_tol=1e-4)


def test_transcendental_chain_e2e():
    n = 2048
    df = pl.DataFrame(
        {
            "a": pl.Series([0.1 + float(i) * 0.001 for i in range(n)], dtype=pl.Float32),
        }
    )
    expr = pl.col("a").log().sqrt()
    cpu_result = df.lazy().with_columns(y=expr).collect()
    metal_result = df.lazy().with_columns(y=expr).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu_result, metal_result, check_exact=False, abs_tol=1e-4)
