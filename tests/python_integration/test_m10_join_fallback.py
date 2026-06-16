import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from polars_metal import MetalEngine


def _check_cpu_parity(lf):
    assert_frame_equal(
        lf.collect(),
        lf.collect(engine=MetalEngine()),
        check_dtypes=True,
        rtol=1e-3,
        atol=1e-3,
    )


def test_f64_chain_falls_back():
    f = pl.DataFrame(
        {"id": np.arange(100, dtype=np.int64), "v": np.arange(100, dtype=np.float64)}
    )
    d = pl.DataFrame(
        {
            "id": np.arange(100, dtype=np.int64),
            "vol": np.arange(100, dtype=np.float64),
        }
    )
    _check_cpu_parity(
        f.lazy()
        .join(d.lazy(), on="id")
        .with_columns((pl.col("vol") * pl.col("v")).alias("o"))
    )


def test_string_key_falls_back():
    f = pl.DataFrame({"id": ["a", "b"], "v": np.float32([1, 2])})
    d = pl.DataFrame({"id": ["a", "b"], "vol": np.float32([3, 4])})
    _check_cpu_parity(
        f.lazy()
        .join(d.lazy(), on="id")
        .with_columns((pl.col("vol") * pl.col("v")).alias("o"))
    )


def test_outer_join_falls_back():
    f = pl.DataFrame({"id": np.int64([0, 1, 2]), "v": np.float32([1, 2, 3])})
    d = pl.DataFrame({"id": np.int64([0, 1]), "vol": np.float32([3, 4])})
    _check_cpu_parity(
        f.lazy()
        .join(d.lazy(), on="id", how="full")
        .with_columns((pl.col("vol") * pl.col("v")).alias("o"))
    )
