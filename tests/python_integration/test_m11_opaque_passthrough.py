"""A frame carrying a Struct/List column alongside F32 columns must run an F32
with_columns chain correctly (or on CPU) and preserve the opaque column byte-exact
-- the OPAQUE-scan passthrough must NOT crash (regression: 'unknown MetalDtype tag').
"""

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from polars_metal import MetalEngine


def _check(lf):
    assert_frame_equal(
        lf.collect(),
        lf.collect(engine=MetalEngine()),
        check_dtypes=True,
        rel_tol=1e-3,
        abs_tol=1e-3,
    )


def test_hstack_chain_over_frame_with_struct_col():
    rng = np.random.default_rng(0)
    n = 2_000_000  # clears the fusion gate WITHOUT force_fusion (the crashing path)
    df = pl.DataFrame(
        {
            "a": rng.uniform(1, 5, n).astype(np.float32),
            "b": rng.uniform(1, 5, n).astype(np.float32),
        }
    ).with_columns(s=pl.struct([(pl.col("a") > 3).alias("hi"), pl.col("b").alias("bb")]))
    lf = df.lazy().with_columns(o=(pl.col("a") * pl.col("b").log()))
    _check(lf)


def test_select_chain_over_frame_with_list_col():
    rng = np.random.default_rng(1)
    n = 2_000_000
    df = pl.DataFrame(
        {
            "a": rng.uniform(1, 5, n).astype(np.float32),
            "b": rng.uniform(1, 5, n).astype(np.float32),
        }
    ).with_columns(lst=pl.concat_list([pl.col("a"), pl.col("b")]))
    lf = df.lazy().with_columns(o=(pl.col("a") * 0.5 + pl.col("b").exp()))
    _check(lf)
