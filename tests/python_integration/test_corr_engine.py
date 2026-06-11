import numpy as np
import polars as pl
import pytest

import polars_metal  # noqa: F401  (registers namespace + patches collect)


def _frame(n=2000, p=10, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p)).astype(np.float32)
    return pl.DataFrame(x, schema=[f"c{i}" for i in range(p)])


def test_corr_sentinel_raises_on_plain_cpu():
    # .metal.corr() builds a sentinel lf; collected WITHOUT engine="metal" it must raise.
    lf = _frame().lazy().metal.corr()
    with pytest.raises(RuntimeError):
        lf.collect()
