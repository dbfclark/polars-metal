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


def test_corr_detect_finds_binding():
    from polars_metal import _corr_detect

    lf = _frame(p=4).lazy().metal.corr()
    bindings = _corr_detect.find_corr_bindings(lf)
    assert len(bindings) == 1
    assert bindings[0].out_name  # the sentinel column name
    assert isinstance(bindings[0].handle, int)
