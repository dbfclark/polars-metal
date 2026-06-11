import numpy as np
import polars as pl
import pytest

import polars_metal as pm  # registers namespace + patches collect


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


def test_apply_corr_matches_numpy_gpu_path():
    from polars_metal import _corr_detect, _corr_dispatch

    df = _frame(n=4000, p=12, seed=1)
    lf = df.lazy().metal.corr()  # p=12 >= CORR_P_MIN → GPU
    bindings = _corr_detect.find_corr_bindings(lf)

    def _collect_rest(rest_lf):
        return rest_lf.collect()

    out = _corr_dispatch.apply_corr(lf, bindings[0], _collect_rest)
    assert out.shape == (12, 12)
    assert all(dt == pl.Float32 for dt in out.dtypes)
    assert out.columns == [f"c{i}" for i in range(12)]
    expected = np.corrcoef(df.to_numpy().T).astype(np.float32)
    np.testing.assert_allclose(out.to_numpy(), expected, atol=1e-4)


def test_metal_corr_end_to_end():
    df = _frame(n=5000, p=16, seed=2)
    out = df.lazy().metal.corr().collect(engine=pm.MetalEngine())
    assert out.shape == (16, 16)
    assert all(dt == pl.Float32 for dt in out.dtypes)
    expected = df.corr().cast(pl.Float32)
    np.testing.assert_allclose(out.to_numpy(), expected.to_numpy(), atol=1e-4)
    assert out.columns == expected.columns


def test_corr_small_p_cpu_fallback_correct():
    # p=3 < CORR_P_MIN → CPU fallback path; result still correct + F32.
    df = _frame(n=3000, p=3, seed=3)
    out = df.lazy().metal.corr().collect(engine=pm.MetalEngine())
    assert out.shape == (3, 3)
    assert all(dt == pl.Float32 for dt in out.dtypes)
    expected = df.corr().cast(pl.Float32)
    np.testing.assert_allclose(out.to_numpy(), expected.to_numpy(), atol=1e-4)


def test_corr_force_gpu_small_p_matches():
    # force_gpu=True drives p=3 through the GPU path; must still match oracle.
    df = _frame(n=3000, p=3, seed=4)
    out = df.lazy().metal.corr(force_gpu=True).collect(engine=pm.MetalEngine())
    expected = df.corr().cast(pl.Float32)
    np.testing.assert_allclose(out.to_numpy(), expected.to_numpy(), atol=1e-4)


def test_corr_p_min_constant_is_eight():
    from polars_metal._corr_dispatch import CORR_P_MIN

    assert CORR_P_MIN == 8


def test_corr_nulls_route_to_cpu_and_match():
    # Null-bearing input → CPU fallback (exact Polars null semantics), F32 out.
    df = _frame(n=2000, p=10, seed=5).with_columns(
        pl.when(pl.int_range(pl.len()) % 7 == 0).then(None).otherwise(pl.col("c0")).alias("c0")
    )
    out = df.lazy().metal.corr().collect(engine=pm.MetalEngine())
    expected = df.corr().cast(pl.Float32)
    assert all(dt == pl.Float32 for dt in out.dtypes)
    # Compare with NaN-equal semantics (constant/degenerate cells may be NaN).
    np.testing.assert_allclose(out.to_numpy(), expected.to_numpy(), atol=1e-4, equal_nan=True)


def test_corr_integer_and_f64_inputs_cast():
    # Int + F64 numeric columns are accepted (cast to F32). p>=8 → GPU path.
    rng = np.random.default_rng(6)
    df = pl.DataFrame(
        {f"c{i}": rng.integers(-100, 100, size=4000) for i in range(10)}
    )  # Int64 columns
    out = df.lazy().metal.corr().collect(engine=pm.MetalEngine())
    expected = df.corr().cast(pl.Float32)
    np.testing.assert_allclose(out.to_numpy(), expected.to_numpy(), atol=1e-3)


def test_corr_non_numeric_raises():
    df = _frame(n=100, p=9).with_columns(pl.lit("x").alias("c0"))
    with pytest.raises(ValueError):
        df.lazy().metal.corr().collect(engine=pm.MetalEngine())


def test_corr_single_row_raises_clear_error():
    df = pl.DataFrame({f"c{i}": [1.0] for i in range(10)})  # N=1
    with pytest.raises(pl.exceptions.ComputeError, match="at least 2 rows"):
        df.lazy().metal.corr().collect(engine=pm.MetalEngine())
