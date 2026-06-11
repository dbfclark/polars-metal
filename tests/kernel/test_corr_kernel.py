import numpy as np

from polars_metal import _native


def _gpu_corr(x: np.ndarray) -> np.ndarray:
    n, p = x.shape
    flat = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    out = _native.execute_corr((flat.ctypes.data, int(flat.size)), int(n), int(p))
    return np.asarray(out, dtype=np.float32).reshape(p, p)


def test_corr_kernel_vs_numpy_wide():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((10000, 50)).astype(np.float32)
    got = _gpu_corr(x)
    expected = np.corrcoef(x.T).astype(np.float32)
    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_corr_kernel_p1_is_one():
    rng = np.random.default_rng(8)
    x = rng.standard_normal((500, 1)).astype(np.float32)
    got = _gpu_corr(x)
    assert got.shape == (1, 1)
    assert abs(got[0, 0] - 1.0) < 1e-4


def test_corr_kernel_constant_column_is_nan():
    # A zero-variance column → division by zero → NaN, matching df.corr().
    x = np.ones((500, 2), dtype=np.float32)
    x[:, 1] = np.linspace(0, 1, 500, dtype=np.float32)  # col1 varies, col0 constant
    got = _gpu_corr(x)
    assert np.isnan(got[0, 0]) and np.isnan(got[0, 1]) and np.isnan(got[1, 0])
