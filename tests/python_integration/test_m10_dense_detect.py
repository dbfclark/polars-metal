import numpy as np

from polars_metal._join_gather import dense_positions


def test_dense_key_detected_and_reordered():
    # dim keyed 0..n in shuffled order; value follows key.
    key = np.array([2, 0, 3, 1], dtype=np.int64)
    val = np.array([20., 0., 30., 10.], dtype=np.float32)
    is_dense, reordered = dense_positions(key, val, dim_height=4)
    assert is_dense
    np.testing.assert_array_equal(reordered, [0., 10., 20., 30.])


def test_nondense_key_rejected():
    key = np.array([0, 1, 5], dtype=np.int64)   # gap -> not 0..n-1
    val = np.array([1., 2., 3.], dtype=np.float32)
    is_dense, _ = dense_positions(key, val, dim_height=3)
    assert not is_dense


def test_duplicate_key_rejected():
    key = np.array([0, 0, 1], dtype=np.int64)
    val = np.array([1., 2., 3.], dtype=np.float32)
    is_dense, _ = dense_positions(key, val, dim_height=3)
    assert not is_dense


def test_empty_dim_not_dense():
    key = np.array([], dtype=np.int64)
    val = np.array([], dtype=np.float32)
    is_dense, _ = dense_positions(key, val, dim_height=0)
    assert not is_dense


def test_negative_or_out_of_range_rejected():
    key = np.array([0, 1, 3], dtype=np.int64)   # max 3 but height 3 -> out of range
    val = np.array([1., 2., 3.], dtype=np.float32)
    is_dense, _ = dense_positions(key, val, dim_height=3)
    assert not is_dense
