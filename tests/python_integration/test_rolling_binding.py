"""
Task 3 — execute_rolling PyO3 binding

Verifies that `_native.execute_rolling` correctly dispatches the Metal
rolling-statistics kernels (sum / mean / var / std) and that the results
are written into the caller-provided output array.
"""

import numpy as np

from polars_metal import _native


def test_execute_rolling_sum_matches_numpy():
    x = np.arange(1, 11, dtype=np.float32)
    out = np.zeros_like(x)
    # op codes: 0=sum, 1=mean, 2=var, 3=std ; ddof for var/std
    _native.execute_rolling(
        inp=(x.ctypes.data, x.size),
        out=(out.ctypes.data, out.size),
        w=3,
        op=0,
        ddof=1,
    )
    assert abs(out[2] - 6.0) < 1e-5, f"expected 6.0 at index 2, got {out[2]}"  # window [1,2,3]=6
    assert abs(out[9] - 27.0) < 1e-5, (
        f"expected 27.0 at index 9, got {out[9]}"
    )  # window [8,9,10]=27


def test_execute_rolling_mean_var_std():
    x = np.arange(1, 11, dtype=np.float32)

    # mean: mean([1,2,3])=2.0, mean([8,9,10])=9.0
    for op, idx, want in [(1, 2, 2.0), (1, 9, 9.0)]:
        out = np.zeros_like(x)
        _native.execute_rolling(
            inp=(x.ctypes.data, x.size),
            out=(out.ctypes.data, out.size),
            w=3,
            op=op,
            ddof=1,
        )
        assert abs(out[idx] - want) < 1e-5, (
            f"mean op={op} idx={idx}: expected {want}, got {out[idx]}"
        )

    # var of [8,9,10] (ddof=1) = 1.0 ; std = 1.0
    for op in (2, 3):
        out = np.zeros_like(x)
        _native.execute_rolling(
            inp=(x.ctypes.data, x.size),
            out=(out.ctypes.data, out.size),
            w=3,
            op=op,
            ddof=1,
        )
        assert abs(out[9] - 1.0) < 1e-4, f"op={op} (var/std) at index 9: expected 1.0, got {out[9]}"
