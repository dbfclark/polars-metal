"""Smoke test for the execute_dtw PyO3 binding (M6 A4)."""

import numpy as np

from polars_metal import _native


def _dtw_ref(q, r, window=-1):
    L = len(q)
    inf = np.inf
    prev = np.full(L + 1, inf)
    cur = np.full(L + 1, inf)
    prev[0] = 0.0
    for i in range(1, L + 1):
        cur[0] = inf
        for j in range(1, L + 1):
            if window >= 0 and abs(i - j) > window:
                cur[j] = inf
                continue
            d = q[i - 1] - r[j - 1]
            cur[j] = d * d + min(prev[j], cur[j - 1], prev[j - 1])
        prev, cur = cur.copy(), prev
    return float(np.sqrt(prev[L]))


def test_execute_dtw_matches_reference():
    rng = np.random.default_rng(7)
    L, N = 12, 5
    r = rng.standard_normal(L).astype(np.float32)
    Q = rng.standard_normal((N, L)).astype(np.float32)
    out = np.empty(N, dtype=np.float32)
    qflat = np.ascontiguousarray(Q).reshape(-1)
    _native.execute_dtw(
        (qflat.ctypes.data, qflat.size),
        (r.ctypes.data, r.size),
        (out.ctypes.data, out.size),
        N,
        L,
        -1,  # window < 0 => full DTW
    )
    for i in range(N):
        assert abs(out[i] - _dtw_ref(Q[i], r)) <= 1e-3 * (1 + abs(_dtw_ref(Q[i], r)))
