"""A4 spike: does batched DTW-vs-reference win on GPU?

DTW(query[i], reference) for N query sequences vs ONE reference, all length L,
F32. DTW's cell recurrence D[i,j] = |q_i - r_j| + min(D[i-1,j], D[i,j-1],
D[i-1,j-1]) is sequential in 2D; the ONLY parallelism is (a) across the N
independent pairs and (b) the anti-diagonal wavefront within a pair. The "vs
reference" shape gives N independent pairs -> embarrassingly parallel across
rows, which is the regime that could win on GPU.

This spike runs the IDENTICAL batched DP on numpy (CPU) and MLX (GPU): the L*L
cell updates, each vectorized over the N-pair batch. Same algorithm both sides
(fair, per the vector-search honesty lesson). CAVEAT: a tight C DTW lib
(dtaidistance, multi-threaded) would be faster than this vectorized-numpy
baseline, so the CPU bar here is OPTIMISTIC-for-GPU; treat a GPU win < ~3x as "a
real C baseline would erase it."

We measure the COMPUTE win (is the GPU faster at the DP?). If yes, A4 builds a
custom MSL kernel (the Python-MLX graph-build overhead here is NOT the shipped
path). If no, A4 should slip to M7.
"""

import time

import numpy as np
import mlx.core as mx


def dtw_scalar(q, r):
    """Reference DTW (single pair), for correctness only."""
    L = len(q)
    D = np.full((L + 1, L + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, L + 1):
        for j in range(1, L + 1):
            cost = abs(q[i - 1] - r[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[L, L]


def dtw_batched_numpy(Q, r):
    """Batched DTW: Q is (N, L), r is (L,). Returns (N,) distances.
    DP over an (N, L+1) rolling row; L*L cell updates each vectorized over N."""
    N, L = Q.shape
    INF = np.float32(1e30)
    prev = np.full((N, L + 1), INF, dtype=np.float32)
    cur = np.empty((N, L + 1), dtype=np.float32)
    # D[0,0]=0 handled by seeding prev[:,0]=0 before first row with a guard
    prev[:, 0] = 0.0  # represents D[i-1=...]; we set base below
    # Standard: D[0,:]=inf except D[0,0]=0; D[:,0]=inf except D[0,0]=0
    prev[:] = INF
    prev[:, 0] = 0.0
    prev[:, 1:] = INF
    # first "prev" row is D[0,*]: only D[0,0]=0
    prev[:] = INF
    prev[:, 0] = 0.0
    for i in range(1, L + 1):
        cur[:, 0] = INF
        qi = Q[:, i - 1]
        for j in range(1, L + 1):
            cost = np.abs(qi - r[j - 1])
            m = np.minimum(np.minimum(prev[:, j], cur[:, j - 1]), prev[:, j - 1])
            cur[:, j] = cost + m
        prev, cur = cur, prev
    return prev[:, L].copy()


def dtw_batched_mlx(Q, r):
    """Same DP in MLX. Q (N,L) mx.array, r (L,) mx.array. Returns (N,) mx.array."""
    N, L = Q.shape
    INF = mx.array(1e30, dtype=mx.float32)
    # columns as a python list of (N,) arrays to avoid in-place scatter
    prev = [mx.zeros((N,), dtype=mx.float32)] + [mx.full((N,), 1e30, dtype=mx.float32) for _ in range(L)]
    for i in range(1, L + 1):
        qi = Q[:, i - 1]
        cur = [mx.full((N,), 1e30, dtype=mx.float32)]
        for j in range(1, L + 1):
            cost = mx.abs(qi - r[j - 1])
            m = mx.minimum(mx.minimum(prev[j], cur[j - 1]), prev[j - 1])
            cur.append(cost + m)
        prev = cur
    out = prev[L]
    mx.eval(out)
    return out


def med(fn, it):
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    rng = np.random.default_rng(0xA4)

    # correctness on tiny input
    q = rng.standard_normal(8).astype(np.float32)
    r = rng.standard_normal(8).astype(np.float32)
    ref = dtw_scalar(q.astype(np.float64), r.astype(np.float64))
    got_np = dtw_batched_numpy(q[None, :], r)[0]
    got_mx = float(dtw_batched_mlx(mx.array(q[None, :]), mx.array(r))[0])
    print(f"correctness: ref={ref:.4f} numpy={got_np:.4f} mlx={got_mx:.4f} "
          f"{'OK' if abs(ref-got_np)<1e-3 and abs(ref-got_mx)<1e-2 else 'MISMATCH!!'}")

    print(f"\n{'N':>8} {'L':>5} {'cpu_ms':>10} {'gpu_ms':>10} {'speedup':>8} {'Ncells':>12}")
    print("-" * 60)
    for L in (64, 128, 256):
        r = rng.standard_normal(L).astype(np.float32)
        rmx = mx.array(r)
        for N in (1_000, 10_000, 100_000):
            Q = rng.standard_normal((N, L)).astype(np.float32)
            Qmx = mx.array(Q)
            mx.eval(Qmx)
            # warmup
            dtw_batched_numpy(Q, r)
            dtw_batched_mlx(Qmx, rmx)
            it = 5 if N < 100_000 else 3
            cpu = med(lambda: dtw_batched_numpy(Q, r), it)
            gpu = med(lambda: dtw_batched_mlx(Qmx, rmx), it)
            print(f"{N:>8,} {L:>5} {cpu*1e3:>10.2f} {gpu*1e3:>10.2f} "
                  f"{cpu/gpu:>7.2f}x {N*L*L:>12,}")


if __name__ == "__main__":
    main()
