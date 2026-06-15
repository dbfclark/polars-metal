"""The CPU<->GPU crossing primitive and its cost model.

A "crossing" on M-series is NOT a PCIe transfer (CPU and GPU share RAM); it is a
RAM->RAM memcpy, because Metal's zero-copy buffer wrap needs 16 KB page alignment
that Polars' 64-byte-aligned Arrow buffers don't satisfy (the StagingPool finding).
The Python `mx.array(host)` / `np.array(device)` round-trip pays exactly that
memcpy (into / out of MLX-allocated unified memory), so it is the representative
primitive for what the engine's Rust StagingPool costs.

We fit  crossing_cost ~= alpha * bytes_crossed + beta * n_crossings  so the M9
verdict generalizes to any pipeline from its (bytes, crossings) profile.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from tests.bench.m8_report._timing import measure


def to_gpu(arr: np.ndarray) -> mx.array:
    """Host -> unified-memory MLX array (pays the copy-in memcpy)."""
    g = mx.array(arr)
    mx.eval(g)
    return g


def to_cpu(arr: mx.array) -> np.ndarray:
    """MLX array -> host numpy (pays the readback memcpy)."""
    mx.eval(arr)
    return np.array(arr)


@dataclass
class CostModel:
    alpha_ms_per_byte: float
    beta_ms_per_crossing: float

    def predict(self, *, bytes_crossed: int, n_crossings: int) -> float:
        return self.alpha_ms_per_byte * bytes_crossed + self.beta_ms_per_crossing * n_crossings


def probe_alpha(byte_sizes: list[int]) -> list[tuple[int, float]]:
    """One round-trip per size; returns (bytes, median_ms). Slope vs bytes = alpha."""
    out = []
    for nbytes in byte_sizes:
        n = max(1, nbytes // 4)  # f32
        a = np.ones(n, dtype=np.float32)
        ms = measure(lambda a=a: to_cpu(to_gpu(a))).median_ms
        out.append((n * 4, ms))
    return out


def probe_beta(counts: list[int]) -> list[tuple[int, float]]:
    """`count` sequential round-trips of a tiny array; (count, median_ms). Slope = beta."""
    tiny = np.ones(4, dtype=np.float32)  # 16 bytes — volume negligible

    def do(count: int) -> None:
        for _ in range(count):
            to_cpu(to_gpu(tiny))

    return [(c, measure(lambda c=c: do(c)).median_ms) for c in counts]


def fit_cost_model() -> CostModel:
    a_pts = probe_alpha([1 << 16, 1 << 20, 1 << 22, 1 << 24])  # 64KB .. 16MB
    xb = np.array([b for b, _ in a_pts], dtype=np.float64)
    yb = np.array([ms for _, ms in a_pts], dtype=np.float64)
    alpha = float(np.polyfit(xb, yb, 1)[0])  # ms per byte
    b_pts = probe_beta([1, 4, 16, 64])
    xc = np.array([c for c, _ in b_pts], dtype=np.float64)
    yc = np.array([ms for _, ms in b_pts], dtype=np.float64)
    beta = float(np.polyfit(xc, yc, 1)[0])  # ms per crossing
    return CostModel(alpha_ms_per_byte=max(alpha, 1e-12), beta_ms_per_crossing=max(beta, 1e-9))
