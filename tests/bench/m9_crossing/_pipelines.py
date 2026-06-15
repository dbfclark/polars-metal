"""Hand-built mixed compute+join pipelines, each with 3-4 execution paths.

Every path of a pipeline MUST return the identical result (the correctness gate);
they differ only in where work runs and how many bytes cross the CPU<->GPU line.
GPU compute uses raw MLX (so crossing placement is explicit and controlled);
CPU uses numpy / Polars. Path names: all_cpu, partial_naive, partial_smart, resident.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from tests.bench.m9_crossing._crossing import to_cpu, to_gpu


@dataclass
class PipelineSpec:
    name: str
    family: str  # "gather" | "asof" | "hash"
    sizes: list[int]
    make_inputs: Callable[[int], Any]
    paths: dict[str, Callable[[Any], Any]]
    check: Callable[[Any, Any], None]  # (result_a, result_b) -> raises on mismatch


# ---------- shared helpers ----------


def _topk_result(hit_idx: np.ndarray, reranked: np.ndarray) -> dict[str, np.ndarray]:
    """Canonical result form: per-row sorted hit indices + sorted reranked scores."""
    return {
        "idx": np.sort(hit_idx, axis=1),
        "score": np.sort(reranked, axis=1),
    }


def _check_topk(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    for i in range(a["idx"].shape[0]):
        assert set(a["idx"][i].tolist()) == set(b["idx"][i].tolist()), f"idx row {i}"
    np.testing.assert_allclose(a["score"], b["score"], rtol=1e-3, atol=1e-3)


# ---------- P1: retrieve -> rerank ----------

_P1_D = 256
_P1_N = 50_000
_P1_K = 10


def _p1_make(q: int, seed: int = 0x91) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "queries": rng.standard_normal((q, _P1_D)).astype(np.float32),
        "corpus": rng.standard_normal((_P1_N, _P1_D)).astype(np.float32),
        "meta": rng.uniform(0.0, 1.0, size=_P1_N).astype(np.float32),
        "k": _P1_K,
    }


def _p1_all_cpu(inp: dict[str, Any]) -> dict[str, np.ndarray]:
    q, c, meta, k = inp["queries"], inp["corpus"], inp["meta"], inp["k"]
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    cn = c / np.linalg.norm(c, axis=1, keepdims=True)
    sims = qn @ cn.T  # (Q, N)
    hit = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]  # (Q, k)
    hit_sim = np.take_along_axis(sims, hit, axis=1)
    feat = meta[hit]  # gather
    reranked = hit_sim * np.exp(-feat)
    return _topk_result(hit, reranked)


def _p1_partial_naive(inp: dict[str, Any]) -> dict[str, np.ndarray]:
    q, c, meta, k = inp["queries"], inp["corpus"], inp["meta"], inp["k"]
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    cn = c / np.linalg.norm(c, axis=1, keepdims=True)
    gq, gc = to_gpu(qn), to_gpu(cn)
    sims_g = mx.matmul(gq, mx.swapaxes(gc, -1, -2))  # (Q, N) on GPU
    sims = to_cpu(sims_g)  # CROSS THE WHOLE Q x N MATRIX (dumb)
    hit = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    hit_sim = np.take_along_axis(sims, hit, axis=1)
    reranked = hit_sim * np.exp(-meta[hit])
    return _topk_result(hit, reranked)


def _p1_partial_smart(inp: dict[str, Any]) -> dict[str, np.ndarray]:
    q, c, meta, k = inp["queries"], inp["corpus"], inp["meta"], inp["k"]
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    cn = c / np.linalg.norm(c, axis=1, keepdims=True)
    gq, gc = to_gpu(qn), to_gpu(cn)
    sims_g = mx.matmul(gq, mx.swapaxes(gc, -1, -2))
    hit_g = mx.argpartition(-sims_g, kth=k - 1, axis=1)[:, :k]  # top-k ON GPU (reducer first)
    hit_sim_g = mx.take_along_axis(sims_g, hit_g, axis=1)
    mx.eval(hit_g, hit_sim_g)
    hit = to_cpu(hit_g).astype(np.int64)  # cross only Q x k
    hit_sim = to_cpu(hit_sim_g)
    reranked = hit_sim * np.exp(-meta[hit])  # gather + rerank on CPU
    return _topk_result(hit, reranked)


def _p1_resident(inp: dict[str, Any]) -> dict[str, np.ndarray]:
    q, c, meta, k = inp["queries"], inp["corpus"], inp["meta"], inp["k"]
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    cn = c / np.linalg.norm(c, axis=1, keepdims=True)
    gq, gc, gmeta = to_gpu(qn), to_gpu(cn), to_gpu(meta)
    sims_g = mx.matmul(gq, mx.swapaxes(gc, -1, -2))
    hit_g = mx.argpartition(-sims_g, kth=k - 1, axis=1)[:, :k]
    hit_sim_g = mx.take_along_axis(sims_g, hit_g, axis=1)
    feat_g = mx.take(gmeta, hit_g, axis=0)  # resident gather
    reranked_g = hit_sim_g * mx.exp(-feat_g)  # resident rerank
    mx.eval(hit_g, reranked_g)
    hit = to_cpu(hit_g).astype(np.int64)  # one small final fold-back
    reranked = to_cpu(reranked_g)
    return _topk_result(hit, reranked)


P1 = PipelineSpec(
    name="retrieve_rerank",
    family="gather",
    sizes=[1_000, 10_000],
    make_inputs=_p1_make,
    paths={
        "all_cpu": _p1_all_cpu,
        "partial_naive": _p1_partial_naive,
        "partial_smart": _p1_partial_smart,
        "resident": _p1_resident,
    },
    check=_check_topk,
)

PIPELINES: list[PipelineSpec] = [P1]


# ---------- P2: fact -> dim lookup -> compute chain ----------

_P2_DIM = 20_000


def _p2_make(n: int, seed: int = 0x92) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "value": rng.uniform(50, 150, size=n).astype(np.float32),
        "id": rng.integers(0, _P2_DIM, size=n).astype(np.int64),
        "dim": rng.uniform(0.1, 0.5, size=_P2_DIM).astype(np.float32),  # e.g. per-key volatility
    }


def _p2_chain_np(s: np.ndarray, vol: np.ndarray) -> np.ndarray:
    # Black-Scholes-shaped chain on fact value `s` and gathered dim `vol`.
    k, r, t = 100.0, 0.02, 1.0
    d1 = (np.log(s / k) + (r + 0.5 * vol * vol) * t) / (vol * np.sqrt(t))
    return s * 0.5 * (1.0 + np.tanh(0.7978845608 * d1))


def _p2_all_cpu(inp):
    vol = inp["dim"][inp["id"]]  # gather
    return _p2_chain_np(inp["value"], vol)


def _p2_partial_naive(inp):
    # dumb: cross the full gathered dim column AND value back and forth without reducing.
    gdim = to_gpu(inp["dim"])
    gid = to_gpu(inp["id"])
    vol_g = mx.take(gdim, gid, axis=0)
    vol = to_cpu(vol_g)  # cross full N gathered vols
    return _p2_chain_np(inp["value"], vol)  # chain on CPU


def _p2_partial_smart(inp):
    # smart: do the gather on CPU (it's an int index op, cheap), cross only the two
    # F32 columns the GPU chain needs, run the heavy transcendental chain on GPU.
    vol = inp["dim"][inp["id"]]  # CPU gather (cheap integer indexing)
    gs, gvol = to_gpu(inp["value"]), to_gpu(vol)
    k, r, t = 100.0, 0.02, 1.0
    d1 = (mx.log(gs / k) + (r + 0.5 * gvol * gvol) * t) / (gvol * (t**0.5))
    out_g = gs * 0.5 * (1.0 + mx.tanh(0.7978845608 * d1))
    return to_cpu(out_g)


def _p2_resident(inp):
    gdim, gid, gs = to_gpu(inp["dim"]), to_gpu(inp["id"]), to_gpu(inp["value"])
    gvol = mx.take(gdim, gid, axis=0)  # resident gather
    k, r, t = 100.0, 0.02, 1.0
    d1 = (mx.log(gs / k) + (r + 0.5 * gvol * gvol) * t) / (gvol * (t**0.5))
    out_g = gs * 0.5 * (1.0 + mx.tanh(0.7978845608 * d1))
    return to_cpu(out_g)


def _check_array(a: np.ndarray, b: np.ndarray) -> None:
    np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3)


P2 = PipelineSpec(
    name="fact_dim_chain",
    family="gather",
    sizes=[1_000_000, 10_000_000],
    make_inputs=_p2_make,
    paths={
        "all_cpu": _p2_all_cpu,
        "partial_naive": _p2_partial_naive,
        "partial_smart": _p2_partial_smart,
        "resident": _p2_resident,
    },
    check=_check_array,
)

PIPELINES.append(P2)
