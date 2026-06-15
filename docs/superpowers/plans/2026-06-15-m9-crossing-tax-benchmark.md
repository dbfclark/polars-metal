# M9 — Crossing-Tax Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hand-build mixed compute+join pipelines across the join spectrum and time four execution strategies (all-CPU / partial-naive / partial-smart / resident), fitting an `α·bytes + β·crossings` cost model, to size whether cheap CPU↔GPU switching makes free per-op routing win — and emit a go/no-go for M10.

**Architecture:** A measurement harness under `tests/bench/m9_crossing/`, no engine changes. The crossing primitive is the Python `mx.array`/`np.array` round-trip (which pays the same page-aligned memcpy the engine's Rust `StagingPool` does). Each pipeline exposes its inputs + a dict of path-callables that must all return the identical result (the correctness gate). The driver times paths across sweeps and fits the cost model; the emitter writes a report + JSON + verdict.

**Tech Stack:** Python 3.11, NumPy (competent CPU baseline), MLX (GPU compute via `mx.matmul`/`mx.take`/`mx.argpartition`/transcendentals), Polars (`join_asof`, hash join), pytest. Reuses `tests/bench/m8_report/_timing.py`. Spec: `docs/superpowers/specs/2026-06-15-m9-crossing-tax-benchmark-design.md`.

---

## File Structure

| File | Responsibility |
|---|---|
| `tests/bench/m9_crossing/__init__.py` | package marker |
| `tests/bench/m9_crossing/_crossing.py` | the crossing primitive (`to_gpu`/`to_cpu`) + the α/β probes (`probe_alpha`, `probe_beta`, `fit_cost_model`) |
| `tests/bench/m9_crossing/_pipelines.py` | `PipelineSpec` dataclass + P1–P4, each with `make_inputs` + a dict of path-callables + a `check` |
| `tests/bench/m9_crossing/run.py` | driver: time pipelines × paths × sizes → `Row`s; fit (α, β); `main()` |
| `tests/bench/m9_crossing/emit.py` | rows + cost model → `docs/crossing-tax-report.md` + `crossing-tax.json` |
| `tests/bench/m9_crossing/test_smoke.py` | smoke+correctness gate: every pipeline's paths agree at smallest size |
| `tests/bench/m9_crossing/test_harness.py` | unit tests for `_crossing` probes + `emit` |
| `docs/crossing-tax-report.md` | generated artifact (committed) |
| `crossing-tax.json` | generated machine-readable twin (committed) |
| `Makefile` | new `crossing-report` target; smoke gate wired into `test-unit` |

**Reused:** `tests/bench/m8_report/_timing.py::measure` (warmup + median-of-N).

**Path naming (used everywhere, keep consistent):** `"all_cpu"`, `"partial_naive"`, `"partial_smart"`, `"resident"`. P3/P4 omit `"resident"`.

---

## Task 1: Crossing primitive + α/β probes (`_crossing.py`)

**Files:**
- Create: `tests/bench/m9_crossing/__init__.py`
- Create: `tests/bench/m9_crossing/_crossing.py`
- Test: `tests/bench/m9_crossing/test_harness.py`

- [ ] **Step 1: Create the package marker**

Create `tests/bench/m9_crossing/__init__.py`:
```python
"""M9 crossing-tax benchmark harness."""
```

- [ ] **Step 2: Write the failing test**

Create `tests/bench/m9_crossing/test_harness.py`:
```python
from __future__ import annotations

import numpy as np

from tests.bench.m9_crossing._crossing import (
    CostModel,
    fit_cost_model,
    to_cpu,
    to_gpu,
)


def test_crossing_roundtrip_preserves_data():
    a = np.arange(64, dtype=np.float32).reshape(8, 8)
    g = to_gpu(a)
    b = to_cpu(g)
    assert np.array_equal(a, b)


def test_fit_cost_model_returns_positive_coeffs():
    cm = fit_cost_model()
    assert isinstance(cm, CostModel)
    # alpha = ms per byte (host->device->host), beta = ms per crossing; both > 0
    assert cm.alpha_ms_per_byte > 0
    assert cm.beta_ms_per_crossing > 0
    # predict() is monotonic in both args
    assert cm.predict(bytes_crossed=10_000_000, n_crossings=1) > cm.predict(bytes_crossed=1_000, n_crossings=1)
    assert cm.predict(bytes_crossed=1_000, n_crossings=10) > cm.predict(bytes_crossed=1_000, n_crossings=1)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/bench/m9_crossing/test_harness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tests.bench.m9_crossing._crossing'`

- [ ] **Step 4: Write minimal implementation**

Create `tests/bench/m9_crossing/_crossing.py`:
```python
"""The CPU<->GPU crossing primitive and its cost model.

A "crossing" on M-series is NOT a PCIe transfer (CPU and GPU share RAM); it is a
RAM->RAM memcpy, because Metal's zero-copy buffer wrap needs 16 KB page alignment
that Polars' 64-byte-aligned Arrow buffers don't satisfy (the StagingPool finding).
The Python `mx.array(host)` / `np.array(device)` round-trip pays exactly that
memcpy (into / out of MLX-allocated unified memory), so it is the representative
primitive for what the engine's Rust StagingPool costs.

We fit  crossing_cost ≈ alpha * bytes_crossed + beta * n_crossings  so the M9
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
    alpha_ms_per_byte: float  # marginal ms per byte of a host->device->host round-trip
    beta_ms_per_crossing: float  # fixed ms per crossing (dispatch/sync), volume-independent

    def predict(self, *, bytes_crossed: int, n_crossings: int) -> float:
        return self.alpha_ms_per_byte * bytes_crossed + self.beta_ms_per_crossing * n_crossings


def probe_alpha(byte_sizes: list[int]) -> list[tuple[int, float]]:
    """One round-trip per size; returns (bytes, median_ms). Slope vs bytes = alpha."""
    out = []
    for nbytes in byte_sizes:
        n = max(1, nbytes // 4)  # f32
        a = np.ones(n, dtype=np.float32)
        ms = measure(lambda a=a: to_cpu(to_gpu(a)))
        out.append((n * 4, ms))
    return out


def probe_beta(counts: list[int]) -> list[tuple[int, float]]:
    """`count` sequential round-trips of a tiny array; (count, median_ms). Slope = beta."""
    tiny = np.ones(4, dtype=np.float32)  # 16 bytes — volume negligible

    def do(count: int) -> None:
        for _ in range(count):
            to_cpu(to_gpu(tiny))

    return [(c, measure(lambda c=c: do(c))) for c in counts]


def fit_cost_model() -> CostModel:
    # alpha: large sizes so the byte term dominates the fixed term.
    a_pts = probe_alpha([1 << 16, 1 << 20, 1 << 22, 1 << 24])  # 64KB .. 16MB
    xb = np.array([b for b, _ in a_pts], dtype=np.float64)
    yb = np.array([ms for _, ms in a_pts], dtype=np.float64)
    alpha = float(np.polyfit(xb, yb, 1)[0])  # ms per byte
    # beta: many tiny crossings so the fixed term dominates.
    b_pts = probe_beta([1, 4, 16, 64])
    xc = np.array([c for c, _ in b_pts], dtype=np.float64)
    yc = np.array([ms for _, ms in b_pts], dtype=np.float64)
    beta = float(np.polyfit(xc, yc, 1)[0])  # ms per crossing
    return CostModel(alpha_ms_per_byte=max(alpha, 1e-12), beta_ms_per_crossing=max(beta, 1e-9))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/bench/m9_crossing/test_harness.py -v`
Expected: PASS (round-trip preserved; α, β positive; predict monotonic).

- [ ] **Step 6: Lint + commit**

```bash
ruff check tests/bench/m9_crossing/ && ruff format tests/bench/m9_crossing/
git add tests/bench/m9_crossing/__init__.py tests/bench/m9_crossing/_crossing.py tests/bench/m9_crossing/test_harness.py
git commit -m "M9: crossing primitive + alpha/beta cost-model probes"
```

---

## Task 2: Pipeline interface + P1 (retrieve→rerank), the template

**Files:**
- Create: `tests/bench/m9_crossing/_pipelines.py`

P1 is the canonical gather pipeline and defines the pattern every later pipeline follows. Inputs: `Q` query vectors and an `N×D` corpus (F32), plus one F32 metadata feature per corpus row. Logic: top-k corpus by cosine similarity per query → gather the metadata feature for the k hits → rerank `final = sim * exp(-feature)` → return, per query, the **set of hit indices** and the **sorted reranked scores**. All four paths must produce the identical result.

The four paths differ only in *where the work runs and how much data crosses*:
- `all_cpu`: everything in numpy.
- `partial_naive`: GPU computes the full `Q×N` similarity matrix, then **crosses the whole matrix** to CPU for top-k + gather + rerank (dumb: huge volume).
- `partial_smart`: GPU computes similarity **and top-k**, crosses only the `Q×k` hit indices + sims (reducer pushed before the crossing), CPU does gather + rerank.
- `resident`: GPU does similarity + top-k + **gather (`mx.take`)** + rerank, crosses only the final `Q×k` result once.

- [ ] **Step 1: Write `_pipelines.py` with the interface + P1**

Create `tests/bench/m9_crossing/_pipelines.py`:
```python
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
import polars as pl

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
```

- [ ] **Step 2: Smoke-check P1 paths agree (manual, before the gate exists)**

Run:
```bash
python -c "
from tests.bench.m9_crossing._pipelines import P1
inp = P1.make_inputs(min(P1.sizes))
base = P1.paths['all_cpu'](inp)
for name, fn in P1.paths.items():
    P1.check(base, fn(inp)); print(name, 'agrees')
"
```
Expected: all four print `agrees`. If `partial_smart`/`resident` disagree, the likely cause is top-k tie ordering — confirm `_check_topk` compares index **sets** (it does) and reranked **sorted** scores; with random normal data exact ties are measure-zero.

- [ ] **Step 3: Lint + commit**

```bash
ruff check tests/bench/m9_crossing/ && ruff format tests/bench/m9_crossing/
git add tests/bench/m9_crossing/_pipelines.py
git commit -m "M9: PipelineSpec interface + P1 retrieve->rerank (4 paths)"
```

---

## Task 3: Smoke+correctness gate (`test_smoke.py`)

**Files:**
- Create: `tests/bench/m9_crossing/test_smoke.py`

- [ ] **Step 1: Write the gate**

Create `tests/bench/m9_crossing/test_smoke.py`:
```python
"""Smoke + correctness gate: for each pipeline, every path produces the IDENTICAL
result at the smallest size. A fast-but-wrong path is caught, and timing is only
ever apples-to-apples. Runs in `make test-unit`.
"""

from __future__ import annotations

import pytest

from tests.bench.m9_crossing._pipelines import PIPELINES


def _cases():
    for p in PIPELINES:
        for path in p.paths:
            yield pytest.param(p, path, id=f"{p.name}:{path}")


@pytest.mark.parametrize("pipeline,path", list(_cases()))
def test_paths_agree(pipeline, path):
    inp = pipeline.make_inputs(min(pipeline.sizes))
    base = pipeline.paths["all_cpu"](inp)
    pipeline.check(base, pipeline.paths[path](inp))
```

- [ ] **Step 2: Run the gate**

Run: `pytest tests/bench/m9_crossing/test_smoke.py -v`
Expected: 4 PASS (P1 × {all_cpu, partial_naive, partial_smart, resident}).

- [ ] **Step 3: Commit**

```bash
git add tests/bench/m9_crossing/test_smoke.py
git commit -m "M9: smoke+correctness gate (all pipeline paths agree)"
```

---

## Task 4: P2 (fact→dim lookup)

**Files:**
- Modify: `tests/bench/m9_crossing/_pipelines.py`

P2 is the second gather pipeline: a fact table with a dense integer `id` column → gather a per-dim F32 feature `dim[id]` → a Black-Scholes-shaped transcendental chain on `(fact_value, dim_feature)`. Result: the F32 output column. Gather is by **dense id**, so `resident` uses `mx.take`.

- [ ] **Step 1: Add P2 to `_pipelines.py`** (append helpers + spec; extend `PIPELINES`)

```python
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
```

- [ ] **Step 2: Run the gate for P2**

Run: `pytest tests/bench/m9_crossing/test_smoke.py -v -k fact_dim_chain`
Expected: 4 PASS. If a mismatch appears, it's almost certainly F32 vs F64 in the chain — both `_p2_chain_np` (numpy, F64 intermediates) and the MLX path (F32) should agree within rtol=1e-3; if not, loosen this pipeline's `check` to rtol=1e-2 and note it.

- [ ] **Step 3: Lint + commit**

```bash
ruff check tests/bench/m9_crossing/ && ruff format tests/bench/m9_crossing/
git add tests/bench/m9_crossing/_pipelines.py
git commit -m "M9: add P2 fact->dim lookup->chain (gather, 4 paths)"
```

---

## Task 5: P3 (as-of join → compute)

**Files:**
- Modify: `tests/bench/m9_crossing/_pipelines.py`

P3 is the sort-merge case: a left time-series (`ts_left`, `x`) as-of-joined to a right time-series (`ts_right`, `y`) — each left row matched to the latest right row with `ts_right <= ts_left` — then a compute step `out = sqrt(x*x + y*y)`. The match is **searchsorted** (both sides sorted by ts). No full `resident` path (merge is sequential-awkward); paths are `all_cpu`, `partial_naive`, `partial_smart`. The GPU-able lever is doing the elementwise compute on GPU after the (CPU) match; `partial_smart` crosses only the two matched F32 columns.

- [ ] **Step 1: Add P3 to `_pipelines.py`**

```python
# ---------- P3: as-of join -> compute ----------

def _p3_make(n: int, seed: int = 0x93) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    ts_left = np.sort(rng.integers(0, 4 * n, size=n)).astype(np.int64)
    ts_right = np.sort(rng.integers(0, 4 * n, size=n)).astype(np.int64)
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    return {"ts_left": ts_left, "x": x, "ts_right": ts_right, "y": y}


def _p3_match(inp) -> np.ndarray:
    # as-of (backward): latest right with ts_right <= ts_left. searchsorted on sorted ts_right.
    pos = np.searchsorted(inp["ts_right"], inp["ts_left"], side="right") - 1
    pos = np.clip(pos, 0, len(inp["ts_right"]) - 1)
    return inp["y"][pos]  # matched y per left row


def _p3_all_cpu(inp):
    yj = _p3_match(inp)
    return np.sqrt(inp["x"] * inp["x"] + yj * yj)


def _p3_partial_naive(inp):
    # dumb: cross x and the full matched y, but also redundantly round-trip y unmatched first.
    yj = _p3_match(inp)
    gx, gy = to_gpu(inp["x"]), to_gpu(yj)
    return to_cpu(mx.sqrt(gx * gx + gy * gy))


def _p3_partial_smart(inp):
    # match on CPU (searchsorted is sequential/CPU), cross only the two matched F32 cols,
    # compute on GPU. (Same crossing volume as naive here; the lever is the GPU compute —
    # this pipeline tests whether as-of's CPU match + GPU compute beats all-CPU at all.)
    yj = _p3_match(inp)
    gx, gy = to_gpu(inp["x"]), to_gpu(yj)
    return to_cpu(mx.sqrt(gx * gx + gy * gy))


P3 = PipelineSpec(
    name="asof_compute",
    family="asof",
    sizes=[1_000_000, 10_000_000],
    make_inputs=_p3_make,
    paths={
        "all_cpu": _p3_all_cpu,
        "partial_naive": _p3_partial_naive,
        "partial_smart": _p3_partial_smart,
    },
    check=_check_array,
)

PIPELINES.append(P3)
```

Note: `_p3_partial_naive` and `_p3_partial_smart` are intentionally near-identical here because the as-of *match* is irreducibly CPU (searchsorted) and the only GPU-able part is the small elementwise compute — so this pipeline's honest finding is "does CPU-match + GPU-compute beat all-CPU?", which the `all_cpu` vs `partial_*` comparison answers. Do **not** invent a fake GPU match to differentiate them.

- [ ] **Step 2: Run the gate for P3**

Run: `pytest tests/bench/m9_crossing/test_smoke.py -v -k asof_compute`
Expected: 3 PASS.

- [ ] **Step 3: Lint + commit**

```bash
ruff check tests/bench/m9_crossing/ && ruff format tests/bench/m9_crossing/
git add tests/bench/m9_crossing/_pipelines.py
git commit -m "M9: add P3 as-of join->compute (sort-merge, 3 paths)"
```

---

## Task 6: P4 (hash-equi-join → compute) — the boundary

**Files:**
- Modify: `tests/bench/m9_crossing/_pipelines.py`

P4 documents the no-resident-path boundary: a relational equi-join of a left table (`key`, `x`) to a right table (`key`, `y`) on `key`, then `out = sqrt(x*x + y*y)` on the joined rows. The join is a **hash build+probe** (Polars `.join`), which has no GPU path. Paths: `all_cpu`, `partial_naive`, `partial_smart`. Expected finding: partial-GPU does not beat all-CPU (the join dominates and is CPU-bound).

- [ ] **Step 1: Add P4 to `_pipelines.py`**

```python
# ---------- P4: hash equi-join -> compute ----------

def _p4_make(n: int, seed: int = 0x94) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    # right has unique keys 0..n-1; left references them (many-to-one), so the join is 1:1 per left row.
    left = pl.DataFrame({
        "key": rng.integers(0, n, size=n).astype(np.int64),
        "x": rng.standard_normal(n).astype(np.float32),
    })
    right = pl.DataFrame({
        "key": np.arange(n, dtype=np.int64),
        "y": rng.standard_normal(n).astype(np.float32),
    })
    return {"left": left, "right": right}


def _p4_joined_xy(inp) -> tuple[np.ndarray, np.ndarray]:
    j = inp["left"].join(inp["right"], on="key", how="inner")
    return j["x"].to_numpy(), j["y"].to_numpy()


def _p4_all_cpu(inp):
    x, y = _p4_joined_xy(inp)
    return np.sort(np.sqrt(x * x + y * y))  # sort: join output order is not defined


def _p4_partial_naive(inp):
    x, y = _p4_joined_xy(inp)  # hash join on CPU (no GPU path)
    gx, gy = to_gpu(x), to_gpu(y)
    return np.sort(to_cpu(mx.sqrt(gx * gx + gy * gy)))


def _p4_partial_smart(inp):
    x, y = _p4_joined_xy(inp)
    gx, gy = to_gpu(x), to_gpu(y)
    return np.sort(to_cpu(mx.sqrt(gx * gx + gy * gy)))


P4 = PipelineSpec(
    name="hashjoin_compute",
    family="hash",
    sizes=[1_000_000, 10_000_000],
    make_inputs=_p4_make,
    paths={
        "all_cpu": _p4_all_cpu,
        "partial_naive": _p4_partial_naive,
        "partial_smart": _p4_partial_smart,
    },
    check=_check_array,
)

PIPELINES.append(P4)
```

Note: the result is sorted because an inner join's output row order is not defined — sorting makes the correctness comparison order-independent. `partial_naive`/`partial_smart` are again near-identical (the join is wholly CPU; only the trivial elementwise post-compute is GPU-able) — this pipeline exists to *document* that partial-GPU can't beat all-CPU when the join is the work, not to differentiate crossing strategies.

- [ ] **Step 2: Run the gate for P4**

Run: `pytest tests/bench/m9_crossing/test_smoke.py -v -k hashjoin_compute`
Expected: 3 PASS.

- [ ] **Step 3: Run the full gate**

Run: `pytest tests/bench/m9_crossing/test_smoke.py -v`
Expected: all PASS (P1 4 + P2 4 + P3 3 + P4 3 = 14).

- [ ] **Step 4: Lint + commit**

```bash
ruff check tests/bench/m9_crossing/ && ruff format tests/bench/m9_crossing/
git add tests/bench/m9_crossing/_pipelines.py
git commit -m "M9: add P4 hash-join->compute (boundary, 3 paths)"
```

---

## Task 7: Driver (`run.py`) + emitter (`emit.py`)

**Files:**
- Create: `tests/bench/m9_crossing/run.py`
- Create: `tests/bench/m9_crossing/emit.py`
- Modify: `tests/bench/m9_crossing/test_harness.py` (emit test)

- [ ] **Step 1: Write `run.py`**

Create `tests/bench/m9_crossing/run.py`:
```python
"""Driver: time every pipeline path across sizes, fit the (alpha, beta) cost model,
emit the report. No timing logic of its own (delegates to m8 measure)."""

from __future__ import annotations

from dataclasses import dataclass

from tests.bench.m8_report._timing import measure
from tests.bench.m9_crossing._crossing import CostModel, fit_cost_model
from tests.bench.m9_crossing._pipelines import PIPELINES, PipelineSpec


@dataclass
class Row:
    pipeline: str
    family: str
    size: int
    path: str
    ms: float
    vs_all_cpu: float  # all_cpu_ms / this_ms  (>1 = faster than all-CPU)


def run(pipelines: list[PipelineSpec] = PIPELINES) -> tuple[list[Row], CostModel]:
    rows: list[Row] = []
    for p in pipelines:
        for size in p.sizes:
            inp = p.make_inputs(size)
            times = {
                name: measure(lambda fn=fn, inp=inp: fn(inp)) for name, fn in p.paths.items()
            }
            base = times["all_cpu"]
            for name, ms in times.items():
                rows.append(
                    Row(p.name, p.family, size, name, ms, base / ms)
                )
                print(f"{p.name:18s} N={size:>10,} {name:14s} {ms:9.2f}ms  {base / ms:6.2f}x vs all_cpu")
    cost = fit_cost_model()
    print(f"\ncost model: alpha={cost.alpha_ms_per_byte:.3e} ms/byte  beta={cost.beta_ms_per_crossing:.4f} ms/crossing")
    return rows, cost


def main() -> None:
    from tests.bench.m9_crossing.emit import write_report

    rows, cost = run()
    write_report(rows, cost, md_path="docs/crossing-tax-report.md", json_path="crossing-tax.json")
    print(f"\nWrote docs/crossing-tax-report.md + crossing-tax.json ({len(rows)} rows)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `emit.py`**

Create `tests/bench/m9_crossing/emit.py`:
```python
"""rows + cost model -> markdown report + JSON. No measurement."""

from __future__ import annotations

import json
import platform
from collections import defaultdict
from dataclasses import asdict

import polars as pl

from tests.bench.m9_crossing._crossing import CostModel


def _header() -> dict[str, str]:
    try:
        import mlx.core as mx

        mlxv = getattr(mx, "__version__", "unknown")
    except Exception:
        mlxv = "unavailable"
    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "polars_version": pl.__version__,
        "mlx_version": mlxv,
    }


def to_markdown(rows, cost: CostModel, header: dict[str, str]) -> str:
    lines = ["# polars-metal — crossing-tax benchmark (M9)", ""]
    lines.append("> Internal decision-input. Sizes the CPU<->GPU crossing tax on mixed compute+join")
    lines.append("> pipelines. Ratios are vs the all-CPU path (what `engine=\"metal\"` does today on a join).")
    lines.append("")
    lines.append("## Environment")
    for k, v in header.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Crossing cost model")
    lines.append("")
    lines.append("`crossing_ms ≈ alpha · bytes_crossed + beta · n_crossings`")
    lines.append("")
    lines.append(f"- **alpha** = {cost.alpha_ms_per_byte:.3e} ms/byte "
                 f"(≈ {1.0 / (cost.alpha_ms_per_byte * 1e9):.1f} GB/s round-trip)")
    lines.append(f"- **beta** = {cost.beta_ms_per_crossing:.4f} ms/crossing (fixed dispatch/sync)")
    lines.append("")
    by_pipe: dict[str, list] = defaultdict(list)
    for r in rows:
        by_pipe[r.pipeline].append(r)
    for pipe, prows in by_pipe.items():
        lines.append(f"## {pipe}  ({prows[0].family})")
        lines.append("")
        lines.append("| size | path | ms | × vs all_cpu |")
        lines.append("|---:|---|---:|---:|")
        for r in prows:
            lines.append(f"| {r.size:,} | {r.path} | {r.ms:.2f} | {r.vs_all_cpu:.2f}× |")
        lines.append("")
    lines.append("<!-- VERDICT: filled in Task 9 after a real run -->")
    lines.append("")
    return "\n".join(lines)


def to_json(rows, cost: CostModel, header: dict[str, str]) -> str:
    return json.dumps(
        {
            "header": header,
            "cost_model": {"alpha_ms_per_byte": cost.alpha_ms_per_byte,
                           "beta_ms_per_crossing": cost.beta_ms_per_crossing},
            "rows": [asdict(r) for r in rows],
        },
        indent=2,
    )


def write_report(rows, cost: CostModel, *, md_path: str, json_path: str) -> None:
    header = _header()
    with open(md_path, "w") as f:
        f.write(to_markdown(rows, cost, header))
    with open(json_path, "w") as f:
        f.write(to_json(rows, cost, header))
```

- [ ] **Step 3: Add an emit unit test** to `tests/bench/m9_crossing/test_harness.py` (imports at top):

```python
import json as _json

from tests.bench.m9_crossing._crossing import CostModel as _CM
from tests.bench.m9_crossing.emit import to_json, to_markdown
from tests.bench.m9_crossing.run import Row


def test_emit_wellformed():
    rows = [Row("retrieve_rerank", "gather", 1000, "resident", 5.0, 3.0),
            Row("retrieve_rerank", "gather", 1000, "all_cpu", 15.0, 1.0)]
    cm = _CM(alpha_ms_per_byte=1e-7, beta_ms_per_crossing=0.05)
    md = to_markdown(rows, cm, {"machine": "arm64"})
    assert "crossing cost model" in md.lower()
    assert "retrieve_rerank" in md and "resident" in md
    parsed = _json.loads(to_json(rows, cm, {"machine": "arm64"}))
    assert parsed["rows"][0]["path"] == "resident"
    assert parsed["cost_model"]["beta_ms_per_crossing"] == 0.05
```

- [ ] **Step 4: Run harness tests**

Run: `pytest tests/bench/m9_crossing/test_harness.py -v`
Expected: PASS (crossing + emit tests).

- [ ] **Step 5: Lint + commit**

```bash
ruff check tests/bench/m9_crossing/ && ruff format tests/bench/m9_crossing/
git add tests/bench/m9_crossing/run.py tests/bench/m9_crossing/emit.py tests/bench/m9_crossing/test_harness.py
git commit -m "M9: driver (run) + emitter (report+json) with cost-model output"
```

---

## Task 8: `make crossing-report` + wire smoke into `test-unit`

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Inspect the Makefile**

Run: `grep -n "test-unit\|perf-report\|^.PHONY" Makefile` and read the `test-unit` + `perf-report` recipe blocks (note the M8 smoke line already appended to `test-unit`).

- [ ] **Step 2: Edit the Makefile**

(a) Add `crossing-report` to `.PHONY`.
(b) Add the target near `perf-report` (literal TAB indent):
```makefile
crossing-report:
	python -m tests.bench.m9_crossing.run
```
(c) Append the M9 smoke+harness gate to the END of the existing `test-unit` recipe (a new line after the M8 line, matching its style):
```makefile
	pytest tests/bench/m9_crossing/test_smoke.py tests/bench/m9_crossing/test_harness.py -q
```

- [ ] **Step 3: Verify**

Run: `make -n crossing-report` (expect it to echo `python -m tests.bench.m9_crossing.run`) and `make -n test-unit` (expect the M9 line present, no Makefile syntax error).
Run: `pytest tests/bench/m9_crossing/test_smoke.py tests/bench/m9_crossing/test_harness.py -q` — expect all PASS.

- [ ] **Step 4: Commit**

```bash
git add Makefile
git commit -m "M9: make crossing-report target + smoke gate wired into test-unit"
```

---

## Task 9: Real run + verdict

**Files:**
- Generate + edit: `docs/crossing-tax-report.md`, `crossing-tax.json`

- [ ] **Step 1: Generate the report**

Run: `make crossing-report`
Expected: per-row timings printed, cost model printed, `docs/crossing-tax-report.md` + `crossing-tax.json` written. Capture stdout. (Heed the M8 variance lesson: if any `vs_all_cpu` ratio is wildly non-monotonic across sizes, re-run — in-process numpy/MLX baselines vary; medians of 7 should be stable.)

- [ ] **Step 2: Sanity-check**

Run: `head -50 docs/crossing-tax-report.md` — confirm the cost model (α, β, GB/s) and per-pipeline tables are populated.

- [ ] **Step 3: Write the verdict** — replace the `<!-- VERDICT ... -->` marker in `docs/crossing-tax-report.md` with prose grounded in the measured rows + cost model. It MUST resolve to one of the three outcomes and state the crossing-volume threshold the (α, β) model implies:

```markdown
## Verdict

### Per-pipeline read
- **P1 retrieve_rerank (gather):** partial_smart <X>× vs all_cpu; resident <Y>×. partial_naive <Z>× (crossing the full QxN matrix). <one line>
- **P2 fact_dim_chain (gather):** <numbers + one line>
- **P3 asof_compute (sort-merge):** <numbers> — does CPU-match + GPU-compute beat all-CPU?
- **P4 hashjoin_compute (hash):** <numbers> — confirms partial-GPU <= all_cpu (the join is the work).

### The cost-model rule
With alpha=<..> ms/byte (<..> GB/s) and beta=<..> ms/crossing, free per-op routing wins when the
GPU compute saved exceeds alpha·bytes_crossed + beta·n_crossings — i.e. roughly when
<crossing volume threshold derived from the numbers>.

### Decision (pick one)
- **(a)** free-routing (partial_smart) beats all-CPU broadly -> M10 = build the boundary-aware router.
- **(b)** resident >> partial_smart only for gather -> M10 = narrower resident-gather build.
- **(c)** crossings/joins dominate (partial_smart <= all_cpu) -> drop; re-center on hardening existing wins.
State which, with the evidence.
```

- [ ] **Step 4: Commit**

```bash
git add docs/crossing-tax-report.md crossing-tax.json
git commit -m "M9: generate crossing-tax report + verdict"
```

- [ ] **Step 5: Final gate**

Run: `make gate`
Expected: green (the slow `crossing-report` is not part of `gate`; the M9 smoke gate under `test-unit` is). Confirm conformance shows only the documented pre-existing deviations.

---

## Self-review notes (addressed)

- **Spec coverage:** §3 pipelines P1–P4 → Tasks 2,4,5,6. §4 paths (all_cpu/partial_naive/partial_smart/resident) → implemented per pipeline (resident only P1/P2 per spec). §5 α/β cost model → Task 1 (`fit_cost_model`) + surfaced in Task 7 emit + Task 9 rule. §6 components → Tasks 1,2,7,8. §7 testing (paths-agree gate + harness units + out-of-gate full run) → Tasks 3,7,8. §9 done/verdict (3 outcomes + volume threshold) → Task 9.
- **No engine changes:** every file is under `tests/bench/m9_crossing/` or the Makefile — the walker/router/UDF are untouched, per spec §8 guardrail 1.
- **Honest baselines:** CPU = numpy/Polars (competent); GPU = raw MLX; crossing = real `mx.array`/`np.array` memcpy. Variance caveat carried into Task 9 (the M8 lesson).
- **Type consistency:** `PipelineSpec` fields (name/family/sizes/make_inputs/paths/check) used identically across P1–P4; `Row` fields (pipeline/family/size/path/ms/vs_all_cpu) match between `run.py` and `emit.py` and the emit test; `CostModel` (alpha_ms_per_byte/beta_ms_per_crossing/predict) consistent across `_crossing.py`, `run.py`, `emit.py`.
- **Honest P3/P4 framing:** their partial_naive/partial_smart are near-identical *by design* (the match is CPU-irreducible) — flagged in-task so the implementer doesn't fabricate a fake GPU match to differentiate them.
