# M7 Workstream A — Python `.metal` Namespace Spine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the 6 copy-pasted `.metal` verb detect/dispatch/cache/sentinel templates in `python/polars_metal/` into one parameterized spine (~525 → ~165 LOC), and make the per-verb null/error/streaming contracts *intentional and documented* — correcting only the accidental inconsistencies, each pinned by a test.

**Architecture:** Two phases. **A1 (contract-first)** pins each verb's current null/boundary-error/streaming behavior with characterization tests, writes a verb-contract doc, then corrects only the accidental inconsistencies (`RuntimeError` → `ComputeError`; document FFT's no-cache divergence and rolling/dt's silent-CPU-on-streaming divergence). **A2 (collapse to spine)** adds shared infra to `_detect_common.py` (a generic `CaptureCache`, a `SentinelBinding` dataclass, a parameterized sentinel parser, a shared candidate-iteration scaffold, and a sentinel-field builder), then migrates each verb onto it one at a time (each migration gate-green and independently revertable), and finally replaces the 5 copy-paste dispatch blocks in `__init__.py` with a loop-driven registry.

**Tech Stack:** Python 3, Polars (LazyFrame plugin via `post_opt_callback`), pytest. No Rust in this workstream. Protected by C1's Python differential slice (`tests/diff/`) plus the existing per-verb `tests/python_integration/` suites.

---

## Background the engineer needs

The `.metal` namespace has **6 verbs**, each implemented as a serialize-detect → collect-and-stitch pipeline:

| Verb | namespace module | detect module | dispatch module | family |
|------|------------------|---------------|-----------------|--------|
| cosine_topk / knn | `_vector_namespace.py` | `_vector_detect.py` | `_vector_dispatch.py` | struct-sentinel (capture) |
| fft / ifft | `_fft_namespace.py` | `_fft_detect.py` | `_fft_dispatch.py` | struct-sentinel (no cache; op inlined) |
| dtw | `_dtw_namespace.py` | `_dtw_detect.py` | `_dtw_dispatch.py` | struct-sentinel (capture) |
| corr | `_corr_namespace.py` | `_corr_detect.py` | `_corr_dispatch.py` | struct-sentinel (capture, frame-replacing) |
| rolling_{mean,sum,var,std} | (none) | `_rolling_detect.py` | `_rolling_dispatch.py` | native-expr (schema-driven) |
| dt.{year,month,day} | (none) | `_dt_detect.py` | `_dt_dispatch.py` | native-expr (schema-driven) |

**Two detect families:**
- **struct-sentinel** (vector/fft/dtw/corr): the namespace wraps the user expr in a `pl.struct([...])` carrying a tagged `Int64` literal (a cache handle or an op code). Detection finds the tag in the serialized plan and pulls `(col, payload)` out of the tagged alias.
- **native-expr** (rolling/dt): detection recognizes a *native* Polars `Function` node (`RollingExpr` / `TemporalExpr`) and validates it against the schema. No sentinel, no struct.

Both families share the same **candidate-iteration scaffold**: a fast path over exprs captured by a `LazyFrame.with_columns` monkey-patch, and a slow fallback that `explain()`-pre-filters then parses the last `"exprs":[...]` fragment of the serialized plan. `_detect_common.py` already factored out the JSON-walk helpers (`_alias_name`, `_struct_fields`, `_literal_int`), the cache machinery (`install_with_columns_capture`, `lookup`, `_make_evictor`). It does **not** yet factor out the candidate-iteration scaffold, the sentinel parser, the binding dataclass, the capture cache, or the sentinel builder — that is this plan's A2.

**Current contract reality (verified 2026-06-12, the baseline A1 pins):**

| Verb | null input | boundary error type | streaming=True |
|------|-----------|---------------------|----------------|
| vector | **raise** `ValueError` | `ValueError` (dtype/dim) + **`RuntimeError`** (handle missing, `_vector_dispatch.py:82`) | **raise** `ComputeError` (shared guard) |
| fft | **raise** `ValueError` | `ValueError` | **raise** `ComputeError` (shared guard) |
| dtw | **mask+restore** | `ValueError` (dtype/shape/NaN) + **`RuntimeError`** (handle missing `:67`; missing dtaidistance `:33`) | **raise** `ComputeError` (shared guard) |
| corr | **CPU fallback** | `ComputeError` (N<2) + `ValueError` (non-numeric) + **`RuntimeError`** (handle missing `:86`) | **raise** `ComputeError` (shared guard) |
| rolling | **CPU fallback** | `ValueError` | **silent CPU fallback** (`[] if streaming`) |
| dt | **mask+restore** | (validation rejects → CPU) | **silent CPU fallback** (`[] if streaming`) |

**The accidental inconsistencies to fix (A1):**
1. `RuntimeError` for the "handle missing (already consumed?)" internal-state errors (vector `:82`, dtw `:67`, corr `:86`) and the missing-dtaidistance error (dtw `:33`) → `pl.exceptions.ComputeError`, per the CLAUDE.md convention "Errors propagate as `PolarsError::ComputeError` at the engine boundary."

**The divergences that are LEGITIMATE — document, do NOT force uniform (A1):**
- Null handling differs *by design*: corr falls back to CPU (pairwise-complete correlation is well-defined); dtw masks+restores (row semantics); vector/fft raise (a null in an embedding/signal is meaningless). Keep as-is; document.
- The dtype-validation `ValueError`s (wrong dtype, dim mismatch, non-numeric) are **user-argument validation**, kept as `ValueError`. Only the internal "handle missing" / environment "missing library" `RuntimeError`s are converted. Document the rule: *user-input validation → `ValueError`; engine-internal / boundary failure → `ComputeError`.*
- FFT has **no capture cache** (its op code is inlined in the sentinel literal), so it has no handle-evicted guard — this is correct, not a missing guard. Document it; the survey's "FFT missing the guard" was a misread.
- Streaming: vector/fft/dtw/corr **raise** (no CPU implementation exists); rolling/dt **silently fall back to CPU** (they *have* CPU equivalents). This is a legitimate divergence — document it.

**Guardrails (from the spec §4):** behavior-preserving except the test-pinned contract corrections; `make gate` green at every step; run `ruff` per-task (don't defer to the final gate — see the subagent-fmt-drift lesson); do **not** touch `_walker.py` / `_udf.py` / `_fusion_analyzer.py` (out of A scope).

**Commands:**
- Per-verb tests: `make wheel && python -m pytest tests/python_integration/test_<verb>*.py -v` (the namespace verbs need the built wheel; `make wheel` = `maturin develop --release`).
- Lint: `ruff check python/ && ruff format python/`
- Full gate: `make gate`
- Differential net: `make test-diff`

> **Note on `make wheel`:** Workstream A is pure Python, but the verbs dispatch into the compiled engine, so the integration tests require the installed wheel. If the wheel is already built and only Python changed, `maturin develop` (without `--release`) is faster for iteration; run `make wheel` (release) before the final gate.

---

# Phase A1 — Contract-first

## Task 1: Characterization tests pinning current verb contracts

**Files:**
- Create: `tests/python_integration/test_metal_namespace_contracts.py`

These tests assert the **current** behavior (the baseline table above). They should PASS against today's code — they lock reality before any refactor. The two corrections in Task 3 will flip specific assertions.

- [ ] **Step 1: Write the characterization tests**

```python
"""Characterization tests pinning the .metal namespace verb contracts on
three axes: null handling, boundary error type, and streaming. These lock
CURRENT behavior before the M7-A consolidation refactor. Intentional
divergences (corr CPU-fallback vs vector raise; rolling silent-CPU vs vector
raise-on-streaming) are pinned here as deliberate, not bugs.

See docs/metal-namespace-contracts.md for the documented contract.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polars_metal  # noqa: F401  (registers the .metal namespace + collect patch)

ComputeError = pl.exceptions.ComputeError


# ---------- null handling ----------

def test_vector_nulls_raise():
    corpus = pl.DataFrame({"emb": [[1.0, 0.0], [0.0, 1.0]]}).select(
        pl.col("emb").cast(pl.Array(pl.Float32, 2))
    )
    q = pl.LazyFrame({"emb": [[1.0, 0.0], None]}).select(
        pl.col("emb").cast(pl.Array(pl.Float32, 2))
    )
    with pytest.raises((ValueError, ComputeError)):
        q.with_columns(
            pl.col("emb").metal.cosine_topk(corpus, k=1, corpus_col="emb")
        ).collect(engine="metal")


def test_fft_nulls_raise():
    lf = pl.LazyFrame({"x": [1.0, 2.0, None, 4.0]}).select(
        pl.col("x").cast(pl.Float32)
    )
    with pytest.raises((ValueError, ComputeError)):
        lf.with_columns(pl.col("x").metal.fft().alias("f")).collect(engine="metal")


def test_corr_nulls_fall_back_to_cpu():
    # corr tolerates nulls by routing to CPU — no raise, finite result.
    lf = pl.LazyFrame(
        {"a": [1.0, 2.0, 3.0, None], "b": [2.0, 4.0, 6.0, 8.0]}
    )
    out = lf.metal.corr().collect(engine="metal")
    assert out.shape == (2, 2)


def test_rolling_nulls_fall_back_to_cpu():
    lf = pl.LazyFrame({"x": [1.0, None, 3.0, 4.0, 5.0]}).select(
        pl.col("x").cast(pl.Float32)
    )
    out = lf.with_columns(
        pl.col("x").rolling_mean(window_size=2).alias("r")
    ).collect(engine="metal")
    cpu = lf.with_columns(
        pl.col("x").rolling_mean(window_size=2).alias("r")
    ).collect(engine="cpu")
    assert out.equals(cpu)


def test_dt_nulls_restored():
    lf = pl.LazyFrame({"d": [pl.date(2021, 1, 1), None]})
    out = lf.with_columns(pl.col("d").dt.year().alias("y")).collect(engine="metal")
    cpu = lf.with_columns(pl.col("d").dt.year().alias("y")).collect(engine="cpu")
    assert out.equals(cpu)


# ---------- streaming ----------

@pytest.mark.parametrize(
    "build",
    [
        lambda lf, corpus: lf.with_columns(
            pl.col("emb").metal.cosine_topk(corpus, k=1, corpus_col="emb")
        ),
    ],
)
def test_vector_streaming_raises(build):
    corpus = pl.DataFrame({"emb": [[1.0, 0.0]]}).select(
        pl.col("emb").cast(pl.Array(pl.Float32, 2))
    )
    lf = pl.LazyFrame({"emb": [[1.0, 0.0]]}).select(
        pl.col("emb").cast(pl.Array(pl.Float32, 2))
    )
    with pytest.raises(ComputeError):
        build(lf, corpus).collect(engine="metal", streaming=True)


def test_fft_streaming_raises():
    lf = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0]}).select(pl.col("x").cast(pl.Float32))
    with pytest.raises(ComputeError):
        lf.with_columns(pl.col("x").metal.fft().alias("f")).collect(
            engine="metal", streaming=True
        )


def test_corr_streaming_raises():
    lf = pl.LazyFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    with pytest.raises(ComputeError):
        lf.metal.corr().collect(engine="metal", streaming=True)


def test_rolling_streaming_silent_cpu_fallback():
    # rolling HAS a CPU implementation, so streaming silently runs on CPU
    # (no raise). This divergence from vector/fft/dtw/corr is intentional.
    lf = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}).select(
        pl.col("x").cast(pl.Float32)
    )
    out = lf.with_columns(
        pl.col("x").rolling_mean(window_size=2).alias("r")
    ).collect(engine="metal", streaming=True)
    cpu = lf.with_columns(
        pl.col("x").rolling_mean(window_size=2).alias("r")
    ).collect(engine="cpu")
    assert out.equals(cpu)


def test_dt_streaming_silent_cpu_fallback():
    lf = pl.LazyFrame({"d": [pl.date(2021, 1, 1), pl.date(2022, 6, 15)]})
    out = lf.with_columns(pl.col("d").dt.year().alias("y")).collect(
        engine="metal", streaming=True
    )
    cpu = lf.with_columns(pl.col("d").dt.year().alias("y")).collect(engine="cpu")
    assert out.equals(cpu)


# ---------- fft has no capture cache (repeated collect must work) ----------

def test_fft_repeated_collect_no_eviction():
    # fft inlines its op code in the sentinel literal (no handle, no cache),
    # so repeated collects of the same lf must both succeed. Pins the
    # "fft has no handle-evicted guard, by design" contract.
    lf = pl.LazyFrame({"x": [1.0, 2.0, 3.0, 4.0]}).select(pl.col("x").cast(pl.Float32))
    built = lf.with_columns(pl.col("x").metal.fft().alias("f"))
    first = built.collect(engine="metal")
    second = built.collect(engine="metal")
    assert first.equals(second)
```

- [ ] **Step 2: Run the tests — they pin current behavior, so they PASS**

Run: `make wheel && python -m pytest tests/python_integration/test_metal_namespace_contracts.py -v`
Expected: ALL PASS. If any fails, the baseline table is wrong for that verb — STOP and reconcile the table with reality before continuing (do not "fix" the verb; update the test to match observed behavior and note the discrepancy).

> If a verb's GPU path is environment-gated (no Metal device in CI), guard the GPU-only asserts with the same skip the existing `tests/python_integration/test_<verb>*.py` use. Check how `test_vector_search.py` / `test_fft.py` skip, and mirror it.

- [ ] **Step 3: Commit**

```bash
git add tests/python_integration/test_metal_namespace_contracts.py
git commit -m "M7 A-1: characterization tests pinning .metal verb contracts (null/error/streaming)"
```

---

## Task 2: Write the verb-contract doc

**Files:**
- Create: `docs/metal-namespace-contracts.md`

- [ ] **Step 1: Write the contract doc**

```markdown
# `.metal` namespace — per-verb contracts

The `.metal` namespace verbs are serialize-detected and run on the GPU via a
collect-and-stitch pipeline. Their contracts differ **by design** along three
axes — null handling, boundary error type, streaming — because the verbs have
different semantics and different CPU-fallback availability. This document is
the source of truth; the characterization tests in
`tests/python_integration/test_metal_namespace_contracts.py` pin it.

## The three axes

| Verb | null input | boundary error | streaming=True |
|------|-----------|----------------|----------------|
| `cosine_topk` / `knn` | **raise** — a null in an embedding is meaningless | `ValueError` (user input: dtype/dim) · `ComputeError` (engine internal) | **raise** `ComputeError` |
| `fft` / `ifft` | **raise** — a null in a signal is meaningless | `ValueError` (user input: dtype) · `ComputeError` (engine internal) | **raise** `ComputeError` |
| `dtw` | **mask + restore** — null rows pass through positionally | `ValueError` (user input: dtype/shape/NaN cell) · `ComputeError` (engine internal / missing optional dep) | **raise** `ComputeError` |
| `corr` | **CPU fallback** — pairwise-complete correlation is well-defined | `ValueError` (non-numeric col) · `ComputeError` (N<2, engine internal) | **raise** `ComputeError` |
| `rolling_{mean,sum,var,std}` | **CPU fallback** | `ValueError` (user input) | **silent CPU fallback** |
| `dt.{year,month,day}` | **mask + restore** | validation reject → CPU | **silent CPU fallback** |

## Why the divergences are intentional

- **Null handling is semantic.** corr's pairwise-complete behavior is a real,
  well-defined statistic, so it routes to CPU rather than refusing. dtw's rows
  are sequences; a null *row* is a missing sequence and is restored
  positionally (a null *cell* inside a non-null sequence is still an error —
  the GPU kernel can't match NaN against the oracle). For vector search and
  FFT a null inside an embedding/signal has no meaning, so they refuse.

- **Error-type rule.** *User-input validation* (wrong dtype, dimension
  mismatch, non-numeric column) raises `ValueError`. *Engine-internal or
  boundary failures* (a capture handle that went missing, an N<2 frame, a
  missing optional dependency) raise `pl.exceptions.ComputeError`, per the
  engine convention that boundary errors look native to Polars users.

- **Streaming.** The vector/fft/dtw/corr kernels have **no CPU
  implementation**, so requesting `streaming=True` over a plan that contains
  one of their sentinels raises `ComputeError` rather than silently producing
  a wrong/absent result. rolling and dt **do** have exact CPU equivalents, so
  under streaming they silently fall back to CPU — the user still gets the
  correct answer, just not on the GPU.

- **fft has no capture cache.** Unlike vector/dtw/corr, fft encodes its op code
  (`OP_FFT` / `OP_IFFT`) directly in the sentinel's `Int64` literal — there is
  no by-reference spec to cache and no handle to evict. Repeated collects of
  the same LazyFrame therefore always succeed. fft correctly has no
  handle-evicted guard; this is not a missing feature.

## Internal cache lifetime (vector / dtw / corr)

These verbs capture a by-reference spec (corpus / reference sequence / column
set) in a per-verb `CaptureCache` keyed by an `Int64` handle embedded in the
sentinel. The dispatcher registers a `weakref.finalize(lf, evict, handle)` so
the spec is freed when the dispatched LazyFrame is GC'd, while repeated
collects of a *live* lf reuse it. If a handle is missing at dispatch time
(evicted early), the verb raises `ComputeError("... handle missing ...")`.
```

- [ ] **Step 2: Commit**

```bash
git add docs/metal-namespace-contracts.md
git commit -m "M7 A-1: document .metal verb contracts (null/error/streaming, intentional divergences)"
```

---

## Task 3: Correct the accidental inconsistency — `RuntimeError` → `ComputeError`

**Files:**
- Modify: `python/polars_metal/_vector_dispatch.py` (the `RuntimeError("... handle missing ...")` ~line 82)
- Modify: `python/polars_metal/_dtw_dispatch.py` (handle-missing ~line 67; missing-dtaidistance ~line 33)
- Modify: `python/polars_metal/_corr_dispatch.py` (handle-missing ~line 86)
- Modify: `tests/python_integration/test_metal_namespace_contracts.py` (add the handle-missing tests)

- [ ] **Step 1: Add failing tests asserting `ComputeError` on a missing handle**

Append to `tests/python_integration/test_metal_namespace_contracts.py`:

```python
# ---------- handle-missing → ComputeError (A1 correction) ----------

def _evict_all(cache_module):
    # helper: simulate the race where the spec was GC'd before dispatch
    cache = cache_module._CACHE  # type: ignore[attr-defined]
    for h in list(cache._specs.keys()):
        cache.evict(h)


def test_vector_handle_missing_raises_compute_error():
    import polars_metal._vector_namespace as ns

    corpus = pl.DataFrame({"emb": [[1.0, 0.0]]}).select(
        pl.col("emb").cast(pl.Array(pl.Float32, 2))
    )
    lf = pl.LazyFrame({"emb": [[1.0, 0.0]]}).select(
        pl.col("emb").cast(pl.Array(pl.Float32, 2))
    )
    built = lf.with_columns(pl.col("emb").metal.cosine_topk(corpus, k=1, corpus_col="emb"))
    _evict_all(ns)  # drop the captured corpus spec before collect
    with pytest.raises(ComputeError):
        built.collect(engine="metal")


def test_corr_handle_missing_raises_compute_error():
    import polars_metal._corr_namespace as ns

    lf = pl.LazyFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    built = lf.metal.corr()
    _evict_all(ns)
    with pytest.raises(ComputeError):
        built.collect(engine="metal")
```

> The `_CACHE` / `_specs` names match the `CaptureCache` introduced in Task 5. If executing Task 3 *before* Task 5 (recommended A1-before-A2 order), the cache is still the per-verb dict (`_CORPUS_CACHE` etc.) — write `_evict_all` against whatever the cache currently is, and update it in Task 5 when the cache type changes. Simplest: in this task, evict via the module's existing `evict_capture` over the live handles. If the current handle set isn't readily enumerable, instead assert the conversion by a direct unit call: import the dispatch's run path and pass a binding whose `payload`/`handle` is a value never captured (e.g. `10_000_000`), then assert `ComputeError`. Pick whichever is reachable in the current code and leave a comment.

- [ ] **Step 2: Run the new tests — expect FAIL (`RuntimeError`, not `ComputeError`)**

Run: `python -m pytest tests/python_integration/test_metal_namespace_contracts.py -k handle_missing -v`
Expected: FAIL — raises `RuntimeError`, which is not a subclass of `pl.exceptions.ComputeError`.

- [ ] **Step 3: Convert the four `RuntimeError` sites**

In `python/polars_metal/_vector_dispatch.py`, change:
```python
raise RuntimeError("polars_metal: vector-search corpus handle missing (already consumed?)")
```
to:
```python
raise pl.exceptions.ComputeError(
    "polars_metal: vector-search corpus handle missing (already consumed?)"
)
```
(Ensure `import polars as pl` is present at the top of the file; add it if absent.)

In `python/polars_metal/_dtw_dispatch.py`, change the handle-missing site:
```python
raise RuntimeError("polars_metal: dtw spec handle missing (already consumed?)")
```
to:
```python
raise pl.exceptions.ComputeError(
    "polars_metal: dtw spec handle missing (already consumed?)"
)
```
and the missing-dependency site:
```python
raise RuntimeError(
    "polars_metal: .metal.dtw allow_cpu_fallback=True needs the 'dtaidistance' "
    "package for unsupported shapes; install it (pip install dtaidistance)."
)
```
to:
```python
raise pl.exceptions.ComputeError(
    "polars_metal: .metal.dtw allow_cpu_fallback=True needs the 'dtaidistance' "
    "package for unsupported shapes; install it (pip install dtaidistance)."
)
```
(Ensure `import polars as pl` at the top of `_dtw_dispatch.py`.)

In `python/polars_metal/_corr_dispatch.py`, change:
```python
raise RuntimeError("polars_metal: corr spec handle missing (already consumed?)")
```
to:
```python
raise pl.exceptions.ComputeError(
    "polars_metal: corr spec handle missing (already consumed?)"
)
```
(`_corr_dispatch.py` already imports `pl` — it uses `pl.exceptions.ComputeError` for the N<2 case.)

- [ ] **Step 4: Verify no stray `RuntimeError` remains in the dispatch modules**

Run: `grep -rn "RuntimeError" python/polars_metal/_*_dispatch.py`
Expected: no matches.

- [ ] **Step 5: Run the tests — expect PASS**

Run: `python -m pytest tests/python_integration/test_metal_namespace_contracts.py -v`
Expected: ALL PASS (including the handle-missing cases now raising `ComputeError`).

- [ ] **Step 6: ruff + commit**

```bash
ruff check python/ && ruff format python/
git add python/polars_metal/_vector_dispatch.py python/polars_metal/_dtw_dispatch.py \
        python/polars_metal/_corr_dispatch.py tests/python_integration/test_metal_namespace_contracts.py
git commit -m "M7 A-1: convert handle-missing/missing-dep RuntimeError -> ComputeError (test-pinned)"
```

---

# Phase A2 — Collapse to one spine

## Task 4: Add the shared spine to `_detect_common.py`

This task is purely **additive** — new code in `_detect_common.py` plus unit tests. No verb is migrated yet, so every existing test stays green.

**Files:**
- Modify: `python/polars_metal/_detect_common.py`
- Modify: `tests/python_integration/test_detect_common.py`

- [ ] **Step 1: Write unit tests for the new spine primitives**

Append to `tests/python_integration/test_detect_common.py`:

```python
import itertools

import polars_metal._detect_common as dc


def test_capture_cache_roundtrip():
    cache = dc.CaptureCache()
    h1 = cache.capture("spec-a")
    h2 = cache.capture("spec-b")
    assert h1 != h2
    assert cache.get(h1) == "spec-a"
    assert cache.get(h2) == "spec-b"
    cache.evict(h1)
    assert cache.get(h1) is None
    assert cache.get(h2) == "spec-b"
    cache.evict(99999)  # evicting an absent handle is a no-op


def test_capture_cache_handles_isolated_per_instance():
    a = dc.CaptureCache()
    b = dc.CaptureCache()
    ha = a.capture("x")
    hb = b.capture("y")
    # independent counters: both start at 1, no cross-talk
    assert a.get(hb) is None or hb not in a._specs
    assert b.get(ha) is None or ha not in b._specs


def test_sentinel_binding_fields():
    b = dc.SentinelBinding(out_name="o", col="c", payload=7)
    assert (b.out_name, b.col, b.payload) == ("o", "c", 7)


def test_make_sentinel_parser_prefix():
    # struct node: {"Function": as_struct over fields}, one field aliased
    # "<TAG><col>" carrying an Int64 literal.
    tag = "__pm_test__"
    parse = dc.make_sentinel_parser(tag)
    fields = [
        {"Alias": [{"Column": "x"}, "__pm_in"]},
        {"Alias": [{"Literal": {"Scalar": {"Int64": 42}}}, f"{tag}myCol"]},
    ]
    node = {"Function": {"input": fields, "function": {"AsStruct": None}}}
    b = parse(node, "out")
    assert b == dc.SentinelBinding(out_name="out", col="myCol", payload=42)


def test_make_sentinel_parser_exact():
    tag = "__pm_corr__"
    parse = dc.make_sentinel_parser(tag, exact=True)
    fields = [{"Alias": [{"Literal": {"Scalar": {"Int64": 5}}}, tag]}]
    node = {"Function": {"input": fields, "function": {"AsStruct": None}}}
    b = parse(node, "out")
    assert b == dc.SentinelBinding(out_name="out", col="", payload=5)


def test_make_sentinel_parser_no_tag_returns_none():
    parse = dc.make_sentinel_parser("__pm_test__")
    assert parse({"Column": "x"}, "out") is None
```

> The exact JSON node shapes (`{"Function": {"input": [...], "function": {"AsStruct": ...}}}` and the `Alias` / `Literal.Scalar.Int64` shapes) must match what `_struct_fields`, `_alias_name`, and `_literal_int` already parse. **Before writing the test, confirm the shapes** by reading the existing `_struct_fields` / `_literal_int` in `_detect_common.py` and one real serialized sentinel (e.g. add a throwaway `print(pl.col("x").metal.fft().meta.serialize(format="json"))` in a scratch script). Adjust the literal node shapes in the test to match. The helpers already handle the real shape — the test just needs to feed them that shape.

- [ ] **Step 2: Run — expect FAIL (`AttributeError: module has no attribute 'CaptureCache'`)**

Run: `python -m pytest tests/python_integration/test_detect_common.py -v`
Expected: FAIL — the new symbols don't exist yet.

- [ ] **Step 3: Add the spine primitives to `_detect_common.py`**

Add near the top imports (if not already present): `import itertools`, `from dataclasses import dataclass`, `from typing import Any, Callable, Iterator`. Then append:

```python
# --------------------------------------------------------------------------
# M7-A spine: one capture cache, one sentinel binding, one parser, one
# candidate-iteration scaffold, one sentinel-field builder. Replaces the
# 4-6 near-identical detect modules + 3 cache triplets + 3 sentinel builders.
#
# NOTE: the spine does NOT own the _raise_cpu guard. Each verb keeps its own
# _raise_cpu stub (with its verb-specific ComputeError message, already pinned
# by match= patterns in test_vector_search/test_corr_engine/test_fft) and
# passes it to sentinel_fields(raise_fn=...). This keeps sentinels and raise
# messages byte-identical across the migration.
# --------------------------------------------------------------------------


class CaptureCache:
    """Handle -> by-reference spec registry shared by the capture-based
    .metal verbs (vector search, dtw, corr). Each verb instantiates its own
    cache so handle spaces stay isolated and specs stay typed. fft needs no
    cache (its op code is inlined in the sentinel literal)."""

    def __init__(self) -> None:
        self._specs: dict[int, Any] = {}
        self._counter = itertools.count(1)

    def capture(self, spec: Any) -> int:
        handle = next(self._counter)
        self._specs[handle] = spec
        return handle

    def get(self, handle: int) -> Any | None:
        return self._specs.get(handle)

    def evict(self, handle: int) -> None:
        self._specs.pop(handle, None)


@dataclass(frozen=True)
class SentinelBinding:
    """A detected struct-sentinel. ``col`` is the source column the tag was
    suffixed with (``""`` for corr's exact tag). ``payload`` is the Int64
    carried in the tagged literal: a cache handle (vector/dtw/corr) or an op
    code (fft)."""

    out_name: str
    col: str
    payload: int


def make_sentinel_parser(
    tag: str, *, exact: bool = False
) -> Callable[[dict, str], "SentinelBinding | None"]:
    """Return a parser ``(inner_json, out_name) -> SentinelBinding | None``.
    ``exact=False`` (default): the tag is a prefix and its suffix is the
    source column (vector/fft/dtw). ``exact=True``: the tag alias matches
    exactly and there is no source column (corr)."""

    def parse(inner_json: dict, out_name: str) -> "SentinelBinding | None":
        try:
            if tag not in json.dumps(inner_json):
                return None
            col = ""
            payload: int | None = None
            for fld in _struct_fields(inner_json):
                alias = _alias_name(fld)
                if exact:
                    if alias == tag:
                        payload = _literal_int(fld)
                elif alias and alias.startswith(tag):
                    col = alias[len(tag):]
                    payload = _literal_int(fld)
            if payload is None or (not exact and not col):
                return None
            return SentinelBinding(out_name=out_name, col=col, payload=payload)
        except Exception:
            return None

    return parse


def iter_candidate_nodes(
    lf: "pl.LazyFrame", *, cache: dict, explain_tags: tuple[str, ...]
) -> Iterator[tuple[dict, str]]:
    """Yield ``(inner_expr_json, out_name)`` for each top-level expression in
    ``lf`` that might carry a sentinel or native marker. Fast path: serialize
    each expr captured by the verb's with_columns monkey-patch (``cache``).
    Slow fallback: ``explain()``-pre-filter on ``explain_tags`` then parse the
    last ``"exprs":[...]`` fragment of the serialized plan. Any error -> stop
    (yields nothing further)."""
    try:
        cached = lookup(cache, lf)
        if cached is not None:
            for expr in cached:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    j = json.loads(expr.meta.serialize(format="json"))
                name = _alias_name(j)
                yield (j["Alias"][0] if name else j, name or "")
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            if not any(t in lf.explain() for t in explain_tags):
                return
            plan = lf.serialize(format="json")
        key = '"exprs":['
        i = plan.rfind(key)
        if i == -1:
            return
        start = i + len(key) - 1
        j = plan.rfind(',"options":', start)
        frag = plan[start:j] if j != -1 else plan[start:]
        nodes = json.loads(frag)
        for node in nodes if isinstance(nodes, list) else []:
            name = _alias_name(node)
            yield (node["Alias"][0] if name else node, name or "")
    except Exception:
        return


def sentinel_fields(
    expr: "pl.Expr",
    *,
    tag: str,
    payload: int,
    raise_alias: str,
    raise_fn: "Callable",
    col: str = "",
    in_alias: str | None = None,
    tag_exact: bool = False,
    raise_expr: "pl.Expr | None" = None,
) -> list["pl.Expr"]:
    """Build the struct field list every sentinel shares: an optional
    pass-through input field, the tagged Int64 payload literal, and the
    guard field (``raise_fn`` is the verb's own ``_raise_cpu`` stub, kept
    per-verb so its ComputeError message — pinned by match= patterns — is
    unchanged). Preserves the exact aliases/order of the pre-spine builders
    so serialized plans are byte-identical."""
    fields: list[pl.Expr] = []
    if in_alias is not None:
        fields.append(expr.alias(in_alias))
    fields.append(
        pl.lit(payload, dtype=pl.Int64).alias(tag if tag_exact else f"{tag}{col}")
    )
    raise_src = raise_expr if raise_expr is not None else expr
    fields.append(
        raise_src.map_batches(raise_fn, return_dtype=pl.Float32).alias(raise_alias)
    )
    return fields
```

> Confirm `json` and `warnings` are already imported at the top of `_detect_common.py` (the existing helpers use them). If not, add `import json` and `import warnings`.

- [ ] **Step 4: Run — expect PASS**

Run: `python -m pytest tests/python_integration/test_detect_common.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Confirm nothing else broke (spine is additive)**

Run: `python -m pytest tests/python_integration/ -q`
Expected: same pass/skip set as before this task (no regressions).

- [ ] **Step 6: ruff + commit**

```bash
ruff check python/ && ruff format python/
git add python/polars_metal/_detect_common.py tests/python_integration/test_detect_common.py
git commit -m "M7 A-2: add namespace spine to _detect_common (CaptureCache, SentinelBinding, sentinel parser/iter/fields)"
```

---

## Task 5: Migrate VECTOR onto the spine

**Files:**
- Modify (rewrite): `python/polars_metal/_vector_detect.py`
- Modify: `python/polars_metal/_vector_namespace.py` (cache triplet → `CaptureCache`; builder → `sentinel_fields`)
- Modify: `python/polars_metal/_vector_dispatch.py` (binding field reads `query_col`/`handle` → `col`/`payload`; binding import)

- [ ] **Step 1: Replace `_vector_detect.py` body with the spine wiring**

Read the current `_vector_detect.py` to confirm the with_columns capture attr name (`_polars_metal_vs_original_with_columns`) and the captured-expr cache variable. Then replace its parser + binding + find with:

```python
"""Serialize-detect .metal.cosine_topk/.knn struct sentinels in a LazyFrame.

The candidate-iteration scaffold and the struct-sentinel parser live in
_detect_common; this module only wires the vector-search tag and the
with_columns capture cache onto that spine.
"""
from __future__ import annotations

import polars as pl

from polars_metal import _detect_common as dc
from polars_metal._detect_common import SentinelBinding
from polars_metal._vector_namespace import SENTINEL_TAG

# Captured-expr cache: LazyFrame.with_columns monkey-patch records the exprs
# added to each lf so the fast path can serialize them individually.
_vs_exprs_cache: dict = {}
dc.install_with_columns_capture("_polars_metal_vs_original_with_columns", _vs_exprs_cache)

_parse = dc.make_sentinel_parser(SENTINEL_TAG)


def find_vector_bindings(lf: pl.LazyFrame) -> list[SentinelBinding]:
    out: list[SentinelBinding] = []
    for inner, name in dc.iter_candidate_nodes(
        lf, cache=_vs_exprs_cache, explain_tags=(SENTINEL_TAG,)
    ):
        b = _parse(inner, name)
        if b is not None and b.out_name:
            out.append(b)
    return out
```

> Preserve the **exact** attr name and any module-level capture setup the current file uses. If the current `_vector_detect.py` installs the capture under a different attr, use that name. The capture installation must remain functionally identical.

- [ ] **Step 2: Migrate the vector namespace cache + builder**

In `python/polars_metal/_vector_namespace.py`, replace the cache triplet:
```python
_CORPUS_CACHE: dict[int, CorpusSpec] = {}

def _capture_corpus(corpus, corpus_col, k, metric, query_col="") -> int:
    handle = next(_HANDLE_COUNTER)
    _CORPUS_CACHE[handle] = CorpusSpec(corpus, corpus_col, k, metric, query_col)
    return handle

def get_capture(handle: int) -> CorpusSpec | None:
    return _CORPUS_CACHE.get(handle)

def evict_capture(handle: int) -> None:
    _CORPUS_CACHE.pop(handle, None)
```
with:
```python
from polars_metal._detect_common import CaptureCache

_CACHE = CaptureCache()

def _capture_corpus(corpus, corpus_col, k, metric, query_col="") -> int:
    return _CACHE.capture(CorpusSpec(corpus, corpus_col, k, metric, query_col))

get_capture = _CACHE.get
evict_capture = _CACHE.evict
```
(Keep the `CorpusSpec` dataclass and `_HANDLE_COUNTER` removal — `_HANDLE_COUNTER` is now unused; delete it if nothing else references it.)

And replace the `build_sentinel` body:
```python
def build_sentinel(query_col_expr, query_col_name, handle):
    return pl.struct([
        query_col_expr.alias("__pm_vs_query"),
        pl.lit(handle, dtype=pl.Int64).alias(f"{SENTINEL_TAG}{query_col_name}"),
        query_col_expr.map_batches(_raise_cpu, return_dtype=pl.Float32).alias("__pm_vs_raise"),
    ])
```
with:
```python
from polars_metal._detect_common import sentinel_fields

def build_sentinel(query_col_expr, query_col_name, handle):
    return pl.struct(
        sentinel_fields(
            query_col_expr,
            tag=SENTINEL_TAG,
            payload=handle,
            col=query_col_name,
            in_alias="__pm_vs_query",
            raise_alias="__pm_vs_raise",
            raise_fn=_raise_cpu,
        )
    )
```
**Keep** the existing local `_raise_cpu` in `_vector_namespace.py` and pass it as `raise_fn=_raise_cpu` (above). Do NOT delete it or change its message — the `match=` pattern in `test_vector_search.py` pins that ComputeError text.

- [ ] **Step 3: Update `_vector_dispatch.py` binding field reads**

In `python/polars_metal/_vector_dispatch.py`:
- Replace the import `from polars_metal._vector_detect import VectorBinding` (if present) with `from polars_metal._detect_common import SentinelBinding` and update any type hints (`VectorBinding` → `SentinelBinding`).
- Replace every `b.query_col` → `b.col` and every `b.handle` → `b.payload` (and the same for any differently-named binding variable). Use grep to find them: `grep -n "\.query_col\|\.handle" python/polars_metal/_vector_dispatch.py`.

- [ ] **Step 4: Run the vector tests**

Run: `make wheel && python -m pytest tests/python_integration/test_vector_search.py tests/python_integration/test_metal_namespace_contracts.py -k "vector or handle_missing" -v`
Expected: ALL PASS (byte-identical sentinel ⇒ detection + dispatch unchanged).

- [ ] **Step 5: ruff + commit**

```bash
ruff check python/ && ruff format python/
git add python/polars_metal/_vector_detect.py python/polars_metal/_vector_namespace.py python/polars_metal/_vector_dispatch.py
git commit -m "M7 A-2: migrate vector search onto the namespace spine"
```

---

## Task 6: Migrate FFT onto the spine

**Files:**
- Modify (rewrite): `python/polars_metal/_fft_detect.py`
- Modify: `python/polars_metal/_fft_namespace.py` (builder → `sentinel_fields`)
- Modify: `python/polars_metal/_fft_dispatch.py` (binding field reads `input_col`/`op` → `col`/`payload`)

FFT has **no capture cache** — only the builder and detect change.

- [ ] **Step 1: Replace `_fft_detect.py` body**

```python
"""Serialize-detect .metal.fft()/.ifft() struct sentinels in a LazyFrame.

fft inlines its op code in the sentinel literal (no capture cache). The
candidate scaffold + parser come from _detect_common.
"""
from __future__ import annotations

import polars as pl

from polars_metal import _detect_common as dc
from polars_metal._detect_common import SentinelBinding
from polars_metal._fft_namespace import FFT_SENTINEL_TAG

_fft_exprs_cache: dict = {}
dc.install_with_columns_capture("_polars_metal_fft_original_with_columns", _fft_exprs_cache)

_parse = dc.make_sentinel_parser(FFT_SENTINEL_TAG)


def find_fft_bindings(lf: pl.LazyFrame) -> list[SentinelBinding]:
    out: list[SentinelBinding] = []
    for inner, name in dc.iter_candidate_nodes(
        lf, cache=_fft_exprs_cache, explain_tags=(FFT_SENTINEL_TAG,)
    ):
        b = _parse(inner, name)
        if b is not None and b.out_name:
            out.append(b)
    return out
```
(Confirm the with_columns attr name matches the current file.)

- [ ] **Step 2: Migrate the fft builder**

In `python/polars_metal/_fft_namespace.py`, replace `build_fft_sentinel`:
```python
from polars_metal._detect_common import sentinel_fields

def build_fft_sentinel(input_expr, input_col, op):
    return pl.struct(
        sentinel_fields(
            input_expr,
            tag=FFT_SENTINEL_TAG,
            payload=op,
            col=input_col,
            in_alias="__pm_fft_in",
            raise_alias="__pm_fft_raise",
            raise_fn=_raise_cpu,
        )
    )
```
**Keep** the existing local `_raise_cpu` and pass `raise_fn=_raise_cpu`. Do NOT delete it or change its message — the `match=` pattern in `test_fft.py` pins it.

- [ ] **Step 3: Update `_fft_dispatch.py` binding field reads**

`grep -n "\.input_col\|\.op\b" python/polars_metal/_fft_dispatch.py`, then `b.input_col` → `b.col`, `b.op` → `b.payload`. Update any `FftBinding` import/type-hint to `SentinelBinding`.

- [ ] **Step 4: Run the fft tests**

Run: `make wheel && python -m pytest tests/python_integration/test_fft.py tests/python_integration/test_metal_namespace_contracts.py -k fft -v`
Expected: ALL PASS (including `test_fft_repeated_collect_no_eviction`).

- [ ] **Step 5: ruff + commit**

```bash
ruff check python/ && ruff format python/
git add python/polars_metal/_fft_detect.py python/polars_metal/_fft_namespace.py python/polars_metal/_fft_dispatch.py
git commit -m "M7 A-2: migrate fft onto the namespace spine"
```

---

## Task 7: Migrate DTW onto the spine

**Files:**
- Modify (rewrite): `python/polars_metal/_dtw_detect.py`
- Modify: `python/polars_metal/_dtw_namespace.py` (cache triplet → `CaptureCache`; builder → `sentinel_fields`)
- Modify: `python/polars_metal/_dtw_dispatch.py` (binding reads `query_col`/`handle` → `col`/`payload`)

- [ ] **Step 1: Replace `_dtw_detect.py` body** (mirror Task 5 Step 1 with the DTW tag/attr)

```python
"""Serialize-detect .metal.dtw struct sentinels in a LazyFrame."""
from __future__ import annotations

import polars as pl

from polars_metal import _detect_common as dc
from polars_metal._detect_common import SentinelBinding
from polars_metal._dtw_namespace import DTW_SENTINEL_TAG

_dtw_exprs_cache: dict = {}
dc.install_with_columns_capture("_polars_metal_dtw_original_with_columns", _dtw_exprs_cache)

_parse = dc.make_sentinel_parser(DTW_SENTINEL_TAG)


def find_dtw_bindings(lf: pl.LazyFrame) -> list[SentinelBinding]:
    out: list[SentinelBinding] = []
    for inner, name in dc.iter_candidate_nodes(
        lf, cache=_dtw_exprs_cache, explain_tags=(DTW_SENTINEL_TAG,)
    ):
        b = _parse(inner, name)
        if b is not None and b.out_name:
            out.append(b)
    return out
```

- [ ] **Step 2: Migrate the dtw cache + builder**

In `python/polars_metal/_dtw_namespace.py`, replace the cache triplet (`_DTW_CACHE` / `_capture` / `get_capture` / `evict_capture`) with:
```python
from polars_metal._detect_common import CaptureCache

_CACHE = CaptureCache()

def _capture(reference, window, allow_cpu_fallback, query_col) -> int:
    return _CACHE.capture(DtwSpec(reference, window, allow_cpu_fallback, query_col))

get_capture = _CACHE.get
evict_capture = _CACHE.evict
```
(Keep `DtwSpec`; delete `_HANDLE_COUNTER` if now unused.)

Replace `build_dtw_sentinel`:
```python
from polars_metal._detect_common import sentinel_fields

def build_dtw_sentinel(seq_expr, query_col, handle):
    return pl.struct(
        sentinel_fields(
            seq_expr,
            tag=DTW_SENTINEL_TAG,
            payload=handle,
            col=query_col,
            in_alias="__pm_dtw_seq",
            raise_alias="__pm_dtw_raise",
            raise_fn=_raise_cpu,
        )
    )
```
**Keep** the existing local `_raise_cpu` and pass `raise_fn=_raise_cpu` (do NOT delete or change its message). **Leave `make_dtw_expr`'s own `ValueError`s** (single-root / negative-window) unchanged — those are user-argument validation (kept `ValueError` per the contract).

- [ ] **Step 3: Update `_dtw_dispatch.py` binding reads** — `b.query_col` → `b.col`, `b.handle` → `b.payload`; `DtwBinding` import/hint → `SentinelBinding`.

- [ ] **Step 4: Run the dtw tests**

Run: `make wheel && python -m pytest tests/python_integration/test_dtw_detect.py tests/python_integration/test_dtw_e2e.py tests/python_integration/test_execute_dtw.py tests/python_integration/test_metal_namespace_contracts.py -k dtw -v`
Expected: ALL PASS.

- [ ] **Step 5: ruff + commit**

```bash
ruff check python/ && ruff format python/
git add python/polars_metal/_dtw_detect.py python/polars_metal/_dtw_namespace.py python/polars_metal/_dtw_dispatch.py
git commit -m "M7 A-2: migrate dtw onto the namespace spine"
```

---

## Task 8: Migrate CORR onto the spine

**Files:**
- Modify (rewrite): `python/polars_metal/_corr_detect.py`
- Modify: `python/polars_metal/_corr_namespace.py` (cache triplet → `CaptureCache`; builder → `sentinel_fields` with `tag_exact` + struct alias)
- Modify: `python/polars_metal/_corr_dispatch.py` (binding read `handle` → `payload`)

CORR is the exact-match, frame-replacing verb. The parser uses `exact=True`.

- [ ] **Step 1: Replace `_corr_detect.py` body**

```python
"""Serialize-detect lf.metal.corr() struct sentinel in a LazyFrame.

corr's sentinel uses an EXACT tag alias (no source-column suffix) and is
frame-replacing — at most one sentinel per lf is meaningful.
"""
from __future__ import annotations

import polars as pl

from polars_metal import _detect_common as dc
from polars_metal._detect_common import SentinelBinding
from polars_metal._corr_namespace import CORR_SENTINEL_TAG

_corr_exprs_cache: dict = {}
dc.install_with_columns_capture("_polars_metal_corr_original_with_columns", _corr_exprs_cache)

_parse = dc.make_sentinel_parser(CORR_SENTINEL_TAG, exact=True)


def find_corr_bindings(lf: pl.LazyFrame) -> list[SentinelBinding]:
    out: list[SentinelBinding] = []
    for inner, name in dc.iter_candidate_nodes(
        lf, cache=_corr_exprs_cache, explain_tags=(CORR_SENTINEL_TAG,)
    ):
        b = _parse(inner, name)
        if b is not None and b.out_name:
            out.append(b)
    return out
```

- [ ] **Step 2: Migrate the corr cache + builder**

In `python/polars_metal/_corr_namespace.py`, replace the cache triplet (`_CORR_CACHE` / `_capture` / `get_capture` / `evict_capture`) with:
```python
from polars_metal._detect_common import CaptureCache

_CACHE = CaptureCache()

def _capture(columns, force_gpu) -> int:
    return _CACHE.capture(CorrSpec(columns, force_gpu))

get_capture = _CACHE.get
evict_capture = _CACHE.evict
```
(Keep `CorrSpec`; delete `_HANDLE_COUNTER` if unused.)

Replace `build_corr_sentinel` — note `tag_exact=True`, no `in_alias`, the raise field reads `pl.col(any_col)`, and the struct gets the `CORR_SENTINEL_COL` alias:
```python
from polars_metal._detect_common import sentinel_fields

def build_corr_sentinel(any_col, handle):
    return pl.struct(
        sentinel_fields(
            pl.col(any_col),
            tag=CORR_SENTINEL_TAG,
            payload=handle,
            raise_alias="__pm_corr_raise",
            tag_exact=True,
            raise_fn=_raise_cpu,
        )
    ).alias(CORR_SENTINEL_COL)
```
**Keep** the existing local `_raise_cpu` and pass `raise_fn=_raise_cpu` (do NOT delete or change its message — pinned by `test_corr_engine.py`).

- [ ] **Step 3: Update `_corr_dispatch.py` binding read** — `binding.handle` → `binding.payload`; `CorrBinding` import/hint → `SentinelBinding`. (`binding.out_name` stays.)

- [ ] **Step 4: Run the corr tests**

Run: `make wheel && python -m pytest tests/python_integration/test_corr_engine.py tests/python_integration/test_metal_namespace_contracts.py -k corr -v`
Expected: ALL PASS (including `test_corr_handle_missing_raises_compute_error`).

- [ ] **Step 5: ruff + commit**

```bash
ruff check python/ && ruff format python/
git add python/polars_metal/_corr_detect.py python/polars_metal/_corr_namespace.py python/polars_metal/_corr_dispatch.py
git commit -m "M7 A-2: migrate corr onto the namespace spine"
```

---

## Task 9: Bring dt + rolling detect onto the shared candidate scaffold

The native-expr verbs (dt, rolling) keep their **bespoke parsers** (schema-driven, `RollingExpr`/`TemporalExpr` validation) but adopt `iter_candidate_nodes` for the fast/slow candidate iteration, eliminating their copied scaffold. This is the "where clean" part of the spec — do it only if it reduces duplication without contorting the parser.

**Files:**
- Modify: `python/polars_metal/_dt_detect.py`
- Modify: `python/polars_metal/_rolling_detect.py`

- [ ] **Step 1: Read both files and identify the scaffold vs the parser**

Read `_dt_detect.py` and `_rolling_detect.py`. Identify (a) the with_columns capture cache + attr, (b) the fast-path/slow-path iteration (the part `iter_candidate_nodes` replaces), and (c) the bespoke `parse_node(inner, out_name, schema)` (the part that STAYS). The slow-path `explain_tags` for dt are `(".dt.year(", ".dt.month(", ".dt.day(")`; for rolling, the rolling native tags the current code pre-filters on (read them from the file — likely `("rolling_mean", "rolling_sum", "rolling_var", "rolling_std")` or the serialized `RollingExpr` marker).

- [ ] **Step 2: Refactor `find_dt_bindings` to use the scaffold**

Replace the hand-rolled fast/slow iteration with:
```python
def find_dt_bindings(lf: pl.LazyFrame) -> list[DtBinding]:
    schema = None  # resolved lazily on first candidate
    out: list[DtBinding] = []
    for inner, name in dc.iter_candidate_nodes(
        lf, cache=_dt_exprs_cache, explain_tags=(".dt.year(", ".dt.month(", ".dt.day(")
    ):
        if schema is None:
            schema = lf.collect_schema()
        b = _parse_dt_node(inner, name, schema)  # existing bespoke parser
        if b is not None and b.out_name:
            out.append(b)
    return out
```
Keep `_parse_dt_node` (the existing schema-validating parser, possibly currently inlined — extract it into a named function with signature `(inner_json, out_name, schema) -> DtBinding | None` that swallows its own exceptions and returns `None` on reject/shadow). `DtBinding` stays as-is (rich dataclass — NOT collapsed to `SentinelBinding`; document this as an intentional divergence in a comment).

- [ ] **Step 3: Refactor `find_rolling_bindings` the same way**

```python
def find_rolling_bindings(lf: pl.LazyFrame) -> list[RollingBinding]:
    schema = None
    out: list[RollingBinding] = []
    for inner, name in dc.iter_candidate_nodes(
        lf, cache=_rolling_exprs_cache, explain_tags=ROLLING_EXPLAIN_TAGS
    ):
        if schema is None:
            schema = lf.collect_schema()
        b = _parse_rolling_node(inner, name, schema)  # existing bespoke parser
        if b is not None and b.out_name:
            out.append(b)
    return out
```
Where `ROLLING_EXPLAIN_TAGS` is the tuple the current slow-path pre-filters on. `RollingBinding` stays as-is.

> **Divergence note to add as a comment in both files:** "dt/rolling keep their rich, schema-validated bindings (not the generic `SentinelBinding`) because they carry multiple typed fields (window/ddof/units_per_day/is_date). They share only the candidate-iteration scaffold (`iter_candidate_nodes`), not the parser or the binding type."

> **If adopting the scaffold contorts the parser** (e.g. the current dt/rolling slow-path fragment extraction differs materially from the struct-sentinel one), STOP and leave that verb's detection as-is — document in the commit message why it stayed off the scaffold. The scaffold adoption is opportunistic, not mandatory; the contract doc and the struct-sentinel collapse are the load-bearing wins.

- [ ] **Step 4: Run the dt + rolling tests**

Run: `make wheel && python -m pytest tests/python_integration/test_dt_detect.py tests/python_integration/test_dt_e2e.py tests/python_integration/test_dt_binding.py tests/python_integration/test_rolling_detect.py tests/python_integration/test_rolling_e2e.py tests/python_integration/test_rolling_binding.py tests/python_integration/test_rolling_fallback.py tests/python_integration/test_rolling_var_std.py tests/python_integration/test_metal_namespace_contracts.py -k "dt or rolling" -v`
Expected: ALL PASS.

- [ ] **Step 5: ruff + commit**

```bash
ruff check python/ && ruff format python/
git add python/polars_metal/_dt_detect.py python/polars_metal/_rolling_detect.py
git commit -m "M7 A-2: bring dt/rolling detect onto the shared candidate scaffold (bespoke parsers retained)"
```

---

## Task 10: Loop-driven dispatch registry in `__init__.py`

Replace the 5 copy-paste column-stitch dispatch blocks (rolling, vector, fft, dtw, dt) with a registry loop; keep corr as the frame-replacing special case.

**Files:**
- Modify: `python/polars_metal/__init__.py` (the dispatch blocks, ~lines 300–372)

- [ ] **Step 1: Read the current dispatch section** (the streaming guard + the 6 blocks) to capture the exact surrounding context (`streaming`, `cb`, `original_collect`, `kwargs`, `self` are all in scope inside the patched `collect`).

- [ ] **Step 2: Replace the 5 stitch blocks + corr with the registry**

Replace the block that currently spans the rolling → corr dispatches (from `rolling_bindings = ...` through the corr `return _corr_dispatch.apply_corr(...)`) with:

```python
            # M5/M6 serialize-detected .metal verbs run on the GPU via the
            # collect-and-stitch template. The five column-stitch verbs share
            # one dispatch shape (detect -> if bindings: apply); corr is
            # frame-replacing (single binding) and stays separate below.
            from polars_metal import (
                _dt_detect,
                _dt_dispatch,
                _dtw_detect,
                _dtw_dispatch,
                _fft_detect,
                _fft_dispatch,
                _rolling_detect,
                _rolling_dispatch,
                _vector_detect,
                _vector_dispatch,
            )

            def _collect_rest(rest_lf: Any) -> Any:
                return original_collect(rest_lf, engine="cpu", post_opt_callback=cb, **kwargs)

            _STITCH_VERBS = (
                (_rolling_detect.find_rolling_bindings, _rolling_dispatch.apply_rolling),
                (_vector_detect.find_vector_bindings, _vector_dispatch.apply_vector_search),
                (_fft_detect.find_fft_bindings, _fft_dispatch.apply_fft),
                (_dtw_detect.find_dtw_bindings, _dtw_dispatch.apply_dtw),
                (_dt_detect.find_dt_bindings, _dt_dispatch.apply_dt),
            )
            for _find_fn, _apply_fn in _STITCH_VERBS:
                _bindings = [] if streaming else _find_fn(self)
                if _bindings:
                    return _apply_fn(self, _bindings, _collect_rest)

            # corr is frame-replacing (REPLACES the frame with the pxp matrix),
            # so it passes a SINGLE binding, not the list — kept separate.
            from polars_metal import _corr_detect, _corr_dispatch

            corr_bindings = [] if streaming else _corr_detect.find_corr_bindings(self)
            if corr_bindings:
                return _corr_dispatch.apply_corr(self, corr_bindings[0], _collect_rest)
```

**Preserve verb order exactly** (rolling, vector, fft, dtw, dt, then corr) — the first verb with bindings wins, and the current order is load-bearing (the comments in the original note rolling consumes first). **Leave the streaming guard above this block unchanged** (the `_SENTINEL_TAGS` raise for vector/fft/dtw/corr).

> The one `_collect_rest` closure replaces the five identical per-verb closures (`_collect_rest`, `_collect_rest_vs`, `_collect_rest_fft`, `_collect_rest_dtw`, `_collect_rest_dt`, `_collect_rest_corr`) — they were byte-identical, so a single shared closure is behavior-preserving.

- [ ] **Step 3: Run the full python_integration suite**

Run: `make wheel && python -m pytest tests/python_integration/ -q`
Expected: same pass/skip set as before A2 began (no regressions across all verbs).

- [ ] **Step 4: ruff + commit**

```bash
ruff check python/ && ruff format python/
git add python/polars_metal/__init__.py
git commit -m "M7 A-2: replace 5 copy-paste dispatch blocks with a loop-driven registry"
```

---

## Task 11: LOC tally, full gate, status update

**Files:**
- Modify: `docs/superpowers/specs/2026-06-12-m7-consolidation-design.md` (mark Workstream A delivered)
- Modify: the memory `m7-consolidation-scope.md` (mark A done; M7 complete)

- [ ] **Step 1: Tally the LOC reduction**

Run: `wc -l python/polars_metal/_*detect*.py python/polars_metal/_*namespace*.py python/polars_metal/_*dispatch*.py python/polars_metal/__init__.py`
Record the before (from the pre-A baseline: vector_detect 109, fft_detect 93, dtw_detect 93, corr_detect 88, dt_detect 184, rolling_detect 414, the namespace/dispatch files, __init__ 404) vs after. The spec target is ≈525 → ≈165 for the namespace machinery (the detect + sentinel + cache + dispatch-registration surface). Note the realized number in the status update — if it lands materially above 165, note which files didn't collapse and why (e.g. rolling_detect's 414 lines are mostly the bespoke `RollingExpr` validation, which is parser logic, not scaffold — that's expected to stay).

- [ ] **Step 2: Run the differential net**

Run: `make test-diff`
Expected: PASS (C1's Python plan-level slice + the Rust proptest subset — the safety net for the A refactor).

- [ ] **Step 3: Run the full gate**

Run: `make gate`
Expected: PASS — `cargo fmt`/clippy/ruff clean, unit + kernel + conformance green. (Conformance baseline must still show only the documented pre-existing deviations; no NEW failures.)

> If `make gate` does not include the `python_integration` suite (per the memory, it historically did not), additionally run `make wheel && python -m pytest tests/python_integration/ -q` and confirm green.

- [ ] **Step 4: Update the spec status**

In `docs/superpowers/specs/2026-06-12-m7-consolidation-design.md`, update the Workstream A section and the §6 definition-of-done to mark A delivered, mirroring how Workstream B's status block is written (commit range, what landed, realized LOC). State that M7 (A+B+C) is complete and gate-green.

- [ ] **Step 5: Update the memory file**

Update `/Users/dclark/.claude/projects/-Users-dclark-dev-polars-metal-main-polars-metal/memory/m7-consolidation-scope.md`: change the status to "A1 + A2 DONE, gate-green; **M7 complete** (A+B+C all delivered on branch m7-consolidation)"; record the realized namespace LOC reduction, the contract doc path, the `SentinelBinding`/`CaptureCache`/`iter_candidate_nodes` spine, the `RuntimeError→ComputeError` correction, and the documented intentional divergences (fft no-cache; rolling/dt silent-CPU-on-streaming; per-verb null semantics).

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-06-12-m7-consolidation-design.md
git commit -m "M7 A: mark Workstream A delivered; M7 consolidation complete"
```

- [ ] **Step 7: Surface the branch-finishing decision**

M7 is now complete on `m7-consolidation`, which is based on the **unmerged** `m6-vector-search` (PR #6). Do NOT merge unilaterally. Report to the architect: M7 is gate-green; the branch stacks on the open PR #6; ask how to integrate (e.g. land PR #6 first, then open an M7 PR onto `main`; or stack the M7 PR onto PR #6). Use the `superpowers:finishing-a-development-branch` skill to present the structured options.

---

## Self-review checklist (run before declaring the plan ready)

- **Spec coverage:** A1 contract doc (Task 2) ✓; A1 characterization tests (Task 1) ✓; A1 RuntimeError→ComputeError + FFT-guard documentation (Task 3 + the contract doc) ✓; A2 parameterized detect factory (`make_sentinel_parser` + `iter_candidate_nodes`, Task 4) ✓; generic `CaptureCache` (Task 4, adopted Tasks 5/7/8) ✓; one sentinel builder (`sentinel_fields`, Task 4, adopted Tasks 5–8) ✓; generic `Binding` (`SentinelBinding`, Task 4) ✓; loop-driven dispatch registry (Task 10) ✓; rolling/dt divergence handled + documented (Task 9) ✓; LOC target + gate (Task 11) ✓.
- **Out of scope respected:** `_walker.py`/`_udf.py`/`_fusion_analyzer.py` never touched ✓; no Rust changes ✓; no new ops/kernels ✓.
- **Behavior-preserving:** every sentinel builder migration preserves exact field aliases/order (byte-identical serialized plans) ✓; the only deliberate behavior change is RuntimeError→ComputeError, test-pinned in Task 3 ✓.
- **Type consistency:** `SentinelBinding(out_name, col, payload)` used uniformly; dispatch field reads renamed `query_col/input_col → col`, `handle/op → payload` in Tasks 5–8; `CaptureCache.get/evict/capture` and `_CACHE` instance names consistent across Tasks 4–8; `iter_candidate_nodes(lf, *, cache, explain_tags)` signature consistent across Tasks 4–9.
```

