# M6 Cleanup + Memory-Usage Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate avoidable host-side copies across the M6 `.metal` dispatch paths, fix the repeated-collect O(N)-serialize footgun, and close a set of correctness/cleanup loose ends — all differentially tested against Polars/CPU oracles.

**Architecture:** Five independently-shippable phases. Phase 1 = surgical copy-elimination (dt Int8 narrowing, FFT clone gate, dt staging guard, DTW copy guard). Phase 2 = correctness fixes (corr N<2 error, streaming fallback, vector null guard). Phase 3 = DRY the detect modules into `_detect_common.py`, then redesign its cache to survive repeated collects via an `id()`-keyed dict + `weakref.ref` eviction. Phase 4 = move the FFT CPU interleave/split onto the GPU (new MSL kernels), folding in the readback. Phase 5 = cosmetics, dead code, edge tests.

**Tech Stack:** Python (Polars dispatch + numpy), Rust (PyO3 + MLX FFI), MSL (Metal shaders for the FFT interleave/split kernels).

**Source:** the three M6 memory/cleanup audits (2026-06-11). Each fix below cites the audited file:line and the measured cost.

**Convention reminders (from CLAUDE.md):** no `unwrap()` in non-test Rust; `// SAFETY:` on unsafe; tests run `--test-threads=1`; run `cargo fmt`/`ruff` per task (don't let drift accumulate to the final gate); Polars CPU is the oracle — debug toward it, never loosen tolerances.

---

## File Structure

- `python/polars_metal/_dt_dispatch.py` — M1 (Int8 astype), M4 (staging guard input shape).
- `crates/polars-metal-core/src/udf.rs` — M4 (dt staging alignment guard in `execute_dt`).
- `crates/polars-metal-kernels/src/fft.rs` — M2 (fourstep forward-clone gate).
- `python/polars_metal/_dtw_dispatch.py` — M6 (copy-only-on-null).
- `python/polars_metal/_corr_dispatch.py` — C1 (N<2 clean error).
- `python/polars_metal/_vector_dispatch.py` — C3 (null query-embedding guard).
- `python/polars_metal/__init__.py` — C2 (streaming sentinel fallback).
- `python/polars_metal/_detect_common.py` — **new.** C4 (shared patch + JSON helpers) + M7 (weakref cache).
- `python/polars_metal/_{vector,fft,dtw,dt,rolling,corr}_detect.py` — C4/M7 (adopt `_detect_common`).
- `crates/polars-metal-core/src/fft.rs`, `shaders/fft_pack.metal` (**new**), `crates/polars-metal-kernels/src/fft.rs` — M5 (GPU interleave/split).
- `crates/polars-metal-core/src/arena.rs`, `tests/bench/bench_corr.py`, misc — C5.

---

# PHASE 1 — Surgical copy-elimination

## Task 1 (M1): dt month/day output — `astype(np.int8)` instead of `Series.cast(Int8)`

`pl.Series(int32).cast(pl.Int8)` costs ~19–20ms at 10M rows; `out.astype(np.int8)` costs ~0.65ms (30×). Year is Int32 (no cast) and stays as-is.

**Files:**
- Modify: `python/polars_metal/_dt_dispatch.py:51`
- Test: `tests/python_integration/test_dt_e2e.py` (or the existing dt test file — confirm name with `ls tests/python_integration/ | grep dt`)

- [ ] **Step 1: Read the current `_dt_series` to confirm context**

Read `python/polars_metal/_dt_dispatch.py` lines 24–60. Confirm line 51 is:
```python
    dense = pl.Series(b.out_name, out, dtype=pl.Int32).cast(out_dtype)
```
and `out` is a numpy int32 array, `out_dtype` is `_FIELD_DTYPE[b.field]` (`pl.Int32` for year, `pl.Int8` for month/day), and `_NUMPY_DTYPE` does not yet exist.

- [ ] **Step 2: Write a failing perf-independent correctness test**

Add to the dt engine test file (byte-exact vs Polars CPU for all three fields, which already must pass — this test guards the dtype after the change):
```python
def test_dt_month_day_dtype_and_values_after_astype():
    import polars as pl
    import polars_metal as pm

    df = pl.DataFrame({"d": pl.date_range(
        pl.date(2000, 1, 1), pl.date(2000, 12, 31), interval="1d", eager=True
    )})
    out = df.lazy().select(
        pl.col("d").dt.year().alias("y"),
        pl.col("d").dt.month().alias("m"),
        pl.col("d").dt.day().alias("dd"),
    ).collect(engine=pm.MetalEngine())
    exp = df.select(
        pl.col("d").dt.year().alias("y"),
        pl.col("d").dt.month().alias("m"),
        pl.col("d").dt.day().alias("dd"),
    )
    assert out.schema["y"] == pl.Int32
    assert out.schema["m"] == pl.Int8 and out.schema["dd"] == pl.Int8
    assert out.equals(exp)
```

- [ ] **Step 3: Run the test to confirm it passes today (baseline) then we keep it green**

Run: `python -m pytest tests/python_integration/test_dt_e2e.py::test_dt_month_day_dtype_and_values_after_astype -v`
Expected: PASS (current code already produces correct dtype/values). This locks behavior before the refactor.

- [ ] **Step 4: Add a numpy-dtype map and switch to `astype`**

In `python/polars_metal/_dt_dispatch.py`, near `_FIELD_DTYPE` (line 21) add:
```python
import numpy as np  # confirm already imported at top; if so, skip this line

_FIELD_NUMPY_DTYPE = {"year": np.int32, "month": np.int8, "day": np.int8}
```
Replace line 51:
```python
    dense = pl.Series(b.out_name, out, dtype=pl.Int32).cast(out_dtype)
```
with:
```python
    # Narrow on the numpy side (out is int32): astype(int8) is ~30x cheaper than
    # pl.Series(int32).cast(pl.Int8) (~19ms -> ~0.6ms at 10M rows). year stays int32.
    narrowed = out.astype(_FIELD_NUMPY_DTYPE[b.field], copy=False)
    dense = pl.Series(b.out_name, narrowed, dtype=out_dtype)
```
Note: `astype(int32, copy=False)` on an already-int32 array is a no-op view (year path); `astype(int8)` allocates the small narrowed buffer (month/day).

- [ ] **Step 5: Run the test + the full dt suite**

Run: `python -m pytest tests/python_integration/ -k "dt" -v`
Expected: all PASS (byte-exact vs CPU preserved).

- [ ] **Step 6: ruff + commit**

```bash
ruff check --fix python/polars_metal/_dt_dispatch.py tests/python_integration/test_dt_e2e.py
git add python/polars_metal/_dt_dispatch.py tests/python_integration/test_dt_e2e.py
git commit -m "M6 cleanup M1: dt month/day astype(int8) (~30x cheaper than Series.cast)"
```

---

## Task 2 (M2): FFT four-step — skip the forward-path input clone

`fft_recursive_fourstep` does `let mut staged = input.to_vec();` unconditionally, but only mutates it when `inverse`. For the forward path the clone is wasted (~2.7ms at 2²⁴); `from_f32_slice` already copies into Metal memory.

**Files:**
- Modify: `crates/polars-metal-kernels/src/fft.rs:321` (the `staged`/`data_buf` block)

- [ ] **Step 1: Read the block**

Read `crates/polars-metal-kernels/src/fft.rs` lines 318–328. Confirm it is:
```rust
    // Stage the (optionally conjugated) input into the data buffer.
    let mut staged = input.to_vec();
    if inverse {
        for c in staged.chunks_exact_mut(2) {
            c[1] = -c[1];
        }
    }
    let mut data_buf = MetalBuffer::from_f32_slice(device, &staged)?;
```

- [ ] **Step 2: Gate the clone on `inverse`**

Replace that block with:
```rust
    // Stage into the data buffer. Forward: no clone — from_f32_slice copies into
    // Metal memory anyway. Inverse: clone to conjugate (input is borrowed).
    let mut data_buf = if inverse {
        let mut staged = input.to_vec();
        for c in staged.chunks_exact_mut(2) {
            c[1] = -c[1];
        }
        MetalBuffer::from_f32_slice(device, &staged)?
    } else {
        MetalBuffer::from_f32_slice(device, input)?
    };
```
(Keep `let mut data_buf` — it is mutated later in the function.)

- [ ] **Step 3: Build + run the FFT kernel tests**

Run: `cargo test -p polars-metal-kernels fft -- --test-threads=1`
Expected: all FFT tests PASS (forward + inverse round-trip unchanged).

- [ ] **Step 4: fmt/clippy + commit**

```bash
cargo fmt -p polars-metal-kernels && cargo clippy -p polars-metal-kernels -- -D warnings
git add crates/polars-metal-kernels/src/fft.rs
git commit -m "M6 cleanup M2: skip forward-path input clone in fft_recursive_fourstep"
```

---

## Task 3 (M4): dt Datetime path — skip StagingPool when input is already page-aligned

`execute_dt` (`crates/polars-metal-core/src/udf.rs` ~2924) unconditionally `memcpy`s through `DT_STAGING.stage()`. For the Datetime path at N≥~25k, the numpy `(phys // units).astype(int32)` buffer is mmap-backed → 16KB page-aligned → eligible for `from_borrowed_i32`'s `newBufferWithBytesNoCopy` zero-copy path. The Date path (Arrow 64-byte aligned, not page-aligned) genuinely needs the StagingPool.

**Files:**
- Modify: `crates/polars-metal-core/src/udf.rs` (the `execute_dt` staging site)

- [ ] **Step 1: Read the staging site**

Read `crates/polars-metal-core/src/udf.rs` around `fn execute_dt` (grep `fn execute_dt` then read ~40 lines). Identify: the input `(ptr,len)`, the `DT_STAGING.stage(...)` call, and whether a `from_borrowed_i32` / `is_ptr_page_aligned` helper already exists (grep `is_ptr_page_aligned` — it's used by `execute_dtw`).

- [ ] **Step 2: Add the alignment guard**

Replace the unconditional `DT_STAGING.stage(...)` with a branch: if `is_ptr_page_aligned(in_ptr)` use `MetalBuffer::from_borrowed_i32(&device, in_ptr as *const i32, len)` (zero-copy `bytesNoCopy`); else use the existing `DT_STAGING.stage(...)` path. Mirror the exact idiom `execute_dtw` uses for `from_borrowed_f32` + the `is_ptr_page_aligned` check (read it first; reuse the same SAFETY-comment shape). Concretely the shape is:
```rust
    use polars_metal_buffer::is_ptr_page_aligned;
    // Datetime path produces a page-aligned numpy buffer (mmap) at large N → zero-copy.
    // Date path / small N is 64-byte Arrow-aligned only → stage through the pool.
    let in_buf = if is_ptr_page_aligned(in_ptr) {
        // SAFETY: in_ptr addresses `len` contiguous live i32 for the call (Python holds it).
        unsafe { MetalBuffer::from_borrowed_i32(&device, in_ptr as *const i32, len) }
            .map_err(/* PyRuntimeError, mirror execute_dtw */)?
    } else {
        DT_STAGING.stage(&device, in_ptr, len)/* existing signature */?
    };
```
Adapt names/signatures to what the file actually has (the `DT_STAGING.stage` signature and the `MetalBuffer::from_borrowed_i32` existence must be verified; if `from_borrowed_i32` doesn't exist but `from_borrowed_f32` does, check `staging.rs`/`buffer` for the i32 variant or add a trivial one mirroring the f32 one). If `from_borrowed_i32` is genuinely absent and non-trivial to add, STOP and report — do not hand-roll an unsafe buffer without the established helper.

- [ ] **Step 3: Run the dt kernel + engine tests**

Run: `cargo test -p polars-metal-core -- --test-threads=1` then `python -m pytest tests/python_integration/ -k "dt" -v`
Expected: all PASS (byte-exact preserved; this is a staging-path change only).

- [ ] **Step 4: fmt/clippy + commit**

```bash
cargo fmt -p polars-metal-core && cargo clippy -p polars-metal-core -- -D warnings
git add crates/polars-metal-core/src/udf.rs
git commit -m "M6 cleanup M4: dt Datetime path uses zero-copy borrow when page-aligned"
```

---

## Task 4 (M6): DTW — copy the matrix only when nulls are present

`_run_binding` does `safe = mat.copy()` unconditionally before neutralizing null rows. When there are no nulls (the common case) the copy is wasted. Payoff is small (~0.1–1.9ms, dwarfed by the O(N·L²) kernel) but the fix is trivial and removes a needless allocation.

**Files:**
- Modify: `python/polars_metal/_dtw_dispatch.py` (the `safe = mat.copy()` block, ~line 87)

- [ ] **Step 1: Read the block**

Read `python/polars_metal/_dtw_dispatch.py` lines 78–96. Confirm the current logic: `null_mask = s.is_null().to_numpy()`, then `safe = mat.copy(); safe[null_mask] = 0.0; if np.isnan(safe).any(): raise ...; qflat = np.ascontiguousarray(safe, ...).reshape(-1)`.

- [ ] **Step 2: Guard the copy on `null_mask.any()`**

Replace the block so the copy + neutralize only happens when there are nulls; otherwise validate NaN on the read-only view and reshape it directly:
```python
    if null_mask.any():
        safe = mat.copy()
        safe[null_mask] = 0.0
        # A non-null row containing a genuine NaN cell would be silently mis-scored
        # by the kernel's fmin (drops NaN), so reject it (nulls ok, NaN cells not).
        if np.isnan(safe).any():
            raise ValueError(
                "polars_metal: .metal.dtw: a non-null sequence contains NaN, which the "
                "GPU kernel cannot match against the oracle (nulls are supported; NaN cells are not)."
            )
        qflat = np.ascontiguousarray(safe, dtype=np.float32).reshape(-1)
    else:
        if np.isnan(mat).any():
            raise ValueError(
                "polars_metal: .metal.dtw: a non-null sequence contains NaN, which the "
                "GPU kernel cannot match against the oracle (nulls are supported; NaN cells are not)."
            )
        qflat = np.ascontiguousarray(mat, dtype=np.float32).reshape(-1)
```
(Keep the surrounding `null_mask` computation and the post-call `res.scatter(...)` null restoration unchanged. The error message must stay byte-identical to the existing one so any test asserting on it still matches.)

- [ ] **Step 3: Run the DTW tests**

Run: `python -m pytest tests/python_integration/ -k "dtw" -v`
Expected: all PASS, including the existing NaN-in-non-null-row raise test and the null-row handling test.

- [ ] **Step 4: ruff + commit**

```bash
ruff check --fix python/polars_metal/_dtw_dispatch.py
git add python/polars_metal/_dtw_dispatch.py
git commit -m "M6 cleanup M6: DTW copies the query matrix only when nulls present"
```

---

# PHASE 2 — Correctness fixes

## Task 5 (C1): corr N<2 — raise a clean ComputeError instead of numpy's TypeError

`_run_corr` routes `N<2` to `_cpu_corr_f32` → `df.corr()`, which on a 1-row frame raises a raw `TypeError: DataFrame constructor called with unsupported type 'float64'`. Surface a clear, Polars-native error instead.

**Files:**
- Modify: `python/polars_metal/_corr_dispatch.py` (`_run_corr` / `_cpu_corr_f32`)
- Test: `tests/python_integration/test_corr_engine.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/python_integration/test_corr_engine.py`:
```python
def test_corr_single_row_raises_clear_error():
    import polars as pl
    import polars_metal as pm

    df = pl.DataFrame({f"c{i}": [1.0] for i in range(10)})  # N=1
    with pytest.raises(pl.exceptions.ComputeError, match="at least 2 rows"):
        df.lazy().metal.corr().collect(engine=pm.MetalEngine())
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/python_integration/test_corr_engine.py::test_corr_single_row_raises_clear_error -v`
Expected: FAIL — currently raises `TypeError`, not `pl.exceptions.ComputeError`.

- [ ] **Step 3: Add an explicit N<2 guard in `_run_corr`**

In `python/polars_metal/_corr_dispatch.py`, in `_run_corr`, replace the `if has_null or df.height < 2:` branch so the `N<2` case raises a clear error before reaching `df.corr()`:
```python
    if df.height < 2:
        raise pl.exceptions.ComputeError(
            "polars_metal: .metal.corr() needs at least 2 rows to compute a "
            f"correlation (got {df.height})."
        )
    if has_null:
        return _cpu_corr_f32(df, columns)
```
(Keep the subsequent `if p < CORR_P_MIN ...` and GPU branches unchanged. Confirm `pl` is imported at module top.)

- [ ] **Step 4: Run the test + full corr suite**

Run: `python -m pytest tests/python_integration/test_corr_engine.py -v`
Expected: the new test PASSES; all prior corr tests still PASS.

- [ ] **Step 5: ruff + commit**

```bash
ruff check --fix python/polars_metal/_corr_dispatch.py tests/python_integration/test_corr_engine.py
git add python/polars_metal/_corr_dispatch.py tests/python_integration/test_corr_engine.py
git commit -m "M6 cleanup C1: corr raises clear ComputeError on N<2 (not numpy TypeError)"
```

---

## Task 6 (C3): vector search — guard null query embeddings

`_array_col_to_matrix` calls `s.to_numpy()` on an `Array(Float32, D)` column with no null check. A query/corpus column with null rows produces silently-wrong results. DTW already guards this; vector search should too.

**Files:**
- Modify: `python/polars_metal/_vector_dispatch.py` (`_array_col_to_matrix` and/or `_run_binding`)
- Test: `tests/python_integration/` vector test file (confirm name via `ls tests/python_integration/ | grep vector`)

- [ ] **Step 1: Write the failing test**

Add to the vector test file:
```python
def test_vector_null_query_row_raises():
    import polars as pl
    import polars_metal as pm

    # Build a query column of Array(Float32, 4) with one null row.
    q = pl.Series("emb", [[1.0, 0.0, 0.0, 0.0], None, [0.0, 1.0, 0.0, 0.0]],
                  dtype=pl.Array(pl.Float32, 4))
    corpus = pl.DataFrame({"emb": pl.Series(
        "emb", [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=pl.Array(pl.Float32, 4))})
    qdf = pl.DataFrame({"q": q})
    with pytest.raises(Exception, match="null"):
        qdf.lazy().select(
            pl.col("q").metal.cosine_topk(corpus, k=1, corpus_col="emb")
        ).collect(engine=pm.MetalEngine())
```
Confirm the exact `.metal.cosine_topk` signature against `_vector_namespace.py` before finalizing (arg names/order may differ — match reality).

- [ ] **Step 2: Run to verify it fails**

Run the new test. Expected: FAIL (currently returns garbage / does not raise). If it happens to raise a different unclear error, note it; the goal is a clear null message.

- [ ] **Step 3: Add the null guard**

In `python/polars_metal/_vector_dispatch.py`, in the function that materializes a query/corpus `Array` column to a matrix (e.g. `_array_col_to_matrix` or where `s.rechunk().to_numpy()` is called on the query/corpus Series), add before `to_numpy()`:
```python
    if s.null_count() > 0:
        raise ValueError(
            "polars_metal: .metal.cosine_topk/.knn does not support null rows in the "
            f"query or corpus embedding column (column {s.name!r} has "
            f"{s.null_count()} null rows). Drop or impute nulls first."
        )
```
Apply it to BOTH the query column and the corpus column paths (check whether they share a helper; if so, one site suffices). Mirror the message style of DTW's null/NaN guard.

- [ ] **Step 4: Run the test + full vector suite**

Run: `python -m pytest tests/python_integration/ -k "vector or cosine or knn" -v`
Expected: the new test PASSES; all prior vector tests still PASS.

- [ ] **Step 5: ruff + commit**

```bash
ruff check --fix python/polars_metal/_vector_dispatch.py <vector test file>
git add python/polars_metal/_vector_dispatch.py <vector test file>
git commit -m "M6 cleanup C3: vector search rejects null query/corpus embedding rows"
```

---

## Task 7 (C2): streaming — sentinel verbs fall back cleanly instead of raising

Under `streaming=True`, `collect_wrapper` sets each verb's bindings to `[]`, so dispatch is skipped — but the plan still carries the `map_batches(_raise_cpu)` sentinel, so the CPU collect raises `RuntimeError("requires collect(engine='metal')")`. The user expects a clean CPU fallback (like rolling, which has no sentinel). Fix: when streaming AND a sentinel verb is detected, raise a *clear* up-front error explaining streaming is unsupported for these verbs (a clear targeted error is the honest contract; true CPU fallback isn't possible because these ops have no CPU implementation).

**Files:**
- Modify: `python/polars_metal/__init__.py` (`collect_wrapper`, the streaming guard region)
- Test: `tests/python_integration/test_corr_engine.py` (+ optionally one per verb)

- [ ] **Step 1: Write the failing test**

Add to `tests/python_integration/test_corr_engine.py`:
```python
def test_corr_streaming_raises_clear_error():
    import polars as pl
    import polars_metal as pm

    df = _frame(n=1000, p=10)
    with pytest.raises(Exception, match="streaming"):
        df.lazy().metal.corr().collect(engine=pm.MetalEngine(), streaming=True)
```

- [ ] **Step 2: Run to verify it fails**

Run the new test. Expected: FAIL or wrong-message — currently raises the misleading `_raise_cpu` "requires collect(engine='metal')" RuntimeError, not a streaming message.

- [ ] **Step 3: Detect sentinel-bearing verbs under streaming and raise clearly**

In `python/polars_metal/__init__.py`, in `collect_wrapper`, AFTER `streaming` is computed and BEFORE the per-verb `[] if streaming else ...` blocks, add a guard that — only when `streaming` is true — checks whether the plan carries any `.metal` sentinel and raises a clear error. Use the cheap `lf.explain()` text check (the detectors already use it as a prefilter) against the known sentinel tags:
```python
            if streaming:
                import warnings as _w

                from polars_metal._vector_namespace import SENTINEL_TAG as _VEC_TAG
                from polars_metal._fft_namespace import FFT_SENTINEL_TAG as _FFT_TAG
                from polars_metal._dtw_namespace import DTW_SENTINEL_TAG as _DTW_TAG
                from polars_metal._corr_namespace import CORR_SENTINEL_TAG as _CORR_TAG

                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    _plan_txt = self.explain()
                if any(t in _plan_txt for t in (_VEC_TAG, _FFT_TAG, _DTW_TAG, _CORR_TAG)):
                    raise pl.exceptions.ComputeError(
                        "polars_metal: .metal vector/fft/dtw/corr verbs are not supported "
                        "under streaming=True (they have no CPU implementation). Collect "
                        "without streaming, or use the CPU equivalent."
                    )
```
Verify the exact exported tag names by grepping each `_*_namespace.py` for `SENTINEL_TAG`/`*_SENTINEL_TAG`; fix the imports to match. Place this guard so it runs for `MetalEngine` collects only (inside the `if isinstance(engine, MetalEngine):` block, where `streaming` is already defined).

- [ ] **Step 4: Run the test + a regression sweep**

Run: `python -m pytest tests/python_integration/test_corr_engine.py -v` then `python -m pytest tests/python_integration/ -k "rolling or dt_" -q` (rolling/dt must STILL fall back cleanly under streaming — they have no sentinel, so the guard won't fire for them).
Expected: new test PASSES; rolling/dt streaming behavior unchanged.

- [ ] **Step 5: ruff + commit**

```bash
ruff check --fix python/polars_metal/__init__.py tests/python_integration/test_corr_engine.py
git add python/polars_metal/__init__.py tests/python_integration/test_corr_engine.py
git commit -m "M6 cleanup C2: sentinel verbs raise a clear streaming-unsupported error"
```

---

# PHASE 3 — DRY the detect modules + fix repeated-collect (C4 → M7)

These compose: C4 extracts the shared machinery into `_detect_common.py`; M7 then upgrades that ONE place from pop-on-consume to a weakref-evicted, get-not-pop cache, fixing repeated-collect for every verb at once.

## Task 8 (C4): extract `_detect_common.py`

The 5 detect modules each duplicate: the `with_columns` monkeypatch boilerplate, and (for 3 of them) the JSON helpers `_alias_name`/`_literal_int`/`_struct_fields`. `_fft_detect` already imports the helpers from `_vector_detect`; consolidate all of it into a dedicated module.

**Files:**
- Create: `python/polars_metal/_detect_common.py`
- Modify: `python/polars_metal/_{vector,fft,dtw,corr,rolling}_detect.py`
- Test: `tests/python_integration/test_detect_common.py` (new)

- [ ] **Step 1: Read all five detect modules to extract the common shape**

Read `_vector_detect.py`, `_fft_detect.py`, `_dtw_detect.py`, `_corr_detect.py`, `_rolling_detect.py`. Catalog: (a) the JSON helpers (`_alias_name`, `_literal_int`, `_struct_fields`) — confirm they are byte-identical across the modules that define them; (b) the patch-install boilerplate (each has a unique `_PATCH_ATTR` string and a per-module `_lf_exprs_cache` dict); (c) `rolling`'s detector — note whether it uses the same with_columns-cache shape or a different one (it may detect `rolling_*` differently; if so, it adopts only the JSON helpers, not the cache).

- [ ] **Step 2: Write a test for the shared helpers**

Create `tests/python_integration/test_detect_common.py`:
```python
from polars_metal import _detect_common as dc


def test_alias_name_extracts():
    node = {"Alias": [{"Literal": {"Scalar": {"Int64": 7}}}, "__pm_x__"]}
    assert dc._alias_name(node) == "__pm_x__"
    assert dc._alias_name({"Column": "a"}) is None


def test_literal_int_extracts():
    node = {"Alias": [{"Literal": {"Scalar": {"Int64": 42}}}, "tag"]}
    assert dc._literal_int(node) == 42


def test_install_patch_idempotent():
    cache1 = {}
    # Installing twice with the same attr must not double-wrap.
    dc.install_with_columns_capture("_test_attr_xyz", cache1)
    dc.install_with_columns_capture("_test_attr_xyz", cache1)
    import polars as pl
    lf = pl.DataFrame({"a": [1]}).lazy().with_columns((pl.col("a") + 1).alias("b"))
    assert id(lf) in cache1  # the patch captured the exprs
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest tests/python_integration/test_detect_common.py -v`
Expected: FAIL (`ModuleNotFoundError: _detect_common`).

- [ ] **Step 4: Create `_detect_common.py` with the helpers + a patch installer**

Create `python/polars_metal/_detect_common.py`. Move the three JSON helpers verbatim from `_vector_detect.py`, and add an `install_with_columns_capture(attr, cache)` that encapsulates the monkeypatch (this is the M7-ready seam — Task 9 will change only its internals):
```python
"""M6: shared serialize-detect machinery for the .metal verbs.

Houses the JSON-walk helpers (one copy, was duplicated across vector/fft/dtw/corr
detect modules) and the with_columns-capture monkeypatch installer. The cache is a
plain id()-keyed dict in this task; Task 9 (M7) upgrades it to a weakref-evicted,
get-not-pop cache so repeated collect() of one LazyFrame stays on the fast path.
"""

from __future__ import annotations

import polars as pl
import polars.lazyframe.frame as _plf


def _alias_name(node) -> str | None:
    if isinstance(node, dict):
        a = node.get("Alias")
        if isinstance(a, list) and len(a) == 2 and isinstance(a[1], str):
            return a[1]
    return None


def _struct_fields(expr_json: dict) -> list:
    fn = expr_json.get("Function")
    if isinstance(fn, dict):
        inp = fn.get("input")
        if isinstance(inp, list):
            return inp
    return []


def _literal_int(node) -> int | None:
    if isinstance(node, dict):
        a = node.get("Alias")
        if isinstance(a, list) and len(a) == 2 and isinstance(a[0], dict):
            lit = a[0].get("Literal")
            if isinstance(lit, dict):
                scalar = lit.get("Scalar")
                if isinstance(scalar, dict):
                    for key in ("Int64", "Int32", "Int"):
                        v = scalar.get(key)
                        if isinstance(v, int):
                            return v
                for key in ("Int64", "Int32", "Int"):
                    v = lit.get(key)
                    if isinstance(v, int):
                        return v
            if isinstance(lit, int):
                return lit
    return None


def install_with_columns_capture(attr: str, cache: dict) -> None:
    """Idempotently install a with_columns wrapper that records each call's exprs
    into `cache` keyed by id(result). Chains with other installs (each wraps the
    previous). No-op if `attr` already installed."""
    if hasattr(_plf.LazyFrame, attr):
        return
    orig = _plf.LazyFrame.with_columns
    setattr(_plf.LazyFrame, attr, orig)

    def _patched(self, *exprs, **named):  # type: ignore[no-untyped-def]
        result = orig(self, *exprs, **named)
        try:
            flat = [e for e in exprs if isinstance(e, pl.Expr)]
            flat += [e.alias(n) for n, e in named.items() if isinstance(e, pl.Expr)]
            if flat:
                cache[id(result)] = flat
        except Exception:
            pass
        return result

    _plf.LazyFrame.with_columns = _patched  # type: ignore[method-assign]
```

- [ ] **Step 5: Point the detect modules at `_detect_common`**

In `_vector_detect.py`, `_fft_detect.py`, `_dtw_detect.py`, `_corr_detect.py`: delete the local copies of `_alias_name`/`_literal_int`/`_struct_fields` and import them from `_detect_common`; replace the inline `if not hasattr(...): ...` patch block with a call to `dc.install_with_columns_capture(_PATCH_ATTR, _lf_exprs_cache)` (each keeps its own `_PATCH_ATTR` and cache dict for now). For `_rolling_detect.py`, replace only its JSON helpers (if it defines any) and its patch block if it uses the same shape; if rolling's detection differs, import only the helpers it uses. Keep each module's `find_*_bindings` and `*Binding` dataclass in place.

- [ ] **Step 6: Run the full detect/verb suite**

Run: `python -m pytest tests/python_integration/test_detect_common.py tests/python_integration/ -k "vector or fft or dtw or corr or rolling or dt" -q`
Expected: all PASS (pure refactor — behavior identical).

- [ ] **Step 7: ruff + commit**

```bash
ruff check --fix python/polars_metal/_detect_common.py python/polars_metal/_*_detect.py tests/python_integration/test_detect_common.py
git add python/polars_metal/_detect_common.py python/polars_metal/_*_detect.py tests/python_integration/test_detect_common.py
git commit -m "M6 cleanup C4: extract _detect_common (shared JSON helpers + patch installer)"
```

---

## Task 9 (M7): weakref-evicted, get-not-pop cache — fix repeated-collect O(N) serialize

`pl.LazyFrame` is weakref-able but unhashable (verified), so the cache is `id() -> (weakref.ref, exprs)`. `find_*_bindings` reads with `get` + identity-validate (no pop), so a 2nd collect of the same lf stays on the fast path instead of paying O(N) `lf.serialize()`. A weakref callback evicts the entry when the lf is GC'd, bounding growth and defeating the id-reuse hazard. This is changed in ONE place (`_detect_common`) and every verb inherits it.

**Files:**
- Modify: `python/polars_metal/_detect_common.py` (capture + a `lookup` helper)
- Modify: `python/polars_metal/_{vector,fft,dtw,corr,rolling}_detect.py` (use `dc.lookup` instead of `cache.pop`)
- Test: `tests/python_integration/test_detect_common.py` + per-verb repeated-collect tests

- [ ] **Step 1: Write the failing repeated-collect test**

Add to `tests/python_integration/test_corr_engine.py` (corr is the strictest case — today its 2nd collect *raises*):
```python
def test_corr_repeated_collect_same_lf():
    import numpy as np
    import polars_metal as pm

    df = _frame(n=3000, p=12, seed=11)
    lf = df.lazy().metal.corr()
    out1 = lf.collect(engine=pm.MetalEngine())
    out2 = lf.collect(engine=pm.MetalEngine())  # must NOT raise, must match
    np.testing.assert_allclose(out1.to_numpy(), out2.to_numpy(), atol=1e-5, equal_nan=True)
```
And a cache-level test in `test_detect_common.py`:
```python
def test_lookup_does_not_pop_and_evicts_on_gc():
    import gc
    import polars as pl
    from polars_metal import _detect_common as dc

    cache = {}
    dc.install_with_columns_capture("_test_attr_m7", cache)
    lf = pl.DataFrame({"a": [1]}).lazy().with_columns((pl.col("a") + 1).alias("b"))
    assert dc.lookup(cache, lf) is not None       # found
    assert dc.lookup(cache, lf) is not None       # still found (no pop)
    del lf
    gc.collect()
    assert len(cache) == 0                          # weakref evicted on GC
```

- [ ] **Step 2: Run to verify both fail**

Run: `python -m pytest tests/python_integration/test_corr_engine.py::test_corr_repeated_collect_same_lf tests/python_integration/test_detect_common.py::test_lookup_does_not_pop_and_evicts_on_gc -v`
Expected: FAIL — corr 2nd collect raises `RuntimeError("corr spec handle missing")`; `dc.lookup` doesn't exist.

- [ ] **Step 3: Upgrade the cache in `_detect_common`**

In `python/polars_metal/_detect_common.py`, change the capture to store a weakref + exprs and add `lookup`:
```python
import weakref

# cache: id(lf) -> (weakref.ref(lf), exprs). Get-not-pop + weakref eviction so a
# repeated collect() of the same LazyFrame stays on the fast path (no O(N)
# lf.serialize()), growth stays bounded, and a reused id can't return stale exprs.

def install_with_columns_capture(attr: str, cache: dict) -> None:
    if hasattr(_plf.LazyFrame, attr):
        return
    orig = _plf.LazyFrame.with_columns
    setattr(_plf.LazyFrame, attr, orig)

    def _patched(self, *exprs, **named):  # type: ignore[no-untyped-def]
        result = orig(self, *exprs, **named)
        try:
            flat = [e for e in exprs if isinstance(e, pl.Expr)]
            flat += [e.alias(n) for n, e in named.items() if isinstance(e, pl.Expr)]
            if flat:
                key = id(result)
                cache[key] = (weakref.ref(result, _make_evictor(cache, key)), flat)
        except Exception:
            pass
        return result

    _plf.LazyFrame.with_columns = _patched  # type: ignore[method-assign]


def _make_evictor(cache: dict, key: int):
    def _evict(_ref) -> None:
        cache.pop(key, None)
    return _evict


def lookup(cache: dict, lf) -> list | None:
    """Return captured exprs for `lf` WITHOUT removing them (fast path survives
    repeated collect). Identity-validates via the stored weakref to reject the
    rare id-reuse case."""
    entry = cache.get(id(lf))
    if entry is None:
        return None
    ref, exprs = entry
    if ref() is not lf:  # id was reused by a different (live) object
        return None
    return exprs
```

- [ ] **Step 4: Switch each detector's fast path from `pop` to `lookup`**

In each `find_*_bindings`, replace `cached = _lf_exprs_cache.pop(id(lf), None)` with `cached = dc.lookup(_lf_exprs_cache, lf)`. Leave the slow serialize fallback unchanged (it still handles the rare cache-miss, e.g. an lf that had further ops chained after the sentinel layer). The cache now holds tuples, so any other direct `.pop`/`[id]` access in these modules must go through `dc.lookup` — grep each module for `_lf_exprs_cache` and `_*_exprs_cache` and convert all reads.

- [ ] **Step 5: Make the spec caches survive repeated collect too**

The expr cache now survives, but corr/dtw/vector also pop their *spec* cache (handle→spec) at dispatch, so a 2nd collect would still fail. For each of `_corr_dispatch.py` (`pop_capture`), `_dtw_dispatch.py` (`pop_capture`), `_vector_dispatch.py` (corpus/spec pop): change dispatch to READ without removing (`get`), and evict the spec via the SAME lf weakref. Concretely, in each namespace (`_corr_namespace.corr`, `_dtw_namespace.make_dtw_expr`, `_vector_namespace`), after building `result_lf = self._lf.with_columns(sentinel)`, register a weakref evictor that drops the handle from the spec cache when `result_lf` dies:
```python
        result_lf = self._lf.with_columns(build_corr_sentinel(cols[0], handle))
        # Tie the captured spec's lifetime to the returned lf so repeated collects
        # of the same lf reuse it, and it's freed when the lf is GC'd.
        weakref.finalize(result_lf, _CORR_CACHE.pop, handle, None)
        return result_lf
```
and change the dispatch's `pop_capture(handle)` to a non-removing `get_capture(handle)` (rename + keep returning the spec; the `weakref.finalize` now owns removal). Apply the same pattern to dtw and vector. (If `weakref.finalize` on a LazyFrame is rejected because LazyFrame is unhashable — finalize uses an internal registry, not a dict key, so it should work; verify with a one-liner. If it fails, fall back to `weakref.ref(result_lf, lambda _r, h=handle: _CACHE.pop(h, None))` stored in a module-level set to keep the ref alive.)

- [ ] **Step 6: Run the repeated-collect tests + full suite**

Run: `python -m pytest tests/python_integration/test_detect_common.py tests/python_integration/test_corr_engine.py -v` then `python -m pytest tests/python_integration/ -k "vector or fft or dtw or rolling or dt" -q`
Expected: the two new tests PASS; all verb suites still PASS. If a verb's repeated-collect now works, optionally add a one-line repeated-collect assertion to its test file (vector/dtw/fft) mirroring corr's.

- [ ] **Step 7: ruff + commit**

```bash
ruff check --fix python/polars_metal/_detect_common.py python/polars_metal/_*_detect.py python/polars_metal/_*_dispatch.py python/polars_metal/_*_namespace.py tests/python_integration/test_detect_common.py tests/python_integration/test_corr_engine.py
git add -A
git commit -m "M6 cleanup M7: weakref-evicted get-not-pop cache (fixes repeated-collect O(N) serialize)"
```

---

# PHASE 4 — FFT GPU interleave/split (M5)

Move the CPU interleave (`vec![0.;2n]` + stride-2 scatter) and split (stride-2 gather into two Vecs) onto the GPU. At N=2²⁴ this removes ~23ms of host-bandwidth work (the FFT kernel is already GPU). Two tiny MSL kernels: real→interleaved-complex pack, and interleaved-complex→planar unpack. Folds in the M3 readback (the unpack writes the two planar Metal buffers we read back).

## Task 10 (M5a): MSL pack/unpack kernels + kernel tests

**Files:**
- Create: `shaders/fft_pack.metal`
- Modify: `crates/polars-metal-kernels/src/fft.rs` (dispatch wrappers)
- Test: `tests/kernel/test_fft_pack.py` (or a Rust `#[cfg(test)]` in the kernels crate — match how `shaders/*.metal` are tested in `tests/kernel/`)

- [ ] **Step 1: Read an existing simple shader + its test harness**

Read `shaders/dt_gregorian.metal` (or another small shader) and its test in `tests/kernel/` to learn the project's MSL entry-point conventions, threadgroup sizing comment style, and how a `.metal` file is loaded/dispatched from `crates/polars-metal-kernels/src/`. Read the top of `crates/polars-metal-kernels/src/fft.rs` to see how the existing FFT library is compiled/loaded (`shared_library`, pipeline creation) — the pack kernels join the same `.metallib`/library path.

- [ ] **Step 2: Write the pack/unpack shader**

Create `shaders/fft_pack.metal`:
```metal
#include <metal_stdlib>
using namespace metal;

// Pack a real signal into interleaved complex: out[2i]=re[i], out[2i+1]=0.
// grid = n threads (one per sample). One .metal file per kernel family.
kernel void fft_pack_real_to_interleaved(
    device const float* re   [[buffer(0)]],
    device float*       out  [[buffer(1)]],   // length 2n
    constant uint&      n    [[buffer(2)]],
    uint                gid  [[thread_position_in_grid]]) {
    if (gid >= n) return;
    out[2 * gid]     = re[gid];
    out[2 * gid + 1] = 0.0f;
}

// Unpack interleaved complex into planar: re_out[i]=in[2i], im_out[i]=in[2i+1].
kernel void fft_unpack_interleaved_to_planar(
    device const float* in       [[buffer(0)]],   // length 2n
    device float*       re_out   [[buffer(1)]],
    device float*       im_out   [[buffer(2)]],
    constant uint&      n        [[buffer(3)]],
    uint                gid      [[thread_position_in_grid]]) {
    if (gid >= n) return;
    re_out[gid] = in[2 * gid];
    im_out[gid] = in[2 * gid + 1];
}
```
(If the input may be complex — `FftInput::Complex(re, im)` — add a third kernel `fft_pack_complex_to_interleaved(re, im, out, n)` writing `out[2i]=re[i]; out[2i+1]=im[i]`. Check `fft_core` for whether the Complex path is reachable from `.metal.fft()`; if only Real is reachable from the engine, still add it for the `execute_fft` Complex branch used by tests.)

- [ ] **Step 3: Write a kernel test (round-trip)**

Create the test (Python via `_native`, or Rust unit — match the repo). The behavior to verify: pack then unpack reproduces the input real signal and zero imaginary. If exposing via `_native` is heavy, prefer a Rust `#[cfg(test)]` in `crates/polars-metal-kernels/src/fft.rs` that dispatches both kernels on a small buffer and asserts the round-trip. Example Rust test shape:
```rust
#[test]
fn pack_unpack_roundtrip() {
    let device = MetalDevice::system_default().unwrap();
    let re: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let inter = dispatch_pack_real(&device, &re, 8).unwrap();      // len 16
    let (ro, io) = dispatch_unpack(&device, &inter, 8).unwrap();
    assert_eq!(ro, re);
    assert!(io.iter().all(|&x| x == 0.0));
}
```

- [ ] **Step 4: Run the kernel test to verify it fails then passes**

Run: `cargo test -p polars-metal-kernels pack_unpack -- --test-threads=1` (FAIL first — dispatch fns/kernels absent), then implement the `dispatch_pack_real`/`dispatch_unpack` wrappers in `crates/polars-metal-kernels/src/fft.rs` (mirror the existing FFT dispatch: create pipeline for the new entry points from the same library, set buffers, dispatch `n` threads), and re-run. Expected: PASS.

- [ ] **Step 5: kernel test in tests/kernel + fmt/clippy + commit**

Add the corresponding entry to `tests/kernel/` if the repo requires a Python-visible kernel test per `shaders/` file (CLAUDE.md: "Don't add files to `shaders/` without a corresponding test in `tests/kernel/`"). Then:
```bash
cargo fmt -p polars-metal-kernels && cargo clippy -p polars-metal-kernels -- -D warnings
git add shaders/fft_pack.metal crates/polars-metal-kernels/src/fft.rs tests/kernel/
git commit -m "M6 cleanup M5a: GPU fft pack/unpack kernels + round-trip test"
```

## Task 11 (M5b): wire pack/unpack into `fft_core`, drop the CPU interleave/split

**Files:**
- Modify: `crates/polars-metal-core/src/fft.rs` (`fft_core`)
- Modify: `crates/polars-metal-kernels/src/fft.rs` (expose a `fft_gpu_planar` entry that takes `re`/(`im`) and returns planar `re_out`/`im_out`, doing pack→fft→unpack all on device)

- [ ] **Step 1: Add a planar GPU entry that keeps interleave/split on-device**

In `crates/polars-metal-kernels/src/fft.rs`, add `fft_gpu_planar(device, re: &[f32], im: Option<&[f32]>, n, inverse) -> Result<(Vec<f32>, Vec<f32>), FftError>` that: stages `re` (+`im` or implicit zeros) to a Metal buffer, dispatches the pack kernel into an interleaved Metal buffer, runs the existing interleaved FFT on it (reuse `fft_gpu`'s core on the Metal buffer, not the host slice — refactor the existing `fft_gpu` to share its buffer-level body), dispatches the unpack kernel into two planar output Metal buffers, reads those back. Only the final two planar readbacks cross to host.

- [ ] **Step 2: Replace `fft_core`'s CPU interleave/split with the planar call**

In `crates/polars-metal-core/src/fft.rs` `fft_core`, delete the `let mut interleaved = vec![0.0f32; 2*len]` + scatter loop (lines ~25–38) and the output split loop (lines ~45–52); call `fft_gpu_planar(&device, re, im_opt, n, inverse)` and return its `(re_out, im_out)` directly. The `execute_fft` PyO3 wrapper still returns the two buffers as `PyBytes` (unchanged readback contract — the Python `np.frombuffer` side is already zero-copy), so M3's ~2.4ms PyBytes copy remains but is now the only host copy on the output side; leave it (a further `numpy`-crate move-out is out of scope).

- [ ] **Step 3: Run the FFT correctness tests (engine + kernel)**

Run: `cargo test -p polars-metal-core fft -- --test-threads=1` and `python -m pytest tests/python_integration/ -k "fft" -v` and `python -m pytest tests/kernel/ -k "fft" -v`. Expected: all PASS — output must remain L2 < 1e-3 vs numpy (the existing FFT differential tests are the oracle; do NOT loosen them).

- [ ] **Step 4: Re-bench FFT to confirm the win**

Run the FFT bench (find it: `ls tests/bench/ | grep fft`; e.g. `PYTHONPATH=. python tests/bench/bench_fft.py` or the relevant entry). Record before/after at N=2²³–2²⁴. Expected: ~15–23ms less host time at 2²⁴ (the interleave/split is now on GPU). Paste the table into the commit.

- [ ] **Step 5: fmt/clippy + commit**

```bash
cargo fmt -p polars-metal-core -p polars-metal-kernels && cargo clippy -p polars-metal-core -p polars-metal-kernels -- -D warnings
git add crates/polars-metal-core/src/fft.rs crates/polars-metal-kernels/src/fft.rs
git commit -m "M6 cleanup M5b: FFT interleave/split on GPU (drops ~20ms host work at 2^24)

<paste before/after FFT bench table>"
```

---

# PHASE 5 — Cosmetics, dead code, edge tests (C5)

## Task 12 (C5): grab-bag cleanups

**Files:** various (each step is independent; commit individually or as one).

- [ ] **Step 1: bench_corr force_gpu note**

In `tests/bench/bench_corr.py`, the sweep uses `force_gpu=True` at p∈{10,25,50} (all ≥ CORR_P_MIN=8, so routing would pick GPU anyway). Either drop `force_gpu=True` from the sweep (so it exercises the real routing path) OR add a one-line comment that it's there to force GPU at the small-p sweep points regardless of any future threshold change. Pick the comment (keeps the bench measuring the GPU path deterministically). Run `PYTHONPATH=. python tests/bench/bench_corr.py` to confirm it still runs and the gate passes.

- [ ] **Step 2: Remove `StubArena` dead code**

In `crates/polars-metal-core/src/arena.rs` and `lib.rs`: `StubArena` is re-exported but unused (audit 1.4). Grep `StubArena` across the repo to confirm zero non-definition references, then delete the struct + its `pub` re-export. If `BumpArena` is similarly unused (audit 2.4) and you can confirm zero references, leave it (it has integration tests and is documented as a deliberate dormant subsystem) — only remove `StubArena`. Run `cargo build -p polars-metal-core` to confirm nothing breaks.

- [ ] **Step 3: Fix stale FFT comment**

In `crates/polars-metal-kernels/src/fft.rs:24`, the comment says sizes above `FFT_BASE_MAX` "return `FftError::Unsupported`" — stale, since four-step/Bluestein paths now handle them. Update the wording to reflect that radix-2 covers ≤ base and larger sizes route to four-step/Bluestein. (Comment-only; no behavior change.)

- [ ] **Step 4: Add corr constant-column integration test + N=0 test**

In `tests/python_integration/test_corr_engine.py`, add an engine-level (not just kernel-level) constant-column test and an empty-frame test:
```python
def test_corr_constant_column_nan_via_engine():
    import numpy as np
    import polars as pl
    import polars_metal as pm

    rng = np.random.default_rng(21)
    cols = {f"c{i}": rng.standard_normal(2000).astype(np.float32) for i in range(9)}
    cols["c0"] = np.ones(2000, dtype=np.float32)  # zero-variance
    df = pl.DataFrame(cols)
    out = df.lazy().metal.corr().collect(engine=pm.MetalEngine())
    exp = df.corr().cast(pl.Float32)
    np.testing.assert_allclose(out.to_numpy(), exp.to_numpy(), atol=1e-4, equal_nan=True)
```
Run: `python -m pytest tests/python_integration/test_corr_engine.py -k "constant_column_nan_via_engine" -v`. Expected: PASS (matches Polars, NaN cells equal under equal_nan).

- [ ] **Step 5: ruff/fmt + commit**

```bash
ruff check --fix tests/python_integration/test_corr_engine.py tests/bench/bench_corr.py
cargo fmt -p polars-metal-core -p polars-metal-kernels
git add -A
git commit -m "M6 cleanup C5: bench comment, remove StubArena, fix stale FFT comment, corr edge tests"
```

---

## Task 13: Final gate + docs/memory update

- [ ] **Step 1: Full gate**

Run: `make gate`. Expected: lint clean, all unit/kernel/integration/conformance green (only the known pre-existing conformance deferrals, if any, unchanged). Fix any fmt/lint drift.

- [ ] **Step 2: Update the docs with the memory-pass results**

In `docs/open-questions.md` (and CLAUDE.md if a roadmap line references these costs), add a short "M6 memory pass (2026-06-11)" note: dt month/day ~30× cheaper output narrowing; FFT interleave/split moved to GPU (~20ms@2²⁴); repeated-collect O(N)-serialize fixed via weakref get-not-pop cache; streaming verbs now error clearly; vector null guard; `_detect_common` consolidation. Keep numbers honest (cite measured before/after).

- [ ] **Step 3: Update memory**

Update `/Users/dclark/.claude/projects/-Users-dclark-dev-polars-metal-main-polars-metal/memory/` — extend `[[metal-input-staging-pool]]` or add a short `m6-memory-pass.md` capturing the cross-cutting lessons (vector/rolling already zero-copy; dt cast-vs-astype; FFT GPU pack/unpack; the weakref get-not-pop cache pattern for repeated collect; `_detect_common`). Add the one-line MEMORY.md pointer. Link `[[proactively-hunt-big-copies]]`.

- [ ] **Step 4: Commit + push**

```bash
git add docs/ "/Users/dclark/.claude/projects/-Users-dclark-dev-polars-metal-main-polars-metal/memory/"
git commit -m "M6 cleanup: final gate, memory-pass docs + memory"
git push origin m6-vector-search
```

---

## Self-Review Notes (for the executor)

- **Coverage vs the audits:** M1 (Task 1), M2 (Task 2), M4 (Task 3), M6 (Task 4), C1 (Task 5), C3 (Task 6), C2 (Task 7), C4 (Task 8), M7 (Task 9), M5/M3 (Tasks 10–11), C5 (Task 12), gate+docs (Task 13). Vector-search and rolling ingest needed NO change (already zero-copy) — deliberately absent.
- **Phase independence:** Phases 1, 2, 5 are independent. Phase 3 is ordered (C4 then M7 — M7 edits the module C4 creates). Phase 4 (M5) is independent but biggest-risk (new MSL); if it slips, Phases 1–3 + 5 still ship a complete memory pass.
- **Oracle discipline:** every behavioral change is gated by a Polars/numpy differential test; FFT changes must hold L2<1e-3, corr byte/F32-tol, dt byte-exact. Never loosen a tolerance to pass.
- **Risk flags for the executor:** Task 3 (M4) depends on `from_borrowed_i32` existing — verify first, STOP if absent. Task 9 (M7) step 5 (`weakref.finalize` on a LazyFrame) — verify with a one-liner before relying on it; documented fallback included. Task 11 (M5b) refactors `fft_gpu` to share a buffer-level body — keep the existing host-slice `fft_gpu` entry working (other callers/tests use it) or update them.
