# A4 — `.metal.dtw`: Sakoe-Chiba-banded DTW kernel — Design

> M6 Track A, item A4 (the pairwise-distance sub-project). Branch `m6-vector-search` (PR #6), reusing the A2 vector-search `.metal`-namespace machinery. Spec supersedes the "speculative, thin spec" placeholder in `docs/superpowers/specs/2026-06-04-m6-metal-namespace-design.md` §A4. See [[m6-scope-and-api-direction]] for the umbrella and [[m6-vector-search-execution-state]] for the reused assets.

## Summary

Add `pl.col("seq").metal.dtw(reference, window=None, allow_cpu_fallback=False)` — a Dynamic Time Warping distance between each row's fixed-length F32 sequence and one broadcast reference sequence, executed on a custom Metal kernel under `collect(engine="metal")`. Euclidean DTW (matches `dtaidistance`), optional Sakoe-Chiba band, one `Float32` distance per row.

## Why this is a real win (spike-grounded, 2026-06-10, M2 Ultra)

DTW is the **opposite of B4**: B4's bare reductions were bandwidth-bound (1 flop/element) and lost on GPU because the host→MLX ingest exceeded Polars' SIMD scan. **DTW is compute-bound** — `O(L²)` work per pair, high arithmetic intensity; ingest (`N·L` floats in, `N` floats out) is negligible against `N·L²` compute. It sits squarely in the Mission's target class ("compute-shaped F32 work wins").

Measured against `dtaidistance` (the standard multi-threaded C DTW library — the fair bar, per the vector-search "match the baseline to the real algorithm" honesty lesson):

| | N=10k | N=100k |
|---|---|---|
| L=64 | 2.6× | 19× |
| L=128 | 3.1× | 25× |
| L=256 | 2.9× | 23× |

**Honest headline: ~23× at N=100k**, and that is a *conservative floor* — the spike's GPU path was an overhead-laden MLX graph (L² nodes) used only as a proxy; the shipped custom kernel should exceed it. The win **scales with N** (more independent pairs → more parallelism) and has a **small-N floor** (≲1–5k rows: GPU dispatch overhead dominates, ~break-even or slower — same structural floor B4 documented). Spike scripts: `scripts/spike_a4_dtw.py` (+ the dtaidistance race inline in the session).

**Caveat carried forward:** the ~23× is vs `dtaidistance`; a naive numpy DTW baseline would inflate it (vectorized-numpy was ~73s vs dtaidistance ~38s at N=100k/L=256 — same ballpark, both crushed by the GPU's 1.6s). The benchmark and any quoted number use `dtaidistance` as the bar.

## API & semantics

```python
pl.col("seq").metal.dtw(reference, window=None, allow_cpu_fallback=False)
```

- **`seq`** — an `Array(Float32, L)` column (fixed inner length `L`). The supported GPU shape.
- **`reference`** — one length-`L` sequence (numpy array / Python list / `Array` literal), **broadcast to every row**. Length must equal `L`.
- **`window`** — Sakoe-Chiba radius (non-negative int) or `None`. Cell `(i, j)` is evaluated iff `|i − j| ≤ window`; `None` = unconstrained full DTW. **Differential-oracle mapping (equal-length): our `window=w` ⇄ `dtaidistance(window=w+1)`** (verified 2026-06-10).
- **`allow_cpu_fallback`** — `bool`, default `False`. Governs **unsupported inputs only** (see below), not a perf switch.
- **Output** — one `Float32` per row, **height-preserving**. **Euclidean DTW**: cell cost `(q_i − r_j)²`, accumulated along the optimal warping path, **`sqrt` of the final cumulative** (byte-confirmed equal to `dtaidistance.distance`). A **null Array row → null distance** (positional restore, mirroring `dt`'s `dense.set(~mask, None)`).

### Unsupported inputs and the fallback switch

"Unsupported on the GPU kernel" = ragged/`List` sequences, non-F32 element dtype, `reference` length ≠ `L`, or `L` above the threadgroup-memory bound (≈2048).

- **`allow_cpu_fallback=False` (default):** raise a clear `ComputeError` (matches the A2 `.metal` precedent — explicit verb, explicit failure).
- **`allow_cpu_fallback=True`:** compute the result on CPU. The CPU engine **lazily imports `dtaidistance`** (the canonical lib and our differential oracle — avoids maintaining a second DTW implementation). If `dtaidistance` is not installed, raise a clear "install dtaidistance for CPU fallback" error. This makes `dtaidistance` an **optional runtime dependency**, loaded only on an explicit opt-in path that is actually triggered. *(Open for veto: alternative is a slow in-house numpy DTW with no runtime dep — see Open questions.)*

Length-mismatch between `reference` and the column's `L` is always an error regardless of the switch (it's a user mistake, not an unsupported-but-valid shape).

## Architecture

### Recognition & dispatch — reuse A2 machinery

`dtw` is a **height-preserving per-row verb**, so it reuses the A2 **`as_struct` sentinel** path end-to-end (the same one `cosine_topk` uses):

- **Namespace:** add a `dtw` method to the existing `.metal` expression namespace in `python/polars_metal/_vector_namespace.py` (`register_expr_namespace("metal")`). It builds a sentinel: the `seq` column + a handle-id-tagged literal capturing `(reference, window, allow_cpu_fallback)` + the CPU-raising `map_batches(_raise)` field. Reference/window are held in a capture dict keyed by handle id and `pop`-ed on consume (exactly like `_CORPUS_CACHE`).
- **Detect:** `python/polars_metal/_dtw_detect.py` — its own `with_columns` monkey-patch + cache (separate from rolling/vector/fft, chained), serialize-detecting the dtw sentinel shape. Mirrors `_vector_detect.py`. `find_dtw_bindings(lf)`.
- **Dispatch:** `python/polars_metal/_dtw_dispatch.py` — `apply_dtw(lf, bindings, collect_fn)`: collect-and-stitch (drop sentinel cols so projection pushdown elides them, CPU-collect the rest, stage the `Array` column + reference as F32, call the kernel, restore nulls positionally, reassemble in schema order). Mirrors `_vector_dispatch.py`. Output is a single `Float32` column (simpler than A2's struct).
- **Wire:** `python/polars_metal/__init__.py` — import `_dtw_detect` eagerly (installs the patch) and add a dtw block in `collect_wrapper` after the FFT block (`fft_bindings` → `vector` → `rolling` chain).

This sidesteps the NodeTraverser opacity wall entirely — an op we own needs no recognition ([[m6-scope-and-api-direction]]).

### The kernel — `shaders/dtw.metal`

Entry point `dtw_banded`. **Best-perf threading: one threadgroup per query row (pair); the threadgroup's threads cooperate on the anti-diagonal wavefront** (DTW's only intra-pair parallelism — cells on an anti-diagonal `k = i+j` are mutually independent; each depends only on diagonals `k−1`, `k−2`).

- **Threadgroup memory** holds the reference (`L` f32, loaded once per group) + the rolling DP state. Two rolling rows (or three rolling anti-diagonals) of `≈L` f32 suffice: `≈3·L·4B` (L=1024 → 12 KB, within the 32 KB threadgroup floor; query box runtime-queries `MTLDevice` per the threadgroup-sizing gotcha).
- **Squared-diff cell cost; final `sqrt`.** Band: a thread only touches `j ∈ [max(1, i−w), min(L, i+w)]`; out-of-band cells are implicitly `+∞`.
- **L bound** ≈ 2048 (threadgroup-mem limit). Larger `L` or ragged → the `allow_cpu_fallback` path (or raise). Documented at the top of the file with threadgroup/grid assumptions (per "one MSL kernel per file" convention).
- **Grid:** `N` threadgroups (one per pair); within-group thread count tuned to the wavefront width (device-queried).

### Rust dispatch & FFI

- `crates/polars-metal-kernels/src/dtw.rs` — `DtwError`, `dispatch_dtw_buf` (zero-copy core over pre-staged `MetalBuffer`s — the PyO3 path) + `dispatch_dtw` (slice wrapper for kernel tests). Mirrors `rolling.rs`/`dt.rs`. Add `pub mod dtw;` to `lib.rs`.
- `crates/polars-metal-core/src/udf.rs` — `execute_dtw` PyO3 entry: stages the `N·L` F32 input + `L` F32 reference via **`from_borrowed_f32`** (zero-copy when page-aligned; copy-back fallback otherwise), output `N` F32. **No StagingPool** — DTW is compute-bound, ingest is negligible (unlike `dt`, which needed it). Register in `lib.rs`.
- The shader auto-compiles via `crates/polars-metal-kernels/build.rs` (every `shaders/*.metal` whose stem doesn't start with `_`) — dropping `shaders/dtw.metal` is sufficient.

## Testing — `dtaidistance` as the differential oracle

`dtaidistance` (C, fast) is the oracle; metric matches exactly, window maps `our w ⇄ lib w+1` (equal-length).

- **Kernel-level** (`crates/polars-metal-kernels/tests/test_dtw.rs`): vs a CPU scalar DTW reference (the textbook `O(L²)` DP) — multiple `L`, full + banded, identical-sequences→0, single-element, anti-diagonal correctness, `n=0/1` rows.
- **Engine-level** (`tests/python_integration/test_dtw_e2e.py`): differential vs `dtaidistance` — sweep `L ∈ {16,64,256}`, `N`, `window ∈ {None, 0, 1, small, ≥L}`, random + edges; **F32 tolerance** (`abs_tol` ~1e-3, GPU f32 vs C f64 accumulation); ≥1 genuine GPU-path case proven via an `execute_dtw` dispatch counter (the B2 lesson: assert dispatch==1, not just equals-oracle); null-bearing rows; `allow_cpu_fallback` True/False on a ragged input (raise vs correct).
- **Detect-level** (`tests/python_integration/test_dtw_detect.py`): sentinel recognized; non-dtw exprs ignored; coexistence with rolling/vector/fft in separate layers.

## Perf & routing

- **Honest target ~23× at N=100k vs `dtaidistance`** (conservative; custom kernel should exceed). Banded `window` cuts compute to `O(L·w)`.
- `.metal.dtw` is an **explicit opt-in verb** → **always route to GPU when detected** (no auto-threshold; the user chose Metal). The **small-N floor** (≲5k rows ≈ break-even) is documented in the docstring — transparency like B4, but no silent fallback (the user asked for it).
- **Benchmark:** `tests/bench/m4_survey/bench_dtw.py` (cpu=`dtaidistance` vs gpu, swept N/L/window) + a `baseline.json` entry with a `ratio_lt` gate (GPU < CPU at the headline N=100k shape). Matches the bench conventions B4 used.

## Scope / non-goals

- **In:** `.metal.dtw(reference, window, allow_cpu_fallback)`, equal-length `Array(F32, L)`, single broadcast reference, Euclidean metric, Sakoe-Chiba band, `Float32`/row out, nulls (positional), the engine path, `dtaidistance` differential oracle + bench bar.
- **Out:** ragged/`List` sequences on GPU (→ fallback or raise); row-aligned two-column DTW (`dtw(col_a, col_b)` — future); the warping *path* (distance only); non-Euclidean metrics; multivariate/dependent DTW; Levenshtein / string distances (standing Non-goal); cross-collect caching (an explicit future `build_index()`, never a hidden cache).

## Dependencies

- **`dtaidistance`** — added as a **test/dev dependency** (differential oracle + bench baseline). Justification (per CLAUDE.md "no new dependency without written justification"): it is the de-facto-standard fast DTW library, gives a free ULP-class differential oracle (the repo's testing strategy), and a fair perf bar. It also serves as the **optional runtime** engine for `allow_cpu_fallback=True`, lazily imported only when that path is both enabled and hit (clear error if absent) — so it is not a hard runtime dependency of the engine.

## Open questions (resolve at plan/drill time)

- **CPU-fallback engine:** lazy `dtaidistance` (chosen — single source of truth, fast) vs a slow in-house numpy DTW (no runtime dep). Default-off either way; the switch's *behavior* (raise vs compute) is fixed.
- **Within-group threading detail:** exact thread-count-per-pair and the anti-diagonal buffer layout (2 rolling rows vs 3 rolling diagonals) — a kernel micro-design decision; pin during the kernel drill with a threadgroup-occupancy check.
- **`L` bound exact value** — derive from the runtime-queried threadgroup memory limit, not hardcoded (the M1/M2/M3/M4 threadgroup-portability gotcha).
- **Reference passing** — handle-id capture (like A2 corpus) vs inline literal; the reference is small (`L` f32), so either works; pick the simpler at impl time.
