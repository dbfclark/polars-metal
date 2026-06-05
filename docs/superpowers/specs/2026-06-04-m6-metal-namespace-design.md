# M6 — The `.metal` Namespace (+ Integer Support) Design

**Status:** Approved (brainstorm 2026-06-04). Umbrella/program spec; the vector-search
sub-project is drilled to implementable depth here. Other sub-projects (FFT, pairwise, the
Track-B integer work) are thin specs to be drilled in their own brainstorm→spec→plan cycles.

**Goal:** Ship the deferred "fancy" compute ops (roadmap items 10–13) by introducing one
coherent user-facing surface — the `.metal` namespace — for GPU-only verbs, plus extending
the engine to integer dtypes. Deliver real compute-bound wins; do **not** chase a single new
headline benchmark. Also assess master-plan position (consolidation deliverable).

---

## Why this design (decision record)

The deferred items (FFT, list/array dot, `corr`, `dt.*`, pairwise distance) were each blocked
on the **NodeTraverser opacity wall** ([[m4-nodetraverser-opacity]]): the py-1.40.1 engine
plugin can't *recognize* list/array/`corr`/`reshape`/`int_range` expressions, so the walker
can't route them. M5 hit this twice and pivoted to focused detection + a custom kernel.

The M6 framing memo proposed "build the general serialize→scope opacity unlock." Brainstorming
**overrode** that in favor of **delivery-first**, and surfaced a cleaner organizing principle:

1. **An op we *own* doesn't need recognition at all.** The entire opacity/sentinel/serialize
   apparatus exists to intercept *native* Polars ops. If we expose our own `.metal.*` verbs,
   we dispatch them directly — the opacity wall is **sidestepped, not climbed.** This is the
   real consolidation: one new surface, recognized one way, replacing four per-op opacity fights.

2. **The native shapes are bandwidth-bound; the wins are batched/compute-bound.** The
   recognizable native lazy shapes are roofline losers (the project's non-goal):
   - single-query dot `(col*lit).arr.sum()` is a **GEMV** (~0.5 FLOP/byte) — bandwidth-bound;
   - `pl.corr(a,b)` is ~constant-work-per-element — bandwidth-bound.

   The survey wins came from the **batched** forms — cosine top-k 29× (Q×N GEMM, each db
   vector reused Q times → arithmetic intensity ~2Q → compute-bound), `df.corr()` 7.8× (M×M
   matrix). These are not clean native expressions, which is *why* they want an invented
   `.metal` verb. (Note: this also retired the "recognize native dot/corr" approach.)

3. **Integer plumbing is a horizontal capability, not a `dt` private detail.** `dt.*`
   acceleration needs net-new Int32/Int64 buffer/bridge/kernel support (the engine is F32-only).
   Since we pay for it anyway, we extend the *whole* fused walker to integers and re-baseline
   the benchmark suite — `dt` becomes the flagship compute-bound consumer, not the sole one.

**Decisions (architect-approved):**

- **M6 objective = ship the deferred ops** (not "build the general unlock as a subsystem").
  The serialize→scope analyzer is built only as far as a given op needs it.
- **M6 = the `.metal` namespace suite** (Track A: FFT, vector search, pairwise) **+ native
  integer/`dt` support** (Track B). `df.corr()` (eager matrix) is **split out** to its own
  later sub-project; the lazy `pl.corr(a,b)` scalar is bandwidth-bound and **not pursued**.
- **Track A is lazy / engine-integrated.** `.metal.*` methods emit recognizable markers that
  `collect(engine="metal")` dispatches; without `engine="metal"` they raise cleanly.
- **Two recognition mechanisms, composable:**
  - the `as_struct` expression sentinel ([[m4-nodetraverser-opacity]]) recognizes
    *height-preserving* per-row verbs. Used by **FFT**, **pairwise-vs-literal**, and (for op
    recognition) **vector search**.
  - the M5 **capture + collect-and-stitch** pattern ([[m5-rolling-execution-state]]) carries
    *by-reference arguments* (and truly cardinality-changing verbs). **Vector search** uses it
    only to hold the corpus `LazyFrame` handle — its op is recognized via the sentinel, since
    queries-as-frame keeps it height-preserving (Q in, Q out).
- **Vector search = "option 3, lazy corpus."** Queries are the frame (height = Q, preserved →
  rides the per-row sentinel kind); the corpus is a **`LazyFrame` argument** materialized at
  dispatch via a normal Polars collect (full optimizer: predicate/projection/slice pushdown).
- **No reuse cache.** Intra-collect reuse/optimization is the lazy engine's job — delegate
  corpus materialization to Polars, own only the GEMM + top-k. Cross-collect "index once,
  query many" reuse is **inherently stateful, outside the lazy model, and deferred** to a
  future explicit `build_index()` handle — never a hidden identity-keyed GPU cache (that
  reintroduces the M5 stale-id / eviction footguns).
- **F32-first stands** (no Apple-GPU f64 / no Metal `double` / MLX refuses f64 on GPU —
  confirmed). Track B adds **Int32/Int64** only.
- **Sequencing:** vector search → FFT → integer/`dt` → pairwise (re-confirm pairwise on arrival).

---

## Program architecture (umbrella)

```
                         M6
        ┌─────────────────┴───────────────────┐
   Track A: .metal namespace            Track B: integer support
        │                                      │
   A1 shared machinery                    B1 Int32/Int64 buffer/bridge
   A2 vector search   ◀── FLAGSHIP        B2 int parity in fused walker
   A3 fft                                 B3 dt gregorian MSL kernel
   A4 pairwise (speculative)             B4 re-baselined int benchmarks
        │                                      │
        └──────────────┬───────────────────────┘
                       │
          Consolidation: roadmap-status audit + measure unblocked candidates
```

Each lettered item is its own spec→plan→implement cycle. This document drills **A2** to
implementable depth and stubs the rest.

### A1 — shared `.metal` namespace machinery

- Register a Polars `metal` namespace (expression and/or LazyFrame, via
  `pl.api.register_*_namespace`). Methods **do not compute**; they emit a recognizable marker.
- **Per-row expr verbs** emit the `as_struct` hybrid sentinel: a viewable `Function('as_struct')`
  carrying the op tag + scalar params in viewable fields, plus an opaque `map_batches` marker
  that raises cleanly on plain CPU. The walker reads op+params, dispatches, folds back. Confirm
  via output dtype.
- **Frame/cardinality verbs** use the M5 capture: the namespace method records intent
  (by-reference args, params) in a side dict keyed by the result frame `id()` with
  **pop-on-consume** eviction; dispatch at `collect(engine="metal")` runs MLX and stitches the
  result. Never `json.loads` a full plan (M5 gotcha: `lf.serialize` embeds the DataFrame).
- CPU behavior (no `engine="metal"`): raise a clear `requires engine="metal"` error rather than
  silently wrong results.

---

## A2 — Vector search (FLAGSHIP, drilled)

### API (expression namespace)

```python
result = queries_lf.with_columns(
    pl.col("emb").metal.cosine_topk(corpus_lf, k=10).alias("hits")   # also .metal.knn(...)
).collect(engine="metal")

# result = queries_lf's columns + "hits", height Q. "hits" is one column:
#   Struct{ indices: List[UInt32](len k), scores: List[Float32](len k) }
# split with: pl.col("hits").struct.field("indices") / .struct.field("scores")
```

- **Namespace kind:** **expression** namespace (`pl.api.register_expr_namespace("metal")`). The verb
  is a per-row expression on the query embedding column → one struct column, height-preserving.
- **Verbs:** `cosine_topk(corpus, k)` (cosine similarity, descending) and `knn(corpus, k)`
  (**true L2 distance**, `sqrt`, ascending). Both = normalize/prep + one GEMM + one top-k.
- **Queries:** the frame's `emb` column (`Array[Float32, D]`, height Q). Composes with arbitrary
  query-metadata columns (carried through unchanged).
- **Corpus:** a `LazyFrame` (also accept eager `DataFrame` / numpy `(N,D)` F32), with an
  `Array[Float32, D]` embedding column named by a `corpus_col=` arg (default `"emb"`). Captured
  by-reference under a monotonic **handle-id**; materialized at dispatch via a normal Polars
  `.collect()` so pushdown applies — only surviving rows' embedding column reaches GPU memory.
- **Output:** one `Struct{indices: List[UInt32], scores: List[Float32]}` column per query row,
  length k. Tie-break: score (sim desc / dist asc), then `index asc`.
- **Mismatch → raise** (resolved): non-F32 embedding, ragged/non-Array `List`, D-mismatch
  query↔corpus, missing corpus column, or use without `engine="metal"` → clear `ComputeError`/raise.
  No silent CPU fallback (the op has no native Polars equivalent to fall back to).

### Mechanism (expr namespace + handle-id capture, M5 collect-and-stitch)

Recognition reuses the **M5 template** (serialize-detect + collect-and-stitch), *not* the
post-opt NodeTraverser. The expr namespace is just the user-facing surface:

1. `pl.col("emb").metal.cosine_topk(corpus, k, corpus_col="emb")` allocates a monotonic
   **handle-id**, stashes `(corpus, corpus_col, k, metric)` in a module-global capture dict
   under that id (pop-on-consume eviction), and returns a **sentinel expression**: a struct-shaped
   marker carrying the query column ref + handle-id as a viewable literal, plus an opaque
   `map_batches(_raise)` field so plain-CPU collect (no `engine="metal"`) **raises** rather than
   silently mis-computing.
2. `collect(engine="metal")` dispatch (in `collect_wrapper`): serialize-detect the sentinel
   (find `(out_name, query_col, handle_id)` bindings), `lf.drop(out_names)` → CPU-collect the rest
   (projection pushdown elides the sentinel), then for each binding pop the corpus from the cache,
   `corpus.collect()` (Polars optimizer/pushdown) → run the GPU op → stitch the result struct
   column back in schema order. Same shape as `_rolling_dispatch.apply_rolling`.

### Backend (MLX composition + new FFI building blocks)

The GEMM exists (`mlx_op_matmul`); on-GPU top-k extraction needs **5 new MLX FFI wrappers**
(reusable for FFT/future ops): `transpose`, `reshape`, `slice` (first-k along an axis),
`take_along_axis` (gather), and **I32 readback** (`argpartition` returns I32; none exists today).

- **Buffer staging:** host query/corpus F32 `(rows·D)` → `MetalBuffer::from_borrowed_f32` →
  `mlx_array_view_metal_buffer(buf, [rows, D], F32)` (the existing 2-D view path; same staging as
  `execute_fused_expr`).
- **cosine:** normalize each side `x / ‖x‖₂` (`square`→`sum_axis(1)`→`reshape (rows,1)`→`sqrt`→
  `div` broadcast); `sims = Qn @ transpose(Cn)` → `(Q,N)`; `idx = argpartition(neg(sims))`,
  `slice [:, :k]` → `(Q,k)` indices; `take_along_axis(sims, idx_k)` → `(Q,k)` scores; sort the k
  for ranked order (sim desc).
- **knn (true L2):** `‖q−c‖² = ‖q‖² + ‖c‖² − 2·q·cᵀ` (GEMM is the cross term, norms broadcast);
  smallest-k via `argpartition(d2)`; `sqrt` the gathered `(Q,k)` distances; ascending.
- **Readback:** scores `(Q,k)` via `mlx_array_copy_to_f32`; indices `(Q,k)` via the new I32
  readback → host. Build the `Struct{indices, scores}` column from the `(Q,k)` host arrays.
- **Tiling over N:** the `(Q,N)` matrix can be large (Q=100,N=1M = 400 MB). Above a byte threshold,
  tile the corpus over N: per tile compute `(Q,tile)` top-k, merge into a running per-query top-k
  (host-side merge of `(Q,k)` partials with index-offset correction). Each tile resident.

### Scope / fallbacks

- **F32 only.** Non-F32 embedding, ragged `List` (non-fixed D), D-mismatch query/corpus → clear
  raise (or CPU brute-force fallback — see open question).
- **In-memory corpus only.** If the filtered corpus column exceeds unified memory we tile; truly
  out-of-core / streaming-during-GEMM is out of scope (adapter is in-memory by construction, M5).
- **No cross-collect reuse** (deferred `build_index()`).

### Correctness oracle

Brute-force numpy/Polars cosine + L2 top-k on random inputs (incl. D=1, k=1, k≥N clamped, ties,
empty corpus, single query). Assert exact index/score match under the defined tie-break; F32
tolerance only where normalization/accumulation order legitimately differs (document it).

### Perf target

Compute-bound GEMM should reproduce survey-class wins (cosine top-k ~29× vs NumPy at
Q≈100, N≈1M, D≈768). Add a `tests/bench/` case; gate with `ratio_lt` (correctness/cliff guard,
not a headline).

---

## A3 — FFT (thin spec)

- **Verb:** `pl.col("signal").metal.fft()` → `Struct{real: F32, imag: F32}` (per-row expr kind;
  the `as_struct` sentinel is the *exact* mechanism — op already wired in `fusion/subgraph.rs`).
- **Remaining work:** complex → `Struct[real, imag]` FFI readback; `ifft`; window/axis semantics;
  match numpy/Polars-CPU reference. Drill in its own cycle.

## A4 — Pairwise distance (speculative, thin spec)

- **Verbs (TBD):** `.metal.dtw(other)` / `.metal.levenshtein(other)` — pairwise vs a literal/
  reference sequence (per-row sentinel kind) or pairwise-within-column (frame kind). New MSL
  kernels; genuine Polars vocabulary gap. **Re-confirm operand shape and value on arrival**;
  candidate for demotion to M7 if the other tracks consume the milestone.

---

## Track B — Integer support (thin spec)

- **B1 — buffer/bridge:** Int32/Int64 `ColumnBuffer` alongside `Vec<f32>`; Arrow ↔ MTLBuffer for
  integer columns, validity preserved.
- **B2 — fused walker int parity:** extend the M4 fused element-wise path to integers (arithmetic,
  comparison, cast, bit ops, compute-bound reductions). **Gated by the existing FLOPs/row
  routing** — bandwidth-bound int (bare `sum`/`min`/filters/`group_by` keys) stays on CPU; only
  compute-bound int chains route to MLX. Null semantics match Polars exactly.
- **B3 — `dt` gregorian kernel:** `dt.year/month/day` via a custom MSL gregorian-calendar kernel
  (~30–40× target); the flagship compute-bound int consumer. `dt` recognition is native (likely
  opacity-bound — verify the path during its drill).
- **B4 — re-baselined benchmarks:** integer variants of the survey workloads so the perf story
  isn't F32-only; record baselines + gates.

---

## Consolidation deliverable

- Roadmap-status audit: per roadmap item, mark shipped / conformance-only / deferred; reconcile
  CLAUDE.md, the master plan, and reality after M6.
- Measure the now-unblocked candidates (vector search; FFT once landed) and record honest numbers.
- Update [[m6-scope-and-api-direction]] with what actually shipped vs. deferred.

---

## Resolved (brainstorm 2026-06-04/05)

- **F32-mismatch → raise** (no silent fallback; the op has no native equivalent).
- **Namespace = expression** namespace (`register_expr_namespace`); output a single `Struct` column.
- **`knn` → true L2** (`sqrt`), ascending.
- **Top-k path = MLX composition** + 5 new reusable FFI wrappers (transpose, reshape, slice,
  take_along_axis, I32 readback).

## Open questions (deferred to plan/drill time)

- **Tiling threshold** for the `(Q,N)` matrix — fixed bytes vs. queried from `MTLDevice`.
- **`argpartition` 2-D semantics** — confirm last-axis default + tie-break in a kernel test before
  relying on it (first Phase-0 task).
- **dt recognition path** — confirm whether `dt.year/month/day` is NodeTraverser-viewable or needs
  serialize/sentinel handling (Track-B drill time).
