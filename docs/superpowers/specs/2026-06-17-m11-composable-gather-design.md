# M11 — Composable resident gather + retrieval flagship (design)

> Status: approved design, pre-plan. Turns M10's narrow `Scan join Scan` single-column
> resident gather into a generally-useful multi-column gather, unblocks the end-to-end
> retrieval pipeline, and adds `.select` recognition for the `.metal` verbs. **No new
> execution architecture** — the scope was collapsed empirically during brainstorming.

## Background

M10 shipped a resident GPU gather for `fact.join(dim, on=<int key>) → F32 chain`: when the
dim key is a dense `0..n-1` permutation and all fact keys are in range, the gather + chain run
resident on GPU (one MLX eval). It is **MVP-narrow**: the dim may contribute **exactly one** non-key value column (multi-column
dims fall to the 2.16× CPU-lookup branch instead of the 4.7–7.1× resident path). **M11 removes
this single-column limit** — the one real gap the spikes found.

`_walk_join` also requires both join inputs to be plain `Scan`s. M11 **keeps** that requirement;
the retrieval flagship satisfies it by materializing the fact (an eager explode), which the spikes
showed already fires the resident gather. (Relaxing the Scan requirement — auto-staging a non-scan
fact side — is the one genuinely-new architecture piece and is explicitly out of scope.)

M10 also left the `.metal` vector verbs un-composable (their opaque sentinel can't be read
downstream) and the verb family invisible to `.select`. M11 addresses `.select`; the sentinel
rework is out of scope (the two-phase retrieval doesn't need it).

### What brainstorming spikes established (so we don't re-derive)

- **Resident gather over a *materialized* fact already works.** A spike confirmed that
  `materialized_fact.lazy().join(dense_dim, on=key).with_columns(F32 chain)` — where the fact is
  a `DataFrame` (so its `.lazy()` is a `Scan`) — **fires M10's resident gather** (`_M10_DENSE_GATHERS
  == 1`, byte-exact). So "resident gather over a non-scan fact side" reduces entirely to
  **materializing the fact first**, and the explode that makes a retrieval fact non-scan can be
  done eagerly (cheap). **No staging architecture is required.**
- **Multi-column is the one real gap.** The same spike with a two-column dim (`price`, `rating`)
  routed to the CPU-lookup branch (`_M10_DENSE_GATHERS == 0`) because of the single-dim-col guard.
- **The retrieval pipeline composes correctly** (byte-exact vs CPU) as a normal LazyFrame once its
  fact is materialized; the metadata lookup is a dense gather, not a hash join.
- **`cosine_topk`'s opaque sentinel blocks only *inline* downstream use**, and inline use is **not
  needed** for the resident retrieval win (the two-phase form below delivers it). Reworking the
  verb to a real struct is therefore **out of scope** for M11 (pure ergonomics, highest blast
  radius). The column-output verbs (`rolling`/`dt`/`fft`/`dtw`) already compose downstream.

## Goal

1. **Multi-column resident gather** — a `fact.join(dim) → F32 chain` where the chain reads
   several dim value columns runs the gather resident on GPU, gathering all needed columns in one
   MLX subgraph.
2. **`.select` recognition** — the five `.metal` stitch verbs (rolling, vector, fft, dtw, dt) and
   `corr` are detected under `LazyFrame.select`, not only `with_columns`.
3. **Retrieval flagship** — an end-to-end `embeddings → cosine_topk → explode → join metadata →
   rerank` benchmark, resident, measured engine-vs-CPU (medians) in the M8 registry.

Honest baselines only (engine=metal vs Polars CPU + raw-MLX/numpy ceiling, medians).

## Non-goals / guardrails (unchanged from M10/M9)

- **No GPU hash join.** The metadata lookup is a dense gather; non-dense keys → CPU lookup.
- **F64 stays CPU** (no GPU FP64); chains must be F32-fusable.
- **groupby / sort / hash-join kernels not extended** past conformance.
- **No `cosine_topk` sentinel rework** (deferred — inline one-LazyFrame retrieval is not needed
  for the win).
- **No auto-staging of a non-scan fact side at a join boundary** (the one genuinely-new
  architecture piece; the two-phase retrieval delivers the win without it).

## 1. Multi-column resident gather

### Recognition (`_walk_join` / `_attach_gather_scope`, `python/polars_metal/_walker.py`)

M10 gates the resident path on `len(_dim_value_cols) == 1`. Relax to **N ≥ 1**:

- `_dim_value_cols` = all non-key columns of the right (dim) schema (already computed).
- The fused chain may reference **any subset** of them. Identify the **gather set** = the dim
  value columns the chain actually reads (leaves whose name is a dim value column).
- Build the gather scope for the full gather set (see analyzer below). Stash on the Join plan:
  `_gather = {scope, descriptors, out_dtype, gather_cols: [...], key_col, out_name}` (note
  `gather_cols` is now a list) and `_out_schema` (the HStack output schema, ordered — for exact
  column reconstruction).

### Analyzer (`analyze_ir_with_columns_gather`, `python/polars_metal/_fusion_analyzer.py`)

M10 splices one `Take(dim_value, key)` for one `gather_col`. Generalize the gather-aware pass:

- PASS 1: on first encounter of the **key** add one shared `("gather_key", key_col)` input (I32).
  For **each distinct dim value column** referenced by the chain, add one
  `("gather_value", col)` input (F32, SHORT = dim_n) and record its idx in
  `gather_ctx["idxs"][col]`.
- PASS 2: on `Column(c)` where `c` is in the gather set, push `Take([value_idx[c], key_idx])` and
  return that node idx. The key input is shared across all `Take`s.
- Every other leaf stages normally (LONG = N). The base `analyze_ir_with_columns` stays
  byte-identical when no gather context is passed.

This yields a single MLX subgraph: N `Take`s of short dim columns by the shared long key, feeding
the F32 chain — one eval, cross only the result.

### Dispatch (`_try_resident_gather`, `python/polars_metal/_udf.py`)

- Compute `reordered[col]` for **each** non-key dim column via `dense_positions` on the shared dim
  key (the permutation is shared; reorder each value column by it).
- Build inputs in descriptor order: one int32 `key` (shared), one F32 SHORT `reordered[col]` per
  `("gather_value", col)`, plus the fact `("col", …)` / `("lit", …)` inputs. Run
  `execute_fused_expr`, output length = fact rows.
- **Output frame:** the chain output column **plus every non-key dim column** the HStack output
  schema carries (a `with_columns(join(...))` keeps them all), each produced via
  `reordered[col][key]` (cheap CPU dense index, as M10 does for the single `vol` column —
  generalized to N). Reorder to `_out_schema`. dtypes/null positions must match CPU exactly.
- **Guards unchanged:** inner/left, dim key a dense `0..dim_n-1` permutation, all fact keys in
  `[0, dim_n)`, no null keys. Any failure → CPU-lookup branch (untouched). The density gate
  (`1e7` FLOPs / `1e5` rows, `force_fusion` override) is unchanged.

### Note on output vs gather sets

The HStack output keeps **all** non-key dim columns (the join produces them); the **chain** reads
only the gather set. M11 produces the chain's inputs via GPU `Take` and the *output* dim columns
via `reordered[col][key]` on CPU. When a dim column is in the output but not read by the chain,
only the CPU index runs for it (no GPU `Take`) — correct and cheap.

## 2. `.select` detection

The `.metal` verbs detect via (a) a `with_columns` capture monkey-patch and (b) a serialize-detect
slow path that matches the HStack `"exprs":[` plan key. `.select` serializes as a `Select` node
with an `"expr":[` key and is missed (confirmed during M10 Task 4.2).

- Add a **`select` capture hook** mirroring `install_with_columns_capture`
  (`python/polars_metal/_detect_common.py`), recording `select` expressions in a parallel cache.
- Extend `iter_candidate_nodes` (the slow path) to also yield candidates from the `Select`
  `"expr"` shape, so a verb used in `.select` is recognized even without the fast-path capture.
- Wire both into the existing detectors (rolling/vector/fft/dtw/dt/corr) — they consume
  `iter_candidate_nodes` / the cache, so the change is centralized in `_detect_common`.

Scope guard: a single `.select`/`.with_columns` layer mixing two different verb sentinels remains
unsupported (M6 limitation, unchanged).

## 3. Retrieval flagship

A two-phase pipeline, both phases on GPU, the boundary a cheap M9 crossing:

```python
hits = queries.with_columns(hit=pl.col("emb").metal.cosine_topk(corpus, k)).collect(engine=metal)  # GPU top-k
fact = hits.explode over hit.indices / hit.scores                                                   # eager, cheap
out  = (fact.lazy()
        .join(metadata, left_on="idx", right_on="id", how="left")
        .with_columns(rerank = f(score, metadata_col_a, metadata_col_b, ...))                       # F32 chain
        .collect(engine=metal))                                                                     # ← resident multi-col gather
```

Added to `tests/bench/m8_report/registry.py` as a report-only case (no failing perf gate),
reporting engine-vs-Polars-CPU + raw-MLX/numpy ceiling + tax, medians. Sized so the rerank clears
the gather gate (≥ ~500k exploded rows). Expectation: resident multi-col gather in the 3–7× range
on the rerank phase; top-k phase at the M6 vector rate.

## Testing

- **Differential vs CPU (the gate):** extend M10's join-gather suite to **multiple dim value
  columns** — 2–3 dim columns, chain reading all/some, across {dense, sparse, nulls, missing keys,
  dup keys}, byte-exact. Output column order/dtypes/null positions exact.
- **`.select` parity:** each `.metal` verb under `.select` matches the same verb under
  `.with_columns` and CPU (rolling, cosine_topk, fft, dtw, dt, corr).
- **Flagship correctness:** the two-phase retrieval byte-exact vs an all-CPU run.
- **Guardrails:** no GPU hash join; F64 multi-col chain → CPU; `make gate` (now including
  `tests/python_integration` via the M10 `test-integration` target) green; no new conformance
  failures.

## Internal phasing (one PR)

1. **Multi-column analyzer splice + dispatch** (the core; extends M10's gather pass + `_try_resident_gather`).
2. **`_walk_join` N-column relaxation** + multi-col differential suite.
3. **`.select` detection** in `_detect_common` + per-verb parity tests.
4. **Retrieval flagship** registry case + medians.

## Open questions

- Does any dim value column dtype other than F32 appear in a realistic metadata rerank (e.g. an
  Int code used in the chain)? MVP: F32 dim value columns only; non-F32 dim columns in the output
  (not read by the chain) are fine via the CPU index, but a non-F32 column *read by the chain*
  forces CPU fallback. Confirm during implementation.
- `.select` fast-path capture: is a `select` monkey-patch sufficient, or do some verbs only flow
  through the slow path? Pin during implementation.
