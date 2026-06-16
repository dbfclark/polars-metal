# M10 — Resident gather via join recognition (design)

> Status: approved design, pre-plan. The first M9→engine translation: M9 proved the
> resident gather→compute prize in raw MLX; M10 makes `engine="metal"` capture it on a
> natural Polars expression shape. One PR.

## Background

M9's crossing-tax benchmark (`docs/crossing-tax-report.md`, decision **(b)**) established:

- Crossings are cheap: α ≈ 4.96e-8 ms/byte (≈ 20.2 GB/s round-trip), β ≈ 0.0016 ms/crossing.
- The **only** pipeline family with a GPU win is **gather → compute** (P1 retrieve→rerank
  22–30×; P2 fact→dim chain up to 8.2× at 10M). The join families (P3 as-of, P4 hash) tie
  all-CPU at ~1.0× under every crossing strategy.
- On the gather family the **resident** strategy (fold the gather — `mx.take` / dense index —
  into the resident GPU subtree, cross only the reduced result) is never worse than per-op
  routing and up to ~2× better, because it kills the cache-hostile CPU scatter.
- The cost-model rule: route to GPU iff `GPU_compute_saved > α·bytes_crossed + β·n_crossings`.
  Two corollaries: (1) cross *reduced* data, never the full intermediate; (2) you need real
  compute density to clear the ~2 ms/full-column tax — a lone `sqrt` (P3/P4) never does.

M9 explicitly ruled out a **general per-op CPU↔GPU router** (outcome (a)) and **GPU joins**:
the join boundary is where the win dies, not where a router rescues it.

## What changed during brainstorming (recorded so we don't re-derive)

The original M10 brief targeted an **explicit `Expr.gather(idx)`** feeding an F32 chain. A
recognition spike (`tests/bench/m9_crossing/_m10_recognition_probe.py`) confirmed the
post-optimization `NodeTraverser` **does** expose a `Gather` node (attrs `.expr` source,
`.idx` index, `.scalar`) — unlike the M4 opacity wall (list/array/`corr`/`rolling` all raise) —
so the explicit-gather form is walker-recognizable.

**But the architect's steer:** requiring the user to write `dim.gather(id)` instead of the
`fact.join(dim, on="id")` they would naturally reach for is exactly the query-restructuring the
M8 vision says to avoid ("blazing fast in general without making the user structure queries").
`Expr.gather` is standard Polars (not a new surface), but the *natural* idiom for a per-key
lookup is a **join**. And a lookup only arises when the dim is a **separate table** — if the
per-key value is already a column, there is no gather and the existing fused walker already
accelerates the chain. Therefore: **P2 recognizes the `join → F32 chain` shape, not an explicit
gather.**

### The integration constraint that forces model 1

Two integration models exist in the codebase:

1. **Walker + `set_udf` (model 1).** The walker (`python/polars_metal/_walker.py` → `_callback.py`)
   walks the IR; on a fully-`Handled` plan it installs a UDF via `nt.set_udf(...)`. That UDF is a
   **`PythonScanSource::Cuda`** scan-source replacement (signature
   `(with_columns, predicate, n_rows, should_time)` — **no input-DataFrame parameter**). It is
   installed at the **root** and *produces* the whole (sub)plan's output from embedded scan dfs.
   The walker is **all-or-nothing**: `_walk_select` (line ~318–321) propagates any child
   `FallBack` to the root, and `execute_with_metal` (line ~42) sends the whole query to CPU on any
   `FallBack`. So today a `Join` anywhere below a compute chain poisons the entire plan → all-CPU.

2. **collect-and-stitch (model 2).** The `.metal` verbs (rolling/dt/vector/fft/dtw/corr) intercept
   at `collect`, run the non-recognized remainder on CPU via `_collect_rest` (a join just lands in
   the CPU remainder — model 2 *splits at joins natively*), GPU the recognized op, and stitch.

**Model 2 can never keep the gather resident** — it forces the lookup onto CPU, yielding at best
partial_smart (2–4×), never the resident 3–8×. To capture the resident win, the GPU node must own
the gather, which means the GPU UDF must root **at or above** the join and produce its output
itself. Because the scan-source UDF cannot *consume* a CPU node's output, the only way to put GPU
compute above a join is to make the UDF **replace the whole plan** (join included) and do the join
equivalent itself. Hence **model 1**, extended to recognize `Join`.

This is **not** a GPU hash join (the rejected non-goal): for a dense integer key the join *is* a
gather (no hash table); for a non-dense key the UDF performs the lookup on CPU via Polars and only
the F32 chain runs on GPU.

## Goal

Make `engine="metal"` capture the M9 resident gather→compute prize on the natural Polars idioms:

- **P2:** `fact.join(dim, on=<int key>) → <F32 compute chain>` — accelerated with no query
  restructuring and no new user API.
- **P1:** retrieve→rerank — `.metal.cosine_topk(corpus, k, rerank_weight=…)` folds a resident
  post-top-k gather + combine on GPU.

Honest baselines only: `engine="metal"` vs **Polars CPU** (mission bar) plus the raw-MLX/numpy
ceiling; report medians.

## Non-goals / guardrails (from the M9 verdict + standing non-goals)

- **No general per-op CPU↔GPU router** (M9 outcome (a) — rejected).
- **No GPU hash joins.** The dense path is a *gather*; the non-dense path does the lookup on CPU.
  We never build a GPU hash table.
- **F64 stays CPU** (no GPU FP64). Chains must be F32-fusable or we fall back.
- **Do not extend hash-groupby / sort past conformance.**
- No string/list/struct kernels.

## P2 — recognition & integration

### Plan shape (verified by the spike)

```
HStack / Select(chain)     ← F32 compute chain; walker-viewable (BinaryExpr, Function, …)
  Join(on <int key>, how)  ← IR node viewable: left_on / right_on (Column), how
    Scan(fact)             ← walker-handleable DataFrameScan
    Scan(dim)              ← walker-handleable DataFrameScan  (dim is SMALL)
```

### `_walk_join`

Add a `Join` arm to `_walk_at_current`. Return `Handled` only when **all** hold (else `FallBack`):

- equi-join on a **single** key column, integer dtype on both sides;
- `how ∈ {inner, left}`;
- both inputs walk to `Handled` `Scan` nodes (the dim and fact frames are materializable);
- the parent compute (Select/HStack) is an **F32-fusable chain** the existing analyzer accepts,
  whose leaves resolve against the **post-join** schema (fact columns + dim value columns).

The resulting `Handled` plan carries a new node kind (`Join`) with: key column name(s), `how`,
left/right scan plans, and the dim value column names the chain consumes. The fused-chain scope
(from the existing M4 analyzer) is attached as a side-channel on the parent, exactly as HStack
fusion does today (`_fused_scope`).

### Whole-plan scan-source UDF (two-scan capture)

`build_udf` today captures **one** scan df and reconstructs a single-input subtree. M10 adds a
`Join` plan kind whose UDF:

1. captures **both** embedded scan dfs (fact, dim) at walk time;
2. produces the looked-up dim column(s) (§ Execution);
3. assembles input buffers `{fact columns, looked-up dim column(s)}` and runs the **existing fused
   MLX subgraph** over them — one `mx.eval`, one fold-back;
4. returns the resulting DataFrame (in output-schema order).

`set_udf` is installed at the root, so Polars executes **only** our UDF — it never runs the join.

## P2 — execution: one UDF, dense / non-dense branch

Let `key = fact[keycol]` (length N), and the dim frame (small, `dim_n` rows) provide
`dim[keycol]` and the value column(s) the chain reads.

### Dense-key resident gather (3–8×)

When the **guards** below hold, the join is a pure gather:

- reorder dim value columns so position = key (dim already keyed `0..dim_n-1` ⇒ identity);
- on GPU: `vol = mx.take(dim_value, key)` (resident), feed the F32 chain, eval once, fold back.

`mx.take` needs a 1-D gather wrapper. M6 shipped `mlx_take_along_axis` (2-D, axis-aware) in
`crates/polars-metal-mlx-sys/src/shape.rs`; add a thin `mlx_take(a, indices)` (1-D) — or reshape
`(n,1)`/`(k,1)` through the existing axis-aware call. The resident subgraph mirrors
`vector_search.rs` (view buffers → build graph → single eval → readback).

### Non-dense-key CPU lookup (2–4×, partial_smart)

When the key is not a dense range (but guards otherwise allow correctness via Polars semantics):

- perform the lookup on CPU via Polars (`fact.join(dim, …)` on the **small** dim — correct join
  semantics for free), producing the looked-up dim column;
- run the F32 chain **resident on GPU** over `{fact columns, looked-up column}`, one fold-back.

The expensive part (the N-row chain) is still GPU; only the small lookup is CPU. This is the M9
partial_smart shape and clears the tax whenever the chain is dense.

### Correctness guards (the cardinal-rule surface)

A dense-key gather equals the join **only** under all of:

- `how ∈ {inner, left}`;
- dim keys **unique** (no one-to-many row explosion);
- key column is a **dense `0..dim_n-1` range** (the gather-position test);
- **no null keys** on either side within the matched set.

Failing the dense test → **non-dense CPU-lookup branch**. Failing a semantic guard the CPU branch
also can't satisfy cheaply → **full CPU fallback**. Output **row count, null positions, dtypes, and
(where defined) row order must match CPU exactly** — differentially tested. For a left join, missing
keys must yield nulls in the looked-up column with the chain evaluating to the same null-propagated
result as CPU.

> Implementation risk to spike first: confirm the two-scan scan-source UDF produces a correct
> end-to-end result on the **non-dense / CPU-lookup** branch before building the dense path on top.
> If two-scan capture or root `set_udf` over a Join misbehaves, the whole model-1 approach is wrong
> and we stop and reconsider (rather than papering over with collect-and-stitch, which can't be
> resident).

## P1 — resident vector rerank

Extend the existing `.metal.cosine_topk(corpus, k)` (expr namespace over `Array(Float32, D)`):

- new optional `rerank_weight`: a per-corpus-row F32 vector, length = corpus rows;
- new optional `rerank`: a fixed, documented form. v1 ships **`"exp_decay"`** →
  `reranked = score * exp(-weight[hit])` (M9 P1's rerank). Default `None` = current behavior.

`vector_search.rs` already produces `idx_k` (top-k indices) and `val_k` (top-k similarities) as a
resident subgraph (matmul → argpartition → `take_along_axis`). Add, resident:
`feat = take(weight, idx_k)`; `reranked = val_k * exp(-feat)`; return `reranked` in place of raw
similarities. The rerank gather is **engine-internal** — the user never writes a gather. Result
struct shape unchanged (`Struct{indices, scores}`); `scores` now carries reranked values when
`rerank` is set. F32-only; non-F32 / length-mismatch `weight` → raise.

## Routing guard + force override

Reuse `crates/polars-metal-core/src/fusion/density.rs::density_routes_gpu` (route iff
`est_flops ≥ 5e7` **and** `n_rows ≥ 1e5`):

- feed the chain's **output length** (N fact rows) as `n_rows`;
- the gather/lookup contributes ~0 FLOPs; the post-lookup chain supplies the density. A bare or
  cheap lookup (gather + `sqrt`) therefore correctly stays CPU (the P3/P4 trap).

**Force-route override.** No routing override exists today. Add a `MetalEngine` field
`force_fusion: bool = False` (default off), plumbed to the density decision so an above-threshold
override forces GPU for a recognized fused/gather subtree regardless of the FLOPs/rows gate.
Intended for benchmarking and power users; the default path stays honest.

## Testing strategy

### Differential (the gate — correctness regressions are bugs)

P2, byte-exact vs CPU (`collect()` vs `collect(engine="metal")`):

- dense key vs non-dense key;
- `how` = inner and left;
- null keys (left and right), duplicate dim keys, missing keys (left-join nulls), empty dim,
  single-row, key dtypes (i8/i16/i32/i64/u8/u16/u32 where index-valid);
- F32 chain present vs absent; non-F32 / F64 chain → CPU fallback (assert dispatch count == CPU);
- below-threshold N and below-FLOPs chain → CPU (assert no GPU dispatch), and with
  `force_fusion=True` → GPU.

P1 rerank vs numpy (`score * exp(-weight[hit])`), rtol/atol 1e-3; weight length mismatch raises;
`rerank=None` unchanged from current cosine_topk.

### Honest perf (regenerable, not a gate)

Add P2 (dense + non-dense) and P1-rerank cases to the **M8 perf-report registry**
(`tests/bench/m8_report/`), reporting both columns (engine vs Polars CPU; engine vs raw-MLX
ceiling) + the tax column, medians. Targets are honest expectations, not thresholds: dense ~3–8×,
non-dense ~2–4×, P1 rerank ~20–30×. Losers (below-threshold) reported, not hidden.

### Guardrails verification

No GPU hash join (dense = gather; non-dense = CPU lookup); F64 → CPU; groupby/sort untouched;
`make gate` green (clippy/fmt/ruff + unit/kernel/conformance).

## Internal phasing (one PR)

1. **Two-scan UDF + `_walk_join` recognition → CPU-lookup branch + resident chain.** Proves the
   model-1 integration end-to-end (the spike). Non-dense path only.
2. **Dense-key resident gather** (`mlx_take` 1-D + dense-key detection + resident take subgraph).
3. **Correctness guards + full differential suite** (the bulk of the risk).
4. **P1 vector rerank** (`rerank_weight` / `rerank` + resident take in `vector_search.rs`).
5. **Force-route override + perf-report wiring** (M8 registry cases, medians).

## Open questions (track in `docs/open-questions.md` if they survive implementation)

- Does the Polars optimizer ever reorder the chain below/around the join (projection pushdown)
  such that the `HStack(chain) → Join` shape isn't stable? If so, recognition must handle the
  pushed-down variant or fall back cleanly.
- Dense-key detection cost: is the `0..dim_n-1` range check on the (small) dim cheap enough to run
  unconditionally, or does it need its own size guard?
- Should `rerank` grow beyond `exp_decay` (e.g. a small enum of forms) in a later milestone, or
  stay single-form until a real workload asks?
