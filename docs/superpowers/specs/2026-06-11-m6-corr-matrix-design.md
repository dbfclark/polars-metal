# M6 — `.metal.corr()` GPU correlation matrix (design)

Date: 2026-06-11
Branch: `m6-vector-search` (Track A follow-on)
Status: design approved, plan pending

## Summary

Add a GPU-accelerated Pearson **correlation matrix** to the `.metal` namespace as a
LazyFrame verb: `lf.metal.corr()`. The correlation matrix of an `N×p` F32 matrix is
`standardize columns → C = Zᵀ Z / (N−1)`, a GEMM — exactly the "fuse the whole compute
subtree into one MLX subgraph" shape (Mission → Architectural principle 1). This is
roadmap item 10's correlation-matrix piece (survey 7.8×), previously deferred as
"eager + opaque."

## Motivation / data

`scripts/spike_corr_crossover.py` (committed alongside this spec) measured GPU
(MLX standardize + GEMM + eval + readback) vs Polars CPU `df.corr()`. The compute/ingest
ratio scales with column count `p`, so the crossover is a function of `p`:

| regime              | result                                            |
|---------------------|---------------------------------------------------|
| p = 2 (single pair) | CPU wins / tie — bandwidth-bound (the B4 loser)   |
| p = 5               | mixed (CPU at 10K rows, GPU at ≥100K)             |
| p ≥ 10              | GPU wins everywhere, **5–20×**                    |
| sweet spot p≈25–50  | up to **20× at N=1M**                             |
| p ≥ 100             | win shrinks (p×p GEMM grows quadratically) but 3–7×|

Full table is in the spike header. This confirms the multi-column correlation matrix is a
genuine GPU win and matches/exceeds the survey's 7.8×. It is the **inverse of B4**: B4
refuted bare single-column reductions (bandwidth-bound); corr at p≥~8 is compute-bound.

## Surface / API

```python
df.lazy().select("a", "b", "c").metal.corr().collect(engine="metal")   # → 3×3 F32 matrix
```

- New `pl.api.register_lazyframe_namespace("metal")` → `MetalLazyNamespace`.
- `MetalLazyNamespace.corr(force_gpu: bool = False) -> pl.LazyFrame`.
- Correlates **all columns of the frame** it is called on (mirrors eager `df.corr()`; the
  user narrows columns with an upstream `.select(...)`).
- Returns a LazyFrame carrying a **corr sentinel**.
- The existing Expr-level `.metal` namespace (`cosine_topk`, `knn`, `fft`, `dtw`) is
  unchanged; this adds a LazyFrame flavor of the same `"metal"` name.

Rationale for the LazyFrame surface (vs an eager `polars_metal.corr(df)` helper or an
asymmetric capture-arg Expr verb): keeps the engine-plugin philosophy (only works under
`collect(engine="metal")`, raises otherwise), reuses serialize-detect + collect-and-stitch,
and reads naturally for a symmetric all-columns operation. `df.corr()` is eager and never
reaches our engine, so it cannot be hooked the way the existing ops are — the LazyFrame
verb is the bridge.

## Detection & dispatch

1. At `.corr()` call time, read the frame schema (`collect_schema()`) → column names,
   count `p`, dtypes. Build a one-column projection holding a **corr sentinel** encoding
   `{op: "corr", columns: [...], force_gpu: bool}`.
2. At `collect(engine="metal")`, the collect-hook serializes the plan; `_corr_detect`
   recognizes the sentinel (same serialize-detect path as `dtw`/`fft`, **not** the
   NodeTraverser walker — corr is NodeTraverser-opaque).
3. The hook collects the **source** `N×p` columns on CPU (a cheap projection), and hands
   the F32 matrix to `execute_corr`.
4. **Frame-replacing stitch (the one genuinely new dispatch primitive).** Unlike the
   existing verbs, which stitch a result column back into an `N`-row frame, corr collapses
   `N` rows → `p` rows. The hook therefore *replaces the entire frame* with the `p×p`
   result rather than inserting a column. This is the only net-new plumbing.
5. Collected on plain CPU (no metal engine), the sentinel **raises**, consistent with the
   other `.metal` verbs.

## Kernel — `execute_corr(X: N×p F32) → p×p F32`

MLX op-graph, **one `mx.eval`** (no custom MSL — MLX's matmul is Apple-tuned and the spike
proves it wins; CLAUDE.md: wrap MLX where it is already fast):

```
mu  = mean(X, axis=0, keepdims)
Xc  = X - mu
var = sum(Xc * Xc, axis=0, keepdims) / (N - 1)
Z   = Xc / sqrt(var)
C   = matmul(Z.T, Z) / (N - 1)
eval(C); contiguous; readback → p×p F32
```

- Reuses the A2 vector-search FFI wrappers (matmul + reductions already bound).
- `contiguous()` before raw readback (the A2 transpose/slice readback gotcha).
- Zero-variance (constant) column → std 0 → NaN row/col, which **matches** Polars'
  constant-column correlation output.
- A custom `shaders/corr.metal` GEMM (fusing standardize into the GEMM to avoid
  materializing `Z`, possibly recovering the p≈100 efficiency dip) is an explicit
  **non-goal for v1** — premature per "profile first." Revisit only if a profile shows
  MLX leaving wins on the table; if so, gate it with the same differential test.

## Routing guard

- Named constant `CORR_P_MIN = 8` (spike: p≥10 solid GPU, p=5 mixed → 8 is the
  conservative crossover).
- The hook selects the path:
  - `p ≥ CORR_P_MIN` **or** `force_gpu=True` → GPU subgraph.
  - else → **CPU fallback**: Polars `df.corr()` on the collected frame, cast to F32.
- `force_gpu=True` always dispatches to GPU regardless of `p`.

## Output contract

- `p×p` **Float32** DataFrame.
- Column names = input column names; row order = column order (matches `df.corr()`'s
  shape — no separate row-label column).
- **Documented divergence from Polars CPU:** dtype is F32, not Polars' F64; values are
  F32-precision (≈1e-5 tolerance), not byte-equal. Same divergence class as the M3
  "Mean F32 returns F32 not F64." Add to the conformance/divergence ledger.
- Output dtype is **path-independent**: the CPU-fallback path also casts to F32, so the
  result dtype does not depend on which branch the guard took.

## Dtype / null handling

- **Pearson only.** Spearman is out of scope for this verb (it needs GPU ranking first);
  not implemented, not silently wrong — outside the verb's contract.
- Numeric inputs (`Int*`, `F64`) are cast to F32 on ingest (lossy, consistent with the
  F32-kernel philosophy). A non-numeric column (string/bool/etc.) → clear raise.
- **Nulls → CPU fallback** (result cast to F32). The standardize+GEMM cannot replicate
  Polars' pairwise-null correlation semantics (a single NaN poisons the whole matrix);
  deferring null-bearing inputs to the CPU oracle keeps null semantics exact at the cost
  of running on CPU for null data. Documented.

## Testing

- **Kernel** (`tests/kernel/`): `execute_corr` vs numpy `corrcoef` / `df.corr()` at F32
  tolerance — random `N×p`; degenerate `p=1` (1×1 = 1.0); `p=2`; a constant column (NaN);
  wide `p`.
- **Differential** (`tests/`): engine `.metal.corr()` vs `df.corr()` cast F32 — random
  frames; null-bearing inputs (exercises the fallback); `force_gpu` on/off; `p` below and
  above `CORR_P_MIN`.
- **Bench** (`tests/bench/`): `bench_corr` sweeping `p`, with a `_gate.ratio_lt` floor
  derived from the spike numbers to catch dispatch-cliff regressions (bench ≠ test).
- No `shaders/` file is added (MLX-only), so no new kernel-shader test is required.

## Out of scope (v1)

- Spearman correlation.
- Custom `shaders/corr.metal` GEMM kernel.
- GPU pairwise-null correlation semantics.
- Covariance matrix / other reductions over the standardized matrix (corr only).
- F64-precision output.
