# M6 Track B — Integer & Temporal Support Design

**Status:** Approved (brainstorm 2026-06-09). Umbrella spec for Track B of M6
([[m6-scope-and-api-direction]]); implementation decomposes into four staged plans
(B1→B2→B3→B4), each its own plan→implement cycle under this one spec.

**Goal:** Extend the Metal execution engine — currently **F32-first** — to **all integer dtypes**
(`Int8/16/32/64`, `UInt8/16/32/64`); ship a **gregorian `dt` MSL kernel** (`dt.year/month/day`) as
the flagship compute-bound
consumer; and — a finding surfaced during this brainstorm — **re-tune bare reduction routing for
both int AND F32**, where GPU reductions are a measured 3.5–10× win at scale.

**Parent spec:** `docs/superpowers/specs/2026-06-04-m6-metal-namespace-design.md` (Track B was a
thin stub there: B1 buffer/bridge, B2 fused-walker int parity, B3 dt kernel, B4 benchmarks).

**Sequencing note:** this restores the parent spec's original order (integer/`dt` *before*
pairwise). A4 pairwise was briefly considered first this session but deferred: Levenshtein needs the
byte/Int plumbing Track B builds, so integers come first.

---

## Why this design (decision record)

**The type system is already integer-aware; the runtime just never consults it.** A survey found
`InputDtype::I32`, `MlxDtype::I32`, `DtypeReq::Numeric` (covers I32 for add/sub/mul/cmp), and
`mlx_array_to_i32_vec` all already exist. The F32 coupling is in *runtime* hard-codes, not
architecture: the subgraph builder pins `MlxDtype::F32` (`subgraph.rs:170/184/227`), Python dispatch
force-casts every column to `np.float32` (`_udf.py:170/181`), and reductions hard-output
`ScalarF32`. So Track B is largely *"make the runtime read the dtype that's already declared"* +
add the I64 leg + the `dt` kernel.

**Empirical spikes resolved two forks that would otherwise have been guessed** (per the
"spike unknowns during brainstorm" discipline — Python MLX 0.25.1 is importable in-repo and matches
the vendored build):

1. **All eight integer dtypes work on MLX Metal.** Live spikes: `Int8/16/32/64` AND
   `UInt8/16/32/64` elementwise (add/sub/mul/floordiv/mod), comparison→bool, reductions
   (sum/min/max), and cast→f32 all execute correctly on `mx.gpu`, including genuine 64-bit values
   (`3e9+2e9=6e9`) and `UInt64` beyond I64 range (`1e19+5` exact — the case widening *can't* handle).
   The M2 "no 64-bit atomics" limit doesn't bite (reductions are tree-based, not atomic). → **all
   integer dtypes first-class, no widening, no remaining unknown.** One semantic wrinkle: MLX keeps
   the reduction accumulator's width by default, but Polars upcasts integer `sum` to 64-bit
   (`sum(Int32)→Int64`, `sum(UInt32)→UInt64`) — we cast on fold-back to match.

2. **Bare reductions are a real GPU win — and not just for int.** Spike (pure-compute, resident
   data, Polars CPU vs MLX GPU):

   | size | op | Polars | MLX GPU | speedup |
   |---|---|---|---|---|
   | 10M Int64 | sum/mean/max | 1.3–1.8 ms | 0.36–0.38 ms | **3.5–4.9×** |
   | 100M Int64 | sum/mean/max | 14–17 ms | 1.4–1.7 ms | **10×** |
   | 10M F32 | sum | 0.66 ms | ~0.3 ms (noisy) | ~wash–2× |
   | 10M F32 | mean/min/max | ~0.83 ms | ~0.27 ms | **~3×** |
   | 100M F32 | sum/mean/min/max | 7–9 ms | ~1 ms | **6–10×** |

   This **contradicts the working assumption** ("bandwidth-bound bare reductions → CPU"). GPU
   reductions decisively beat Polars CPU at scale; the win is size-dependent (F32 `sum` at 10M is a
   wash — Polars SIMD is near-bandwidth there — but everything wins at 100M, and int wins even at
   10M). **Caveat:** these are pure-compute numbers; the *engine* path (PyO3 + collect-and-fold +
   zero-copy staging + scalar readback) carries overhead M4's original CPU-pinning measured. The
   *compute* unlock is proven; the *engine* win must be confirmed in-engine (B4's job). The
   staged-with-copy numbers (GPU loses at 10M) are a Python-MLX artifact — our engine stages via
   `from_borrowed_*` (zero-copy on unified memory), so it pays the resident cost, not the copy.

**Architect-approved decisions:**
- **Full Track B** (B1+B2+B3+B4), one umbrella spec, four staged plans.
- **All eight integer dtypes first-class** — `Int8/16/32/64`, `UInt8/16/32/64` — mapped natively to
  MLX dtypes (spike-verified, incl. `UInt64` beyond I64 range). No widening, no unsigned fallback;
  covering them all is uniform (`dtype → MlxDtype`) and is the only correct path for `UInt64`.
- **`dt` = native acceleration** of `pl.col(d).dt.year/month/day` (no new verb) via the
  serialize-detect + collect-and-stitch template (rolling/FFT proven path); `dt.*` is
  NodeTraverser-opaque. Date(I32) + Datetime(I64→days), all time units.
- **Reduction routing (int + F32) elevated to a first-class B4 deliverable** — measure end-to-end,
  install a size-aware per-op threshold, flip routing where GPU genuinely wins. This expands Track
  B's blast radius to touch the F32 reduction path; explicitly endorsed.
- **Honest framing:** the headline perf wins are (a) the `dt` kernel (~30–40×, always routes) and
  (b) the reduction-routing unlock at scale. General int *arithmetic* parity is mostly
  correctness/capability — it lets mixed int↔F32 chains fuse instead of falling back; pure-int
  arithmetic chains rarely clear the FLOPs/row bar. The spec says so rather than implying an
  int-arithmetic speedup that isn't there.

---

## Architecture & decomposition

```
B1  Int buffer/FFI/subgraph dtype-awareness  ──┐  (foundation; self-contained)
B2  Fused-walker int parity  ──────────────────┤  (depends on B1)
B3  dt gregorian MSL kernel  ◀── FLAGSHIP ──────┤  (depends on B1; orthogonal to B2)
B4  Reduction routing + re-baselined benchmarks ┘  (depends on B1–B3; HEADLINE)
```

Each lettered item is its own plan→implement cycle. The load-bearing seam is B1: once the subgraph
builder reads the declared `InputDtype` instead of hard-coding F32, B1 is self-contained and B2/B3
build on it.

### B1 — Integer buffer/FFI/subgraph dtype-awareness

- **`InputDtype`** (`fusion/scope.rs`) + **`MlxDtype`** (`mlx-sys/array.rs`): add variants for all
  integer widths — `I8/I16/I64` and `U8/U16/U32/U64` (I32 already present) — each with its
  `element_size`. The variants are mechanical (mirror the existing pattern).
- **Buffer crate:** add `from_<t>_slice` / `to_<t>_vec` / `from_borrowed_<t>` per width (the
  Arrow↔MTL bridge and validity bitmap are already byte-agnostic, so each is a thin convenience
  wrapper; consider a generic-over-width helper to avoid eight near-copies). Zero-copy where
  alignment permits, mirroring `from_borrowed_f32`.
- **MLX FFI:** add `mlx_array_from_<t>_slice` + `mlx_array_to_<t>_vec` readback per width (the I32
  readback already exists from vector search). `mlx_array_view_metal_buffer` already takes a
  `MlxDtype` → the view path is ready for every width.
- **`subgraph.rs`** (the hard-codes at `:170/:184/:227`): map `InputDtype → MlxDtype` when wrapping
  each buffer; derive element count from `dtype.element_size()` (not `/4`); dispatch readback on the
  *output* dtype (`to_i32_vec` / `to_i64_vec` / `to_f32_vec`).
- **Python dispatch** (`_udf.py:170/181`): pass each column as its **native dtype** bytes
  (i32/i64/f32), not force-cast to `np.float32`; pre-allocate the output by the analyzer-inferred
  dtype. Analyzer (`_fusion_analyzer.py`): map every Polars integer dtype to its `InputDtype`
  (`Int8/16/64`, `UInt8/16/32/64`; I32 already mapped).
- **Exit bar:** an Int32 *and* an Int64 column round-trip through a trivial fused chain (`col + 1`)
  end-to-end, byte-exact vs Polars CPU, nulls preserved.

### B2 — Fused-walker integer parity

- **Op coverage:** arithmetic (`add/sub/mul/floordiv/mod/neg/abs`), comparison
  (`eq/ne/lt/le/gt/ge`→Bool), and **cast** (int↔int, int↔f32) — the ops the spike verified on Metal
  for both widths. Transcendentals/`pow`/rounding stay F32-only (an int feeding them casts to f32
  first, which already works).
- **Reduction dtype semantics — matched to Polars exactly, all GPU-capable as chain terminators:**
  - `sum` upcasts to 64-bit per Polars: signed → **Int64**, unsigned → **UInt64** (MLX keeps the
    input width → cast on fold-back; the differential test pins the exact per-dtype result).
  - `min/max(<int>)`→ same dtype (preserve).
  - `mean(IntN)`→**F32** (MLX upcasts int→f32 and divides). Polars returns Float64; we return F32 —
    the same documented divergence as the existing "Mean F32 returns F32 not F64" baseline. The
    int-mean path exists and is used when `mean` terminates a fused compute-bound chain (bare mean
    routing decided in B4).
- **Null semantics:** match Polars exactly via the existing (dtype-agnostic) validity bitmap
  (`null + x = null`). Differential tests include null-heavy int inputs.
- **Routing:** the FLOPs/row router decides fused-chain routing as today. Honest reach: mixed
  int↔f32 chains and dt are the realistic GPU consumers; pure-int arithmetic chains rarely clear the
  bar. (Bare *reduction* routing — int and F32 — is B4.)

### B3 — `dt` gregorian MSL kernel (flagship)

- **Recognition:** `dt.year/month/day` are NodeTraverser-opaque, so detect them via the
  pre-optimization `lf.serialize` plan (the rolling/FFT template) + collect-and-stitch: drop the dt
  output columns, CPU-collect the rest (projection pushdown elides the dt fields), run the GPU kernel
  per binding, stitch results back in schema order. No new user API — native
  `pl.col(d).dt.year()` just gets faster under `engine="metal"`.
- **Kernel** (`shaders/dt_gregorian.metal`, + a `tests/kernel/` test per the shaders convention):
  branchless civil-from-days (Howard Hinnant's algorithm — the settled approach), with a field
  selector (`0=year,1=month,2=day`) or three entry points. Input: I32 days-since-1970. Document
  threadgroup/grid assumptions at the top of the file.
- **Input prep by dtype:**
  - **`Date`** (I32 days): fed directly.
  - **`Datetime`** (I64, `time_unit` ms/us/ns): `days = floor_div(value, units_per_day)` (handle
    negatives), then the same kernel. `time_unit` from the column dtype, passed as a scalar.
- **Output dtypes — matched to Polars exactly:** `dt.year`→**Int32**, `dt.month`/`dt.day`→**Int8**.
  The kernel computes Int32; month/day are narrowed to Int8 on fold-back (host-side) so the stitched
  column dtype matches Polars.
- **Correctness:** exact match vs Polars CPU `dt.*` across the epoch, leap/century rules
  (2000 leap, 1900 not), **pre-1970 negative days**, year/month boundaries, all three Datetime time
  units, and null-bearing columns.
- **Perf target:** ~30–40× vs Polars CPU at 10M rows; always routes (compute-bound consumer, no
  FLOPs/row gating).

### B4 — Reduction routing + re-baselined benchmarks (HEADLINE)

- **Outcome — premise refuted by end-to-end measurement (2026-06-10, M2 Ultra).** The pre-plan
  spike drove bare int (I32/I64) and F32 reductions through the *full* engine path (collect +
  zero-copy-view stage + MLX reduce + scalar readback) vs Polars CPU across 1M/10M/100M. **GPU
  loses 2–5× at every size with no crossover.** A bare reduction is bandwidth-bound (1 flop/element);
  the host→MLX ingest alone (≥7ms @100M, the unified-memory wall) exceeds Polars' multithreaded SIMD
  scan (2.4ms). The brainstorm spike's 3.5–10× was an artifact of comparing *resident* MLX (no ingest,
  0.95ms) against *single-threaded* numpy (~17ms), not real Polars CPU. **`std`/`var` stay on GPU** —
  they are genuine 5–9× wins (Polars CPU std/var is a slow two-pass Welford, far from bandwidth). The
  M4 compute-intensity gate (route on FLOPs/row, not op identity) is vindicated, not changed.
- **No routing flip, no `N₀` threshold.** `_BARE_GPU_WORTHY_REDUCTIONS = {std, var}` is correct as-is;
  there is no size crossover to gate on. `StagingPool` is moot here — the reduction input is already a
  zero-copy MLX view (no per-call `newBufferWithBytes` tax, unlike `execute_dt`).
- **Shipped as guard + re-baseline:** a dispatch-asserted regression test pinning the decision
  (`tests/python_integration/test_reduction_routing.py`), a permanent end-to-end benchmark
  (`tests/bench/m4_survey/bench_reductions.py`) with honest `baseline.json` gates (std/var gated,
  bare reductions informational). Full data: memory `m6-b4-reduction-routing-spike`.

---

## Testing strategy

Differential vs **Polars CPU (the oracle), byte-exact** unless a documented F32 divergence applies:
- **Int arith/cmp/cast:** the full dtype matrix — `Int8/16/32/64`, `UInt8/16/32/64` — including
  per-dtype overflow/wraparound domain (e.g. `Int8` wrap mod 256, `UInt64` beyond I64 range),
  null-heavy inputs, and mixed int↔f32 chains.
- **Reductions:** `sum`→Int64, `min/max`→IntN, `mean`→F32 (F32 tolerance); correct dtype + value;
  nulls; the size-aware threshold routes correctly (small→CPU, large→GPU) and both paths agree.
- **dt kernel:** year/month/day on Date + Datetime (all 3 time units); epoch, leap/century,
  pre-1970 negatives, boundaries, nulls; exact match incl. output dtypes (Int32 year, Int8
  month/day). Kernel-level test in `tests/kernel/` + engine-level differential.
- **Conformance:** the existing Polars-suite-through-`engine="metal"` gate must stay green; integer
  columns that previously forced CPU fallback now fuse — verify no regression.

All `cargo test` runs use `--test-threads=1` (Metal command-queue contention).

---

## Scope / non-goals (explicit)

- **First-class:** all eight integer dtypes — `Int8/16/32/64`, `UInt8/16/32/64` — natively (no
  widening, no fallback; spike-verified on Metal incl. `UInt64`).
- **Out:** bit-ops (`and/or/xor/shift` — rare in OLAP; revisit on demand); integer groupby-keys /
  hash-join (stays CPU/conformance — standing bandwidth non-goal); `mean`→Float64 precision (we
  return F32 — documented divergence, same as existing F32-mean baseline).
- **No new public API** — `dt` recognized natively via serialize-detect; no `.metal` verb.
- **F32 *arithmetic* routing unchanged** — B4 touches only bare-*reduction* routing; the broader
  "should more F32 paths route to GPU" question stays out of Track B.

---

## Open questions (resolve at plan/drill time)

- **Reduction crossover `N₀`** — fixed per-op constants vs. queried from `MTLDevice` bandwidth. B4
  measures; start with measured constants.
- **Datetime negative-days floor-div** — confirm MSL integer division rounds toward zero and adjust
  for floor semantics on pre-epoch timestamps (test pre-1970 explicitly).
- **Zero-copy staging for int columns** — confirm `from_borrowed_i32/i64` achieves zero-copy at
  Polars' Arrow buffer alignment, or measures the copy cost where it can't (affects B4 thresholds).
- **`dt` recognition robustness** — confirm the serialize-detect path distinguishes
  `dt.year/month/day` from other temporal exprs and carries the `time_unit` (Datetime) reliably; the
  M5 "lf.serialize embeds data" gotcha applies (use expr-capture, not full-plan json).
