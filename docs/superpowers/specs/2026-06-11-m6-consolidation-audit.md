# M6 Consolidation Audit — roadmap status + M7 candidate pool

Date: 2026-06-11
Branch: `m6-vector-search` (PR #6)
Purpose: the M6 spec's **consolidation deliverable** — per roadmap item, mark
shipped / conformance-only / deferred; reconcile CLAUDE.md's roadmap against
what actually shipped; seed M7 planning.

---

## 1. Roadmap status (CLAUDE.md items 1–13)

| # | Item | Status | Note |
|---|------|--------|------|
| 1 | Buffer bridge | **shipped (M0)** | foundation; `MetalBuffer`, page-align regimes |
| 2 | Engine skeleton | **shipped (M1)** | plugin registration + walker |
| 3 | Scan + project + filter | **shipped (M1)** | routed where it can; CPU fallback otherwise |
| 4 | Elementwise + reductions | **shipped (M2/M3)** | folded into the M4 fusion walker |
| 5 | Hash groupby | **conformance-only** | shipped M2; do not extend (bandwidth-bound) |
| 6 | Radix sort (fixed-width) | **conformance-only** | partial M3 |
| 7 | TPC-H Q1/Q6 walker | **conformance-only** | M3; perf non-goal |
| 8 | MLX subgraph fusion (F32 chains) | **shipped (M4)** | haversine 22×, BS 28×, std/var 6–7×, etc. + compute-intensity routing (B4-confirmed) |
| 9 | Rolling windowed stats | **shipped (M5, merged PR #5)** | custom `shaders/rolling.metal`, ~25×, F32-only |
| 10 | List/Array dot + **corr matrix** | **shipped (M6)** — *reframed* | see §2: vector-search use-case shipped as `.metal.cosine_topk/.knn`; corr shipped as `lf.metal.corr()` (~9.9×). Generic `list.eval/arr.sum` *recognition* NOT built (superseded by the namespace). |
| 11 | `Expr.fft()` | **shipped (M6 A3 + memory pass)** | hand-rolled planar MSL FFT, all sizes on-GPU, ~3–4.6× vs numpy |
| 12 | dt gregorian kernel | **shipped (M6 B3)** | `dt.year/month/day`, honest 10–27×, byte-exact |
| 13 | Pairwise distance (Levenshtein, DTW) | **partial** | DTW shipped (M6 A4, 13.4× vs dtaidistance); **Levenshtein deferred** |

**Also shipped in M6 but NOT a numbered roadmap item:**
- **Track B integer parity + reductions** (B1/B2): all 8 int dtypes end-to-end; int sum/min/max on GPU.
- **`StagingPool`** (B3b): reusable page-aligned host→Metal staging; adopted by dt (and the audit confirmed rolling/vector don't need it — already zero-copy).
- **8 pre-existing conformance fixes** (the lazyframe/group_by failures — non-fused HStack handler, the "non-F32 HStack → CPU fallback" invariant, CSE-off gating). See [[m6-conformance-fixes]].
- **Memory/copy pass** (2026-06-11): dt `astype(int8)` 30×, the repeated-collect weakref-cache fix (all 6 verbs), correctness fixes (corr N<2 / streaming / vector null guard), `_detect_common` DRY. See [[m6-memory-pass]].
- **FFT planar (SoA) rewrite**: eliminated the host interleave/split reshuffle (~20% @2^24). See [[m6-fft-planar-rewrite]].

## 2. The reframe that defines M6 (reconciliation)

The M4/M5 roadmap framed items 9–13 as **blocked on a NodeTraverser "opacity unlock"** — a net-new recognition subsystem to let the walker *see* list/array/`corr`/`rolling_*`/`dt.*` nodes. **M6 did NOT build that.** The M6 brainstorm (2026-06-04, [[m6-scope-and-api-direction]]) overrode it: ship the ops via a **`.metal` namespace** (`cosine_topk`, `knn`, `fft`, `dtw`, `corr`) — **an op we own needs no recognition** — detected via `lf.serialize` + collect-and-stitch, not the walker. So:

- "Recognize generic `list.eval(...).list.sum()` / `arr.dot` shapes" (item 10's literal framing) was **not built and is no longer the plan** — the namespace replaces it.
- The general opacity-unlock is **still unbuilt**, and now **largely unnecessary** for the ops we care about (they're namespace verbs or serialize-detected). It only matters if a future goal needs the *walker* to fuse an opaque node mid-tree.

**Honest-perf reconciliation:** the M4 survey numbers (FFT 77×, corr 7.8×, dt 30–40×) were NumPy/raw-MLX comparisons; the **engine-path** numbers are lower and bandwidth-shaped for several ops — dt 10–27×, FFT 3–4.6×, corr ~9.9×, DTW 13.4× (compute-bound, the exception). The roofline finding holds: **native single-column dot/reduction is bandwidth-bound (loses or ties); wins live in batched-GEMM / high-FLOP-per-byte forms** (corr, vector search, DTW).

## 3. CLAUDE.md staleness fixed alongside this audit

- Item 8 said "**Remaining to land M4:** Phase 11 … merge" and the M4 header "landing in progress" — **stale**: M4 landed (M5/M6 build on it). Corrected.
- The M5 section header "**Next (M5) … to be planned**" and M6's framing as "all gated on a recognition mechanism" — **stale**: M5 + M6 shipped via the namespace/serialize-detect path. Corrected to reflect delivered status.

## 4. M7 candidate pool (for the M7 brainstorm)

Deferred / unbuilt, grouped by how strong the case is:

**Compute-bound (roofline-favorable — strongest candidates):**
- **Levenshtein / edit-distance** pairwise (roadmap item 13, deferred) — same `.metal.dtw` template, high compute density per pair. A real Polars vocabulary gap.
- **Cooperative-wavefront DTW** — the shipped DTW is one-thread-per-pair; a cooperative wavefront would push past 13.4×. ([[m6-a4-dtw-execution-state]])
- **Batched / other GEMM-shaped ops** — covariance matrix, pairwise distance matrices, attention-shaped kernels. The roofline says this is where M-series wins live.

**Namespace breadth (more `.metal` verbs we own):**
- **Spearman correlation** (needs a GPU rank/argsort first; corr spec out-of-scope).
- **Custom `shaders/corr.metal` GEMM** (corr uses MLX matmul; a fused standardize+GEMM could recover the p≈100 efficiency dip — only if profiled worth it).
- **Rader-path FFT** (deliberately skipped; prime-size FFT currently bridges through Bluestein).

**Statefulness / API surface:**
- **Cross-collect vector index reuse** (`build_index()`) — the "build once, query many" reuse cache, deferred from M6 as inherently stateful / outside the lazy model. Biggest API question.

**Coverage / dtype breadth:**
- **rolling F64 / integer** support (rolling is F32-only).
- **FFT real-input (rfft)** specialization — Hermitian symmetry halves the work for the common real case (planar rewrite makes this cleaner to add now).

**Infrastructure (not user-facing):**
- The general **NodeTraverser opacity unlock** — only if a future goal needs walker-level fusion of opaque nodes mid-tree. Currently moot.
- **Fused single-command-buffer pipelines** — the FFT pack/unpack regression showed per-dispatch barriers dominate; a general "fuse adjacent kernels into one command buffer" capability could matter for any multi-kernel op.

**Strategic note for M7 planning:** M6 proved the `.metal`-namespace pattern (own the op, serialize-detect, collect-and-stitch) is the productive path — it sidestepped the opacity fight entirely. M7 should likely keep extending it toward **compute-bound / GEMM-shaped** ops (where the roofline says we win) rather than chasing bandwidth-shaped ops to modest single-digit wins. Brainstorm M7 deliberately (the statefulness question for `build_index()` is the one genuinely new subsystem).
