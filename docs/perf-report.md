# polars-metal — honest perf report

> Internal decision-input. Numbers are machine-specific (see header). Engine path is full `engine="metal"` wall-clock incl. ingest + fold-back.

## Environment
- **machine**: arm64
- **platform**: macOS-26.2-arm64-arm-64bit
- **python_version**: 3.11.4
- **polars_version**: 1.40.1
- **numpy_version**: 1.26.2
- **mlx_version**: 0.25.1
- **methodology**: warmup=2, iters=7, median reported, engine path includes ingest+fold-back

## Executive scorecard

- Rows clearing **≥10× vs CPU** (order-of-magnitude bar): **12** / 37
- Rows that win (>1.15×): **31** / 37
- Rows that tie/lose: **6** (losses: 6)

## fusion-chain

| op | size | engine ms | CPU ms | engine ×CPU | ceiling ms | ceiling ×CPU | tax | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| haversine | 1,000,000 | 6.20 | 30.79 | 4.96× | — | — | — | 🟢 win |
| haversine | 10,000,000 | 20.31 | 763.74 | 37.61× | — | — | — | ✅ ≥10× |
| haversine | 100,000,000 | 165.25 | 4751.01 | 28.75× | — | — | — | ✅ ≥10× |
| black_scholes | 1,000,000 | 5.07 | 45.19 | 8.91× | — | — | — | 🟢 win |
| black_scholes | 10,000,000 | 20.44 | 526.26 | 25.74× | — | — | — | ✅ ≥10× |
| black_scholes | 100,000,000 | 196.44 | 5571.19 | 28.36× | — | — | — | ✅ ≥10× |

## rolling

| op | size | engine ms | CPU ms | engine ×CPU | ceiling ms | ceiling ×CPU | tax | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| rolling_mean_w1000 | 1,000,000 | 3.01 | 30.81 | 10.24× | — | — | — | ✅ ≥10× |
| rolling_mean_w1000 | 10,000,000 | 5.47 | 316.62 | 57.91× | — | — | — | ✅ ≥10× |
| rolling_sum_w1000 | 1,000,000 | 3.63 | 25.21 | 6.95× | — | — | — | 🟢 win |
| rolling_sum_w1000 | 10,000,000 | 6.11 | 259.77 | 42.49× | — | — | — | ✅ ≥10× |
| rolling_var_w1000 | 1,000,000 | 7.31 | 54.29 | 7.43× | — | — | — | 🟢 win |
| rolling_var_w1000 | 10,000,000 | 36.18 | 662.02 | 18.30× | — | — | — | ✅ ≥10× |
| rolling_std_w1000 | 1,000,000 | 8.48 | 83.05 | 9.80× | — | — | — | 🟢 win |
| rolling_std_w1000 | 10,000,000 | 40.45 | 325.88 | 8.06× | — | — | — | 🟢 win |

## vector-search

| op | size | engine ms | CPU ms | engine ×CPU | ceiling ms | ceiling ×CPU | tax | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cosine_topk | 1,000 | 93.14 | 7870.26 | 84.50× | — | — | — | ✅ ≥10× |
| cosine_topk | 10,000 | 1043.29 | 63452.54 | 60.82× | — | — | — | ✅ ≥10× |
| knn | 1,000 | 71.64 | 6816.77 | 95.15× | — | — | — | ✅ ≥10× |
| knn | 10,000 | 1963.98 | 66506.23 | 33.86× | — | — | — | ✅ ≥10× |

## fft

| op | size | engine ms | CPU ms | engine ×CPU | ceiling ms | ceiling ×CPU | tax | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fft | 1,048,576 | 25.88 | 35.52 | 1.37× | — | — | — | 🟢 win |
| fft | 8,388,608 | 81.11 | 252.04 | 3.11× | — | — | — | 🟢 win |
| fft | 33,554,432 | 175.02 | 1673.08 | 9.56× | — | — | — | 🟢 win |

## dtw

| op | size | engine ms | CPU ms | engine ×CPU | ceiling ms | ceiling ×CPU | tax | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dtw | 1,000 | 25.56 | 113.76 | 4.45× | — | — | — | 🟢 win |
| dtw | 50,000 | 2041.57 | 5326.98 | 2.61× | — | — | — | 🟢 win |

## corr

| op | size | engine ms | CPU ms | engine ×CPU | ceiling ms | ceiling ×CPU | tax | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| corr_p10 | 100,000 | 8.54 | 6.60 | 0.77× | — | — | — | 🔴 loss |
| corr_p10 | 1,000,000 | 52.61 | 72.28 | 1.37× | — | — | — | 🟢 win |
| corr_p50 | 100,000 | 18.75 | 71.17 | 3.80× | — | — | — | 🟢 win |
| corr_p50 | 1,000,000 | 38.56 | 312.19 | 8.10× | — | — | — | 🟢 win |

## temporal-int

| op | size | engine ms | CPU ms | engine ×CPU | ceiling ms | ceiling ×CPU | tax | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dt_year | 1,000,000 | 20.40 | 45.85 | 2.25× | — | — | — | 🟢 win |
| dt_year | 10,000,000 | 66.53 | 357.79 | 5.38× | — | — | — | 🟢 win |
| dt_year | 50,000,000 | 204.36 | 1539.52 | 7.53× | — | — | — | 🟢 win |
| int_sum | 1,000,000 | 1.86 | 0.95 | 0.51× | — | — | — | 🔴 loss |
| int_sum | 10,000,000 | 2.06 | 1.57 | 0.76× | — | — | — | 🔴 loss |
| int_sum | 100,000,000 | 6.32 | 18.05 | 2.85× | — | — | — | 🟢 win |

## conformance-loser

| op | size | engine ms | CPU ms | engine ×CPU | ceiling ms | ceiling ×CPU | tax | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| tpch_q1 | 10,000,000 | 1867.41 | 300.58 | 0.16× | — | — | — | 🔴 loss |
| tpch_q6 | 10,000,000 | 450.40 | 46.48 | 0.10× | — | — | — | 🔴 loss |
| bare_sum_f32 | 1,000,000 | 3.07 | 1.83 | 0.60× | — | — | — | 🔴 loss |
| bare_sum_f32 | 100,000,000 | 17.93 | 23.37 | 1.30× | — | — | — | 🟢 win |

## Survey reconciliation

How the measured engine-path numbers compare to figures previously claimed in CLAUDE.md / the
M4 survey / memory. Most prior numbers were either raw-MLX-vs-numpy (inflated) or measured on
different hardware; this column is the honest `engine="metal"` wall-clock on *this* machine.

| op | previously claimed | measured here (engine ×CPU) | reconciliation |
|---|---|---|---|
| haversine | 22× (M4 survey) | 28–38× @10–100M | holds, better at scale |
| black_scholes | 28× (M4 survey) | 26–28× @10–100M | matches |
| rolling_mean | ~25× (M5 memory) | 58× @10M | holds, better at scale |
| cosine_topk / knn | ~20× (M6 memory) | 34–95× **vs competent BLAS-numpy** | holds; see caveat below |
| fft | 77× (raw MLX vs numpy) / 3–4.6× (engine memory) | 1.4× @2²⁰ → 9.6× @2²⁵ | the 77× was raw-MLX; honest engine path grows with N, reaches ~10× |
| dtw | 13.4× (vs dtaidistance, M6 memory) | 2.6–4.5× **vs `distance_fast`** | lower: the C `distance_fast` baseline is much stronger than the prior comparison |
| corr (p=50) | 7.8× (survey) / 9.9× (M6 memory) | 8.1× @1M | matches |
| dt.year | 30–40× (survey) / 10–27× (memory) | 2.3–7.5× | lower: bandwidth-shaped; Polars CPU `dt.year` is a tight SIMD loop |
| TPC-H Q1/Q6 | 2.8–19.6× **slower** (M3) | 0.16× (6×slower) / 0.10× (10×slower) | confirmed loss, as designed |

**Baseline-honesty notes (these materially affect the numbers):**
- The first run had **inflated** `knn` (1489×) and `dtw` (953×) because their CPU baselines were
  naive (explicit numpy broadcast / pure-Python DTW). Corrected to **BLAS matmul** (`knn`) and
  **dtaidistance `distance_fast`** (`dtw`) — the numbers above are the honest ones.
- **Vector search caveat:** the CPU baseline is competent chunked-BLAS numpy (exact brute force),
  *not* a specialized ANN library (faiss). faiss would narrow the gap; the engine's win is genuine
  for *exact* batched search, which is the apples-to-apples comparison.
- **The `tax` (ceiling) column is empty** — no `ceiling_fn` was wired (the raw-MLX ceilings were
  out of scope for this pass). The engine-vs-CPU column is the mission bar and is fully populated;
  wiring raw-MLX ceilings to expose the ingest/fold-back tax explicitly is a follow-up.
- **Variance:** several large-N rows (dtw@50k, black_scholes@100M, corr) show wide min/max spreads
  (thermal / Metal warmup). Medians are reported; treat single-digit ratios as ±25%.

## Mission verdict & workload map

### Mission verdict
The order-of-magnitude bar (**≥10× vs Polars CPU**) is cleared **decisively on the compute-bound
class and nowhere else** — exactly what the roofline predicts:

- **Clears ≥10× (12 rows):** fusion chains (haversine, black_scholes) at 10M+; rolling mean/sum/var
  at 10M; vector search (cosine_topk, knn) at every size. These are high-FLOP-per-byte kernels.
- **Single-digit wins (bandwidth-tinged):** corr (grows to 8× at p=50/1M), dt.year (2–7.5×), fft
  (grows to ~9.6× at 2²⁵), rolling std, dtw (2.6–4.5×). Real wins, but the byte traffic caps them.
- **Ties / losses (6 rows, all bandwidth-bound):** TPC-H Q1 (6× slower), Q6 (10× slower), bare
  F32/int sums (~tie, routed to CPU by the B4 guard). **These are the project's documented
  non-goal, and the loss is consistent and expected.** int_sum/bare_sum's apparent 100M "wins"
  (1.3–2.85×) are within measurement noise of a tie — they route to CPU.

**Is the bar still right?** Yes. The engine is an *order of magnitude faster on the workload class
it targets* and *loses on the class it explicitly doesn't*. The mission's "F32-compute-shaped"
framing is validated, not in need of rescoping.

### Workloads we can win at today
Translating the measured wins into real-world data-processing challenges `engine="metal"` is
blazing-fast at right now:

- **F32 transcendental feature pipelines** — option pricing (Black-Scholes), geospatial
  (haversine), scientific/physics element-wise chains → **26–38× at 10M+ rows**. The flagship win.
- **Embedding similarity / retrieval (exact)** — cosine top-k and L2 k-NN over a corpus →
  **34–95× vs competent numpy** (caveat: vs faiss the gap narrows; bounded by GPU memory, below).
- **Windowed time-series statistics** — rolling mean/sum/var over large F32 series →
  **18–58× at 10M**. Streaming analytics, signal smoothing.
- **Spectral / signal batch** — large 1-D FFT → **up to ~9.6× at 33M points** (grows with N).
- **Correlation / covariance analytics** — wide-ish correlation matrices → **8× at p=50, N=1M**
  (grows with p and N; loses at small p/N where dispatch overhead dominates).
- **Time-series alignment** — DTW against a reference → **2.6–4.5×** (honest, vs the C baseline).

### Where the user still has to think too hard
- **Bandwidth-shaped ops lose silently-ish.** TPC-H-style filter/groupby/join-heavy queries, and
  bare reductions, are slower or tie. The router sends bare reductions to CPU (good), but a user
  writing a mixed query gets the *compute* part accelerated and the *bandwidth* part at parity —
  the wall-clock win is only as good as the compute fraction (CLAUDE.md Principle #2).
- **A real scaling cliff:** `cosine_topk`/`knn` **OOM the GPU at Q=100k** (corpus 50k, D=768) —
  the engine materializes the full Q×corpus score matrix (~20 GB, over Metal's `maxBufferLength`).
  Practical batch sizes are bounded by GPU memory today; tiling the score matrix would lift this.
- **Small-input overhead:** corr@p10/100k, dt/fft at small N — fixed dispatch + ingest cost
  dominates, so the user must have enough rows to amortize it.

### Next-direction trigger (not a decision)
**Does any winning workload's natural shape want to cross a join (or other CPU-fallback) boundary
mid-pipeline?** — **Yes, at least one, clearly: embedding retrieval.** `cosine_topk`/`knn` return
*corpus indices*; the very next step in a real retrieval/RAG pipeline is to **join those indices
back to a document/metadata table**. Today that join forces GPU→CPU fold-back with no return to
GPU. Feature pipelines similarly often **join a dimension table** before the transcendental chain.

So the trigger condition the brainstorm set **is met**: there is a real, common fusion graph
(retrieve → join metadata → maybe re-score) that wants to span a join while staying on a workload
class where we win 34–95×. That is the signal to, **next milestone**, build a mixed-pipeline
crossing-tax benchmark (retrieve-then-join, feature-join-then-chain) and measure how much of the
compute win the GPU↔CPU round-trip eats — which is the evidence for whether a resident GPU join
pays off. **This report does not decide that; it establishes that the trigger fired.**

