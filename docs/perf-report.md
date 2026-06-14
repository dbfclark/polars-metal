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
- Rows that win (>1.15×): **29** / 37
- Rows that tie/lose: **8** (losses: 6)

## fusion-chain

| op | size | engine ms | CPU ms | engine ×CPU | ceiling ms | ceiling ×CPU | tax | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| haversine | 1,000,000 | 2.55 | 12.42 | 4.88× | 1.08 | 11.49 | 2.36 | 🟢 win |
| haversine | 10,000,000 | 8.32 | 175.01 | 21.03× | 3.38 | 51.71 | 2.46 | ✅ ≥10× |
| haversine | 100,000,000 | 60.99 | 1718.30 | 28.17× | 25.20 | 68.20 | 2.42 | ✅ ≥10× |
| black_scholes | 1,000,000 | 2.53 | 15.50 | 6.13× | 0.57 | 27.02 | 4.40 | 🟢 win |
| black_scholes | 10,000,000 | 8.70 | 219.53 | 25.24× | 2.58 | 85.21 | 3.38 | ✅ ≥10× |
| black_scholes | 100,000,000 | 72.13 | 2003.67 | 27.78× | 19.80 | 101.21 | 3.64 | ✅ ≥10× |

## rolling

| op | size | engine ms | CPU ms | engine ×CPU | ceiling ms | ceiling ×CPU | tax | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| rolling_mean_w1000 | 1,000,000 | 1.54 | 10.81 | 7.02× | — | — | — | 🟢 win |
| rolling_mean_w1000 | 10,000,000 | 3.37 | 110.84 | 32.90× | — | — | — | ✅ ≥10× |
| rolling_sum_w1000 | 1,000,000 | 2.70 | 11.08 | 4.10× | — | — | — | 🟢 win |
| rolling_sum_w1000 | 10,000,000 | 5.05 | 112.28 | 22.21× | — | — | — | ✅ ≥10× |
| rolling_var_w1000 | 1,000,000 | 4.56 | 22.98 | 5.04× | — | — | — | 🟢 win |
| rolling_var_w1000 | 10,000,000 | 33.37 | 230.82 | 6.92× | — | — | — | 🟢 win |
| rolling_std_w1000 | 1,000,000 | 5.40 | 22.93 | 4.24× | — | — | — | 🟢 win |
| rolling_std_w1000 | 10,000,000 | 33.20 | 233.35 | 7.03× | — | — | — | 🟢 win |

## vector-search

| op | size | engine ms | CPU ms | engine ×CPU | ceiling ms | ceiling ×CPU | tax | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| cosine_topk | 1,000 | 31.98 | 717.08 | 22.42× | 16.68 | 42.99 | 1.92 | ✅ ≥10× |
| cosine_topk | 10,000 | 276.47 | 7379.90 | 26.69× | 155.30 | 47.52 | 1.78 | ✅ ≥10× |
| knn | 1,000 | 33.36 | 839.17 | 25.15× | 21.54 | 38.97 | 1.55 | ✅ ≥10× |
| knn | 10,000 | 291.31 | 8669.81 | 29.76× | 166.80 | 51.98 | 1.75 | ✅ ≥10× |

## fft

| op | size | engine ms | CPU ms | engine ×CPU | ceiling ms | ceiling ×CPU | tax | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fft | 1,048,576 | 10.67 | 13.53 | 1.27× | — | — | — | 🟢 win |
| fft | 8,388,608 | 43.85 | 141.55 | 3.23× | — | — | — | 🟢 win |
| fft | 33,554,432 | 129.89 | 599.11 | 4.61× | — | — | — | 🟢 win |

## dtw

| op | size | engine ms | CPU ms | engine ×CPU | ceiling ms | ceiling ×CPU | tax | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dtw | 1,000 | 23.72 | 39.80 | 1.68× | — | — | — | 🟢 win |
| dtw | 50,000 | 916.24 | 2144.93 | 2.34× | — | — | — | 🟢 win |

## corr

| op | size | engine ms | CPU ms | engine ×CPU | ceiling ms | ceiling ×CPU | tax | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| corr_p10 | 100,000 | 2.82 | 1.89 | 0.67× | 1.21 | 1.56 | 2.32 | 🔴 loss |
| corr_p10 | 1,000,000 | 8.82 | 19.88 | 2.25× | 4.32 | 4.60 | 2.04 | 🟢 win |
| corr_p50 | 100,000 | 3.79 | 20.19 | 5.33× | 1.57 | 12.83 | 2.41 | 🟢 win |
| corr_p50 | 1,000,000 | 16.92 | 172.21 | 10.18× | 8.45 | 20.39 | 2.00 | ✅ ≥10× |

## temporal-int

| op | size | engine ms | CPU ms | engine ×CPU | ceiling ms | ceiling ×CPU | tax | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dt_year | 1,000,000 | 6.08 | 15.88 | 2.61× | — | — | — | 🟢 win |
| dt_year | 10,000,000 | 18.83 | 158.21 | 8.40× | — | — | — | 🟢 win |
| dt_year | 50,000,000 | 74.19 | 787.36 | 10.61× | — | — | — | ✅ ≥10× |
| int_sum | 1,000,000 | 0.84 | 0.55 | 0.65× | — | — | — | 🔴 loss |
| int_sum | 10,000,000 | 0.88 | 0.60 | 0.68× | — | — | — | 🔴 loss |
| int_sum | 100,000,000 | 2.46 | 2.23 | 0.91× | — | — | — | 🟡 tie |

## conformance-loser

| op | size | engine ms | CPU ms | engine ×CPU | ceiling ms | ceiling ×CPU | tax | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| tpch_q1 | 10,000,000 | 559.19 | 50.13 | 0.09× | — | — | — | 🔴 loss |
| tpch_q6 | 10,000,000 | 154.91 | 7.65 | 0.05× | — | — | — | 🔴 loss |
| bare_sum_f32 | 1,000,000 | 0.74 | 0.57 | 0.78× | — | — | — | 🔴 loss |
| bare_sum_f32 | 100,000,000 | 2.47 | 2.19 | 0.89× | — | — | — | 🟡 tie |

## Survey reconciliation

How the measured engine-path numbers compare to figures previously claimed in CLAUDE.md / the
M4 survey / memory. Most prior numbers were raw-MLX-vs-numpy (inflated) or measured on different
hardware; this column is the honest `engine="metal"` wall-clock on *this* machine.

| op | previously claimed | measured here (engine ×CPU) | reconciliation |
|---|---|---|---|
| haversine | 22× (M4 survey) | 21–28× @10–100M | matches |
| black_scholes | 28× (M4 survey) | 25–28× @10–100M | matches |
| rolling_mean | ~25× (M5 memory) | 33× @10M | holds, better at scale |
| cosine_topk / knn | ~20× (M6 memory) | 22–30× **vs competent BLAS-numpy** | matches; see caveats |
| fft | 77× (raw MLX vs numpy) / 3–4.6× (engine memory) | 1.3× @2²⁰ → 4.6× @2²⁵ | the 77× was raw-MLX; honest engine path is single-digit, grows with N |
| dtw | 13.4× (vs dtaidistance, M6 memory) | 1.7–2.3× **vs `distance_fast`** | much lower: the C `distance_fast` baseline is far stronger than the prior comparison |
| corr (p=50) | 7.8× (survey) / 9.9× (M6 memory) | 10.2× @1M | matches/holds |
| dt.year | 30–40× (survey) / 10–27× (memory) | 2.6–10.6× | lower: bandwidth-shaped; Polars CPU `dt.year` is a tight SIMD loop |
| TPC-H Q1/Q6 | 2.8–19.6× **slower** (M3) | 0.09× (11×slower) / 0.05× (20×slower) | confirmed loss, as designed |

**Baseline-honesty + measurement notes (these materially affect the numbers):**
- An earlier run had **inflated** `knn` (1489×) and `dtw` (953×) from naive CPU baselines (explicit
  numpy broadcast / pure-Python DTW). Corrected to **BLAS matmul** (`knn`) and **dtaidistance
  `distance_fast`** (`dtw`). The numbers above are the honest ones.
- **Run-to-run variance is real and large for the CPU baselines.** Between two runs the `cosine_topk`
  CPU baseline swung ~10× (BLAS thread warm-up / thermal), moving the reported win from ~84× to ~25×.
  The lower (this-run) numbers are the more trustworthy. **Treat every ratio as ±30% and don't over-read
  any single figure** — the *category-level* pattern (compute wins big, bandwidth loses) is the signal,
  not the exact multiplier.
- **Vector-search caveat:** CPU baseline is competent chunked-BLAS numpy doing *exact* brute force, not a
  specialized ANN library (faiss). faiss would narrow the gap; the win is genuine for exact batched search.

## The tax column — what the engine pays over raw GPU compute

The `tax` column (`engine_ms / ceiling_ms`, where ceiling is raw-MLX compute with host→device transfer
excluded) is wired for the five ops with a clean raw-MLX form. It is **the most decision-relevant number
in this report**:

| op | tax (engine ÷ raw-MLX compute) |
|---|---|
| cosine_topk / knn | **1.6–1.9×** |
| corr | **2.0–2.4×** |
| haversine | **2.4×** |
| black_scholes | **3.4–4.4×** |

Reading: even on the ops we win 22–28× at, the engine leaves **~1.5–4.4× on the table** versus raw GPU
compute — that gap is host→Metal ingest + readback/fold-back + FFI, paid **once per fused subtree**. The
win survives only because raw compute is 40–100× faster than CPU (the `ceiling ×CPU` column), so even
after the tax we clear the bar. This is CLAUDE.md Principle #2 quantified. (Ops without a raw-MLX form —
rolling, dt, fft-vs-numpy, dtw, tpch — show `—`; no meaningful raw ceiling exists.)

## Mission verdict & workload map

### Mission verdict
The order-of-magnitude bar (**≥10× vs Polars CPU**) is cleared **on the compute-bound class and nowhere
else** — exactly what the roofline predicts:

- **Clears ≥10× (12 rows):** fusion chains (haversine, black_scholes) at 10M+; rolling mean/sum at 10M;
  vector search (cosine_topk, knn) at every size; corr at p=50/1M; dt.year at 50M. High-FLOP-per-byte work.
- **Single-digit wins (bandwidth-tinged):** corr at small p/N, dt.year below 50M, fft (1.3→4.6×, grows with
  N), rolling var/std, dtw (1.7–2.3× vs the strong C baseline). Real but byte-capped.
- **Ties / losses (8 rows, all bandwidth-bound):** TPC-H Q1 (11× slower), Q6 (20× slower), bare F32/int
  sums (~tie, routed to CPU by the B4 guard). **The documented non-goal; the loss is consistent and
  expected.**

**Is the bar still right?** Yes. The engine is an order of magnitude faster on the workload class it
targets and loses on the class it explicitly doesn't. The "F32-compute-shaped" framing is validated, not
in need of rescoping.

### Workloads we can win at today
- **F32 transcendental feature pipelines** — option pricing (Black-Scholes), geospatial (haversine),
  scientific element-wise chains → **25–28× at 10M+**. The flagship win.
- **Embedding similarity / retrieval (exact)** — cosine top-k and L2 k-NN over a corpus → **22–30× vs
  competent numpy** (faiss narrows it; bounded by GPU memory, below).
- **Windowed time-series statistics** — rolling mean/sum over large F32 series → **22–33× at 10M**.
- **Correlation / covariance analytics** — wide correlation matrices → **10× at p=50/1M** (grows with p, N).
- **Spectral / signal batch** — large 1-D FFT → up to ~4.6× at 33M points (grows with N).
- **Temporal extraction at scale** — dt.year → ~10× at 50M (bandwidth-shaped; needs scale to win).

### Where the user still has to think too hard
- **Bandwidth-shaped ops lose or tie.** TPC-H-style filter/groupby/join-heavy queries and bare reductions
  are slower or parity. A mixed query is only as fast as its compute fraction (Principle #2).
- **A real scaling cliff:** `cosine_topk`/`knn` **OOM the GPU at Q=100k** (corpus 50k, D=768) — the engine
  materializes the full Q×corpus score matrix (~20 GB, over Metal's `maxBufferLength`). Practical batch
  sizes are GPU-memory-bound today; tiling the score matrix would lift this.
- **Small-input overhead:** corr@p10/100k, dt/fft at small N — fixed dispatch + the 1.5–4.4× ingest tax
  dominate, so the user needs enough rows to amortize.

### Next-direction trigger (not a decision)
**Does any winning workload's natural shape want to cross a join (or other CPU-fallback) boundary
mid-pipeline?** — **Yes, clearly: embedding retrieval.** `cosine_topk`/`knn` return *corpus indices*; the
next step in any retrieval/RAG pipeline is to **join those indices back to a document/metadata table**.
Today that join forces a GPU→CPU fold-back with no return to GPU. Feature pipelines similarly often **join
a dimension table** before the transcendental chain.

The tax column sharpens this: the engine already pays **1.5–4.4× per resident segment** for host↔device +
fold-back. A join that splits a pipeline makes you pay that crossing **twice** (fold out to join on CPU,
re-ingest for the next compute segment) plus the join itself. So the question a resident GPU join answers
is concrete and measurable: *does keeping the pipeline on-GPU across the join recover enough of that
double-crossing tax to beat routing the join to CPU?*

So the trigger the brainstorm set **is met**: a real, common fusion graph (retrieve → join metadata →
re-score) wants to span a join on a class we win 22–30× at. The signal is to, **next milestone**, build a
mixed-pipeline crossing-tax benchmark (retrieve-then-join, feature-join-then-chain) and measure how much of
the compute win the GPU↔CPU round-trip eats — the evidence for whether a resident GPU join pays off, in
service of "blazing fast without making the user structure the query." **This report establishes that the
trigger fired; it does not decide to build joins.**

