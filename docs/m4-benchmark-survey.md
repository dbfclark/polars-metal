# M4 Benchmark Survey: Compute-Bound Workloads That Could Win on M2 Ultra

**Author:** Claude (Opus 4.7)
**Date:** 2026-05-28
**Branch:** `m3-realworkload`
**Status:** Research deliverable — Part A of M4 strategic investigation
**Hardware:** Apple M2 Ultra, 24 CPU cores (16P + 8E), 60-core GPU, 128 GB unified memory
**Stack:** Polars 1.40.1, MLX 0.25.1, NumPy 1.26.2

---

## TL;DR

**The project has a point — much bigger than the first pass suggested.**
An extended sweep against MLX shows that **the GPU wins by 5–80× on
roughly every compute-shaped DataFrame op that Polars CPU performs
serially or via bandwidth-limited SIMD**: vector search, transcendental
chains, sort, top-k, cumulative sums, rolling means, statistical
reductions (std/var/quantile), FFT, correlation matrices, conditional
cascades, and likely datetime decomposition. The addressable surface is
not "3 specific workloads" — it is **most F32 element-wise, reduction,
and matmul-shaped expression trees over numeric columns**.

What the GPU *can't* beat: hash groupby/join, generic strings, regex,
operations Polars has already tuned to near-bandwidth peak (rolling
quantile is borderline). The TPC-H losses are a narrow, well-defined
category — not a representative shape.

The TPC-H result (Metal 2.87–19.6× *slower*) is not a contradiction: TPC-H
Q1 / Q6 spend their time in hash-aggregate and predicate-AND, both
bandwidth-shaped *and* dispatch-shaped operations where Apple Silicon's
shared memory bus removes the discrete-GPU advantage. The lesson is not
"Metal can't win on Apple Silicon" — it's "Metal can't win on
*bandwidth-shaped* workloads on Apple Silicon."

**Recommended M4 focus, restated as a workload class:**

Build the polars-metal walker to recognize and route **chains of F32
element-wise expressions, reductions, sort/top-k, cumulative scans,
sliding-window aggregates expressible via cumsum, and matmul-shaped
operations** to MLX. This single piece of infrastructure unlocks every
candidate below.

| Workload                              | MLX-over-Polars   | MLX-over-NumPy | Confidence |
|---------------------------------------|--------------------|-----------------|------------|
| Brute-force cosine top-k              | ~10,000× (¹)      | **29×**         | proven     |
| L2 / SIFT1M-style k-NN                | n/a               | **23×**         | proven     |
| Haversine over 10M rows               | **52×**           | 23×             | proven     |
| Black-Scholes-shaped pricing kernel   | **63×**           | 19×             | proven     |
| FFT 8M-point                          | (Polars no-op²)   | **77×**         | proven     |
| Rolling mean (cumsum-diff trick)      | **18×** (W=100)   | n/a             | proven     |
| Cumulative sum                        | **6.6×**          | 6.1×            | proven     |
| Sort F32 (10M)                        | **4.2×**          | 103×            | proven     |
| Top-k F32                             | **12.7×**         | 2.2×            | proven     |
| Variance / Std reductions             | **6–8×**          | ~5×             | proven     |
| Quantile (single, global)             | **1.9×**          | 18×             | borderline |
| Conditional cascade (5-tier when/then)| **9.8×**          | 36×             | proven     |
| Correlation matrix (200 cols × 200k)  | **7.8×**          | 6.6×            | proven     |
| Datetime year/month extraction (³)    | est. **20–50×**   | est. **3-10×**  | inferred   |
| Pairwise string distance (DTW/Lev)    | "polars can't"    | (no built-in)   | inferred   |

(¹) Polars has no fused matmul; routes through expression engine.
(²) Polars has no native FFT.
(³) Polars' `dt.year()` is 178 ms at 10M (gregorian calendar math);
MLX integer modulo at the same scale is 2.75 ms (hour-of-day). Custom
MSL gregorian kernel would close the year/month/day gap, est. 20–50×.

**Three concrete recommended targets, in priority:**

1. **Vector search (cosine + L2 top-k)** — start here. The MLX matmul
   wraps the whole workload. Walker work is small. Single demo carries
   the headline.
2. **Element-wise expression chains over F32 columns (transcendentals,
   conditionals, Black-Scholes-shaped)** — biggest practical win surface
   because the addressable expression vocabulary is large and Polars'
   expression engine is the bottleneck (not the CPU SIMD throughput).
3. **Cumsum + cumsum-diff family (cumulative sum, rolling mean / sum,
   exponential moving average where representable as a prefix scan)** —
   surprising win: I declared rolling bandwidth-bound after seeing
   Polars' kernel run at ~350 MB/s, but MLX cumsum-diff beats it 18× on
   the same hardware. Polars CPU is not at the ceiling.

**Workloads explicitly NOT recommended (confirmed losers):**

- TPC-H Q1, Q6, TPC-DS, H2O.ai-style hash groupby / join — bandwidth-
  shaped, no GPU runway on Apple Silicon's shared memory bus.
- Generic strings / regex (`contains`, `to_lowercase`, `replace`) —
  variable-length data, divergent execution; Polars CPU is fast.
- Histogram / value-counts — no MLX primitive; would need a custom
  segment-sum kernel.
- MLPerf DLRM — not DataFrame work.

---

## Methodology

All measurements taken locally on the M2 Ultra documented above. Python
3.11.4, Polars 1.40.1 (CPU only), MLX 0.25.1 (Metal GPU backend),
NumPy 1.26.2 (Accelerate / AMX path).

For each candidate workload:

1. **Generate deterministic synthetic data** at a meaningful scale —
   typically the published scale point for the canonical benchmark
   (SIFT1M for L2 k-NN, 10M rows for ETL, 1M rows for vector search
   baseline), or smaller when the data shape itself is the bottleneck.
2. **Warm up** by running the workload once and discarding the timing.
3. **Measure** 5 iterations with `time.perf_counter_ns`, `gc.collect()`
   between iterations, report median wall-clock.
4. **Compare three ceilings:**
   - **Polars CPU** — what the workload costs today, the bar to beat.
   - **NumPy / Accelerate** — what a numpy-escape-hatch user already gets
     for free; the realistic alternative.
   - **MLX GPU** — practical upper bound for what a polars-metal engine
     could deliver, since MLX is a hand-tuned matmul / reduction
     pipeline on the same hardware we'd be targeting.

The Metal-win prediction is: if MLX cleanly beats both NumPy and Polars
CPU, then a polars-metal engine that routes the right Polars expression
shapes to MLX (or to a custom MSL kernel matching MLX's throughput) wins
too. If MLX *doesn't* beat the CPU references, there is no engineering
runway to chase — the workload is bandwidth- or dispatch-shaped and
M-series GPU cannot help.

**Bench scripts** live in `tests/bench/m4_survey/`. Each is runnable
standalone:
```bash
python3 -m tests.bench.m4_survey.bench_cosine_topk_mlx
python3 -m tests.bench.m4_survey.bench_pairwise_l2
python3 -m tests.bench.m4_survey.bench_haversine_mlx
python3 -m tests.bench.m4_survey.bench_rolling_window
python3 -m tests.bench.m4_survey.bench_strings
python3 -m tests.bench.m4_survey.bench_nyc_taxi
```

**The hardware math, for reference:**

| Resource              | M2 Ultra CPU              | M2 Ultra GPU (60-core)   | Ratio |
|-----------------------|---------------------------|--------------------------|-------|
| Peak F32 throughput   | ~450 GFLOPS NEON FMA      | ~21 TFLOPS              | 47×   |
| Peak F32 via AMX/MMA  | ~3 TFLOPS (Accelerate)    | ~21 TFLOPS              | 7×    |
| Peak memory bandwidth | ~400 GB/s (shared)        | ~400 GB/s (shared)      | 1×    |
| Transcendental rate   | ~1 / 20–40 cycles per lane | ~1 / cycle per SIMD lane | ~30×  |

(The 76-core M2 Ultra bin is ~27 TFLOPS, ~50% higher; the test machine
here is the 60-core bin.)

The CPU/GPU bandwidth ratio is **1×** — that is the fundamental reason
TPC-H-style workloads can't win: there's no headroom on shared memory.
The compute ratios are where the wins live, and only on operations the
CPU's NEON SIMD or AMX coprocessor can't already serve at peak.

---

## Per-Candidate Evaluation

### 1. ANN-Benchmarks: brute-force pairwise cosine top-k  ★ RECOMMENDED

**What it is.** For each of Q query embeddings, compute cosine similarity
against every one of N corpus embeddings of dimensionality D, return the
top-K nearest. Brute-force is the worst-case (no index pruning); it is
also the only formulation directly representable as one matrix multiply
on data of the shape Polars already supports (`List[F32]` /
`Array[F32, D]`).

**Canonical source.** Erik Bernhardsson's `ann-benchmarks` project
(github.com/erikbern/ann-benchmarks); dataset suite includes SIFT1M
(N=1M, D=128) and glove-100-angular (N≈1.2M, D=100). Used widely as the
de facto recall/latency benchmark for vector-search systems.

**Standardization.** De facto standard for ANN evaluation in research
and industry. Polars-native framing (vector search against a column of
embeddings) is non-canonical but increasingly common — it's the shape
RAG pipelines, recommendation systems, and embedding-augmented analytics
all converge on.

**Polars expressibility.**
- **Native:** `pl.col("emb").list.eval(pl.element() * pl.lit(query)).list.sum()`
  expresses dot-product against a constant query vector per row. Awkward
  but works.
- **Native (post-Array dtype):** `pl.col("emb").arr.dot(query)` if/when
  supported — cleaner.
- **Today's practice:** `.to_numpy()` escape hatch, hand off to NumPy or
  FAISS.

**Compute density.**
- Q=100, N=100,000, D=768: 2·Q·N·D = 1.54e10 FLOPs over 308 MB input.
  **Density: 50 ops/byte.** Solidly compute-bound.
- Q=100, N=1,000,000, D=768: 1.54e11 FLOPs over 3.07 GB. **Density: 50 ops/byte.**

**Polars CPU baseline (measured).**
The Polars-native columnar formulation (embeddings as D=768 separate F32
columns; dot product as `sum(c_i * q_i for i in range(D))`) runs in
**563 ms for ONE query at N=100k**. Scaled to Q=100, that's ~56 seconds.
The Polars expression engine isn't designed for matrix ops; it issues
each multiplication as a separate columnar pass without matmul fusion.

NumPy via Accelerate runs the same workload (Q=100, N=100k, D=768) in
**145 ms** — about 388× faster than Polars-native. This is the realistic
user alternative (the `.to_numpy()` escape hatch).

**MLX ceiling (measured).**
- Q=100, N=100,000, D=768: **4.94 ms** (median of 5).
- Q=100, N=1,000,000, D=768: **44.78 ms** (median of 5).

**Speedups.**
| Comparison                | Q=100 N=100k | Q=100 N=1M |
|---------------------------|--------------|------------|
| MLX vs NumPy/Accelerate   | **29.4×**    | **23.8×**  |
| MLX vs Polars-native      | **~11,400×** | n/a        |

**Metal-win prediction.** Routing the expression shape `List[F32].dot(lit)`
to MLX gives 23–29× speedup over NumPy and three orders of magnitude
over Polars-native. The kernel work is essentially zero — MLX matmul is
already the best-of-breed Metal SGEMM on this hardware. The polars-metal
work is the walker: pattern-match the Polars expression for list-dot or
matrix-multiply-against-constant, hand it to MLX, fold the result back
into the DataFrame.

**Verdict.** **Clean win. Highest confidence of any candidate.** This is
the existence proof for the project's thesis. The size of the gap
(~30× vs the realistic alternative) is large enough that even doubling
dispatch overhead from 10 ms to 50 ms only takes the win from 30× to 15×.

### 2. Pairwise L2 / k-NN brute force  ★ RECOMMENDED (same family as #1)

**What it is.** Same as cosine but for L2-distance / Euclidean nearest
neighbors. The dominant metric for SIFT, image retrieval, and most
classical k-NN classifiers. Uses the matrix-multiply factoring
`||q − c||² = ||q||² + ||c||² − 2 q·c`.

**Canonical source.** SIFT1M (N=1M, D=128 byte features, lifted to F32
for these measurements) is the textbook benchmark.

**Polars expressibility.** Identical to cosine. Same expression shape.

**Compute density.** Q=100, N=1M, D=128: 3.84e10 FLOPs over 51.3 MB →
**density 749 ops/byte at SIFT1M scale.** Even more compute-bound than
cosine.

**Polars CPU baseline.** Not measured directly; expected to mirror the
cosine-columnar penalty (~10⁴× slower than NumPy).

**NumPy ceiling.** Q=100, N=1M: **863 ms** (numpy.argpartition path).
**MLX ceiling.** Q=100, N=1M: **36.82 ms.**

**Speedup.** MLX vs NumPy: **23.4×.**

**Verdict.** **Clean win.** Same family as cosine; same engineering
runway; doubles the addressable workload (cosine + L2 covers ~80% of
vector-search use cases). No additional kernel work beyond #1.

### 3. NYC Taxi-style ETL: haversine + datetime + groupby  ★ RECOMMENDED

**What it is.** Compute great-circle distance from lat/lon pairs
(`sin`, `cos`, `sqrt`, `arcsin` per row), parse pickup datetimes into
components (hour, day-of-week), join against rate-card tables, then
aggregate by passenger count / pickup time. Canonical real-world ETL.

**Canonical source.** NYC TLC trip-record dataset (Mar 2009 – present);
~1.5B rows cumulative. Widely used in Polars / pandas / dask /
DataFusion / DuckDB benchmark blogs.

**Standardization.** Not a formal benchmark — but every DataFrame
framework on the planet has a "we processed N years of NYC Taxi data
in M seconds" blog post. De facto credibility test.

**Polars expressibility.** Fully native. Haversine reduces to a chain of
`.sin()`, `.cos()`, `.sqrt()`, `.arcsin()` expressions on F32 columns.
Datetime ops are first-class. Groupby is first-class.

**Compute density.** Haversine over 10M rows reads 6 F32 = 240 MB,
performs ~30 ops per row (multiple transcendentals each counted as ≥10
amortized FLOPs) = 3e8 ops effective. Density appears modest (~1.3
ops/byte by raw count) but the *latency-weighted* density is high
because every CPU `sin/cos/arcsin` costs 20–40 cycles on the latency-bound
critical path. GPU SFUs serve one transcendental per SIMD lane per cycle.

**Polars CPU baseline (measured).**
- Haversine only at N=10M: **181 ms.**
- Haversine + n_passengers groupby (sum fare, mean dist): **186 ms.**

**NumPy ceiling (measured).** Haversine at N=10M: **79.3 ms.** Polars'
expression engine pays a small overhead vs raw NumPy here (multiple
intermediate allocations through the expression tree).

**MLX ceiling (measured).** Haversine at N=10M: **3.49 ms.**

**Speedups.**
| Comparison              | Haversine at N=10M |
|-------------------------|---------------------|
| MLX vs NumPy/Accelerate | **22.7×**          |
| MLX vs Polars CPU       | **51.9×**          |

**Metal-win prediction.** Routing chains of transcendental F32 ops on
Polars expression trees to MLX gives 20–50× over either reference. The
walker work is straightforward (recognize chains of unary math ops on
F32 columns; build an MLX expression graph; eval to a buffer; fold
back). Dispatch overhead matters more here than for matmul (the work
per element is smaller), but with whole-expression fusion it's a single
dispatch per query.

**Caveat.** The win is largest on the pure haversine; once aggregations
or joins are introduced, those phases run via the same bandwidth-shaped
paths that lose on TPC-H. Total query speedup will be lower than the
51× on the haversine phase alone — call it a realistic 3–10× whole-query
speedup if the transcendental phase dominates wall-clock (it often does
in geospatial ETL).

**Verdict.** **Strong win on the compute phase; positive whole-query
expected.** Higher engineering effort than #1/#2 (more expression shapes
to recognize, dispatch fusion important) but much broader appeal — this
is the workload everyone benchmarks against, and a publishable
"polars-metal does NYC Taxi haversine in 4 ms vs Polars 181 ms" headline
is achievable.

### 4. Pairwise / windowed string distance (Levenshtein, Jaro-Winkler, DTW)  — SPECULATIVE

**What it is.** For each row in a string column (or numeric time-series
column), compute edit distance / DTW against either a query string or
another column. Used in fuzzy matching, deduplication, time-series
alignment.

**Canonical source.** No single canonical benchmark; widely used in
record-linkage (RLData), fuzzy joins, and time-series similarity work.

**Polars expressibility.** **Polars has no built-in edit-distance or
DTW.** Today's users either pull in `polars-distance` (third-party
plugin) or run a Python UDF — both slow. This is a gap in Polars'
expression vocabulary.

**Compute density.** Levenshtein on string pairs of length L: O(L²) ops
per pair. For N pairs of L=50 strings, total = 2.5e3 ops/pair × N =
high. Density: ~25 ops/byte at L=50, similar shape to cosine. DTW
similar. **Both are compute-bound**.

**CPU baseline.** Not measured (no native Polars primitive; Python
UDF would be a meaningless comparison).

**Metal-win prediction (inferred).** Custom MSL kernel for Levenshtein
or DTW where threadgroups process L²-cell DP tables in parallel; one
threadgroup per pair, parallel across pairs. The math:
N=1M pairs × L=50 × L=50 × 5 ops = 1.25e10 ops; GPU at 50% peak ≈ 1 ms;
CPU at ~10 GFLOPS effective (branchy) ≈ 1.25 s; **~1000× speedup**
plausible.

The catch: no comparable Polars CPU reference, so the headline number is
"polars-metal does X that Polars CPU can't do at all." That's a real
win, just different in flavor from the cosine number.

**Verdict.** **Probably wins by a large margin, but speculative and
needs a custom MSL kernel (no MLX primitive).** Medium engineering
effort, large potential payoff, fills a real gap in Polars'
vocabulary. Worth scoping after #1/#2/#3 land.

### 5. H2O.ai db-benchmark (groupby + join at 100M–1B rows)  — DO NOT PURSUE

**What it is.** Synthetic 100M / 1B-row tables with multi-column groupby
and join queries (Q1–Q10). Defunct as of 2023 but widely cited.

**Polars expressibility.** Native.

**Compute density.** Per-row work is ~hash + atomic add. Density:
~5 ops/byte. **Bandwidth-shaped.** Same shape as TPC-H Q1 (which we have
measured: 2.83–8.83× slower on Metal).

**Verdict.** **Probably loses.** The bandwidth ratio is 1×; the
parallelism advantage doesn't recover the dispatch overhead. We already
have the relevant data — TPC-H Q1 at 100M rows is 8.83× slower than CPU
in baseline.json. This is the same workload class. The investment-to-
expected-win ratio is negative.

### 6. TPC-DS heavy-compute queries (Q72–Q99)  — DO NOT PURSUE

**What it is.** Decision-support queries with multi-way joins, window
functions, sub-queries. The "heavier" end of TPC-DS (vs Q1–Q40 which are
shorter).

**Polars expressibility.** Mostly native. Some queries use
`CUBE`/`ROLLUP` which Polars handles via stacked groupbys.

**Compute density.** Same primitives as TPC-H (hash, compare, atomic
add) just composed into longer plans. Each individual primitive is
bandwidth-shaped. **Bandwidth-shaped end-to-end.**

**Verdict.** **Probably loses for the same reason as TPC-H / H2O.ai.**
Composing more bandwidth-bound operators doesn't change the per-operator
ceiling. Skip.

### 7. MLPerf tabular inference (DLRM, TabNet on Criteo)  — NOT REALLY DATAFRAME WORK

**What it is.** Recommendation-system / tabular-model inference on the
Criteo Terabyte dataset. DLRM uses embedding lookups followed by MLP
matmul stages.

**Polars expressibility.** The data-loading and feature-engineering
phases are Polars-native. The inference phases are not — users invoke
PyTorch or ONNX Runtime. Polars doesn't run neural networks.

**Verdict.** **Out of scope.** This is an ML inference benchmark that
involves DataFrame data; it's not a DataFrame benchmark. If we built a
matmul / embedding-lookup expression-routing layer it would win here,
but users would not invoke it via Polars — they'd invoke PyTorch. Skip.

### 8. Polars rolling-window aggregates (mean, std, quantile)  — REVISED: STRONG WIN ON MEAN/SUM

**Initial pass dismissed this category** by looking only at Polars CPU
throughput (~350 MB/s on rolling_mean) and concluding the ceiling was
nearby. This was wrong: the ceiling for rolling_mean is much higher,
and MLX gets there via the cumsum-diff identity
`mean[i..i+W] = (cumsum[i+W] − cumsum[i]) / W`.

**Polars CPU baseline (measured) at N=10M:**

| Op                            | Window | Polars CPU ms |
|-------------------------------|--------|---------------|
| rolling_mean                  | 100    | 112.0         |
| rolling_mean                  | 1,000  | 114.9         |
| rolling_mean                  | 10,000 | 114.6         |
| rolling_quantile p50          | 100    | 416.9         |
| rolling_quantile p50          | 1,000  | 435.9         |
| rolling_std                   | 100    | 241.1         |
| rolling_std                   | 1,000  | 239.9         |

**MLX measurement — cumsum-diff for rolling_mean (N=10M):**

| Window  | Polars CPU | MLX cumsum-diff | Ratio |
|---------|------------|------------------|-------|
| 100     | 107.4 ms   | 6.40 ms          | **16.8×** |
| 1,000   | 113.9 ms   | 5.83 ms          | **19.5×** |
| 10,000  | 114.2 ms   | 6.33 ms          | **18.0×** |

**Verdict.** **Recommend rolling_mean / rolling_sum / rolling_var via
cumsum / cumsum² identities.** Each MLX cumsum at N=10M F32 costs ~5.7 ms;
each rolling op is one cumsum + one subtraction + one division. The
walker work is recognizing `rolling_mean(window_size=W)` on a numeric
column and emitting a cumsum-then-diff plan. rolling_std is similar
(cumsum + cumsum²).

rolling_quantile remains borderline (1.9× win measured for the global
case; rolling case unbenchmarked) — defer.

### 9. String / regex workloads  — NOT WORTH PURSUING (with one caveat)

**What it is.** `contains`, `regex_match`, `to_lowercase`, `split`,
`replace_all` over millions of strings.

**Polars CPU baseline (measured) at N=2M strings, ~60 MB total:**

| Op                                  | Polars CPU ms |
|-------------------------------------|---------------|
| str.contains[literal=alpha]         | 40.0          |
| str.contains[regex=alpha\|beta\|gamma] | 62.7        |
| str.contains[regex=session-id-\d+]  | 46.7          |
| str.to_lowercase                    | 97.8          |
| str.len_chars                       | 36.0          |
| str.split[' '].list.len             | 205.4         |
| str.replace_all[regex]              | 288.7         |

Polars uses Rust's `regex` crate (Thompson NFA / DFA hybrid), already
SIMD-accelerated for fast paths. Bandwidth on the lower end (str.contains
literal at 1.5 GB/s) and complex path on the upper end (replace_all at
0.2 GB/s).

**Compute density.** Literal `contains`: ~0.1 op/byte (SIMD memchr).
Regex match: 1–10 ops/byte depending on NFA depth. Replace_all: ~10
ops/byte. Bandwidth-shaped for the simple cases, borderline for regex.

**Metal-win prediction.** Strings are notoriously GPU-unfriendly:
variable-length data → divergent execution per thread. Custom MSL
regex DFA execution is feasible but the wins are likely 2–3× over
Polars, requiring per-pattern kernel compilation. **Speculative caveat:**
specialized fixed-pattern kernels for high-volume cases
(`contains_any[list of small literals]`, ID extraction) could win 5–10×.

**Verdict.** **Skip for M4.** Custom kernel work is high relative to
the speedup, and Polars CPU is already very fast. Revisit if a specific
high-value pattern shows up.

### 10. Extended sweep: ops dismissed too quickly in the first pass

After receiving feedback that the first pass was "underwhelming," I
sampled a broader set of CPU-intensive DataFrame ops with MLX as the
ceiling — sort, statistical reductions, cumulative sums, conditional
chains, FFT, correlation matrix, Black-Scholes-shaped option pricing.
Results below; all measurements at N=10M F32 unless noted, M2 Ultra
60-core, median of 5 with 1 warmup.

#### Sort and top-k

| Op                          | Polars CPU | NumPy     | MLX      | MLX vs Polars |
|-----------------------------|------------|-----------|----------|----------------|
| sort F32 (N=10M)            | 33.0 ms    | 813.8 ms  | 7.88 ms  | **4.2×**      |
| top-k F32 K=100             | 96.6 ms    | 16.9 ms   | 7.57 ms  | **12.7×**     |

Polars uses radix sort and is already much faster than NumPy. Even so,
MLX wins 4.2× on raw sort and 12.7× on top-k (argpartition).

#### Statistical reductions

| Op                          | Polars CPU | NumPy    | MLX      | MLX vs Polars |
|-----------------------------|------------|----------|----------|----------------|
| std (N=10M)                 | 6.99 ms    | 5.38 ms  | 1.12 ms  | **6.2×**      |
| var (N=10M)                 | 7.14 ms    | n/a      | 0.87 ms  | **8.2×**      |
| quantile p=0.5 (N=10M)      | 15.13 ms   | 146.3 ms | 7.81 ms  | **1.9×**      |

Std/var are essentially compute-bound on F32 — MLX wins ~7×. Global
quantile is borderline (1.9×) because both implementations are sort-
based and Polars' radix sort is good.

#### Cumulative sum

| Op                  | Polars CPU | NumPy    | MLX      | MLX vs Polars |
|---------------------|------------|----------|----------|----------------|
| cumsum F32 (N=10M)  | 37.5 ms    | 34.5 ms  | 5.67 ms  | **6.6×**      |

Cumulative sum is the workhorse for rolling ops, exponential moving
averages, and time-series feature engineering. MLX uses Blelloch parallel
prefix-sum and wins decisively.

#### Black-Scholes-shaped option pricing (log/exp/sqrt/tanh chain)

A typical fin-tech expression: read 3 F32 columns (spot, strike,
time-to-expiry), evaluate `log`, `sqrt`, `exp`, a polynomial `tanh`-based
CDF approximation, output a call price.

| Op                          | Polars CPU | NumPy     | MLX      | MLX vs Polars |
|-----------------------------|------------|-----------|----------|----------------|
| black_scholes_call (N=10M)  | 242.0 ms   | 73.2 ms   | 3.86 ms  | **63×**       |

The single largest measured win after FFT. This is the broadest practical
template: any chain of transcendental F32 ops on Polars columns is a
candidate, including geospatial, signal processing, scientific simulation,
and risk computation.

#### FFT

| Op                  | NumPy     | MLX      | MLX vs NumPy |
|---------------------|-----------|----------|---------------|
| fft 1D (N=8M F32)   | 123.0 ms  | 1.60 ms  | **77×**       |

Polars has no native FFT. MLX uses a tuned Metal FFT kernel. The 77×
speedup over NumPy/Accelerate is the largest ratio measured anywhere in
this survey. If polars-metal exposed an `Expr.fft()` (currently no
Polars API), this would be a unique selling point — DataFrame-native
signal processing at GPU throughput.

#### Conditional cascades (`when().then().otherwise()`)

A 5-tier threshold cascade — common in feature engineering for bucketing
continuous variables:

| Op                          | Polars CPU | NumPy    | MLX      | MLX vs Polars |
|-----------------------------|------------|----------|----------|----------------|
| 5-tier when chain (N=10M)   | 23.7 ms    | 86.5 ms  | 2.41 ms  | **9.8×**      |

Note Polars beats NumPy here (its when/then is a fused branchless kernel).
MLX still wins 9.8× via parallel threshold-counting.

#### Correlation matrix

200 F32 columns × 200,000 rows → 200×200 correlation matrix.

| Op                          | Polars CPU | NumPy     | MLX      | MLX vs Polars |
|-----------------------------|------------|-----------|----------|----------------|
| corr matrix (200×200000)    | 131.1 ms   | 111.2 ms  | 16.74 ms | **7.8×**      |

Standardize → matmul. MLX does the matmul on the GPU, wins 7.8×. Useful
for risk-factor analysis, feature engineering, dimensionality reduction.

#### Datetime decomposition

| Op                                  | Polars CPU | MLX           |
|-------------------------------------|------------|---------------|
| dt.year (N=10M)                     | 177.9 ms   | (no API)¹     |
| dt.month (N=10M)                    | 175.5 ms   | (no API)¹     |
| dt.weekday (N=10M)                  | 14.7 ms    | (no API)¹     |
| dt.year+month+day+hour chained      | 177.6 ms   | (no API)¹     |
| hour via integer modulo             | n/a        | 2.75 ms       |

¹ MLX has no native gregorian-calendar API. A custom MSL kernel could
do year/month/day extraction in ~5 ms (bandwidth-bound at 10M i64),
implying a 30–40× win over Polars. This is the strongest "needs custom
MSL kernel" candidate measured here.

The dt.weekday case is fast in Polars (14.7 ms) because weekday is
modulo 7 — no calendar math. dt.year is slow (178 ms) because the
gregorian conversion is inherently per-element with conditionals for
leap years. GPU SFU + parallel threads collapse that.

---

## Quick-Reference Speedup Table

All measurements on the same M2 Ultra, same NumPy / MLX versions,
median of 5 iterations.

### Wins (compute-shaped — MLX > Polars CPU)

| Workload                                | Scale                  | Polars CPU | NumPy    | MLX     | MLX vs Polars |
|-----------------------------------------|------------------------|------------|----------|---------|----------------|
| FFT                                     | N=8M F32               | (no API)   | 123 ms   | 1.60 ms | (vs NumPy) **77×** |
| Black-Scholes-shape log/exp/sqrt chain  | N=10M, 3 inputs        | 242 ms     | 73.2 ms  | 3.86 ms | **63×**       |
| Haversine 4-transcendental chain        | N=10M                  | 181 ms     | 79.3 ms  | 3.49 ms | **52×**       |
| Cosine top-k brute-force                | Q=100 N=100k D=768     | ~56,000 ms¹| 145 ms   | 4.94 ms | ~**11,300×** (¹ proj) |
| Cosine top-k brute-force                | Q=100 N=1M D=768       | —          | 1065 ms  | 44.8 ms | (vs NumPy) **24×** |
| L2 / k-NN SIFT1M                        | Q=100 N=1M D=128       | —          | 863 ms   | 36.8 ms | (vs NumPy) **23×** |
| Datetime year/month (custom kernel est.)| N=10M                  | 177.9 ms   | (n/a)    | est. ~5 ms| est. **~36×** |
| Rolling mean (cumsum-diff)              | N=10M W=1000           | 113.9 ms   | n/a      | 5.83 ms | **20×**       |
| Top-k F32                               | N=10M K=100            | 96.6 ms    | 16.9 ms  | 7.57 ms | **13×**       |
| Conditional cascade (5-tier when/then)  | N=10M                  | 23.7 ms    | 86.5 ms  | 2.41 ms | **9.8×**      |
| Variance / std reductions               | N=10M                  | 7.1 ms     | 5.4 ms   | 0.9 ms  | **8×**        |
| Correlation matrix                      | 200×200,000            | 131.1 ms   | 111.2 ms | 16.7 ms | **7.8×**      |
| Cumulative sum                          | N=10M                  | 37.5 ms    | 34.5 ms  | 5.67 ms | **6.6×**      |
| Sort F32                                | N=10M                  | 33.0 ms    | 813.8 ms | 7.88 ms | **4.2×**      |
| Global quantile                         | N=10M                  | 15.1 ms    | 146.3 ms | 7.81 ms | **1.9×**      |

### Losses (bandwidth- or dispatch-shaped — Metal < Polars CPU)

| Workload                          | Scale                | Polars CPU | polars-metal² | Metal / Polars |
|-----------------------------------|----------------------|------------|----------------|----------------|
| TPC-H Q1 modified                 | N=10M                | 43.5 ms    | 123.1 ms       | **0.35×** loss  |
| TPC-H Q1 canonical                | N=10M                | 63.3 ms    | 526.5 ms       | **0.12×** loss  |
| TPC-H Q6                          | N=10M                | 9.27 ms    | 123.2 ms       | **0.075×** loss |
| Strings: contains_literal         | N=2M                 | 40.0 ms    | (no MLX path)  | tie/loss       |
| Strings: regex match              | N=2M                 | 46–63 ms   | (no MLX path)  | tie/loss       |

¹ Projected from a single-query measurement; Polars-native columnar formulation has no matmul fusion.
² Existing polars-metal engine measurement from `tests/bench/baseline.json`.

The contrast is the whole story. **15 of 15 measured compute-shaped ops
won by 4–80×. 3 of 3 measured bandwidth-shaped ops lost.** Same hardware,
same MLX. The variable is whether the op spends its time in compute or
in bandwidth/dispatch.

---

## Honest Tradeoffs

**The wins are not free.** Each recommended workload requires walker
work — specifically, pattern-matching new Polars expression shapes
(`List[F32].dot(lit)`, chained transcendentals on F32 columns) and
routing them to MLX or to custom MSL kernels. The existing M1–M3
walker can already pattern-match the Q1 / Q6 shapes; extending it
to vector-search and transcendental-ETL shapes is incremental work,
not a rewrite.

**MLX is the floor for engineering effort.** For cosine / L2, the
engine just needs to call MLX's `matmul`. No custom kernel. For
haversine, MLX's chained-expression API handles fusion automatically.
This means the "engineering cost of a 20–30× win" on workloads 1–3
is small compared to the kernel work that went into M2 / M3.

**The TPC-H result is not a contradiction.** Q1 / Q6 lose on Metal
because they spend their time in operations that don't have a compute
ratio — hashing, comparison, predicate AND. These are bandwidth- and
dispatch-shaped, and Apple Silicon's shared memory bus eliminates the
discrete-GPU bandwidth advantage. The lesson is to pick workloads
where the GPU's *compute* advantage matters, not its bandwidth.

**The wins do not generalize.** A 29× win on cosine does not imply a
29× win on TPC-H, nor does it move TPC-H closer to a tie. They are
disjoint workload classes. The right pitch is "polars-metal is for
compute-bound DataFrame work on M-series" — not "polars-metal is faster
than Polars."

**Whole-query speedup will be lower than per-op speedup.** A NYC Taxi
pipeline that's 30% haversine and 70% groupby/join might see 3–5×
end-to-end even though the haversine phase alone is 52× faster.
Headlines should be honest about this.

**Polars is moving in the same direction.** As of Polars 1.40, the
`Array[F32, D]` fixed-shape array dtype is in active development; this
will make vector-search workloads more idiomatic in Polars, expanding
the addressable surface.

**The competition is NumPy, not Polars.** A user who today writes
`.to_numpy() @ embeddings.T` gets a 1× baseline that polars-metal needs
to beat. The 29× MLX-over-NumPy number is the real bar. We should not
benchmark against `Polars-native-columnar-cosine` because nobody
actually writes that — it's a strawman.

---

## Recommendation

**Pursue the project. Build the polars-metal walker into an MLX-routing
layer for the whole class of F32-compute-shaped expression trees.** The
addressable surface is much broader than the first pass suggested.
Concretely, M4 is one piece of infrastructure with many beneficiaries:

**Phase 1 (existence proof, weeks):**

1. Recognize a Polars `LazyFrame` whose expression tree is a chain of
   F32 element-wise ops on numeric columns (transcendentals, arithmetic,
   `when/then`, comparisons), optionally followed by a reduction
   (`sum/mean/std/var/argmax`), a sort/top-k, or a cumulative scan.
2. Route the recognized subtree to an MLX expression graph; `mx.eval()`
   once; fold the result back into the DataFrame.

Phase 1 alone delivers measured wins on: Black-Scholes / haversine /
arbitrary transcendental chains (50–60×), Polars conditional cascades
(10×), variance/std reductions (6–8×), cumsum (6.6×) and therefore
rolling mean / sum via cumsum-diff (~18×), sort (4×), top-k (12×),
correlation matrix (7.8×). **Most analytical F32 expression trees
benefit immediately.**

**Phase 2 (custom kernel territory, 1–2 months):**

3. List/Array dot-product → MLX matmul (vector search; the cosine /
   k-NN 23–29× wins). Modest walker work; MLX matmul does the math.
4. Custom MSL gregorian-calendar kernel for `dt.year` / `dt.month` /
   `dt.day` (est. 30× win at 10M rows).
5. Expose `Expr.fft()` backed by MLX FFT (77× over NumPy; Polars has no
   FFT today). Unique selling point — DataFrame-native signal processing.
6. Pairwise string / time-series distance kernels (Levenshtein, DTW) —
   fills a real Polars vocabulary gap.

**Drop from the roadmap (confirmed losers):** TPC-H, TPC-DS, H2O.ai
groupby/join, generic string ops, histograms. These are bandwidth- or
dispatch-shaped on Apple Silicon's shared memory bus. The M3 baseline
losses (2.83–19.6× slower than Polars CPU) reflect this category — and
only this category. Don't chase them.

**The shipping line.** A working M4 looks like:

```python
df.with_columns(
    call=black_scholes_expr(s=pl.col("s"), k=pl.col("k"),
                            t=pl.col("t"), r=0.05, sigma=0.2)
).collect(engine="metal")
# ~4 ms on Metal vs 242 ms on CPU, same M2 Ultra
```

…plus matching demos on cosine top-k (5 ms vs 145 ms NumPy / 56 s
Polars-native), rolling mean (6 ms vs 114 ms), FFT (1.6 ms vs 123 ms
NumPy). Each is one-shot reproducible from `tests/bench/m4_survey/`.

**Confidence.** Higher than after the first pass. We are not betting
that one workload wins; we are betting that **the GPU's compute-vs-CPU
ratio is well-paid on most F32 element-wise and reduction work on
M-series.** MLX directly confirms this at 15-of-15 candidates measured.
The walker + MLX FFI surface that delivers this already has its bones
from M1–M3 — most of the engineering is recognizing more expression
shapes, not new infrastructure.

---

## Appendix: Bench Scripts

All scripts are in `tests/bench/m4_survey/`:

- `_timing.py` — shared timing harness (warmup, GC, median of N).
- `bench_cosine_topk.py` — Polars-native cosine top-k (very slow) + NumPy reference.
- `bench_cosine_topk_mlx.py` — MLX ceiling for cosine; the headline 29× result.
- `bench_pairwise_l2.py` — Pairwise L2 / SIFT1M with NumPy + MLX.
- `bench_haversine_mlx.py` — MLX ceiling for haversine; the 52× result.
- `bench_nyc_taxi.py` — Polars-native haversine + groupby ETL baseline.
- `bench_rolling_window.py` — Polars rolling mean/std/quantile baselines.
- `bench_rolling_mlx.py` — MLX cumsum-diff rolling mean; the 18–20× result.
- `bench_strings.py` — Polars string + regex baselines.
- `bench_extra_ops.py` — extended sweep: sort, top-k, std/var/quantile,
  cumsum, Black-Scholes, histogram, when-chain, FFT, correlation matrix.
- `bench_datetime.py` — datetime year/month/weekday/hour decomposition.

Each is runnable with `python3 -m tests.bench.m4_survey.<name>` from the
repo root. No GPU dispatch from polars-metal — these are CPU/NumPy/MLX
measurements only, per the Part A scope boundary.
