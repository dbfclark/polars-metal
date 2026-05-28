# M4 Benchmark Survey: Compute-Bound Workloads That Could Win on M2 Ultra

**Author:** Claude (Opus 4.7)
**Date:** 2026-05-28
**Branch:** `m3-realworkload`
**Status:** Research deliverable — Part A of M4 strategic investigation
**Hardware:** Apple M2 Ultra, 24 CPU cores (16P + 8E), 60-core GPU, 128 GB unified memory
**Stack:** Polars 1.40.1, MLX 0.25.1, NumPy 1.26.2

---

## TL;DR

**The project has a point.** There is a concrete class of DataFrame-shaped
workloads where Metal beats CPU on M2 Ultra by **20–50×** — not 1.1×, not
a wash. The clean existence proof is **brute-force pairwise vector
similarity** (cosine / L2 top-k); the broadest-appeal candidate is
**transcendental-heavy ETL** (haversine, datetime decomposition,
geospatial transforms). Both ride on M2 Ultra GPU's hardware Special
Function Units and ~21 TFLOPS F32 peak (60-core bin; ~27 TFLOPS on the
76-core bin) vs CPU NEON's ~450 GFLOPS — a 45–60× compute ratio that
the workload class is selected to expose.

The TPC-H result (Metal 2.87–19.6× *slower*) is not a contradiction: TPC-H
Q1 / Q6 spend their time in hash-aggregate and predicate-AND, both
bandwidth-shaped *and* dispatch-shaped operations where Apple Silicon's
shared memory bus removes the discrete-GPU advantage. The lesson is not
"Metal can't win on Apple Silicon" — it's "Metal can't win on
*bandwidth-shaped* workloads on Apple Silicon."

**Recommended M4 focus (ordered by confidence):**

1. **Brute-force vector search (cosine + L2 top-k) over embedding columns** —
   29× win measured (MLX 4.94 ms vs NumPy 145 ms at Q=100, N=100k, D=768).
   The existence proof. Custom-kernel work is minimal; MLX matmul does
   the job.
2. **Transcendental-heavy ETL (NYC Taxi-style haversine, datetime
   decomposition)** — 52× win measured against Polars CPU
   (3.49 ms MLX vs 181 ms Polars at N=10M). Broader workload appeal;
   dispatch-overhead-sensitive but the per-op latency gap is huge.
3. **(Speculative) Pairwise / windowed string distance (Levenshtein,
   Jaro-Winkler, DTW) over a string column vs a query string** — high
   compute density per pair, no good Polars CPU primitive, fertile ground
   for a custom MSL kernel. No CPU baseline measured (Polars has no
   built-in); inferred from pairwise-similarity math.

**Workloads explicitly NOT recommended:** H2O.ai db-benchmark, TPC-DS,
generic groupby/join (all bandwidth-shaped). MLPerf DLRM (not actually
DataFrame work). Polars rolling-window aggregates (already well-tuned
on CPU; gap too small).

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

### 8. Polars rolling-window aggregates (mean, std, quantile)  — NOT WORTH PURSUING

**What it is.** Sliding-window mean / std / median / quantile over
time-series. Polars has well-tuned CPU implementations using ring
buffers and incremental order statistics.

**Polars CPU baseline (measured) at N=10M.**

| Op                            | Window | Polars CPU ms |
|-------------------------------|--------|---------------|
| rolling_mean                  | 100    | 112.0         |
| rolling_mean                  | 1,000  | 114.9         |
| rolling_mean                  | 10,000 | 114.6         |
| rolling_quantile p50          | 100    | 416.9         |
| rolling_quantile p50          | 1,000  | 435.9         |
| rolling_std                   | 100    | 241.1         |
| rolling_std                   | 1,000  | 239.9         |

**Compute density.** rolling_mean: incremental — ~4 ops per output, ~1
op/byte. **Bandwidth-bound.** rolling_quantile: insertion sort within a
ring of W → O(log W) per output, ~9 ops/byte at W=1000. Borderline.

**Metal-win prediction.** rolling_mean is already at ~350 MB/s read
throughput on Polars CPU — close to the bandwidth ceiling for cache-
resident data. No GPU runway. rolling_quantile *could* win on a custom
kernel that uses simdgroup-sort for within-window sorting, but estimated
2–4× speedup, with substantial engineering. Marginal.

**Verdict.** **Skip.** Polars CPU is already near the ceiling for the
shape of these workloads. The engineering ROI is too low.

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

---

## Quick-Reference Speedup Table

All measurements on the same M2 Ultra, same NumPy / MLX versions,
median of 5 iterations.

| Workload                          | Scale                | Polars CPU | NumPy   | MLX     | MLX vs NumPy | MLX vs Polars |
|-----------------------------------|----------------------|------------|---------|---------|--------------|----------------|
| Cosine top-k                      | Q=100, N=100k, D=768 | ~56,000 ms¹ | 145 ms  | 4.94 ms | **29.4×**    | ~11,300×       |
| Cosine top-k                      | Q=100, N=1M, D=768   | —          | 1065 ms | 44.8 ms | **23.8×**    | —              |
| L2 k-NN                           | Q=100, N=100k, D=128 | —          | 110.2 ms| 3.96 ms | **27.8×**    | —              |
| L2 k-NN                           | Q=100, N=1M, D=128   | —          | 863 ms  | 36.8 ms | **23.4×**    | —              |
| Haversine (4 transcendentals/row) | N=10M                | 181 ms     | 79.3 ms | 3.49 ms | **22.7×**    | **51.9×**      |
| **— vs. —**                       |                      |            |         |         |              |                |
| TPC-H Q1 modified (baseline)      | N=10M                | 43.5 ms    | n/a     | (polars-metal 123 ms²) | n/a | **0.35×** (loss) |
| TPC-H Q1 canonical (baseline)     | N=10M                | 63.3 ms    | n/a     | (polars-metal 527 ms²) | n/a | **0.12×** (loss) |
| TPC-H Q6 (baseline)               | N=10M                | 9.27 ms    | n/a     | (polars-metal 123 ms²) | n/a | **0.075×** (loss)|

¹ Projected from a single-query measurement; Polars-native columnar formulation has no matmul fusion so scales linearly with Q.
² Existing polars-metal engine measurement from `tests/bench/baseline.json`, not MLX direct.

The contrast is the whole story. **Compute-bound shapes (top half):
~25× wins. Bandwidth-bound shapes (bottom half): 3–13× losses.** Same
hardware. Same MLX. The variable is the workload's compute-to-bandwidth
ratio.

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

**Pursue the project, focused on the compute-bound workload class.**
Build out M4 around three workloads:

1. **Brute-force pairwise vector similarity (cosine + L2 top-k) over
   `List[F32]` / `Array[F32, D]` columns.** This is the existence proof.
   Engineering: pattern-match `list.dot(lit)`, `list.eval(self * lit)
   .list.sum()`, and the `Array.matmul` form (when available); route to
   MLX `matmul`. Add `argpartition` / top-k as a follow-up kernel.
2. **Transcendental-heavy expression chains on F32 columns
   (haversine, geospatial, signal processing).** Pattern-match chains of
   `sin/cos/sqrt/exp/log/arcsin/arctan` on F32 columns; build an MLX
   expression graph; eval once; fold back. The walker work is recognizing
   the chain shape and avoiding intermediate materialization.
3. **(Phase 2 / speculative)** Pairwise string / time-series distance
   kernels (Levenshtein, Jaro-Winkler, DTW) — fill the gap in Polars'
   expression vocabulary while delivering big speedups.

**Drop bandwidth-shaped workloads from the roadmap.** TPC-H, TPC-DS,
H2O.ai-style groupby/join, rolling-window aggregates — these are
hardware-bound on Apple Silicon and chasing them is throwing engineering
at a ceiling. The lesson from M3 is that the prior roadmap (M2 = groupby,
M3 = sort + join) optimized for the wrong workload class. M4 should
correct course.

**The shipping line.** A working M4 deliverable looks like:
"`pl.DataFrame({'emb': ...}).with_columns(sim=pl.col('emb').list.dot(query))
.top_k(10, 'sim').collect(engine='metal')` runs in 5 ms where CPU
takes ~50 seconds and `.to_numpy()` takes ~150 ms." That is a real
existence proof for the project's thesis, on a workload users actually
have, with a number nobody can dispute.

**If we don't get there, the project still doesn't have a point.** But
we should get there: the math says we will, and the MLX measurements
say we will, and the engineering work to do so is much smaller than what
went into M2 or M3.

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
- `bench_strings.py` — Polars string + regex baselines.

Each is runnable with `python3 -m tests.bench.m4_survey.<name>` from the
repo root. No GPU dispatch from polars-metal — these are CPU/NumPy/MLX
measurements only, per the Part A scope boundary.
