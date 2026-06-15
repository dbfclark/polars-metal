# polars-metal — crossing-tax benchmark (M9)

> Internal decision-input. Sizes the CPU<->GPU crossing tax on mixed compute+join
> pipelines. Ratios are vs the all-CPU path (what `engine="metal"` does today on a join).

## Environment
- **machine**: arm64
- **platform**: macOS-26.2-arm64-arm-64bit
- **polars_version**: 1.40.1
- **mlx_version**: 0.25.1

## Crossing cost model

`crossing_ms ≈ alpha · bytes_crossed + beta · n_crossings`

- **alpha** = 4.960e-08 ms/byte (≈ 20.2 GB/s round-trip)
- **beta** = 0.0016 ms/crossing (fixed dispatch/sync)

## retrieve_rerank  (gather)

| size | path | ms | × vs all_cpu |
|---:|---|---:|---:|
| 1,000 | all_cpu | 466.59 | 1.00× |
| 1,000 | partial_naive | 376.15 | 1.24× |
| 1,000 | partial_smart | 20.41 | 22.87× |
| 1,000 | resident | 20.58 | 22.67× |
| 10,000 | all_cpu | 4078.65 | 1.00× |
| 10,000 | partial_naive | 3719.61 | 1.10× |
| 10,000 | partial_smart | 134.17 | 30.40× |
| 10,000 | resident | 134.25 | 30.38× |

## fact_dim_chain  (gather)

| size | path | ms | × vs all_cpu |
|---:|---|---:|---:|
| 1,000,000 | all_cpu | 4.67 | 1.00× |
| 1,000,000 | partial_naive | 4.99 | 0.94× |
| 1,000,000 | partial_smart | 2.16 | 2.17× |
| 1,000,000 | resident | 1.39 | 3.37× |
| 10,000,000 | all_cpu | 50.31 | 1.00× |
| 10,000,000 | partial_naive | 46.19 | 1.09× |
| 10,000,000 | partial_smart | 12.07 | 4.17× |
| 10,000,000 | resident | 6.11 | 8.24× |

## asof_compute  (asof)

| size | path | ms | × vs all_cpu |
|---:|---|---:|---:|
| 1,000,000 | all_cpu | 69.55 | 1.00× |
| 1,000,000 | partial_naive | 70.66 | 0.98× |
| 1,000,000 | partial_smart | 70.68 | 0.98× |
| 10,000,000 | all_cpu | 822.13 | 1.00× |
| 10,000,000 | partial_naive | 823.83 | 1.00× |
| 10,000,000 | partial_smart | 820.90 | 1.00× |

## hashjoin_compute  (hash)

| size | path | ms | × vs all_cpu |
|---:|---|---:|---:|
| 1,000,000 | all_cpu | 83.11 | 1.00× |
| 1,000,000 | partial_naive | 81.46 | 1.02× |
| 1,000,000 | partial_smart | 81.11 | 1.02× |
| 10,000,000 | all_cpu | 992.07 | 1.00× |
| 10,000,000 | partial_naive | 998.10 | 0.99× |
| 10,000,000 | partial_smart | 1018.62 | 0.97× |

## Verdict

### Per-pipeline read
- **P1 retrieve_rerank (gather):** partial_smart **22.9× / 30.4×** vs all_cpu (Q=1k / 10k); resident **22.7× / 30.4×** — a statistical tie with partial_smart. partial_naive only **1.24× / 1.10×** because it crosses the full Q×N similarity matrix back to CPU, spending the whole α·bytes budget on the crossing and erasing the GPU matmul win. **The lever is pushing the top-k reducer to the GPU side of the crossing (cross Q×k, not Q×N), not keeping the tail resident** — once the data is reduced, the post-reduce gather+rerank is tiny and CPU vs GPU placement is a wash (smart ≈ resident).
- **P2 fact_dim_chain (gather):** partial_naive **0.94× / 1.09×** (no win — crosses the full N gathered column, runs the chain on CPU); partial_smart **2.17× / 4.17×** (CPU gather, cross 2 F32 cols, transcendental chain on GPU); resident **3.37× / 8.24×**. **Here resident genuinely beats partial_smart (~2× at N=10M)** — keeping the gather on-GPU (`mx.take`) avoids the cache-hostile CPU scatter over N rows, which is the dominant non-compute cost once the chain is on the GPU.
- **P3 asof_compute (sort-merge):** all paths **0.98×–1.00×** — a dead heat. The as-of match (`searchsorted`) is CPU-irreducible and dominates wall-clock; the only GPU-able step is a single `sqrt(x²+y²)`, whose speedup is exactly cancelled by the crossing tax. **CPU-match + GPU-compute does not beat all-CPU when the post-join compute is light.**
- **P4 hashjoin_compute (hash):** all paths **0.97×–1.02×** — confirms partial-GPU ≤ all_cpu when the join is the work. The hash build+probe is wholly CPU (no GPU path) and dominates; the trivial elementwise post-compute can't amortize even a near-free crossing, and at N=10M the extra round-trip drags partial_smart slightly *below* all_cpu (0.97×).

### The cost-model rule
With **α = 4.96e-8 ms/byte (≈ 20.2 GB/s round-trip)** and **β = 0.0016 ms/crossing**, crossings are *cheap*: the fixed per-crossing cost is negligible (β ≈ 1.6 µs), and moving a full N=10M F32 column (40 MB) costs only **α·40e6 ≈ 2 ms** each way. Free per-op routing wins when the GPU compute it enables saves more than that tax:

  `GPU_compute_saved  >  α · bytes_crossed + β · n_crossings`

Two corollaries the data makes concrete:
1. **Cross reduced data, never the full intermediate.** P1 proves it: crossing the Q×N matrix (partial_naive) blows the entire α·bytes budget → 1.1×; pushing the top-k first so only Q×k crosses → 30×. The reducer must sit on the GPU side of the boundary.
2. **You need real compute density to clear the ~2 ms/full-column tax.** A transcendental chain (P2, ~50 ms of CPU work at 10M) clears it easily; a lone `sqrt` (P3/P4) does not. Bandwidth-shaped or CPU-irreducible work (the as-of match, the hash join) never clears it regardless of how cheap the crossing is.

So the binding constraint is **not** the crossing cost (switching is cheap, as M9 set out to test) — it is (a) reducer placement and (b) compute density. Routing never *hurts* materially (join pipelines tie at ~1.0×), but it only *helps* on the gather→compute family.

### Decision: **(b)** — M10 = narrower resident-gather build, not a general router.

The evidence:
- **The only family with any GPU win is gather→compute (P1, P2).** The two join families (P3 asof, P4 hash) tie all-CPU at ~1.0× under every crossing strategy — the join/match is CPU-irreducible and the surrounding compute is too light to amortize. A general boundary-aware per-op router (outcome **(a)**) would spend its complexity adjudicating exactly these join boundaries, for zero measured return.
- **On the gather family, the resident path is the one to build.** It is never worse than partial_smart (P1: tie at ~30×) and up to ~2× better (P2: 8.24× vs 4.17× at 10M), because folding the gather (`mx.take`) into the resident GPU subtree eliminates the CPU scatter that otherwise dominates. It is also *simpler* than per-op crossing-cost routing: recognize a `gather → F32-compute` subtree, keep it resident, fold back once.
- Outcome **(c)** (drop) is ruled out — partial_smart/resident beat all-CPU by 2–30× on both gather pipelines, a real and sizeable win. Outcome **(a)** is over-scoped for what the numbers support.

**M10:** extend the resident-subtree fusion to cover a `gather`/dense-`take` feeding an F32 compute chain or a top-k+rerank, keeping the gather and everything downstream of it on the GPU and crossing only the reduced result. Do **not** build a general per-op CPU↔GPU router, and do **not** build GPU joins — the as-of and hash pipelines show the join boundary is where the win dies, not where a crossing-cost router would rescue it.

