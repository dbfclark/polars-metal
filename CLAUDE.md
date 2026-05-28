# polars-metal

A Metal-backed execution engine for [Polars](https://github.com/pola-rs/polars), targeting Apple Silicon. Plugs into Polars' GPU engine interface; users opt in via `df.collect(engine="metal")`. Not a fork, not a new dataframe library.

## Mission

Make compute-shaped DataFrame work on M-series faster than Polars CPU by an order of magnitude or more — for any expression tree dominated by F32 element-wise math, reductions, sort / scan, or matmul-shaped operations. Bridge the gap between `df.collect()` and writing MLX or Metal by hand.

The M4 survey (see `docs/m4-benchmark-survey.md`) measured 15-of-15 wins of 4–80× across this workload class on M2 Ultra: Black-Scholes-shape transcendental chains 63×, FFT 77×, haversine 52×, cosine top-k 29× vs NumPy, rolling mean via cumsum-diff 18×, sort 4×, top-k 12×, variance / std 6–8×, cumsum 6.6×, conditional cascades 10×, correlation matrix 7.8×. The TPC-H Q1 / Q6 losses (2.83–19.6× slower than Polars CPU) are a narrow and well-understood category — bandwidth-shaped queries on Apple Silicon's shared memory bus — and are not the bar.

Unified memory means zero host↔device copies, which we lean on for fast routing of MLX subgraphs back into Polars buffers. It does not give us GDDR6X-class bandwidth, which is why the cuDF-on-4090 framing was wrong (see Non-goals).

### Architectural principles that fall out of this

1. **Fuse the whole compute subtree into one MLX subgraph, not per-op.** When the walker recognizes a chain of F32 compute ops in the Polars expression tree (transcendentals, arithmetic, reductions, sort, cumulative scans, matmul), it builds a single `mlx::core::array` graph, `mx.eval()`s it once, and folds the result back. Per-op routing pays dispatch overhead per op, which is the same fragmentation trap that made TPC-H Q1 lose despite individually fast kernels. One subtree → one dispatch (or as close as MLX's own kernel-fusion lets us get).

2. **Every op around the fused compute subtree must be at CPU parity, not faster.** The 50× compute win only shows up in wall-clock if filter, scan, predicate-eval, HStack-CSE materialization, and result fold-back add zero penalty. If we shave 200 ms off a compute phase but pay 50 ms of FFI marshalling around it, the user sees 5× instead of 50×. Treat parity for non-compute ops as a hard requirement, not an optimization. Where Polars CPU is fast on a Polars-native op (filter, take, materialize), route to Polars CPU with zero-copy buffer hand-off — do not write a Metal version that ties.

3. **Compute-intensity detection drives routing, not op-by-op pattern matching.** The walker should look at the expression tree, decide "this subtree has enough compute density to warrant a Metal subgraph," and either fuse it whole or leave it alone. A two-op chain with one matmul is worth fusing; a 30-op chain that's all filter+take+cast is not. The trigger is FLOPs / row, not op identity.

## Non-goals

- **Not a standalone dataframe library.** No public `pl_metal.DataFrame`. The only user-facing surface is the engine plugin.
- **Not a Polars fork.** We register through the existing engine plugin mechanism. If we need a Polars change, upstream it.
- **Not a general MLX wrapper.** MLX is an implementation detail.
- **Not full op coverage in v0.1.** Unsupported ops fall back to CPU; this is correct behavior, not failure.
- **Not training/autograd.** This is an OLAP engine. MLX's autograd is irrelevant here.
- **Not trying to beat cuDF on TPC-H.** Bandwidth-shaped queries on Apple Silicon's shared memory bus have no discrete-GPU runway. cuDF on a 4090 wins TPC-H via 1 TB/s GDDR6X and 24 GB of dedicated VRAM — neither exists on M-series. We measured 2.83–19.6× losses vs Polars CPU on TPC-H Q1 / Q6 and will not close them. That is not the workload class we target.
- **Not investing in hash groupby / join past correctness.** The M2 / M3 groupby, sort, and join kernels stay as conformance code so that `engine="metal"` still produces correct results on those expression shapes — but we do not extend them. New shape support and new kernels go to compute-shaped ops (per Mission).
- **Not building string / regex kernels.** Variable-length data + divergent execution lose on GPU; Polars CPU is already fast. Fall back to CPU.

## Architecture

```
User Polars code
      │
      ▼
Polars logical plan ── optimizer ──▶ physical plan (IR)
                                         │
                                         ▼
                              MetalEngine::execute(node)
                                         │
                       ┌─────────────────┼─────────────────┐
                       ▼                 ▼                 ▼
                 MLX primitives    Custom MSL kernels   CPU fallback
                       │                 │                 │
                       └──────── Arrow-layout buffers ─────┘
                                  (zero-copy MTLBuffer)
```

Three layers:

1. **Engine adapter (Rust).** Pattern-matches Polars IR nodes, owns the physical plan walker, dispatches to kernels, and assembles results back into Polars `DataFrame`s. This is the bulk of the code.
2. **Kernel layer.** Thin Rust wrappers around MLX ops for the easy stuff (elementwise, reductions, matmul, numeric sort). Custom MSL files for the hard stuff (hash groupby, hash join, radix sort over arbitrary keys, string kernels, null-aware variants).
3. **Buffer bridge.** Arrow buffers ↔ `MTLBuffer` with zero-copy where alignment permits. Handles validity bitmaps, offset buffers, and dictionary columns.

## Key references (read these first)

Both cuDF and Polars are checked out locally under `references/` (gitignored, refreshed via `scripts/refresh-references.sh`). Read with `Read`/`Grep`/`Glob` — no need to WebFetch.

- **`references/cudf/python/cudf_polars/`** — the structural template. Read `_dask.py`, `dsl/`, and the IR-node dispatch before writing any engine code. Our adapter is shape-isomorphic to this.
- **`references/cudf/cpp/src/`** — CUDA kernel sources. Read the matching kernel before writing the MSL port (per "Working with Claude Code in this repo" below).
- **`references/polars/crates/polars-plan/`** — defines the IR node types we must handle. The `IR` enum is the spec.
- **`references/polars/crates/polars-core/src/chunked_array/`** — Polars' columnar layout. Our buffers must round-trip through this.
- **`references/candle/candle-metal-kernels/`** — HuggingFace's Metal-shader collection (binary ops, cast, fill, indexing, gemm, quantized, etc.). Read for MSL idioms and Apple-Silicon-specific patterns when authoring new shaders; the `src/metal_src/` directory has the `.metal` source files and `src/metal/` has Rust-side Metal API abstractions (buffer, command_buffer, compute_pipeline, etc.). Particularly useful where cuDF's CUDA reference doesn't translate cleanly to MSL.
- **MLX docs** — `ml-explore/mlx`, especially the C++ API and the `mlx::core::array` memory model. We mostly use the C++ side from Rust.
- **Apple Metal Shading Language spec** — for the custom kernels. The MSL 3.x feature set is our floor.
- **Apache Arrow columnar format spec** — the source of truth for null semantics and layout.

## Tech stack

- **Rust** for the engine, FFI to MLX (C++), and the Python wheel.
- **MSL (Metal Shading Language)** for custom kernels. One `.metal` file per kernel family; compile via `metal-rs` or `objc2-metal`.
- **MLX** (C++) for primitive ops. Linked via `cxx` or hand-written FFI.
- **PyO3 + maturin** for the Python wheel.
- **Polars** as a path dependency during development, pinned to a known-good rev. Track upstream's `plugin/engine` API carefully.

## Repo layout

```
crates/
  polars-metal-core/       # engine adapter, IR walker, plan dispatch
  polars-metal-kernels/    # Rust-side kernel wrappers
  polars-metal-mlx-sys/    # MLX FFI bindings
  polars-metal-buffer/     # Arrow ↔ MTLBuffer bridge
shaders/
  groupby.metal
  join.metal
  sort.metal
  string.metal
  ...
python/
  polars_metal/            # thin Python package, registers the engine
tests/
  conformance/             # runs Polars' own test suite through engine="metal"
  kernel/                  # per-kernel correctness tests
  bench/                   # criterion + pytest-benchmark
docs/
  architecture.md
  kernel-authoring.md
```

## Build / test / run

```bash
# build everything
make build

# build the Python wheel and install in editable mode
make wheel    # equivalent to: maturin develop --release

# unit tests
make test-unit

# kernel correctness tests (M0 is a placeholder; real kernels arrive in M1)
make test-kernel

# conformance: Polars-shaped tests with engine=MetalEngine()
make test-conformance

# differential: random small DataFrames, our engine vs CPU
make test-diff

# benchmarks
make bench

# lint (clippy, fmt, ruff)
make lint

# everything (the gate)
make gate

# refresh local reference clones (cuDF, Polars) — needed only after a rev bump
make refresh-refs
```

**Always run `make lint` (or `make gate` for the full check) before declaring a task done.** CI will reject otherwise (when CI exists; currently lint runs locally).

## Roadmap

### Shipped (M0–M3) — kept as conformance code

1. **Buffer bridge.** Zero-copy Arrow `Buffer` ↔ `MTLBuffer`, null bitmap handling, alignment. Shipped M0.
2. **Engine skeleton.** Register with Polars, accept a physical plan, walk it, fall back to CPU for everything. Shipped M1.
3. **Scan + project + filter.** Routed where it can be; falls back via `_filter_via_polars` for shapes the walker doesn't handle. Shipped M1.
4. **Elementwise + reductions (limited).** MLX passthroughs for a subset of F32 / F64 ops. Shipped M2; expanded in M3 walker.
5. **Hash groupby with sum/mean/count/min/max.** Shipped in M2 with PerAgg + fused F32 paths. Maintained as **conformance only**. Do not extend — the workload class loses to Polars CPU on M-series for the reasons in Mission / Non-goals.
6. **Radix sort on fixed-width keys.** Shipped partial in M3. Maintained as **conformance only**.
7. **Walker for canonical TPC-H Q1, Q6, modified Q1.** Shipped in M3 (Phases 0–14). Used for conformance gates only; not a perf bar.

### Now (M4) — the compute-intensity pivot

Build the walker and MLX-subgraph-fusion infrastructure that delivers wins on the compute-shaped workload class identified in `docs/m4-benchmark-survey.md`. The architectural principles in **Mission → Architectural principles** govern this work: fuse whole subtrees, not per-op; require CPU parity for surrounding ops; route based on compute intensity, not op identity.

Phases below are not strictly ordered — they share a single subtree-recognition + MLX-fusion infrastructure that all of them benefit from. Pick the order that hits a runnable, demoable example earliest.

8. **MLX subgraph fusion for F32 expression chains.** Recognize Polars expression trees of element-wise F32 ops (transcendentals, arithmetic, `when/then`, comparisons, casts) optionally terminated by a reduction (`sum/mean/std/var/argmax`), a sort, a top-k, or a cumulative scan. Build one `mlx::core::array` graph, eval once, fold back into a Polars Series. **This step is the foundation for nearly every M4 win.** Targets: Black-Scholes (63× measured), haversine (52×), variance/std (6–8×), conditional cascade (10×), sort (4×), top-k (12×), cumsum (6.6×).
9. **Cumsum-diff family.** Rolling mean / sum / variance via the `mean[i..i+W] = (cumsum[i+W] − cumsum[i]) / W` identity. Reuses the Phase 8 infrastructure. Target: 18–20× measured at all window sizes.
10. **List / Array dot-product → MLX matmul.** Pattern-match `List[F32].dot(lit)` / `Array[F32, D].dot(lit)` / equivalent `list.eval(...).list.sum()` shapes; emit a single matmul. Target: cosine top-k (29× vs NumPy, ~10,000× vs Polars-native) and L2 k-NN (23× vs NumPy). First custom Polars-expression-shape recognizer.
11. **`Expr.fft()` exposing MLX FFT.** New Polars vocabulary item (Polars has no native FFT). 77× vs NumPy. Unique selling point — DataFrame-native signal processing.
12. **Custom MSL gregorian-calendar kernel** for `dt.year` / `dt.month` / `dt.day`. Estimated 30–40× over Polars (`dt.year` is currently 178 ms at 10M rows). First brand-new MSL kernel of the M4 direction.
13. **Speculative: pairwise distance kernels** (Levenshtein, DTW). Fills a real Polars vocabulary gap; high compute density per pair. Custom MSL.

### Explicitly demoted / dropped

- ~~Hash join.~~ Bandwidth-shaped on M-series; deferred indefinitely unless a non-TPC-H workload demands it.
- ~~Window functions via partition-then-per-partition kernels.~~ Replaced by cumsum-diff (Phase 9) for the shapes that matter (rolling mean / sum / std / EMA). Skip the per-partition kernel design; it was sized for TPC-H Q4-shaped queries that aren't the target anymore.
- ~~Strings.~~ Dropped. CPU is fast; GPU loses on variable-length data.

## Conventions

- **Rust 2021, MSRV pinned in `rust-toolchain.toml`.** Don't bump casually.
- **No `unwrap()` in non-test code.** Use `Result` with `thiserror`-defined error types per crate.
- **No `unsafe` outside `*-sys` crates and the buffer bridge.** Both require a `// SAFETY:` comment explaining the invariant.
- **Errors propagate as `PolarsError::ComputeError` at the engine boundary** so they look native to Polars users.
- **Null semantics match Polars exactly.** When in doubt, write a test comparing engine output to CPU output on inputs containing nulls. Mismatches are bugs, even if "mathematically reasonable."
- **One MSL kernel per file.** Filename matches the entry point. Document threadgroup/grid assumptions at the top of the file.
- **Allocations:** prefer reusing scratch buffers via a per-query arena. The unified-memory model means there's no GPU OOM cliff, but fragmentation still hurts.
- **Don't introduce a new dependency without a written justification in the PR description.** Especially for the kernel and buffer crates — they should stay lean.

## Testing strategy

We have three free, high-quality test oracles. Use them.

1. **Polars' own test suite, run with `engine="metal"`.** This is the conformance bar. A test that passes on CPU and fails on Metal is a bug in this engine, full stop. Set up a CI job that runs the full Polars Python test suite with the engine flag flipped.
2. **Differential testing against CPU Polars.** For every kernel, generate random inputs (including null-heavy, empty, and single-row cases) and assert byte-exact equality with the CPU execution of the same plan. `proptest` for Rust-level, `hypothesis` for Python-level.
3. **cuDF-Polars output.** Where we have access to an NVIDIA box, run the same queries through cuDF-Polars and diff. Useful for catching cases where CPU Polars and we both agree but are both wrong about edge cases that the GPU world has already debugged.

Benchmarks are not tests. A perf regression is not a failing test; a correctness regression is. Keep them separate.

### Headline benchmark suite (M4 onward)

The numbers we ship and quote come from compute-shaped workloads, not TPC-H. The headline suite under `tests/bench/m4_survey/`:

- `bench_extra_ops.py::black_scholes_call` — F32 transcendental chain, 10M rows. The Phase 8 fusion target.
- `bench_cosine_topk_mlx.py` — vector search, Q=100 N=1M D=768. The Phase 10 target.
- `bench_haversine_mlx.py` — haversine 10M rows. The Phase 8 motivating case.
- `bench_rolling_mlx.py` — rolling_mean cumsum-diff. The Phase 9 target.
- `bench_extra_ops.py::fft` — 8M-point 1D FFT. The Phase 11 target.

TPC-H Q1 / Q6 benches stay in `tests/bench/` as **conformance gates only**. Their `_gate.ratio_lt` thresholds (currently 3.5×–22× depending on shape) catch correctness regressions and prevent us falling off a kernel-dispatch cliff. They do not represent the bar the project is trying to clear.

## Gotchas

- **MLX is dense-numeric-first.** It won't help you with strings, structs, lists, or null bitmaps. Don't try to force it.
- **Polars' IR is not stable across versions.** Pin the Polars rev in `Cargo.toml` and bump deliberately. Each bump may require adapter changes.
- **Threadgroup sizing is not portable across M1/M2/M3/M4.** Query `MTLDevice` capabilities at runtime; do not hardcode.
- **Hash-table sizing on GPU is awkward.** Two-pass (count then fill) is usually right; resist the urge to use atomics for the count.
- **Validity bitmaps are little-endian, bit-packed, Arrow-aligned.** Your MSL kernels need to load 8 rows of nulls at a time. Write a tested helper and reuse it everywhere.
- **`engine="metal"` must always produce a Polars DataFrame indistinguishable from the CPU result.** Same dtypes, same null positions, same row order where order is defined. No "approximately equal."
- **Don't write a new public API just because it would be convenient.** The engine plugin surface is the only surface.

## Working with Claude Code in this repo

- **When implementing a kernel:** read the matching cuDF CUDA kernel first (in `cudf/cpp/src/`). Don't reinvent the algorithm; the parallel-primitives literature is settled. Port it to MSL.
- **When touching the IR adapter:** look at how `cudf_polars` handles the same node type. If our handling diverges, document why.
- **Before writing new code:** check whether MLX already has the op. If yes, wrap it; do not write a custom kernel.
- **Prefer MLX subgraph fusion over per-op routing.** Whenever you find a chain of compute ops the walker can recognize, build it into a single MLX subgraph and `mx.eval()` once. Issuing the same ops as separate MLX calls or separate Metal dispatches pays the same fragmentation cost that lost TPC-H Q1. The shape to aim for is "one Polars subtree → one MLX expression graph → one MLX dispatch (or as close as MLX's own kernel fusion lets you get)."
- **When you find a compute-heavy subtree, push for CPU parity on what's around it.** Routing 50× faster compute into a pipeline that pays 50 ms of FFI marshalling per fold-back collapses the wall-clock win. If a surrounding op (filter, take, materialize, cast) is fast on Polars CPU, hand it back to Polars with zero-copy buffer pass-through — don't write a Metal version that ties. The win comes from making the compute phase fast, not from making everything Metal.
- **When in doubt about semantics:** write a tiny Polars CPU script that exercises the case, observe the output, then match it. Polars CPU is the spec.
- **Don't speculatively optimize.** Land a correct, slow version with tests first. Profile, then optimize.
- **Don't add files to `shaders/` without a corresponding test in `tests/kernel/`.**

## Open questions (track in `docs/open-questions.md`)

- Should the engine own its own thread/stream pool, or piggyback on Polars'?
- How do we expose `MTLCommandQueue` for advanced users without leaking Metal types into the Python API?
- Spill-to-CPU policy when a kernel's working set exceeds available unified memory (it can happen on 8/16GB machines).
- Do we support `LazyFrame.profile()` natively, or rely on Polars' built-in?
