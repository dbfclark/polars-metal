# polars-metal

A Metal-backed execution engine for [Polars](https://github.com/pola-rs/polars), targeting Apple Silicon. Plugs into Polars' GPU engine interface; users opt in via `df.collect(engine="metal")`. Not a fork, not a new dataframe library.

## Mission

Match cuDF-Polars-on-a-4090 performance for realistic analytical workloads on an M-series Mac, by routing Polars' physical plan to MLX ops and custom MSL kernels where MLX is insufficient. Unified memory means zero host↔device copies; design around that advantage from day one.

## Non-goals

- **Not a standalone dataframe library.** No public `pl_metal.DataFrame`. The only user-facing surface is the engine plugin.
- **Not a Polars fork.** We register through the existing engine plugin mechanism. If we need a Polars change, upstream it.
- **Not a general MLX wrapper.** MLX is an implementation detail.
- **Not full op coverage in v0.1.** Unsupported ops fall back to CPU; this is correct behavior, not failure.
- **Not training/autograd.** This is an OLAP engine. MLX's autograd is irrelevant here.

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

## Roadmap (work in this order)

The order is chosen so each step produces a runnable engine, and so that the hardest design decisions (nulls, strings, memory ownership) get locked in early on the simplest cases.

1. **Buffer bridge.** Zero-copy Arrow `Buffer` ↔ `MTLBuffer`, null bitmap handling, alignment. No kernels yet.
2. **Engine skeleton.** Register with Polars, accept a physical plan, walk it, fall back to CPU for everything. Verify end-to-end with `engine="metal"` returning correct results on simple queries (via CPU fallback).
3. **Scan + project + filter.** First real GPU path. Filter is the canonical null-aware test case.
4. **Elementwise + reductions.** Mostly MLX passthroughs. Confirm null propagation matches Polars exactly.
5. **Hash groupby with sum/mean/count/min/max.** First custom MSL kernel. This is where the project becomes interesting.
6. **Radix sort on fixed-width keys.** Foundation for sort, sort-merge join, and window functions.
7. **Hash join (inner, then left/outer).** Build side on GPU, probe side streamed.
8. **Window functions (`.over()`).** Partition by hash, then per-partition kernels. Directly relevant to time-series and panel workloads.
9. **Strings.** Custom kernels for offsets + data buffers. Comparison, hashing, contains/starts-with, length. Defer regex.
10. **Everything else** as user demand dictates.

Don't skip ahead. Each stage's correctness tests are prerequisites for the next.

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
- **When in doubt about semantics:** write a tiny Polars CPU script that exercises the case, observe the output, then match it. Polars CPU is the spec.
- **Don't speculatively optimize.** Land a correct, slow version with tests first. Profile, then optimize.
- **Don't add files to `shaders/` without a corresponding test in `tests/kernel/`.**

## Open questions (track in `docs/open-questions.md`)

- Should the engine own its own thread/stream pool, or piggyback on Polars'?
- How do we expose `MTLCommandQueue` for advanced users without leaking Metal types into the Python API?
- Spill-to-CPU policy when a kernel's working set exceeds available unified memory (it can happen on 8/16GB machines).
- Do we support `LazyFrame.profile()` natively, or rely on Polars' built-in?
