# M0 — end-to-end skeleton with CPU fallback

**Status.** Draft, approved 2026-05-19.
**Master plan.** [`2026-05-19-master-plan-design.md`](2026-05-19-master-plan-design.md). This document fulfills M0's slot.
**Scope.** The design for milestone M0 — not its implementation plan. The implementation plan is the next artifact, produced from this spec.

## Deliverable

After M0 ships:

- `import polars_metal` succeeds; the import wires up the `MetalEngine` config object and monkey-patches `polars.lazyframe.frame._gpu_engine_callback` to route `isinstance(engine, MetalEngine)` instances to our callback.
- `df.collect(engine=polars_metal.MetalEngine())` returns results indistinguishable from `df.collect()` (CPU) on any query Polars itself can run. Mechanism: our callback walks the IR, recognizes nothing as GPU-supportable, returns without calling `nt.set_udf(...)`, and Polars' CPU executor takes over.
- The wheel builds via `maturin develop` and on a fresh `pip install`; the `polars_metal._native` Rust extension loads.
- The buffer bridge (`polars-metal-buffer`) and MLX FFI (`polars-metal-mlx-sys`) crates exist as separate units with their own correctness tests, but are **not yet used by the engine path**. They are proven in isolation so M1 can wire them in.
- The conformance harness runs and reports green: `pytest tests/conformance -k "not skip_metal"` (Polars' own test suite via `engine=MetalEngine()`) passes.
- All master-plan gates run as local commands; `Makefile` exposes them under short names.

## Out of scope for M0

- Any GPU code path being exercised by a real query.
- Any MSL kernel.
- Any kernel wrapper in `polars-metal-kernels`.
- Any IR-node-specific handler in the walker beyond "fall back."
- Spill-to-CPU policy, perf tuning, multi-Mac portability work beyond the gate.

## Why this slice

M0 forces every cross-cutting decision (entry point, walker shape, FFI strategy, error propagation, build/test infrastructure, alignment policy) to be made for real, with real tests, before any kernel work. Kernel work is the expensive code; deciding what shape it plugs into is cheap by comparison and easier to redo if we get it wrong while there's nothing yet plugged in.

## Architecture

### Repo layout at end of M0

```
polars-metal/
├── crates/
│   ├── polars-metal-buffer/        # Arrow ↔ MTLBuffer bridge, null-bitmap helpers
│   ├── polars-metal-mlx-sys/       # FFI bindings to MLX C++
│   ├── polars-metal-kernels/       # (empty scaffolding)
│   └── polars-metal-core/          # PyO3 extension: engine entry point + per-op dispatch
├── python/polars_metal/
│   ├── __init__.py                 # exports MetalEngine; monkey-patches Polars on import
│   ├── _engine.py                  # MetalEngine config dataclass
│   ├── _callback.py                # execute_with_metal — the IR walker
│   └── _native/                    # the compiled Rust extension lives here
├── shaders/                        # (empty)
├── tests/
│   ├── conformance/                # Polars test suite via engine=MetalEngine()
│   ├── kernel/                     # buffer-bridge + mlx-sys correctness tests
│   ├── diff/                       # hypothesis end-to-end differential
│   └── bench/                      # (criterion + pytest-benchmark scaffolding)
├── Makefile                        # one entry point per gate
└── docs/                           # design specs, architecture, open-questions
```

### Data flow at M0

```
df.collect(engine=MetalEngine())
    └─> Polars' optimizer + plan builder (CPU code, unchanged)
        └─> _gpu_engine_callback (monkey-patched) sees MetalEngine instance
            └─> polars_metal._callback.execute_with_metal(nt, ...)
                ├─> Walks the IR via nt.view_current_node() etc.
                ├─> No node is GPU-supportable (M0)
                └─> Returns without calling nt.set_udf(...)
        └─> Polars' CPU executor runs the (unchanged) IR
```

### Crate roles

- **`polars-metal-buffer`** and **`polars-metal-mlx-sys`** are pure Rust crates with their own correctness tests; no PyO3, no Python dependency, no engine dependency. Can be unit-tested via `cargo test` without `maturin develop`.
- **`polars-metal-core`** is the only crate that depends on PyO3 and produces the `_native` extension module. It wraps `polars-metal-buffer` + `polars-metal-mlx-sys` behind a Python-callable API surface, defines the engine-level error type, and hosts the (stubbed) `ScratchArena`.
- **`polars-metal-kernels`** exists in M0 as scaffolding only (`Cargo.toml` + empty `lib.rs`). Landing the crate now means M1's plan doesn't include a "create new crate" task — M1 just adds kernels.

### Python-side shape

The walker is a thin Python module (`_callback.py`) using `singledispatch` on Polars IR node types via `nt.view_current_node()`. In M0 the dispatch has no registered handlers — every node hits the default `unhandled` path, the walker exits without `set_udf`, and CPU takes over. M1 extends this by registering a handler for `Filter`.

`MetalEngine` is a small Python dataclass mirroring `GPUEngine`'s shape:

- `device: int | None = None` — index into available Metal devices; default uses `MTLCreateSystemDefaultDevice()`.
- `debug: bool = False` — verbose dispatch logging.
- Reserved fields for future kernels (tolerances, threadgroup overrides) are not present yet; YAGNI.

### Engine registration mechanics

`polars_metal/__init__.py` performs the monkey-patch at import time:

1. Read `polars.lazyframe.frame._gpu_engine_callback` and verify its signature matches what M0 was developed against (defensive check; fails loudly with a clear error if Polars refactored the call site).
2. Replace it with a wrapper that recognizes `isinstance(engine, MetalEngine)` and dispatches to `polars_metal._callback.execute_with_metal`. All other engine values flow to the original implementation unchanged.
3. The wrapper is idempotent — re-importing `polars_metal` does not double-patch.

## M0-defining decisions

### MLX FFI strategy: commit to `cxx`

The master plan called for a spike. Reconsidering at the M0-spec level: M0's MLX surface is one trivial op (`mlx::core::add` on a small `mlx::core::array`), switching cost is bounded by ~tens of lines, and `cxx` is the strong prior for C++-shaped APIs. We commit to `cxx` for M0 and reserve the right to switch to hand-written C shim + `bindgen` by M2 if friction shows up. The wrong call here costs days, not weeks.

### Polars rev pin

`9d8a77e9569779550405fd6ce7fecefcf58f5ca4` (today's `pola-rs/polars` main, same commit as `references/polars/`). Pinned in `Cargo.toml`. `scripts/refresh-references.sh` stays in sync. Bump policy from the master plan applies: only at milestone boundaries.

### Buffer-bridge alignment policy

Two regimes inside `polars-metal-buffer`:

- **Zero-copy (preferred).** When an Arrow `Buffer`'s underlying pointer is page-aligned and length is page-aligned, wrap it with `MTLDevice.makeBuffer(bytesNoCopy:length:options:deallocator:)`. The deallocator holds an `Arc` to the Arrow buffer so refcount semantics are preserved.
- **Copy fallback.** Otherwise, allocate via `MTLDevice.makeBuffer(length:options:)` and `memcpy` in.

M0's tests exercise both regimes deliberately by feeding the bridge an aligned input and a deliberately misaligned input. We do not attempt to influence Arrow allocations to land page-aligned in M0; that's a perf optimization for later if profiling shows it matters.

### Error propagation shape

Three `thiserror`-based error types, one per crate:

- `polars_metal_buffer::BufferError` — alignment failures, validity-bitmap shape errors, `MTLBuffer` allocation failures.
- `polars_metal_mlx_sys::FfiError` — MLX exceptions caught at the `cxx` boundary and converted to a Rust error variant; pointer/lifetime errors.
- `polars_metal_core::EngineError` — wraps the above via `#[from]`; this is the only error type the engine-boundary code surfaces.

At the PyO3 boundary, `EngineError` converts to `PolarsError::ComputeError(format!("polars-metal: {e}"))` via `impl From<EngineError> for PyErr`. Per CLAUDE.md, that's the conversion that makes errors look native to Polars users.

### Per-query arena strategy (deferred scaffolding only)

No kernels run in M0, so the arena is unused at runtime. We do define the trait surface so M1+ kernel wrappers can be written against it without rework:

```rust
pub trait ScratchArena {
    fn reserve(&mut self, bytes: usize) -> Result<MTLBuffer, EngineError>;
    fn reset(&mut self);
}
```

M0's implementation is a stub: `reserve` always allocates a fresh `MTLBuffer`, `reset` is a no-op. M1+ replaces the stub with a free-list-by-size-class.

## Testing strategy

Five layers, with one Makefile target per gate. Total M0 test code is mostly Layer 1 (heavy proptest on the buffer bridge); higher layers are scaffolded so M1 inherits them ready-to-use.

### Layer 1 — Per-crate Rust proptest (heavily populated in M0)

- **`polars-metal-buffer`.** Alignment math, null-bitmap helpers (load/store 8 rows at a time), offset-buffer round-trip, dictionary-column round-trip. Aggressive `proptest` coverage on null-bitmap helpers — random row counts, null densities, alignment offsets — validating bit-exact round-trip through `MTLBuffer`. The master plan flags this as the bug source that every subsequent kernel inherits if we get it wrong, so this gets the heaviest test investment in M0.
- **`polars-metal-mlx-sys`.** One trivial-op proptest: random `f32` arrays of varying length through `mlx_add`, compared against a Rust reference (`a + b`). Catches FFI lifetime/marshalling bugs and confirms the build/runtime-link path works end to end.
- **`polars-metal-core`.** Small unit tests for `EngineError` conversion to `PolarsError::ComputeError`, and that the `ScratchArena` stub returns valid buffers and `reset()` is a no-op.

### Layer 1.5 — Kernel-level Rust differential (scaffolded but empty in M0)

Compares each GPU kernel's output against a CPU reference implementation written in Rust. End to end at the kernel level, no Polars and no Python in the loop. This is where the bulk of kernel-correctness verification lives once kernels exist. M0 ships the harness shape (a `tests/kernel/` directory and one placeholder test that confirms `cargo test -p polars-metal-kernels` runs); first real differential test arrives with M1's filter kernel.

### Layer 2 — Python integration (handful of tests)

- Wheel builds via `maturin develop`; `import polars_metal` succeeds.
- The monkey-patch installs cleanly, asserts on the expected `_gpu_engine_callback` signature, and is idempotent across re-imports.
- `MetalEngine()` constructs; passing one to `df.collect(engine=...)` does not raise.
- The callback returns without calling `set_udf` on a trivial plan (verified via a mock `NodeTraverser`).

### Layer 3 — Conformance (Polars' own suite via `engine=MetalEngine()`)

`pytest tests/conformance -k "not skip_metal"`. A pytest module re-runs Polars' Python test suite with `engine=MetalEngine()` substituted in. The skip/xfail registry (`tests/conformance/_skips.toml`) starts empty in M0 — everything should pass via fallback. A test passing on `engine="cpu"` but failing on `engine=MetalEngine()` is a hard fail and is M0's most informative correctness signal.

### Layer 4 — End-to-end differential (Python hypothesis, scaffolded)

`tests/diff/` with three properties (numeric, string, null-heavy): random small DataFrames driven through `df.collect(engine=MetalEngine())` vs `engine="cpu"`. M0 verifies fallback parity (trivially identical because fallback IS CPU); real coverage starts at M1. Hundreds of cases, not millions — this is the slower end-to-end layer that complements the fast Rust proptest workhorse.

### Gate command tooling — `Makefile`

One entry point per gate; each is a one-liner the user (and Claude) can run before declaring work done:

```makefile
make build              # cargo build --workspace --release
make wheel              # maturin develop --release  (pyproject.toml at root points to crates/polars-metal-core/)
make test-unit          # cargo test --workspace
make test-kernel        # cargo test -p polars-metal-buffer -p polars-metal-mlx-sys --features=gpu-tests
make test-conformance   # pytest tests/conformance -k "not skip_metal"
make test-diff          # pytest tests/diff
make bench              # cargo bench && pytest tests/bench --benchmark-only
make lint               # cargo clippy --workspace --all-targets -- -D warnings \
                        # && cargo fmt --check \
                        # && ruff check python/ \
                        # && ruff format --check python/
make gate               # everything above; "is M0 done" check
make refresh-refs       # scripts/refresh-references.sh
```

`make gate` is the single thing the user runs to validate "M0 is done." The portability gate (small M2 and M1) is `make gate` invoked manually on those machines at the milestone boundary.

## Risks & open questions

Specific to M0; risks owned by later milestones live in the master plan or `docs/open-questions.md`.

**Monkey-patch fragility.** Polars could refactor `_gpu_engine_callback` between pin bumps and our patch silently breaks. *Mitigation:* an integration test asserts the patch site exists with the expected signature on import; failure raises a clear error rather than silently doing nothing. When we bump the Polars rev, this test fails loudly if the signature changed. Tracked in `docs/open-questions.md` as a coupling point.

**Upstream coordination.** "Monkey-patch for M0, upstream a proper hook later" is the plan. M0's implementation plan includes a sub-task to land an issue or discussion thread with the Polars maintainers documenting the use case and proposing a hook shape — not the PR itself, just the conversation, so they have time to weigh in before we're committed.

**`cxx` build complexity.** `cxx` requires invoking a C++ compiler from cargo. On Apple Silicon this is `clang++` from the Xcode command-line tools — should be straightforward but is the first time a build dep outside cargo's world enters this project. *Mitigation:* M0's plan has an early task to get a hello-world `cxx` build working end to end (Rust calls one C++ function returning `int`) before wiring in MLX. If `cxx` hits an unrecoverable build issue, we switch to hand-written C shim + `bindgen` — same surface area in M0, bounded cost.

**MLX availability on the build machine.** MLX (`ml-explore/mlx`, the C++ side) must be installable somewhere the build can find. *Decision deferred to M0's implementation plan:* pick whichever has the most predictable behavior across the M2 Ultra + M2 + M1 portability matrix. Weak prior: build from source via git submodule pinned to a known-good MLX rev, mirrored under `references/mlx/` or `vendor/mlx/`.

**Conformance harness scale.** Polars' Python test suite is large (thousands of tests). Running the whole thing with `engine=MetalEngine()` substituted in might be slow even when everything falls back — every test pays the round-trip-through-our-callback cost. *Mitigation:* M0 measures wall-clock for the full conformance run and documents it. If it's >30 minutes on M2 Ultra, restrict the default conformance gate to a subset (the lazy-frame tests specifically) with a `make test-conformance-full` for the complete sweep.

**Polars rev pin and IR version.** We pinned today's main; IR version `(12, 1)`. Polars main could land an IR breaking change before M0 ships, which we'd detect via the IR-version mismatch raising `NotImplementedError`. *Mitigation:* the pin holds for M0. If upstream breaks, we hold; we don't chase. This is exactly why the master plan's pinning policy exists.

**Inherited from master plan, owned by M0.** Buffer-bridge null-bitmap correctness (addressed by the Layer 1 proptest investment). Plugin API stability (addressed by pinning and the patch-site assertion test).

**Not M0's problem.** Hash groupby perf (M3). MLX null story for reductions (M2). Spill-to-CPU (post-M3).

## What this document does not do

- It does not specify M0's implementation plan — that is the next artifact, produced by `writing-plans`.
- It does not commit to a calendar — M0 completes when `make gate` passes on M2 Ultra and the portability gate passes on small M2 and M1.
- It does not specify M1 onward — each milestone gets its own spec when its turn comes.
