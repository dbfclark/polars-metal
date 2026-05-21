# polars-metal architecture

See [CLAUDE.md](../CLAUDE.md) for the high-level mission statement and
[the master plan](superpowers/specs/2026-05-19-master-plan-design.md) for
the milestone-by-milestone roadmap. This document is the maintainer's
first read: it describes the engine's structure as it stands at the end
of M1.

## Milestone status

M1 ships GPU execution for the narrow vertical slice `scan → project →
filter` over `Int64 / Float64 / Boolean` columns. Anything outside the
M1 closed set (Section [Walker](#the-walker)) falls back cleanly to CPU,
as it did in M0. Later milestones layer additional operators (arithmetic
+ reductions in M2, hash groupby in M3, etc.) onto the same dispatch
spine.

The [M1 design spec](superpowers/specs/2026-05-20-m1-design.md) is the
canonical statement of *what* M1 ships; this file describes *how* the
implementation is laid out so a maintainer can navigate the code.

## End-to-end execution flow

```
df.collect(engine=MetalEngine())
        │
        ▼
LazyFrame.collect (monkey-patched in polars_metal.__init__)
        │  injects execute_with_metal as post_opt_callback
        ▼
Polars optimizer runs, then calls back into Python:
        │
        ▼
_callback.execute_with_metal(nt, …, config=MetalEngine(…))
        │
        ▼
_walker.walk(nt)        bottom-up — returns Handled(plan_dict) | FallBack
        │
        ├── FallBack  ─▶  return; Polars' CPU executor runs the query.
        │
        └── Handled   ─▶  nt.set_udf(_udf.build_udf(plan_dict))
                          │
                          ▼
                Polars invokes the UDF when it materialises the subtree:
                          │
                          ▼
                _udf._dispatch peels Filter at the plan root and routes
                to one of:
                  • _native.execute_plan        (Scan / Project only)
                  • _native.execute_filter_compact (Filter + per-column
                                                    compaction)
                  • _native.cmp_*_col_(col|scalar)  (predicate evaluation)
                  • _native.bool_(and|or)_dispatch  (predicate combinators)
                          │
                          ▼
                polars-metal-kernels::pipeline::compact_{i64,f64,bool}
                runs the three-pass compaction on the GPU. Result bytes
                travel back as `PyBytes`; `_udf._assemble_series` rebuilds
                a `pl.Series` per surviving column via
                `pa.Array.from_buffers`.
                          │
                          ▼
                pl.DataFrame returned to Polars; query done.
```

Two Polars patch sites in `polars_metal.__init__._patch_gpu_engine_callback`
make this flow possible: `_gpu_engine_callback` (defensive — guards
against `MetalEngine` reaching Polars' engine-string allow-list) and
`LazyFrame.collect` (primary — short-circuits `engine=MetalEngine()` to
`collect(engine="cpu", post_opt_callback=ours)`). Both sites are
verified at import time by `_verify_patch_site`; the spec lives in
[`open-questions.md` § Monkey-patch coupling](open-questions.md) and the
upstream-hook proposal is in
[`upstream-polars-engine-hook.md`](upstream-polars-engine-hook.md).

## The walker

`python/polars_metal/_walker.py` implements a bottom-up traversal of the
Polars IR exposed through the `NodeTraverser` (`nt`) handed to the
post-optimisation callback. The two outcomes are:

- `Handled(plan: dict)` — every IR node in the subtree below the
  current position is in the closed set; `plan` is a serialised
  `MetalPlanNode` tree (see [`MetalPlanNode`](#metalplannode-intermediate-ir))
  with walker-only side-channel keys for the Scan leaf (the captured
  `PyDataFrame` and an optional projection list).
- `FallBack(reason: str)` — at least one node or expression in the
  subtree falls outside the closed set. Any `FallBack` poisons all
  ancestors immediately; the walker never lifts partial `Handled`
  subtrees to GPU in M1.

The accepted IR shapes (`_walk_at_current` in `_walker.py`):

| Polars IR node      | Handler                  | Notes                                                                                            |
| ------------------- | ------------------------ | ------------------------------------------------------------------------------------------------ |
| `DataFrameScan`     | `_walk_dataframe_scan`   | Falls back if `.selection` is set (optimizer-pushed predicate) or any column dtype is not in the type matrix. |
| `SimpleProjection`  | `_walk_simple_projection`| Metadata-only column re-selection by name, validated against `nt.get_schema()`.                  |
| `Select`            | `_walk_select`           | Accepts only plain `Column(name)` expressions with no alias and no cast.                          |
| `Filter`            | `_walk_filter`           | Accepts only predicates in the closed set described in [Predicate AST](#predicate-ast).           |

Anything else returns `FallBack("unsupported IR node: …")`.

### Partial-dispatch policy (M1: none)

M1 holds the line: the entire query subtree must be `Handled` or the
whole query runs on CPU. Rationale per the
[M1 design spec § Partial dispatch policy](superpowers/specs/2026-05-20-m1-design.md):
materialising scan output onto GPU, blitting back to CPU, then running
CPU filter is guaranteed slower than the all-CPU path. M2's
elementwise + reductions will close enough of the predicate set that
fallback rates drop sharply; partial-dispatch heuristics are post-M3.

### Comparison with cuDF-polars

The Handled/FallBack pattern mirrors cuDF-polars' translation walk in
`references/cudf/python/cudf_polars/cudf_polars/dsl/translate.py`
(`Translator.translate_ir`). Differences a reader of both should know:

- **Version handshake.** cuDF asserts
  `(version := self.visitor.version()) >= (12, 1)` at walk entry; we do
  not. The cost of drifting past `py-1.40.1` is a silent miss
  (`type(node).__name__` returns a name we don't recognise → FallBack
  → CPU). Tracked in
  [`open-questions.md` § Walker / UDF integration item 2](open-questions.md).
- **IR-class identification.** cuDF uses `multipledispatch` over real
  Python classes; we use `type(node).__name__` string comparison because
  Polars' PyO3-generated IR classes live in the unnamed `builtins`
  module in `py-1.40.1` (no stable Python type to dispatch on).
- **Partial subtree lifting.** cuDF lifts the largest GPU subtree it
  can and bridges CPU↔GPU at the boundary; we deliberately do not. See
  spec § Partial dispatch policy for the rationale.
- **IR coverage.** cuDF covers most of Polars' IR; we cover three node
  types. Roadmap expansion is per-milestone.

## `MetalPlanNode` intermediate IR

`crates/polars-metal-core/src/plan/mod.rs` defines a small Rust enum
that sits between "the Python walker accepted this Polars node" and
"the kernel layer is about to run." Its job is isolation:
`polars-metal-kernels` operates on `&[i64]`, bit-packed `&[u8]`
bitmaps, and a `PredicateAst` — it never imports Polars or PyO3.

Variants (M1):

- `Scan { n_rows, columns: Vec<(String, MetalDtype)> }` — leaf; the
  captured `PyDataFrame` itself is passed alongside the plan dict
  through the UDF boundary (see `_udf._extract_scan_df_and_wire_plan`).
- `Project { input, columns: Vec<String> }` — column re-selection.
  `_udf.py` lifts `Scan.projection` into a synthetic `Project` wrapper
  so the Rust dispatch handles projection uniformly.
- `Filter { input, predicate: PredicateAst }` — predicate compaction.

New variants land alongside the kernels that implement them — the
enum's smallness is intentional.

## Predicate AST

`PredicateAst` (in the same file as `MetalPlanNode`) is the closed set
of predicate shapes M1 will dispatch. The walker constructs the dict
form in `_walker._walk_predicate`; Rust reconstructs the enum in
`udf::deserialize_predicate`. Variants:

- `Column { name, dtype: I64 | F64 | Bool }`
- `LiteralI64(i64)` / `LiteralF64(f64)` / `LiteralBool(bool)`
- `Compare { op: CompareOp, lhs, rhs }` — six ops `Eq/Ne/Lt/Le/Gt/Ge`;
  both operands must resolve to the same numeric dtype (`I64` or
  `F64`); at least one operand must be a `Column`
  (constant-folded literal-vs-literal is rejected).
- `And(lhs, rhs)` / `Or(lhs, rhs)` — both sides must resolve to `Bool`.
  `LiteralBool` is rejected on either side of And/Or, matching the
  bare-literal-predicate rejection at the Filter root.

Anything outside this set — `is_null`, `NOT`, casts, arithmetic in the
predicate, multi-column function calls, string ops, types other than
`I64 / F64 / Bool` — falls back. The validator (`_walker._walk_predicate`)
and the kernel dispatch share the variant set via the Rust enum, so
adding a kernel without extending the walker (or vice versa) is a
compile-time mismatch.

## UDF + Rust dispatch path

`_callback.execute_with_metal` is the post-optimisation callback Polars
invokes. On `Handled`, it calls `nt.set_udf(_udf.build_udf(plan))`.
Polars invokes the UDF when it materialises the subtree; the UDF body
in `_udf.py` does the following:

1. **Strip walker-only side channels** from the plan
   (`_extract_scan_df_and_wire_plan`): capture the underlying
   `PyDataFrame` from the (unique) `Scan` leaf, rewrite the tree into
   the Rust wire format, lift `Scan.projection` into a synthetic
   `Project` wrapper. Multi-Scan plans (joins / unions) are
   not supported in M1 and raise rather than mis-route silently.
2. **Route the wire plan** in `_dispatch`:
   - `Filter` at the root → `_dispatch_filter` → compaction pipeline.
   - `Project(Filter(…))` → compact first, then apply the projection
     via the underlying `PyDataFrame.select` (never
     `pl.DataFrame.select`, which would re-enter `LazyFrame.collect`
     and the monkey-patch — infinite recursion;
     see [`open-questions.md` § Walker / UDF integration item 1](open-questions.md)).
   - Otherwise → `_native.execute_plan(df_pydf, wire_plan)`, which
     handles Scan (pass-through) and Project (column re-selection) in
     Rust.
3. **Filter dispatch** in `_dispatch_filter`:
   - Resolve the upstream DataFrame (run scan/project under the
     Filter via `execute_plan`).
   - Evaluate the predicate AST via `_evaluate_predicate`. Leaves:
     `Column(Bool)` reads the precomputed bool column.
     `Compare` calls one of `_native.cmp_{i64,f64}_col_{col,scalar}`,
     which runs the matching MSL `cmp_*` kernel against the upstream
     column buffers. `And`/`Or` recursively evaluates both sides, then
     combines via `_native.bool_{and,or}_dispatch` (3-valued Kleene
     logic in `shaders/logical_bool.metal`).
   - The result is a bit-packed `(data, valid)` predicate buffer pair.
4. **Per-column compaction** in `_native.execute_filter_compact`
   (`crates/polars-metal-core/src/udf.rs`). For each surviving column,
   Python extracts the Arrow `(data, valid)` bytes
   (`Series.to_arrow().buffers()`), hands them to Rust, which calls
   `polars-metal-kernels::pipeline::compact_{i64,f64,bool}` — the
   three-pass compaction described in
   [§ Three-pass compaction pipeline](#three-pass-compaction-pipeline).
5. **Reassembly** in `_assemble_series`: trim the kernel's
   4-byte-aligned padding to the Arrow-canonical
   `ceil(n_out / 8)` byte length, build a `pa.Array.from_buffers`,
   wrap in a `pl.Series`, return a `pl.DataFrame` to Polars.

The Python-side Arrow extraction is a deliberate choice: PyArrow's
`Buffer` exposes `bytes(buf)` directly, which is much simpler than
pulling the buffers through PyO3 ChunkedArray plumbing. The cost is
two byte copies (Arrow → bytes for input, bytes → Arrow for output) per
column; profiling at the end of M1 (see
[Performance, current state](#performance-current-state)) puts the per-query
dispatch + buffer marshalling overhead, not the kernels themselves, on
the hot path.

## Three-pass compaction pipeline

`crates/polars-metal-kernels/src/pipeline.rs` splits the work into two
phases. Passes 1 and 2 (`compute_keep_and_prefix`) run **once per
filter dispatch**; pass 3 (`compact_{i64,f64,bool}`) runs **once per
surviving column**, taking the shared `keep` and `prefix` buffers
plus the precomputed survivor count `n_out`.

1. **Predicate to dense u8.** `dispatch_predicate_to_u8`
   (`shaders/filter_predicate.metal::filter_predicate_to_u8`) reads
   the bit-packed predicate data + validity and writes
   `keep: u8[n_rows]` where `out[i] = 1` iff `pred_data[i] AND
   pred_valid[i]`. The 8× byte blowup over bit-packed bool is the
   price for MLX cumsum's dense-input requirement.
2. **Inclusive prefix sum.** MLX `cumsum` over `keep` produces
   `prefix: u32[n_rows]`. The wrapper lives in
   `polars-metal-mlx-sys::cumsum_u8_to_u32`, forced onto `Device::gpu`
   via `StreamContext`. `prefix[n_rows - 1]` is the survivor count
   `n_out`. Zero survivors short-circuits the third pass.
3. **Scatter.** `dispatch_scatter_{i64,f64,bool}`
   (`shaders/filter_scatter.metal`) writes each surviving row to
   `dst_data[prefix[i] - 1]` and ORs the validity bit into
   `dst_valid`. Validity is bit-packed and 4-byte-aligned (the buffer
   is bound to the kernel as `device atomic_uint*`, atomics are
   used because eight output rows share one validity byte). For
   `filter_scatter_bool` the data buffer is itself bit-packed and uses
   the same atomic-OR pattern. An overrun sentinel (i64 / f64 only —
   the bool variant cannot reserve a slot since every bit is
   meaningful) is allocated and checked post-dispatch.

Before T30 Step 2 (`fcf7a1f`) passes 1 and 2 were inlined into each
per-column `compact_*` call and ran redundantly once per surviving
column. Hoisting them out is the largest single contribution to the
post-T30 perf numbers below.

The `keep_flags` u8 buffer, the u32 prefix buffer, and the per-column
dst buffers are each freshly allocated per filter dispatch; per-query
arena reuse is deferred to M2 — see
[Arena and deallocator keep-alive](#arena-and-deallocator-keep-alive).

### MLX cumsum copy cost

`polars-metal-mlx-sys::cumsum_u8_to_u32` is the only MLX op M1 uses.
Its FFI signature is `(input: &[u8], out: &mut [u32]) -> Result<(),
FfiError>`, implemented over `rust::Slice` (thin pointer + length) on
both sides — no per-element marshalling across the FFI boundary. The
remaining cost inside the bridge is the MLX `array(ptr, shape, dtype)`
constructor's copy-in and a memcpy of the scan result back into the
caller's slice. At M1 sizes that's small; fully eliminating it
requires direct MLX-over-`MTLBuffer` views, which is the scheduled M2
FFI revisit tracked in
[`open-questions.md` § FFI choice (cxx)](open-questions.md).

## Arena and deallocator keep-alive

M0's `polars-metal-buffer` crate gave us the buffer bridge: an
`Arc<ArrowBuffer>` is captured in a Metal deallocator block so a
zero-copy `MTLBuffer` keeps its source Arrow allocation alive
(`crates/polars-metal-buffer/src/bridge.rs::MetalBuffer::zero_copy`).

M1's compaction pipeline reuses that symmetry in the opposite
direction. Each compacted column comes back from Rust as a `Vec<u8>` of
result bytes; the Python side wraps those bytes in
`pa.py_buffer(...)` and hands them to `pa.Array.from_buffers`. The
PyArrow buffer's lifetime is owned by Python (the underlying `bytes`
object keeps the allocation alive); there is no Rust-side arena to
keep alive across the UDF boundary in M1, because per-column
allocations are owned by the `Vec<u8>` results that PyO3 copies into
`PyBytes`.

The full per-query arena and `Arc<ScratchArena>`-in-deallocator pattern
described in the M1 design spec is *not* the shape M1 actually ships —
the data path turned out to be simpler (Python owns the resulting
bytes; no Metal buffer needs to outlive the Rust function call).
`BumpArena` ships in `crates/polars-metal-core/src/arena.rs` with
integration tests at `crates/polars-metal-core/tests/test_arena.rs`,
but the M1 hot path does not call it — M2 will wire it in. The
spec's arena design is the right shape for M2, where direct
MLX-over-MTLBuffer (above) will mean the GPU buffer must outlive the
FFI call; the arena will land then. The buffer bridge's
deallocator-keep-alive remains the canonical pattern for "Metal
resource that needs to outlive a Rust scope," and is unchanged from M0.

## Shader build pipeline

`crates/polars-metal-kernels/build.rs` is the build-time step that
turns `shaders/*.metal` into one embedded `polars_metal.metallib`:

1. Enumerate `shaders/*.metal`. Files whose stem starts with `_`
   (today: `_validity.metal`) are header-only — they are `#include`d
   by other kernels and excluded from compilation. The `-I
   <shaders_dir>` flag passed to `xcrun metal` makes the includes
   resolve.
2. For each non-header source, run `xcrun metal -c -frecord-sources -I
   <shaders_dir> -o <stem>.air <source>` to produce one `.air` per
   kernel file.
3. Link the `.air` files into a single `.metallib` via `xcrun
   metallib`.
4. Export the metallib's absolute path via `cargo:rustc-env=POLARS_METAL_METALLIB=…`.
   `crates/polars-metal-kernels/src/shader_lib.rs` then
   `include_bytes!(env!("POLARS_METAL_METALLIB"))` the metallib into
   the compiled binary — no runtime path lookup.

At runtime, `ShaderLibrary` (`shader_lib.rs`):

1. **Materialise once per process.** A `OnceLock` writes the embedded
   bytes to a per-PID temp file the first time `shared_library()` is
   called. `objc2-metal 0.2` exposes `newLibraryWithURL:` but not the
   `dispatch_data_t`-based `newLibraryWithData:`, so the brief
   on-disk hop is unavoidable. The file is unlinked after the first
   successful load.
2. **Cache pipeline state objects per entry point.** `ShaderLibrary.psos`
   is a `Mutex<HashMap<String, MTLComputePipelineState>>`. Each kernel
   entry point (e.g. `cmp_i64_lt`, `filter_scatter_bool`) is built at
   most once per process; concurrent builders deduplicate via the
   cache.
3. **Query threadgroup size at runtime.** Per CLAUDE.md's portability
   gotcha, `crates/polars-metal-kernels/src/command.rs::dispatch_1d`
   reads `MTLComputePipelineState::maxTotalThreadsPerThreadgroup` off
   the PSO and clamps to it (with a `DEFAULT_THREADGROUP_WIDTH`
   ceiling). The per-kernel dispatchers in `cmp.rs`, `filter.rs`, and
   `logical.rs` go through `CommandQueue::dispatch_1d`; threadgroup
   sizes are never hardcoded.

The `Send + Sync` impl on `ShaderLibrary` documents that `MTLLibrary`
and `MTLComputePipelineState` are Apple-documented thread-safe and the
cache is `Mutex`-guarded. This lets the library be shared from any
thread, which matters for the kernel proptest suite (parallel cargo
test workers in the same binary).

## Error & fallback policy

Two categories, per the
[M1 design spec § Error & fallback policy](superpowers/specs/2026-05-20-m1-design.md):

- **Plan-shape rejections** are normal and frequent. The walker
  returns `FallBack(reason)`; no `set_udf` call; Polars runs CPU.
  Reasons are logged at DEBUG level when `MetalEngine(debug=True)`.
- **Runtime failures** during GPU execution are rare and fatal. There
  is no mid-execution fallback. `EngineError` variants (allocation
  exhausted, FFI failure, kernel dispatch error, scatter overrun
  sentinel tripped) convert to `polars.exceptions.ComputeError` at the
  PyO3 boundary — see `crates/polars-metal-core/src/error.rs`. The M0
  follow-up commit (`c6d9558`) made `ComputeError` the canonical
  surface; the test asserting this lives in the same crate.

## Performance (current state)

End-to-end M1 benchmarks at 10M rows on Apple M2 Ultra
(`tests/bench/baseline.json`) put `engine=MetalEngine()` between 5.3×
and 17.6× **slower** than `engine="cpu"` across the five M1 queries
(`filter_simple` 8.4×, `filter_compound` 9.0×, `filter_then_project`
8.7×, `filter_then_project_high_selectivity` 17.6×,
`filter_then_project_low_selectivity` 5.3×). T30 cut `filter_simple`
from 220ms to 78ms (−65%) by hoisting the predicate-to-u8 and MLX
cumsum out of the per-column loop and by replacing the
`CxxVector`-based cumsum FFI with a `rust::Slice` bridge. The
pre-T30 picture put cumsum FFI marshalling × per-column redundancy on
the hot path; that is now fixed.

What's left in the surviving ~78ms (per T30 Step 1 profiling) is
kernel-level dispatch (predicate + scatter — threadgroup tuning,
validity-bitmap load patterns), the per-query Arrow→MTLBuffer setup
(~18ms — the original zero-copy bridge wiring), and the two
host↔MLX memcpys still inside the cumsum call. All three are
M2-shaped; the 5%-of-CPU bar from the design spec is not met in M1.
Details are in [`open-questions.md`](open-questions.md) and the M1
retrospective.

## Where the code lives

```
crates/
  polars-metal-buffer/      Arrow ↔ MTLBuffer bridge (M0); deallocator
                            keep-alive pattern.
  polars-metal-mlx-sys/     MLX FFI bindings (cxx). M1 export:
                            cumsum_u8_to_u32.
  polars-metal-kernels/     Rust-side kernel wrappers + dispatchers.
    src/cmp.rs              cmp_{i64,f64}_{col,scalar} dispatchers.
    src/logical.rs          bool_{and,or} dispatchers.
    src/filter.rs           predicate-to-u8 + scatter_{i64,f64,bool}
                            dispatchers.
    src/pipeline.rs         compact_{i64,f64,bool} — orchestrates the
                            three-pass pipeline.
    src/shader_lib.rs       metallib loader + per-entry-point PSO cache.
    src/command.rs          CommandQueue wrapper.
    build.rs                Shader build pipeline.
  polars-metal-core/        Engine adapter + PyO3 entry points.
    src/plan/mod.rs         MetalPlanNode + PredicateAst.
    src/udf.rs              execute_plan, execute_filter_compact,
                            cmp_*_col_*, bool_{and,or}_dispatch.
    src/error.rs            EngineError → polars.exceptions.ComputeError.
shaders/
  _validity.metal           Bit-packed-validity helpers (header-only).
  hello.metal               Build-pipeline canary (validates metallib).
  cmp_i64.metal             Six i64 comparison ops × col/scalar.
  cmp_f64.metal             Six f64 comparison ops × col/scalar.
                            NaN semantics differ from Polars'
                            TotalOrd — see open-questions.md.
  logical_bool.metal        3-valued AND / OR.
  filter_predicate.metal    Bit-packed bool + valid → dense u8 keep.
  filter_scatter.metal      Scatter surviving rows for i64/f64/bool.
python/polars_metal/
  __init__.py               Monkey-patches LazyFrame.collect +
                            _gpu_engine_callback.
  _engine.py                MetalEngine config dataclass.
  _callback.py              post-optimisation callback; calls walker;
                            installs UDF.
  _walker.py                Bottom-up IR walk → Handled/FallBack.
  _udf.py                   UDF body: routing, predicate evaluation,
                            Arrow extraction, reassembly.
tests/
  kernel/                   Per-kernel correctness + proptest.
  conformance/              Polars' own tests/unit/lazyframe/ +
                            operations/filter/comparison/select/
                            expr/binary, run with engine=MetalEngine().
  diff/                     Hypothesis-driven differential vs CPU.
  bench/                    pytest-benchmark E2E + baseline.json.
  python_integration/       Python ↔ Rust seam tests (walker, UDF
                            dispatch, monkey-patch).
```

## Open caveats

Forward-pointers to [`open-questions.md`](open-questions.md):

- **`cmp_f64` NaN vs Polars TotalOrd.** IEEE 754 semantics; one
  conformance test is `xfail(strict=True)`. *Owner:* M1 post-T18 / M2.
- **MLX FFI revisit.** The slice-based `cumsum_u8_to_u32` bridge
  retired the per-element marshalling cost; the remaining two
  host↔MLX memcpys inside the call need the direct-MLX-over-MTLBuffer
  rewrite, which M2's reductions will force anyway.
- **Per-query dispatch + marshalling overhead.** What remains of the
  M1 E2E gap after T30 (above); arena + direct MLX wiring +
  threadgroup tuning are the planned M2 levers.
- **Polars rev drift / walker IR-class identification.** M1 has no
  version handshake; cuDF's pattern is the model when we add one.
- **Monkey-patch coupling.** Two patch sites today; upstream-hook
  proposal in `upstream-polars-engine-hook.md` retires both.
