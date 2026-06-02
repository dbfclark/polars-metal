# Open questions

Questions and risks that don't block current work but warrant tracking. Each entry has a one-line title, the open question, and where it gets owned.

## Monkey-patch coupling

The M0 patch turned out to need two sites, not one:
1. `polars.lazyframe.frame._gpu_engine_callback` — defensive wrap so MetalEngine reaching this function returns our callback rather than raising on the engine-string allow-list.
2. `polars.lazyframe.frame.LazyFrame.collect` — primary intercept. Polars 1.40+ routes `engine=` straight to Rust's `ldf.collect()`, which only accepts strings, so we short-circuit at the Python level: when `engine=MetalEngine()`, call `original_collect(self, engine="cpu", post_opt_callback=cb)` with our callback. `post_opt_callback` is an internal/test-only Polars hook.

Both sites are now verified at import time. Site 1 checks the `_gpu_engine_callback` parameter set; Site 2 does a runtime probe call (`LazyFrame.collect(engine="cpu", post_opt_callback=lambda *a, **kw: None)` on a trivial frame) because `post_opt_callback` flows through Polars' `**_kwargs` catch-all and isn't visible to `inspect.signature`. If either check fails on a future Polars rev, `import polars_metal` raises with a clear message rather than letting the patch silently no-op.

The real fix is an upstream engine-registration hook in Polars; once that lands, both patches retire. *Owner:* see `docs/upstream-polars-engine-hook.md` (drafted in T37).

## MLX null story

MLX is dense-numeric-first and has no first-class null representation. Reductions in M2 need either custom MSL null-aware kernels or MLX-op-on-data + separate-validity-reduction. *Owner:* M2's design spec.

## Hash groupby GPU performance

The biggest single unknown in M0–M3. Two-pass count-then-fill is the textbook approach (cuDF reference: `references/cudf/cpp/src/groupby/`). *Owner:* M3's design spec.

## Memory pressure on base M1

M2 Ultra has plenty of memory; M1 (8 GB) does not. We don't plan spill-to-CPU in M0–M3, but the portability gate catches OOM regressions. *Owner:* future, post-M3.

## Buffer-bridge alignment assumptions

We assume Arrow `Buffer`s sometimes arrive page-aligned and sometimes don't. The bridge handles both via two regimes. If profiling shows we're nearly always in the copy regime, we may want to influence Arrow allocations to land page-aligned. *Owner:* future, after M1 perf data.

## EngineError surfaces as PyRuntimeError, not polars.exceptions.ComputeError

`crates/polars-metal-core/src/error.rs` converts `EngineError` to `PyRuntimeError("polars-metal: ...")`. The spec wanted errors to surface as `polars.exceptions.ComputeError` so they look native to Polars users. The current behaviour is dormant in M0 (the callback never raises), but M1+ kernels can fail and users will see the wrong exception type. *Follow-up:* convert to `polars.exceptions.ComputeError` directly via PyO3 type lookup, and add a Python integration test asserting `pl.exceptions.ComputeError` is what callers catch. *Owner:* pre-M1.

## FFI choice (cxx)

We committed to `cxx` for M0 based on a weak prior. If friction emerges before M2, switch to hand-written C shim + `bindgen`. *Owner:* M0 (revisit at M2's design).

**M1 friction observation (T5, 2026-05-20):** ~~the `cxx::CxxVector` push-in / iter-copy-out pattern forces a per-element copy across the FFI boundary for every binding (`add_f32`, `cumsum_u8_to_u32`). At M1 sizes this is invisible; at M2's groupby/join scale (10M+ rows, multi-column) the u32-per-row copy on cumsum alone will dominate.~~ **Resolved for cumsum in T30 Step 3 (`ae63acb`):** `cumsum_u8_to_u32` now uses `rust::Slice` on both sides — no `CxxVector`, no per-element marshalling. T30 Step 2 (`fcf7a1f`) also hoisted the cumsum out of the per-column compaction loop, so the call now happens once per filter dispatch rather than once per surviving column. The remaining cost inside the bridge is the MLX `array(ptr, shape, dtype)` constructor's copy-in and a memcpy of the scan result back into the caller's slice.

**M1 confirmation (T24, 2026-05-21):** the M1 baseline (`tests/bench/baseline.json`) showed the cumsum-via-`CxxVector` step was a confirmed contributor to the end-to-end gap. T30 (above) resolved the per-element marshalling component; the two host↔MLX memcpys still inside the cumsum call require direct MLX-over-`MTLBuffer` wiring (construct `mlx::core::array` views over our already-allocated `MTLBuffer` and read the result back in place).

**Partial resolution of the cxx-vs-bindgen question:** the T30 Step 3 slice bridge demonstrates that `cxx` *can* do raw-pointer-plus-len performantly via `rust::Slice`. The remaining M2 decision — whether direct MLX-over-`MTLBuffer` views are cleaner under (a) extended `cxx` shims or (b) hand-written C shim + `bindgen` — is narrower than originally framed.

## Walker / UDF integration with Polars internals (M1, T7 2026-05-20)

Three friction points discovered while building the IR walker in Task 7. Each will affect every later task that touches the walker or the UDF path, so they live here rather than buried in T7's commit body.

1. **UDF re-entry via `df.select`.** `pl.DataFrame.select` internally calls `lazy().collect()`. When `MetalEngine` is installed, our patched `LazyFrame.collect` intercepts that re-entrant call and dispatches *back* through the walker — infinite recursion. T7 mitigates by using `pl.DataFrame._from_pydf(df._df.select(list(names)))` (the underlying PyDataFrame's sync `select`, no LazyFrame round-trip). **Task 8's Rust dispatch will hit the same trap** the moment it constructs a result DataFrame from kernel outputs; any reassembly path that uses `df.select`/`df.with_columns` triggers re-entry. *Owner:* M1 (T8 onward — always use PyDataFrame internals when assembling UDF outputs).

2. **Polars NodeTraverser version handshake.** T7 identifies IR nodes by `type(node).__name__` because the PyO3-generated IR classes in py-1.40.1 live in the unnamed `builtins` module. This is the canonical Polars-Python idiom for one pinned rev. cuDF-Polars guards against silent breakage on Polars version drift by asserting `(version := nt.version()) < (12, 1)` at walk entry. We don't have an equivalent guard. Adding one before any Polars rev bump (current pin: `py-1.40.1`) prevents silently wrong behavior when an IR class is renamed. *Owner:* M1 (T8 or first conformance regression on a rev bump).

3. **`DataFrameScan.selection` predicate pushdown triggers M1 fallback.** Polars' optimizer can push a Filter predicate onto a DataFrameScan as a `.selection` attribute, eliminating the explicit Filter node. The walker currently rejects any DataFrameScan with a non-None `selection` (because we'd need to evaluate that predicate via the same kernels Filter uses). Once Phase 5+ Filter lands, the walker should lift these to GPU instead — otherwise many CSE-optimized queries fall back unnecessarily even when they're entirely within the M1 supported set. *Owner:* M1 (Phase 5+, T15+).

## cmp_f64 NaN semantics: IEEE 754 vs Polars TotalOrd (M1, T18 2026-05-20)

`shaders/cmp_f64.metal` (Task 17) implements IEEE 754 ordered comparison: `NaN OP x` is `false` for `==, <, <=, >, >=` and `true` only for `!=`. Polars CPU implements `TotalOrd` semantics for these ops — NaN is treated as **greater than any non-NaN** value, so `NaN > 0 = true`, `NaN > NaN = false` (per total ordering), `NaN == NaN = true`.

Concrete check (`py-1.40.1`):

```
pl.Series([1.0, NaN, 3.0]) > 0  →  [True, True, True]
pl.Series([1.0, NaN, 3.0]) == pl.Series([1.0, NaN, 3.0])  →  [True, True, True]
```

Discovered while landing the Task 18 end-to-end filter+comparison wiring. The integer path (`cmp_i64.metal`) is unaffected. T18 marks the failing test (`test_filter_with_nan_f64_total_ord`) `xfail(strict=True)` so when the kernel is fixed, the strict-xfail will flip and force us to drop the marker.

**Fix sketch:** rework the six `f64_<op>` helpers in `cmp_f64.metal` to short-circuit NaN-presence into the matching TotalOrd outcome. `f64_eq` returns true when both inputs are NaN; the order helpers (`<, <=, >, >=`) treat NaN as larger than any non-NaN value. `f64_total_order_key` is already nearly the right primitive — it just needs to map NaN consistently above ±Inf.

*Owner:* M1 (post-T18 follow-up alongside Task 21's hypothesis differential strategies, which would have caught this immediately).

## MLX install path

Resolved in T19: git submodule under `vendor/mlx`, pinned to v0.22.0, built via cmake. *Owner:* M0 (resolved). Refresh via standalone cmake invocation, not via `scripts/refresh-references.sh` (which is for read-only references, not build deps).

## Conformance harness scale

`pytest tests/conformance` (14 tests, 7 operations × cpu+metal) ran in 0.461s wall-clock on an M-series Mac (measured T33, 2026-05-20). Will grow as M1+ ops land; keep an eye on suite time crossing 60s and add pytest-xdist parallelism if needed.

## Metal toolchain (resolved)

Initially missing on the dev host: T19's MLX build had to use `-DMLX_BUILD_METAL=OFF` because the Metal shader compiler couldn't be invoked. Root cause was a stale x86_64 `CoreSimulator.framework` under `/Library/Developer/PrivateFrameworks/` blocking Xcode's plugin loading. Resolved 2026-05-20 by running `sudo xcodebuild -runFirstLaunch`, then `xcodebuild -downloadComponent MetalToolchain`. MLX rebuilt with `-DMLX_BUILD_METAL=ON`; `build.rs` now links `Metal`, `Foundation`, `QuartzCore`, `Accelerate` frameworks. *Owner:* M0 (resolved).

## M0 retrospective (2026-05-20, updated with conformance findings)

**Outcome.** `make gate` passes end-to-end on M2 Ultra in ~6s wall-clock. 30 Python tests + 20 Rust unit/proptest tests across the workspace, all green. `df.collect(engine=polars_metal.MetalEngine())` returns CPU-equivalent results on select/filter/group_by/join/sort/with_columns.

**Conformance against Polars' own test suite.** A post-review investigation revealed M0 conformance is stronger than the original retrospective implied. `tests/conformance/test_polars_suite.py` now runs Polars' own `tests/unit/lazyframe/` (~722 tests) with `engine=MetalEngine()` forced via `polars_metal._pytest_plugin`. With the Polars wheel pinned exactly to 1.40.1, the references/polars submodule at the matching `py-1.40.1` tag, and `cloudpickle` in dev deps, the suite shows **721 passed / 0 failed / 1 skipped** under our engine. Same numbers as pure Polars baseline — our engine adds zero new failures. The 31 "failures" reported in earlier sessions were entirely version skew (tests for newer-Polars features absent in 1.40.1) plus missing optional deps; they fail in pure CPU runs too. The one real patch-induced bug — a `post_opt_callback` collision in `tests/unit/lazyframe/cuda/test_node_visitor.py` — was fixed by chaining caller-provided callbacks ahead of ours in `python/polars_metal/__init__.py`.

**Surprises during execution (vs. the plan):**

- **Rust 1.81 was too old.** Proptest 1.x pulls `getrandom 0.4.2`, which needs the `edition2024` Cargo feature, stable starting Rust 1.85. Toolchain bumped to 1.85.0.
- **objc2-metal API rough edges.** Plan's snippets needed several adjustments — `ProtocolObject` lives in `objc2::runtime`, `MTLCreateSystemDefaultDevice` returns a raw pointer, `MTLResourceOptions::MTLResourceStorageModeShared` is the actual constant name, and the deallocator block requires `block2::RcBlock::new` from the `block2 = "0.5"` crate (not `objc2::block`).
- **MLX C++ API rough edges.** `eval` is in `transforms.h` (not `utils.h`); MLX `Shape` is `std::vector<int32_t>` (not int64). For cxx return-type interop, the C++ side returns `std::unique_ptr<std::vector<float>>`.
- **Metal toolchain was missing.** Initially blocked GPU build; resolved by `sudo xcodebuild -runFirstLaunch` + `xcodebuild -downloadComponent MetalToolchain`. See dedicated entry above.
- **The monkey-patch needed two sites, not one.** Polars 1.40+ routes `engine=` straight to Rust's `ldf.collect()`. The Python-side `_gpu_engine_callback` wrap is insufficient on its own; we also patch `LazyFrame.collect` and inject our callback via `post_opt_callback`. See dedicated entry above.

**Resolved in PR #1 follow-up commits (kept M0 cohesive rather than queueing for a separate pre-M1 PR):**

- ~~Add a signature assertion test for the `LazyFrame.collect` patch site.~~ Done — `_verify_patch_site` now probes `post_opt_callback` at import time (commit `c6d9558`).
- ~~`EngineError → PyErr` surfaces as `PyRuntimeError` rather than `polars.exceptions.ComputeError`.~~ Done — now uses `polars.exceptions.ComputeError` with `PyRuntimeError` fallback; new Rust test asserts the produced exception is a real `ComputeError` instance (commit `c6d9558`).
- ~~MLX GPU dispatch isn't separately validated.~~ Done — `add_f32_on_gpu` forces `Device::gpu` via `mlx::core::StreamContext`, tested on a 4096-element array (commit `58e9fb5`). MLX throws if Metal is unavailable, so a passing test = working dispatch.

**Still to revisit at M1 (not blockers):**

- The portability gate (small M2, M1) is still a manual run-on-your-other-machine step. Document the procedure in `docs/` if we want subsequent milestones to enforce it more uniformly.
- The `pyo3 0.22` macro emits `useless_conversion` clippy warnings in `polars-metal-core`; suppressed file-scoped with `#![allow(clippy::useless_conversion)]`. Revisit when pyo3 is upgraded.
- ~~The hypothesis differential harness in `tests/diff/` generates bare scans (no varied operations), so it's only really testing fallback parity. Add filter/select/groupby strategies once M1 has kernels producing real GPU output.~~ Filter+select portion done in T21 (M1). `tests/diff/` retired in M2 T33–T35: property tests migrated to `crates/polars-metal-kernels/tests/test_compaction_pipeline.rs` (Rust proptest); explicit edge cases migrated to `tests/python_integration/test_filter_edges.py`. Groupby differential strategies remain deferred to M3 — see "Groupby differential strategies still missing" under "M1 retrospective notes" below.

## ~~Routing layer: per-op GPU-vs-CPU dispatch on unified memory (M1, post-T30 2026-05-21)~~

_Resolved in M2 Phases 1–2._ Per-op routing layer shipped in production.
Walker emits `MetalPlanNode` trees; `compute_lifting_plan` in
`crates/polars-metal-core/src/router/` walks the tree, applies cost rules,
and returns a `LiftingPlan`. Python callback reads the root decision and
either lifts to GPU (via UDF) or returns to CPU. See
`docs/architecture.md` § M2: routing layer and hash groupby.

## Routing layer: per-op GPU-vs-CPU dispatch on unified memory (M1, post-T30 2026-05-21)

The M1 perf investigation surfaced an architectural assumption we'd been carrying from cuDF without noticing. On a discrete GPU, the dispatch decision is dominated by "is this data already on the GPU?" because PCIe transfer costs swamp everything — so cuDF keeps everything on GPU. On Apple Silicon's unified memory that constraint vanishes: the same bytes are equally addressable from both processors with zero transfer cost.

This changes the right framing of the engine. Today the walker says "if I have a kernel for this shape, dispatch to GPU." The correct rule is "if I have a kernel AND the kernel beats CPU at this input shape, dispatch to GPU." Per-op, possibly per-row-count, decided at plan time from a small cost-model table.

Worked example from M1: filter is memory-bound. CPU Polars hits ~200–400 GB/s effective via SIMD on the same DRAM the GPU would read. The GPU has no fundamental bandwidth advantage on memory-bound ops on unified memory; the realistic ceiling for GPU filter at 10M rows is 2–3× CPU, not 5% slower. The M1 baseline (5.3–17.6×) is close to the structural ceiling for filter specifically. M2's stated ops (groupby/sort/join) are compute-/parallelism-dense and should beat CPU — that's where dispatch to GPU is the right answer.

Concrete design items for M2:

1. **Cost model in the walker.** Per-op, per-input-shape estimates seeded from `tests/bench/baseline.json` and the criterion microbenches. Starts dumb (constants table), refines over time.
2. **`Handled` becomes explicit GPU-plan; the CPU path is also explicit** rather than implicit-via-fallback. Three states, not two: `Handled(GpuPlan)`, `Handled(CpuPlan)`, `FallBack(reason)`.
3. **Per-op crossover thresholds.** Filter: probably CPU below 100M rows when isolated. Groupby/sort/join: GPU at much lower row counts. Updated as kernels improve.
4. **Spec language fix.** The ≤5% Metal/CPU bar should become "≤5% slower than the best routing." If the router picks CPU, we *are* CPU on that path — the gate is automatically met by definition.

The M1 filter kernels are not wasted under this framing — they're the reference for what GPU filter looks like when it's competitive (large-N, chained with other GPU ops, etc.). They're called when the cost model says yes; they sit dormant when it says no.

**Mission reframe.** CLAUDE.md says "Match cuDF-Polars-on-a-4090 performance for realistic analytical workloads." That survives intact: cuDF wins on a 4090 because of dedicated VRAM bandwidth on bandwidth-bound ops; we win on Apple Silicon by routing to whichever processor is best per-op. Same end result (best-available perf on the user's hardware) without pretending the GPU is always the right answer. *Owner:* M2 design spec — this is the dominant architectural input.

## ~~M1 end-to-end performance gap (M1, T24 2026-05-21)~~

_Resolved for M2's stated workload._ M2 confirms filter belongs on CPU under
unified memory; the router routes it to CPU at all sizes. The modified TPC-H
Q1 benchmark (Metal-routed GroupBy, CPU-routed Filter + Sort) records
`ratio_metal_over_cpu = 0.914` in `tests/bench/baseline.json::tpch_q1_modified`
— Metal 8.6% faster than CPU, meeting the M2 perf gate (`ratio < 1.0`).
M1-only filter queries remain 5–17× slower on Metal; the router suppresses
that path in production.

## M1 end-to-end performance gap (M1, T24 2026-05-21)

The M1 design spec § Layer 4 sets a `ratio_metal_over_cpu <= 1.05` gate on five E2E queries. `tests/bench/baseline.json` (M2 Ultra, post-T30 at `ae63acb`) shows the actual ratios at 10M rows are 5.27× – 17.65× — Metal still slower than CPU on every query, not within 5%, but down from the pre-T30 12.14× – 54.82× range. `filter_simple` cumulatively dropped from 220ms to 78ms (−65%) across T30.

Kernel-only criterion numbers (T23 — `cmp_i64` ~20 ms per 10M rows, `filter_scatter` ~19 ms per 10M rows) account for the bulk of the remaining ~50–100 ms per-query Metal wall-clock. The remaining gap is dispatch + Arrow↔MTLBuffer setup per query, not kernel compute.

Remaining root causes, in rough order of suspected impact (post-T30):

- **Predicate-kernel and scatter-kernel dispatch.** Threadgroup sizing is left at a portable default rather than tuned per device class; validity-bitmap loads are issued one byte at a time (vs the 8-rows-at-a-time pattern called out in CLAUDE.md's gotchas); scatter writes serialise on the prefix-sum dependency. Owner: M2 kernel-level tuning pass.
- **Per-query Arrow→MTLBuffer setup (~18 ms).** The M0 zero-copy bridge has both a zero-copy regime (page-aligned, refcount-bumped) and a copy regime; M1 queries are taking the copy regime more often than expected. Owner: M2 perf response (direct MLX-over-MTLBuffer wiring will also bear on this).
- **Two host↔MLX memcpys inside the cumsum call.** MLX's `array(ptr, shape, dtype)` constructor copies bytes into MLX-managed memory and we memcpy the scan result back into the caller's slice. Per-element marshalling was eliminated in T30 Step 3 (`ae63acb`); full elimination of the data-touching passes needs the MLX-over-`MTLBuffer` rewrite tracked in "FFI choice (cxx)" above.

T28's M1 retrospective took the "ship M1 with the gap documented" path; T30 followed up with the cumsum/per-column-redundancy fix; the three bottlenecks above are the M2 design inputs. *Owner:* M2 design.

## BumpArena ships but is unwired in the M1 hot path (M1, T25 2026-05-21)

The M1 design spec (`docs/superpowers/specs/2026-05-20-m1-design.md:134-147`) described an arena that owns predicate intermediates, `keep_flags`, the `prefix_sum`, and the output column buffers, with output Arrow buffers' custom-deallocator closures keeping an `Arc<ScratchArena>` alive across the UDF boundary.

The shipped M1 differs on the output side: `crates/polars-metal-core/src/udf.rs` materialises kernel outputs as `Vec<u8>` and copies them into `PyBytes::new_bound(py, &out_data)` (and the matching validity buffer) before returning to the walker, which round-trips them through Polars Arrow buffers. `BumpArena` (`crates/polars-metal-core/src/arena.rs`) is fully implemented with integration tests but is not called from `pipeline.rs` or `udf.rs` — it ships dormant.

This is intentional for M1 (the keep-alive closure model needs design work to play nicely with Python's GC and Polars' Arrow buffer ownership) but is a known item for M2 — especially if the perf-gap investigation above points at allocation cost or the second copy through `PyBytes` as a contributor. *Owner:* M2 design.

## M1 retrospective notes (2026-05-21)

This is a placeholder structure for T28's full retrospective. T27 captures only the highest-signal items here; T28 will populate the rest.

**Resolved in M1:**

- ~~Differential harness only tested fallback parity (M0 retrospective).~~ Filter + projection strategies and edge cases landed in T21 — see strikethrough under "Still to revisit at M1" above.
- ~~MLX FFI revisit was speculative (M0 "FFI choice" entry).~~ Backed by T5 friction and T24 baseline data; cumsum-specific component resolved in T30 Step 3 (`ae63acb`) via the `rust::Slice` bridge. The broader MLX-over-`MTLBuffer` decision remains an M2 design item — see "FFI choice (cxx)".

**Still to revisit at M2 (not blockers):**

- **End-to-end perf gap.** The 5.3–17.6× CPU/Metal ratio at 10M rows (post-T30) is the dominant M2 design input — see dedicated entry above.
- **BumpArena wiring.** Implemented but unwired in the M1 hot path — see dedicated entry above.
- **Groupby differential strategies still missing.** T21 closed the filter/select half of the M0-era harness gap; groupby strategies wait on M3 kernels producing real GPU output, just as filter/select did for M1.
- **Threadgroup tuning per device class.** `cmp_i64` and `filter_scatter` use portable defaults from `MTLDevice.maxThreadsPerThreadgroup()` (per CLAUDE.md), but the ~5 GB/s effective throughput on M2 Ultra suggests these are leaving silicon on the table. M2 owns the per-device-class tuning pass — not before, because we don't yet have a profiling story.

## M2 new open questions (2026-05-22)

### Apple Silicon Metal 64-bit atomics gap

The current Metal toolchain (32023.883) supports `atomic_uint` /
`atomic_int` fully but does not support `atomic_fetch_add_explicit` or
`atomic_compare_exchange_weak_explicit` on `atomic_long` / `atomic_ulong`.
M2 routes 64-bit aggregation through a CPU finalise path
(`aggregate_sum_i64_cpu` etc. in `crates/polars-metal-kernels/src/groupby.rs`).

Open question: when Apple ships the wider atomic set, should we add native
64-bit kernels behind a build-time toolchain probe + runtime feature flag,
or remain CPU-finalise for simplicity? The CPU path is not obviously a
bottleneck at M2 scale (10M rows, low cardinality); the answer depends on
M3's high-cardinality and multi-column benchmarks. *Owner:* M3+ design.

### GroupBy build phase on CPU

M2 originally planned a GPU atomic-CAS build phase for the hash table.
The design deadlocked due to SIMD-group lockstep execution: one thread in
a SIMD group cannot spin-wait on a sibling thread's write to advance. After
a failed redesign to a 3-state non-spinning machine (Task 20), the build
was moved to CPU HashMap-based find-or-insert.

Open question: for high-cardinality groupby (1M+ unique groups), the CPU
HashMap build may become the bottleneck (large map, cache-hostile random
access). An alternative is GPU sort-then-segment-reduce, which avoids CAS
atomics entirely — each row writes to a sorted position, then a prefix scan
identifies group boundaries. This is the planned M3 redesign; the hash
kernel (`shaders/groupby_hash.metal`) is retained because M3's sort-based
build would consume the hashes. *Owner:* M3 design.

### String-key groupby

Out of scope for M2. M3 must decide: dictionary-encode string keys at the
buffer bridge so the existing u128 encoder can handle them (dictionary
index is an i32 or i64, fits the current `KeyDtype` set), or run a
dedicated string-hash kernel that bypasses the encoder? Likely both: bridge
handles the common case where cardinality is low (dictionary fits in memory);
kernel handles large-cardinality string keys where dictionary encoding would
itself be expensive. *Owner:* M3 design, after multi-chunk Series support.

### Multi-chunk Series support

M2 falls back to CPU for any input Series with more than one Arrow chunk
(T31 — `dispatch_groupby` checks `series.n_chunks() > 1` and raises
`EngineError` which surfaces as `ComputeError`). Full multi-chunk support
is deferred to M3+ and likely belongs at the buffer-bridge layer
(`crates/polars-metal-buffer/`): the bridge would concatenate chunks into
a single contiguous `MTLBuffer` before handing to the kernel layer, hiding
the chunking from all kernels. *Owner:* M3+ (prerequisite for string-key
groupby).

### Composite key 128-bit limit

M2's `encode_keys` caps composite key width at 128 bits. Q1's original
spec called for `i64` keys (`l_returnflag`, `l_linestatus`); two `i64`
columns require 130 bits (65 bits × 2), exceeding the cap. The M2
benchmark fixture uses `Boolean` keys (2 bits × 2 = 4 bits total) to
avoid the overflow. A more principled fix is to extend `KeyDtype` with
`I8` / `I16` / `I32` variants that consume 9 / 17 / 33 bits respectively,
allowing common low-cardinality categorical keys to fit. *Owner:* M3 (the
string-key work will force a `KeyDtype` extension anyway).

### Conformance baseline drift on upstream Polars updates

When `make refresh-refs` bumps the pinned Polars rev in `references/`, the
`tests/conformance/_polars_known_failures_*.txt` baselines may need
recapture because tests that previously passed may fail on the new rev (new
test preconditions, renamed APIs) or vice versa. Currently the recapture
procedure is documented in the `test_polars_suite.py` module docstring and
must be run manually. Consider automating: a CI job that runs on rev bumps,
diffs the captured baseline against the committed one, and either updates
the baseline or reports drift as a review item. *Owner:* post-M2 CI
infrastructure.

### M2 perf finding — GroupBy aggregation kernels are not the bottleneck

The Q1 benchmarks shipped with M2 produce surprisingly tight ratios:

| Variant | n_groups | CPU ms | Metal ms | Ratio |
|---|---|---|---|---|
| Q1-64bit (i64 keys + f64 values, CPU finalize for aggs) | 4 | 339 | 309 | 0.914 |
| Q1-32bit (Bool keys + i32/f32 values, GPU aggs) | 4 | 27 | 26 | 0.988 |
| Q1-32bit high-card (Int32 group key, GPU aggs) | ~1024 | 67 | 66 | 0.991 |

The high-cardinality variant was added to test the "atomic contention is
killing GPU aggregation" hypothesis (low cardinality → 7M rows contending
on 4 atomic slots). **The hypothesis was disproved**: 256× lower contention
(4 → 1024 groups) moved the ratio by only ~0.003. The absolute time jumped
~40ms equally for both engines.

What this tells us:

- **Both engines pay ~40ms more at 1024 groups vs 4 groups.** That cost is
  *shared* between CPU and Metal because the build phase (CPU HashMap) and
  result-DataFrame assembly (CPU) scale with cardinality. The GPU
  aggregation kernels themselves did not change cost meaningfully —
  whatever they were doing at 4 groups, they do at 1024 groups.

- **The GPU aggregation kernels are already fast enough to be invisible
  in Q1-shaped queries.** The remaining 1-2% gap is split between
  CPU-routed phases (filter, sort), the CPU build phase, MetalBuffer
  marshalling overhead, and PyO3 boundary crossings. None of those can be
  improved by tuning the GPU aggregation code further.

**The largest perf lever for M3+ is therefore not "tune the aggregation
kernels" — it's:**

1. **Move the build phase back to GPU.** The current CPU HashMap build is the
   single largest residual CPU cost at high cardinality. A sort-then-
   segment-reduce design (cuDF has one) avoids the atomics-and-lockstep
   issues that forced the M2 build-phase pivot to CPU. *Owner:* M3.

2. **Kernel-fusion across aggregations.** Q1's 8 aggregations dispatch 8+
   separate kernels (each ~50-100μs Metal launch overhead + buffer setup).
   A fused kernel that reads each row once and updates all aggregations in
   one pass would cut dispatch overhead ~10×. *Owner:* M3 / kernel-tuning
   pass.

3. **GPU filter when it's the immediate parent of a GPU GroupBy.** The
   current cost model routes filter to CPU unconditionally. For
   filter→groupby pipelines, fusing the filter into the GPU path
   eliminates the post-filter MetalBuffer round-trip. *Owner:* M3 cost-
   model extension.

4. **Larger workloads.** At 100M rows the build phase scaling becomes
   genuinely painful for the CPU HashMap; the GPU advantage should widen.
   Worth re-measuring before deciding M3 priorities. *Owner:* M3 pre-
   planning.

Recorded as an explicit M3 design input. The empirical evidence is in
`tests/bench/baseline.json` (entries `tpch_q1_modified`,
`tpch_q1_modified_32bit`, `tpch_q1_modified_32bit_high_card`).

### M3 Phase 5b spike — Single-pass global-atomic GPU hash table doesn't work on Apple Silicon yet (M3, 2026-05-26)

After Phase 4 (A1 partitioned-hash) and Phase 5 (A2 sort-then-segment)
shipped, profiling showed A2 is strictly worse than CPU at every tested
cardinality (10M × 65K: A2 2.09s vs CPU 211ms). The Phase 5b spike asked
whether a single-pass global-atomic GPU hash table — different algorithm
from A1's partitioned design — might fill the gap.

**Result: the spike's negative finding.** The kernel
(`shaders/groupby_global_hash.metal`) and dispatch
(`crates/polars-metal-kernels/src/groupby_global_hash/`) compile and run
on Apple Silicon, **don't deadlock** (Phase 4's spin-wait risk was
mitigated by the global hash spreading collisions across the entire
table rather than a small TGSM), but **produce wrong results**: the test
`ten_thousand_unique_keys_in_hundred_thousand_rows` observed 20,697
groups for 10,000 unique keys (~2× inflation).

**Root cause:** the MSL toolchain (32023.883) only accepts
`memory_order_relaxed` on atomic ops; `acquire`/`release`/`acq_rel` are
rejected at compile time. The kernel needs to write the non-atomic
`slot_key` between CAS-claim and state-publish, but without acquire/
release the slot_key write isn't reliably visible to peer threads when
they observe the state publish — peers read stale zeros, fail the key
match, and probe to the next slot, where they also write their key. Same
key ends up in multiple slots, inflating the group count.

**Mitigations tried (none work on this toolchain):**
- `acquire` / `release` orderings on individual ops → MSL compile error
- `__atomic_thread_fence` exists but only accepts `memory_order_relaxed`
- `threadgroup_barrier(mem_flags::mem_device)` requires non-divergent
  control flow; our threads return at different points (no go)
- 128-bit atomic key+gid pack → MSL has no `atomic_ulong2`
- 64-bit hash summary in state → 32-bit collisions at high cardinality
  (~100 false-positives per 1M keys) are unacceptable

**Kept in tree as a documented experiment.** The kernel + smoke tests
(`#[ignore]`'d so CI stays green) serve as the regression-detection
signal: re-run with `--ignored` after any MSL toolchain bump. When Apple
ships acquire/release, the failing test will start passing and A3 can
be productionized.

**Implications for M3:**
- A1 (post-Phase 4.5 optimization) is the only viable GPU build path
  shipping in M3. Phase 6 router uses A1 in its narrow win band
  (≥ 1M rows, est_cardinality ≤ A1's overflow ceiling) and CPU
  everywhere else.
- A2 stays in tree but is not wired into the router.
- The "different algorithm for high cardinality" question (>16K groups
  where A1 overflows) is *unresolved* on this toolchain. The Phase 6
  router falls back to CPU for that range.

*Owner:* M4 (revisit when Apple ships better atomics) or earlier if a
fundamentally different algorithm avoids the memory-ordering trap.

---

## Correlation matrix has no engine hook (M4 Task 29) — RESOLVED: defer to Phase 10 (2026-06-01)

**Question:** How do we expose the matmul-shaped correlation-matrix win
(survey: 7.8×) through the engine, given the engine's only opt-in is
`collect(engine=MetalEngine())`?

**Findings:**
- `df.corr()` is **eager** — returns a `DataFrame` with no `collect` and no
  `engine=` parameter. Nothing to intercept the way we intercept `collect`.
- `pl.corr(a, b)` is **invisible to the NodeTraverser**: `view_expression` on
  a `corr` node raises `NotImplementedError: corr` (py-1.40.1). So even
  `df.lazy().select(pl.corr(...)).collect(engine="metal")` can't be walked.
- The corr **matrix** (standardize → X^T X) is a DataFrame→NxN-matrix op, not
  expressible as a Polars expression at all.

**Options weighed:** (1) monkey-patch eager `df.corr()` behind an explicit
`enable_corr()`/context-manager; (2) new public `polars_metal.corr_matrix(df)`
function; (3) defer, capture the matmul win via Phase 10; (4) upstream a
Polars change exposing `corr` to the visitor.

**Decision (dbfclark):** **(3) Defer Task 29.** #2 violates CLAUDE.md's
"engine plugin is the only user-facing surface / no new public API"; #1 adds
always-on/context magic to a core eager method (a real departure from the
per-call `engine=` model); #4 depends on upstream. The matmul lever moves to
**Phase 10** (`Array[F32, D].dot(lit)` → MLX matmul), a walkable expression
that fits the plugin — so the matmul FFI/kernel work is not wasted. Revisit
only if a headline `df.corr()` number is later required (then prefer #1 with
explicit activation) or upstream makes `corr` walkable (#4).

---

## NodeTraverser opacity blocks list/array/corr/FFT recognition (M4 Phase 10/11) — DEFERRED (2026-06-01)

**Finding (py-1.40.1).** The engine-plugin `NodeTraverser` — the walker's only
window at the post-optimization callback — exposes only a fixed "core"
expression set via `view_expression`: `Column`, `Literal`, `BinaryExpr`,
`Function` with a viewable `function_data` tuple (sin/cos/log/.../cum_sum/
`as_struct`), `Cast`, `Agg`, `Ternary`, `Sort`. Everything else raises:
- list-namespace exprs  -> `"list expr"`
- array-namespace exprs -> `"array expr"`
- `pl.corr`             -> `"corr"`
- `map_batches`/`map_elements` -> `"anonymousfunction"`
- `reshape`             -> `"reshape"`
- top_k/bottom_k null-drop -> dynamic predicate (unviewable; see Task 27)

`PyExprIR` carries only `.node`/`.output_name` (no serialize). So the walker
**cannot recognize** the expressions Phase 10 (list/array `.dot`) and the
plan's Phase 11 FFT (`map_batches` placeholder) depend on. Under
`engine="metal"` these **fall back to CPU cleanly today** (correct results, no
GPU win). Also: the plan's `pl.col(...).arr.dot(lit)` API does not exist in
this Polars version (real dot shapes are `list.eval(element()*lit).list.sum()`
and `(arr * arr_lit).arr.sum()`, both unviewable).

**Escape hatches that exist (not where the walker is):**
- `expr.meta.serialize(format="json")` and `lf.serialize(...)` DO expose
  list/array/corr/eval — but only at the **pre-optimization LazyFrame**, not at
  the post-opt NodeTraverser the engine callback receives.
- `nt.get_dtype(node)` works even on opaque nodes (returns the output dtype).

**FFT-specific unlock (discovered, not yet built):** `pl.struct([...])` IS
viewable (`Function('as_struct')`, viewable field inputs). A *hybrid* sentinel
`pl.struct([col("x").alias("real"), col("x").map_batches(_raise).alias("imag")])`
is walker-routable (view `as_struct`, read field[0]=`Column(x)` for the input,
field[1] opaque marker, confirm via output dtype `Struct{real,imag}`) AND
raises cleanly on CPU (the `map_batches` field). This makes a custom
`.metal.fft()` recognizable. It only works for **struct-output** ops, so it
unblocks FFT but NOT scalar-output dot or the corr matrix.

**Decision (dbfclark): record + defer.** M4 is treated as complete on the
walker-visible compute class (haversine, Black-Scholes, std/var/sum/mean
reductions, sort/top-k, cumsum, and on-GPU null handling — all shipped). The
remaining matmul (Phase 10) and FFT (Phase 11) phases are deferred. When picked
up:
- **FFT** has a clear path: viewable `as_struct` sentinel + walker recognition +
  new Rust/FFI for MLX complex -> two F32 arrays -> Polars `Struct` readback
  (the `Fft` op is already wired in `fusion/subgraph.rs`; only the complex
  readback + Struct assembly is missing).
- **list/array dot + corr** need a different mechanism: a pre-optimization
  plan-capture recognizer (`lf.serialize` exposes them), a viewable data layout
  (D separate F32 columns -> MAC chain through the existing fused path), or an
  upstream polars-python change exposing these exprs to the visitor.

---

## Null-bearing chain reductions: GPU not worth it (measured 2026-06-02)

**Question:** a chain reduction over a null column (e.g. `(x.log().exp()).sum()`
with nulls) can't be replayed from the wire plan (the fused scope is GPU-only;
no CPU-evaluable AST). Should we compute it on the GPU instead of falling back?

**Tried two builds:**
1. *Naive* (chain → null-correct Polars Series via the HStack path → host
   reduce): **65 ms vs 44 ms CPU** at 10M for `log().exp().sum()` — a loss. The
   cost is null marshalling: `to_numpy()` on a null F32 column NaN-injects
   (~27 ms), `pa.array(out, mask=...)` builds the masked Series (~20 ms).
2. *Optimized* (raw Arrow F32 buffer in, skipping NaN-inject; numpy `where=`
   reduction, skipping the Series build): `log().exp().sum()` 28 ms (**1.5×**,
   a win), but `(x*3).min()` 27 ms vs 5 ms CPU (**0.2×**) and `(x*2+1).std()`
   46 ms vs 16 ms (**0.4×**) — still losses. Irreducible overhead: `is_null`
   mask (~8 ms) + the host reduction itself (numpy min ~5 ms, std two-pass
   ~20 ms). Only **heavy chain + cheap reduction** clears it. Worse, raw
   buffers are **incorrect for `where` chains**: a null cond keeps the `else`
   branch *valid* (value 0.0), but the GPU chain over garbage computes the
   `then` branch there → wrong result. `where` chains need the NaN-inject
   path, which erases the win.

**Decision (dbfclark): fall back null chains to CPU** (the behavior shipped in
b2ae73a). Capturing only the winning slice (heavy elementwise chain + sum/min/
max) would need an elementwise-only + compute-density gate + a `where` carve-
out — too much machinery for a ~1.5× win on one shape. Null-*free* chains keep
their big GPU wins (Increment 2). If revisited, the lever is reducing the ~8 ms
`is_null` cost and host-reduction cost, or a density gate.

**Resolved (2026-06-02): do the null handling in Polars via `drop_nulls`.** The
masked attempts paid to *preserve null positions* (NaN-inject + masked-Series
build). But a reduction *skips* nulls — positions are irrelevant and there's
nothing to rejoin (output is a scalar). So for an **elementwise** null chain,
let Polars `drop_nulls(subset=cols)` compact the nulls away (~4.5 ms native),
hand the GPU the dense survivors (zero-copy, no NaN-inject), reduce → scalar.
Measured: `log().exp().sum()` **4.6×**, `(x*2+1).std()` **1.6×** (trivial chains
like `x*3 → min` ~0.6×, a small rare loss — could density-gate later). All match
CPU. `where` chains still fall back (a null cond keeps the else branch valid, so
dropping the row is wrong). The key realization: `drop_nulls` rescues
*reductions* (no rejoin) but not HStack (row-shaped output must preserve null
positions — that's the genuine "rejoin" cost).
