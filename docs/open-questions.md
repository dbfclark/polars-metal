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
