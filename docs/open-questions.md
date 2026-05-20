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
- The hypothesis differential harness in `tests/diff/` generates bare scans (no varied operations), so it's only really testing fallback parity. Add filter/select/groupby strategies once M1 has kernels producing real GPU output.
