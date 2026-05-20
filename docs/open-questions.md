# Open questions

Questions and risks that don't block current work but warrant tracking. Each entry has a one-line title, the open question, and where it gets owned.

## Monkey-patch coupling

The M0 patch turned out to need two sites, not one:
1. `polars.lazyframe.frame._gpu_engine_callback` — defensive wrap so MetalEngine reaching this function returns our callback rather than raising on the engine-string allow-list.
2. `polars.lazyframe.frame.LazyFrame.collect` — primary intercept. Polars 1.40+ routes `engine=` straight to Rust's `ldf.collect()`, which only accepts strings, so we short-circuit at the Python level: when `engine=MetalEngine()`, call `original_collect(self, engine="cpu", post_opt_callback=cb)` with our callback. `post_opt_callback` is an internal/test-only Polars hook.

Only site 1's signature is asserted at import time. Site 2 (`LazyFrame.collect`) is not yet asserted — if its signature changes (e.g., `post_opt_callback` is removed/renamed), our patch silently breaks. *Follow-up:* add a signature assertion for `LazyFrame.collect` and a smoke test that asserts `post_opt_callback` exists. *Owner:* M0 follow-up / pre-M1.

The real fix is an upstream engine-registration hook in Polars; once that lands, both patches retire. *Owner:* see `docs/upstream-polars-engine-hook.md` (drafted in T37).

## MLX null story

MLX is dense-numeric-first and has no first-class null representation. Reductions in M2 need either custom MSL null-aware kernels or MLX-op-on-data + separate-validity-reduction. *Owner:* M2's design spec.

## Hash groupby GPU performance

The biggest single unknown in M0–M3. Two-pass count-then-fill is the textbook approach (cuDF reference: `references/cudf/cpp/src/groupby/`). *Owner:* M3's design spec.

## Memory pressure on base M1

M2 Ultra has plenty of memory; M1 (8 GB) does not. We don't plan spill-to-CPU in M0–M3, but the portability gate catches OOM regressions. *Owner:* future, post-M3.

## Buffer-bridge alignment assumptions

We assume Arrow `Buffer`s sometimes arrive page-aligned and sometimes don't. The bridge handles both via two regimes. If profiling shows we're nearly always in the copy regime, we may want to influence Arrow allocations to land page-aligned. *Owner:* future, after M1 perf data.

## FFI choice (cxx)

We committed to `cxx` for M0 based on a weak prior. If friction emerges before M2, switch to hand-written C shim + `bindgen`. *Owner:* M0 (revisit at M2's design).

## MLX install path

Resolved in T19: git submodule under `vendor/mlx`, pinned to v0.22.0, built via cmake. *Owner:* M0 (resolved). Refresh via standalone cmake invocation, not via `scripts/refresh-references.sh` (which is for read-only references, not build deps).

## Conformance harness scale

`pytest tests/conformance` (14 tests, 7 operations × cpu+metal) ran in 0.461s wall-clock on an M-series Mac (measured T33, 2026-05-20). Will grow as M1+ ops land; keep an eye on suite time crossing 60s and add pytest-xdist parallelism if needed.

## Metal toolchain (resolved)

Initially missing on the dev host: T19's MLX build had to use `-DMLX_BUILD_METAL=OFF` because the Metal shader compiler couldn't be invoked. Root cause was a stale x86_64 `CoreSimulator.framework` under `/Library/Developer/PrivateFrameworks/` blocking Xcode's plugin loading. Resolved 2026-05-20 by running `sudo xcodebuild -runFirstLaunch`, then `xcodebuild -downloadComponent MetalToolchain`. MLX rebuilt with `-DMLX_BUILD_METAL=ON`; `build.rs` now links `Metal`, `Foundation`, `QuartzCore`, `Accelerate` frameworks. *Owner:* M0 (resolved).
