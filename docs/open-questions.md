# Open questions

Questions and risks that don't block current work but warrant tracking. Each entry has a one-line title, the open question, and where it gets owned.

## Monkey-patch coupling

We monkey-patch `polars.lazyframe.frame._gpu_engine_callback`. The patch site is asserted at import time, but the callback signature could shift under us. *Owner:* M0 (patch-site assertion test). *Resolution path:* upstream a proper engine-registration hook to Polars; replace the patch with the hook once it lands.

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

## Metal toolchain unavailable on dev host

T19 built MLX with `-DMLX_BUILD_METAL=OFF` because the host's Metal shader compiler is missing (macOS 26.5 beta + Xcode toolchain ABI mismatch surfaced by `xcodebuild -downloadComponent MetalToolchain`). MLX runs on CPU/Accelerate. For M0 this is fine — M0 falls back to CPU on every IR node and the MLX bridge is only tested in isolation, where CPU MLX is sufficient for FFI correctness. **Must be resolved before M1 begins**, since M1's first GPU kernel (filter) needs real Metal dispatch. Resolution path: install an Xcode version compatible with the host macOS, then rebuild MLX with `-DMLX_BUILD_METAL=ON` and switch `build.rs` to `cargo:rustc-link-lib=dylib=mlx`. *Owner:* pre-M1.
