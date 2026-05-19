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

Choose between git submodule, Homebrew, or build-from-source for the MLX C++ dependency. *Owner:* M0's implementation plan (Task 19).
