# M5 Rolling — Custom MSL Kernel Design

**Status:** Approved (brainstorm 2026-06-02). Supersedes the rolling-via-cumsum-diff
mechanism in [`2026-06-02-m5-rolling.md`](../plans/2026-06-02-m5-rolling.md).

**Goal:** Transparently accelerate native `pl.col(x).rolling_{sum,mean,var,std}(window)`
under `engine="metal"` via a custom, numerically-stable Metal kernel, matching Polars
CPU within a tight tolerance — or falling back to CPU (correct) for anything the kernel
doesn't handle.

---

## Why this design (decision record)

The original M5 plan rewrote `rolling_*` into a cumsum-diff expression
(`(c - c.shift(w))`, off-by-one fixed with `when(int_range(len)==w-1)`) and routed it
through the MLX fusion walker. Two findings during execution killed that approach:

1. **`int_range` is opaque to the NodeTraverser** (`view_expression` raises
   `NotImplementedError: range`; it serializes as `Range`/`IntRange`, while `Len` is
   viewable). The walker therefore cannot recognize the off-by-one fix, so the rewrite
   would always fall back to CPU — zero speedup.
2. **F32 cumsum-diff is numerically unsound.** `c[i] - c[i-w]` differences two *global*
   prefix sums whose magnitude grows to ~`N·mean`; stored in F32 each carries absolute
   error ~`N·mean·2⁻²⁴`, which survives the subtraction (catastrophic cancellation
   scaling with `N`). This violates CLAUDE.md's "no approximately equal" principle, and
   the accumulation-order fix (F64) is unavailable on Apple Silicon / MLX.

A custom kernel computes the windowed statistic **directly**, keeps accumulation
magnitudes **block-bounded** (not `N`-scaled), owns the window `w` (so head-nulls and the
first-full-window are structural, no `int_range`), and needs no MLX graph. It therefore
**obviates** the general serialize→scope analyzer, `int_range`→`RowIndex` recognition, and
the cumsum-diff graph — all explicitly **dropped from M5** (deferred to a future opacity
subproject for `dt`/list/array/`fft`/`corr`).

**Decisions (architect-approved):**
- Custom MSL kernel; general serialize→scope analyzer **dropped** from M5.
- **F32-only** acceleration. F64 → CPU (no Metal/MLX f64). Integer columns → CPU
  (engine is F32-only; i64 sums overflow F32's 24-bit mantissa anyway).
- **Streaming out of scope, guarded.** The polars-metal adapter is in-memory-only by
  construction (the `collect` patch forces `engine="cpu"`); it does not integrate with
  Polars' streaming executor. If streaming is requested, the rolling rewrite is skipped
  (query runs correctly via plain Polars, just unaccelerated). A genuinely streaming
  rolling (online windowed kernel with carried state) is a separate future subproject.
- **Tile-blocked O(N)** algorithm (over the simpler O(N·w) direct sum) for the large-`w`
  perf target. **Shifted-data variance** (over two-pass) for var/std stability.

**Already shipped, kept, now unused by rolling** (do not revert): the `Shift` op
(T1–T4), `mlx_iota_f32`+`RowIndex` (4b/4c), and the NodeTraverser-side `shift` recognition
+ structural-null validity (Task 5 — retains independent value: raw `pl.col(x).shift(n)`
under `engine="metal"` matches CPU). See [[m5-shift-validity-design]].

---

## Architecture

```
df.collect(engine=MetalEngine())
  └─ collect_wrapper (__init__.py):
       1. if streaming requested → skip (guard); plain Polars handles rolling on CPU
       2. bindings = find_rolling_bindings(lf)        # focused serialize parse
       3. if none → unchanged collect
       4. split: lf_rest = lf without the rolling output columns
       5. df = lf_rest.collect(... existing in-memory metal path ...)
       6. for each binding: result = execute_rolling(df[col].rechunk(), w, op)
                            df = df.hstack(result_series)   # first w-1 null
       7. restore original column order; return df
```

Execution is **collect-and-stitch over fully-materialized, whole columns** — chunk-safe by
construction (no `map_batches`, no streaming morsels). The MLX fusion walker is untouched.

### Component boundaries

| Unit | Responsibility | Interface | Depends on |
|------|----------------|-----------|------------|
| `shaders/rolling.metal` | Windowed F32 statistics, tile-blocked, stable | `rolling_sum_f32`, `rolling_var_f32` entry points | `_validity.metal` (head-null bitmap helpers) |
| `crates/polars-metal-kernels/src/rolling.rs` | Dispatcher: bind buffers, params, tiling, dispatch | `rolling_sum_f32(in, n, w) -> out`, `rolling_var_f32(in, n, w, ddof) -> out` | `command.rs`, `shader_lib.rs`, buffer bridge |
| PyO3 binding (`fusion/py.rs` or new module) | Expose dispatcher to Python; wrap result+validity into a Series | `execute_rolling(col_buf, w, op) -> Series-ready (data, valid)` | kernels crate, buffer bridge |
| `python/polars_metal/_rolling_detect.py` | Find handleable `rolling_*` bindings from `lf.serialize` | `find_rolling_bindings(lf) -> list[RollingBinding]` | Polars serialize JSON |
| `python/polars_metal/__init__.py` (collect wrapper) | Streaming guard; orchestrate detect → collect-rest → dispatch → stitch | — | the above |

---

## The kernel (`shaders/rolling.metal`)

Port the algorithm from `references/cudf/cpp/src/rolling/` (read first, per CLAUDE.md);
adapt to F32-stability via tile-blocking. Include `_validity.metal` for the head-null
bitmap. Document threadgroup/grid assumptions at the top of the file.

**Tiling.** Each threadgroup owns `T` output rows. It loads
`input[tile_start-(w-1) .. tile_start+T)` (a `w-1` left halo) into threadgroup memory.
`T` is derived at runtime from the device's max threadgroup memory (query `MTLDevice`;
never hardcode — M1–M4 differ). The tile size bounds accumulation magnitude to ~`T·mean`,
making the differencing error `T`-scaled rather than `N`-scaled.

**`rolling_sum_f32` (and mean).** Cooperative tile-local **inclusive prefix sum** `P`
over the loaded region; output `i` = `P[local_i] − P[local_i − w]`. O(N) regardless of
`w`. For `mean`, scale by `1/w` (in-kernel via an op flag, or a trivial post-step).

**`rolling_var_f32` (and std).** **Shifted-data** variance to avoid mean-dominated
cancellation: pick a tile-local shift `k` (the tile's first loaded value), accumulate
tile-local prefix sums of `x' = x − k` and `x'²`; for each window
`S1' = Σx'`, `S2' = Σx'²` over the `w` values, then
`var = (S2' − S1'²/w) / (w − 1)` (ddof=1, Polars default), `std = √var`.
Shift-invariance makes this exact; centering near zero keeps `S2' − S1'²/w ≈ w·var`.
(Fallback if a differential test shows drift: two-pass per window — window mean, then
`Σ(x−μ)²` — bulletproof, O(w)/output.)

**Head nulls.** First `w−1` outputs are structurally null (input is null-free, gated).
The dispatcher writes the validity bitmap host-side (first `w−1` bits clear, rest set)
using `_validity.metal` conventions — no value-graph validity, no `int_range`.

**Guard.** If `w > T_max` (window can't fit one tile), the detector rejects that binding
→ CPU. Also reject `w < 1`.

---

## Detection (`_rolling_detect.py`)

`find_rolling_bindings(lf) -> list[RollingBinding(op, column, window, out_name, ddof)]`,
parsing `lf.serialize(format="json")`. Return a binding **only** when all hold (else omit
→ native Polars → CPU):

- function is `rolling_mean` / `rolling_sum` / `rolling_var` / `rolling_std`;
- argument is a **bare column reference** (not a computed sub-expression);
- column dtype is `Float32`;
- the source column is **null-free** (mirror M4's `_fused_inputs_null_free` check);
- **default options**: no `weights`, `center=False`, `min_samples`/`min_periods`
  unset (== window), no `by`/temporal rolling;
- `1 ≤ window ≤ T_max`.

**Open item (resolve in plan, pin in a comment):** the exact JSON node shape for
`rolling_*` at the pinned Polars rev — verify empirically (`expr.meta.serialize`) before
relying on field names. The serialized format is **deprecated** and the logical IR is not
version-stable (CLAUDE.md gotcha); add an import-time **version-probe guard** (in the
spirit of `_verify_patch_site`) that disables the rolling rewrite if the schema it expects
isn't present, so a Polars bump degrades to CPU rather than mis-parsing.

---

## Dispatch & fold-back (collect wrapper)

1. **Streaming guard.** If `streaming` / `new_streaming` is requested (or the future
   `engine="streaming"`), skip the rewrite entirely.
2. **Split.** Build `lf_rest` = the LazyFrame without the recognized rolling output
   columns (keep any sibling expressions in the same `with_columns`). If the binding
   can't be cleanly removed/reconstructed, don't rewrite it.
3. **Collect** `lf_rest` via the existing in-memory metal path.
4. **Dispatch.** For each binding: `s = df[column].rechunk()` (contiguous F32 buffer for
   zero-copy + correct whole-column stats), `execute_rolling(s, w, op)`, build the result
   Float32 Series with the first `w−1` rows null.
5. **Stitch** results onto `df`; restore the original output schema/column order.

`.rechunk()` is mandatory — the buffer bridge needs a single contiguous F32 buffer, and it
guarantees the kernel sees the whole column.

---

## Testing

1. **Kernel correctness** — `crates/polars-metal-kernels/tests/test_rolling.rs`: each
   entry point vs an exact F64-computed reference, across `w ∈ {1, 2, N}`, single-tile,
   multi-tile, and windows straddling tile boundaries; null-free inputs; assert a **tight**
   tolerance (the blocking exists precisely to make this pass — e.g. `|metal − f64ref|`
   within a few ULP of the F32 result).
2. **Differential vs Polars CPU** — `tests/python_integration/test_rolling_*.py`: random
   `n`/`w`/values × {sum, mean, var, std}, `engine="metal"` == CPU within `rtol/atol`;
   head nulls match (first `w−1` null); var/std use ddof=1.
3. **Fallback** — nulls / `min_samples` / `center` / `weights` / non-F32 / `w > T_max` /
   streaming → not rewritten (dispatch count 0), result equals CPU.
4. **Conformance** — `make test-conformance` stays at baseline (`lazyframe` +
   `operations_group_by` known failures only).
5. **Bench** — `tests/bench/m4_survey/bench_rolling_mlx.py` engine path at 10M F32,
   `rolling_mean(1000)`; flip `phase9_rolling_mean_w1000_10m` in `baseline.json` from
   `_pending` to measured with a `ratio_lt` gate (target ~10–18×).

---

## Conventions / constraints

- Rust 2021; no `unwrap()` outside tests; no `unsafe` outside `*-sys`/buffer (with
  `// SAFETY:`); errors → `PolarsError::ComputeError` at the engine boundary.
- One MSL kernel family per file; document threadgroup/grid assumptions at the top.
- Don't add a `shaders/` file without a `tests/kernel/` (kernels-crate) test.
- Null semantics match Polars exactly; differential test on null-bearing inputs (which
  must fall back) confirms parity.
- `make lint` / `make gate` before declaring done.

## Out of scope (explicit)

- Streaming / out-of-core rolling (adapter isn't a streaming engine; needs an online
  kernel — future subproject).
- F64 / integer rolling acceleration (CPU fallback).
- The general serialize→scope analyzer, `int_range`→`RowIndex` recognition, cumsum-diff
  graph (deferred opacity subproject).
- Centered / weighted / `min_samples`-override / temporal (`by=`) rolling (CPU fallback).
