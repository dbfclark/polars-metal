# B2 — Integer Reduction Parity Implementation Plan

**REQUIRED SUB-SKILL: superpowers:subagent-driven-development** — execute this plan task-by-task via dispatched implementer subagents; each task is a self-contained TDD loop (failing test → run/FAIL → minimal impl → run/PASS → commit), and the orchestrator reviews each task's diff before dispatching the next.

**Goal:** Make the fused-reduction path carry an integer column's declared dtype end-to-end — for the **GPU-admissible** integer (op, dtype) combos only — instead of hard-coding `F32`. Concretely, admit on the GPU:

- integer **`sum`** for `{Int32, Int64, UInt32, UInt64}` (MLX-native reduction dtype == Polars: `int32→Int32`, `int64→Int64`, `uint32→UInt32`, `uint64→UInt64` — *no cast needed*), and
- integer **`min`/`max`** for **all 8** integer widths (MLX preserves input width == Polars preserves input width).

Everything else integer — **narrow `sum`** (`Int8/Int16/UInt8/UInt16`, which Polars upcasts to Int64/UInt64 but MLX widens to int32/uint32 → MISMATCH), and **`mean`/`std`/`var`** over any integer (MLX→float32, Polars→Float64 → MISMATCH) — **falls back to CPU** (the analyzer aborts the reduction so the walker routes it to the conformance GroupBy/CPU path). This is correct, not a failure (CLAUDE.md Non-goals: "Unsupported ops fall back to CPU").

The "no-cast" framing is what makes B2 clean: there is **no `CastI64`/`CastU64` op** in the subgraph (only `CastF32`/`CastF64`/`CastI32`/`CastBool` exist), so B2 admits *exactly* the op×dtype pairs where the MLX-native reduction dtype already equals the Polars output dtype — no fold-back cast, no new op.

**Null handling — reuse the existing mechanism, do NOT bypass to CPU.** The fused-reduction path already has a GPU null path (B1/M4):
- **Chain reductions** over a null column: the walker stamps `_drop_nulls`; the dispatch does `upstream.drop_nulls(subset=cols)` on CPU (one pass) then reduces the **dense survivors on the GPU** (`_udf.py::_build_select_reduction_fused`, the `is_chain` / `_drop_nulls` branch). Lossless — a reduction skips nulls, the output is a scalar so positions don't matter, and `count = frame.height` is the non-null count. B2's int chain reductions **reuse this exact path** unchanged.
- **Bare single-column** null reductions fall to CPU (the `if n < 2 or series.null_count() > 0` branch — `getattr(series, kind)()` on the source column). B2 **keeps that for int too** (it already produces the exact Polars dtype/value because it calls Polars on the source `series`).

No sentinel-fill for reductions — `drop_nulls` is the right tool (and the only one that gives the non-null count std/var would need; those are F32-only and thus unaffected here).

**Exit bar (the acceptance criterion for B2):** for `sum` over `{Int32, Int64, UInt32, UInt64}` and `min`/`max` over all 8 integer widths, a reduction (`df.select(pl.col("x").<op>())`) under `engine="metal"` is **byte-exact** vs Polars CPU (`got.equals(want)` — same dtype, same value), across null-free / null-bearing-chain / empty / single-element inputs; **at least one case genuinely exercises the GPU int-reduction path** (verified via the `_dispatches` dispatch counter); and narrow-int `sum` / int `mean` correctly **fall back to CPU** and still match.

**Out of scope (do NOT plan these):**
- B3 — the `dt` gregorian MSL kernel.
- B4 — bare-reduction routing thresholds (the compute-intensity gate at `_walker.py::_try_fused_select_reduction` line ~453 stays as-is; B2 only ensures correctness *when* a reduction is routed to GPU).
- Bool-final-output reductions; narrow-int `sum` upcast to Int64/UInt64 on the GPU (would need a `CastI64`/`CastU64` op — explicitly deferred, narrow sum stays CPU); int `mean`/`std`/`var` on the GPU (MLX→f32 vs Polars→Float64 mismatch — stays CPU).
- The elementwise HStack path (`analyze_ir_with_columns`) — that was B1; B2 touches only the *reduction* path (`analyze_ir_reduction`).

---

## Architecture

The fused-reduction path for `df.select(pl.col("x").sum())` flows:

`_walker._walk_select_reduction` → `_try_fused_select_reduction` (per-agg) → `_fusion_analyzer.analyze_ir_reduction(nt, agg_node_id, in_schema)` builds a `PyFusionScope` whose single output is the reduction op (`Sum`/`Min`/`Max`/…), returns `(scope, descriptors, agg_kind, is_chain, arg_id)` → the walker stamps a binding dict (`_fused_scope`, `_fused_columns`, `_agg_kind`, `_is_chain`, `_null_mode`) → `_route_fused_reduction` decides `_drop_nulls` per chain → at dispatch, `_udf._build_select_reduction_fused` stages inputs as `np.float32`, calls `_native.execute_fused_expr` with an `np.float32` `out_arr` + tag 0, reads back a scalar, applies the Bessel correction (std/var only), and builds a `pl.Series(name, [val], dtype=pl.Float32)`.

B2 changes three hard-coded `F32` assumptions in that path:

1. **`analyze_ir_reduction`** — the F32-only gate (`if any(d_kind == "col" and schema.get(payload) != pl.Float32 ...): raise _Aborted`) is *relaxed* to admit the GPU-admissible int (op, dtype) combos, and the function now **infers and returns the reduction output dtype** as a wire string (`out_dtype_str`) — `"F32"` for the existing float path, `"I32"`/`"I64"`/`"U32"`/`"U64"`/etc. for an admitted int reduction. It aborts (→ CPU) for the non-admissible combos.
2. **`_build_select_reduction_fused`** — stages each column at its native width (reusing B1's `_series_input_dtype_str` / `_np_dtype_and_tag`), pre-allocates `out_arr` at the reduction's output dtype, passes the right tags through `execute_fused_expr`, and builds the result `pl.Series` at the output dtype. The `drop_nulls` / bare / Bessel logic is untouched (Bessel is std/var-only → F32-only → unaffected).
3. **`_walk_select_reduction` / `_try_fused_select_reduction`** — stamps the analyzer's inferred output dtype on each binding (`_fused_out_dtype`) and adds an **`int_fused_ok` safety check** (mirroring B1's HStack defense): admit an int reduction onto the GPU only when the analyzer's inferred wire tag corresponds to the *real* Polars output dtype (`nt.get_dtype(agg_node_id)`); otherwise fall back. This guards against an analyzer mis-inference corrupting the result dtype.

**No Rust change is needed.** Verified while drafting:
- The reduction output dtype on the eval path comes from **MLX itself** via `mlx_array_dtype` inside `MlxSubgraph::eval_into_typed` (B1), which *asserts* the eval'd dtype == the declared output tag and then writes back width-aware. `execute_fused_expr` already takes typed `(ptr, n, tag)` I/O (B1). So passing the correct output tag from Python is sufficient.
- `DtypeOut::ScalarF32` in `supported_ops.rs` (the `Sum | Mean | Min | Max => OpSpec { output_dtype: O::ScalarF32, .. }` arm) does **not** force F32 on the eval path. `op_spec(...).output_dtype` is consulted in exactly two places — `scope.rs::est_flops_for` (routing FLOPs estimate) and `py.rs::push_op` (arg-count validation) — **never** to coerce the output dtype. The subgraph `build_op` arm for `Sum`/`Min`/`Max` (`subgraph.rs:504-507`) wires `mlx_sum`/`mlx_min`/`mlx_max`, which produce MLX's native int reduction dtype; `eval_into_typed` reads it back and asserts against the Python-declared tag. This is the same "Python infers the output dtype, Rust validates via the `mlx_array_dtype` assertion" contract B1 established for HStack — reductions ride it for free. **B2 is Python-only.** (Task 4 verifies this empirically rather than assuming it.)

## Tech Stack

Python 3 + Polars (the differential oracle) + numpy + pyarrow + pytest. The Metal engine is reached via `engine=MetalEngine()`. No Rust edits (verified above) → **no `make wheel` rebuild is required** for B2's code changes; the existing B1 wheel already exposes the typed `execute_fused_expr`. (Task 4 runs `make wheel` once defensively only if a verification step surfaces an unexpected Rust dependency — the plan's default assumption is no rebuild.) All reused dtype machinery (`_DTYPE_STR_TO_NP_AND_TAG`, `_np_dtype_and_tag`, `_series_input_dtype_str`, `_POLARS_DTYPE_TO_WIRE` in `_udf.py`; `_INT_TAG_TO_POLARS` in `_walker.py`; the drift-guard test `test_dtype_tag_tables_match_canonical_mlx_dtype`) is B1's — **do not duplicate it**.

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `python/polars_metal/_fusion_analyzer.py` | Modify | `analyze_ir_reduction`: relax the F32-only gate to admit GPU-admissible int (op,dtype) combos; infer the reduction **output dtype** (wire string) per the admit rules; return it as a 6th tuple element; abort (→ CPU) for the non-admissible int combos. Add the admit-rule helper `_int_reduction_out_dtype`. |
| `python/polars_metal/_udf.py` | Modify | `_build_select_reduction_fused`: stage int columns/literals at native width (reuse `_series_input_dtype_str`/`_np_dtype_and_tag`), allocate `out_arr` + pass tags at the reduction's output dtype, build the result `pl.Series` at the output dtype. Both bare and chain/`drop_nulls` paths. |
| `python/polars_metal/_walker.py` | Modify | `_try_fused_select_reduction`: capture the analyzer's `out_dtype_str`, stamp it on each binding (`_fused_out_dtype`); add the `int_fused_ok` safety check (`out_dtype_str` int tag must match `nt.get_dtype(agg_node_id)`), else abort that select to CPU. |
| `tests/python_integration/test_int_reductions.py` | Create | The B2 differential matrix: `sum/min/max` × admitted int dtypes × {null-free, null-chain, empty, single}, byte-exact vs CPU; ≥1 GPU-path case via the `_dispatches` counter; narrow-`sum` / int-`mean` fall-back-to-CPU-and-match cases. |

---

## Task 1 — Analyzer: admit-rule helper + a unit test pinning the (op, dtype) → out-dtype matrix

Add a pure helper `_int_reduction_out_dtype(agg_kind, col_dtype) -> str | None` encoding the *exact* GPU-admit matrix, and lock it with a table-driven unit test. This is the load-bearing decision; isolating it as a tested pure function keeps the gate edit in Task 2 trivial and unambiguous. No engine call here — pure Python.

**Files**
- Modify: `python/polars_metal/_fusion_analyzer.py`
- Test (create): `tests/python_integration/test_int_reductions.py` (the unit-test portion; the differential matrix lands in Task 5)

**Steps**

- [ ] Add a failing unit test to a **new** file `tests/python_integration/test_int_reductions.py` that imports the not-yet-existing helper and asserts the full admit matrix:
  ```python
  import polars as pl
  import pytest

  from polars_metal._fusion_analyzer import _int_reduction_out_dtype


  # (agg_kind, polars dtype) -> expected wire output-dtype string, or None for
  # "not GPU-admissible → CPU fallback". This table IS the B2 scope decision.
  _ADMIT = [
      # sum: admitted only for the four widths where MLX-native == Polars.
      ("sum", pl.Int32, "I32"),
      ("sum", pl.Int64, "I64"),
      ("sum", pl.UInt32, "U32"),
      ("sum", pl.UInt64, "U64"),
      # sum of narrow ints → Polars upcasts to Int64/UInt64, MLX widens to
      # int32/uint32 → MISMATCH → CPU.
      ("sum", pl.Int8, None),
      ("sum", pl.Int16, None),
      ("sum", pl.UInt8, None),
      ("sum", pl.UInt16, None),
      # min / max: preserve input width for all 8 → admitted everywhere.
      ("min", pl.Int8, "I8"),
      ("min", pl.Int16, "I16"),
      ("min", pl.Int32, "I32"),
      ("min", pl.Int64, "I64"),
      ("min", pl.UInt8, "U8"),
      ("min", pl.UInt16, "U16"),
      ("min", pl.UInt32, "U32"),
      ("min", pl.UInt64, "U64"),
      ("max", pl.Int8, "I8"),
      ("max", pl.Int16, "I16"),
      ("max", pl.Int32, "I32"),
      ("max", pl.Int64, "I64"),
      ("max", pl.UInt8, "U8"),
      ("max", pl.UInt16, "U16"),
      ("max", pl.UInt32, "U32"),
      ("max", pl.UInt64, "U64"),
      # mean/std/var of int → MLX→f32, Polars→Float64 → MISMATCH → CPU.
      ("mean", pl.Int32, None),
      ("mean", pl.Int64, None),
      ("std", pl.Int64, None),
      ("var", pl.Int64, None),
  ]


  @pytest.mark.parametrize("kind,dtype,expected", _ADMIT)
  def test_int_reduction_out_dtype_matrix(kind, dtype, expected):
      assert _int_reduction_out_dtype(kind, dtype) == expected
  ```
- [ ] Run it, expect FAIL (helper does not exist → ImportError):
  `pytest tests/python_integration/test_int_reductions.py -v`
  Expected: collection error `cannot import name '_int_reduction_out_dtype'`.
- [ ] Implement `_int_reduction_out_dtype` in `python/polars_metal/_fusion_analyzer.py` (place it just above `analyze_ir_reduction`, after the `_REDUCTION_OP` dict at line ~634):
  ```python
  # B2: GPU-admissible integer reductions. The fused reduction output dtype must
  # equal the Polars output dtype with NO fold-back cast (there is no CastI64 /
  # CastU64 op), so we admit exactly the (op, dtype) pairs where MLX's native
  # reduction dtype already matches Polars:
  #   - sum: MLX keeps the input width; Polars keeps it ONLY for the 32/64-bit
  #     widths (Int32→Int32, Int64→Int64, UInt32→UInt32, UInt64→UInt64). Polars
  #     upcasts narrow-int sum to Int64/UInt64 while MLX widens to int32/uint32 →
  #     mismatch → those stay CPU.
  #   - min/max: both MLX and Polars preserve the input width for all 8 widths.
  #   - mean/std/var: MLX→float32, Polars→Float64 → mismatch → CPU.
  _SUM_ADMIT: dict[Any, str] = {
      pl.Int32: "I32",
      pl.Int64: "I64",
      pl.UInt32: "U32",
      pl.UInt64: "U64",
  }
  _MINMAX_ADMIT: dict[Any, str] = {
      pl.Int8: "I8",
      pl.Int16: "I16",
      pl.Int32: "I32",
      pl.Int64: "I64",
      pl.UInt8: "U8",
      pl.UInt16: "U16",
      pl.UInt32: "U32",
      pl.UInt64: "U64",
  }


  def _int_reduction_out_dtype(agg_kind: str, col_dtype: Any) -> str | None:
      """Wire output-dtype string for a GPU-admissible integer reduction, or
      ``None`` when the (op, dtype) pair is not GPU-admissible (→ CPU fallback).

      Admits int ``sum`` for {Int32, Int64, UInt32, UInt64} and int ``min``/
      ``max`` for all 8 integer widths — exactly the pairs where MLX's native
      reduction dtype equals the Polars output dtype (no fold-back cast). All
      other int reductions (narrow ``sum``, ``mean``/``std``/``var``) return
      ``None`` so the caller aborts to CPU.
      """
      if agg_kind == "sum":
          return _SUM_ADMIT.get(col_dtype)
      if agg_kind in ("min", "max"):
          return _MINMAX_ADMIT.get(col_dtype)
      return None
  ```
  (`Any` and `pl` are already imported at the top of `_fusion_analyzer.py`.)
- [ ] Run it, expect PASS:
  `pytest tests/python_integration/test_int_reductions.py -v`
  Expected: all parametrizations green.
- [ ] Lint:
  `ruff check python/polars_metal/_fusion_analyzer.py tests/python_integration/test_int_reductions.py`
  Expected: clean.
- [ ] Commit:
  `git add python/polars_metal/_fusion_analyzer.py tests/python_integration/test_int_reductions.py && git commit -m "B2 T1: analyzer admit-rule helper _int_reduction_out_dtype + (op,dtype)→out-dtype matrix test"`

---

## Task 2 — Analyzer: relax the F32-only gate + return the inferred output dtype

Wire `_int_reduction_out_dtype` into `analyze_ir_reduction`: relax the F32-only column gate to also admit the GPU-admissible int combos, infer the reduction output dtype, and return it as a 6th tuple element. The float path must keep returning `"F32"` (no behavior change for F32 reductions).

**Files**
- Modify: `python/polars_metal/_fusion_analyzer.py`
- Test (create): a focused analyzer-level test in `tests/python_integration/test_int_reductions.py`

**Steps**

- [ ] Add a failing test that drives `analyze_ir_reduction` through a real `NodeTraverser` and asserts the returned output dtype. (The simplest harness that reaches the analyzer is to run a select-reduction through the engine and assert the inferred dtype via the engine result, but the *direct* analyzer call needs an `nt`/`node_id`/`schema` which only the walker has. So assert at the engine level that the returned-tuple arity change didn't break F32 AND that an admitted int reduction now produces the int dtype — this also covers the integration in Tasks 2+3 together. For a pure-analyzer probe, add a test that imports `analyze_ir_reduction` and checks its return *arity* is 6 by inspecting a constructed-via-engine call is overkill; instead pin the contract at the engine boundary.) Append:
  ```python
  import numpy as np

  from polars_metal import MetalEngine


  def test_int64_sum_returns_int64_dtype_via_engine():
      # An int sum that is admitted (Int64) must come back as Int64, byte-exact.
      df = pl.DataFrame({"x": pl.Series([1, 2, 3, 4], dtype=pl.Int64)})
      lf = df.lazy().select(pl.col("x").sum().alias("s"))
      got = lf.collect(engine=MetalEngine())
      want = lf.collect()
      assert got.equals(want)
      assert got["s"].dtype == pl.Int64


  def test_f32_sum_still_works_after_arity_change():
      # The analyzer-tuple arity grew from 5 to 6; the F32 path must be intact.
      df = pl.DataFrame({"x": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float32)})
      lf = df.lazy().select((pl.col("x") * 2.0).sum().alias("s"))  # chain → GPU
      got = lf.collect(engine=MetalEngine())
      want = lf.collect()
      from polars.testing import assert_frame_equal
      assert_frame_equal(got, want, check_exact=False, abs_tol=1e-4)
  ```
- [ ] Run, expect FAIL — `analyze_ir_reduction` still returns a 5-tuple and `_try_fused_select_reduction` unpacks 5; once we change arity in this task the walker (Task 3) must match, so **this task's two tests stay red until Task 3**. To get a strictly-RED-then-GREEN-per-task cadence, in *this* task run only the F32 test and expect it to FAIL on the unpack mismatch after the analyzer change, then make it pass by also adjusting the single walker unpack site (the minimal coupling). Decision: **fold the one walker unpack-site edit into this task** (it is a one-line arity change; the full `int_fused_ok` stamping logic is Task 3). Run now to capture the pre-change baseline:
  `pytest tests/python_integration/test_int_reductions.py::test_int64_sum_returns_int64_dtype_via_engine -v`
  Expected: FAIL — currently the Int64 sum is rejected by the F32-only gate (`schema.get(payload) != pl.Float32`) → the select routes to CPU GroupBy; result is correct Int64 but **the assertion that exercises the int path** is what we're building; right now `got.equals(want)` may already pass via CPU. So assert the *mechanism* too in Task 5. For this task, the meaningful RED is the F32-arity test after the analyzer edit. Proceed.
- [ ] Implement in `python/polars_metal/_fusion_analyzer.py` — change `analyze_ir_reduction`'s signature/return type and replace the F32-only gate. New return type annotation:
  ```python
  def analyze_ir_reduction(
      nt: Any, agg_node_id: int, schema: dict[str, Any]
  ) -> tuple[PyFusionScope, list[tuple[str, str | float]], str, bool, int, str] | None:
  ```
  (The 6th element is the wire output-dtype string.) Then replace the gate block (current lines ~690–697, the `if any(... != pl.Float32 ...)` + the "need at least one real column" check) with:
  ```python
          # Need at least one real column (literal-only reductions are degenerate).
          col_descriptors = [(k, p) for k, p in descriptors if k == "col"]
          if not col_descriptors:
              raise _Aborted

          # Resolve the binding's single column dtype. B2 admits only monomorphic
          # reductions (every column leaf shares one dtype); a mixed-dtype chain
          # falls back so we never mis-infer the output width.
          col_dtypes = {schema.get(p) for _, p in col_descriptors}
          if len(col_dtypes) != 1:
              raise _Aborted
          col_dtype = next(iter(col_dtypes))

          # Output-dtype inference + GPU-admit gate:
          #   - Float32 column → "F32" (the existing float path, unchanged).
          #   - GPU-admissible int (op, dtype) → its wire int dtype (no cast).
          #   - anything else (narrow int sum, int mean/std/var, F64, …) → CPU.
          if col_dtype == pl.Float32:
              out_dtype_str = "F32"
          else:
              out_dtype_str = _int_reduction_out_dtype(kind, col_dtype)
              if out_dtype_str is None:
                  raise _Aborted
  ```
  Then update the bare-single-column check + the terminus push + the return to thread `out_dtype_str` (the lines that currently read `if not is_chain and (len(descriptors) != 1 ...)`, `red_idx = scope.push_op(...)`, `return scope, descriptors, kind, is_chain, arg_id`):
  ```python
          if not is_chain and (len(descriptors) != 1 or descriptors[0][0] != "col"):
              # Bare reduction must be a single column.
              raise _Aborted
          red_idx = scope.push_op(op_id, [inner_idx])
          scope.mark_output(red_idx)
          return scope, descriptors, kind, is_chain, arg_id, out_dtype_str
  ```
  Note: `kind` is the lowercase agg name already computed at the top of the function (`kind = str(getattr(agg_node, "name", "")).lower()`); `_int_reduction_out_dtype` takes exactly that. Update the function docstring's "Returns `(scope, descriptors, agg_kind, is_chain, arg_id)`" line to add `, out_dtype_str` and its "All column inputs must be Float32" paragraph to describe the new admit rule.
- [ ] In `python/polars_metal/_walker.py::_try_fused_select_reduction`, change the single unpack site (line ~424) from 5 to 6 elements (minimal arity fix; the stamping logic is Task 3 — here just keep it compiling and ignore the new value for now with a clear marker):
  ```python
          scope, columns, agg_kind, is_chain, arg_id, out_dtype_str = result
  ```
  Do **not** yet use `out_dtype_str` (Task 3 stamps `_fused_out_dtype` and adds `int_fused_ok`). Leave a `# noqa` is unnecessary — `out_dtype_str` is consumed in Task 3; to avoid an unused-variable lint failure *now*, add it to the binding dict immediately as `"_fused_out_dtype": out_dtype_str` (this is forward-compatible and harmless — the dispatch defaults to `"F32"` and Task 3 adds the gate):
  ```python
          bindings.append(
              {
                  "name": output_name,
                  "_fused_scope": scope,
                  "_fused_columns": columns,
                  "_agg_kind": agg_kind,
                  "_is_chain": is_chain,
                  "_fused_out_dtype": out_dtype_str,
                  "_null_mode": null_mode_ir(nt, arg_id, in_schema) if is_chain else None,
              }
          )
  ```
- [ ] Run, expect PASS (F32 arity test green; the Int64 engine test passes via whichever path — its GPU-mechanism assertion is Task 5):
  `pytest tests/python_integration/test_int_reductions.py -v`
  Expected: green (all Task-1 matrix tests + the two new engine tests).
- [ ] Run the existing F32 reduction routing suite to confirm no regression from the arity change:
  `pytest tests/python_integration/test_reduction_routing.py -v`
  Expected: green (5 tests: bare sum/min/max/mean stay CPU, std/var use GPU, sum rides along with std).
- [ ] Lint:
  `ruff check python/polars_metal/_fusion_analyzer.py python/polars_metal/_walker.py tests/python_integration/test_int_reductions.py`
  Expected: clean.
- [ ] Commit:
  `git add python/polars_metal/_fusion_analyzer.py python/polars_metal/_walker.py tests/python_integration/test_int_reductions.py && git commit -m "B2 T2: analyzer — relax F32-only reduction gate to admit int sum/min/max, return inferred out-dtype (6-tuple)"`

---

## Task 3 — Walker: stamp `_fused_out_dtype` + `int_fused_ok` dtype-match safety

Make the walker defend against an analyzer mis-inference: an int reduction is admitted onto the GPU only when the analyzer's inferred wire output tag corresponds to the **real** Polars reduction output dtype (`nt.get_dtype(agg_node_id)`). On mismatch (or any int reduction the analyzer didn't tag F32 but whose real dtype doesn't match), the whole select falls back to CPU. This mirrors B1's HStack `int_fused_ok` defense (`_walker.py:604-614`).

**Files**
- Modify: `python/polars_metal/_walker.py`
- Test: extend `tests/python_integration/test_int_reductions.py`

**Steps**

- [ ] Add a failing safety test asserting the dtype-match guard rejects a hypothetical mismatch by confirming the real-dtype check is consulted. Since we can't easily fabricate a mis-inference, assert the *positive* contract precisely — an admitted int reduction's result dtype equals `nt.get_dtype`'s dtype — and add a regression that an int reduction whose real Polars output dtype is Int64 (e.g. narrow `sum` upcast) is NOT GPU-routed (it must already be aborted by the analyzer, but the walker guard is the belt-and-braces). Append:
  ```python
  from polars_metal import _native


  def _reduction_dispatches(lf, eng) -> int:
      """Count GPU fused-reduction dispatches via the execute_fused_expr hook."""
      n = {"c": 0}
      orig = _native.execute_fused_expr

      def cnt(scope, inputs, out):
          n["c"] += 1
          return orig(scope=scope, inputs=inputs, out=out)

      _native.execute_fused_expr = cnt
      try:
          lf.collect(engine=eng)
      finally:
          _native.execute_fused_expr = orig
      return n["c"]


  def test_narrow_int_sum_never_routes_to_gpu():
      # Int8 sum → Polars upcasts to Int64; MLX→int32. Not admitted; must stay
      # CPU (0 GPU dispatches) AND match Polars exactly.
      eng = MetalEngine()
      df = pl.DataFrame({"x": pl.Series([1, 2, 3, 100], dtype=pl.Int8)})
      lf = df.lazy().select(pl.col("x").sum().alias("s"))
      assert _reduction_dispatches(lf, eng) == 0
      got, want = lf.collect(engine=eng), lf.collect()
      assert got.equals(want)
      assert got["s"].dtype == pl.Int64  # Polars upcast preserved on CPU


  def test_int64_chain_sum_routes_to_gpu_and_matches():
      # A compute chain ending in sum is always GPU-worthy (is_chain=True), so
      # this genuinely exercises the GPU int-reduction path.
      eng = MetalEngine()
      df = pl.DataFrame({"x": pl.Series([1, 2, 3, 4, 5], dtype=pl.Int64)})
      lf = df.lazy().select(((pl.col("x") * 2) + 1).sum().alias("s"))
      assert _reduction_dispatches(lf, eng) == 1, "int chain sum should use GPU"
      got, want = lf.collect(engine=eng), lf.collect()
      assert got.equals(want)
      assert got["s"].dtype == pl.Int64
  ```
- [ ] Run, expect FAIL — `test_int64_chain_sum_routes_to_gpu_and_matches` is the meaningful RED: today the F32-only gate (now relaxed in T2) admits the chain, but the dispatch (`_build_select_reduction_fused`, still F32-hardcoded until Task 4) stages the Int64 column as `np.float32` and pre-allocates an `np.float32` `out_arr` with tag 0 → `execute_fused_expr` declares output tag F32 while MLX evals an int64 reduction → `eval_into_typed`'s dtype assertion fires → `PyRuntimeError`. So expect an error/exception, not a clean fail:
  `pytest tests/python_integration/test_int_reductions.py::test_int64_chain_sum_routes_to_gpu_and_matches -v`
  Expected: FAIL (RuntimeError "output dtype mismatch: declared F32, eval'd I64" from `eval_into_typed`). `test_narrow_int_sum_never_routes_to_gpu` should already PASS (the analyzer aborted Int8 sum in T2 → CPU). This RED proves the dispatch must be made dtype-aware (Task 4) AND motivates the walker guard here.
- [ ] Implement the walker guard in `python/polars_metal/_walker.py::_try_fused_select_reduction`. After `analyze_ir_reduction` returns and before appending the binding, add the real-dtype safety check (mirroring `_INT_TAG_TO_POLARS` use at HStack lines 604-608). Insert right after the `scope, columns, agg_kind, is_chain, arg_id, out_dtype_str = result` unpack:
  ```python
          # B2 safety (mirrors the HStack `int_fused_ok` defense): admit an int
          # reduction onto the GPU only when the analyzer's inferred wire output
          # tag corresponds to the REAL Polars reduction output dtype. The F32
          # path is unconditionally fine. A mismatch (analyzer mis-inference)
          # aborts the whole fused select to CPU, which preserves Polars' dtype.
          if out_dtype_str != "F32":
              try:
                  real_out = str(nt.get_dtype(node_id))
              except Exception:
                  return None
              if (
                  out_dtype_str not in _INT_TAG_TO_POLARS
                  or real_out != _INT_TAG_TO_POLARS[out_dtype_str]
              ):
                  return None
  ```
  (`node_id` is the agg's arena id, already bound at the top of the per-`agg_expr` loop; `_INT_TAG_TO_POLARS` is already imported at `_walker.py:100`. Returning `None` aborts the whole fused-select attempt → the walker falls through to the conformance GroupBy/CPU path, exactly like a non-eligible agg.)
- [ ] Run, expect: `test_narrow_int_sum_never_routes_to_gpu` PASS; `test_int64_chain_sum_routes_to_gpu_and_matches` still FAIL (the dispatch is still F32-hardcoded — Task 4 fixes it). This is the expected intermediate state — the walker now *admits* the Int64 chain (guard passes: `out_dtype_str="I64"`, `_INT_TAG_TO_POLARS["I64"]=="Int64"`, real dtype Int64), and the failure has moved entirely into the dispatch:
  `pytest tests/python_integration/test_int_reductions.py -v`
  Expected: all green EXCEPT `test_int64_chain_sum_routes_to_gpu_and_matches` (RuntimeError from `eval_into_typed`). Note this in the commit.
- [ ] Lint:
  `ruff check python/polars_metal/_walker.py tests/python_integration/test_int_reductions.py`
  Expected: clean.
- [ ] Commit:
  `git add python/polars_metal/_walker.py tests/python_integration/test_int_reductions.py && git commit -m "B2 T3: walker — stamp _fused_out_dtype + int_fused_ok dtype-match safety on fused reductions (dispatch still F32; T4)"`

---

## Task 4 — Dispatch: dtype-aware int staging + output in `_build_select_reduction_fused`

Make the reduction dispatch carry the binding's `_fused_out_dtype`: stage each column at its native width, stage literals at the output dtype, pre-allocate `out_arr` + pass tags at the output dtype, and build the result `pl.Series` at the output dtype. The `drop_nulls` / bare-fallback / Bessel logic is untouched (Bessel is std/var-only → F32-only → never reached by an admitted int reduction). This turns the Task-3 chain test green and is the last code change (verified Python-only — no Rust edit).

**Files**
- Modify: `python/polars_metal/_udf.py`
- Test: `tests/python_integration/test_int_reductions.py` (Task-3 case turns green; no new test needed here — Task 5 adds the full matrix)

**Steps**

- [ ] Confirm the failing target is green-able with a Python-only edit (no Rust): the RED from Task 3 is `eval_into_typed`'s assertion firing because Python declared output tag 0 (F32) for an int64 eval. Run it to re-confirm the exact error before editing:
  `pytest tests/python_integration/test_int_reductions.py::test_int64_chain_sum_routes_to_gpu_and_matches -v`
  Expected: FAIL with `RuntimeError: ... output dtype mismatch: declared F32, eval'd I64` (or similar) — proving the only fix needed is Python passing the right tag/dtype.
- [ ] Implement in `python/polars_metal/_udf.py::_build_select_reduction_fused`. Resolve the binding's output dtype once at the top of the per-`agg` loop, then thread it. Inside `for agg in fused_aggs:` (after `is_chain = agg.get("_is_chain", False)`), add:
  ```python
          out_dtype_str = agg.get("_fused_out_dtype", "F32")
          out_np_dtype, out_tag = _np_dtype_and_tag(out_dtype_str)
          out_pl_dtype = _wire_str_to_polars_dtype(out_dtype_str)
  ```
  Replace the **bare** branch's staging (current lines ~659-672). The null/degenerate CPU fallback already builds `pl.Series(name, [value], dtype=series.dtype)` which is correct for any dtype — keep it. Only the GPU-staging line changes from forced `np.float32`:
  ```python
          if not is_chain:
              # Bare single column (any admitted int width, or F32). Enforced by
              # `analyze_ir_reduction` to be a single column.
              col_name = descriptors[0][1]
              series = upstream.get_column(col_name)
              # Nulls / <2 rows: replay the reduction on the source column with
              # Polars — exact (same null skipping, n=0/1, dtype) for any dtype.
              if n < 2 or series.null_count() > 0:
                  value = getattr(series, kind)()
                  out_series.append(pl.Series(name, [value], dtype=series.dtype))
                  continue
              col_dtype_str = _series_input_dtype_str(series)
              col_np_dtype, col_tag = _np_dtype_and_tag(col_dtype_str)
              input_arrays = [np.ascontiguousarray(series.to_numpy(), dtype=col_np_dtype)]
              input_tags = [col_tag]
              count = n
  ```
  Replace the **chain** branch's staging (current lines ~694-702) to stage columns at native width + literals at the output dtype, and to build a parallel `input_tags`:
  ```python
              input_arrays = []
              input_tags = []
              for d_kind, payload in descriptors:
                  if d_kind == "col":
                      col = frame.get_column(payload)
                      col_dtype_str = _series_input_dtype_str(col)
                      col_np_dtype, col_tag = _np_dtype_and_tag(col_dtype_str)
                      input_arrays.append(np.ascontiguousarray(col.to_numpy(), dtype=col_np_dtype))
                      input_tags.append(col_tag)
                  elif d_kind == "lit":
                      # Stage the literal at the reduction's output dtype so an int
                      # chain stays integer (mirrors the HStack literal staging).
                      input_arrays.append(np.asarray([payload], dtype=out_np_dtype))
                      input_tags.append(out_tag)
                  else:
                      raise RuntimeError(f"polars_metal: unknown reduction descriptor {d_kind!r}")
  ```
  Note the chain branch's `count == 0` / `count < 2` early-outs currently build `pl.Series(name, [val0], dtype=pl.Float32)` — for an admitted int reduction `count == 0` can occur (all-null after drop). `sum` of an empty int column in Polars is `0` typed at the reduction dtype; `kind in ("std","var")` is F32-only so unreachable for int. Change the `count == 0` early-out to use the output dtype:
  ```python
              count = frame.height
              if count == 0:
                  val0 = 0 if kind == "sum" else None
                  out_series.append(pl.Series(name, [val0], dtype=out_pl_dtype))
                  continue
              if kind in ("std", "var") and count < 2:
                  out_series.append(pl.Series(name, [None], dtype=pl.Float32))
                  continue
  ```
  Then replace the shared eval + result-build tail (current lines ~704-720) to allocate/tag/wrap at the output dtype:
  ```python
          out_arr = np.empty(1, dtype=out_np_dtype)
          inputs = [
              (int(a.__array_interface__["data"][0]), int(a.size), tag)
              for a, tag in zip(input_arrays, input_tags, strict=True)
          ]
          _native.execute_fused_expr(
              scope=scope,
              inputs=inputs,
              out=(int(out_arr.__array_interface__["data"][0]), int(out_arr.size), out_tag),
          )
          # Bessel correction is std/var-only (F32-only — never an int output).
          if out_dtype_str == "F32":
              val = float(out_arr[0])
              if kind == "var":
                  val *= count / (count - 1)
              elif kind == "std":
                  val *= math.sqrt(count / (count - 1))
              out_series.append(pl.Series(name, [val], dtype=pl.Float32))
          else:
              # Integer reduction: the scalar is already the exact Polars dtype
              # (MLX-native == Polars for the admitted combos). Wrap losslessly via
              # numpy item() so no float round-trip touches the integer value.
              out_series.append(pl.Series(name, [out_arr[0].item()], dtype=out_pl_dtype))
  ```
  (`out_arr[0].item()` yields a Python int for an int dtype — no float truncation. The F32 branch keeps the existing `float(out_arr[0])` + Bessel path verbatim.)
- [ ] Add the `_wire_str_to_polars_dtype` helper near the other `_udf.py` dtype helpers (after `_series_input_dtype_str`, ~line 184). Reuse `_INT_TAG_TO_POLARS` from the walker via Polars' own name lookup to avoid a third table:
  ```python
  # Wire dtype string -> Polars dtype, for building int reduction result Series.
  # Mirrors _DTYPE_STR_TO_NP_AND_TAG; F32 maps to Float32 (the float path uses it
  # directly, but include it for completeness).
  _WIRE_STR_TO_POLARS: dict[str, "pl.DataType"] = {
      "F32": pl.Float32,
      "I8": pl.Int8,
      "I16": pl.Int16,
      "I32": pl.Int32,
      "I64": pl.Int64,
      "U8": pl.UInt8,
      "U16": pl.UInt16,
      "U32": pl.UInt32,
      "U64": pl.UInt64,
  }


  def _wire_str_to_polars_dtype(dtype_str: str) -> "pl.DataType":
      dt = _WIRE_STR_TO_POLARS.get(dtype_str)
      if dt is None:
          raise RuntimeError(f"polars_metal: no Polars dtype for wire dtype {dtype_str!r}")
      return dt
  ```
  (This third table is unavoidable here — `_DTYPE_STR_TO_NP_AND_TAG` maps to numpy/tag, `_INT_TAG_TO_POLARS` maps wire→Polars-*string* and lives in `_walker.py`. Importing `_walker` into `_udf` would create a cycle. The drift-guard test in Task 5 pins `_WIRE_STR_TO_POLARS` against `_INT_TAG_TO_POLARS` so the two can't drift.)
- [ ] Run the Task-3 chain test, expect PASS:
  `pytest tests/python_integration/test_int_reductions.py::test_int64_chain_sum_routes_to_gpu_and_matches -v`
  Expected: green (Int64 chain sum now stages int64, declares tag 6, `eval_into_typed` assertion passes, result is Int64).
- [ ] Run the whole int-reduction file + the F32 routing suite (no-regression):
  `pytest tests/python_integration/test_int_reductions.py tests/python_integration/test_reduction_routing.py -v`
  Expected: all green.
- [ ] Lint:
  `ruff check python/polars_metal/_udf.py tests/python_integration/test_int_reductions.py`
  Expected: clean.
- [ ] Commit:
  `git add python/polars_metal/_udf.py tests/python_integration/test_int_reductions.py && git commit -m "B2 T4: dispatch — dtype-aware int staging + output in _build_select_reduction_fused; Python-only (no Rust change)"`

---

## Task 5 — Differential matrix: sum/min/max × admitted int dtypes × null modes, + CPU-fallback correctness

Lock the B2 exit bar with a parametrized differential test against Polars CPU, byte-exact, covering: `sum` over {Int32, Int64, UInt32, UInt64} and `min`/`max` over all 8 int widths; null-free / null-bearing-chain / empty / single-element; ≥1 genuine GPU-path case (the chain cases via the dispatch counter); and the CPU-fallback cases (narrow `sum`, int `mean`) matching exactly. Also add the `_WIRE_STR_TO_POLARS` ↔ `_INT_TAG_TO_POLARS` drift guard.

**Files**
- Modify: `tests/python_integration/test_int_reductions.py`

**Steps**

- [ ] Append the full differential matrix. Use chain forms for the GPU-path assertions (a bare int sum/min/max is bandwidth-bound → the B4-untouched gate keeps it on CPU; `got.equals(want)` still holds on CPU, and we separately prove the GPU path via the chain cases):
  ```python
  _SUM_DTYPES = [pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]
  _MINMAX_DTYPES = [
      pl.Int8, pl.Int16, pl.Int32, pl.Int64,
      pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
  ]


  def _vals_for(dtype) -> list[int]:
      # Small, in-range, mixed-sign where signed. Chosen so +/* in a chain stay
      # in range for the narrowest min/max types (Int8/UInt8).
      if dtype in (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
          return [0, 1, 2, 7, 9, 3]
      return [-3, 0, 1, 2, 7, -2]


  @pytest.mark.parametrize("dtype", _SUM_DTYPES)
  def test_int_sum_byte_exact_no_nulls(dtype):
      eng = MetalEngine()
      df = pl.DataFrame({"x": pl.Series(_vals_for(dtype), dtype=dtype)})
      lf = df.lazy().select(pl.col("x").sum().alias("r"))
      got, want = lf.collect(engine=eng), lf.collect()
      assert got.equals(want), f"{dtype}: {got} != {want}"
      assert got["r"].dtype == dtype


  @pytest.mark.parametrize("dtype", _MINMAX_DTYPES)
  @pytest.mark.parametrize("op", ["min", "max"])
  def test_int_minmax_byte_exact_no_nulls(dtype, op):
      eng = MetalEngine()
      df = pl.DataFrame({"x": pl.Series(_vals_for(dtype), dtype=dtype)})
      lf = df.lazy().select(getattr(pl.col("x"), op)().alias("r"))
      got, want = lf.collect(engine=eng), lf.collect()
      assert got.equals(want), f"{op} {dtype}: {got} != {want}"
      assert got["r"].dtype == dtype


  @pytest.mark.parametrize("dtype", _SUM_DTYPES)
  def test_int_chain_sum_gpu_path(dtype):
      # Chain → is_chain=True → routes to GPU. Proves the GPU int-reduction path
      # (dispatch count == 1) AND byte-exact dtype/value.
      eng = MetalEngine()
      df = pl.DataFrame({"x": pl.Series(_vals_for(dtype), dtype=dtype)})
      lf = df.lazy().select(((pl.col("x") + 1) * 2).sum().alias("r"))
      assert _reduction_dispatches(lf, eng) == 1, f"chain sum {dtype} should use GPU"
      got, want = lf.collect(engine=eng), lf.collect()
      assert got.equals(want), f"chain {dtype}: {got} != {want}"


  @pytest.mark.parametrize("op", ["min", "max"])
  def test_int_chain_minmax_gpu_path(op):
      eng = MetalEngine()
      df = pl.DataFrame({"x": pl.Series([-3, 0, 1, 2, 7], dtype=pl.Int64)})
      lf = df.lazy().select(getattr((pl.col("x") + 1), op)().alias("r"))
      assert _reduction_dispatches(lf, eng) == 1, f"chain {op} should use GPU"
      got, want = lf.collect(engine=eng), lf.collect()
      assert got.equals(want)


  @pytest.mark.parametrize("dtype", _SUM_DTYPES)
  def test_int_chain_sum_with_nulls_drop_nulls_path(dtype):
      # Null-bearing chain → walker stamps _drop_nulls → CPU drop + GPU reduce of
      # survivors. Byte-exact vs Polars (which skips nulls).
      eng = MetalEngine()
      vals = _vals_for(dtype)
      vals_n = vals[:2] + [None] + vals[2:]
      df = pl.DataFrame({"x": pl.Series(vals_n, dtype=dtype)})
      lf = df.lazy().select(((pl.col("x") + 1) * 2).sum().alias("r"))
      got, want = lf.collect(engine=eng), lf.collect()
      assert got.equals(want), f"null chain {dtype}: {got} != {want}"


  @pytest.mark.parametrize("dtype", _SUM_DTYPES)
  def test_int_sum_empty(dtype):
      eng = MetalEngine()
      df = pl.DataFrame({"x": pl.Series([], dtype=dtype)})
      lf = df.lazy().select(pl.col("x").sum().alias("r"))
      got, want = lf.collect(engine=eng), lf.collect()
      assert got.equals(want), f"empty {dtype}: {got} != {want}"


  @pytest.mark.parametrize("dtype", _SUM_DTYPES)
  def test_int_sum_single_element(dtype):
      eng = MetalEngine()
      df = pl.DataFrame({"x": pl.Series([5], dtype=dtype)})
      lf = df.lazy().select(pl.col("x").sum().alias("r"))
      got, want = lf.collect(engine=eng), lf.collect()
      assert got.equals(want), f"single {dtype}: {got} != {want}"


  @pytest.mark.parametrize("op", ["min", "max"])
  def test_int_minmax_empty_and_single(op):
      eng = MetalEngine()
      for vals in ([], [5]):
          df = pl.DataFrame({"x": pl.Series(vals, dtype=pl.Int64)})
          lf = df.lazy().select(getattr(pl.col("x"), op)().alias("r"))
          got, want = lf.collect(engine=eng), lf.collect()
          assert got.equals(want), f"{op} {vals}: {got} != {want}"


  # ── CPU-fallback combos: must NOT route to GPU, must still match Polars ──

  @pytest.mark.parametrize("dtype", [pl.Int8, pl.Int16, pl.UInt8, pl.UInt16])
  def test_narrow_int_sum_falls_back_to_cpu_and_matches(dtype):
      eng = MetalEngine()
      df = pl.DataFrame({"x": pl.Series([1, 2, 3, 4], dtype=dtype)})
      # Bare AND chain forms: narrow sum is never admitted (analyzer aborts).
      for lf in (
          df.lazy().select(pl.col("x").sum().alias("r")),
          df.lazy().select(((pl.col("x") + 1)).sum().alias("r")),
      ):
          assert _reduction_dispatches(lf, eng) == 0, f"narrow sum {dtype} must stay CPU"
          got, want = lf.collect(engine=eng), lf.collect()
          assert got.equals(want), f"narrow sum {dtype}: {got} != {want}"


  @pytest.mark.parametrize("dtype", [pl.Int32, pl.Int64, pl.UInt32, pl.UInt64])
  def test_int_mean_falls_back_to_cpu_and_matches(dtype):
      eng = MetalEngine()
      df = pl.DataFrame({"x": pl.Series([1, 2, 3, 4], dtype=dtype)})
      for lf in (
          df.lazy().select(pl.col("x").mean().alias("r")),
          df.lazy().select(((pl.col("x") + 1)).mean().alias("r")),
      ):
          assert _reduction_dispatches(lf, eng) == 0, f"int mean {dtype} must stay CPU"
          got, want = lf.collect(engine=eng), lf.collect()
          assert got.equals(want), f"int mean {dtype}: {got} != {want}"


  def test_wire_str_to_polars_matches_walker_table():
      # Drift guard: _udf._WIRE_STR_TO_POLARS must agree with _walker._INT_TAG_TO_POLARS
      # (both map wire str -> the same Polars dtype) for every integer wire tag.
      from polars_metal._udf import _WIRE_STR_TO_POLARS
      from polars_metal._walker import _INT_TAG_TO_POLARS

      for wire, pl_str in _INT_TAG_TO_POLARS.items():
          assert wire in _WIRE_STR_TO_POLARS, f"{wire} missing from _WIRE_STR_TO_POLARS"
          assert str(_WIRE_STR_TO_POLARS[wire]) == pl_str, (
              f"{wire}: {_WIRE_STR_TO_POLARS[wire]} != {pl_str}"
          )
  ```
  (The bare int sum/min/max byte-exact tests assert correctness *regardless of path* — they pass whether the gate routes them to CPU or GPU. The chain tests assert the GPU path explicitly via `_reduction_dispatches == 1`. Together they satisfy the exit bar's "(a) correct regardless of path AND (b) genuinely exercise the GPU path" requirement.)
- [ ] Run the whole file, expect PASS:
  `pytest tests/python_integration/test_int_reductions.py -v`
  Expected: all parametrizations green. If a `min`/`max` chain narrows out of an Int8 range, adjust `_vals_for` (the chosen values keep `(x+1)` in range for Int8/UInt8).
- [ ] Lint:
  `ruff check tests/python_integration/test_int_reductions.py`
  Expected: clean.
- [ ] Commit:
  `git add tests/python_integration/test_int_reductions.py && git commit -m "B2 T5: differential matrix — int sum/min/max × dtypes × null modes byte-exact; GPU-path + CPU-fallback proofs; drift guard"`

---

## Task 6 — Full `make gate` + conformance no-regression

Run the full gate to confirm B2 introduces no Rust regression (it shouldn't touch Rust) and no Python/conformance regression. Confirm the only failures are the documented pre-existing baseline divergences (MEMORY: `M3 conformance deferrals`, `M6 conformance fixes`).

**Files**
- None (verification only; a doc note if any baseline shifts).

**Steps**

- [ ] Run the targeted Python integration suites that touch the fused-reduction + HStack machinery, to catch any cross-talk:
  `pytest tests/python_integration/test_int_reductions.py tests/python_integration/test_reduction_routing.py tests/python_integration/test_int_foundation.py tests/python_integration/test_execute_fused_expr.py -v`
  Expected: green (B1's `test_int_foundation` + `test_execute_fused_expr` must still pass — B2 didn't touch the HStack path; the arity change to `analyze_ir_reduction` is reduction-only).
- [ ] Run the full gate:
  `make gate`
  Expected: green, OR only the documented pre-existing baseline divergences (no NEW failures). Per MEMORY, the known conformance baseline is the lazyframe/group_by set that M6 already fixed and the F32-mean-returns-F32 divergence — none of which B2 changes. If a NEW failure appears, **stop and triage** (it's a B2 regression) before committing.
- [ ] If `make gate` is green with no new failures, no commit is needed (no files changed). If a baseline note needs updating (e.g. a previously-failing int-reduction conformance test now passes), record it in the commit:
  `git commit --allow-empty -m "B2 T6: make gate green — int reductions land with no conformance/Rust regression"`

---

## Self-review: B2 scope coverage

Mapping each B2 scope item (from the prompt + spec B2 reduction-semantics) to a task:

| B2 scope item | Covered by |
|---|---|
| Admit int `sum` for {Int32, Int64, UInt32, UInt64} (MLX-native == Polars, no cast) | T1 (`_SUM_ADMIT` matrix) → T2 (gate) → T4 (dispatch) → T5 (differential) |
| Admit int `min`/`max` for all 8 int widths | T1 (`_MINMAX_ADMIT`) → T2 → T4 → T5 |
| Narrow `sum` (Int8/16, UInt8/16) → CPU fallback (mismatch) | T1 (returns None) → T2 (abort) + T3 (guard) → T5 (`test_narrow_int_sum_falls_back_to_cpu_and_matches`) |
| Int `mean`/`std`/`var` → CPU fallback (MLX→f32 vs Polars→Float64) | T1 (returns None for mean; std/var never in `_REDUCTION_OP` int admit) → T5 (`test_int_mean_falls_back_to_cpu_and_matches`) |
| Analyzer: relax F32-only gate + infer + return output dtype | T2 |
| Walker: stamp `_fused_out_dtype` + `int_fused_ok` dtype-match safety (mirror B1) | T2 (stamp) + T3 (guard) |
| Dispatch: dtype-aware int staging (cols + literals) + out-dtype alloc/tags + result Series (bare + chain/drop_nulls) | T4 |
| Null handling — REUSE `_drop_nulls` (chain) + bare-CPU fallback, no sentinel | T4 (preserves both branches verbatim) + T5 (`test_int_chain_sum_with_nulls_drop_nulls_path`) |
| Empty / single-element | T5 (`test_int_sum_empty`, `test_int_sum_single_element`, `test_int_minmax_empty_and_single`) |
| ≥1 genuine GPU int-reduction path, verified | T3 + T5 (`test_int_chain_*_gpu_path` via the `_reduction_dispatches` counter) |
| Reuse B1's dtype tables (`_DTYPE_STR_TO_NP_AND_TAG`, `_np_dtype_and_tag`, `_series_input_dtype_str`, `_INT_TAG_TO_POLARS`), don't duplicate | T4 (reuses all; the one new `_WIRE_STR_TO_POLARS` table is drift-guarded in T5) |
| No conformance/Rust regression | T6 (`make gate`) |

**Rust:** no change. Verified (Architecture §): `op_spec(...).output_dtype` / `DtypeOut::ScalarF32` is consulted only by `est_flops_for` (routing) and `push_op` (arg validation), never to coerce eval output dtype; the reduction output dtype is produced by MLX (`mlx_sum`/`mlx_min`/`mlx_max` in `subgraph.rs:504-507`) and read back + asserted-against-the-declared-tag by `eval_into_typed` (B1). Passing the correct tag from Python is sufficient. Task 4 re-confirms this empirically (the RED is `eval_into_typed`'s assertion firing, the GREEN is Python passing the right tag — no Rust touched).

**Out of scope, explicitly not touched** (would be scope creep): the compute-intensity routing gate (`_try_fused_select_reduction` line ~453, `_BARE_GPU_WORTHY_REDUCTIONS`) — B4 tunes it; narrow-int `sum` upcast on the GPU (needs a `CastI64`/`CastU64` op — deferred); int `mean`/`std`/`var` on the GPU; the `dt` kernel (B3); the elementwise HStack path (B1); Bool-final reductions; bit-ops.

**Conventions honored:** Python-only (no `make wheel` needed for the code changes — Task 4 confirms); per-task `ruff` lint on touched files; reuse of B1's dtype machinery (no duplicate tables; the single unavoidable `_WIRE_STR_TO_POLARS` is drift-guarded against `_INT_TAG_TO_POLARS`); walker `int_fused_ok` defense mirrored from B1's HStack path; differential vs Polars CPU byte-exact (`got.equals(want)`) for every admitted combo; GPU path verified via the `execute_fused_expr` dispatch counter (the same mechanism `test_reduction_routing.py` uses).
