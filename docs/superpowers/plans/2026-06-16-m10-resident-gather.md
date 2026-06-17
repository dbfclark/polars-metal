# M10 — Resident gather via join recognition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `engine="metal"` accelerate the natural `fact.join(dim, on=<int key>) → <F32 compute chain>` idiom (resident GPU gather when the key is a dense range, CPU-lookup + resident GPU chain otherwise), plus a resident vector rerank on `.metal.cosine_topk`.

**Architecture:** Recognize `Join(int key) → F32 chain` in the walker (model 1: a whole-plan `PythonScanSource::Cuda` UDF rooted at both scans — never collect-and-stitch, which can't keep the gather resident). The UDF produces the looked-up dim column then runs the **existing** fused MLX subgraph over `{fact cols, looked-up col}`. Phase 1 ships the CPU-lookup branch by reusing today's `_dispatch_hstack_fused` after a CPU join (no new kernels, proves the integration). Phases 2–3 add a 1-D `mlx_take` + `OpId::Take` so the dense-key case gathers on GPU resident, byte-matching the Phase-1 CPU result. Phase 4 adds the vector rerank. Phase 5 adds a force-route override + honest perf wiring.

**Tech Stack:** Rust (engine core + PyO3 `_native`), MLX C++ FFI via `cxx` (`polars-metal-mlx-sys`), Python engine glue (`polars_metal`), Polars 1.40.1 engine-plugin `NodeTraverser`, pytest + cargo test (differential vs Polars CPU / numpy).

**Spec:** `docs/superpowers/specs/2026-06-16-m10-resident-gather-design.md`

**Reference symbols (verified current, this branch):**
- Walker: `python/polars_metal/_walker.py` — `_walk_at_current:157`, `_walk_hstack:557`, `_probe_fusion_analyzer:521`, `Handled`/`FallBack:125`.
- Analyzer: `python/polars_metal/_fusion_analyzer.py` — `analyze_ir_with_columns:564` returns `(PyFusionScope, [("col",name)|("lit",val)...], out_dtype_str)`.
- UDF: `python/polars_metal/_udf.py` — `build_udf:51`, `_dispatch:89`, `_dispatch_hstack_fused:208`.
- Callback: `python/polars_metal/_callback.py` — `execute_with_metal:25` (installs `nt.set_udf` at root on `Handled`).
- Density: `crates/polars-metal-core/src/fusion/density.rs` — `density_routes_gpu(scope, n_rows):23` (FLOPs ≥ 5e7 AND rows ≥ 1e5).
- Engine: `python/polars_metal/_engine.py` — `MetalEngine` frozen dataclass.
- Rust fusion: `fusion/supported_ops.rs` `OpId` enum (ends ~line 94) + `op_spec`/`est_flops_for`; `fusion/scope.rs` `FusionScope`/`add_input`/`push_op_param`; `fusion/py.rs` `PyFusionScope.add_input/push_op`; `fusion/subgraph.rs` `from_fusion_scope_buffers:248` + `build_op:390`.
- MLX FFI: `polars-metal-mlx-sys/src/shape.rs` `mlx_take_along_axis:39`; `cxx/mlx_bridge.{h,cc}`; `src/lib.rs` `#[cxx::bridge]`; `src/elementwise.rs` `mlx_mul/mlx_exp/mlx_neg`.
- Vector: `crates/polars-metal-core/src/vector_search.rs` `vector_search_topk:42` / `execute_vector_search:165`; `python/polars_metal/_vector_namespace.py` `cosine_topk:98`; `_vector_dispatch.py` `apply_vector_search:116` / `_run_binding:79`.
- Native registry: `crates/polars-metal-core/src/lib.rs` `#[pymodule] polars_metal_native:57`.

**Conventions (CLAUDE.md):** no `unwrap()` outside tests; no `unsafe` outside `*-sys`/buffer; errors → `PolarsError::ComputeError` at the boundary; null semantics match Polars exactly (differential test on nulls). Run `cargo fmt` + `ruff` per task (a subagent-fmt-discipline gotcha — don't let drift accumulate to the final gate). `make test-unit` runs `--test-threads=1` (Metal command-queue contention). `make wheel` (= `maturin develop --release`) after Rust changes, before Python tests that import `_native`.

---

## Phase 0 — Branch hygiene & smoke baseline

### Task 0.1: Confirm clean baseline

**Files:** none (verification only)

- [ ] **Step 1: Verify branch + green baseline**

Run:
```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
git status -sb            # expect: ## m10-resident-gather...
git log --oneline -1      # expect the M10 spec commit (23a27d4) on top
make wheel && make test-unit
```
Expected: build succeeds, unit tests pass (baseline). If red, STOP and report — do not build on a red baseline.

---

## Phase 1 — Model-1 integration: `_walk_join` + two-scan UDF + CPU-lookup branch

> This phase ships a real win (~2–4×) on the natural `.join()` idiom with **no new kernels**, by doing the join on CPU and reusing the existing fused-chain dispatch. It also de-risks the load-bearing unknown (two-scan scan-source UDF over a Join). If this phase can't be made correct, STOP and reconsider model 1 (per spec) rather than papering over with collect-and-stitch.

### Task 1.1: Characterize the `join → chain` IR shape (probe test)

**Files:**
- Create: `tests/python_integration/test_m10_join_ir_shape.py`

- [ ] **Step 1: Write the probe test** — documents the exact IR the walker sees, so `_walk_join` targets reality (not assumptions).

```python
"""Characterization: the IR shape a `join -> F32 chain` produces at the
post-optimization NodeTraverser. Pins what `_walk_join` must handle."""
from __future__ import annotations

import numpy as np
import polars as pl


def _capture_ir(lf: pl.LazyFrame) -> dict:
    report: dict = {}

    def cb(nt, _d=None):
        def visit(nid, depth):
            nt.set_node(nid)
            node = nt.view_current_node()
            report.setdefault("nodes", []).append((depth, type(node).__name__))
            if type(node).__name__ == "Join":
                report["join_attrs"] = sorted(a for a in dir(node) if not a.startswith("_"))
                report["join_how"] = repr(getattr(node, "options", None))
                report["left_on"] = [type(nt.view_expression(e.node)).__name__
                                     for e in getattr(node, "left_on", [])]
                report["right_on"] = [type(nt.view_expression(e.node)).__name__
                                      for e in getattr(node, "right_on", [])]
            for inp in nt.get_inputs():
                visit(inp, depth + 1)
        visit(nt.get_node(), 0)

    lf.collect(engine="cpu", post_opt_callback=cb)
    return report


def test_join_then_chain_ir_shape():
    rng = np.random.default_rng(0)
    fact = pl.DataFrame({
        "id": rng.integers(0, 500, 2000).astype(np.int64),
        "value": rng.uniform(50, 150, 2000).astype(np.float32),
    })
    dim = pl.DataFrame({
        "id": np.arange(500, dtype=np.int64),
        "vol": rng.uniform(0.1, 0.5, 500).astype(np.float32),
    })
    lf = (fact.lazy()
          .join(dim.lazy(), on="id", how="left")
          .with_columns((pl.col("vol").tanh() * pl.col("value")).alias("out")))
    rep = _capture_ir(lf)
    # The compute chain sits in an HStack above a Join above two scans.
    kinds = [k for _, k in rep["nodes"]]
    assert "Join" in kinds, rep
    assert "HStack" in kinds or "Select" in kinds, rep
    assert rep["left_on"] == ["Column"] and rep["right_on"] == ["Column"], rep
    print("M10 IR shape:", rep)  # captured for _walk_join authoring
```

- [ ] **Step 2: Run it (must pass; prints the shape)**

Run: `python -m pytest tests/python_integration/test_m10_join_ir_shape.py -v -s`
Expected: PASS, prints `M10 IR shape: {...}`. **Record `join_attrs`, `join_how`, and the node depth ordering** — Tasks 1.2/1.3 reference them. If `Join` is absent (optimizer fused it differently), STOP and report the actual shape.

- [ ] **Step 3: Commit**
```bash
git add tests/python_integration/test_m10_join_ir_shape.py
git commit -m "M10: characterize join->chain IR shape (probe)"
```

### Task 1.2: `_walk_join` recognition (guards + Handled plan dict)

**Files:**
- Modify: `python/polars_metal/_walker.py` (add `Join` arm to `_walk_at_current:157`; add `_walk_join`)
- Test: `tests/python_integration/test_m10_walk_join.py`

- [ ] **Step 1: Write the failing test** — `_walk_join` returns a `Join` plan when guards pass, `FallBack` otherwise.

```python
"""Unit-level: _walk_join recognizes equi-join(int key, inner/left) -> fused F32
chain and returns a Handled plan dict; falls back otherwise."""
from __future__ import annotations

import numpy as np
import polars as pl
from polars_metal._walker import Handled, FallBack, walk


def _walk_plan(lf: pl.LazyFrame):
    out = {}
    def cb(nt, _d=None):
        out["res"] = walk(nt)
    lf.collect(engine="cpu", post_opt_callback=cb)
    return out["res"]


def _frames(key_dtype=np.int64, how="left"):
    rng = np.random.default_rng(1)
    fact = pl.DataFrame({
        "id": rng.integers(0, 500, 2000).astype(key_dtype),
        "value": rng.uniform(50, 150, 2000).astype(np.float32),
    })
    dim = pl.DataFrame({
        "id": np.arange(500, dtype=key_dtype),
        "vol": rng.uniform(0.1, 0.5, 500).astype(np.float32),
    })
    return (fact.lazy().join(dim.lazy(), on="id", how=how)
            .with_columns((pl.col("vol").tanh() * pl.col("value")).alias("out")))


def test_walk_join_handled_for_int_key_fused_chain():
    res = _walk_plan(_frames())
    assert isinstance(res, Handled), res
    # Top plan is the HStack(chain) whose input is a Join node.
    plan = res.plan
    def find_join(p):
        if p.get("kind") == "Join":
            return p
        inner = p.get("input")
        return find_join(inner) if isinstance(inner, dict) else None
    jp = find_join(plan)
    assert jp is not None, plan
    assert jp["how"] in ("left", "inner")
    assert jp["key"] == "id"
    assert jp["left"]["kind"] == "Scan" and jp["right"]["kind"] == "Scan"


def test_walk_join_fallback_on_f64_chain():
    rng = np.random.default_rng(2)
    fact = pl.DataFrame({"id": rng.integers(0, 100, 500).astype(np.int64),
                         "value": rng.uniform(1, 2, 500).astype(np.float64)})  # F64
    dim = pl.DataFrame({"id": np.arange(100, dtype=np.int64),
                        "vol": rng.uniform(0.1, 0.5, 100).astype(np.float64)})
    lf = (fact.lazy().join(dim.lazy(), on="id", how="left")
          .with_columns((pl.col("vol") * pl.col("value")).alias("out")))
    assert isinstance(_walk_plan(lf), FallBack)


def test_walk_join_fallback_on_string_key():
    fact = pl.DataFrame({"id": ["a", "b", "a"], "value": np.float32([1, 2, 3])})
    dim = pl.DataFrame({"id": ["a", "b"], "vol": np.float32([0.1, 0.2])})
    lf = (fact.lazy().join(dim.lazy(), on="id", how="left")
          .with_columns((pl.col("vol") * pl.col("value")).alias("out")))
    assert isinstance(_walk_plan(lf), FallBack)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/python_integration/test_m10_walk_join.py -v`
Expected: FAIL (`_walk_at_current` returns `FallBack(unsupported IR node: Join)` → `test_walk_join_handled...` fails).

- [ ] **Step 3: Implement `_walk_join`**

Add to `_walk_at_current` (after the `HStack` arm at `_walker.py:173`):
```python
    if cls == "Join":
        return _walk_join(nt, node)
```

Add `_walk_join` (model on `_walk_hstack`'s input-recursion + guard style). Use the attribute names recorded in Task 1.1 (`left_on`/`right_on`/`options`):
```python
# Integer key dtypes valid as gather indices (dense-path check happens at execution).
_JOIN_INT_KEY_DTYPES = {"Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32"}


def _walk_join(nt: Any, node: Any) -> WalkResult:
    """Lower an equi-join on a single integer key feeding an F32 fused chain.

    Returns a `Join` plan node carrying both scan sub-plans + key/how metadata.
    The PARENT (Select/HStack) recognizes this via `_walk_at_current` recursion;
    here we only validate the join itself and lower its two scan inputs. The
    dense-vs-CPU-lookup execution choice is deferred to dispatch (Task 1.3 / 3.2).
    """
    left_on = list(getattr(node, "left_on", []) or [])
    right_on = list(getattr(node, "right_on", []) or [])
    if len(left_on) != 1 or len(right_on) != 1:
        return FallBack(reason="join: only single-key equi-join supported")

    # Resolve key column names + dtypes from each input's schema.
    inputs = nt.get_inputs()
    if len(inputs) != 2:
        return FallBack(reason=f"join expected 2 inputs, got {len(inputs)}")

    parent_id = nt.get_node()
    nt.set_node(inputs[0])
    left_schema = dict(nt.get_schema())
    nt.set_node(inputs[1])
    right_schema = dict(nt.get_schema())
    nt.set_node(parent_id)

    lkey = _column_expr_name(nt, left_on[0])    # helper below
    rkey = _column_expr_name(nt, right_on[0])
    if lkey is None or rkey is None:
        return FallBack(reason="join: non-Column key expression")
    if str(left_schema.get(lkey)) not in _JOIN_INT_KEY_DTYPES:
        return FallBack(reason=f"join: key dtype {left_schema.get(lkey)} not an integer key")

    how = _join_how(node)                       # helper below; maps options -> "left"|"inner"
    if how not in ("left", "inner"):
        return FallBack(reason=f"join: how={how} not in (left, inner)")

    # Lower both scan inputs.
    nt.set_node(inputs[0]); left = _walk_at_current(nt)
    nt.set_node(inputs[1]); right = _walk_at_current(nt)
    nt.set_node(parent_id)
    if isinstance(left, FallBack):
        return left
    if isinstance(right, FallBack):
        return right
    if left.plan.get("kind") != "Scan" or right.plan.get("kind") != "Scan":
        return FallBack(reason="join: inputs are not plain scans")

    return Handled(plan={
        "kind": "Join",
        "left": left.plan,
        "right": right.plan,
        "key": lkey,
        "right_key": rkey,
        "how": how,
    })
```

Add helpers near the top-of-file helpers:
```python
def _column_expr_name(nt: Any, ir_expr: Any) -> str | None:
    try:
        inner = nt.view_expression(getattr(ir_expr, "node"))
    except Exception:
        return None
    return str(getattr(inner, "name", "")) or None if type(inner).__name__ == "Column" else None


def _join_how(node: Any) -> str:
    """Map the Join node's options to a lowercase how string. Uses the attr
    recorded in Task 1.1; defensively handles both `.options.args.how` shapes."""
    opts = getattr(node, "options", None)
    raw = repr(opts).lower()
    if "inner" in raw:
        return "inner"
    if "left" in raw:
        return "left"
    return "unsupported"
```

> NOTE for implementer: Task 1.1 prints the real `join_attrs`/`join_how`. If `left_on` items are raw ints (not objects with `.node`), adjust `_column_expr_name` to take the int directly. If `how` lives somewhere other than `options`, fix `_join_how` to read the recorded attribute. Keep the guard semantics identical.

Also: the parent `_walk_hstack`/`_walk_select` already recurse into their input via `_walk_at_current`, so once `_walk_join` is wired, an `HStack(chain) → Join` plan becomes fully `Handled` automatically — confirm no change needed there (the existing `inner = _walk_at_current(nt)` now succeeds).

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/python_integration/test_m10_walk_join.py -v`
Expected: PASS (all three). If `test_walk_join_handled` still fails because the parent HStack rejects the fused chain over post-join columns, inspect `_probe_fusion_analyzer` output — the chain leaves must resolve against `in_schema` = post-join schema (which includes `vol`). The `in_schema` HStack passes is its input's schema (the Join output), so this should already include the dim column.

- [ ] **Step 5: Commit**
```bash
cargo fmt --manifest-path crates/polars-metal-core/Cargo.toml 2>/dev/null; ruff check python/polars_metal/_walker.py --fix
git add python/polars_metal/_walker.py tests/python_integration/test_m10_walk_join.py
git commit -m "M10: _walk_join recognizes int-key equi-join -> fused F32 chain"
```

### Task 1.3: Two-scan capture + `_dispatch_join` (CPU lookup → existing fused chain)

**Files:**
- Modify: `python/polars_metal/_udf.py` (`build_udf:51` add `Join` kind; add `_dispatch_join`; teach `_dispatch:89` to route `Join`)
- Test: `tests/python_integration/test_m10_join_dispatch.py`

- [ ] **Step 1: Write the failing end-to-end differential test**

```python
"""End-to-end: engine='metal' on join->chain == Polars CPU, byte-exact.
Phase 1 = CPU-lookup branch (join on CPU, fused chain on GPU)."""
from __future__ import annotations

import numpy as np
import polars as pl
import polars_metal
from polars_metal import MetalEngine
from polars.testing import assert_frame_equal


def _pipeline(fact, dim, how="left"):
    return (fact.lazy().join(dim.lazy(), on="id", how=how)
            .with_columns(
                (pl.col("value") * 0.5
                 * (1.0 + (0.7978845608 * (pl.col("vol").log())).tanh())).alias("out")))


def test_join_chain_matches_cpu_dense_key():
    rng = np.random.default_rng(10)
    n, dim_n = 1_000_000, 20_000
    fact = pl.DataFrame({"id": rng.integers(0, dim_n, n).astype(np.int64),
                         "value": rng.uniform(50, 150, n).astype(np.float32)})
    dim = pl.DataFrame({"id": np.arange(dim_n, dtype=np.int64),
                        "vol": rng.uniform(0.1, 0.5, dim_n).astype(np.float32)})
    lf = _pipeline(fact, dim)
    cpu = lf.collect()
    gpu = lf.collect(engine=MetalEngine())
    assert_frame_equal(cpu, gpu, check_dtypes=True, rtol=1e-3, atol=1e-3)
```

- [ ] **Step 2: Run to verify failure**

Run: `make wheel && python -m pytest tests/python_integration/test_m10_join_dispatch.py -v`
Expected: FAIL — `build_udf` raises `NotImplementedError`/KeyError for `plan["kind"] == "Join"` (callback then silently falls back, so the result is actually correct-but-CPU). To make the test meaningful, assert the GPU path ran: add `MetalEngine(debug=True)` and check logs, OR (preferred) make the test FAIL loudly first by asserting a dispatch counter (Step 3 adds one).

- [ ] **Step 3: Implement two-scan capture + `_dispatch_join`**

In `build_udf` (`_udf.py:51`), add before the scan extraction:
```python
    if plan["kind"] == "Join":
        return _build_join(plan)
```
Add (the `Join` plan's `left`/`right` are full Scan sub-plans, each carrying its `df` side-channel — capture BOTH):
```python
def _build_join(plan: dict) -> Any:
    """Whole-plan scan-source UDF rooted at a Join. Phase 1: CPU lookup +
    existing fused-chain dispatch. Phase 3 adds the resident-gather branch."""
    left_df = plan["left"]["df"]
    right_df = plan["right"]["df"]

    def udf(with_columns, predicate, n_rows, should_time):
        df = _dispatch_join(left_df, right_df, plan)
        if n_rows is not None:
            df = df.slice(0, n_rows)
        return (df, []) if should_time else df

    return udf


def _dispatch_join(left_pydf: Any, right_pydf: Any, plan: dict) -> pl.DataFrame:
    """Phase 1 CPU-lookup branch: do the join on CPU (correct semantics), then
    run the fused chain that sits ABOVE the join via the existing HStack path."""
    left = pl.DataFrame._from_pydf(left_pydf)
    right = pl.DataFrame._from_pydf(right_pydf)
    joined = left.join(right, left_on=plan["key"], right_on=plan["right_key"], how=plan["how"])
    # The parent plan above this Join node is an HStack/Project carrying the
    # fused chain. We rebuilt `joined` as the upstream; delegate the chain.
    parent = plan["_parent_chain"]           # attached by the walker (Step 4)
    return _dispatch_chain_over_frame(joined, parent)
```

Add a small adapter that runs an HStack-fused (or Project-of-HStack) wire plan over an in-memory frame, reusing `_dispatch_hstack_fused`'s body. The cleanest approach: refactor `_dispatch_hstack_fused` (`_udf.py:208`) so its core takes an already-materialized `upstream` frame:
```python
def _dispatch_chain_over_frame(upstream: pl.DataFrame, parent: dict) -> pl.DataFrame:
    if parent["kind"] == "Project":
        inner = _dispatch_chain_over_frame(upstream, parent["input"])
        return pl.DataFrame._from_pydf(inner._df.select(list(parent["columns"])))
    if parent["kind"] == "HStack":
        return _hstack_fused_over_upstream(upstream, parent)   # extracted core
    raise NotImplementedError(f"join parent chain kind {parent['kind']!r}")
```
Refactor `_dispatch_hstack_fused` to call a new `_hstack_fused_over_upstream(upstream, wire_plan)` containing today's per-binding loop (lines ~225–313), and have the original compute `upstream = _dispatch(df_pydf, wire_plan["input"])` then call it. This is a pure extraction — no behavior change; the existing HStack tests must still pass.

- [ ] **Step 4: Attach the parent chain in the walker**

The `Join` plan needs a back-reference to the chain above it. In `_walk_hstack` (and `_walk_select_projection` if a `Select` projection sits above), when the lowered `inner.plan["kind"] == "Join"`, attach the just-built parent dict onto it under `_parent_chain`. Concretely, at the end of `_walk_hstack`'s `Handled(...)` construction (`_walker.py:757`), add:
```python
    handled = Handled(plan={"kind": "HStack", "input": inner.plan, "exprs": out_exprs})
    if inner.plan.get("kind") == "Join":
        inner.plan["_parent_chain"] = handled.plan
    return handled
```
And strip `_parent_chain` in `_strip_side_channels` (`_callback.py:115`) — but note `Join` plans aren't sent to the Rust router at all (the UDF is built directly from the Python plan with side-channels intact). Confirm `execute_with_metal` installs the UDF for `Join`-rooted plans: it calls `_native.compute_lifting_plan(wire_plan)` first. Add an early branch in `execute_with_metal` (`_callback.py`, after `walk`): if the plan tree contains a `Join` node, skip the router and `build_udf` directly (mirror the `has_fused_binding` override at `_callback.py:70`). Implement `_plan_has_join(plan)` like `_plan_has_fused_binding`.

- [ ] **Step 5: Run to verify pass**

Run: `make wheel && python -m pytest tests/python_integration/test_m10_join_dispatch.py -v`
Expected: PASS, byte-exact vs CPU. Confirm the GPU path actually ran via `MetalEngine(debug=True)` logs showing `installed UDF for plan kind=HStack` over a Join.

- [ ] **Step 6: Commit**
```bash
ruff check python/polars_metal/_udf.py python/polars_metal/_callback.py --fix
git add python/polars_metal/_udf.py python/polars_metal/_callback.py python/polars_metal/_walker.py tests/python_integration/test_m10_join_dispatch.py
git commit -m "M10: two-scan UDF + CPU-lookup join dispatch (reuses fused chain)"
```

### Task 1.4: Fallback correctness (non-fusable / unsupported → clean CPU)

**Files:**
- Test: `tests/python_integration/test_m10_join_fallback.py`

- [ ] **Step 1: Write the test** — each unsupported shape must produce the correct CPU result (engine returns identical frame, just on CPU).

```python
import numpy as np, polars as pl
from polars_metal import MetalEngine
from polars.testing import assert_frame_equal

def _check_cpu_parity(lf):
    assert_frame_equal(lf.collect(), lf.collect(engine=MetalEngine()),
                       check_dtypes=True, rtol=1e-3, atol=1e-3)

def test_f64_chain_falls_back():
    f = pl.DataFrame({"id": np.arange(100, dtype=np.int64), "v": np.arange(100, dtype=np.float64)})
    d = pl.DataFrame({"id": np.arange(100, dtype=np.int64), "vol": np.arange(100, dtype=np.float64)})
    _check_cpu_parity(f.lazy().join(d.lazy(), on="id").with_columns((pl.col("vol")*pl.col("v")).alias("o")))

def test_string_key_falls_back():
    f = pl.DataFrame({"id": ["a","b"], "v": np.float32([1,2])})
    d = pl.DataFrame({"id": ["a","b"], "vol": np.float32([3,4])})
    _check_cpu_parity(f.lazy().join(d.lazy(), on="id").with_columns((pl.col("vol")*pl.col("v")).alias("o")))

def test_outer_join_falls_back():
    f = pl.DataFrame({"id": np.int64([0,1,2]), "v": np.float32([1,2,3])})
    d = pl.DataFrame({"id": np.int64([0,1]), "vol": np.float32([3,4])})
    _check_cpu_parity(f.lazy().join(d.lazy(), on="id", how="full").with_columns((pl.col("vol")*pl.col("v")).alias("o")))
```

- [ ] **Step 2: Run** — `make wheel && python -m pytest tests/python_integration/test_m10_join_fallback.py -v` → PASS (fallback produces correct CPU result). Fix `_walk_join` guards if any wrongly route to GPU.

- [ ] **Step 3: Commit**
```bash
git add tests/python_integration/test_m10_join_fallback.py
git commit -m "M10: join-path fallbacks produce correct CPU results"
```

---

## Phase 2 — 1-D `mlx_take` + `OpId::Take` + mixed-length subgraph

> Adds the GPU gather primitive so the dense-key case can be resident. Each step is differentially tested against numpy / the Phase-1 CPU result.

### Task 2.1: 1-D `mlx_take` FFI wrapper

**Files:**
- Modify: `crates/polars-metal-mlx-sys/cxx/mlx_bridge.h` (declare `mlx_op_take`)
- Modify: `crates/polars-metal-mlx-sys/cxx/mlx_bridge.cc` (impl)
- Modify: `crates/polars-metal-mlx-sys/src/lib.rs` (cxx extern decl)
- Modify: `crates/polars-metal-mlx-sys/src/shape.rs` (Rust wrapper `mlx_take`)
- Test: `crates/polars-metal-mlx-sys/tests/test_take.rs`

- [ ] **Step 1: Verify the MLX C++ signature**

Run: `grep -n "array take(" vendor/mlx/mlx/ops.h | head`
Expected: a `take(const array& a, const array& indices, int axis, ...)` and/or `take(const array& a, const array& indices, ...)` (flattened). Use `mlx::core::take(a, indices, /*axis=*/0)` for a 1-D gather over axis 0. **Record the exact signature** before writing the impl.

- [ ] **Step 2: Write the failing kernel test**

```rust
// crates/polars-metal-mlx-sys/tests/test_take.rs
use polars_metal_mlx_sys::array::{mlx_array_from_f32_slice, mlx_array_from_i32_slice, mlx_array_to_f32_vec, mlx_array_eval};
use polars_metal_mlx_sys::shape::mlx_take;

#[test]
fn take_1d_gathers_by_index() {
    // source = [10, 20, 30, 40]; idx = [3, 0, 0, 2] -> [40, 10, 10, 30]
    let src = mlx_array_from_f32_slice(&[10.0, 20.0, 30.0, 40.0], &[4]).unwrap();
    let idx = mlx_array_from_i32_slice(&[3, 0, 0, 2], &[4]).unwrap();
    let out = mlx_take(&src, &idx).unwrap();
    mlx_array_eval(&[out.clone()]).unwrap();
    assert_eq!(mlx_array_to_f32_vec(&out).unwrap(), vec![40.0, 10.0, 10.0, 30.0]);
}
```
> If `mlx_array_from_i32_slice` / `mlx_array_from_f32_slice` helper names differ, grep `crates/polars-metal-mlx-sys/src/array.rs` for the constructors used by existing tests and match them.

- [ ] **Step 3: Run to verify failure** — `cargo test -p polars-metal-mlx-sys take_1d -- --nocapture` → FAIL (no `mlx_take`).

- [ ] **Step 4: Implement (replicate the `take_along_axis` pattern exactly)**

`mlx_bridge.h` (next to the existing `mlx_op_take_along_axis` decl ~line 283):
```cpp
std::shared_ptr<MlxArray> mlx_op_take(
    const std::shared_ptr<MlxArray>& a,
    const std::shared_ptr<MlxArray>& indices);
```
`mlx_bridge.cc` (next to `mlx_op_take_along_axis` ~line 525):
```cpp
std::shared_ptr<MlxArray> mlx_op_take(
    const std::shared_ptr<MlxArray>& a,
    const std::shared_ptr<MlxArray>& indices) {
    auto base = std::make_shared<mlx::core::array>(
        mlx::core::take(*a, *indices, /*axis=*/0));
    return std::shared_ptr<MlxArray>(base, static_cast<MlxArray*>(base.get()));
}
```
`src/lib.rs` (`#[cxx::bridge]` block, next to `mlx_op_take_along_axis`):
```rust
        fn mlx_op_take(
            a: &SharedPtr<MlxArray>,
            indices: &SharedPtr<MlxArray>,
        ) -> Result<SharedPtr<MlxArray>>;
```
`src/shape.rs` (replicate `mlx_take_along_axis:39` ref-chaining):
```rust
/// 1-D gather over axis 0: `out[i] = a[indices[i]]`.
pub fn mlx_take(a: &MlxArrayHandle, indices: &MlxArrayHandle) -> Result<MlxArrayHandle, FfiError> {
    let ptr = ffi::mlx_op_take(&a.ptr, &indices.ptr).map_err(FfiError::from)?;
    let mut refs = a._input_refs.clone();
    refs.extend(indices._input_refs.iter().cloned());
    Ok(MlxArrayHandle { ptr, _input_refs: refs })
}
```

- [ ] **Step 5: Run to verify pass** — `cargo test -p polars-metal-mlx-sys take_1d -- --nocapture` → PASS.

- [ ] **Step 6: Commit**
```bash
cargo fmt -p polars-metal-mlx-sys
git add crates/polars-metal-mlx-sys/
git commit -m "M10: add 1-D mlx_take FFI wrapper"
```

### Task 2.2: `OpId::Take` + op_spec

**Files:**
- Modify: `crates/polars-metal-core/src/fusion/supported_ops.rs` (enum variant + `op_spec`)
- Modify: `crates/polars-metal-core/src/fusion/py.rs` (`op_id_from_str` if it has an explicit map)
- Test: `crates/polars-metal-core/src/fusion/supported_ops.rs` (`#[cfg(test)]`)

- [ ] **Step 1: Write the failing test** (in the `supported_ops.rs` test module):
```rust
#[test]
fn take_op_spec_is_binary_low_flops() {
    let spec = op_spec(OpId::Take);
    assert_eq!(spec.n_args, 2);          // (source, index)
    assert!(spec.flops_per_row <= 1);    // gather is bandwidth, ~0 compute
}
```

- [ ] **Step 2: Run** — `cargo test -p polars-metal-core take_op_spec` → FAIL (no `OpId::Take`).

- [ ] **Step 3: Implement** — add `Take` to the `OpId` enum (after `MatMul`, near the binary ops ~line 90):
```rust
    // Gather: out[i] = source[index[i]] (binary: source, index). Output length =
    // index length (may differ from source length — the only mixed-length op).
    Take,
```
In `op_spec` add an arm:
```rust
        OpId::Take => OpSpec { n_args: 2, flops_per_row: 1, dynamic_flops: false, /* ...match struct fields used by neighbors... */ },
```
> Copy the `OpSpec { ... }` field layout from an existing binary arm (e.g. `Mul`) and set `n_args: 2`, `flops_per_row: 1`. If `op_id_from_str` (in `py.rs` or `supported_ops.rs`) is an explicit match, add `"Take" => Some(OpId::Take)`.

- [ ] **Step 4: Run** — `cargo test -p polars-metal-core take_op_spec` → PASS.

- [ ] **Step 5: Commit**
```bash
cargo fmt -p polars-metal-core
git add crates/polars-metal-core/src/fusion/
git commit -m "M10: add OpId::Take (binary gather, low flops)"
```

### Task 2.3: `build_op` Take arm + mixed-length subgraph eval

**Files:**
- Modify: `crates/polars-metal-core/src/fusion/subgraph.rs` (`build_op:390` add `Take` arm)
- Test: `crates/polars-metal-core/src/fusion/subgraph.rs` (`#[cfg(test)]`) or `crates/polars-metal-core/tests/test_take_subgraph.rs`

- [ ] **Step 1: Write the failing test** — a 2-input scope (short source + long index) with a `Take` then a unary op, evaluated end-to-end, matches a numpy gather. Model on existing subgraph tests (grep `from_fusion_scope_buffers` in tests). The source buffer length differs from the index/output length — this pins the mixed-length contract.

```rust
// Build scope: in0 = source(F32, len 4), in1 = index(I32, len 5);
// op0 = Take(in0, in1) -> len 5; op1 = Sqrt(op0). Compare to numpy.
#[test]
fn take_then_sqrt_mixed_length() {
    use crate::fusion::scope::{FusionScope, InputDtype};
    use crate::fusion::supported_ops::OpId;
    let mut s = FusionScope::new();
    let src = s.add_input("src", InputDtype::F32);   // len 4
    let idx = s.add_input("idx", InputDtype::I32);    // len 5
    let t = s.push_op_param(OpId::Take, vec![src, idx], None);
    let r = s.push_op_param(OpId::Sqrt, vec![t], None);
    s.mark_output(r);
    // source = [1,4,9,16]; idx = [3,0,2,1,0] -> take=[16,1,9,4,1] -> sqrt=[4,1,3,2,1]
    let src_buf = make_f32_buffer(&[1.0, 4.0, 9.0, 16.0]);
    let idx_buf = make_i32_buffer(&[3, 0, 2, 1, 0]);
    let sg = MlxSubgraph::from_fusion_scope_buffers(&s, &[src_buf, idx_buf]).unwrap();
    let out = sg.eval_to_f32_vec().unwrap();   // use the eval/readback helper existing tests use
    assert_eq!(out, vec![4.0, 1.0, 3.0, 2.0, 1.0]);
}
```
> Use whatever buffer-construction + eval helpers the existing subgraph tests use (grep the test module). If none expose a single-output f32 readback, eval `sg.outputs[0]` via `mlx_array_eval` + `mlx_array_to_f32_vec`.

- [ ] **Step 2: Run** — `cargo test -p polars-metal-core take_then_sqrt_mixed_length` → FAIL (`build_op` has no `Take` arm → `BuildError`).

- [ ] **Step 3: Implement the `Take` arm** in `build_op` (`subgraph.rs:390`). The source handle is `args[0]`, index handle is `args[1]`. The index buffer is viewed by `from_fusion_scope_buffers` as F32/I32 per its `InputDtype`; MLX `take` needs an integer index array — ensure the index input dtype is I32/I64 (the analyzer stages it as such) and cast if needed:
```rust
        Take => {
            arg_count(node.op, 2, &args)?;
            // args[1] is the index column; MLX take wants integer indices.
            let idx = mlx_cast(args[1], MlxDtype::I32).map_err(|e| BuildError::MlxError(format!("{e:?}")))?;
            ffi(mlx_take(args[0], &idx))
        }
```
> `mlx_cast` + `MlxDtype` are already imported in this file (used by other arms). `ffi(...)` is the existing helper wrapping `Result<MlxArrayHandle, FfiError>` into `Result<_, BuildError>`. Import `mlx_take` from `polars_metal_mlx_sys::shape` alongside `mlx_take_along_axis`.

**Mixed-length note:** `from_fusion_scope_buffers` already views each input at its own `buf.len()/elt` length (`subgraph.rs:248`), so the short source and long index coexist. The output of `Take` is index-length; all downstream ops operate on index-length arrays. The ONLY constraint: the short source input must be consumed *only* by `Take` (never elementwise-combined with a long input before the gather). The analyzer (Task 3.2) guarantees this by construction.

- [ ] **Step 4: Run** — `cargo test -p polars-metal-core take_then_sqrt_mixed_length` → PASS.

- [ ] **Step 5: Commit**
```bash
cargo fmt -p polars-metal-core
git add crates/polars-metal-core/src/fusion/subgraph.rs crates/polars-metal-core/tests/ 2>/dev/null
git commit -m "M10: build_op Take arm + mixed-length subgraph eval"
```

---

## Phase 3 — Dense-key detection + resident-gather wiring + differential suite

### Task 3.1: Dense-key detection + dim reordering helper

**Files:**
- Create: `python/polars_metal/_join_gather.py`
- Test: `tests/python_integration/test_m10_dense_detect.py`

- [ ] **Step 1: Write the failing test**

```python
from polars_metal._join_gather import dense_positions
import numpy as np, polars as pl

def test_dense_key_detected_and_reordered():
    # dim keyed 0..n in shuffled order; value follows key.
    key = np.array([2, 0, 3, 1], dtype=np.int64)
    val = np.array([20., 0., 30., 10.], dtype=np.float32)
    # dense_positions returns (is_dense, value_reordered_by_key) s.t. out[k]=val for key k
    is_dense, reordered = dense_positions(key, val, dim_height=4)
    assert is_dense
    np.testing.assert_array_equal(reordered, [0., 10., 20., 30.])

def test_nondense_key_rejected():
    key = np.array([0, 1, 5], dtype=np.int64)   # gap -> not 0..n-1
    val = np.array([1., 2., 3.], dtype=np.float32)
    is_dense, _ = dense_positions(key, val, dim_height=3)
    assert not is_dense

def test_duplicate_key_rejected():
    key = np.array([0, 0, 1], dtype=np.int64)
    val = np.array([1., 2., 3.], dtype=np.float32)
    is_dense, _ = dense_positions(key, val, dim_height=3)
    assert not is_dense
```

- [ ] **Step 2: Run** — `python -m pytest tests/python_integration/test_m10_dense_detect.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement**

```python
"""Dense-key detection for join->gather lowering. A left/inner equi-join on an
integer key equals `value_reordered[fact_key]` IFF the dim key is a permutation
of 0..dim_height-1 (unique, contiguous, no nulls). Otherwise the caller takes
the CPU-lookup branch."""
from __future__ import annotations
import numpy as np


def dense_positions(key: np.ndarray, value: np.ndarray, dim_height: int):
    """Return (is_dense, value_reordered) where value_reordered[k] is the dim
    value for key k. is_dense is False when key is not a 0..dim_height-1
    permutation (gaps, duplicates, out-of-range, or nulls)."""
    if key.shape[0] != dim_height or value.shape[0] != dim_height:
        return False, None
    if key.min() != 0 or key.max() != dim_height - 1:
        return False, None
    reordered = np.empty(dim_height, dtype=value.dtype)
    reordered[key] = value                      # scatter; duplicates -> lost slots
    # Verify it was a true permutation (no duplicate overwrote a slot).
    seen = np.zeros(dim_height, dtype=bool)
    seen[key] = True
    if not seen.all():
        return False, None
    return True, reordered
```
> Null keys: a nullable dim key arrives as a numpy array with a separate null mask — the caller passes `key` only when `dim.get_column(rkey).null_count() == 0` (checked in Task 3.2); otherwise non-dense branch.

- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit**
```bash
ruff check python/polars_metal/_join_gather.py --fix
git add python/polars_metal/_join_gather.py tests/python_integration/test_m10_dense_detect.py
git commit -m "M10: dense-key detection + dim value reordering"
```

### Task 3.2: Resident-gather branch in `_dispatch_join`

**Files:**
- Modify: `python/polars_metal/_udf.py` (`_dispatch_join` dense branch)
- Modify: `python/polars_metal/_fusion_analyzer.py` (the chain's `vol`-leaf must become `Take(dim_value, fact_key)`)
- Test: `tests/python_integration/test_m10_resident_gather.py`

- [ ] **Step 1: Write the failing test** — dense path must byte-match the Phase-1 CPU result AND must actually run the GPU gather (assert via a module-level counter the dense branch increments).

```python
import numpy as np, polars as pl
from polars_metal import MetalEngine
from polars_metal import _udf
from polars.testing import assert_frame_equal

def _pipeline(fact, dim, how="left"):
    return (fact.lazy().join(dim.lazy(), on="id", how=how)
            .with_columns((pl.col("value")*0.5*(1.0+(0.7978845608*pl.col("vol").log()).tanh())).alias("out")))

def test_dense_resident_gather_matches_cpu():
    rng = np.random.default_rng(11)
    n, dim_n = 1_000_000, 20_000
    fact = pl.DataFrame({"id": rng.integers(0, dim_n, n).astype(np.int64),
                         "value": rng.uniform(50,150,n).astype(np.float32)})
    dim = pl.DataFrame({"id": rng.permutation(dim_n).astype(np.int64),   # dense, shuffled
                        "vol": rng.uniform(0.1,0.5,dim_n).astype(np.float32)})
    lf = _pipeline(fact, dim)
    _udf._M10_DENSE_GATHERS = 0
    gpu = lf.collect(engine=MetalEngine())
    assert _udf._M10_DENSE_GATHERS == 1, "dense branch did not run"
    assert_frame_equal(lf.collect(), gpu, check_dtypes=True, rtol=1e-3, atol=1e-3)
```

- [ ] **Step 2: Run** — `make wheel && python -m pytest tests/python_integration/test_m10_resident_gather.py -v` → FAIL (counter 0: today everything takes the CPU-lookup branch).

- [ ] **Step 3: Implement the dense branch**

The challenge: the fused chain above the join references `vol` (the dim value column) as a plain Column leaf resolving to a length-N input. For the resident gather, `vol` must instead be `Take(dim_value[len dim_n], fact_key[len N])`. Two viable approaches — pick **A** (simpler, localized):

**Approach A — rewrite the fused inputs at dispatch.** Keep the analyzer/scope as built (chain over N-length `vol` + `value`), but in the dense branch of `_dispatch_join`, instead of materializing `vol` on CPU, build an *augmented scope*: prepend the dim-value column + fact-key column as inputs and a `Take` op, and rebind the chain's `vol` input slot to the Take output. This requires scope surgery — fragile.

**Approach B (chosen) — let the chain run over the gathered column, but do the gather resident as a separate fused output, then feed it.** Simplest correct resident form: build ONE scope = `[fact_key(I32 idx), dim_value(F32 src), fact_value(F32)] + Take(src, idx) + <chain ops over Take-output and fact_value>`. The analyzer already builds the chain over `{vol, value}`; we splice a Take in front of the `vol` leaf.

Concretely, add a helper in `_fusion_analyzer.py` that, given an accepted chain scope + the descriptor index of the `vol` column, returns a NEW scope where that input is replaced by a `Take(dim_value_input, key_input)` subgraph. Implement as a scope-rebuild:
```python
def splice_gather_input(scope, descriptors, gather_col, key_col):
    """Return (new_scope, new_descriptors) where `gather_col`'s input slot is
    fed by Take(dim_value, key). dim_value stays SHORT (dim_n); key + all other
    inputs stay LONG (N). Preserves op order and output marking."""
    # Rebuild: PASS-1 inputs = [key(I32), dim_value(F32), <other long cols/lits>...]
    # then push Take(dim_value_idx, key_idx); remap the old gather_col leaf idx to
    # the Take node idx; replay original ops with remapped arg indices.
    ...
```
> This is genuine new logic — the implementer writes it against `FusionScope`/`PyFusionScope` (`add_input`/`push_op`). The differential test (Step 1) + the mixed-length subgraph test (Task 2.3) are the safety net; the byte-exact match to CPU is the gate. Keep the remap explicit and unit-test `splice_gather_input` on a 2-op chain.

In `_dispatch_join` dense branch:
```python
_M10_DENSE_GATHERS = 0   # module-level test counter

def _dispatch_join(left_pydf, right_pydf, plan):
    left = pl.DataFrame._from_pydf(left_pydf)
    right = pl.DataFrame._from_pydf(right_pydf)
    parent = plan["_parent_chain"]
    binding = _single_fused_binding(parent)            # the HStack's one fused expr
    gather_col = plan["right_key_value_col"]           # the dim value col the chain reads
    rkey = plan["right_key"]; lkey = plan["key"]
    if (plan["how"] in ("left", "inner")
            and right.get_column(rkey).null_count() == 0
            and left.get_column(lkey).null_count() == 0):
        key = left.get_column(lkey).to_numpy()
        val = right.get_column(gather_col).to_numpy()
        is_dense, reordered = dense_positions(np.asarray(right.get_column(rkey).to_numpy()),
                                              np.asarray(val), right.height)
        if is_dense and left.height >= 1:
            global _M10_DENSE_GATHERS; _M10_DENSE_GATHERS += 1
            return _hstack_resident_gather(left, key, reordered, binding, parent)
    # fall through to CPU-lookup branch (Phase 1)
    joined = left.join(right, left_on=lkey, right_on=rkey, how=plan["how"])
    return _dispatch_chain_over_frame(joined, parent)
```
`_hstack_resident_gather` builds the spliced scope and calls `execute_fused_expr` with inputs `[key(int32), reordered_dim_value(f32), <other fact cols>...]`, output length = `left.height`, then stitches the result column(s) and returns `left.with_columns(...)` matching `parent`'s output schema (apply the Project if `parent` is a Project-of-HStack).

> The walker must record `right_key_value_col` (the dim column the chain consumes) and ensure the chain has exactly ONE gathered dim column for the dense path (multiple dim value columns → MVP CPU-lookup branch). Add that to `_walk_join`'s `Handled` plan: scan the fused binding's `_fused_columns` for names present in the dim (right) schema but not the fact (left) schema.

- [ ] **Step 4: Run** — PASS (counter == 1, byte-exact). If the spliced-scope result diverges, debug with a tiny N=8 case comparing `execute_fused_expr` output to numpy gather+chain before the 1M case.

- [ ] **Step 5: Commit**
```bash
cargo fmt -p polars-metal-core 2>/dev/null; ruff check python/polars_metal/ --fix
git add python/polars_metal/_udf.py python/polars_metal/_fusion_analyzer.py python/polars_metal/_walker.py tests/python_integration/test_m10_resident_gather.py
git commit -m "M10: resident dense-key gather branch (Take spliced into fused chain)"
```

### Task 3.3: Full differential suite

**Files:**
- Create: `tests/python_integration/test_m10_join_differential.py`

- [ ] **Step 1: Write the parametrized differential suite** — every case byte-exact vs CPU.

```python
import numpy as np, polars as pl, pytest
from polars_metal import MetalEngine
from polars.testing import assert_frame_equal

KEY_DTYPES = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32]

def _chain(): return (pl.col("value")*0.5*(1.0+(0.7978845608*pl.col("vol").log()).tanh())).alias("out")

def _run(fact, dim, how):
    lf = fact.lazy().join(dim.lazy(), on="id", how=how).with_columns(_chain())
    assert_frame_equal(lf.collect(), lf.collect(engine=MetalEngine()),
                       check_dtypes=True, rtol=1e-3, atol=1e-3)

@pytest.mark.parametrize("how", ["left", "inner"])
@pytest.mark.parametrize("kd", KEY_DTYPES)
def test_dense(how, kd):
    rng = np.random.default_rng(20)
    dim_n = min(200, np.iinfo(kd).max)
    n = 5000
    fact = pl.DataFrame({"id": rng.integers(0, dim_n, n).astype(kd), "value": rng.uniform(50,150,n).astype(np.float32)})
    dim = pl.DataFrame({"id": rng.permutation(dim_n).astype(kd), "vol": rng.uniform(0.1,0.5,dim_n).astype(np.float32)})
    _run(fact, dim, how)

@pytest.mark.parametrize("how", ["left", "inner"])
def test_nondense_sparse_keys(how):
    rng = np.random.default_rng(21)
    keys = (rng.choice(10_000, 300, replace=False)).astype(np.int64)   # sparse, gaps
    fact = pl.DataFrame({"id": rng.choice(keys, 5000).astype(np.int64), "value": rng.uniform(50,150,5000).astype(np.float32)})
    dim = pl.DataFrame({"id": keys, "vol": rng.uniform(0.1,0.5,len(keys)).astype(np.float32)})
    _run(fact, dim, how)

def test_left_join_missing_keys_yield_nulls():
    fact = pl.DataFrame({"id": np.int64([0,1,2,3]), "value": np.float32([10,20,30,40])})
    dim = pl.DataFrame({"id": np.int64([0,2]), "vol": np.float32([0.1,0.2])})  # 1,3 missing
    _run(fact, dim, "left")

def test_duplicate_dim_keys_explode():
    fact = pl.DataFrame({"id": np.int64([0,1]), "value": np.float32([10,20])})
    dim = pl.DataFrame({"id": np.int64([0,0,1]), "vol": np.float32([0.1,0.2,0.3])})  # dup -> 1:many
    _run(fact, dim, "inner")

def test_null_keys():
    fact = pl.DataFrame({"id": pl.Series([0,None,2], dtype=pl.Int64), "value": np.float32([10,20,30])})
    dim = pl.DataFrame({"id": np.int64([0,1,2]), "vol": np.float32([0.1,0.2,0.3])})
    _run(fact, dim, "left")

def test_empty_dim():
    fact = pl.DataFrame({"id": np.int64([0,1]), "value": np.float32([10,20])})
    dim = pl.DataFrame({"id": np.array([], np.int64), "vol": np.array([], np.float32)})
    _run(fact, dim, "left")

def test_single_row():
    fact = pl.DataFrame({"id": np.int64([0]), "value": np.float32([7])})
    dim = pl.DataFrame({"id": np.int64([0]), "vol": np.float32([0.3])})
    _run(fact, dim, "inner")
```

- [ ] **Step 2: Run** — `make wheel && python -m pytest tests/python_integration/test_m10_join_differential.py -v` → all PASS. Any failure is a cardinal correctness bug; fix the guard/branch (likely: duplicate/missing/null cases must NOT take the dense branch — `dense_positions` + null-count guards enforce this; verify the CPU-lookup branch reproduces Polars join semantics exactly, including left-join nulls propagating through the chain).

- [ ] **Step 3: Commit**
```bash
git add tests/python_integration/test_m10_join_differential.py
git commit -m "M10: full join-gather differential suite (dense/nondense/nulls/dups)"
```

---

## Phase 4 — P1 resident vector rerank

### Task 4.1: `vector_search_topk` resident rerank

**Files:**
- Modify: `crates/polars-metal-core/src/vector_search.rs` (`vector_search_topk:42`, `execute_vector_search:165`)
- Test: `crates/polars-metal-core/tests/test_vector_rerank.rs`

- [ ] **Step 1: Write the failing test** — with a weight vector, scores become `sim * exp(-weight[hit])`.

```rust
// Tiny corpus; verify reranked = val_k * exp(-weight[idx]) vs hand math.
#[test]
fn cosine_topk_with_exp_decay_rerank() {
    // 1 query, 3 corpus rows, D=2, k=2, weights bias toward low-weight rows.
    let q = vec![1.0f32, 0.0];
    let c = vec![1.0,0.0,  0.9,0.1,  0.0,1.0];  // row0 best, row2 worst
    let w = vec![0.0f32, 2.0, 0.0];
    let (idx, val) = vector_search_topk_rerank(&q,1,&c,3,2,2, OP_COSINE, Some(&w)).unwrap();
    // For each returned (i, score): score ≈ cos_sim_i * exp(-w[i]).
    // (exact assertion: recompute sims, pick top-2 by reranked, compare set+values)
    assert_eq!(idx.len(), 2);
    // ... recompute expected reranked and assert_allclose ...
}
```
> Name the rerank-enabled entry `vector_search_topk_rerank` (or add an `Option<&[f32]>` weight param to `vector_search_topk` and update callers). Match the existing tile wrapper.

- [ ] **Step 2: Run** — `cargo test -p polars-metal-core cosine_topk_with_exp_decay` → FAIL.

- [ ] **Step 3: Implement** — after `val_k` is computed (`vector_search.rs:84`), if a weight is provided, gather it resident and combine. The rerank applies BEFORE the final eval/readback so it stays in one graph:
```rust
    // val_k: (Q,k) similarities. Resident rerank: feat = take(weight, idx_k); val_k *= exp(-feat).
    let final_val = if let Some(w) = weight {
        let wv = view1d(w, n_rows)?;                              // (N,)
        let idx_flat = mlx_reshape(&idx_k_i, &[(q_rows * k) as i32])?;   // (Q*k,)
        let feat = mlx_take(&wv, &idx_flat)?;                     // (Q*k,)
        let feat2d = mlx_reshape(&feat, &[q_rows as i32, k as i32])?;    // (Q,k)
        let neg = mlx_neg(&feat2d)?;
        let decay = mlx_exp(&neg)?;
        mlx_mul(&val_k, &decay)?
    } else {
        val_k.clone()
    };
    mlx_array_eval(&[idx_k_i.clone(), final_val.clone()])?;
    let indices = mlx_array_to_i32_vec(&idx_k_i)?;
    let values = mlx_array_to_f32_vec(&final_val)?;
```
> `view1d` / `mlx_take` / `mlx_neg` / `mlx_exp` / `mlx_mul` are available (`mlx_take` from Task 2.1; rest from `elementwise.rs`). `idx_k_i` is already I32 — reshape is contiguous (memory: reshape of a fresh array is safe; argpartition/take_along_axis produce contiguous output). The reranked `values` are no longer raw cosine sims, so the Python `_build_struct` sort must sort by reranked score (still "desc by score" for cosine) — confirm ordering in Task 4.3.

Thread an `Option<&[f32]>` weight through `vector_search_topk_tiled` and `execute_vector_search`. In the PyO3 signature add `weight: Option<(usize, usize)>`:
```rust
#[pyo3(signature = (query, q_rows, corpus, n_rows, d, k, op, tile_rows, weight=None))]
pub fn execute_vector_search(... , weight: Option<(usize, usize)>) -> PyResult<(Vec<u32>, Vec<f32>)> {
    let w = weight.map(|(p, l)| unsafe { std::slice::from_raw_parts(p as *const f32, l) });
    ...
}
```

- [ ] **Step 4: Run** — `cargo test -p polars-metal-core cosine_topk_with_exp_decay` → PASS.
- [ ] **Step 5: Commit**
```bash
cargo fmt -p polars-metal-core
git add crates/polars-metal-core/src/vector_search.rs crates/polars-metal-core/tests/test_vector_rerank.rs
git commit -m "M10: resident exp-decay rerank in vector_search_topk"
```

### Task 4.2: Namespace API + dispatch wiring

**Files:**
- Modify: `python/polars_metal/_vector_namespace.py` (`cosine_topk:98` — add `rerank_weight`, `rerank`)
- Modify: `python/polars_metal/_vector_dispatch.py` (`_run_binding:79` — pass weight)
- Test: `tests/python_integration/test_m10_vector_rerank.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np, polars as pl
from polars_metal import MetalEngine

def test_cosine_topk_rerank_matches_numpy():
    rng = np.random.default_rng(30)
    N, D, Q, k = 2000, 64, 50, 10
    corpus = rng.standard_normal((N, D)).astype(np.float32)
    weight = rng.uniform(0, 1, N).astype(np.float32)
    queries = rng.standard_normal((Q, D)).astype(np.float32)
    qdf = pl.DataFrame({"emb": [list(map(float, r)) for r in queries]},
                       schema={"emb": pl.Array(pl.Float32, D)})
    cdf = pl.DataFrame({"emb": [list(map(float, r)) for r in corpus]},
                       schema={"emb": pl.Array(pl.Float32, D)})
    res = (qdf.lazy().select(pl.col("emb").metal.cosine_topk(cdf, k, rerank_weight=pl.Series(weight), rerank="exp_decay").alias("hit"))
           .collect(engine=MetalEngine()))
    # numpy oracle: cosine sims -> top-k -> sim*exp(-weight[hit]), re-sort desc
    qn = queries/np.linalg.norm(queries,axis=1,keepdims=True)
    cn = corpus/np.linalg.norm(corpus,axis=1,keepdims=True)
    sims = qn@cn.T
    for qi in range(Q):
        top = np.argpartition(-sims[qi], k-1)[:k]
        rer = sims[qi, top]*np.exp(-weight[top])
        order = top[np.argsort(-rer)]
        got = res["hit"][qi]["indices"]
        assert set(int(x) for x in got) == set(int(x) for x in top)
        np.testing.assert_allclose(sorted(res["hit"][qi]["scores"], reverse=True),
                                   sorted(rer, reverse=True), rtol=1e-3, atol=1e-3)

def test_rerank_weight_length_mismatch_raises():
    import pytest
    # weight length != corpus rows -> raise
    ...
```

- [ ] **Step 2: Run** — `make wheel && python -m pytest tests/python_integration/test_m10_vector_rerank.py -v` → FAIL (`cosine_topk` has no `rerank_weight`).

- [ ] **Step 3: Implement** — extend `cosine_topk` (and the corpus capture) to carry `rerank_weight` (a `pl.Series`/array, F32, len = corpus rows) + `rerank` (only `"exp_decay"` for v1; `None` = today's behavior). Store on the `CorpusSpec`. In `_run_binding` (`_vector_dispatch.py:79`), if the spec has a weight, materialize it as a contiguous f32 numpy array, validate `len == n_rows` (else `ValueError`), and pass `weight=(w.ctypes.data, w.size)` to `_native.execute_vector_search`. Keep `_build_struct` ordering "desc by score" for cosine (reranked scores are still larger-is-better).

- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit**
```bash
ruff check python/polars_metal/_vector_namespace.py python/polars_metal/_vector_dispatch.py --fix
git add python/polars_metal/_vector_namespace.py python/polars_metal/_vector_dispatch.py tests/python_integration/test_m10_vector_rerank.py
git commit -m "M10: cosine_topk rerank_weight/exp_decay (resident rerank)"
```

---

## Phase 5 — Force-route override + honest perf wiring + gate

### Task 5.1: `MetalEngine.force_fusion` override

**Files:**
- Modify: `python/polars_metal/_engine.py` (add field)
- Modify: `python/polars_metal/_callback.py` / wherever the density decision is consulted for fused/join plans
- Test: `tests/python_integration/test_m10_force_route.py`

- [ ] **Step 1: Write the failing test** — a below-threshold join→chain stays CPU by default; with `force_fusion=True` it runs the GPU path (assert via the dense counter or a debug log / dispatch flag).

```python
import numpy as np, polars as pl
from polars_metal import MetalEngine
from polars_metal import _udf
from polars.testing import assert_frame_equal

def _small_dense_pipeline():
    rng = np.random.default_rng(40)
    n, dim_n = 2000, 100   # n < MIN_ROWS_THRESHOLD (1e5) -> default CPU
    fact = pl.DataFrame({"id": rng.integers(0,dim_n,n).astype(np.int64), "value": rng.uniform(50,150,n).astype(np.float32)})
    dim = pl.DataFrame({"id": rng.permutation(dim_n).astype(np.int64), "vol": rng.uniform(0.1,0.5,dim_n).astype(np.float32)})
    return fact.lazy().join(dim.lazy(), on="id").with_columns((pl.col("value")*(pl.col("vol").exp())).alias("out"))

def test_below_threshold_defaults_cpu():
    lf = _small_dense_pipeline()
    _udf._M10_DENSE_GATHERS = 0
    out = lf.collect(engine=MetalEngine())
    assert _udf._M10_DENSE_GATHERS == 0           # routed CPU by density guard
    assert_frame_equal(lf.collect(), out, check_dtypes=True, rtol=1e-3, atol=1e-3)

def test_force_fusion_overrides():
    lf = _small_dense_pipeline()
    _udf._M10_DENSE_GATHERS = 0
    out = lf.collect(engine=MetalEngine(force_fusion=True))
    assert _udf._M10_DENSE_GATHERS == 1           # forced GPU
    assert_frame_equal(lf.collect(), out, check_dtypes=True, rtol=1e-3, atol=1e-3)
```
> This requires the join path to consult the density guard. Wire it: in `_walk_join` (or dispatch), call `scope.route_decision(n_rows)` (the same `route_decision` `_probe_fusion_analyzer:540` uses) with `n_rows = left.height` and, when it routes CPU and `force_fusion` is False, return `FallBack`. Pass `force_fusion` from `MetalEngine` through `execute_with_metal(config=...)` into the walker (thread it via the plan or a context).

- [ ] **Step 2: Run** — FAIL (no `force_fusion` field; join path doesn't consult density).

- [ ] **Step 3: Implement** — add to `_engine.py`:
```python
    force_fusion: bool = False
    """Force recognized fused/gather subtrees onto the GPU past the density gate
    (FLOPs/rows). For benchmarking and power users; default keeps routing honest."""
```
Plumb `config.force_fusion` into `execute_with_metal` (it already receives `config`) and have the join/fused routing honor it: route GPU iff `force_fusion or density_routes_gpu(...) == Gpu`. The density n_rows for a join is the fact (left) height.

- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit**
```bash
ruff check python/polars_metal/_engine.py python/polars_metal/_callback.py --fix
git add python/polars_metal/_engine.py python/polars_metal/_callback.py python/polars_metal/_walker.py tests/python_integration/test_m10_force_route.py
git commit -m "M10: MetalEngine.force_fusion override + density gate on join path"
```

### Task 5.2: Honest perf-report registry cases

**Files:**
- Modify: `tests/bench/m8_report/registry.py` (add P2 dense, P2 non-dense, P1 rerank cases)
- Test: smoke via `make perf-report` (or the m8 smoke gate)

- [ ] **Step 1: Add registry entries** following the existing m8 registry pattern (grep the file for an existing case to copy). Each case: a `setup` building fact/dim (or corpus/weight), an `engine_fn` (`collect(engine=MetalEngine())`), a `cpu_fn` (`collect()`), and a `ceiling_fn` (raw MLX/numpy: P2 = `_p2_resident`-shape from `tests/bench/m9_crossing/_pipelines.py`; P1 = numpy argpartition+rerank). Report medians; both columns (engine-vs-CPU + engine-vs-ceiling) + tax.
  - `m10_join_gather_dense_n10m` (dim_n=20k, N=10M, Black-Scholes chain)
  - `m10_join_gather_nondense_n10m` (sparse keys)
  - `m10_vector_rerank_q100_n200k_d256_k10`

- [ ] **Step 2: Smoke** — `python -m pytest tests/bench/m8_report -k smoke -v` (or the wired smoke gate) → PASS. Do NOT add `ratio_lt` gates that would fail CI on perf; these are honest-report rows, not gates (per CLAUDE.md: benchmarks are not tests).

- [ ] **Step 3: Run the report + record numbers**

Run: `make perf-report` (or the registry runner). Record the engine-vs-CPU medians for the three cases in the PR description. Expect ~3–8× (dense), ~2–4× (non-dense), ~20–30× (rerank). If dense ≤ non-dense, the resident path isn't engaging — debug before claiming the win.

- [ ] **Step 4: Commit**
```bash
git add tests/bench/m8_report/registry.py
git commit -m "M10: perf-report cases (join-gather dense/nondense, vector rerank)"
```

### Task 5.3: Full gate + memory + finish

**Files:** none (verification) + memory file

- [ ] **Step 1: Run the full gate**

Run: `make gate`
Expected: clippy + fmt + ruff clean; `test-unit`, `test-kernel`, `test-conformance` green. Fix any drift (the subagent-fmt gotcha — mod/use order, multiline-return wrap, `pytest.raises(Exception)` B017). Re-run until green. Capture the conformance baseline — known pre-existing divergences (Mean F32 dtype, etc.) are expected per memory; M10 must not ADD failures.

- [ ] **Step 2: Differential sweep sanity** — `python -m pytest tests/python_integration/test_m10_*.py -v` all green.

- [ ] **Step 3: Write the M10 execution-state memory**

Create `/Users/dclark/.claude/projects/-Users-dclark-dev-polars-metal-main-polars-metal/memory/m10-resident-gather-execution-state.md` (type: project) capturing: shipped scope (join→gather recognition via `_walk_join`, two-scan UDF, dense/non-dense branch, vector rerank, `force_fusion`); the honest medians; the load-bearing gotchas found during build (the real Join IR attrs from Task 1.1, the mixed-length subgraph contract, the `splice_gather_input` remap, any optimizer-shape surprises); and add a one-line pointer to `MEMORY.md`. Link `[[m9 verdict]]`, `[[m8-honest-perf-direction]]`, `[[m6-vector-search-execution-state]]`, `[[m4-nodetraverser-opacity]]`.

- [ ] **Step 4: Finish the branch** — invoke `superpowers:finishing-a-development-branch` to open the PR (one combined PR, per the architect's decision). PR body: the scope, the honest engine-vs-CPU medians, the guardrails honored (no GPU hash join, F64→CPU, groupby/sort untouched), and the differential-suite coverage.

---

## Self-review notes (addressed)

- **Spec coverage:** P2 join recognition (Tasks 1.2–1.3), dense resident gather (Phase 2–3), non-dense CPU lookup (Task 1.3), correctness guards (Tasks 1.4, 3.1, 3.3), P1 rerank (Phase 4), density guard + `force_fusion` override (Task 5.1), honest perf (Task 5.2), guardrails verification (Task 5.3). All spec sections map to tasks.
- **Risk-first ordering:** the two-scan UDF integration (the spec's "spike first" risk) is Phase 1 and ships a real win with no kernels; the harder mixed-length/Take work follows with the Phase-1 CPU result as a byte-exact oracle.
- **Type consistency:** `Join` plan dict keys (`left`/`right`/`key`/`right_key`/`how`/`_parent_chain`/`right_key_value_col`) are used consistently across `_walk_join`, `_dispatch_join`, and the dense branch. `_M10_DENSE_GATHERS` counter name is consistent across Tasks 3.2/5.1. `mlx_take`/`OpId::Take`/`splice_gather_input` names are consistent.
- **Known discovery points (flagged inline, not placeholders):** the real Join node attributes (`_join_how`/`_column_expr_name`) are pinned by Task 1.1's printed output; `splice_gather_input` is genuine new logic gated by the mixed-length subgraph test + byte-exact differential. These are de-risked by tests, not hand-waved.
