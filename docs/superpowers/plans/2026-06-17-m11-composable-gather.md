# M11 — Composable resident gather + retrieval flagship — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend M10's single-column resident GPU gather to multiple dim value columns, add `.select` recognition for the `.metal` verbs, and ship a two-phase retrieval flagship that runs resident end-to-end.

**Architecture:** M10 already gathers one dim value column resident (`Take(dim_value, key)` spliced into the F32 chain) and already fires when the fact frame is a materialized `Scan`. M11 generalizes the gather to N columns (one shared key input + one `Take` per referenced dim column), relaxes the single-dim-col walker guard, and produces all output dim columns. `.select` detection extends the existing `with_columns` capture + serialize slow-path. No new execution architecture — the retrieval fact is materialized (eager explode) so it's a `Scan`.

**Tech Stack:** Python engine glue (`polars_metal`), Rust `_native` (already has `OpId::Take` + `mlx_take` from M10 — no Rust changes expected), Polars 1.40.1, pytest differential vs CPU.

**Spec:** `docs/superpowers/specs/2026-06-17-m11-composable-gather-design.md`

**Key current symbols (verified, post-M10-merge):**
- `python/polars_metal/_fusion_analyzer.py`: `analyze_ir_with_columns_gather(nt, node_id, schema, gather_col, key_col):624` (M10 single-col); the gather-aware pass-1 branch at `_gather_leaves_ir:1408` (`name_s == gather_ctx["gather_col"]` → add `("gather_key",k)` I32 + `("gather_value",v)` F32, record `gather_ctx["idxs"]={"key":ki,"value":vi}`); the pass-2 Take push at `_visit_ir_ops:1633-1639` (`scope.push_op("Take", [idxs["value"], idxs["key"]])`); the empty-check at `:676` (`if not gather_ctx["idxs"]`).
- `python/polars_metal/_walker.py`: `_attach_gather_scope:787` (the `if len(dim_cols) != 1: return` guard at `:800`; stashes `join_plan["_gather"]={scope,descriptors,out_dtype,gather_col,key_col,out_name}` + `_out_schema`); `_dim_value_cols` set at `:905`. The fused binding carries `_fused_columns` (descriptor list) + `_expr_node`.
- `python/polars_metal/_udf.py`: `_try_resident_gather(left, right, plan, gather):189` — single-col (uses `gather["gather_col"]`, one `reordered`, one `vol_series = reordered[key_np]`).
- `python/polars_metal/_join_gather.py`: `dense_positions(key, value, dim_height) -> (is_dense, reordered)`.
- `python/polars_metal/_detect_common.py`: `install_with_columns_capture(attr, cache):97` (patches `LazyFrame.with_columns`); `iter_candidate_nodes(lf, *, cache, explain_tags):201` (fast path = serialize cached exprs; slow path matches `key='"exprs":['` at `:226`).
- `tests/bench/m8_report/registry.py`: M10 added `m10_join_gather_dense` etc. — copy that entry shape.

**Conventions:** no `unwrap()` outside `*-sys`/tests (and `#![allow(clippy::unwrap_used, clippy::expect_used)]` atop any new `tests/*.rs`); run `ruff format` + `ruff check --fix` per task (not just `ruff check`); `make wheel` only if Rust changes (none expected); `make gate` now runs `tests/python_integration` via `test-integration`; differential tests use `rel_tol`/`abs_tol` (not deprecated `rtol`/`atol`).

---

## Phase 0 — Baseline

### Task 0.1: Confirm green baseline on the M11 branch

**Files:** none

- [ ] **Step 1: Verify branch + build + gate-lite**

Run:
```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
git rev-parse --abbrev-ref HEAD     # expect: m11-composable-gather
git log --oneline -1                # expect the M11 spec commit
make wheel && python -m pytest tests/python_integration/ -q
```
Expected: build succeeds; `tests/python_integration` = all pass except the known `test_groupby_i32_keys_i32_values_matches_cpu` **xfail** (it's marked xfail now, so the run is green). If red, STOP and report.

---

## Phase 1 — Multi-column resident gather

### Task 1.1: Multi-column gather-aware analyzer

**Files:**
- Modify: `python/polars_metal/_fusion_analyzer.py` (`analyze_ir_with_columns_gather:624`, the pass-1 branch `:1408`, the pass-2 branch `:1633`, the empty-check `:676`)
- Test: `tests/python_integration/test_m11_gather_scope.py`

The M10 analyzer splices ONE `Take` for ONE `gather_col`. Generalize to a SET of gather columns sharing one key input.

- [ ] **Step 1: Write the failing isolated test** (mirrors M10's `test_m10_gather_scope.py`: capture a real chain expr, run the gather scope through `execute_fused_expr`, compare to numpy with TWO gathered dim columns).

```python
"""Isolated: analyze_ir_with_columns_gather with MULTIPLE dim value columns.
The spliced scope (Take(price,key), Take(rating,key) feeding an F32 chain) must
match a numpy gather+chain."""
from __future__ import annotations
import numpy as np
import polars as pl
from polars_metal import _native
from polars_metal._fusion_analyzer import analyze_ir_with_columns_gather


def _capture(lf, gather_cols, key_col):
    out = {}
    def cb(nt, _d=None):
        def visit(nid):
            nt.set_node(nid)
            node = nt.view_current_node()
            if type(node).__name__ == "HStack":
                inputs = nt.get_inputs()
                parent = nt.get_node()
                nt.set_node(inputs[0]); schema = dict(nt.get_schema()); nt.set_node(parent)
                e = node.exprs[0]
                out["res"] = analyze_ir_with_columns_gather(nt, e.node, schema, gather_cols, key_col)
                return True
            for i in nt.get_inputs():
                if visit(i):
                    return True
            return False
        visit(nt.get_node())
    lf.collect(engine="cpu", post_opt_callback=cb)
    return out["res"]


def test_multi_col_gather_scope_matches_numpy():
    rng = np.random.default_rng(7)
    n, dim_n = 4096, 256
    fact_id = rng.integers(0, dim_n, n).astype(np.int64)
    sc = rng.uniform(0, 1, n).astype(np.float32)
    price = rng.uniform(0.1, 2.0, dim_n).astype(np.float32)   # position == key
    rating = rng.uniform(1, 5, dim_n).astype(np.float32)

    fact = pl.DataFrame({"id": fact_id, "sc": sc})
    dim = pl.DataFrame({"id": np.arange(dim_n, dtype=np.int64), "price": price, "rating": rating})
    # chain reads BOTH dim cols: sc * exp(price) * log(rating)
    lf = (fact.lazy().join(dim.lazy(), on="id", how="left")
          .with_columns((pl.col("sc") * pl.col("price").exp() * pl.col("rating").log()).alias("out")))

    res = _capture(lf, ["price", "rating"], "id")
    assert res is not None
    scope, descriptors, out_dtype = res
    assert out_dtype == "F32"
    kinds = [d[0] for d in descriptors]
    assert kinds.count("gather_key") == 1, descriptors          # ONE shared key
    gv = [p for k, p in descriptors if k == "gather_value"]
    assert set(gv) == {"price", "rating"}, descriptors           # one value input per dim col

    # build inputs in descriptor order
    arrays, tags = [], []
    short = {"price": price, "rating": rating}
    for kind, payload in descriptors:
        if kind == "gather_key":
            a = np.ascontiguousarray(fact_id, dtype=np.int32); t = 2
        elif kind == "gather_value":
            a = np.ascontiguousarray(short[payload], dtype=np.float32); t = 0   # SHORT (dim_n)
        elif kind == "col":
            assert payload == "sc"; a = np.ascontiguousarray(sc, dtype=np.float32); t = 0
        elif kind == "lit":
            a = np.asarray([payload], dtype=np.float32); t = 0
        else:
            raise AssertionError(kind)
        arrays.append(a); tags.append(t)
    out = np.empty(n, dtype=np.float32)
    inputs = [(int(a.__array_interface__["data"][0]), int(a.size), t) for a, t in zip(arrays, tags)]
    assert _native.execute_fused_expr(scope=scope, inputs=inputs,
        out=(int(out.__array_interface__["data"][0]), int(out.size), 0)) == n
    expect = sc * np.exp(price[fact_id]) * np.log(rating[fact_id])
    np.testing.assert_allclose(out, expect, rtol=1e-3, atol=1e-3)


def test_single_col_still_works():
    # back-compat: a single gather col still produces one key + one value.
    rng = np.random.default_rng(8)
    n, dim_n = 1024, 64
    fact = pl.DataFrame({"id": rng.integers(0, dim_n, n).astype(np.int64),
                         "sc": rng.uniform(0, 1, n).astype(np.float32)})
    vol = rng.uniform(0.1, 0.5, dim_n).astype(np.float32)
    dim = pl.DataFrame({"id": np.arange(dim_n, dtype=np.int64), "vol": vol})
    lf = (fact.lazy().join(dim.lazy(), on="id", how="left")
          .with_columns((pl.col("sc") * pl.col("vol").log()).alias("out")))
    res = _capture(lf, ["vol"], "id")
    assert res is not None
    _, descriptors, _ = res
    assert [k for k, _ in descriptors].count("gather_key") == 1
    assert [p for k, p in descriptors if k == "gather_value"] == ["vol"]
```

- [ ] **Step 2: Run → FAIL.** `python -m pytest tests/python_integration/test_m11_gather_scope.py -v` — fails because `analyze_ir_with_columns_gather`'s 4th param is a single `gather_col` string, not a list (and the gather_ctx is single-valued).

- [ ] **Step 3: Implement the multi-col change.**

`analyze_ir_with_columns_gather` (`:624`) — change the param + the gather_ctx shape + the empty-check:
```python
def analyze_ir_with_columns_gather(
    nt: Any,
    node_id: int,
    schema: dict[str, Any],
    gather_cols,                      # was: gather_col: str  -> now an iterable of column names
    key_col: str,
) -> tuple[PyFusionScope, list[tuple[str, str | float]], str] | None:
    ...
    gather_ctx: dict = {
        "gather_cols": set(gather_cols),
        "key_col": key_col,
        "idxs": {"key": None, "values": {}},
    }
    ...
    if not gather_ctx["idxs"]["values"]:   # was: if not gather_ctx["idxs"]
        return None
```
Keep the docstring but update it to say "one or more dim value columns, sharing one key input."

Pass-1 gather branch (`_gather_leaves_ir`, `:1408`) — replace:
```python
        if gather_ctx is not None and name_s in gather_ctx["gather_cols"]:
            if gather_ctx["idxs"]["key"] is None:
                key_dtype = schema.get(gather_ctx["key_col"])
                if key_dtype is None:
                    raise _Aborted
                ki = scope.add_input(gather_ctx["key_col"], "I32")
                descriptors.append(("gather_key", gather_ctx["key_col"]))
                gather_ctx["idxs"]["key"] = ki
            if name_s not in gather_ctx["idxs"]["values"]:
                vi = scope.add_input(name_s, "F32")
                descriptors.append(("gather_value", name_s))
                gather_ctx["idxs"]["values"][name_s] = vi
            # gather leaf is not a plain input; pass 2 builds the Take.
            return
```

Pass-2 gather branch (`_visit_ir_ops`, `:1633-1639`) — replace the `name_s == gather_ctx["gather_col"]` Take push:
```python
        if gather_ctx is not None and name_s in gather_ctx["gather_cols"]:
            return scope.push_op(
                "Take",
                [gather_ctx["idxs"]["values"][name_s], gather_ctx["idxs"]["key"]],
            )
```

> IMPORTANT: the base `analyze_ir_with_columns` (no gather_ctx) path must stay byte-identical — the gather branch is still guarded by `gather_ctx is not None`. Run the existing fusion tests to confirm.

- [ ] **Step 4: Run → PASS.** `python -m pytest tests/python_integration/test_m11_gather_scope.py -v` (both tests). Then `python -m pytest tests/python_integration/ -k "fused or gather or hstack" -q` — no regressions (base analyzer + M10 single-col path unchanged; the M10 `test_m10_gather_scope.py` calls the function with a single-string `gather_col` — it now must pass a list: see Step 5).

- [ ] **Step 5: Fix the M10 caller signature in its isolated test.** `tests/python_integration/test_m10_gather_scope.py` calls `analyze_ir_with_columns_gather(nt, e.node, schema, gather_col="vol", key_col="id")`. Update that call to `analyze_ir_with_columns_gather(nt, e.node, schema, ["vol"], "id")` (positional list). Re-run `python -m pytest tests/python_integration/test_m10_gather_scope.py -v` → still passes. (The real M10 caller `_attach_gather_scope` is updated in Task 1.2.)

- [ ] **Step 6: Commit.**
```bash
ruff format python/polars_metal/_fusion_analyzer.py tests/python_integration/test_m11_gather_scope.py tests/python_integration/test_m10_gather_scope.py
ruff check python/polars_metal/_fusion_analyzer.py --fix
git add python/polars_metal/_fusion_analyzer.py tests/python_integration/test_m11_gather_scope.py tests/python_integration/test_m10_gather_scope.py
git commit -m "M11: multi-column gather-aware analyzer (N Takes sharing one key)"
```

### Task 1.2: Walker recognition + dispatch for N dim columns

**Files:**
- Modify: `python/polars_metal/_walker.py` (`_attach_gather_scope:787`)
- Modify: `python/polars_metal/_join_gather.py` (add `is_dense_key`, `reorder_by_key`)
- Modify: `python/polars_metal/_udf.py` (`_try_resident_gather:189`)
- Test: `tests/python_integration/test_m11_multicol_gather.py`

- [ ] **Step 1: Write the failing end-to-end resident test.**

```python
import numpy as np, polars as pl
from polars_metal import MetalEngine
from polars_metal import _udf
from polars.testing import assert_frame_equal


def _pipeline(fact, dim, how="left"):
    # rerank reads TWO dim columns: sc * exp(price) * log(rating)
    return (fact.lazy().join(dim.lazy(), on="id", how=how)
            .with_columns((pl.col("sc") * pl.col("price").exp() * pl.col("rating").log()).alias("rr")))


def test_multicol_resident_gather_matches_cpu():
    rng = np.random.default_rng(11)
    n, dim_n = 1_000_000, 20_000
    fact = pl.DataFrame({"id": rng.integers(0, dim_n, n).astype(np.int64),
                         "sc": rng.uniform(0, 1, n).astype(np.float32)})
    dim = pl.DataFrame({"id": rng.permutation(dim_n).astype(np.int64),
                        "price": rng.uniform(0.1, 2.0, dim_n).astype(np.float32),
                        "rating": rng.uniform(1, 5, dim_n).astype(np.float32)})
    lf = _pipeline(fact, dim)
    _udf._M10_DENSE_GATHERS = 0
    gpu = lf.collect(engine=MetalEngine(force_fusion=True))
    assert _udf._M10_DENSE_GATHERS == 1, "multi-col resident gather did not run"
    assert_frame_equal(lf.collect(), gpu, check_dtypes=True, rel_tol=1e-3, abs_tol=1e-3)
```

- [ ] **Step 2: Run → FAIL.** `make wheel >/dev/null 2>&1; python -m pytest tests/python_integration/test_m11_multicol_gather.py -v` — fails: `_attach_gather_scope` returns early at `len(dim_cols) != 1`, so the path is CPU (counter 0).

- [ ] **Step 3: Add the reorder helpers** to `python/polars_metal/_join_gather.py`:
```python
def is_dense_key(key: np.ndarray, dim_height: int) -> bool:
    """True iff `key` is a permutation of 0..dim_height-1 (unique, contiguous,
    in range, no nulls). Same predicate dense_positions uses internally."""
    if dim_height <= 0 or key.shape[0] != dim_height:
        return False
    if int(key.min()) != 0 or int(key.max()) != dim_height - 1:
        return False
    seen = np.zeros(dim_height, dtype=bool)
    seen[key] = True
    return bool(seen.all())


def reorder_by_key(key: np.ndarray, value: np.ndarray, dim_height: int) -> np.ndarray:
    """Return `value` reordered so position == key: out[key[i]] = value[i].
    Assumes `is_dense_key(key, dim_height)` already passed."""
    out = np.empty(dim_height, dtype=value.dtype)
    out[key] = value
    return out
```
(Leave `dense_positions` as-is; M10's single-col path and its tests still use it.)

- [ ] **Step 4: Generalize `_attach_gather_scope`** (`_walker.py:787`). Replace the single-col body:
```python
    if len(out_exprs) != 1 or "_fused_scope" not in out_exprs[0]:
        return
    dim_cols = join_plan.get("_dim_value_cols", [])
    if not dim_cols:
        return
    key_col = join_plan["key"]
    # The dim columns the fused chain actually reads (a subset of dim_cols).
    binding_cols = {name for kind, name in out_exprs[0].get("_fused_columns", []) if kind == "col"}
    gather_cols = [c for c in dim_cols if c in binding_cols]
    if not gather_cols:
        return                      # chain reads no dim column -> nothing to gather
    expr_node = out_exprs[0].get("_expr_node")
    if expr_node is None:
        return
    try:
        res = analyze_ir_with_columns_gather(nt, expr_node, in_schema, gather_cols, key_col)
    except Exception:
        return
    if res is None:
        return
    join_plan["_gather"] = {
        "scope": res[0],
        "descriptors": res[1],
        "out_dtype": res[2],
        "gather_cols": gather_cols,         # chain-referenced dim cols (Take inputs)
        "out_dim_cols": list(dim_cols),     # ALL non-key dim cols (output reconstruction)
        "key_col": key_col,
        "out_name": out_exprs[0]["name"],
    }
    join_plan["_out_schema"] = out_schema
```

- [ ] **Step 5: Generalize `_try_resident_gather`** (`_udf.py:189`). Replace `gather["gather_col"]`/single-`reordered` logic:
```python
    import numpy as np
    from polars_metal._join_gather import is_dense_key, reorder_by_key

    how = plan["how"]; lkey = plan["key"]; rkey = plan["right_key"]
    if how not in ("left", "inner"):
        return None
    if left.height == 0 or right.height == 0:
        return None
    if left.get_column(lkey).null_count() or right.get_column(rkey).null_count():
        return None
    dim_n = right.height
    rkey_np = np.asarray(right.get_column(rkey).to_numpy())
    if not is_dense_key(rkey_np, dim_n):
        return None
    key_np = np.asarray(left.get_column(lkey).to_numpy())
    if int(key_np.min()) < 0 or int(key_np.max()) >= dim_n:
        return None

    out_dim_cols = gather["out_dim_cols"]
    # reorder every dim column we need (chain inputs + output passthrough) once.
    needed = set(gather["gather_cols"]) | set(out_dim_cols)
    reordered = {
        c: reorder_by_key(rkey_np, np.ascontiguousarray(right.get_column(c).to_numpy(), dtype=np.float32), dim_n)
        for c in needed
    }

    n = left.height
    arrays: list[np.ndarray] = []
    tags: list[int] = []
    for kind, payload in gather["descriptors"]:
        if kind == "gather_key":
            a = np.ascontiguousarray(key_np, dtype=np.int32); tag = 2
        elif kind == "gather_value":
            a = np.ascontiguousarray(reordered[payload], dtype=np.float32); tag = 0
        elif kind == "col":
            a = np.ascontiguousarray(left.get_column(payload).to_numpy(), dtype=np.float32); tag = 0
        elif kind == "lit":
            a = np.asarray([payload], dtype=np.float32); tag = 0
        else:
            return None
        arrays.append(a); tags.append(tag)

    out_arr = np.empty(n, dtype=np.float32)
    inputs = [(int(a.__array_interface__["data"][0]), int(a.size), t)
              for a, t in zip(arrays, tags, strict=True)]
    written = _native.execute_fused_expr(
        scope=gather["scope"], inputs=inputs,
        out=(int(out_arr.__array_interface__["data"][0]), int(out_arr.size), 0))
    if written != n:
        return None

    # output: every non-key dim column (cheap dense index) + the chain result.
    dim_series = [pl.Series(c, reordered[c][key_np]) for c in out_dim_cols]
    out_series = pl.Series(gather["out_name"], out_arr)
    result = left.with_columns([*dim_series, out_series])
    out_schema = plan.get("_out_schema")
    if out_schema is not None:
        result = result.select([name for name, _ in out_schema])
    return result
```
> Note: `reorder_by_key` casts each dim column to F32 (the gather scope is F32). A dim value column that's F32 in the output is reproduced exactly; if a dim column in `out_dim_cols` is NOT F32 in the CPU result, the dtype would mismatch — guard in Step 6 (force CPU fallback when any non-key dim col is non-F32). For the MVP/retrieval target all metadata value cols are F32.

- [ ] **Step 6: Guard non-F32 dim columns.** In `_attach_gather_scope`, before stashing `_gather`, confirm every column in `dim_cols` is F32 in `in_schema` (the join output schema); if any isn't, `return` (→ CPU-lookup, byte-exact). Add:
```python
    if any(str(in_schema.get(c)) != "Float32" for c in dim_cols):
        return
```
(Place after computing `dim_cols`, before building the scope.)

- [ ] **Step 7: Run → PASS.** `python -m pytest tests/python_integration/test_m11_multicol_gather.py -v` (counter 1, byte-exact). Then the M10 resident test still passes: `python -m pytest tests/python_integration/test_m10_resident_gather.py -v`.

Wait — the M10 dispatch reads `gather["gather_col"]`/`gather["out_dim_cols"]`? M10 stashed `gather_col` (singular) + had no `out_dim_cols`. Since `_attach_gather_scope` (the only producer of `_gather`) now stashes `gather_cols`/`out_dim_cols`, and `_try_resident_gather` (the only consumer) now reads them, they're consistent. The M10 single-col case flows through the SAME new code (gather_cols=["vol"], out_dim_cols=["vol"]). Confirm `test_m10_resident_gather.py` + `test_m10_join_differential.py` are green (they now exercise the generalized path).

- [ ] **Step 8: Commit.**
```bash
ruff format python/polars_metal/_walker.py python/polars_metal/_udf.py python/polars_metal/_join_gather.py tests/python_integration/test_m11_multicol_gather.py
ruff check python/polars_metal/ --fix
git add python/polars_metal/_walker.py python/polars_metal/_udf.py python/polars_metal/_join_gather.py tests/python_integration/test_m11_multicol_gather.py
git commit -m "M11: resident gather over N dim value columns (walker + dispatch)"
```

### Task 1.3: Multi-column differential suite

**Files:**
- Create: `tests/python_integration/test_m11_multicol_differential.py`

- [ ] **Step 1: Write the suite** (extends M10's matrix to multiple dim cols; every case byte-exact vs CPU; if any fails it's a correctness bug — do NOT loosen).

```python
import numpy as np, polars as pl, pytest
from polars_metal import MetalEngine
from polars.testing import assert_frame_equal


def _run(fact, dim, chain, how="left"):
    lf = fact.lazy().join(dim.lazy(), on="id", how=how).with_columns(chain)
    assert_frame_equal(lf.collect(), lf.collect(engine=MetalEngine(force_fusion=True)),
                       check_dtypes=True, rel_tol=1e-3, abs_tol=1e-3)


def _dim(dim_n, rng, cols=("price", "rating")):
    d = {"id": rng.permutation(dim_n).astype(np.int64)}
    for c in cols:
        d[c] = rng.uniform(0.5, 3.0, dim_n).astype(np.float32)
    return pl.DataFrame(d)


@pytest.mark.parametrize("how", ["left", "inner"])
def test_two_cols_dense(how):
    rng = np.random.default_rng(20)
    dim_n, n = 2000, 200_000
    fact = pl.DataFrame({"id": rng.integers(0, dim_n, n).astype(np.int64),
                         "sc": rng.uniform(0, 1, n).astype(np.float32)})
    chain = (pl.col("sc") * pl.col("price").exp() * pl.col("rating").log()).alias("rr")
    _run(fact, _dim(dim_n, rng), chain, how)


def test_three_cols_chain_reads_subset():
    # dim has 3 value cols; chain reads only 2 -> the 3rd is output passthrough.
    rng = np.random.default_rng(21)
    dim_n, n = 1000, 100_000
    fact = pl.DataFrame({"id": rng.integers(0, dim_n, n).astype(np.int64),
                         "sc": rng.uniform(0, 1, n).astype(np.float32)})
    dim = _dim(dim_n, rng, cols=("price", "rating", "weight"))
    chain = (pl.col("sc") * pl.col("price").log() + pl.col("rating")).alias("rr")  # reads price, rating; weight passthrough
    _run(fact, dim, chain, "left")


def test_left_join_missing_keys_nulls():
    rng = np.random.default_rng(22)
    fact = pl.DataFrame({"id": np.int64([0, 1, 2, 3]), "sc": np.float32([1, 2, 3, 4])})
    dim = pl.DataFrame({"id": np.int64([0, 2]), "price": np.float32([0.5, 0.7]),
                        "rating": np.float32([2.0, 3.0])})  # 1,3 missing
    _run(fact, dim, (pl.col("sc") * pl.col("price") + pl.col("rating")).alias("rr"), "left")


def test_nondense_sparse_falls_back_correct():
    rng = np.random.default_rng(23)
    keys = rng.choice(50_000, 1000, replace=False).astype(np.int64)
    fact = pl.DataFrame({"id": rng.choice(keys, 20_000).astype(np.int64),
                         "sc": rng.uniform(0, 1, 20_000).astype(np.float32)})
    dim = pl.DataFrame({"id": keys, "price": rng.uniform(0.5, 3, len(keys)).astype(np.float32),
                        "rating": rng.uniform(1, 5, len(keys)).astype(np.float32)})
    _run(fact, dim, (pl.col("sc") * pl.col("price").exp() * pl.col("rating").log()).alias("rr"), "left")


def test_non_f32_dim_col_falls_back():
    # an Int64 dim col in the output -> resident path must decline (dtype guard) but stay byte-exact.
    rng = np.random.default_rng(24)
    dim_n, n = 500, 50_000
    fact = pl.DataFrame({"id": rng.integers(0, dim_n, n).astype(np.int64),
                         "sc": rng.uniform(0, 1, n).astype(np.float32)})
    dim = pl.DataFrame({"id": rng.permutation(dim_n).astype(np.int64),
                        "price": rng.uniform(0.5, 3, dim_n).astype(np.float32),
                        "code": rng.integers(0, 9, dim_n).astype(np.int64)})  # non-F32
    _run(fact, dim, (pl.col("sc") * pl.col("price").exp()).alias("rr"), "left")
```

- [ ] **Step 2: Run → all PASS.** `python -m pytest tests/python_integration/test_m11_multicol_differential.py -v`. For the dangerous cases: `test_three_cols_chain_reads_subset` (the unread `weight` must appear in output via passthrough), `test_non_f32_dim_col_falls_back` (the Int64 `code` forces CPU-lookup — assert it still matches). If any fails, debug the guard/branch (don't loosen tolerances).

- [ ] **Step 3: Commit.**
```bash
ruff format tests/python_integration/test_m11_multicol_differential.py
git add tests/python_integration/test_m11_multicol_differential.py
git commit -m "M11: multi-column gather differential suite (2-3 cols, subset, nulls, non-F32)"
```

---

## Phase 2 — `.select` detection

### Task 2.1: Recognize `.metal` verbs under `LazyFrame.select`

**Files:**
- Modify: `python/polars_metal/_detect_common.py` (`install_with_columns_capture:97`, `iter_candidate_nodes:201`)
- Test: `tests/python_integration/test_m11_select_detection.py`

- [ ] **Step 1: Write the failing parity test** (each verb under `.select` matches `.with_columns` + CPU). Start with two representative verbs (rolling = column-output, dt = column-output; vector = struct sentinel).

```python
import numpy as np, polars as pl
from polars_metal import MetalEngine
from polars.testing import assert_frame_equal


def test_rolling_under_select():
    rng = np.random.default_rng(0)
    df = pl.DataFrame({"x": rng.standard_normal(50_000).astype(np.float32)})
    lf = df.lazy().select(pl.col("x").rolling_mean(window_size=100).alias("rm"))
    assert_frame_equal(lf.collect(), lf.collect(engine=MetalEngine()),
                       check_dtypes=True, rel_tol=1e-3, abs_tol=1e-3)


def test_dt_under_select():
    d = pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 12, 31), interval="1d", eager=True)
    df = pl.DataFrame({"d": d})
    lf = df.lazy().select(pl.col("d").dt.year().alias("yr"))
    assert_frame_equal(lf.collect(), lf.collect(engine=MetalEngine()), check_dtypes=True)


def test_cosine_topk_under_select():
    rng = np.random.default_rng(1)
    N, D, Q, k = 2000, 64, 30, 5
    corpus = pl.DataFrame({"emb": [list(map(float, r)) for r in rng.standard_normal((N, D)).astype(np.float32)]},
                          schema={"emb": pl.Array(pl.Float32, D)})
    q = pl.DataFrame({"emb": [list(map(float, r)) for r in rng.standard_normal((Q, D)).astype(np.float32)]},
                     schema={"emb": pl.Array(pl.Float32, D)})
    # .select must detect the sentinel (today only with_columns does)
    res = q.lazy().select(pl.col("emb").metal.cosine_topk(corpus, k).alias("hit")).collect(engine=MetalEngine())
    assert res.columns == ["hit"]
    assert res["hit"].dtype == pl.Struct({"indices": pl.List(pl.UInt32), "scores": pl.List(pl.Float32)})
    assert res.height == Q
```

- [ ] **Step 2: Run → FAIL.** `python -m pytest tests/python_integration/test_m11_select_detection.py -v` — `test_cosine_topk_under_select` raises (the sentinel `_raise_cpu` fires because `.select` isn't detected → falls through to CPU which raises). rolling/dt may pass (they degrade to CPU gracefully) — but they should run on GPU; assert-parity still holds either way. The decisive failure is the vector one.

- [ ] **Step 3: Extend the capture to also patch `select`.** In `install_with_columns_capture` (`:97`), after installing the `with_columns` wrapper, also wrap `select` into the SAME cache. Rename-safe approach — append:
```python
    # Also capture `.select(...)` exprs so the .metal verbs are detected under
    # the projection idiom, not only with_columns. Same cache, keyed by id(result).
    sel_attr = attr + "_select"
    if not hasattr(_plf.LazyFrame, sel_attr):
        orig_select = _plf.LazyFrame.select
        setattr(_plf.LazyFrame, sel_attr, orig_select)

        def _patched_select(self, *exprs, **named):  # type: ignore[no-untyped-def]
            result = orig_select(self, *exprs, **named)
            try:
                flat = [e for e in exprs if isinstance(e, pl.Expr)]
                # select also accepts a single list arg
                for e in exprs:
                    if isinstance(e, (list, tuple)):
                        flat += [x for x in e if isinstance(x, pl.Expr)]
                flat += [e.alias(n) for n, e in named.items() if isinstance(e, pl.Expr)]
                if flat:
                    key = id(result)
                    cache[key] = (weakref.ref(result, _make_evictor(cache, key)), flat)
            except Exception:
                pass
            return result

        _plf.LazyFrame.select = _patched_select  # type: ignore[method-assign]
```

- [ ] **Step 4: Extend the slow path** in `iter_candidate_nodes` (`:226`) to also match the `Select` node's `"expr":[` fragment when `"exprs":[` is absent. Replace the `key = '"exprs":['` lookup with a fallback:
```python
        key = '"exprs":['
        i = plan.rfind(key)
        if i == -1:
            key = '"expr":['        # Select node (projection idiom)
            i = plan.rfind(key)
        if i == -1:
            return
```
(Keep the rest of the slow-path parse identical — it parses the fragment after `key`.)

- [ ] **Step 5: Run → PASS.** `python -m pytest tests/python_integration/test_m11_select_detection.py -v` (all three). Then broaden the test to the remaining verbs (fft, dtw, corr) following the same `.select` shape and confirm parity; add those cases to the file. Run `python -m pytest tests/python_integration/ -k "vector or rolling or fft or dtw or dt or corr" -q` → no regressions (with_columns paths unaffected — the select patch is additive).

- [ ] **Step 6: Commit.**
```bash
ruff format python/polars_metal/_detect_common.py tests/python_integration/test_m11_select_detection.py
ruff check python/polars_metal/_detect_common.py --fix
git add python/polars_metal/_detect_common.py tests/python_integration/test_m11_select_detection.py
git commit -m "M11: detect .metal verbs under LazyFrame.select (capture + slow-path)"
```

---

## Phase 3 — Retrieval flagship

### Task 3.1: Two-phase retrieval benchmark + correctness

**Files:**
- Modify: `tests/bench/m8_report/registry.py` (add `m11_retrieval_rerank`)
- Test: `tests/python_integration/test_m11_retrieval.py`

- [ ] **Step 1: Write the end-to-end correctness test** (the two-phase pipeline byte-exact vs an all-CPU run).

```python
import numpy as np, polars as pl
from polars_metal import MetalEngine
from polars_metal import _udf
from polars.testing import assert_frame_equal


def _retrieval(queries, corpus, metadata, k, engine):
    hits = queries.lazy().with_columns(
        hit=pl.col("emb").metal.cosine_topk(corpus, k)).collect(engine=engine)
    fact = (hits.lazy()
            .with_columns(idx=pl.col("hit").struct.field("indices"),
                          sc=pl.col("hit").struct.field("scores"))
            .explode(["idx", "sc"])
            .with_columns(idx=pl.col("idx").cast(pl.Int64))
            .collect())                                   # eager -> fact is a Scan
    return (fact.lazy()
            .join(metadata.lazy(), left_on="idx", right_on="id", how="left")
            .with_columns(rr=(pl.col("sc") * pl.col("price").exp() * pl.col("rating").log()))
            .collect(engine=engine))


def test_retrieval_pipeline_resident_matches_cpu():
    rng = np.random.default_rng(3)
    Q, N, D, k = 4000, 5000, 64, 10
    corpus = pl.DataFrame({"emb": [list(map(float, r)) for r in rng.standard_normal((N, D)).astype(np.float32)]},
                          schema={"emb": pl.Array(pl.Float32, D)})
    queries = pl.DataFrame({"emb": [list(map(float, r)) for r in rng.standard_normal((Q, D)).astype(np.float32)]},
                           schema={"emb": pl.Array(pl.Float32, D)})
    metadata = pl.DataFrame({"id": rng.permutation(N).astype(np.int64),
                             "price": rng.uniform(0.1, 2.0, N).astype(np.float32),
                             "rating": rng.uniform(1, 5, N).astype(np.float32)})
    _udf._M10_DENSE_GATHERS = 0
    gpu = _retrieval(queries, corpus, metadata, k, MetalEngine(force_fusion=True))
    assert _udf._M10_DENSE_GATHERS == 1, "metadata gather did not run resident"
    cpu = _retrieval(queries, corpus, metadata, k, "cpu")
    assert_frame_equal(cpu.sort(["idx", "rr"]), gpu.sort(["idx", "rr"]),
                       check_dtypes=True, rel_tol=1e-3, abs_tol=1e-3)
```

- [ ] **Step 2: Run → it should PASS** (the machinery exists after Phases 1–2). `python -m pytest tests/python_integration/test_m11_retrieval.py -v`. The counter==1 proves the rerank phase ran resident multi-col. If counter==0, the explode'd fact wasn't materialized to a Scan before the join — confirm the `.collect()` after explode (the eager boundary).

- [ ] **Step 3: Add the report-only registry case.** In `tests/bench/m8_report/registry.py`, copy the M10 `m10_join_gather_dense` entry shape and add `m11_retrieval_rerank`: setup builds queries/corpus/metadata (seeded); the engine_fn runs the two-phase `_retrieval(... MetalEngine(force_fusion=True))`, cpu_fn runs `_retrieval(... "cpu")`, ceiling_fn = raw numpy (cosine top-k + gather + rerank). Size: Q=2000, N=200k corpus, D=256, k=10 (exploded rows = 20k for the smoke size; headline size Q large enough that exploded rows ≥ ~500k to clear the gather gate). No `ratio_lt` gate (report-only). Match the existing entry fields exactly so the m8 smoke test passes.

- [ ] **Step 4: Smoke the registry + run the report.**
```bash
python -m pytest tests/bench/m8_report/test_smoke.py tests/bench/m8_report/test_harness.py -q   # PASS
```
Then measure (record the medians for the PR): run the registry runner for just `m11_retrieval_rerank` (or `make perf-report`), and note engine-vs-CPU × for the rerank phase + the top-k phase. Expect the rerank phase 3–7× (resident multi-col), top-k at the M6 vector rate.

- [ ] **Step 5: Commit.**
```bash
ruff format tests/bench/m8_report/registry.py tests/python_integration/test_m11_retrieval.py
ruff check tests/bench/m8_report/registry.py --fix
git add tests/bench/m8_report/registry.py tests/python_integration/test_m11_retrieval.py
git commit -m "M11: two-phase retrieval flagship (resident multi-col rerank) + perf case"
```

---

## Phase 4 — Gate, memory, PR

### Task 4.1: Full gate + memory + finish

**Files:** none + memory

- [ ] **Step 1: Full gate.** `make gate` → green ("M0 gate passed."). It now runs `tests/python_integration` (the M10 `test-integration` target), so all M11 tests are gated. Fix any fmt/lint drift (the recurring subagent gotcha: `ruff format`, clippy `expect_used` allow on new `tests/*.rs` — none expected this milestone). The known groupby xfail stays xfail.

- [ ] **Step 2: Differential sanity.** `python -m pytest tests/python_integration/test_m11_*.py -v` → all green; M10 tests (`test_m10_*`) still green (the generalized gather path subsumes single-col).

- [ ] **Step 3: Write the M11 memory.** Create `/Users/dclark/.claude/projects/-Users-dclark-dev-polars-metal-main-polars-metal/memory/m11-composable-gather-execution-state.md` (type: project): shipped scope (multi-col resident gather, `.select` detection, two-phase retrieval flagship); the brainstorming finding that the resident retrieval win needs only fact-materialization + multi-col (NOT cosine_topk rework or auto-staging); honest medians; gotchas (the gather/output-dim-col split, the non-F32 dim-col guard, the `.select` capture+slow-path). Add a one-line pointer to `MEMORY.md`. Link `[[m10-resident-gather-execution-state]]`, `[[m8-honest-perf-direction]]`, `[[m6-vector-search-execution-state]]`.

- [ ] **Step 4: Finish the branch.** Invoke `superpowers:finishing-a-development-branch` → push + PR. PR body: scope, honest medians, guardrails held (no GPU hash join — dense gather; F64/non-F32 dim col → CPU; groupby/sort untouched), the explicit non-goals (no sentinel rework, no auto-staging) with the spike reasoning, and that the retrieval flagship is two-phase by design.

---

## Self-review notes (addressed)

- **Spec coverage:** multi-col gather (Task 1.1 analyzer, 1.2 walker+dispatch, 1.3 differential); `.select` detection (Task 2.1); retrieval flagship (Task 3.1); guardrails + gate (Task 4.1). All spec sections map.
- **Type consistency:** `_gather` dict keys are consistent across producer (`_attach_gather_scope`) and consumer (`_try_resident_gather`): `gather_cols` (list, chain-referenced), `out_dim_cols` (list, all non-key dim cols), `descriptors`, `scope`, `out_name`, `key_col`. `analyze_ir_with_columns_gather`'s 4th param is `gather_cols` (iterable) everywhere (Task 1.1 changes the M10 isolated-test caller too). `is_dense_key`/`reorder_by_key` names consistent across `_join_gather.py` and `_udf.py`.
- **No new Rust:** `OpId::Take` + `mlx_take` already exist (M10); multiple `Take`s in one scope is just more ops — no kernel change. (If the gather scope with N Takes surfaces any subgraph issue, that's a Task 1.1 finding, but the mixed-length contract already supports it.)
- **Risk-first:** Task 1.1 validates the multi-Take scope in isolation (numpy oracle) before the walker/dispatch wiring builds on it; the M10 single-col path is subsumed and re-verified.
