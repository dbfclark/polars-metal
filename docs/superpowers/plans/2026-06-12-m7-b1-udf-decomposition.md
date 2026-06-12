# M7 B-1 — `udf.rs` Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the 3,000-line `crates/polars-metal-core/src/udf.rs` god-module into focused single-responsibility submodules — a pure, behavior-preserving file move that isolates the live compute ops (fused_expr, rolling, dtw, dt) from the conformance-only groupby/compact code and the shared kernels (cmp, predicate).

**Architecture:** Convert `udf.rs` into a `udf/` module directory. Move each pyfunction family verbatim into its own submodule; `udf/mod.rs` keeps the orchestration core (`execute_plan`/`execute_node`, the cache statics, `warmup`) and **re-exports every public item so all existing `udf::NAME` paths in `lib.rs` keep resolving unchanged**. No logic changes; the `cmp` and `build_agg_kind_and_vcol` folds and the groupby core/legacy split are deliberately deferred to plan B-2. This plan is move-only.

**Tech Stack:** Rust 2021, PyO3 0.22, `cargo`. The compiler + `make test-unit` are the oracle for a move refactor — imports and visibility are resolved against compiler errors, not guessed.

---

## Conventions for EVERY extraction task (read once, apply to all)

Each extraction task follows the same shape. Do NOT rewrite function bodies — **move them verbatim**. Locate items by their **function/type name** (line numbers below are from the current 3,000-line file and may drift as earlier tasks land — trust the name, use the line range as a hint).

1. **Move**: cut the named items out of `udf/mod.rs` and paste them verbatim into the new submodule file.
2. **Imports**: add a `use` header to the new file. Resolve it against the compiler — run `cargo build -p polars-metal-core`, add each `use` the errors demand (mirror the relevant lines from `udf/mod.rs`'s original import header, reproduced at the bottom of this plan). Cross-module references to sibling helpers use `use crate::udf::common::*;` etc. or `super::`.
3. **Visibility**:
   - Items that `lib.rs` re-exports or wraps as pyfunctions (`execute_plan`, `execute_filter_compact`, `cmp_i64_col_scalar`, `cmp_i64_col_col`, `cmp_f64_col_scalar`, `cmp_f64_col_col`, `bool_and_dispatch`, `bool_or_dispatch`, `execute_groupby`, `warmup_common_fused_signatures`, `execute_fused_expr`, `execute_rolling`, `execute_dt`, `execute_dtw`, `parse_groupby_plan`, `GroupByParseError`, `ParsedAgg`, `ParsedGroupByPlan`, `ParsedKey`) stay **`pub`**.
   - Items referenced only by sibling submodules become **`pub(crate)`**.
   - Items used only within their own new module stay private.
4. **Re-export**: in `udf/mod.rs`, add `pub use <submodule>::*;` (or explicit `pub use <submodule>::{NAMES};`) so every `udf::NAME` path in `lib.rs` continues to resolve **without editing `lib.rs`**.
   - **PyO3 caveat:** `wrap_pyfunction!(udf::NAME, m)` in `lib.rs` must still resolve `NAME` through the re-export. After moving a pyfunction, the FIRST build will tell you if it doesn't. If `wrap_pyfunction!` fails to resolve through the `pub use` re-export, the fallback is: make the submodule `pub(crate) mod <name>;`, keep the pyfunction `pub`, and change that one `wrap_pyfunction!(udf::NAME, m)` line in `lib.rs` to `wrap_pyfunction!(udf::<submodule>::NAME, m)`. Prefer the re-export (no lib.rs edit); use the fallback only if the compiler forces it. Document in the commit which approach was needed.
5. **Verify**: `cargo build -p polars-metal-core` clean, then `cargo test -p polars-metal-core -- --test-threads=1` green (the move changed nothing, so all tests must still pass), then `cargo fmt && cargo clippy -p polars-metal-core --all-targets -- -D warnings` clean.
6. **Commit** with message `M7 B-1: extract udf/<module>.rs (move-only)`.

**A test failure after a move means a botched move (wrong import, dropped item, visibility error) — fix the move, do NOT change a test.**

---

### Task 1: Scaffold the `udf/` module directory

**Files:**
- Rename: `crates/polars-metal-core/src/udf.rs` → `crates/polars-metal-core/src/udf/mod.rs`
- Create (empty): `crates/polars-metal-core/src/udf/{common,predicate,compare,compact,logical,fused_expr,rolling,dtw,dt,groupby}.rs`

- [ ] **Step 1: Move the file with git (preserves history)**

```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
mkdir -p crates/polars-metal-core/src/udf
git mv crates/polars-metal-core/src/udf.rs crates/polars-metal-core/src/udf/mod.rs
```

- [ ] **Step 2: Create the empty submodule files and declare them**

Create the 10 empty files:
```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal/crates/polars-metal-core/src/udf
for m in common predicate compare compact logical fused_expr rolling dtw dt groupby; do touch $m.rs; done
```

At the TOP of `udf/mod.rs` (right after the existing `//!` doc comment block, before the `use` lines), add the module declarations:
```rust
mod common;
mod compact;
mod compare;
mod dt;
mod dtw;
mod fused_expr;
mod groupby;
mod logical;
mod predicate;
mod rolling;
```
(An empty `.rs` file is a valid empty module — this compiles with all code still living in `mod.rs`.)

- [ ] **Step 3: Verify the scaffold builds (all code still in mod.rs)**

```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
cargo build -p polars-metal-core
```
Expected: clean build. Nothing moved yet — this only proves the directory rename + empty module decls are wired correctly.

- [ ] **Step 4: Commit**

```bash
git add -A crates/polars-metal-core/src/udf
git commit -m "M7 B-1: scaffold udf/ module directory (rename udf.rs -> udf/mod.rs)"
```

---

### Task 2: Extract `udf/common.rs` — shared low-level helpers

Extract first; other modules depend on these. Per the Conventions section.

**Move these items** (by name) from `udf/mod.rs` into `udf/common.rs`, all as **`pub(crate)`**:
- `new_device_and_queue` (~1048-1054) — returns `(MetalDevice, CommandQueue)`
- `check_numeric_buffers` (~1059-1084)
- `check_bitpacked_buffer` (~1088-1100)
- `pack_valid_bitmap` (~1770-1780)

- [ ] **Step 1: Move the four helpers into `udf/common.rs`, set `pub(crate)`, add imports**

Cut the four functions verbatim into `udf/common.rs`. Add the `use` header the compiler demands (these need `polars_metal_buffer::MetalDevice`, `polars_metal_kernels::command::CommandQueue`, and likely `pyo3` error types — mirror from the mod.rs import header at the bottom of this plan). Change each `fn` / `pub fn` to `pub(crate) fn`.

- [ ] **Step 2: Wire callers in `mod.rs` (and re-export not needed — these are internal)**

In `udf/mod.rs`, these helpers were called by code still in mod.rs and by code that will move later. Add `use common::{new_device_and_queue, check_numeric_buffers, check_bitpacked_buffer, pack_valid_bitmap};` near the top of mod.rs so existing callers in mod.rs resolve. (No `pub use` — these are not part of the public API.)

- [ ] **Step 3: Build + test + lint**

```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
cargo build -p polars-metal-core && cargo test -p polars-metal-core -- --test-threads=1
cargo fmt && cargo clippy -p polars-metal-core --all-targets -- -D warnings
```
Expected: clean build, all tests pass, lint clean.

- [ ] **Step 4: Commit**

```bash
git add -A crates/polars-metal-core/src/udf
git commit -m "M7 B-1: extract udf/common.rs (move-only)"
```

---

### Task 3: Extract `udf/predicate.rs` — plan/predicate deserialization

**Move these items** from `udf/mod.rs` into `udf/predicate.rs`:
- `deserialize_plan` (~609-685) — **`pub(crate)`** (called by execute_plan/execute_node in mod.rs)
- `deserialize_predicate` (~687-787) — `pub(crate)`
- `parse_dtype` (~789-795) — `pub(crate)`
- `parse_compare_op` (~800-812) — `pub(crate)`

- [ ] **Step 1: Move the four functions, set `pub(crate)`, add imports**

Cut verbatim into `udf/predicate.rs`. Imports needed include `crate::plan::{MetalPlanNode, PredicateAst, MetalDtype, ...}` and pyo3 dict/error types — resolve against the compiler.

- [ ] **Step 2: Wire callers in mod.rs**

Add `use predicate::{deserialize_plan, deserialize_predicate};` (and `parse_dtype`/`parse_compare_op` if mod.rs calls them) to `udf/mod.rs` so the orchestration code resolves.

- [ ] **Step 3: Build + test + lint**

```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
cargo build -p polars-metal-core && cargo test -p polars-metal-core -- --test-threads=1
cargo fmt && cargo clippy -p polars-metal-core --all-targets -- -D warnings
```
Expected: clean, all pass.

- [ ] **Step 4: Commit**

```bash
git add -A crates/polars-metal-core/src/udf
git commit -m "M7 B-1: extract udf/predicate.rs (move-only)"
```

---

### Task 4: Extract `udf/compare.rs` — comparison kernels (4 pyfunctions)

**Move these items** from `udf/mod.rs` into `udf/compare.rs`:
- `cmp_out_min_bytes` (~816-820) — `pub(crate)`
- `cmp_i64_col_scalar` (~835-879) — **`pub`** (pyfunction + lib.rs re-export)
- `cmp_i64_col_col` (~888-937) — **`pub`**
- `cmp_f64_col_scalar` (~946-989) — **`pub`**
- `cmp_f64_col_col` (~994-1042) — **`pub`**

Keep the `#[pyfunction]` attributes on the four pyfunctions.

- [ ] **Step 1: Move the five items, set visibility, add imports**

Cut verbatim into `udf/compare.rs`. Imports: `polars_metal_kernels::cmp::{dispatch_cmp_f64, dispatch_cmp_f64_scalar, dispatch_cmp_i64, dispatch_cmp_i64_scalar, CompareOp}`, `pyo3` (`PyBytes`, `prelude`), `crate::udf::common::{new_device_and_queue, check_numeric_buffers}`, `crate::lib`'s `engine_err`? (check — cmp uses `PyValueError` directly, likely no engine_err). Resolve against compiler.

- [ ] **Step 2: Re-export from mod.rs so `udf::cmp_*` paths resolve**

In `udf/mod.rs` add: `pub use compare::{cmp_f64_col_col, cmp_f64_col_scalar, cmp_i64_col_col, cmp_i64_col_scalar};`

- [ ] **Step 3: Build (watch the wrap_pyfunction! resolution) + test + lint**

```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
cargo build -p polars-metal-core
```
If `wrap_pyfunction!(udf::cmp_i64_col_scalar, m)` etc. in `lib.rs` fail to resolve through the re-export, apply the PyO3 fallback from the Conventions section (make `pub(crate) mod compare;` and point the four `wrap_pyfunction!` lines at `udf::compare::cmp_*`). Then:
```bash
cargo test -p polars-metal-core -- --test-threads=1
cargo fmt && cargo clippy -p polars-metal-core --all-targets -- -D warnings
```
Expected: clean, all pass. **Also build the wheel and run a Python smoke check** that the cmp pyfunctions are still registered (the filter path uses them):
```bash
make wheel && pytest tests/python_integration/test_filter_comparison.py -q
```
Expected: pass (proves the pyfunction registration survived the move).

- [ ] **Step 4: Commit**

```bash
git add -A crates/polars-metal-core/src/udf crates/polars-metal-core/src/lib.rs
git commit -m "M7 B-1: extract udf/compare.rs (move-only)"
```

---

### Task 5: Extract `udf/compact.rs` — filter compaction

**Move these items** from `udf/mod.rs` into `udf/compact.rs`:
- `execute_filter_compact` (~330-430) — **`pub`** (pyfunction + lib.rs re-export)
- `compact_one_column` (~444-599) — `pub(crate)` (or private if only called by execute_filter_compact)

- [ ] **Step 1: Move both, set visibility, add imports**

Cut verbatim. Imports: `polars_metal_kernels::pipeline::{compact_bool, compact_f64, compact_i64, compute_keep_and_prefix}`, `crate::udf::common::{check_numeric_buffers, check_bitpacked_buffer, pack_valid_bitmap}`, pyo3 types. Resolve against compiler.

- [ ] **Step 2: Re-export + wire**

In `udf/mod.rs`: `pub use compact::execute_filter_compact;`. If `execute_node` (still in mod.rs) calls into the filter path, add `use compact::...;` as the compiler demands.

- [ ] **Step 3: Build (wrap_pyfunction! for execute_filter_compact) + test + lint**

```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
cargo build -p polars-metal-core && cargo test -p polars-metal-core -- --test-threads=1
cargo fmt && cargo clippy -p polars-metal-core --all-targets -- -D warnings
```
Apply the PyO3 fallback for `udf::execute_filter_compact` if needed. Expected: clean, all pass.

- [ ] **Step 4: Commit**

```bash
git add -A crates/polars-metal-core/src/udf crates/polars-metal-core/src/lib.rs
git commit -m "M7 B-1: extract udf/compact.rs (move-only)"
```

---

### Task 6: Extract `udf/logical.rs` — boolean and/or dispatch

**Move these items** from `udf/mod.rs` into `udf/logical.rs`:
- `bool_and_dispatch` (~2409-2420) — **`pub`** (pyfunction + re-export)
- `bool_or_dispatch` (~2427-2438) — **`pub`**
- `dispatch_logical_py` (~2448-2506) — private (shared body, only called by the two above)

- [ ] **Step 1: Move all three, set visibility, add imports**

Cut verbatim. Imports: `polars_metal_kernels::logical::{dispatch_bool_and, dispatch_bool_or}`, `crate::udf::common::*` as needed, pyo3 types. Resolve against compiler.

- [ ] **Step 2: Re-export**

In `udf/mod.rs`: `pub use logical::{bool_and_dispatch, bool_or_dispatch};`

- [ ] **Step 3: Build + test + lint**

```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
cargo build -p polars-metal-core && cargo test -p polars-metal-core -- --test-threads=1
cargo fmt && cargo clippy -p polars-metal-core --all-targets -- -D warnings
```
Apply PyO3 fallback if needed. Expected: clean, all pass.

- [ ] **Step 4: Commit**

```bash
git add -A crates/polars-metal-core/src/udf crates/polars-metal-core/src/lib.rs
git commit -m "M7 B-1: extract udf/logical.rs (move-only)"
```

---

### Task 7: Extract the four live compute ops into separate files

Four isomorphic moves — each is one self-contained pyfunction. Do all four in this task, building between each so a failure is localized.

**Moves** (each pyfunction stays **`pub`**; each file gets its own imports resolved against the compiler):
- `execute_fused_expr` (~2543-2616) → `udf/fused_expr.rs`. Imports include `crate::fusion::py::...` and `polars_metal_buffer`.
- `execute_rolling` (~2653-2771) → `udf/rolling.rs`. F32 rolling kernel imports from `polars_metal_kernels`.
- `execute_dtw` (~2788-2862) → `udf/dtw.rs`. DTW kernel imports.
- `execute_dt` (~2881-3001) → `udf/dt.rs`. Gregorian-calendar kernel + `DT_STAGING` static — **NOTE:** `execute_dt` uses the `DT_STAGING` `OnceLock` static (declared in mod.rs ~195). Either move `DT_STAGING` into `dt.rs` (it is only used by `execute_dt` — preferred) or keep it in mod.rs as `pub(crate)` and `use` it. Move it into `dt.rs` if nothing else references it (grep first: `grep -rn DT_STAGING crates/polars-metal-core/src/`).

- [ ] **Step 1: Move `execute_fused_expr` → `udf/fused_expr.rs`; add `pub use fused_expr::execute_fused_expr;` to mod.rs; build**

```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
cargo build -p polars-metal-core
```
Apply PyO3 fallback for `udf::execute_fused_expr` if needed.

- [ ] **Step 2: Move `execute_rolling` → `udf/rolling.rs`; add `pub use rolling::execute_rolling;`; build**

```bash
cargo build -p polars-metal-core
```

- [ ] **Step 3: Move `execute_dtw` → `udf/dtw.rs`; add `pub use dtw::execute_dtw;`; build**

```bash
cargo build -p polars-metal-core
```

- [ ] **Step 4: Move `execute_dt` (+ `DT_STAGING` if exclusive) → `udf/dt.rs`; add `pub use dt::execute_dt;`; build**

```bash
grep -rn "DT_STAGING" crates/polars-metal-core/src/   # confirm execute_dt is the only user before moving the static
cargo build -p polars-metal-core
```

- [ ] **Step 5: Full test + lint + Python smoke for all four**

```bash
cargo test -p polars-metal-core -- --test-threads=1
cargo fmt && cargo clippy -p polars-metal-core --all-targets -- -D warnings
make wheel && pytest tests/python_integration/test_rolling_e2e.py tests/python_integration/test_dtw_e2e.py tests/python_integration/test_dt_e2e.py -q
```
Expected: clean, all pass (proves the four pyfunctions are still registered + behave).

- [ ] **Step 6: Commit**

```bash
git add -A crates/polars-metal-core/src/udf crates/polars-metal-core/src/lib.rs
git commit -m "M7 B-1: extract udf/{fused_expr,rolling,dtw,dt}.rs (move-only)"
```

---

### Task 8: Extract `udf/groupby.rs` — the whole groupby cluster

The largest move. Keep the entire groupby cluster as ONE module for now — the internal core/legacy split + the `build_agg_kind_and_vcol` fold are plan B-2's job. Move it intact.

**Move these items** from `udf/mod.rs` into `udf/groupby.rs`:
- Conversion helpers: `convert_agg_op` (~53-62), `convert_binary_op` (~65-72), `convert_agg_expr` (~75-86), `wire_dtype_tag_to_kernel` (~89-108) — `pub(crate)` or private as the compiler requires.
- Routing: `enum GroupByDispatchChoice` (~113-116), `decide_groupby_dispatch` (~130-185) — `pub(crate)`/private.
- Types: `ParsedGroupByPlan` (~1108-1111), `ParsedKey` (~1115-1118), `ParsedAgg` (~1123-1137) + `impl ParsedAgg` (~1139-1149), `GroupByParseError` (~1153-1162) — **`pub`** (lib.rs re-exports these).
- Parsing: `parse_agg_expr_dict` (~1177-1248), `parse_groupby_plan` (~1261-1400) (**`pub`**, re-exported), `metal_dtype_to_key_dtype` (~1410-1431).
- Value-column + agg builders: `build_value_column` (~1447-1485), `build_agg_kind_and_vcol` (~1487-1766).
- Encoding: `encode_decoded_column` (~1784-1867), `encode_agg_output` (~1874-1912), `groupby_err` (~1915-1917).
- Execution: `execute_groupby` (~1947-2393) — **`pub`** (pyfunction + re-export).

**Do NOT move** `pack_valid_bitmap` (already in common.rs from Task 2 — `use crate::udf::common::pack_valid_bitmap;`).

- [ ] **Step 1: Move the whole cluster verbatim into `udf/groupby.rs`; add imports**

Cut all the items above into `udf/groupby.rs`. This file needs a large import header — the kernel agg/groupby types (`polars_metal_kernels::groupby::*`, `polars_metal_kernels::aggregate_fused::{cache, signature}::*`), `crate::plan::{AggExpr, AggOp, BinaryOp, MetalDtype}`, `crate::udf::common::{new_device_and_queue, check_numeric_buffers, check_bitpacked_buffer, pack_valid_bitmap}`, pyo3 types, `std::collections::{BTreeMap, HashMap}`. Resolve against the compiler; mirror from the mod.rs import header (bottom of plan).

- [ ] **Step 2: Re-export the public groupby API from mod.rs**

In `udf/mod.rs`:
```rust
pub use groupby::{
    execute_groupby, parse_groupby_plan, GroupByParseError, ParsedAgg, ParsedGroupByPlan, ParsedKey,
};
```
Also add any `use groupby::...;` the orchestration code (`execute_node`) in mod.rs needs to call `execute_groupby`.

- [ ] **Step 3: Build (wrap_pyfunction! for execute_groupby; re-exports for the 5 types/fns) + test + lint**

```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
cargo build -p polars-metal-core && cargo test -p polars-metal-core -- --test-threads=1
cargo fmt && cargo clippy -p polars-metal-core --all-targets -- -D warnings
```
Apply PyO3 fallback for `udf::execute_groupby` if needed. **Critical re-export check:** `router_udf.rs` and other crate code reference `parse_groupby_plan`, `ParsedGroupByPlan`, `ParsedAgg`, etc. via the `lib.rs` `pub use udf::{...}` — confirm those still resolve (the build will fail loudly if not). Expected: clean, all pass.

- [ ] **Step 4: Python smoke for groupby (conformance)**

```bash
make wheel && pytest tests/python_integration/test_groupby.py tests/python_integration/test_groupby_small_int_keys.py -q
```
Expected: pass (groupby conformance survives the move).

- [ ] **Step 5: Commit**

```bash
git add -A crates/polars-metal-core/src/udf crates/polars-metal-core/src/lib.rs
git commit -m "M7 B-1: extract udf/groupby.rs (move-only)"
```

---

### Task 9: Trim `udf/mod.rs` and verify the whole decomposition

After Tasks 2-8, `udf/mod.rs` should hold only: the module `//!` doc, the `mod` declarations, the `pub use` re-exports, the `use` headers, the cache statics (`FUSED_CACHE`, `DT_STAGING` if not moved, `get_or_init_fused_cache`), `warmup_common_fused_signatures` (~216-229, **`pub`** pyfunction), `execute_plan` (~242-249, **`pub`** pyfunction), and `execute_node` (~251-298). Confirm nothing stale remains.

- [ ] **Step 1: Audit `udf/mod.rs` contents**

```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
grep -n "^fn \|^pub fn \|^pub(crate) fn \|^struct \|^enum \|^impl \|^static \|^macro_rules" crates/polars-metal-core/src/udf/mod.rs
wc -l crates/polars-metal-core/src/udf/*.rs
```
Expected: `mod.rs` now contains only the orchestration core (execute_plan, execute_node, warmup, statics) + module decls + re-exports — roughly 150-250 lines, down from 3,000. Any function that belongs to an extracted family but is still in mod.rs → move it to its module (and re-run that module's build/test). No `udf/*.rs` file should be empty (if one is, its extraction was missed).

- [ ] **Step 2: Confirm the public API surface is unchanged**

```bash
git show HEAD~7:crates/polars-metal-core/src/lib.rs > /tmp/lib_before.rs 2>/dev/null || git show <scaffold_commit>^:crates/polars-metal-core/src/lib.rs > /tmp/lib_before.rs
diff /tmp/lib_before.rs crates/polars-metal-core/src/lib.rs || echo "lib.rs changed — review the wrap_pyfunction! fallback edits"
```
The only acceptable `lib.rs` changes are PyO3-fallback `wrap_pyfunction!`/`pub use` path updates (Conventions §4). The `pub use udf::{...}` list and the set of registered pyfunctions must be IDENTICAL in effect. If `lib.rs` is unchanged, even better.

- [ ] **Step 3: Full gate**

```bash
cd /Users/dclark/dev/polars-metal/main/polars-metal
make gate
```
Expected: green end to end (this runs lint + test-unit + test-kernel + wheel + test-conformance + test-diff — the differential net from C1 confirms no behavior drift across the whole decomposition). SLOW (5-15 min) — be patient. A failure here means a move altered behavior; bisect to the offending extraction.

- [ ] **Step 4: Commit (if Step 1 moved any stragglers; otherwise no-op)**

```bash
git add -A crates/polars-metal-core/src
git commit -m "M7 B-1: trim udf/mod.rs to orchestration core; verify decomposition" || echo "nothing to trim — decomposition already clean"
```

---

## Appendix: original `udf.rs` import header (for resolving per-module `use`)

The pre-split `udf.rs` imported (each submodule needs the subset its moved code references):
```rust
use crate::plan::{AggExpr, AggOp, BinaryOp, MetalDtype, MetalPlanNode, PredicateAst};
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::aggregate_fused::cache::FusedLibraryCache;
use polars_metal_kernels::aggregate_fused::signature::{
    AggExpr as KAggExpr, AggOp as KAggOp, AggSpec as KAggSpec, BinaryOp as KBinaryOp,
    MetalDtype as KMetalDtype,
};
use polars_metal_kernels::cmp::{
    dispatch_cmp_f64, dispatch_cmp_f64_scalar, dispatch_cmp_i64, dispatch_cmp_i64_scalar, CompareOp,
};
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::groupby::{
    dispatch_groupby_fused, AggKind, AggRequest, GroupByError, KeyColumn, KeyDtype, ValueColumn,
};
use polars_metal_kernels::logical::{dispatch_bool_and, dispatch_bool_or};
use polars_metal_kernels::pipeline::{
    compact_bool, compact_f64, compact_i64, compute_keep_and_prefix,
};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, OnceLock};
```

## Self-Review

**Spec coverage (against design §3 Workstream B1):** "Split udf.rs into focused modules (parser/cmp/groupby_core/groupby_legacy)." This plan splits into more modules than the spec named because the real file has 9 pyfunction families, not just groupby — the live compute ops (fused_expr/rolling/dtw/dt), compact, logical, and predicate are peeled out too. The groupby **core/legacy** internal split is deferred to B-2 (it is not a pure move — `execute_groupby` has both paths inline — so it belongs with the `build_agg_kind_and_vcol` fold it is coupled to). Documented at the top of this plan. ✓

**Placeholder scan:** No TBD/TODO. The "resolve imports against the compiler" instruction is the correct specification for a move refactor (the compiler is a deterministic oracle), not a placeholder — the import header to draw from is reproduced in the Appendix.

**Consistency:** Every extracted pyfunction's visibility (`pub`) and re-export is tracked against the `lib.rs` `pub use` list + `wrap_pyfunction!` calls reproduced from the actual lib.rs. The PyO3 re-export caveat + fallback is stated once (Conventions §4) and referenced per task. `pack_valid_bitmap` is placed in common.rs (Task 2) and explicitly not re-moved in Task 8.

**Behavior preservation:** Every task ends with `cargo test -p polars-metal-core` green; pyfunction-bearing tasks add a Python smoke test; the final task runs the full `make gate` (including C1's differential net). Move-only — no logic changes.
