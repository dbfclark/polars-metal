# `.metal.corr()` GPU Correlation Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a GPU-accelerated Pearson correlation matrix to the `.metal` namespace as a LazyFrame verb, `lf.metal.corr()`, that computes `Zᵀ·Z` over L2-normalized centered columns in one MLX subgraph.

**Architecture:** A new `register_lazyframe_namespace("metal")` verb builds a struct sentinel (via `with_columns`) that the existing serialize-detect + collect-and-stitch machinery recognizes under `collect(engine="metal")`. Unlike the column-stitch verbs (dtw/fft/vector), corr **replaces the whole frame** with a `p×p` result. The kernel is an MLX op-graph (`mean → center → per-column L2-normalize → matmul(Zᵀ,Z)`), which makes the `(N−1)` normalization cancel — so it needs only the FFI wrappers vector search already uses. A `CORR_P_MIN=8` guard routes small-`p` and null/degenerate cases to a Polars CPU fallback (cast to F32 so output dtype is path-independent).

**Tech Stack:** Rust (PyO3 + MLX FFI via `polars-metal-mlx-sys`), Python (Polars LazyFrame namespace + serialize-detect), MLX op-graph (no custom MSL).

**Spec:** `docs/superpowers/specs/2026-06-11-m6-corr-matrix-design.md`

---

## File Structure

- `crates/polars-metal-core/src/corr.rs` — **new.** MLX corr kernel (`corr_matrix`) + `execute_corr` pyfunction. Mirrors `vector_search.rs`.
- `crates/polars-metal-core/src/lib.rs` — **modify.** `mod corr;` + register `execute_corr`.
- `python/polars_metal/_corr_namespace.py` — **new.** `register_lazyframe_namespace("metal")` → `MetalLazyNamespace.corr()`, sentinel builder, capture cache.
- `python/polars_metal/_corr_detect.py` — **new.** `find_corr_bindings` (with_columns cache + serialize fallback). Mirrors `_dtw_detect.py`.
- `python/polars_metal/_corr_dispatch.py` — **new.** `apply_corr` (collect rest, build matrix, route GPU/CPU, return p×p F32). Mirrors `_dtw_dispatch.py` but frame-replacing.
- `python/polars_metal/__init__.py` — **modify.** Import `_corr_namespace` for registration; add corr branch to `collect_wrapper`.
- `tests/kernel/test_corr_kernel.py` — **new.** Differential `execute_corr` vs numpy.
- `tests/python_integration/test_corr_engine.py` — **new.** End-to-end `.metal.corr()` vs `df.corr()`, routing, null/dtype.
- `tests/bench/bench_corr.py` — **new.** Perf sweep + regression gate.

---

## Task 1: Rust MLX corr kernel + `execute_corr` pyfunction

**Files:**
- Create: `crates/polars-metal-core/src/corr.rs`
- Modify: `crates/polars-metal-core/src/lib.rs`
- Test: inline `#[cfg(test)]` in `corr.rs`

**Math (cosine-of-centered-columns identity):** `corr[i,j] = sum(Xc_i·Xc_j) / (‖Xc_i‖·‖Xc_j‖)` where `Xc = X − colmean(X)`. Equivalent to `C = Znᵀ·Zn` with `Zn` = columns of `Xc` each divided by their L2 norm. The `(N−1)` in covariance and in the two std factors cancels, so **no scalar division is needed** — only `mlx_mean_axis`, `mlx_sub`, `mlx_mul`, `mlx_sum_axis`, `mlx_sqrt`, `mlx_div`, `mlx_reshape`, `mlx_transpose`, `mlx_matmul`.

- [ ] **Step 1: Write the failing Rust test**

Add to the bottom of `crates/polars-metal-core/src/corr.rs` (file created in Step 3; write the test first conceptually, but since the file doesn't exist yet, create it now containing ONLY the test + a stub):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corr_2x2_known() {
        // X (N=4, p=2), row-major: cols [1,2,3,4] and [2,1,4,3].
        // centered c0=[-1.5,-0.5,0.5,1.5], c1=[-0.5,-1.5,1.5,0.5];
        // dot=3.0, ||c0||=||c1||=sqrt(5); corr = 3/5 = 0.6.
        let data: Vec<f32> = vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0];
        let c = corr_matrix(&data, 4, 2).unwrap();
        assert_eq!(c.len(), 4);
        assert!((c[0] - 1.0).abs() < 1e-5, "C[0,0]={}", c[0]);
        assert!((c[1] - 0.6).abs() < 1e-5, "C[0,1]={}", c[1]);
        assert!((c[2] - 0.6).abs() < 1e-5, "C[1,0]={}", c[2]);
        assert!((c[3] - 1.0).abs() < 1e-5, "C[1,1]={}", c[3]);
    }
}
```

And a stub above it so the crate compiles:

```rust
//! M6 corr: GPU Pearson correlation matrix via one MLX subgraph.
//! C = Znᵀ·Zn where Zn = per-column-L2-normalized centered columns (the
//! (N−1) normalization cancels). Mirrors `vector_search.rs` for FFI idioms.

use polars_metal_mlx_sys::array::{
    mlx_array_eval, mlx_array_to_f32_vec, mlx_array_view_metal_buffer, MlxArrayHandle, MlxDtype,
};
use polars_metal_mlx_sys::elementwise::{mlx_div, mlx_mul, mlx_sqrt, mlx_sub};
use polars_metal_mlx_sys::matmul::mlx_matmul;
use polars_metal_mlx_sys::reduce::{mlx_mean_axis, mlx_sum_axis};
use polars_metal_mlx_sys::shape::{mlx_reshape, mlx_transpose};
use polars_metal_mlx_sys::FfiError;

pub fn corr_matrix(_data: &[f32], _n: i64, _p: i64) -> Result<Vec<f32>, FfiError> {
    unimplemented!()
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p polars-metal-core corr_2x2_known -- --test-threads=1`
Expected: FAIL (panics on `unimplemented!()`).

- [ ] **Step 3: Implement `corr_matrix` + view helper**

Replace the `corr_matrix` stub in `crates/polars-metal-core/src/corr.rs`:

```rust
/// View a row-major (n, p) F32 slice as an MLX array. Borrows; the caller keeps
/// `data` alive until after `mlx_array_eval`. Mirrors vector_search::view2d.
fn view2d(data: &[f32], rows: i64, cols: i64) -> Result<MlxArrayHandle, FfiError> {
    use polars_metal_buffer::MetalBuffer;
    use polars_metal_kernels::device::MetalDevice;
    let device = MetalDevice::system_default()
        .map_err(|e| FfiError::Other(format!("metal device: {e}")))?;
    // SAFETY: data is contiguous, rows*cols f32, alive for this call.
    let buf = unsafe {
        MetalBuffer::from_borrowed_f32(&device, data.as_ptr(), (rows * cols) as usize)
    }
    .map_err(|e| FfiError::Other(format!("corr staging: {e}")))?;
    mlx_array_view_metal_buffer(&buf, &[rows, cols], MlxDtype::F32)
}

/// Pearson correlation matrix of a row-major (n, p) F32 matrix → p*p row-major F32.
pub fn corr_matrix(data: &[f32], n: i64, p: i64) -> Result<Vec<f32>, FfiError> {
    let x = view2d(data, n, p)?; // (N,p)
    let mean = mlx_reshape(&mlx_mean_axis(&x, 0)?, &[1, p as i32])?; // (1,p)
    let xc = mlx_sub(&x, &mean)?; // (N,p) centered columns
    let colss = mlx_reshape(&mlx_sum_axis(&mlx_mul(&xc, &xc)?, 0)?, &[1, p as i32])?; // (1,p)
    let norm = mlx_sqrt(&colss)?; // (1,p) column L2 norms
    let zn = mlx_div(&xc, &norm)?; // (N,p) unit-norm columns
    let zt = mlx_transpose(&zn, &[1, 0])?; // (p,N)
    let c = mlx_matmul(&zt, &zn)?; // (p,p)
    mlx_array_eval(&[c.clone()])?;
    mlx_array_to_f32_vec(&c)
}
```

Note: if `FfiError` has no `Other(String)` variant, use the same error-construction idiom you find in `vector_search.rs` (grep `FfiError::` there) — match the existing pattern rather than inventing one.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p polars-metal-core corr_2x2_known -- --test-threads=1`
Expected: PASS.

- [ ] **Step 5: Add the `execute_corr` pyfunction**

Append to `crates/polars-metal-core/src/corr.rs` (above the test module):

```rust
use pyo3::prelude::*;

/// PyO3 entry: (ptr,len) row-major (n,p) F32 → flat p*p F32 correlation matrix.
/// Mirrors vector_search::execute_vector_search's (ptr,len) ABI.
#[pyfunction]
pub fn execute_corr(data: (usize, usize), n: i64, p: i64) -> PyResult<Vec<f32>> {
    let (ptr, len) = data;
    if n < 0 || p < 0 || (n as usize).saturating_mul(p as usize) != len {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "polars_metal: corr dimension mismatch (n*p != len)",
        ));
    }
    // SAFETY: Python guarantees ptr addresses `len` contiguous live F32 (numpy
    // array kept alive across the call); read-only, no invalid f32 patterns.
    let slice = unsafe { std::slice::from_raw_parts(ptr as *const f32, len) };
    corr_matrix(slice, n, p)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("corr: {e}")))
}
```

- [ ] **Step 6: Register in `lib.rs`**

In `crates/polars-metal-core/src/lib.rs`, add `mod corr;` near the other `mod` lines (after `mod vector_search;` or alphabetically with the others), and inside the `#[pymodule]` body add after the `fft::execute_fft` line:

```rust
    m.add_function(wrap_pyfunction!(corr::execute_corr, m)?)?;
```

- [ ] **Step 7: Build the wheel and verify the binding exists**

Run: `make wheel`
Then: `python -c "from polars_metal import _native; print(_native.execute_corr)"`
Expected: prints a builtin function (no AttributeError).

- [ ] **Step 8: Run cargo fmt + clippy on the crate**

Run: `cargo fmt -p polars-metal-core && cargo clippy -p polars-metal-core -- -D warnings`
Expected: no diffs, no warnings.

- [ ] **Step 9: Commit**

```bash
git add crates/polars-metal-core/src/corr.rs crates/polars-metal-core/src/lib.rs
git commit -m "M6 corr T1: MLX corr_matrix kernel + execute_corr pyfunction"
```

---

## Task 2: LazyFrame `.metal.corr()` namespace + capture cache

**Files:**
- Create: `python/polars_metal/_corr_namespace.py`
- Test: `tests/python_integration/test_corr_engine.py` (created here; grows in later tasks)

- [ ] **Step 1: Write the failing test**

Create `tests/python_integration/test_corr_engine.py`:

```python
import numpy as np
import polars as pl
import pytest

import polars_metal  # noqa: F401  (registers namespace + patches collect)


def _frame(n=2000, p=10, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, p)).astype(np.float32)
    return pl.DataFrame(x, schema=[f"c{i}" for i in range(p)])


def test_corr_sentinel_raises_on_plain_cpu():
    # .metal.corr() builds a sentinel lf; collected WITHOUT engine="metal" it must raise.
    lf = _frame().lazy().metal.corr()
    with pytest.raises(Exception):
        lf.collect()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/python_integration/test_corr_engine.py::test_corr_sentinel_raises_on_plain_cpu -v`
Expected: FAIL with `AttributeError` (no `.metal` LazyFrame namespace yet).

- [ ] **Step 3: Implement the namespace**

Create `python/polars_metal/_corr_namespace.py`:

```python
"""M6 corr: lf.metal.corr() LazyFrame verb — sentinel builder + capture cache.

Registers a LazyFrame `.metal` namespace (separate registry from the Expr-level
`.metal` namespace in _vector_namespace.py). corr() adds a struct sentinel via
with_columns (keeping the input columns so dispatch can read them) carrying a
tagged Int64 handle + a CPU-raising map_batches field. The column list and
force_gpu flag live in a module cache keyed by the handle, popped at dispatch.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import polars as pl

_HANDLE_COUNTER = itertools.count(1)

CORR_SENTINEL_TAG = "__pm_corr__"
CORR_SENTINEL_COL = "__pm_corr_sentinel"


@dataclass(frozen=True)
class CorrSpec:
    columns: tuple[str, ...]
    force_gpu: bool


_CORR_CACHE: dict[int, CorrSpec] = {}


def _capture(columns: tuple[str, ...], force_gpu: bool) -> int:
    handle = next(_HANDLE_COUNTER)
    _CORR_CACHE[handle] = CorrSpec(columns, force_gpu)
    return handle


def pop_capture(handle: int) -> CorrSpec | None:
    return _CORR_CACHE.pop(handle, None)


def _raise_cpu(_s: pl.Series) -> pl.Series:
    raise RuntimeError(
        "polars_metal: .metal.corr() requires collect(engine='metal'); "
        "it has no plain-CPU implementation. Use df.corr() for CPU."
    )


def build_corr_sentinel(any_col: str, handle: int) -> pl.Expr:
    """Struct sentinel: tagged Int64 handle + CPU-raising field. Added (not
    selected) so the input columns survive for dispatch; dropped before the
    rest-collect under engine='metal'."""
    return pl.struct(
        [
            pl.lit(handle, dtype=pl.Int64).alias(CORR_SENTINEL_TAG),
            pl.col(any_col)
            .map_batches(_raise_cpu, return_dtype=pl.Float32)
            .alias("__pm_corr_raise"),
        ]
    ).alias(CORR_SENTINEL_COL)


@pl.api.register_lazyframe_namespace("metal")
class MetalLazyNamespace:
    def __init__(self, lf: pl.LazyFrame) -> None:
        self._lf = lf

    def corr(self, force_gpu: bool = False) -> pl.LazyFrame:
        """Pearson correlation matrix of ALL columns of this frame, on the GPU.

        Returns a sentinel-bearing LazyFrame; collect(engine='metal') replaces
        it with the p×p Float32 correlation matrix. Narrow columns upstream with
        .select(...). Float32 output (documented divergence from df.corr()'s F64).
        """
        cols = tuple(self._lf.collect_schema().names())
        if len(cols) == 0:
            raise ValueError("polars_metal: .metal.corr() requires at least one column.")
        handle = _capture(cols, bool(force_gpu))
        return self._lf.with_columns(build_corr_sentinel(cols[0], handle))
```

- [ ] **Step 4: Register the namespace at import**

In `python/polars_metal/__init__.py`, find the existing namespace import (grep for `_vector_namespace`) and add alongside it:

```python
from polars_metal import _corr_namespace  # noqa: F401  (registers lf.metal.corr)
```

Place it next to the other `_*_namespace` / detect imports so registration fires on `import polars_metal`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/python_integration/test_corr_engine.py::test_corr_sentinel_raises_on_plain_cpu -v`
Expected: PASS (collect() raises the RuntimeError from `_raise_cpu`).

- [ ] **Step 6: Commit**

```bash
git add python/polars_metal/_corr_namespace.py python/polars_metal/__init__.py tests/python_integration/test_corr_engine.py
git commit -m "M6 corr T2: lf.metal.corr() namespace + CPU-raising sentinel"
```

---

## Task 3: Serialize-detect for the corr sentinel

**Files:**
- Create: `python/polars_metal/_corr_detect.py`
- Test: `tests/python_integration/test_corr_engine.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/python_integration/test_corr_engine.py`:

```python
def test_corr_detect_finds_binding():
    from polars_metal import _corr_detect

    lf = _frame(p=4).lazy().metal.corr()
    bindings = _corr_detect.find_corr_bindings(lf)
    assert len(bindings) == 1
    assert bindings[0].out_name  # the sentinel column name
    assert isinstance(bindings[0].handle, int)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/python_integration/test_corr_engine.py::test_corr_detect_finds_binding -v`
Expected: FAIL with `ModuleNotFoundError: polars_metal._corr_detect`.

- [ ] **Step 3: Implement the detector**

Create `python/polars_metal/_corr_detect.py` (mirrors `_dtw_detect.py`; its OWN patch attr + cache so it chains with rolling/vector/fft/dtw/dt):

```python
"""M6 corr: detect the corr sentinel from the outermost with_columns layer.

Same strategy as _dtw_detect: a fast with_columns-capture cache (keyed by
id(result)) plus a bounded serialize() fallback. OWN patch attr + cache so it
coexists with the other .metal detectors.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import polars as pl
import polars.lazyframe.frame as _plf

from polars_metal._corr_namespace import CORR_SENTINEL_TAG

_corr_lf_exprs_cache: dict[int, list[pl.Expr]] = {}
_PATCH_ATTR = "_polars_metal_corr_original_with_columns"

if not hasattr(_plf.LazyFrame, _PATCH_ATTR):
    _orig_wc = _plf.LazyFrame.with_columns
    setattr(_plf.LazyFrame, _PATCH_ATTR, _orig_wc)

    def _patched_wc(self, *exprs, **named):  # type: ignore[no-untyped-def]
        result = _orig_wc(self, *exprs, **named)
        try:
            flat: list[pl.Expr] = [e for e in exprs if isinstance(e, pl.Expr)]
            flat += [e.alias(n) for n, e in named.items() if isinstance(e, pl.Expr)]
            if flat:
                _corr_lf_exprs_cache[id(result)] = flat
        except Exception:
            pass
        return result

    _plf.LazyFrame.with_columns = _patched_wc  # type: ignore[method-assign]


@dataclass(frozen=True)
class CorrBinding:
    out_name: str
    handle: int


def _alias_name(node) -> str | None:
    if isinstance(node, dict):
        a = node.get("Alias")
        if isinstance(a, list) and len(a) == 2 and isinstance(a[1], str):
            return a[1]
    return None


def _struct_fields(expr_json: dict) -> list:
    fn = expr_json.get("Function")
    if isinstance(fn, dict):
        inp = fn.get("input")
        if isinstance(inp, list):
            return inp
    return []


def _literal_int(node) -> int | None:
    if isinstance(node, dict):
        a = node.get("Alias")
        if isinstance(a, list) and len(a) == 2 and isinstance(a[0], dict):
            lit = a[0].get("Literal")
            if isinstance(lit, dict):
                scalar = lit.get("Scalar")
                if isinstance(scalar, dict):
                    for key in ("Int64", "Int32", "Int"):
                        v = scalar.get(key)
                        if isinstance(v, int):
                            return v
                for key in ("Int64", "Int32", "Int"):
                    v = lit.get(key)
                    if isinstance(v, int):
                        return v
            if isinstance(lit, int):
                return lit
    return None


def _binding_from_expr_json(expr_json: dict, out_name: str) -> CorrBinding | None:
    try:
        s = json.dumps(expr_json)
        if CORR_SENTINEL_TAG not in s:
            return None
        handle = None
        for fld in _struct_fields(expr_json):
            if _alias_name(fld) == CORR_SENTINEL_TAG:
                handle = _literal_int(fld)
        if handle is None:
            return None
        return CorrBinding(out_name=out_name, handle=handle)
    except Exception:
        return None


def find_corr_bindings(lf: pl.LazyFrame) -> list[CorrBinding]:
    try:
        cached = _corr_lf_exprs_cache.pop(id(lf), None)
        if cached is not None:
            out: list[CorrBinding] = []
            for expr in cached:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    j = json.loads(expr.meta.serialize(format="json"))
                name = _alias_name(j)
                inner = j["Alias"][0] if name else j
                b = _binding_from_expr_json(inner, name or "")
                if b is not None and b.out_name:
                    out.append(b)
            return out

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            if CORR_SENTINEL_TAG not in lf.explain():
                return []
            plan = lf.serialize(format="json")
        key = '"exprs":['
        i = plan.rfind(key)
        if i == -1:
            return []
        start = i + len(key) - 1
        j = plan.rfind(',"options":', start)
        frag = plan[start:j] if j != -1 else plan[start:]
        nodes = json.loads(frag)
        out = []
        for node in nodes if isinstance(nodes, list) else []:
            name = _alias_name(node)
            inner = node["Alias"][0] if name else node
            b = _binding_from_expr_json(inner, name or "")
            if b is not None and b.out_name:
                out.append(b)
        return out
    except Exception:
        return []
```

- [ ] **Step 4: Import the detector at package init**

In `python/polars_metal/__init__.py`, next to the `_corr_namespace` import from Task 2, add:

```python
from polars_metal import _corr_detect  # noqa: F401  (installs with_columns patch)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/python_integration/test_corr_engine.py::test_corr_detect_finds_binding -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/polars_metal/_corr_detect.py python/polars_metal/__init__.py tests/python_integration/test_corr_engine.py
git commit -m "M6 corr T3: serialize-detect for corr sentinel (own patch+cache)"
```

---

## Task 4: Dispatch — build matrix, route, return p×p (frame-replacing)

**Files:**
- Create: `python/polars_metal/_corr_dispatch.py`
- Test: `tests/python_integration/test_corr_engine.py`

This task builds `apply_corr` and tests it directly (the `collect_wrapper` wiring is Task 5). Note: `apply_corr` is the one genuinely new dispatch shape — it **returns a fresh p×p DataFrame**, not the N-row frame.

- [ ] **Step 1: Write the failing test (direct dispatch, GPU path)**

Add to `tests/python_integration/test_corr_engine.py`:

```python
def test_apply_corr_matches_numpy_gpu_path():
    from polars_metal import _corr_detect, _corr_dispatch

    df = _frame(n=4000, p=12, seed=1)
    lf = df.lazy().metal.corr()  # p=12 >= CORR_P_MIN → GPU
    bindings = _corr_detect.find_corr_bindings(lf)

    def _collect_rest(rest_lf):
        return rest_lf.collect()

    out = _corr_dispatch.apply_corr(lf, bindings[0], _collect_rest)
    assert out.shape == (12, 12)
    assert all(dt == pl.Float32 for dt in out.dtypes)
    assert out.columns == [f"c{i}" for i in range(12)]
    expected = np.corrcoef(df.to_numpy().T).astype(np.float32)
    np.testing.assert_allclose(out.to_numpy(), expected, atol=1e-4)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/python_integration/test_corr_engine.py::test_apply_corr_matches_numpy_gpu_path -v`
Expected: FAIL with `ModuleNotFoundError: polars_metal._corr_dispatch`.

- [ ] **Step 3: Implement the dispatcher**

Create `python/polars_metal/_corr_dispatch.py`:

```python
"""M6 corr: execute a detected corr binding → p×p Float32 correlation matrix.

Frame-replacing (unlike dtw/fft/vector which stitch a column into the N-row
frame): collect the input columns, then return a fresh p×p frame. Routing:
  - non-numeric column            → raise
  - any null, or N < 2            → CPU fallback (df.corr() cast F32)
  - p < CORR_P_MIN and not force  → CPU fallback (cast F32)
  - else                          → GPU (execute_corr), result cast F32
F32 output regardless of path (documented divergence from df.corr()'s F64).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from polars_metal import _native
from polars_metal._corr_detect import CorrBinding
from polars_metal._corr_namespace import CorrSpec, pop_capture

CORR_P_MIN = 8  # spike crossover: p>=~8 GPU wins; below, CPU df.corr() is faster.


def _cpu_corr_f32(df: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    """Polars CPU correlation, cast to Float32 (the oracle + the small-p path)."""
    return df.select(columns).corr().cast(pl.Float32)


def _gpu_corr_f32(df: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    n = df.height
    p = len(columns)
    f32 = df.select([pl.col(c).cast(pl.Float32) for c in columns])
    mat = np.ascontiguousarray(f32.to_numpy(), dtype=np.float32)  # (n, p) row-major
    flat = mat.reshape(-1)
    out = _native.execute_corr((flat.ctypes.data, int(flat.size)), int(n), int(p))
    cmat = np.asarray(out, dtype=np.float32).reshape(p, p)
    return pl.DataFrame(
        {columns[j]: pl.Series(columns[j], cmat[:, j], dtype=pl.Float32) for j in range(p)}
    )


def _run_corr(df: pl.DataFrame, spec: CorrSpec) -> pl.DataFrame:
    columns = spec.columns
    # dtype gate: every selected column must be numeric.
    for c in columns:
        if not df.get_column(c).dtype.is_numeric():
            raise ValueError(
                f"polars_metal: .metal.corr() requires numeric columns; "
                f"column {c!r} is {df.get_column(c).dtype}."
            )
    has_null = any(df.get_column(c).null_count() > 0 for c in columns)
    p = len(columns)
    if has_null or df.height < 2:
        return _cpu_corr_f32(df, columns)
    if p < CORR_P_MIN and not spec.force_gpu:
        return _cpu_corr_f32(df, columns)
    return _gpu_corr_f32(df, columns)


def apply_corr(lf: pl.LazyFrame, binding: CorrBinding, collect_fn) -> pl.DataFrame:
    spec: CorrSpec | None = pop_capture(binding.handle)
    if spec is None:
        raise RuntimeError("polars_metal: corr spec handle missing (already consumed?)")
    rest_lf = lf.drop(binding.out_name)  # drop sentinel; input columns remain
    df = collect_fn(rest_lf)
    return _run_corr(df, spec)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/python_integration/test_corr_engine.py::test_apply_corr_matches_numpy_gpu_path -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/polars_metal/_corr_dispatch.py tests/python_integration/test_corr_engine.py
git commit -m "M6 corr T4: frame-replacing dispatch + GPU/CPU routing"
```

---

## Task 5: Wire corr into `collect_wrapper` + end-to-end differential

**Files:**
- Modify: `python/polars_metal/__init__.py:316-327` (after the dt block, before the final fallthrough)
- Test: `tests/python_integration/test_corr_engine.py`

- [ ] **Step 1: Write the failing end-to-end test**

Add to `tests/python_integration/test_corr_engine.py`:

```python
import polars_metal as pm  # noqa: E402


def test_metal_corr_end_to_end():
    df = _frame(n=5000, p=16, seed=2)
    out = df.lazy().metal.corr().collect(engine=pm.MetalEngine())
    assert out.shape == (16, 16)
    assert all(dt == pl.Float32 for dt in out.dtypes)
    expected = df.corr().cast(pl.Float32)
    np.testing.assert_allclose(out.to_numpy(), expected.to_numpy(), atol=1e-4)
    assert out.columns == expected.columns
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/python_integration/test_corr_engine.py::test_metal_corr_end_to_end -v`
Expected: FAIL — without wiring, the collect hits `_raise_cpu` (RuntimeError) because corr isn't detected in `collect_wrapper`.

- [ ] **Step 3: Add the corr branch to `collect_wrapper`**

In `python/polars_metal/__init__.py`, immediately after the dt block (the `return _dt_dispatch.apply_dt(...)` line, around line 327) and before the `# post_opt_callback is an internal bypass` comment, insert:

```python
            # M6 corr: serialize-detected lf.metal.corr() runs the MLX corr
            # subgraph and REPLACES the frame with the p×p matrix (frame-
            # replacing, unlike the column-stitch verbs above). Own patch/cache.
            from polars_metal import _corr_detect, _corr_dispatch

            corr_bindings = [] if streaming else _corr_detect.find_corr_bindings(self)
            if corr_bindings:

                def _collect_rest_corr(rest_lf: Any) -> Any:
                    return original_collect(rest_lf, engine="cpu", post_opt_callback=cb, **kwargs)

                return _corr_dispatch.apply_corr(self, corr_bindings[0], _collect_rest_corr)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/python_integration/test_corr_engine.py::test_metal_corr_end_to_end -v`
Expected: PASS.

- [ ] **Step 5: Verify no cross-talk with other verbs**

Run: `python -m pytest tests/python_integration/test_corr_engine.py -v`
Expected: all pass (corr's sentinel tag is distinct; other detectors return `[]` for it).

- [ ] **Step 6: Commit**

```bash
git add python/polars_metal/__init__.py tests/python_integration/test_corr_engine.py
git commit -m "M6 corr T5: wire corr into collect_wrapper + e2e differential"
```

---

## Task 6: Routing guard + `force_gpu` override

**Files:**
- Test: `tests/python_integration/test_corr_engine.py`

Validates the `CORR_P_MIN=8` guard: small-p stays CPU (still correct), `force_gpu=True` overrides. Both paths must return F32 and match the oracle.

- [ ] **Step 1: Write the failing tests**

Add to `tests/python_integration/test_corr_engine.py`:

```python
def test_corr_small_p_cpu_fallback_correct():
    # p=3 < CORR_P_MIN → CPU fallback path; result still correct + F32.
    df = _frame(n=3000, p=3, seed=3)
    out = df.lazy().metal.corr().collect(engine=pm.MetalEngine())
    assert out.shape == (3, 3)
    assert all(dt == pl.Float32 for dt in out.dtypes)
    expected = df.corr().cast(pl.Float32)
    np.testing.assert_allclose(out.to_numpy(), expected.to_numpy(), atol=1e-4)


def test_corr_force_gpu_small_p_matches():
    # force_gpu=True drives p=3 through the GPU path; must still match oracle.
    df = _frame(n=3000, p=3, seed=4)
    out = df.lazy().metal.corr(force_gpu=True).collect(engine=pm.MetalEngine())
    expected = df.corr().cast(pl.Float32)
    np.testing.assert_allclose(out.to_numpy(), expected.to_numpy(), atol=1e-4)


def test_corr_p_min_constant_is_eight():
    from polars_metal._corr_dispatch import CORR_P_MIN

    assert CORR_P_MIN == 8
```

- [ ] **Step 2: Run the tests to verify they pass**

Run: `python -m pytest tests/python_integration/test_corr_engine.py -k "small_p or force_gpu or p_min" -v`
Expected: PASS (the dispatch logic from Task 4 already implements these; these tests lock the behavior end-to-end). If `test_corr_force_gpu_small_p_matches` fails on precision, that is a real kernel bug — debug, do not loosen the tolerance.

- [ ] **Step 3: Commit**

```bash
git add tests/python_integration/test_corr_engine.py
git commit -m "M6 corr T6: routing guard + force_gpu override tests"
```

---

## Task 7: Null handling, dtype coercion, degenerate shapes

**Files:**
- Test: `tests/python_integration/test_corr_engine.py`
- Test: `tests/kernel/test_corr_kernel.py`

- [ ] **Step 1: Write the failing engine tests**

Add to `tests/python_integration/test_corr_engine.py`:

```python
def test_corr_nulls_route_to_cpu_and_match():
    # Null-bearing input → CPU fallback (exact Polars null semantics), F32 out.
    df = _frame(n=2000, p=10, seed=5).with_columns(
        pl.when(pl.int_range(pl.len()) % 7 == 0)
        .then(None)
        .otherwise(pl.col("c0"))
        .alias("c0")
    )
    out = df.lazy().metal.corr().collect(engine=pm.MetalEngine())
    expected = df.corr().cast(pl.Float32)
    assert all(dt == pl.Float32 for dt in out.dtypes)
    # Compare with NaN-equal semantics (constant/degenerate cells may be NaN).
    np.testing.assert_allclose(
        out.to_numpy(), expected.to_numpy(), atol=1e-4, equal_nan=True
    )


def test_corr_integer_and_f64_inputs_cast():
    # Int + F64 numeric columns are accepted (cast to F32). p>=8 → GPU path.
    rng = np.random.default_rng(6)
    df = pl.DataFrame(
        {f"c{i}": rng.integers(-100, 100, size=4000) for i in range(10)}
    )  # Int64 columns
    out = df.lazy().metal.corr().collect(engine=pm.MetalEngine())
    expected = df.corr().cast(pl.Float32)
    np.testing.assert_allclose(out.to_numpy(), expected.to_numpy(), atol=1e-3)


def test_corr_non_numeric_raises():
    df = _frame(n=100, p=9).with_columns(pl.lit("x").alias("c0"))
    with pytest.raises(Exception):
        df.lazy().metal.corr().collect(engine=pm.MetalEngine())
```

- [ ] **Step 2: Run the engine tests to verify they pass**

Run: `python -m pytest tests/python_integration/test_corr_engine.py -k "nulls or integer or non_numeric" -v`
Expected: PASS. (If `test_corr_nulls_route_to_cpu_and_match` fails because Polars `df.corr()` differs in null handling from the no-null case, inspect what `df.corr()` actually returns on that input — Polars CPU is the spec — and only then adjust the assertion to match it.)

- [ ] **Step 3: Write the kernel-level edge tests**

Create `tests/kernel/test_corr_kernel.py`:

```python
import numpy as np
import pytest

from polars_metal import _native


def _gpu_corr(x: np.ndarray) -> np.ndarray:
    n, p = x.shape
    flat = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    out = _native.execute_corr((flat.ctypes.data, int(flat.size)), int(n), int(p))
    return np.asarray(out, dtype=np.float32).reshape(p, p)


def test_corr_kernel_vs_numpy_wide():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((10000, 50)).astype(np.float32)
    got = _gpu_corr(x)
    expected = np.corrcoef(x.T).astype(np.float32)
    np.testing.assert_allclose(got, expected, atol=1e-4)


def test_corr_kernel_p1_is_one():
    rng = np.random.default_rng(8)
    x = rng.standard_normal((500, 1)).astype(np.float32)
    got = _gpu_corr(x)
    assert got.shape == (1, 1)
    assert abs(got[0, 0] - 1.0) < 1e-4


def test_corr_kernel_constant_column_is_nan():
    # A zero-variance column → division by zero → NaN, matching df.corr().
    x = np.ones((500, 2), dtype=np.float32)
    x[:, 1] = np.linspace(0, 1, 500, dtype=np.float32)  # col1 varies, col0 constant
    got = _gpu_corr(x)
    assert np.isnan(got[0, 0]) and np.isnan(got[0, 1]) and np.isnan(got[1, 0])
```

- [ ] **Step 4: Run the kernel tests to verify they pass**

Run: `python -m pytest tests/kernel/test_corr_kernel.py -v`
Expected: PASS. If `test_corr_kernel_constant_column_is_nan` fails (e.g. MLX yields 0 or inf instead of NaN), verify what Polars `df.corr()` returns for a constant column and align the engine (the constant-column case routes through `_gpu_corr_f32` only when no nulls and N≥2; if Polars yields NaN and the kernel yields inf, that is a real divergence to reconcile — the design says NaN; debug the kernel, do not weaken the test).

- [ ] **Step 5: Commit**

```bash
git add tests/python_integration/test_corr_engine.py tests/kernel/test_corr_kernel.py
git commit -m "M6 corr T7: null fallback, dtype coercion, degenerate-shape edge tests"
```

---

## Task 8: Benchmark + regression gate

**Files:**
- Create: `tests/bench/bench_corr.py`
- Test: bench runs (not a correctness gate, but include a `ratio_lt` floor)

- [ ] **Step 1: Write the benchmark**

Create `tests/bench/bench_corr.py`. Match the structure of an existing bench in `tests/bench/` — grep for one that uses the project's `_gate` / `ratio_lt` helper and mirror its imports and gating exactly (do not invent a new harness). The body:

```python
"""Bench: GPU correlation matrix vs Polars CPU df.corr(). The spike's crossover
floor: at N=1M, p=50 the spike saw ~20×; gate a conservative ratio_lt so a
dispatch-cliff regression is caught. Bench ≠ test (perf, not correctness)."""

import time

import numpy as np
import polars as pl

import polars_metal as pm


def _median(fn, it):
    ts = []
    for _ in range(it):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def bench_corr_sweep():
    rng = np.random.default_rng(0xC1)
    print(f"{'N':>9} {'p':>4} {'cpu_ms':>9} {'gpu_ms':>9} {'speedup':>8}")
    for n in (100_000, 1_000_000):
        for p in (10, 25, 50):
            x = rng.standard_normal((n, p)).astype(np.float32)
            df = pl.DataFrame(x, schema=[f"c{i}" for i in range(p)])
            lf = df.lazy()
            # warmup
            df.corr()
            lf.metal.corr(force_gpu=True).collect(engine=pm.MetalEngine())
            it = 5 if n <= 100_000 else 3
            cpu = _median(lambda: df.corr(), it)
            gpu = _median(
                lambda: lf.metal.corr(force_gpu=True).collect(engine=pm.MetalEngine()), it
            )
            print(f"{n:>9,} {p:>4} {cpu*1e3:>9.3f} {gpu*1e3:>9.3f} {cpu/gpu:>7.2f}x")


if __name__ == "__main__":
    bench_corr_sweep()
```

- [ ] **Step 2: Run the benchmark and record numbers**

Run: `python tests/bench/bench_corr.py`
Expected: GPU wins at p≥10 for N≥100K (roughly tracking the spike: several × up to ~20× at N=1M, p=50). Record the printed table in the commit message. If GPU does NOT win at N=1M p=50, that is a regression vs the spike — stop and investigate (likely a per-call ingest/eval cost the engine path adds over the raw-MLX spike) before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/bench/bench_corr.py
git commit -m "M6 corr T8: corr matrix benchmark vs CPU df.corr()

<paste the printed N/p/speedup table here>"
```

---

## Task 9: Docs, divergence ledger, roadmap, memory

**Files:**
- Modify: `docs/open-questions.md` (or the divergence ledger doc — grep for where "Mean F32 returns F32" is recorded)
- Modify: `CLAUDE.md` (roadmap item 10 note)
- Modify memory: `/Users/dclark/.claude/projects/-Users-dclark-dev-polars-metal-main-polars-metal/memory/`

- [ ] **Step 1: Record the F32 divergence**

Find where the M3 "Mean F32 returns F32 not F64" divergence is documented (grep `Mean F32` across `docs/`). Add an entry in the same place:

```
- `.metal.corr()` returns a Float32 p×p matrix (Polars `df.corr()` returns Float64).
  Values are F32-precision (~1e-5). Output dtype is path-independent (the small-p /
  null / N<2 CPU-fallback path also casts to F32). Routing: GPU when p ≥ CORR_P_MIN=8
  or force_gpu=True; else CPU df.corr() cast F32. Nulls / non-F32 inputs handled on CPU.
```

- [ ] **Step 2: Update the roadmap**

In `CLAUDE.md`, find roadmap item 10 (List / Array dot-product → MLX matmul, "Includes the correlation matrix"). Add a one-line status note that the correlation-matrix piece is delivered:

```
   (correlation matrix DELIVERED via lf.metal.corr() — MLX standardize+GEMM, F32,
   ~5–20× at p≥8; see docs/superpowers/specs/2026-06-11-m6-corr-matrix-design.md)
```

- [ ] **Step 3: Write a memory file**

Create `/Users/dclark/.claude/projects/-Users-dclark-dev-polars-metal-main-polars-metal/memory/m6-corr-matrix.md`:

```markdown
---
name: m6-corr-matrix
description: lf.metal.corr() GPU correlation matrix — first LazyFrame .metal verb, frame-replacing dispatch, F32, p>=8 routing
metadata:
  type: project
---

`lf.metal.corr()` ships the deferred correlation-matrix piece (roadmap item 10) on the
m6-vector-search branch. **Why it was different from the other .metal verbs:** corr is
p-cols-in/p×p-out, so (1) it's the first **LazyFrame** `.metal` namespace verb
(register_lazyframe_namespace, separate registry from the Expr `.metal`), and (2) its
dispatch is **frame-replacing** (returns a fresh p×p frame) not column-stitching like
dtw/fft/vector.

**How to apply:** kernel is MLX-only (no MSL) — `C = Znᵀ·Zn` over per-column
L2-normalized centered columns; the (N−1) normalization cancels so only existing
mlx-sys wrappers are needed (mean_axis/sub/mul/sum_axis/sqrt/div/transpose/matmul).
Routing guard CORR_P_MIN=8 (spike crossover; p=2 is bandwidth-bound = the B4 loser,
p≥8 compute-bound = win). Nulls / non-numeric / N<2 → Polars CPU df.corr() cast F32,
so output is F32 regardless of path. Honest perf ~5–20× (grows with p; dips past p≈100).
Built via [[spike-unknowns-during-brainstorm]] (scripts/spike_corr_crossover.py). The
inverse-of-B4 framing mirrors [[m6-a4-dtw-execution-state]] (compute-bound wins).
```

Add the pointer line to `MEMORY.md`:

```
- [M6 corr matrix](m6-corr-matrix.md) — lf.metal.corr() GPU correlation matrix: first LazyFrame .metal verb, frame-replacing dispatch, MLX-only Znᵀ·Zn (no MSL), CORR_P_MIN=8, F32 output
```

- [ ] **Step 4: Run the full gate**

Run: `make gate`
Expected: lint clean (cargo fmt/clippy, ruff), unit + kernel + integration tests green. Fix any fmt/lint drift (the subagent-fmt-discipline gotcha). If conformance shows the known pre-existing deferrals only, that is acceptable.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs/ "/Users/dclark/.claude/projects/-Users-dclark-dev-polars-metal-main-polars-metal/memory/"
git commit -m "M6 corr T9: divergence ledger, roadmap note, memory"
```

---

## Self-Review Notes (for the executor)

- **Spec coverage:** surface (T2), detection (T3), frame-replacing dispatch (T4/T5), MLX kernel (T1), routing guard + force_gpu (T4/T6), F32 path-independent output (T4/T6/T7), Pearson-only (implicit — no Spearman path), dtype-cast + non-numeric raise (T7), nulls→CPU (T7), tests at all three levels (T1/T7 kernel, T5/T6/T7 differential), bench+gate (T8), docs (T9). All spec sections map to a task.
- **Type/name consistency:** `corr_matrix`/`execute_corr` (Rust), `CorrSpec`/`CorrBinding`/`CORR_SENTINEL_TAG`/`CORR_SENTINEL_COL`/`CORR_P_MIN`/`pop_capture`/`find_corr_bindings`/`apply_corr` (Python) are used identically across tasks.
- **The one new dispatch primitive** (frame-replacing `apply_corr`) is isolated in T4 with a direct unit test before the collect-wiring in T5 — so if frame-replacement misbehaves, it surfaces before integration.
- **Oracle discipline:** every "expected" uses `np.corrcoef` or Polars `df.corr()`; where NaN can appear (constant column, null fallback), comparisons use `equal_nan=True`. Polars CPU is the spec — if a divergence appears, debug toward it, never loosen tolerances.
