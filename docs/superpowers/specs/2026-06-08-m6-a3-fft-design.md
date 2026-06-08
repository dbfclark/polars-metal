# M6 A3 — `.metal.fft()` / `.metal.ifft()` Design

**Status:** Approved (brainstorm 2026-06-08). Drill of A3 from the M6 umbrella spec
(`docs/superpowers/specs/2026-06-04-m6-metal-namespace-design.md` §A3).

**Goal:** Ship GPU-accelerated 1-D FFT as `.metal` expression-namespace verbs —
`pl.col("signal").metal.fft()` and `.metal.ifft()` — executed under `collect(engine="metal")`,
returning a `Struct{real: Float32, imag: Float32}` column. DataFrame-native signal processing;
survey ceiling ~77× vs NumPy.

**Architecture:** Reuse the A2 vector-search template verbatim — a `.metal` expression namespace
that emits a serialize-detectable, CPU-raising `as_struct` sentinel; a serialize-detect pass +
collect-and-stitch dispatch under `engine="metal"`; a Rust PyO3 dispatcher composing existing MLX
FFI. See `[[m6-vector-search-execution-state]]` for the template's reusable parts.

**Tech Stack:** Rust + `cxx` FFI to MLX (C++), PyO3, Polars py-1.40.1, numpy (oracle/marshalling).

---

## User surface

```python
import polars as pl
import polars_metal  # registers engine + .metal namespace

df = pl.DataFrame({"signal": signal_f32})            # N rows, Float32
spectrum = df.with_columns(
    pl.col("signal").metal.fft().alias("spectrum")   # → Struct{real: F32, imag: F32}, N rows
).collect(engine="metal")

recovered = spectrum.with_columns(
    pl.col("spectrum").metal.ifft().alias("recovered")  # Struct → Struct, N rows
).collect(engine="metal")
```

### Semantics

- **Whole column = one 1-D signal.** A column of N rows is one length-N signal. The transform is
  length-preserving: N input samples → N complex outputs, returned as a `Struct{real, imag}`
  column of N rows. Because length is preserved, the result composes with `with_columns`.
- **Full complex transform** (`mlx_fft` / `mlx_ifft`). No `rfft` (its N→N/2+1 length change breaks
  the per-column model — explicitly dropped at brainstorm time).
- **Input dtype — both verbs accept either:**
  - a **real `Float32`** column → treated as complex with zero imaginary part, or
  - a **`Struct{real: Float32, imag: Float32}`** column → true complex (enables `fft().ifft()`
    round-trips on real DataFrame data).
- **Output dtype:** always `Struct{real: Float32, imag: Float32}`.
- **Arbitrary N.** MLX FFT handles non-power-of-2 lengths (smooth sizes are fastest); no padding,
  no truncation. Output length == input length.

### Edge cases (mirror A2)

- Input that is neither `Float32` nor `Struct{Float32, Float32}` → **raise** a `ComputeError`
  (no CPU equivalent; the op has no native Polars fallback).
- **Nulls** in the signal (or in either struct field) → **raise** (FFT over nulls is ill-defined).
- **Empty** column → empty `Struct{real, imag}` column (0 rows).

---

## Components

### 1. MLX FFI — one new wrapper (everything else exists)

Already present in `crates/polars-metal-mlx-sys/src/fft.rs` + `cxx/mlx_bridge.{h,cc}`:
`mlx_fft`, `mlx_ifft` (1-D, last axis, real-or-complex input → complex output), `mlx_real`,
`mlx_imag` (complex → F32). F32 readback via `array::mlx_array_to_f32_vec`. Round-trip already
covered by `tests/test_scan_matmul_fft.rs::fft_ifft_round_trip`.

**New:** `mlx_complex(re, im) -> complex64` — assemble a complex array from two F32 streams, for the
struct-input path. C++ implementation: `astype(re, complex64) + astype(im, complex64) * i`
(MLX has no single complex-constructor op; compose from existing ops). Add as:
- `mlx_bridge.h` / `mlx_bridge.cc`: `mlx_op_complex(const std::shared_ptr<MlxArray>&, const std::shared_ptr<MlxArray>&)`
- `src/lib.rs` extern block declaration
- `src/fft.rs`: `pub fn mlx_complex(re, im) -> Result<MlxArrayHandle, FfiError>` (propagate both `_input_refs`)

### 2. Rust dispatcher — `crates/polars-metal-core/src/fft.rs` (new)

Pure core + PyO3 binding, mirroring `vector_search.rs`:

```
pub fn fft_core(input: FftInput, n: i64, inverse: bool) -> Result<(Vec<f32>, Vec<f32>), FfiError>
// FftInput::Real(&[f32])  → view (n,) F32 array
// FftInput::Complex(&[f32] re, &[f32] im) → mlx_complex(view(re), view(im))
//   → mlx_fft | mlx_ifft → (mlx_real, mlx_imag) → mlx_array_eval → two mlx_array_to_f32_vec
```

PyO3 binding `execute_fft`, registered in `crates/polars-metal-core/src/lib.rs`, using the
established `(ptr, len)` buffer convention:

```
#[pyfunction]
fn execute_fft(
    real: (usize, usize),                  // (ptr, len) of the real stream
    imag: Option<(usize, usize)>,          // Some(...) for struct input; None for real-only input
    n: i64,
    inverse: bool,
) -> PyResult<(Vec<f32>, Vec<f32>)>        // (real_out, imag_out), each length n
```

`mod fft;` added near the other module declarations in `lib.rs`.

### 3. Python namespace — extend the existing `metal` expr namespace

The `metal` namespace is already registered (`register_expr_namespace("metal")`) for
`cosine_topk` / `knn`. Add `.fft()` / `.ifft()` methods to the same namespace class. Each:
- captures nothing external (unlike vector search there is no corpus) — the input column IS the data,
- builds a serialize-detectable, CPU-raising `as_struct` **sentinel** carrying the input column name
  + an op tag (`fft` / `ifft`) under a magic alias prefix (`__pm_fft__`),
- raises on plain-CPU collect via an opaque `map_batches` field (dropped before CPU collect under
  `engine="metal"`).

New/edited Python files (mirroring the `_vector_*` trio):
- `python/polars_metal/_fft_namespace.py` — the `.fft()`/`.ifft()` sentinel builders (or add to the
  existing namespace module if the namespace class lives in one place — follow the A2 layout).
- `python/polars_metal/_fft_detect.py` — serialize-detect `__pm_fft__` sentinel bindings from a LazyFrame.
- `python/polars_metal/_fft_dispatch.py` — collect-and-stitch: read the input column (real F32 or
  struct), rechunk to contiguous F32, call `execute_fft`, build the `Struct{real, imag}` column,
  stitch back into the result frame (drop the sentinel output column first).
- `python/polars_metal/__init__.py` — import the namespace module (registers methods) + wire
  detect/dispatch into `collect_wrapper` alongside the vector-search hooks.

### 4. The existing fused-subgraph `Fft`/`Ifft` arms — left as-is

`fusion/subgraph.rs` (`Fft => ffi(mlx_fft(args[0]))`), `supported_ops.rs` (op defs + n·log2(n)
cost), `py.rs` (string→op), `scope.rs` (allowed set) wire `Fft`/`Ifft` into the fused walker.
That path is **unreachable** — `fft` is not a NodeTraverser-viewable expression
(`[[m4-nodetraverser-opacity]]`) — and would fold a complex result into an F32 Series incorrectly
if ever hit. **Out of scope to remove.** The live path is the `.metal` namespace route. Add a
one-line comment at the `Fft` subgraph arm pointing to this spec; do not rely on or extend it.

---

## Data flow

```
pl.col("signal").metal.fft()
        │  (build as_struct sentinel: {input col, op tag "fft", raise-on-cpu map_batches})
        ▼
df.collect(engine="metal")
        │  collect_wrapper → _fft_detect: serialize plan, find __pm_fft__ bindings
        ▼
_fft_dispatch:
   drop sentinel output col → CPU-collect remaining frame
   read input col: Float32 → real stream; Struct{real,imag} → two streams
   rechunk contiguous → execute_fft(real, imag?, n, inverse)
        │
        ▼ Rust fft_core: view → [mlx_complex] → mlx_fft|mlx_ifft → mlx_real + mlx_imag → eval → readback
        │
        ▼  (real_out, imag_out): two Vec<f32>, length n
   build Struct{real: F32, imag: F32} column (n rows)
   stitch into result frame → return DataFrame
```

---

## Testing strategy

`tests/python_integration/test_fft.py`:
- **Differential vs numpy** — `np.fft.fft` over random F32 signals (various N: power-of-2 and not);
  assert real/imag within tolerance (F32 ≈ 1e-3 relative, scaled for N). `np.fft.ifft` for the
  inverse.
- **Round-trip** — `fft().ifft()` recovers the original signal within tolerance.
- **Struct input** — `ifft` on a `Struct{real,imag}` column equals `np.fft.ifft` on the assembled
  complex array.
- **Raise tests** — non-F32 column; nulls in the signal; (optionally) wrong struct field dtypes.
- **Empty** — empty column → empty struct column.

`crates/polars-metal-mlx-sys/tests/test_vector_ffi.rs` (or the existing fft test file):
- `mlx_complex` building-block test (assemble (re, im) → complex64, verify via `mlx_real`/`mlx_imag`).

`crates/polars-metal-core` `#[cfg(test)]` in `fft.rs`:
- `fft_core` small known-signal test (e.g. impulse → flat spectrum; DC → single bin) for real and
  complex input, forward and inverse.

`tests/bench/m4_survey/bench_fft.py`:
- 8M-point 1-D FFT, metal-engine path vs `numpy.fft.fft`, `_gate.ratio_lt` honest threshold.
  Report the apples-to-apples comparison (engine collect path, not bare FFI) per the A2
  honest-baseline discipline (`[[m6-vector-search-execution-state]]`).

---

## Open questions (resolve at plan/drill time)

- **Tolerance constant** for the numpy differential — FFT error grows with N; pick a bound that is
  N-aware (e.g. `~N · eps_f32` absolute, or relative-to-magnitude), echoing the backward-error
  lesson from `[[m3-conformance-deferrals]]` (don't use an unbounded relative-to-result metric).
- **Struct-input null check granularity** — whether to reject only outer-null rows or also
  validate both inner fields are fully non-null before assembling the complex array.
- **`mlx_complex` constant `i`** — confirm the cleanest C++ idiom for the imaginary unit in the
  installed MLX rev (a `complex64` scalar array vs. a typed literal); de-risk in the first FFI task.
```
