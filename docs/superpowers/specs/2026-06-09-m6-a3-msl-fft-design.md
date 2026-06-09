# M6 A3 — Hand-rolled MSL FFT Design

**Status:** Approved (brainstorm 2026-06-09). Supersedes the MLX-based FFT path for the large-N
regime; the interim MLX path (guarded to pow2 ≤ 2^20, `_GPU_FFT_MAX`) was shipped first and is
documented in `[[m6-a3-fft-execution-state]]`.

**Goal:** A from-scratch Metal (MSL) 1-D FFT kernel that is correct and fast for **all sizes**,
removing the 2^20 ceiling of MLX's Metal FFT (`ml-explore/mlx#1800` — silently wrong ≥2^23,
missing-kernel errors at 2^21/2^22, unfixed upstream through 0.31). It backs the existing
`.metal.fft()` / `.metal.ifft()` expression verbs; the user-facing surface is unchanged.

**Why hand-roll (not patch MLX):** per the `Never patch vendored code` convention — we adapt MLX's
*proven, M-series-tuned* FFT codelets into our own kernel and fix the size/correctness bugs there,
rather than forking `vendor/mlx` (re-apply-every-bump debt). MLX's source is the reference, not the
runtime.

**Tech Stack:** MSL (one `shaders/fft.metal` + headers), Rust dispatcher in `polars-metal-kernels`,
the repo's `ShaderLibrary`/`CommandQueue` abstraction, PyO3, numpy (differential oracle).

**Reference (read before porting each piece):**
- `vendor/mlx/mlx/backend/metal/kernels/fft/radix.h` — radix codelets, `complex_mul`, `get_twiddle`.
- `vendor/mlx/mlx/backend/metal/kernels/fft/readwrite.h` — load/store + real-packing helpers.
- `vendor/mlx/mlx/backend/metal/kernels/fft.metal` — kernel entry points / function-constant params.
- `vendor/mlx/mlx/backend/metal/fft.cpp` — host-side **planner** (radix decomposition, four-step,
  Rader, Bluestein selection, twiddle precompute). Our Rust planner mirrors this.

---

## Scope

- **All sizes**, no artificial cap. Power-of-2 and arbitrary N.
- **Forward `fft` and inverse `ifft`.**
- **Real `Float32` and complex `Struct{real:F32, imag:F32}` input**; output always `Struct{real,imag}`,
  length N (N→N, length-preserving).
- Whole-column-as-one-signal (unchanged from the shipped surface).
- **Out of scope:** multi-dimensional FFT, rfft half-spectrum output (we did full-complex by design),
  windowing, batched/per-row FFT (the column is one signal). 2-D / batched is a possible future.

---

## Algorithm

We port MLX's full algorithm set (it is correctness-and-speed across composite sizes, and everything
is gated by the differential test sweep, so the breadth is free):

1. **In-threadgroup radix Stockham base.** Mixed-radix codelets (radix 2,3,4,5,6,7,8) compute an FFT
   that fits in threadgroup memory (≤ ~4096 points, within the 32 KB threadgroup-memory limit), one
   threadgroup per transform, SIMD-parallel butterflies. Twiddles via `get_twiddle`; complex via
   `float2` + `complex_mul`. Ported from `radix.h`.
2. **Rader's algorithm** for prime factors not covered by the small radices (decomposes an n-point
   prime DFT into an (n-1)-point convolution done with the Stockham FFT). Ported from MLX.
3. **Four-step (Bailey) for large composite/pow2 N.** N = n₁·n₂ (each ≤ base limit): FFT columns →
   twiddle multiply → FFT rows → transpose. **This is the regime MLX breaks** — we instantiate every
   needed sub-size (no missing `four_step_mem_*` kernel) and fix the large-N indexing so it is
   correct end-to-end, gated by differential tests at 2^21…2^24.
4. **Recursive four-step for arbitrary large N.** When n₂ itself exceeds the base limit, four-step it
   too (a four-step pass whose sub-transform is four-stepped). This removes any algorithmic size
   ceiling — N is bounded only by available unified memory (see Error handling).
5. **Bluestein (chirp-z)** for arbitrary N with awkward factorizations: express the N-point DFT as a
   convolution evaluated with a fast pow2 FFT (size M ≥ 2N−1), then de-chirp. Exact, N→N.
6. **`ifft`** = the same kernels with conjugated twiddles and a final `1/N` scale (equivalently
   `conj(fft(conj(x)))/N`).
7. **Real & complex input.** The kernel operates on complex `float2`. Real input is loaded with
   imag=0 (MLX's `readwrite.h` real-packing optimization may be ported later as a perf win; v1 may
   treat real as complex for simplicity — a differential-tested perf decision).

**Planner (host side, Rust).** Mirrors `fft.cpp`: given N, factorize, choose radix decomposition /
Rader / four-step (recursive) / Bluestein, precompute twiddle/chirp buffers once, and emit the
ordered sequence of kernel dispatches. This is the brain; the `.metal` kernels are the codelets.

---

## Components & files

**New:**
- `shaders/fft.metal` — kernel entry points (radix-stage, four-step passes, twiddle-multiply,
  transpose, chirp/Bluestein helpers). Parameterized by function constants (radix, N, stride,
  inverse), echoing MLX's specialization.
- `shaders/_fft_radix.metal` — header-only (the `_` convention): `complex_mul`, `get_twiddle`, radix
  codelets, load/store helpers. Adapted from MLX's `radix.h`/`readwrite.h`.
- `crates/polars-metal-kernels/src/fft.rs` — the **planner + dispatcher**: factorization, plan
  construction, twiddle/chirp precompute, the multi-dispatch loop via `ShaderLibrary`/`CommandQueue`,
  and the buffer-arena management for intermediate passes. Returns `(real, imag)` host F32.
- `crates/polars-metal-kernels/tests/test_fft.rs` — the differential sweep (below).

**Modified:**
- `crates/polars-metal-kernels/build.rs` — already auto-compiles `shaders/*.metal`; the new file is
  picked up. Confirm the metallib includes the new entry points.
- `crates/polars-metal-core/src/fft.rs` — rewire `fft_core` / `execute_fft` to call the new kernel
  dispatcher instead of the MLX FFI. Keep the `(real, imag) → PyBytes` readback (already correct).
- `crates/polars-metal-mlx-sys` — the MLX FFT FFI (`mlx_op_fft_1d`/`ifft_1d`/`real`/`imag`/`complex`)
  becomes unused for the engine path. **Remove** it (and the `fft.rs` wrappers) once the kernel
  lands, unless still used elsewhere — keeping dead FFI is debt. `mlx_complex` may be retained if the
  complex-input staging reuses it; decide at implementation time.
- `python/polars_metal/_fft_dispatch.py` — **remove the `_GPU_FFT_MAX = 2^20` guard** once the kernel
  covers all sizes; the GPU path becomes the default for all N. CPU fallback stays only for the
  genuine edge (see Error handling) and the existing null/dtype guards.

---

## Data flow

```
.metal.fft()/.ifft()  (sentinel, unchanged)
        ▼  collect(engine="metal") → _fft_detect → _fft_dispatch (Python; null/dtype/empty guards)
        ▼  contiguous F32 (real) or two F32 streams (complex) → PyO3 execute_fft
        ▼  Rust planner (polars-metal-kernels/src/fft.rs): factorize N → plan
        ▼  precompute twiddles/chirp (host → MTLBuffer)
        ▼  dispatch loop: [radix stages | four-step passes | transpose | bluestein] via CommandQueue
        ▼  result in a known CONTIGUOUS complex layout (we control it → clean readback)
        ▼  (real, imag) → PyBytes → np.frombuffer → Struct{real,imag} column → stitch
```

---

## Error handling / fallback

- **No artificial size cap.** Apple Silicon's unified memory has no GPU-VRAM cliff; the only limit is
  total system RAM. FFT scratch is O(N) (a few × N×8 bytes for output + transpose/Bluestein buffers).
- **OOM edge:** on small-RAM machines (8/16 GB) at pathological N the working set can exceed RAM. On
  allocation failure, return a clear `ComputeError` (or CPU-fallback as a safety) — this is the
  existing CLAUDE.md "spill-to-CPU policy" open question, handled gracefully, not a hardcoded ceiling.
- **CPU-fallback-for-speed is a post-benchmark decision, not baked in.** Expectation: the GPU kernel
  wins at the sizes that matter (large N is the whole point). After the kernel works, benchmark
  GPU-vs-numpy across sizes; if some small-N regime is faster on CPU (dispatch overhead), route it to
  CPU then — but that threshold is set from measurement, not assumed. Recorded as an open question.
- Nulls / non-F32 / empty input: unchanged (Python dispatch already raises / short-circuits).
- **No silent garbage, ever** — correctness is gated by the differential sweep before the guard is
  lifted.

---

## Testing strategy

The differential oracle (numpy, L2-relative-error) is the safety net that makes aggressive
hand-optimization safe (CLAUDE.md). `crates/polars-metal-kernels/tests/test_fft.rs` (Rust-level,
dispatching the kernel) and `tests/python_integration/test_fft.py` (engine path):

- **Correctness sweep** vs `numpy.fft`: sizes spanning every algorithm path —
  - small pow2: 8, 16, 64, 256, 1024, 4096 (radix base),
  - the previously-broken band: 2^21, 2^22, 2^23, 2^24 (four-step; must be L2 ≤ ~1e-4, *not* garbage),
  - very large: 2^25, and a recursive-four-step size,
  - non-pow2 small: 1000, 100003 (mixed-radix / Bluestein),
  - non-pow2 large: 3_000_000, 8_000_000 (Bluestein / four-step) — exactly the sizes wrong today,
  - prime / awkward: a large prime, a 3·5·7-smooth size (radix + Rader).
  - each × {fft, ifft} × {real input, complex/struct input}; plus fft→ifft round-trip recovery.
- **Tolerance:** L2 relative error (robust to near-zero bins), threshold ~1e-4 for F32 (scale check
  for very large N). Heed `[[m3-conformance-deferrals]]` — bound error against magnitude, never an
  unbounded relative-to-result metric.
- **Performance gate** (`tests/bench/m4_survey/bench_fft.py`): re-target to large pow2 (2^23/2^24) and
  record the honest engine-path ratio vs numpy — the headline this whole effort is for. Add non-pow2
  and the previously-broken sizes. Keep the gate honest (engine path, not bare kernel).
- **Conformance:** `make gate` stays green.

---

## Open questions (resolve at plan / implementation / tuning time)

- **Base radix limit** — the exact max in-threadgroup transform size (threadgroup-memory bound;
  query the PSO's limits at runtime, do not hardcode). MLX uses 4096; confirm on-device.
- **Real-input packing** — port MLX's real-FFT optimization (halve the work) or treat real as complex
  in v1? A differential-tested perf decision; v1 may take the simple path and optimize later.
- **CPU-fallback-for-speed threshold** — set from the benchmark sweep, if any regime warrants it.
- **MLX FFT FFI removal** — confirm nothing else depends on `mlx_op_fft_*` before deleting; keep
  `mlx_complex` only if the complex-staging path reuses it.
- **Recursive four-step depth / transpose strategy** — tiled vs naive transpose for the inter-pass
  shuffle; pick the differential-correct version first, optimize the transpose after profiling.

---

## Delivered (2026-06-09)

**What shipped.** A from-scratch MSL 1-D FFT (`shaders/fft.metal` + the header-only
`shaders/_fft_radix.metal`, driven by the planner/dispatcher in
`crates/polars-metal-kernels/src/fft.rs`) backing `.metal.fft()` / `.metal.ifft()` for **all sizes
on-GPU** — removing MLX's 2^20 ceiling (`ml-explore/mlx#1800`). Algorithm set actually landed:
- **radix-2 Stockham base** for small pow2 in threadgroup memory;
- **mixed-radix (3–8) base** for composite N ≤ 1024;
- **Bailey four-step** for large pow2;
- **recursive batched four-step** for N > 2^20 — this is what fixes the previously-garbage 2^21–2^25
  band that MLX got wrong;
- **Bluestein chirp-z** for primes / non-smooth N (the DFT as a length-`M = next_pow2(2N−1)` pow2
  convolution).
- `ifft` via conjugated twiddles + `1/N` scale; real `F32` and `Struct{real,imag}` input both
  supported; output always `Struct{real,imag}`, length-preserving.

Supported range `n ∈ [1, 2^30]` (guarded; Bluestein's `M = next_pow2(2N−1)` is itself kept ≤ 2^30).

**Correctness.** Differentially verified against three oracles: the CPU O(N²) DFT (`dft_reference`,
small N), an f64 radix-2 host oracle (large pow2), and `numpy.fft` at the engine level. L2 relative
error **< 1e-3** across the sweep (≤ 1e-4 at small sizes), over
`[8, 1024, 4096, 2^21..2^25, 1000, 100003, 3_000_000, 8_000_000] × {fft, ifft} × {real, struct}`,
plus fft→ifft round-trip recovery. No silent garbage at any size — the band MLX returned garbage on
is now correct.

**Performance (honest, M2 Ultra, full engine `collect` path vs `numpy.fft`).** The GPU **wins** at
the large-pow2 regime — `metal/numpy ≈ 0.23–0.26` at 2^23 and 2^24 (~3.8–4.3×), run-to-run band
~0.23–0.36. The bench (`tests/bench/m4_survey/bench_fft.py`) gates `ratio < 0.6` at 2^23 / 2^24 —
the honest measured win with headroom, not an aspirational number. This large-pow2 win is the
headline this whole effort was for.

**Rader's algorithm (Task 7): SKIPPED.** Bluestein already covers prime / non-smooth N **correctly**;
Rader is a perf-only optimization for prime sizes, which are not this milestone's headline workload
(the headline — large pow2 — is delivered by the recursive four-step). The added complexity
(primitive-root finding, modular index permutations, its own (n−1)-point convolution) was not
justified for a non-headline path. Decision recorded per the plan's explicit "skip and record"
authorization.

**MLX FFT FFI removed.** `polars-metal-mlx-sys` no longer wraps `mlx_op_fft_1d` / `ifft_1d` /
`real` / `imag` / `complex` (dead after the kernel landed). The fused-subgraph `Fft` / `Ifft` arms in
`crates/polars-metal-core/src/fusion/subgraph.rs` now return `UnsupportedOp` — those arms are
unreachable from the NodeTraverser walker anyway; the **live FFT path is the `.metal` namespace**
backed by the hand-rolled kernel.

**Deferred (future optimization).** Bluestein's chirp/filter build, the pointwise `FFT·FFT` product,
and the post-multiply are done host-side for simplicity, costing extra host↔device round-trips around
the three GPU FFTs. A fused premul/postmul MSL kernel (the plan's `fft_chirp_premul` /
`fft_chirp_postmul`) would keep that O(M) work on-device — deferred, as Bluestein is the non-headline
path. (Noted in `fft.rs` near `bluestein`.)
