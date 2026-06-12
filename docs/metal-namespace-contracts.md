# `.metal` namespace — per-verb contracts

The `.metal` namespace verbs are serialize-detected and run on the GPU via a
collect-and-stitch pipeline. Their contracts differ **by design** along three
axes — null handling, boundary error type, streaming — because the verbs have
different semantics and different CPU-fallback availability. This document is
the source of truth; the characterization tests in
`tests/python_integration/test_metal_namespace_contracts.py` pin it.

## The three axes

| Verb | null input | boundary error | streaming=True |
|------|-----------|----------------|----------------|
| `cosine_topk` / `knn` | **raise** — a null in an embedding is meaningless | `ValueError` (user input: dtype/dim) · `ComputeError` (engine internal) | **raise** `ComputeError` |
| `fft` / `ifft` | **raise** — a null in a signal is meaningless | `ValueError` (user input: dtype) · `ComputeError` (engine internal) | **raise** `ComputeError` |
| `dtw` | **mask + restore** — null rows pass through positionally | `ValueError` (user input: dtype/shape/NaN cell) · `ComputeError` (engine internal / missing optional dep) | **raise** `ComputeError` |
| `corr` | **CPU fallback** — pairwise-complete correlation is well-defined | `ValueError` (non-numeric col) · `ComputeError` (N<2, engine internal) | **raise** `ComputeError` |
| `rolling_{mean,sum,var,std}` | **CPU fallback** | `ValueError` (user input) | **silent CPU fallback** |
| `dt.{year,month,day}` | **mask + restore** | validation reject → CPU | **silent CPU fallback** |

## Why the divergences are intentional

- **Null handling is semantic.** corr's pairwise-complete behavior is a real,
  well-defined statistic, so it routes to CPU rather than refusing. dtw's rows
  are sequences; a null *row* is a missing sequence and is restored
  positionally (a null *cell* inside a non-null sequence is still an error —
  the GPU kernel can't match NaN against the oracle). For vector search and
  FFT a null inside an embedding/signal has no meaning, so they refuse.

- **Error-type rule.** *User-input validation* (wrong dtype, dimension
  mismatch, non-numeric column) raises `ValueError`. *Engine-internal or
  boundary failures* (a capture handle that went missing, an N<2 frame, a
  missing optional dependency) raise `pl.exceptions.ComputeError`, per the
  engine convention that boundary errors look native to Polars users.

- **Streaming.** The vector/fft/dtw/corr kernels have **no CPU
  implementation**, so requesting `streaming=True` over a plan that contains
  one of their sentinels raises `ComputeError` rather than silently producing
  a wrong/absent result. rolling and dt **do** have exact CPU equivalents, so
  under streaming they silently fall back to CPU — the user still gets the
  correct answer, just not on the GPU.

- **fft has no capture cache.** Unlike vector/dtw/corr, fft encodes its op code
  (`OP_FFT` / `OP_IFFT`) directly in the sentinel's `Int64` literal — there is
  no by-reference spec to cache and no handle to evict. Repeated collects of
  the same LazyFrame therefore always succeed. fft correctly has no
  handle-evicted guard; this is not a missing feature.

## Internal cache lifetime (vector / dtw / corr)

These verbs capture a by-reference spec (corpus / reference sequence / column
set) in a per-verb `CaptureCache` keyed by an `Int64` handle embedded in the
sentinel. The dispatcher registers a `weakref.finalize(lf, evict, handle)` so
the spec is freed when the dispatched LazyFrame is GC'd, while repeated
collects of a *live* lf reuse it. If a handle is missing at dispatch time
(evicted early), the verb raises `ComputeError("... handle missing ...")`.
