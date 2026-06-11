"""M6 vector search: `.metal` expression namespace, corpus capture, sentinel builder.

User surface:
    pl.col("emb").metal.cosine_topk(corpus_lf, k, corpus_col="emb")
    pl.col("emb").metal.knn(corpus_lf, k, corpus_col="emb")

These return a *sentinel* expression that:
  - is serialize-detectable (carries the query column + an integer handle-id),
  - raises on plain-CPU collect (an opaque map_batches marker),
  - is recognized + dispatched to the GPU by collect(engine="metal").

The corpus (a LazyFrame / DataFrame / numpy array) is held by-reference in a module-global
dict keyed by the handle-id; the dispatcher reads it non-removingly and ties eviction to the
collected lf's lifetime via weakref.finalize (so a repeated collect of the same lf reuses it).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import polars as pl

from polars_metal import _dtw_namespace, _fft_namespace

_HANDLE_COUNTER = itertools.count(1)


@dataclass(frozen=True)
class CorpusSpec:
    corpus: Any  # LazyFrame | DataFrame | numpy ndarray (by-reference)
    corpus_col: str
    k: int
    metric: str  # "cosine" | "knn"
    query_col: str


# Handle-id → corpus spec, held BY-REFERENCE. Read non-removingly at dispatch
# (get_capture) and evicted when the collected lf is GC'd (weakref.finalize →
# evict_capture). NOTE: a captured expr that is built but never collected under
# engine="metal" leaks its entry (and the corpus it references) until process exit —
# inherent to the by-reference design; acceptable for the MVP.
_CORPUS_CACHE: dict[int, CorpusSpec] = {}

# Magic prefix embedded in the sentinel's Int64-literal field alias so the
# serialize detector can find our bindings unambiguously.
SENTINEL_TAG = "__pm_vsearch__"


def _capture_corpus(corpus: Any, corpus_col: str, k: int, metric: str, query_col: str = "") -> int:
    handle = next(_HANDLE_COUNTER)
    _CORPUS_CACHE[handle] = CorpusSpec(corpus, corpus_col, k, metric, query_col)
    return handle


def get_capture(handle: int) -> CorpusSpec | None:
    """Non-removing read (repeated collect of the same lf reuses the corpus spec;
    the dispatcher ties eviction to the lf lifetime via weakref.finalize)."""
    return _CORPUS_CACHE.get(handle)


def pop_capture(handle: int) -> CorpusSpec | None:
    return _CORPUS_CACHE.pop(handle, None)


def evict_capture(handle: int) -> None:
    """Remove the spec for *handle* from the cache. Registered as a weakref
    finalizer on the dispatched LazyFrame so the entry is freed on lf GC."""
    _CORPUS_CACHE.pop(handle, None)


def _raise_cpu(_s: pl.Series) -> pl.Series:
    raise RuntimeError(
        "polars_metal: .metal.cosine_topk/.knn require collect(engine='metal'); "
        "they have no CPU implementation."
    )


def build_sentinel(query_col_expr: pl.Expr, query_col_name: str, handle: int) -> pl.Expr:
    """Build the recognizable, CPU-raising sentinel struct expression.

    Shape (serialized): a struct with three fields:
      - field 0: the query column (so the detector reads the input column name),
      - field 1: a literal i64 handle-id tagged with SENTINEL_TAG via its alias,
      - field 2: an opaque map_batches(_raise) over the query column → raises on CPU.
    Under engine="metal", dispatch DROPS this output column before the CPU collect, so the
    map_batches never executes; on plain CPU it executes and raises.
    """
    return pl.struct(
        [
            query_col_expr.alias("__pm_vs_query"),
            pl.lit(handle, dtype=pl.Int64).alias(f"{SENTINEL_TAG}{query_col_name}"),
            query_col_expr.map_batches(_raise_cpu, return_dtype=pl.Float32).alias("__pm_vs_raise"),
        ]
    )


@pl.api.register_expr_namespace("metal")
class MetalExprNamespace:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def _query_col_name(self) -> str:
        # Best-effort: the root column name drives detection. meta.root_names()
        # returns the input column(s); we require exactly one.
        roots = self._expr.meta.root_names()
        if len(roots) != 1:
            raise ValueError(
                "polars_metal: .metal.cosine_topk/.knn must be applied to a single "
                f"column (got roots {roots})."
            )
        return roots[0]

    def cosine_topk(self, corpus: Any, k: int, corpus_col: str = "emb") -> pl.Expr:
        if k < 1:
            raise ValueError("k must be >= 1")
        qcol = self._query_col_name()
        handle = _capture_corpus(corpus, corpus_col, k, "cosine", qcol)
        return build_sentinel(self._expr, qcol, handle)

    def knn(self, corpus: Any, k: int, corpus_col: str = "emb") -> pl.Expr:
        if k < 1:
            raise ValueError("k must be >= 1")
        qcol = self._query_col_name()
        handle = _capture_corpus(corpus, corpus_col, k, "knn", qcol)
        return build_sentinel(self._expr, qcol, handle)

    def _input_col(self) -> str:
        roots = self._expr.meta.root_names()
        if len(roots) != 1:
            raise ValueError(
                "polars_metal: .metal.fft/.ifft must be applied to a single column "
                f"(got roots {roots})."
            )
        return roots[0]

    def fft(self) -> pl.Expr:
        col = self._input_col()
        return _fft_namespace.build_fft_sentinel(self._expr, col, _fft_namespace.OP_FFT)

    def ifft(self) -> pl.Expr:
        col = self._input_col()
        return _fft_namespace.build_fft_sentinel(self._expr, col, _fft_namespace.OP_IFFT)

    def dtw(
        self,
        reference: Any,
        window: int | None = None,
        allow_cpu_fallback: bool = False,
    ) -> pl.Expr:
        return _dtw_namespace.make_dtw_expr(self._expr, reference, window, allow_cpu_fallback)
