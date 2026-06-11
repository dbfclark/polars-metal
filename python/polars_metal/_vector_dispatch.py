"""M6 vector search: execute detected bindings on the GPU and stitch struct columns in.

Collect-and-stitch over whole, materialized columns (chunk-safe), mirroring _rolling_dispatch:
  1. drop the sentinel output columns → CPU-collect the rest (projection pushdown elides them,
     and crucially the opaque map_batches(_raise) field never runs),
  2. for each binding: materialize the query column + collect the corpus (pushdown), run the GPU
     op, sort each query's k by metric order, build the Struct column,
  3. reassemble in original schema order.
"""

from __future__ import annotations

import weakref

import numpy as np
import polars as pl

from polars_metal import _native
from polars_metal._vector_detect import VectorBinding
from polars_metal._vector_namespace import evict_capture, get_capture

_OP_CODE = {"cosine": 0, "knn": 1}
_TILE_ROWS_DEFAULT = 1 << 30  # effectively no tiling unless the corpus is enormous


def _array_col_to_matrix(s: pl.Series) -> tuple[np.ndarray, int, int]:
    """Return (contiguous (N*D) f32, N, D) for an Array(Float32, D) column."""
    if not isinstance(s.dtype, pl.Array) or s.dtype.inner != pl.Float32:
        raise ValueError(f"polars_metal vector search requires Array(Float32, D); got {s.dtype}")
    null_count = s.null_count()
    if null_count > 0:
        raise ValueError(
            f"polars_metal: .metal.cosine_topk/.knn does not support null rows in the "
            f"query or corpus embedding column (column {s.name!r} has {null_count} "
            "null rows). Drop or impute nulls first."
        )
    d = s.dtype.size
    n = s.len()
    flat = s.to_numpy()  # Array(F32, D) → (N, D) ndarray
    m = np.ascontiguousarray(flat, dtype=np.float32).reshape(-1)
    return m, n, d


def _corpus_matrix(spec_corpus, corpus_col: str) -> tuple[np.ndarray, int, int]:
    """Return (contiguous (N*D) f32, N, D) for the corpus embedding column."""
    if isinstance(spec_corpus, np.ndarray):
        m = np.ascontiguousarray(spec_corpus, dtype=np.float32)
        if m.ndim != 2:
            raise ValueError("numpy corpus must be 2-D (N, D)")
        return m.reshape(-1), m.shape[0], m.shape[1]
    corpus_df = spec_corpus.collect() if isinstance(spec_corpus, pl.LazyFrame) else spec_corpus
    s = corpus_df.get_column(corpus_col).rechunk()
    return _array_col_to_matrix(s)


def _build_struct(
    indices: np.ndarray, scores: np.ndarray, q_rows: int, k: int, metric: str
) -> pl.Series:
    """Sort each query's k by metric order and build Struct{indices, scores}."""
    idx_lists: list[list[int]] = []
    score_lists: list[list[float]] = []
    for qi in range(q_rows):
        ii = indices[qi * k : (qi + 1) * k]
        ss = scores[qi * k : (qi + 1) * k]
        if metric == "knn":
            ss = np.sqrt(np.maximum(ss, 0.0))  # squared → true L2
            order = np.lexsort((ii, ss))  # asc dist, then index asc
        else:
            order = np.lexsort((ii, -ss))  # desc sim, then index asc
        idx_lists.append([int(x) for x in ii[order]])
        score_lists.append([float(x) for x in ss[order]])
    return pl.Series(
        "",
        [{"indices": il, "scores": sl} for il, sl in zip(idx_lists, score_lists, strict=True)],
        dtype=pl.Struct({"indices": pl.List(pl.UInt32), "scores": pl.List(pl.Float32)}),
    )


def _run_binding(qframe: pl.DataFrame, b: VectorBinding) -> pl.Series:
    spec = get_capture(b.handle)
    if spec is None:
        raise RuntimeError("polars_metal: vector-search corpus handle missing (already consumed?)")
    qmat, q_rows, qd = _array_col_to_matrix(qframe.get_column(b.query_col).rechunk())
    cmat, n_rows, cd = _corpus_matrix(spec.corpus, spec.corpus_col)
    if qd != cd:
        raise ValueError(f"query D={qd} != corpus D={cd}")
    if n_rows == 0:
        # A query against an empty corpus has zero neighbours. Short-circuit:
        # the GPU staging path can't allocate a 0-byte buffer, and the correct
        # answer is an empty hit-list per query (matches the numpy oracle).
        return _build_struct(
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.float32),
            q_rows,
            0,
            spec.metric,
        ).rename(b.out_name)
    k = min(spec.k, n_rows)
    idx, val = _native.execute_vector_search(
        (qmat.ctypes.data, qmat.size),
        q_rows,
        (cmat.ctypes.data, cmat.size),
        n_rows,
        qd,
        k,
        _OP_CODE[spec.metric],
        _TILE_ROWS_DEFAULT,
    )
    idx = np.asarray(idx, dtype=np.uint32)
    val = np.asarray(val, dtype=np.float32)
    return _build_struct(idx, val, q_rows, k, spec.metric).rename(b.out_name)


def apply_vector_search(
    lf: pl.LazyFrame, bindings: list[VectorBinding], collect_fn
) -> pl.DataFrame:
    """Dispatch vector-search bindings to the GPU and stitch struct columns in.

    *collect_fn(rest_lf) -> DataFrame* collects the non-sentinel columns. Dropping the
    sentinel output columns (lf.drop(out_names)) elides their computation — including the
    opaque map_batches(_raise) — from the CPU path via projection pushdown.
    """
    out_names = [b.out_name for b in bindings]
    order = lf.collect_schema().names()
    # Tie spec eviction to the lf lifetime (repeated collects of the same lf
    # reuse the spec; it is freed when the lf is GC'd). Registering twice is
    # harmless — both do an idempotent dict.pop.
    for b in bindings:
        weakref.finalize(lf, evict_capture, b.handle)
    rest_lf = lf.drop(out_names)
    df = collect_fn(rest_lf)
    cols: dict[str, pl.Series] = {c: df.get_column(c) for c in df.columns}
    for b in bindings:
        cols[b.out_name] = _run_binding(df, b)
    return pl.DataFrame([cols[c] for c in order])
