"""Serialize-detect .metal.cosine_topk/.knn struct sentinels in a LazyFrame.

The candidate-iteration scaffold and the struct-sentinel parser live in
_detect_common; this module only wires the vector-search tag and the
with_columns capture cache onto that spine.
"""

from __future__ import annotations

import polars as pl

from polars_metal import _detect_common as dc
from polars_metal._detect_common import SentinelBinding
from polars_metal._vector_namespace import SENTINEL_TAG

_vs_exprs_cache: dict = {}
dc.install_with_columns_capture("_polars_metal_vs_original_with_columns", _vs_exprs_cache)

_parse = dc.make_sentinel_parser(SENTINEL_TAG)


def find_vector_bindings(lf: pl.LazyFrame) -> list[SentinelBinding]:
    out: list[SentinelBinding] = []
    for inner, name in dc.iter_candidate_nodes(
        lf, cache=_vs_exprs_cache, explain_tags=(SENTINEL_TAG,)
    ):
        b = _parse(inner, name)
        if b is not None and b.out_name:
            out.append(b)
    return out
