"""Serialize-detect lf.metal.corr() struct sentinel in a LazyFrame.

corr's sentinel uses an EXACT tag alias (no source-column suffix) and is
frame-replacing — at most one sentinel per lf is meaningful.
"""

from __future__ import annotations

import polars as pl

from polars_metal import _detect_common as dc
from polars_metal._corr_namespace import CORR_SENTINEL_TAG
from polars_metal._detect_common import SentinelBinding

_corr_exprs_cache: dict = {}
dc.install_with_columns_capture("_polars_metal_corr_original_with_columns", _corr_exprs_cache)

_parse = dc.make_sentinel_parser(CORR_SENTINEL_TAG, exact=True)


def find_corr_bindings(lf: pl.LazyFrame) -> list[SentinelBinding]:
    out: list[SentinelBinding] = []
    for inner, name in dc.iter_candidate_nodes(
        lf, cache=_corr_exprs_cache, explain_tags=(CORR_SENTINEL_TAG,)
    ):
        b = _parse(inner, name)
        if b is not None and b.out_name:
            out.append(b)
    return out
