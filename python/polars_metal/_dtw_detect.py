"""Serialize-detect .metal.dtw struct sentinels in a LazyFrame."""

from __future__ import annotations

import polars as pl

from polars_metal import _detect_common as dc
from polars_metal._detect_common import SentinelBinding
from polars_metal._dtw_namespace import DTW_SENTINEL_TAG

_dtw_exprs_cache: dict = {}
dc.install_with_columns_capture("_polars_metal_dtw_original_with_columns", _dtw_exprs_cache)

_parse = dc.make_sentinel_parser(DTW_SENTINEL_TAG)


def find_dtw_bindings(lf: pl.LazyFrame) -> list[SentinelBinding]:
    out: list[SentinelBinding] = []
    for inner, name in dc.iter_candidate_nodes(
        lf, cache=_dtw_exprs_cache, explain_tags=(DTW_SENTINEL_TAG,)
    ):
        b = _parse(inner, name)
        if b is not None and b.out_name:
            out.append(b)
    return out
