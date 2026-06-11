"""M6 A4 DTW: detect sentinel bindings from the outermost with_columns layer.

Identical strategy to _vector_detect (fast with_columns-capture path + slow
explain()/serialize fallback; never json.loads the full plan). Its OWN patch
attr + cache so it chains safely alongside rolling/vector/fft.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import polars as pl

from polars_metal import _detect_common as dc
from polars_metal._detect_common import _alias_name, _literal_int, _struct_fields
from polars_metal._dtw_namespace import DTW_SENTINEL_TAG

_dtw_lf_exprs_cache: dict = {}  # id(lf) -> (weakref.ref, exprs); managed by _detect_common
_PATCH_ATTR = "_polars_metal_dtw_original_with_columns"

dc.install_with_columns_capture(_PATCH_ATTR, _dtw_lf_exprs_cache)


@dataclass(frozen=True)
class DtwBinding:
    out_name: str
    query_col: str
    handle: int


def _binding_from_expr_json(expr_json: dict, out_name: str) -> DtwBinding | None:
    try:
        s = json.dumps(expr_json)
        if DTW_SENTINEL_TAG not in s:
            return None
        fields = _struct_fields(expr_json)
        query_col = None
        handle = None
        for fld in fields:
            alias_name = _alias_name(fld)
            if alias_name and alias_name.startswith(DTW_SENTINEL_TAG):
                query_col = alias_name[len(DTW_SENTINEL_TAG) :]
                handle = _literal_int(fld)
        if query_col is None or handle is None:
            return None
        return DtwBinding(out_name=out_name, query_col=query_col, handle=handle)
    except Exception:
        return None


def find_dtw_bindings(lf: pl.LazyFrame) -> list[DtwBinding]:
    """Return DtwBinding for each sentinel alias in the outermost with_columns layer."""
    try:
        cached = dc.lookup(_dtw_lf_exprs_cache, lf)
        if cached is not None:
            out: list[DtwBinding] = []
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

        # Slow fallback: pre-filter then bounded scan.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            if DTW_SENTINEL_TAG not in lf.explain():
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
