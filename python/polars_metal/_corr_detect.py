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

from polars_metal import _detect_common as dc
from polars_metal._corr_namespace import CORR_SENTINEL_TAG
from polars_metal._detect_common import _alias_name, _literal_int, _struct_fields

_corr_lf_exprs_cache: dict[int, list[pl.Expr]] = {}
_PATCH_ATTR = "_polars_metal_corr_original_with_columns"

dc.install_with_columns_capture(_PATCH_ATTR, _corr_lf_exprs_cache)


@dataclass(frozen=True)
class CorrBinding:
    out_name: str
    handle: int


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
    """Return CorrBinding for each sentinel alias in the outermost with_columns layer."""
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

        # Slow fallback: pre-filter then bounded scan.
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
