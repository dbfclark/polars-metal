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
import polars.lazyframe.frame as _plf

from polars_metal._dtw_namespace import DTW_SENTINEL_TAG

_dtw_lf_exprs_cache: dict[int, list[pl.Expr]] = {}
_PATCH_ATTR = "_polars_metal_dtw_original_with_columns"

if not hasattr(_plf.LazyFrame, _PATCH_ATTR):
    _orig_wc = _plf.LazyFrame.with_columns
    setattr(_plf.LazyFrame, _PATCH_ATTR, _orig_wc)

    def _patched_wc(self, *exprs, **named):  # type: ignore[no-untyped-def]
        result = _orig_wc(self, *exprs, **named)
        try:
            flat: list[pl.Expr] = [e for e in exprs if isinstance(e, pl.Expr)]
            flat += [e.alias(n) for n, e in named.items() if isinstance(e, pl.Expr)]
            if flat:
                _dtw_lf_exprs_cache[id(result)] = flat
        except Exception:
            pass
        return result

    _plf.LazyFrame.with_columns = _patched_wc  # type: ignore[method-assign]


@dataclass(frozen=True)
class DtwBinding:
    out_name: str
    query_col: str
    handle: int


def _struct_fields(expr_json: dict) -> list:
    fn = expr_json.get("Function")
    if isinstance(fn, dict):
        inp = fn.get("input")
        if isinstance(inp, list):
            return inp
    return []


def _alias_name(node) -> str | None:
    if isinstance(node, dict):
        a = node.get("Alias")
        if isinstance(a, list) and len(a) == 2 and isinstance(a[1], str):
            return a[1]
    return None


def _literal_int(node) -> int | None:
    if isinstance(node, dict):
        a = node.get("Alias")
        if isinstance(a, list) and len(a) == 2 and isinstance(a[0], dict):
            lit = a[0].get("Literal")
            if isinstance(lit, dict):
                scalar = lit.get("Scalar")
                if isinstance(scalar, dict):
                    for key in ("Int64", "Int32", "Int"):
                        v = scalar.get(key)
                        if isinstance(v, int):
                            return v
                for key in ("Int64", "Int32", "Int"):
                    v = lit.get(key)
                    if isinstance(v, int):
                        return v
            if isinstance(lit, int):
                return lit
    return None


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
        cached = _dtw_lf_exprs_cache.pop(id(lf), None)
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
