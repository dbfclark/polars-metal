"""M6 vector search: detect sentinel bindings from a LazyFrame's outermost with_columns layer.

Reuses the M5 detection strategy:
  - Fast path: a `with_columns` monkey-patch records the Python Expr objects keyed by result
    LazyFrame id(); we serialize each expr individually (tiny) and look for our SENTINEL_TAG.
  - Slow fallback: lf.explain() pre-filter, then a bounded parse for the tag.

We never json.loads the full plan (it embeds the DataFrame at scale — the M5 gotcha).

## Coexistence with M5 rolling

`_rolling_detect` ALSO monkey-patches `LazyFrame.with_columns` (its own cache, keyed by
id(result), popped at its own find_rolling_bindings). We install a SEPARATE patch with a
SEPARATE attr (`_polars_metal_vs_original_with_columns`) and a SEPARATE cache; the two patches
chain safely (ours wraps whichever was installed first). Both pop their own caches, so neither
empties the other's.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import polars as pl
import polars.lazyframe.frame as _plf

from polars_metal._vector_namespace import SENTINEL_TAG

# id(result LazyFrame) → captured Expr objects (fast path). Evicted on consume (pop).
_lf_exprs_cache: dict[int, list[pl.Expr]] = {}
_PATCH_ATTR = "_polars_metal_vs_original_with_columns"

if not hasattr(_plf.LazyFrame, _PATCH_ATTR):
    _orig_wc = _plf.LazyFrame.with_columns
    setattr(_plf.LazyFrame, _PATCH_ATTR, _orig_wc)

    def _patched_wc(self, *exprs, **named):  # type: ignore[no-untyped-def]
        result = _orig_wc(self, *exprs, **named)
        try:
            flat: list[pl.Expr] = [e for e in exprs if isinstance(e, pl.Expr)]
            flat += [e.alias(n) for n, e in named.items() if isinstance(e, pl.Expr)]
            if flat:
                _lf_exprs_cache[id(result)] = flat
        except Exception:
            pass
        return result

    _plf.LazyFrame.with_columns = _patched_wc  # type: ignore[method-assign]


@dataclass(frozen=True)
class VectorBinding:
    out_name: str
    query_col: str
    handle: int


def _struct_fields(expr_json: dict) -> list:
    """Return the list of field-expr nodes of an as_struct Function, else []."""
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
    """Extract the Int64 handle from an Alias([Literal, name]) node.

    CONFIRMED at py-1.40.1 (Phase 2): the shape is
        {"Literal": {"Scalar": {"Int64": <value>}}}
    i.e. value at node["Alias"][0]["Literal"]["Scalar"]["Int64"]. We match that
    primarily, with a couple of legacy/fallback shapes for resilience.
    """
    if isinstance(node, dict):
        a = node.get("Alias")
        if isinstance(a, list) and len(a) == 2 and isinstance(a[0], dict):
            lit = a[0].get("Literal")
            if isinstance(lit, dict):
                # Primary (py-1.40.1): {"Scalar": {"Int64": N}}
                scalar = lit.get("Scalar")
                if isinstance(scalar, dict):
                    for key in ("Int64", "Int32", "Int"):
                        v = scalar.get(key)
                        if isinstance(v, int):
                            return v
                # Fallbacks for other Polars revs.
                for key in ("Int64", "Int32", "Int"):
                    v = lit.get(key)
                    if isinstance(v, int):
                        return v
                    if isinstance(v, dict) and isinstance(v.get("Int"), int):
                        return v["Int"]
            if isinstance(lit, int):
                return lit
    return None


def _binding_from_expr_json(expr_json: dict, out_name: str) -> VectorBinding | None:
    """Find the SENTINEL_TAG literal + query column inside a serialized struct expr."""
    try:
        s = json.dumps(expr_json)
        if SENTINEL_TAG not in s:
            return None
        # The tag is the alias of the Int64 literal field: f"{SENTINEL_TAG}{query_col}".
        # Walk the as_struct field aliases to recover query_col + the literal handle value.
        fields = _struct_fields(expr_json)
        query_col = None
        handle = None
        for fld in fields:
            alias_name = _alias_name(fld)
            if alias_name and alias_name.startswith(SENTINEL_TAG):
                query_col = alias_name[len(SENTINEL_TAG):]
                handle = _literal_int(fld)
        if query_col is None or handle is None:
            return None
        return VectorBinding(out_name=out_name, query_col=query_col, handle=handle)
    except Exception:
        return None


def find_vector_bindings(lf: pl.LazyFrame) -> list[VectorBinding]:
    """Return VectorBinding for each sentinel alias in the outermost with_columns layer."""
    try:
        cached = _lf_exprs_cache.pop(id(lf), None)
        if cached is not None:
            out: list[VectorBinding] = []
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
            if SENTINEL_TAG not in lf.explain():
                return []
            plan = lf.serialize(format="json")
        # Bounded parse of the exprs fragment (same rfind trick as _rolling_detect).
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
