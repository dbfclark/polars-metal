"""M6: shared serialize-detect machinery for the .metal struct-sentinel verbs.

One copy of the JSON-walk helpers (were duplicated across vector/fft/dtw/corr
detect modules) and the with_columns-capture monkeypatch installer. A later task
upgrades the cache here (weakref get-not-pop) so repeated collect() stays fast;
keeping it in one place is what makes that a one-line change.
"""

from __future__ import annotations

import polars as pl
import polars.lazyframe.frame as _plf


def _alias_name(node) -> str | None:
    if isinstance(node, dict):
        a = node.get("Alias")
        if isinstance(a, list) and len(a) == 2 and isinstance(a[1], str):
            return a[1]
    return None


def _struct_fields(expr_json: dict) -> list:
    """Return the list of field-expr nodes of an as_struct Function, else []."""
    fn = expr_json.get("Function")
    if isinstance(fn, dict):
        inp = fn.get("input")
        if isinstance(inp, list):
            return inp
    return []


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


def install_with_columns_capture(attr: str, cache: dict) -> None:
    """Idempotently install a with_columns wrapper recording each call's exprs
    into `cache` keyed by id(result). Chains with other installs (wraps whichever
    with_columns is current). No-op if `attr` already installed."""
    if hasattr(_plf.LazyFrame, attr):
        return
    orig = _plf.LazyFrame.with_columns
    setattr(_plf.LazyFrame, attr, orig)

    def _patched(self, *exprs, **named):  # type: ignore[no-untyped-def]
        result = orig(self, *exprs, **named)
        try:
            flat = [e for e in exprs if isinstance(e, pl.Expr)]
            flat += [e.alias(n) for n, e in named.items() if isinstance(e, pl.Expr)]
            if flat:
                cache[id(result)] = flat
        except Exception:
            pass
        return result

    _plf.LazyFrame.with_columns = _patched  # type: ignore[method-assign]
