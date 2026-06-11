"""M6: shared serialize-detect machinery for the .metal struct-sentinel verbs.

One copy of the JSON-walk helpers (were duplicated across vector/fft/dtw/corr
detect modules) and the with_columns-capture monkeypatch installer.  M7a adds
weakref get-not-pop so repeated collect() of the same LazyFrame stays on the
fast path instead of falling through to O(N) lf.serialize().
"""

from __future__ import annotations

import weakref

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


def _make_evictor(cache: dict, key: int):
    """Return a weakref finalizer that removes `key` from `cache` when the
    referent is garbage-collected.  Keeps cache growth bounded and prevents
    a reused id() from returning stale exprs."""

    def _evict(_ref) -> None:
        cache.pop(key, None)

    return _evict


def lookup(cache: dict, lf) -> list | None:
    """Return captured exprs for `lf` WITHOUT removing them (so a repeated collect
    stays on the fast path).  Identity-validates via the stored weakref to reject
    the rare id-reuse case (returns None -> caller takes the slow serialize fallback)."""
    entry = cache.get(id(lf))
    if entry is None:
        return None
    ref, exprs = entry
    if ref() is not lf:
        return None
    return exprs


def install_with_columns_capture(attr: str, cache: dict) -> None:
    """Idempotently install a with_columns wrapper recording each call's exprs
    into `cache` keyed by id(result) as `(weakref.ref(result), exprs)`.  Get-not-pop
    via lookup() + weakref eviction means a repeated collect() of the same lf stays
    on the fast path, growth stays bounded, and a reused id can't return stale exprs.
    Chains with other installs (wraps whichever with_columns is current).
    No-op if `attr` already installed."""
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
                key = id(result)
                cache[key] = (weakref.ref(result, _make_evictor(cache, key)), flat)
        except Exception:
            pass
        return result

    _plf.LazyFrame.with_columns = _patched  # type: ignore[method-assign]
