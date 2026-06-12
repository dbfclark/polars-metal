"""M6: shared serialize-detect machinery for the .metal struct-sentinel verbs.

One copy of the JSON-walk helpers (were duplicated across vector/fft/dtw/corr
detect modules) and the with_columns-capture monkeypatch installer.  M7a adds
weakref get-not-pop so repeated collect() of the same LazyFrame stays on the
fast path instead of falling through to O(N) lf.serialize().
"""

from __future__ import annotations

import itertools
import json
import warnings
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

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


# --------------------------------------------------------------------------
# M7-A spine: one capture cache, one sentinel binding, one parser, one
# candidate-iteration scaffold, one sentinel-field builder. Replaces the
# 4-6 near-identical detect modules + 3 cache triplets + 3 sentinel builders.
#
# NOTE: the spine does NOT own the _raise_cpu guard. Each verb keeps its own
# _raise_cpu stub (verb-specific ComputeError message, pinned by match=
# patterns in test_vector_search/test_corr_engine/test_fft) and passes it to
# sentinel_fields(raise_fn=...). Keeps sentinels + messages byte-identical.
# --------------------------------------------------------------------------


class CaptureCache:
    """Handle -> by-reference spec registry shared by the capture-based
    .metal verbs (vector search, dtw, corr). Each verb instantiates its own
    cache so handle spaces stay isolated and specs stay typed. fft needs no
    cache (its op code is inlined in the sentinel literal)."""

    def __init__(self) -> None:
        self._specs: dict[int, Any] = {}
        self._counter = itertools.count(1)

    def capture(self, spec: Any) -> int:
        handle = next(self._counter)
        self._specs[handle] = spec
        return handle

    def get(self, handle: int) -> Any | None:
        return self._specs.get(handle)

    def evict(self, handle: int) -> None:
        self._specs.pop(handle, None)


@dataclass(frozen=True)
class SentinelBinding:
    """A detected struct-sentinel. ``col`` is the source column the tag was
    suffixed with (``""`` for corr's exact tag). ``payload`` is the Int64
    carried in the tagged literal: a cache handle (vector/dtw/corr) or an op
    code (fft)."""

    out_name: str
    col: str
    payload: int


def make_sentinel_parser(
    tag: str, *, exact: bool = False
) -> Callable[[dict, str], SentinelBinding | None]:
    """Return a parser ``(inner_json, out_name) -> SentinelBinding | None``.
    ``exact=False`` (default): the tag is a prefix and its suffix is the
    source column (vector/fft/dtw). ``exact=True``: the tag alias matches
    exactly and there is no source column (corr)."""

    def parse(inner_json: dict, out_name: str) -> SentinelBinding | None:
        try:
            if tag not in json.dumps(inner_json):
                return None
            col = ""
            payload: int | None = None
            for fld in _struct_fields(inner_json):
                alias = _alias_name(fld)
                if exact:
                    if alias == tag:
                        payload = _literal_int(fld)
                elif alias and alias.startswith(tag):
                    col = alias[len(tag) :]
                    payload = _literal_int(fld)
            if payload is None or (not exact and not col):
                return None
            return SentinelBinding(out_name=out_name, col=col, payload=payload)
        except Exception:
            return None

    return parse


def iter_candidate_nodes(
    lf: pl.LazyFrame, *, cache: dict, explain_tags: tuple[str, ...]
) -> Iterator[tuple[dict, str]]:
    """Yield ``(inner_expr_json, out_name)`` for each top-level expression in
    ``lf`` that might carry a sentinel or native marker. Fast path: serialize
    each expr captured by the verb's with_columns monkey-patch (``cache``).
    Slow fallback: ``explain()``-pre-filter on ``explain_tags`` then parse the
    last ``"exprs":[...]`` fragment of the serialized plan. Any error -> stop
    (yields nothing further)."""
    try:
        cached = lookup(cache, lf)
        if cached is not None:
            for expr in cached:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    j = json.loads(expr.meta.serialize(format="json"))
                name = _alias_name(j)
                yield (j["Alias"][0] if name else j, name or "")
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            if not any(t in lf.explain() for t in explain_tags):
                return
            plan = lf.serialize(format="json")
        key = '"exprs":['
        i = plan.rfind(key)
        if i == -1:
            return
        start = i + len(key) - 1
        j = plan.rfind(',"options":', start)
        frag = plan[start:j] if j != -1 else plan[start:]
        nodes = json.loads(frag)
        for node in nodes if isinstance(nodes, list) else []:
            name = _alias_name(node)
            yield (node["Alias"][0] if name else node, name or "")
    except Exception:
        return


def sentinel_fields(
    expr: pl.Expr,
    *,
    tag: str,
    payload: int,
    raise_alias: str,
    raise_fn: Callable,
    col: str = "",
    in_alias: str | None = None,
    tag_exact: bool = False,
    raise_expr: pl.Expr | None = None,
) -> list[pl.Expr]:
    """Build the struct field list every sentinel shares: an optional
    pass-through input field, the tagged Int64 payload literal, and the guard
    field (``raise_fn`` is the verb's own ``_raise_cpu`` stub, kept per-verb
    so its ComputeError message stays unchanged). Preserves the exact
    aliases/order of the pre-spine builders so serialized plans are
    byte-identical."""
    fields: list[pl.Expr] = []
    if in_alias is not None:
        fields.append(expr.alias(in_alias))
    fields.append(pl.lit(payload, dtype=pl.Int64).alias(tag if tag_exact else f"{tag}{col}"))
    raise_src = raise_expr if raise_expr is not None else expr
    fields.append(raise_src.map_batches(raise_fn, return_dtype=pl.Float32).alias(raise_alias))
    return fields
