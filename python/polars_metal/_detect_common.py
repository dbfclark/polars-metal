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


# id(select-result lf) -> weakref(parent lf, pre-select). Lets the stitch
# dispatch recover a verb's SOURCE column when the `.select` projected only the
# sentinel output (so the source column is absent from the post-select frame but
# still resolvable from its parent). Keyed/evicted exactly like the expr cache.
_select_parents: dict[int, Any] = {}


def _parent_lf(lf) -> Any | None:
    """Return the pre-select parent of `lf` if `.select` capture recorded one,
    else None. The parent is held by a STRONG reference (the `df.lazy()` in a
    `df.lazy().select(...)` chain is a temporary with no other referent, so a
    weakref would die immediately); its lifetime is tied to the result lf via the
    same weakref evictor as the expr cache."""
    return _select_parents.get(id(lf))


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

    # NOTE: `.select(...)` is intentionally NOT captured here. Polars implements
    # eager `DataFrame.select` (and much internal machinery) as
    # `self.lazy().select(...).collect()`, so a global `LazyFrame.select`
    # monkey-patch fires on every internal select and pollutes the detection
    # cache (id(result)-keyed; transient lfs cause nondeterministic stale hits
    # — it flaked the `as_struct`/`value_counts` CSE conformance tests). The
    # `.metal` verbs are detected under `.select` purely via the serialize
    # slow path in `iter_candidate_nodes` (the `'"expr":['` Select fragment) +
    # `_reconstruct_parent`, which needs no global patch. See M11.


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


def _balanced_bracket_slice(s: str, start: int) -> str | None:
    """Return ``s[start:end]`` where ``s[start] == '['`` and ``end`` is one past
    the matching ``']'``, tracking bracket depth and skipping bracket chars that
    fall inside JSON string literals (with backslash escapes). Returns None if no
    balanced close is found. Used to extract a Select node's ``"expr":[...]`` list,
    whose tail is the scanned-subtree ``"input"`` rather than ``"options"``."""
    depth = 0
    in_str = False
    esc = False
    for k in range(start, len(s)):
        ch = s[k]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return s[start : k + 1]
    return None


def _reconstruct_parent(lf: pl.LazyFrame):
    """Recover the pre-select parent LazyFrame from *lf*'s serialized plan when
    no parent was captured (slow serialize path). The top node is a ``Select``
    whose ``input`` subtree is exactly the parent; deserialize that subtree back
    into a LazyFrame. Returns None on any error (caller degrades to the
    with_columns path)."""
    try:
        import io

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plan = json.loads(lf.serialize(format="json"))
        # The plan may be the bare DSL ({"Select": ...}) or wrapped in an IR
        # envelope ({"IR": {"dsl": {"Select": ...}, "version": ...}}) depending on
        # where in the collect path we serialize. Unwrap to the DSL node.
        if isinstance(plan, dict) and "IR" in plan:
            ir = plan["IR"]
            dsl = ir.get("dsl") if isinstance(ir, dict) else None
            node = dsl if isinstance(dsl, dict) else plan
        else:
            node = plan
        sel = node.get("Select") if isinstance(node, dict) else None
        if not isinstance(sel, dict) or "input" not in sel:
            return None
        sub = json.dumps(sel["input"])
        return pl.LazyFrame.deserialize(io.StringIO(sub), format="json")
    except Exception:
        return None


def collect_stitch_base(lf: pl.LazyFrame, out_names, src_cols, collect_fn) -> pl.DataFrame:
    """Collect the base DataFrame the .metal stitch dispatch stitches its GPU
    outputs onto. Handles both detection idioms:

    * ``with_columns`` (no recorded parent): ``lf.drop(out_names)`` — projection
      pushdown elides the sentinel computation (incl. the opaque ``_raise``
      map_batches) from the CPU path; the source columns survive in the result.

    * ``select`` (parent recorded by the capture patch): ``lf.drop(out_names)``
      would NOT elide ``_raise`` (the sentinel select DEFINES the frame), so we
      instead collect the non-sentinel OUTPUT columns directly from ``lf`` (those
      are ordinary, non-raising exprs) and recover any binding SOURCE columns from
      the pre-select parent.

    The returned frame contains every non-sentinel output column plus every needed
    source column (the latter dropped from the final result by the caller, which
    reorders to the schema's ``order``)."""
    parent = _parent_lf(lf)
    needed_src = [c for c in dict.fromkeys(src_cols) if c]
    if parent is None:
        # No captured parent. Distinguish the with_columns idiom (source columns
        # survive in lf's schema) from a select reached via the slow serialize
        # path (source columns absent → reconstruct the parent from the plan).
        try:
            lf_cols = set(lf.collect_schema().names())
        except Exception:
            lf_cols = set()
        if needed_src and not all(c in lf_cols for c in needed_src):
            recovered = _reconstruct_parent(lf)
            if recovered is not None:
                parent = recovered
    if parent is None:
        # with_columns idiom: drop sentinel outputs, source columns ride along.
        df = collect_fn(lf.drop(out_names))
        present = set(df.columns)
        missing = [c for c in needed_src if c not in present]
        if missing:
            src_df = collect_fn(lf.select([pl.col(c) for c in missing]))
            df = df.hstack([src_df.get_column(c) for c in missing])
        return df
    # select idiom: the sentinel select defines lf, so collect the non-sentinel
    # outputs from lf (safe — they don't raise) and the sources from the parent.
    out_set = set(out_names)
    rest_outputs = [c for c in lf.collect_schema().names() if c not in out_set]
    if rest_outputs:
        df = collect_fn(lf.select([pl.col(c) for c in rest_outputs]))
    else:
        # No non-sentinel outputs: an empty frame; the recovered source columns
        # below supply the row count.
        df = pl.DataFrame()
    present = set(df.columns)
    missing = [c for c in needed_src if c not in present]
    if missing:
        src_df = collect_fn(parent.select([pl.col(c) for c in missing]))
        df = src_df if df.width == 0 else df.hstack([src_df.get_column(c) for c in missing])
    return df


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
        if i != -1:
            # HStack/with_columns: the expr list is immediately followed by
            # ',"options":' — slice up to it (proven, kept unchanged).
            start = i + len(key) - 1
            j = plan.rfind(',"options":', start)
            frag = plan[start:j] if j != -1 else plan[start:]
        else:
            # Select node (projection idiom): the expr list is followed by
            # ',"input":{...}' (the scanned subtree), NOT ',"options":', so the
            # ',"options":' shortcut over-captures. Walk to the balanced ']'.
            key = '"expr":['
            i = plan.rfind(key)
            if i == -1:
                return
            start = i + len(key) - 1
            frag = _balanced_bracket_slice(plan, start)
            if frag is None:
                return
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
