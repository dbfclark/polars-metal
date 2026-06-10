"""M6 B3: detect native dt.year/month/day bindings from a LazyFrame's
outermost with_columns layer, for the gregorian custom-kernel path.

dt.* expressions are NodeTraverser-opaque, so (like rolling/FFT) we inspect
the pre-optimization serialized plan. Serialized expr shape (py-1.40.1):

  {"Function": {"input": [{"Column": "d"}],
                "function": {"TemporalExpr": "Year"}}}   // or "Month" / "Day"

wrapped as {"Alias": [<Function>, "out_name"]}. The time_unit (Datetime) is
NOT in the expr JSON -- it comes from the column's schema dtype.

Independent with_columns patch + cache (each detector pops only its own
cache) so this chains safely with the rolling / vector / fft patches.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import polars as pl
import polars.lazyframe.frame as _plf

_TEMPORAL_FN_MAP = {"Year": "year", "Month": "month", "Day": "day"}

# Slow-path pre-filter tags (appear in lf.explain()).
_DT_EXPLAIN_TAGS = (".dt.year(", ".dt.month(", ".dt.day(")

_UNITS_PER_DAY = {
    "ms": 86_400_000,
    "us": 86_400_000_000,
    "ns": 86_400_000_000_000,
}

# -- with_columns expression capture (independent patch + cache) --------------
_dt_lf_exprs_cache: dict[int, list[pl.Expr]] = {}
_PATCH_ATTR = "_polars_metal_dt_original_with_columns"

if not hasattr(_plf.LazyFrame, _PATCH_ATTR):
    _orig_wc = _plf.LazyFrame.with_columns
    setattr(_plf.LazyFrame, _PATCH_ATTR, _orig_wc)

    def _patched_wc(self, *exprs, **named):  # type: ignore[no-untyped-def]
        result = _orig_wc(self, *exprs, **named)
        try:
            flat: list[pl.Expr] = [e for e in exprs if isinstance(e, pl.Expr)]
            flat += [e.alias(n) for n, e in named.items() if isinstance(e, pl.Expr)]
            if flat:
                _dt_lf_exprs_cache[id(result)] = flat
        except Exception:
            pass
        return result

    _plf.LazyFrame.with_columns = _patched_wc  # type: ignore[method-assign]


@dataclass(frozen=True)
class DtBinding:
    field: str  # "year" | "month" | "day"
    column: str
    out_name: str
    is_date: bool  # True -> physical i32 days; False -> Datetime i64
    units_per_day: int | None  # None for Date; ms/us/ns count for Datetime


def _binding_for_column(field: str, col_name: str, col_dtype) -> DtBinding | None:
    """Build a DtBinding from a recognized field + the column's schema dtype,
    or None if the dtype is not a supported temporal type."""
    if col_dtype == pl.Date:
        return DtBinding(field, col_name, "", is_date=True, units_per_day=None)
    if isinstance(col_dtype, pl.Datetime):
        # Time-zone-aware datetimes -> CPU (wall-clock semantics differ).
        if getattr(col_dtype, "time_zone", None) is not None:
            return None
        upd = _UNITS_PER_DAY.get(col_dtype.time_unit)
        if upd is None:
            return None
        return DtBinding(field, col_name, "", is_date=False, units_per_day=upd)
    return None


def _parse_dt_expr(expr_json: dict, schema: dict) -> DtBinding | None:
    """Return a DtBinding if expr_json is a handleable TemporalExpr Function
    over a bare Column, else None. Never raises."""
    try:
        fn_node = expr_json.get("Function")
        if not isinstance(fn_node, dict):
            return None
        inputs = fn_node.get("input")
        if not isinstance(inputs, list) or len(inputs) != 1:
            return None
        col_node = inputs[0]
        if not isinstance(col_node, dict) or list(col_node.keys()) != ["Column"]:
            return None
        col_name = col_node["Column"]
        if not isinstance(col_name, str):
            return None
        function = fn_node.get("function")
        if not isinstance(function, dict):
            return None
        temporal = function.get("TemporalExpr")
        field = _TEMPORAL_FN_MAP.get(temporal) if isinstance(temporal, str) else None
        if field is None:
            return None
        return _binding_for_column(field, col_name, schema.get(col_name))
    except Exception:
        return None


def _bindings_from_polars_exprs(exprs: list[pl.Expr], schema: dict) -> list[DtBinding]:
    results: list[DtBinding] = []
    for expr in exprs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ser = expr.meta.serialize(format="json")
            expr_json = json.loads(ser)
            inner, out_name = expr_json, ""
            alias = expr_json.get("Alias")
            if isinstance(alias, list) and len(alias) == 2:
                inner, out_name = alias[0], alias[1]
            if not isinstance(out_name, str):
                continue
            b = _parse_dt_expr(inner, schema)
            if b is not None:
                results.append(
                    DtBinding(b.field, b.column, out_name or b.column, b.is_date, b.units_per_day)
                )
        except Exception:
            continue
    return results


def find_dt_bindings(lf: pl.LazyFrame) -> list[DtBinding]:
    """Return handleable dt.year/month/day bindings in the outermost
    with_columns layer. Never raises (returns [] on any failure)."""
    try:
        cached = _dt_lf_exprs_cache.pop(id(lf), None)
        if cached is not None:
            schema = dict(lf.collect_schema())
            results = _bindings_from_polars_exprs(cached, schema)
            if results:
                sources = {b.column for b in results}
                if any(b.out_name in sources for b in results):
                    return []
                return results

        # Slow fallback: explain() pre-filter, then serialize + bounded parse.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            explain_text = lf.explain()
        if not any(tag in explain_text for tag in _DT_EXPLAIN_TAGS):
            return []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            plan_str = lf.serialize(format="json")
        schema = dict(lf.collect_schema())

        exprs_key = '"exprs":['
        idx = plan_str.rfind(exprs_key)
        if idx == -1:
            return []
        start = idx + len(exprs_key) - 1
        opts_idx = plan_str.rfind(',"options":', start)
        if opts_idx == -1:
            return []
        alias_nodes = json.loads(plan_str[start:opts_idx])
        if not isinstance(alias_nodes, list):
            return []

        results = []
        for node in alias_nodes:
            try:
                if not isinstance(node, dict):
                    continue
                alias = node.get("Alias")
                if not isinstance(alias, list) or len(alias) != 2:
                    continue
                expr_json, out_name = alias[0], alias[1]
                if not isinstance(out_name, str):
                    continue
                b = _parse_dt_expr(expr_json, schema)
                if b is not None:
                    results.append(
                        DtBinding(b.field, b.column, out_name, b.is_date, b.units_per_day)
                    )
            except Exception:
                continue

        sources = {b.column for b in results}
        if any(b.out_name in sources for b in results):
            return []
        return results
    except Exception:
        return []
