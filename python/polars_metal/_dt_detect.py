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

M7 A-2: dt keeps its rich, schema-validated DtBinding (not the generic
SentinelBinding) -- it shares only the candidate-iteration scaffold from
_detect_common.iter_candidate_nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from polars_metal import _detect_common as dc

_TEMPORAL_FN_MAP = {"Year": "year", "Month": "month", "Day": "day"}

# Slow-path pre-filter tags (appear in lf.explain()).
_DT_EXPLAIN_TAGS = (".dt.year(", ".dt.month(", ".dt.day(")

_UNITS_PER_DAY = {
    "ms": 86_400_000,
    "us": 86_400_000_000,
    "ns": 86_400_000_000_000,
}

# -- with_columns expression capture (independent patch + cache) --------------
# M7a: use shared weakref get-not-pop cache via _detect_common.
_dt_lf_exprs_cache: dict = {}
_PATCH_ATTR = "_polars_metal_dt_original_with_columns"
dc.install_with_columns_capture(_PATCH_ATTR, _dt_lf_exprs_cache)


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


def _parse_dt_node(inner_json: dict, out_name: str, schema: dict) -> DtBinding | None:
    """Bespoke per-node parser: schema-validate inner_json and produce a complete
    DtBinding (with out_name resolved). Returns None on any rejection. Never raises.

    dt keeps its rich, schema-validated DtBinding (not the generic SentinelBinding)
    -- it shares only the candidate-iteration scaffold."""
    try:
        b = _parse_dt_expr(inner_json, schema)
        if b is None:
            return None
        resolved = out_name or b.column
        if not resolved:
            return None
        return DtBinding(b.field, b.column, resolved, b.is_date, b.units_per_day)
    except Exception:
        return None


def find_dt_bindings(lf: pl.LazyFrame) -> list[DtBinding]:
    """Return handleable dt.year/month/day bindings in the outermost
    with_columns layer. Never raises (returns [] on any failure)."""
    try:
        schema = None
        out: list[DtBinding] = []
        for inner, name in dc.iter_candidate_nodes(
            lf, cache=_dt_lf_exprs_cache, explain_tags=_DT_EXPLAIN_TAGS
        ):
            if schema is None:
                schema = dict(lf.collect_schema())
            b = _parse_dt_node(inner, name, schema)
            if b is not None and b.out_name:
                out.append(b)

        if not out:
            return []
        sources = {b.column for b in out}
        if any(b.out_name in sources for b in out):
            return []
        return out
    except Exception:
        return []
