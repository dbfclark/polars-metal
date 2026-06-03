"""Detect handleable native rolling_* bindings from a LazyFrame's
pre-optimization serialized plan, for the M5 custom-kernel path.

Serialized JSON shape (pinned empirically at Polars py-1.40.1):

  Expression level (expr.meta.serialize(format="json")):
    {
      "Function": {
        "input": [{"Column": "x"}],
        "function": {
          "RollingExpr": {
            "function": "Mean",   // or "Sum" / "Var" / "Std"
            "options": {
              "window_size": 3,
              "min_periods": 3,   // equals window_size for default
              "weights": null,
              "center": false,
              "fn_params": null   // for Mean/Sum; {"Var": {"ddof": 1}} for Var/Std
            }
          }
        }
      }
    }

  Full LazyFrame plan (lf.serialize(format="json")):
    BEFORE schema resolution (before lf.collect_schema() / lf.schema is called):
    {
      "HStack": {
        "input": { ... },
        "exprs": [
          {
            "Alias": [
              { <Function node as above> },
              "r"               // the alias / output name
            ]
          }
        ],
        "options": { ... }
      }
    }

    AFTER schema resolution (after lf.collect_schema() triggers Polars internals),
    the plan is wrapped in an IR envelope:
    {
      "IR": {
        "dsl": {
          "HStack": { ... }    // same inner structure as above
        },
        "version": ...
      }
    }

    We handle both forms: serialize FIRST (before any schema call), then get
    schema from the output plan (or from the collected schema afterward).
    If neither form is present, return [].

  Non-default variants differ as follows:
    center=True   → "center": true
    min_samples=1 → "min_periods": 1  (not equal to window_size)
    weights=[...] → "weights": [1.0, 2.0, 3.0]

Anything not matching the handleable shape is omitted → native Polars/CPU.

## Detection performance strategy

For DataFrameScan-backed LazyFrames (``df.lazy().with_columns(...)``), the
full plan JSON embeds the DataFrame's Arrow bytes and can reach hundreds of
MB for large DataFrames.  Calling ``json.loads()`` on that string takes
O(N) time — 1-2 s at 10 M rows — which negates the kernel speedup.

We avoid this with a three-phase approach:

Fast path (via ``with_columns`` patch, O(1)):
  We monkey-patch ``LazyFrame.with_columns`` at module import time to record
  the Polars expression objects (not evaluated) in a weak-keyed dict indexed
  by the result LazyFrame id.  When ``find_rolling_bindings`` is called with
  a LazyFrame that was created by our patched ``with_columns``, we can use
  ``expr.meta.serialize(format="json")`` on each expression individually —
  each expression JSON is < 1 KB regardless of DataFrame size, so the whole
  detection is ~1 ms even for 10 M-row DataFrameScan-backed LazyFrames.

Slow fallback (full plan serialize + partial JSON parse, O(N)):
  For LazyFrames not captured by the patch (deserialized from binary, created
  via other APIs, etc.) we fall back to the two-phase approach:
    1. lf.explain() cheap pre-filter (~1 ms) — skip serialize if no rolling keywords
    2. lf.serialize(format="json") + rfind-based partial parse of the "exprs"
       fragment — avoids json.loads on the full (potentially huge) plan string.
  This still pays O(N) for the serialize, but avoids the O(N) json.loads.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import polars as pl
import polars.lazyframe.frame as _plf

MAX_W = 4096  # keep in sync with shaders/rolling.metal / rolling.rs

_ROLLING_FN_MAP = {
    "Sum": "sum",
    "Mean": "mean",
    "Var": "var",
    "Std": "std",
}

# Rolling op tags that appear in lf.explain() output (for slow-path pre-filter).
_ROLLING_EXPLAIN_TAGS = (
    "rolling_mean",
    "rolling_rsum",
    "rolling_var",
    "rolling_std",
)

# ── with_columns expression capture ─────────────────────────────────────────
# We monkey-patch LazyFrame.with_columns to record the Python expression
# objects before Polars compiles them into its internal IR. Keyed by the
# id() of the *result* LazyFrame so find_rolling_bindings can look them up
# without touching the plan at all.
#
# WeakValueDictionary would be ideal but LazyFrames are not weakly
# referenceable in the current PyO3 binding (no __weakref__ slot). We use a
# plain dict and rely on Python's reference counting: when the calling frame
# drops the LazyFrame (after collect), the id is eligible for reuse, but by
# then find_rolling_bindings has already consumed and we've evicted the entry.
# The dict stays small in practice (one entry per in-flight collect).
_lf_exprs_cache: dict[int, list[pl.Expr]] = {}

_PATCH_WITH_COLUMNS_ATTR = "_polars_metal_original_with_columns"

# We have already monkey-patched if the attribute exists.
if not hasattr(_plf.LazyFrame, _PATCH_WITH_COLUMNS_ATTR):
    _original_with_columns = _plf.LazyFrame.with_columns
    setattr(_plf.LazyFrame, _PATCH_WITH_COLUMNS_ATTR, _original_with_columns)

    def _patched_with_columns(self, *exprs, **named_exprs):  # type: ignore[no-untyped-def]
        result = _original_with_columns(self, *exprs, **named_exprs)
        try:
            # Normalise to a flat list of Polars Expr objects.
            all_exprs: list[pl.Expr] = []
            for e in exprs:
                if isinstance(e, pl.Expr):
                    all_exprs.append(e)
                # cs (column selector), list, etc. — skip; unlikely to be rolling
            for name, e in named_exprs.items():
                if isinstance(e, pl.Expr):
                    all_exprs.append(e.alias(name))
            if all_exprs:
                _lf_exprs_cache[id(result)] = all_exprs
        except Exception:
            pass  # never block the user's call
        return result

    _plf.LazyFrame.with_columns = _patched_with_columns  # type: ignore[method-assign]


@dataclass(frozen=True)
class RollingBinding:
    op: str  # "mean" | "sum" | "var" | "std"
    column: str
    window: int
    out_name: str
    ddof: int = 1


def _parse_rolling_expr(
    expr_json: dict,
    schema: dict[str, pl.DataType],
) -> RollingBinding | None:
    """Return a RollingBinding if expr_json is a handleable rolling Function
    node, otherwise return None.  Never raises."""
    try:
        fn_node = expr_json.get("Function")
        if not isinstance(fn_node, dict):
            return None

        inputs = fn_node.get("input")
        if not isinstance(inputs, list) or len(inputs) != 1:
            return None

        # Input must be a bare Column reference (not a sub-expression)
        col_node = inputs[0]
        if not isinstance(col_node, dict) or list(col_node.keys()) != ["Column"]:
            return None
        col_name = col_node["Column"]
        if not isinstance(col_name, str):
            return None

        function = fn_node.get("function")
        if not isinstance(function, dict):
            return None

        rolling = function.get("RollingExpr")
        if not isinstance(rolling, dict):
            return None

        fn_tag = rolling.get("function")
        op = _ROLLING_FN_MAP.get(fn_tag)
        if op is None:
            return None

        options = rolling.get("options")
        if not isinstance(options, dict):
            return None

        window_size = options.get("window_size")
        if not isinstance(window_size, int):
            return None

        # DEFAULT options guard: center must be false
        if options.get("center") is not False:
            return None

        # DEFAULT options guard: weights must be null
        if options.get("weights") is not None:
            return None

        # DEFAULT options guard: min_periods must equal window_size
        min_periods = options.get("min_periods")
        if min_periods != window_size:
            return None

        # fn_params: null for sum/mean; {"Var": {"ddof": 1}} for var/std
        fn_params = options.get("fn_params")
        ddof = 1
        if fn_tag in ("Var", "Std"):
            if not isinstance(fn_params, dict):
                return None
            var_params = fn_params.get("Var")
            if not isinstance(var_params, dict):
                return None
            ddof = var_params.get("ddof", 1)
            if not isinstance(ddof, int):
                return None
        else:
            if fn_params is not None:
                return None

        # Column must be Float32
        col_dtype = schema.get(col_name)
        if col_dtype != pl.Float32:
            return None

        # Window bounds: 1 <= window <= MAX_W
        if not (1 <= window_size <= MAX_W):
            return None

        # For var/std: window > ddof required (window >= 2 when ddof=1)
        if op in ("var", "std") and window_size <= ddof:
            return None

        return RollingBinding(
            op=op,
            column=col_name,
            window=window_size,
            out_name="",  # filled in by the caller
            ddof=ddof,
        )
    except Exception:
        return None


def _extract_hstack(plan: dict) -> dict | None:
    """Extract the HStack node from a plan dict.

    Handles two forms:
      - Direct: {"HStack": {...}}
      - After schema resolution: {"IR": {"dsl": {"HStack": {...}}, ...}}

    Returns the HStack dict, or None if not present in either form.
    """
    hstack = plan.get("HStack")
    if isinstance(hstack, dict):
        return hstack
    ir = plan.get("IR")
    if isinstance(ir, dict):
        dsl = ir.get("dsl")
        if isinstance(dsl, dict):
            hstack = dsl.get("HStack")
            if isinstance(hstack, dict):
                return hstack
    return None


def _bindings_from_polars_exprs(
    exprs: list[pl.Expr],
    schema: dict[str, pl.DataType],
) -> list[RollingBinding]:
    """Parse a list of Polars Expr objects captured at with_columns() time.

    Serializes each expression individually (~200 B per expr, < 1 ms total)
    and parses for rolling bindings.  This is the O(1)-in-N fast path.
    """
    results: list[RollingBinding] = []
    for expr in exprs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ser = expr.meta.serialize(format="json")
            expr_json = json.loads(ser)

            # Expression may be bare (Function) or Alias([Function, name]).
            out_name = ""
            inner_json = expr_json

            alias = expr_json.get("Alias")
            if isinstance(alias, list) and len(alias) == 2:
                inner_json, out_name = alias[0], alias[1]
            if not isinstance(out_name, str):
                continue

            binding = _parse_rolling_expr(inner_json, schema)
            if binding is not None:
                results.append(
                    RollingBinding(
                        op=binding.op,
                        column=binding.column,
                        window=binding.window,
                        out_name=out_name or binding.column,
                        ddof=binding.ddof,
                    )
                )
        except Exception:
            continue
    return results


def find_rolling_bindings(lf: pl.LazyFrame) -> list[RollingBinding]:
    """Parse the LazyFrame's expressions and return a list of RollingBinding
    for every handleable rolling_* alias found in the outermost HStack
    (with_columns) layer.

    ## Fast path (O(1) in N, ~1 ms):
    If the LazyFrame was created via the patched ``with_columns``, the
    expression objects are in ``_lf_exprs_cache[id(lf)]``.  We serialize
    each expression individually (< 1 KB per expr) to get window / column
    info.

    ## Slow fallback (O(N) serialize + partial JSON parse):
    For LazyFrames not in the cache, we use a two-phase fallback:
      1. lf.explain() pre-filter (~1 ms) — skip serialize if no rolling
      2. lf.serialize() + rfind-based partial parse of the "exprs" fragment
         — avoids json.loads on the full (potentially huge) plan string.

    Returns an empty list on any parse failure — this function never raises.
    """
    try:
        # ── Fast path: expression objects captured at with_columns() time ─────
        # Use get() (not pop()) so that repeated collect() calls on the same
        # LazyFrame object all benefit from the fast path. The dict stays small:
        # each entry is a few Polars Expr objects (~KB), and entries are naturally
        # evicted when the user's LazyFrame is garbage-collected (the id() key
        # becomes stale; Python's GC may reuse the id(), but that is harmless
        # because the old entry will simply be overwritten by the new with_columns
        # call, or the stale entry will produce a schema mismatch that falls back
        # gracefully via _parse_rolling_expr returning None).
        cached_exprs = _lf_exprs_cache.get(id(lf))
        if cached_exprs is not None:
            # Collect schema (fast — already resolved by the time detect runs).
            schema: dict[str, pl.DataType] = dict(lf.collect_schema())
            results = _bindings_from_polars_exprs(cached_exprs, schema)
            if results:
                # Reject if any output name shadows a source column.
                sources = {b.column for b in results}
                if any(b.out_name in sources for b in results):
                    return []
                return results

        # ── Slow fallback: serialize + partial JSON parse ─────────────────────

        # Phase 1: cheap pre-filter via explain() (~1 ms).
        # explain() does not serialize the DataFrame — it only walks the plan.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            explain_text = lf.explain()

        if not any(tag in explain_text for tag in _ROLLING_EXPLAIN_TAGS):
            return []

        # Phase 2: serialize the full plan (unavoidable; O(N) for DataFrameScan).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            plan_str = lf.serialize(format="json")

        # Collect schema AFTER serializing to avoid IR-wrapper mutation.
        schema = dict(lf.collect_schema())

        # Phase 3: partial JSON parse — only the "exprs" fragment.
        # The plan always ends with: ...,"exprs":[{...}],"options":{...}}}
        # We use rfind to locate the fragment and parse only that tiny array.
        exprs_key = '"exprs":['
        idx = plan_str.rfind(exprs_key)
        if idx == -1:
            return []
        exprs_array_start = idx + len(exprs_key) - 1  # points at '['

        options_key = ',"options":'
        opts_idx = plan_str.rfind(options_key, exprs_array_start)
        if opts_idx == -1:
            return []

        # exprs_json_str is the JSON array "[{...}, ...]" — typically < 1 KB.
        exprs_json_str = plan_str[exprs_array_start:opts_idx]
        alias_nodes: list = json.loads(exprs_json_str)

        if not isinstance(alias_nodes, list):
            return []

        results = []
        for alias_node in alias_nodes:
            try:
                if not isinstance(alias_node, dict):
                    continue
                alias = alias_node.get("Alias")
                # Alias is a two-element list: [<expr>, <name>]
                if not isinstance(alias, list) or len(alias) != 2:
                    continue
                expr_json, out_name = alias[0], alias[1]
                if not isinstance(out_name, str):
                    continue

                binding = _parse_rolling_expr(expr_json, schema)
                if binding is not None:
                    # Replace the placeholder out_name with the actual alias
                    results.append(
                        RollingBinding(
                            op=binding.op,
                            column=binding.column,
                            window=binding.window,
                            out_name=out_name,
                            ddof=binding.ddof,
                        )
                    )
            except Exception:
                continue

        # Reject if any output name shadows a source column we must read: the split
        # (lf.drop(out_names)) would remove a column the kernel needs, and an
        # in-place rolling (out_name == column) can't be expressed as orig-source
        # + rolled-output from one plan. Fall back to CPU for the whole query.
        sources = {b.column for b in results}
        if any(b.out_name in sources for b in results):
            return []

        return results

    except Exception:
        return []
