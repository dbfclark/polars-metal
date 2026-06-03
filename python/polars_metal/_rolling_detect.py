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
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import polars as pl

MAX_W = 4096  # keep in sync with shaders/rolling.metal / rolling.rs

_ROLLING_FN_MAP = {
    "Sum": "sum",
    "Mean": "mean",
    "Var": "var",
    "Std": "std",
}


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


def find_rolling_bindings(lf: pl.LazyFrame) -> list[RollingBinding]:
    """Parse the LazyFrame's pre-optimization serialized plan and return a
    list of RollingBinding for every handleable rolling_* alias found in the
    outermost HStack (with_columns) layer.

    IMPORTANT: serialize MUST happen before schema resolution (collect_schema /
    .schema) to avoid Polars wrapping the plan in an IR envelope. We handle
    both forms via _extract_hstack, but collect the schema AFTER serializing.

    Returns an empty list on any parse failure — this function never raises.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            plan_str = lf.serialize(format="json")

        plan = json.loads(plan_str)

        # Collect schema AFTER serializing to avoid IR-wrapper mutation.
        schema: dict[str, pl.DataType] = dict(lf.collect_schema())

        hstack = _extract_hstack(plan)
        if hstack is None:
            return []

        exprs = hstack.get("exprs")
        if not isinstance(exprs, list):
            return []

        results: list[RollingBinding] = []
        for alias_node in exprs:
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
