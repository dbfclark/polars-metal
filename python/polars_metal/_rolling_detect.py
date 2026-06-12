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

M7 A-2: rolling keeps its rich, schema-validated RollingBinding (not the
generic SentinelBinding) -- it shares only the candidate-iteration scaffold
from _detect_common.iter_candidate_nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from polars_metal import _detect_common as dc

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
# M7a: use shared weakref get-not-pop cache via _detect_common.  A repeated
# collect() of the same LazyFrame stays on the fast path; growth is bounded by
# weakref eviction on GC.
_lf_exprs_cache: dict = {}

_PATCH_WITH_COLUMNS_ATTR = "_polars_metal_original_with_columns"
dc.install_with_columns_capture(_PATCH_WITH_COLUMNS_ATTR, _lf_exprs_cache)


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


def _parse_rolling_node(
    inner_json: dict,
    out_name: str,
    schema: dict[str, pl.DataType],
) -> RollingBinding | None:
    """Bespoke per-node parser: schema-validate inner_json and produce a complete
    RollingBinding (with out_name resolved). Returns None on any rejection. Never raises.

    rolling keeps its rich, schema-validated RollingBinding (not the generic
    SentinelBinding) -- it shares only the candidate-iteration scaffold."""
    try:
        binding = _parse_rolling_expr(inner_json, schema)
        if binding is None:
            return None
        resolved = out_name or binding.column
        return RollingBinding(
            op=binding.op,
            column=binding.column,
            window=binding.window,
            out_name=resolved,
            ddof=binding.ddof,
        )
    except Exception:
        return None


def find_rolling_bindings(lf: pl.LazyFrame) -> list[RollingBinding]:
    """Parse the LazyFrame's expressions and return a list of RollingBinding
    for every handleable rolling_* alias found in the outermost HStack
    (with_columns) layer.

    ## Fast path (O(1) in N, ~1 ms):
    If the LazyFrame was created via the patched ``with_columns``, the
    expression objects are retrieved via ``dc.lookup(_lf_exprs_cache, lf)``.  We serialize
    each expression individually (< 1 KB per expr) to get window / column
    info.

    ## Slow fallback (O(N) serialize + partial JSON parse):
    For LazyFrames not in the cache, we use a two-phase fallback:
      1. lf.explain() pre-filter (~1 ms) — skip serialize if no rolling
      2. lf.serialize() + rfind-based partial parse of the "exprs" fragment
         — avoids json.loads on the full (potentially huge) plan string.

    Both paths are provided by dc.iter_candidate_nodes; this function
    applies the bespoke RollingBinding parser and the source-shadowing guard.

    Returns an empty list on any parse failure — this function never raises.
    """
    try:
        schema: dict[str, pl.DataType] | None = None
        results: list[RollingBinding] = []

        for inner, name in dc.iter_candidate_nodes(
            lf, cache=_lf_exprs_cache, explain_tags=_ROLLING_EXPLAIN_TAGS
        ):
            if schema is None:
                schema = dict(lf.collect_schema())
            binding = _parse_rolling_node(inner, name, schema)
            if binding is not None:
                results.append(binding)

        if not results:
            return []

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
