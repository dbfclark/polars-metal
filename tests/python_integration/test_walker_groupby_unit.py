"""Unit-level tests for `_walk_group_by`: dispatch from `_walk_at_current`
and the shape of the emitted plan dict.

These tests inject a ``post_opt_callback`` directly via Polars' internal
hook, bypassing the MetalEngine collect wrapper. The callback invokes the
walker and captures the resulting plan dict (or fallback reason) without
installing a UDF — the CPU executor runs the query. This lets us assert on
the exact plan structure the walker emits without needing a real GPU kernel.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from polars_metal._walker import FallBack, Handled, walk


def _capture_plan(lf: pl.LazyFrame) -> tuple[dict | None, str | None]:
    """Collect with a shim that invokes the walker and captures its output.

    Uses Polars' ``post_opt_callback`` internal hook so the CPU executor
    runs the query correctly even when the walker returns a plan (no UDF
    installed). Returns ``(plan_dict, fallback_reason)`` — exactly one of the
    two will be non-None for any given query.
    """
    captured: dict[str, Any] = {"plan": None, "fallback": None}

    def shim(nt: Any, dur: Any) -> None:
        try:
            result = walk(nt)
        except Exception as e:
            captured["fallback"] = f"exception: {e!r}"
            return
        if isinstance(result, FallBack):
            captured["fallback"] = result.reason
            return
        assert isinstance(result, Handled)
        captured["plan"] = result.plan
        # Do not install a UDF — CPU executes the query.

    # post_opt_callback is an internal Polars bypass that injects a callback
    # without routing through the GPU engine machinery.
    lf.collect(engine="cpu", post_opt_callback=shim)
    return captured["plan"], captured["fallback"]


def test_groupby_single_i64_key_sum_emits_plan() -> None:
    df = pl.DataFrame({"k": [1, 1, 2, 2, 3], "v": [10, 20, 30, 40, 50]})
    plan, fallback = _capture_plan(df.lazy().group_by("k").agg(pl.col("v").sum().alias("sum_v")))
    assert fallback is None, f"unexpected fallback: {fallback}"
    assert plan is not None
    assert plan["kind"] == "GroupBy"
    assert plan["keys"] == [["k", "I64"]]
    assert plan["aggs"] == [
        {"kind": "Simple", "input_col": "v", "op": "Sum", "output_alias": "sum_v"}
    ]
    assert plan["input"]["kind"] == "Scan"


def test_groupby_composite_key_two_i64_keys_emits_plan() -> None:
    df = pl.DataFrame({"a": [1, 1, 2], "b": [10, 20, 30], "v": [1.0, 2.0, 3.0]})
    plan, fallback = _capture_plan(df.lazy().group_by(["a", "b"]).agg(pl.col("v").sum().alias("s")))
    assert fallback is None, f"unexpected fallback: {fallback}"
    assert plan is not None
    assert plan["kind"] == "GroupBy"
    assert plan["keys"] == [["a", "I64"], ["b", "I64"]]
    assert plan["aggs"] == [{"kind": "Simple", "input_col": "v", "op": "Sum", "output_alias": "s"}]


def test_groupby_multiple_aggs_emits_all() -> None:
    df = pl.DataFrame({"k": [1, 1, 2], "v": [10.0, 20.0, 30.0]})
    plan, fallback = _capture_plan(
        df.lazy()
        .group_by("k")
        .agg(
            pl.col("v").sum().alias("s"),
            pl.col("v").mean().alias("m"),
            pl.col("v").min().alias("mn"),
            pl.col("v").max().alias("mx"),
            pl.col("v").count().alias("cnt"),
            pl.len().alias("rows"),
        )
    )
    assert fallback is None, f"unexpected fallback: {fallback}"
    assert plan is not None
    # Simple specs carry "op"; Length specs do not (their wire format is
    # {kind: "Length", output_alias: ...}).
    ops_seen = [a["op"] if a["kind"] == "Simple" else "Len" for a in plan["aggs"]]
    assert sorted(ops_seen) == sorted(["Sum", "Mean", "Min", "Max", "Count", "Len"])
    # Len becomes a Length spec — no input_col, no op fields.
    len_spec = next(a for a in plan["aggs"] if a["kind"] == "Length")
    assert "input_col" not in len_spec
    assert len_spec["output_alias"] == "rows"


def test_groupby_string_key_falls_back() -> None:
    df = pl.DataFrame({"k": ["a", "b", "a"], "v": [1, 2, 3]})
    plan, fallback = _capture_plan(df.lazy().group_by("k").agg(pl.col("v").sum()))
    assert plan is None
    assert fallback is not None
    assert "String" in fallback or "unsupported dtype" in fallback


def test_groupby_unsupported_agg_expression_falls_back() -> None:
    df = pl.DataFrame({"k": [1, 1, 2], "v": [1.0, 2.0, 3.0]})
    # ``abs()`` is a function call (not a BinaryExpr over Add/Sub/Mul/Div),
    # so even with M3 Phase 2's capability-G extractor it falls outside the
    # supported closed set.
    plan, fallback = _capture_plan(df.lazy().group_by("k").agg(pl.col("v").abs().sum().alias("s")))
    assert plan is None
    assert fallback is not None


def test_groupby_binary_expression_agg_emits_expression_kind() -> None:
    """``(pl.col("v") * 2).sum()`` lifts via capability G (M3 Phase 2).

    The walker emits an ``Expression``-kind AggSpec containing the recursive
    ``expr`` sub-tree, the outer ``op``, and the alias.
    """
    df = pl.DataFrame({"k": [1, 1, 2], "v": [1.0, 2.0, 3.0]})
    plan, fallback = _capture_plan(df.lazy().group_by("k").agg((pl.col("v") * 2).sum().alias("s")))
    assert fallback is None
    assert plan is not None
    aggs = plan["aggs"]
    assert len(aggs) == 1
    spec = aggs[0]
    assert spec["kind"] == "Expression"
    assert spec["op"] == "Sum"
    assert spec["output_alias"] == "s"
    expr = spec["expr"]
    assert expr["kind"] == "Binary"
    assert expr["op"] == "Mul"
    assert expr["lhs"] == {"kind": "Column", "name": "v"}
    # ``pl.col("v") * 2`` on an F64 column has its literal cast to F64.
    assert expr["rhs"]["kind"] in ("LiteralF64", "LiteralI64")
