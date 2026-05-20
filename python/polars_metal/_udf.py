"""Polars UDF entry point. Polars invokes this with
``(with_columns, predicate, n_rows, should_time)`` once the optimized plan
has decided our subtree is responsible for producing a DataFrame.

In Task 7 (Phase 4) the UDF is a pure-Python interpreter over the plan
dict produced by ``_walker.walk()``. It does not yet dispatch to MLX or
custom MSL kernels — that is Task 8.

The plan-dict shape mirrors ``MetalPlanNode`` in
``crates/polars-metal-core/src/plan/mod.rs``:

- ``{"kind": "Scan", "columns": [(name, dtype_tag), ...], "df": <PyDataFrame>,
   "projection": list[str] | None}``
- ``{"kind": "Project", "input": <plan>, "columns": list[str]}``
- ``{"kind": "Filter", "input": <plan>, ...}``  (Phase 5+)

The "Scan" node owns the *underlying* DataFrame (un-projected); the
optional ``projection`` describes the order/subset Polars's optimizer
chose. The walker captures it from ``DataFrameScan.projection``.
"""

from __future__ import annotations

from typing import Any

import polars as pl


def build_udf(plan: dict) -> Any:
    """Return a callable suitable for ``nt.set_udf(...)``.

    The returned function matches the polars-mem-engine PythonScanSource::Cuda
    signature: ``(with_columns, predicate, n_rows, should_time)``. We ignore
    ``predicate`` (the optimizer hasn't pushed predicates into our subtree —
    we returned FallBack on DataFrameScan.selection) and ``n_rows`` (no
    slice pushdown handled yet).

    ``with_columns`` arrives as ``None`` because we never opt into
    column-projection pushdown; the walker handles projection internally.

    When ``should_time`` is true Polars expects a ``(df, timings)`` tuple.
    We don't measure kernel timings yet; emit an empty timing list.
    """

    def udf(
        with_columns: list[str] | None,
        predicate: Any,
        n_rows: int | None,
        should_time: bool,
    ) -> Any:
        df = _execute(plan)
        # Apply Polars-requested slice if any. Defensive: in Phase 4 the
        # optimizer should not push a slice into us, but if it does we honor
        # it rather than silently producing a too-large frame.
        if n_rows is not None:
            df = df.slice(0, n_rows)
        if should_time:
            return df, []
        return df

    return udf


def _execute(plan: dict) -> pl.DataFrame:
    kind = plan["kind"]
    if kind == "Scan":
        # Lift the raw PyDataFrame back into a polars.DataFrame, then apply
        # the optimizer-chosen projection (if any). We MUST NOT use
        # ``df.select(...)`` here: pl.DataFrame.select goes through
        # ``self.lazy().select(...).collect()``, which the Polars conformance
        # harness intercepts to re-route through MetalEngine — producing
        # unbounded recursion through this same UDF. We use the underlying
        # PyDataFrame.select (sync, in-place column reorder/subset) via the
        # standard ``df[[names]]`` indexing path, which dispatches to
        # ``_select_columns_by_name`` and stays out of the lazy plan.
        raw = pl.DataFrame._from_pydf(plan["df"])
        proj = plan["projection"]
        if proj is None:
            return raw
        return _select_columns(raw, proj)
    if kind == "Project":
        upstream = _execute(plan["input"])
        return _select_columns(upstream, plan["columns"])
    if kind == "Filter":
        raise NotImplementedError("Filter dispatch lands in M1 Phase 5+ (Tasks 14-15)")
    raise ValueError(f"unknown plan kind: {kind!r}")


def _select_columns(df: pl.DataFrame, names: list[str]) -> pl.DataFrame:
    """Re-order/subset columns without going through LazyFrame.collect.

    Goes directly to PyDataFrame.select to avoid re-entering MetalEngine via
    any LazyFrame.collect interceptor."""
    return pl.DataFrame._from_pydf(df._df.select(list(names)))
