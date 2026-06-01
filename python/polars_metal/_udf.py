"""Polars UDF entry point. Polars invokes this with
``(with_columns, predicate, n_rows, should_time)`` once the optimized plan
has decided our subtree is responsible for producing a DataFrame.

Task 8 (Phase 4) handles scan + project via Rust's ``_native.execute_plan``.
Task 15 (Phase 5) extends the UDF body to handle Filter at the plan root:
the Python side extracts Arrow buffers from the upstream DataFrame, hands
the raw bytes to Rust's ``_native.execute_filter_compact`` for GPU
compaction, then re-assembles a Polars DataFrame via
``pyarrow.Array.from_buffers``.

Why Python-side Arrow extraction?
---------------------------------
Polars + PyArrow + pyo3 interop is much simpler when the Arrow buffer
extraction stays in Python. ``Series.to_arrow().buffers()`` returns
``pyarrow.Buffer`` objects whose memory is directly addressable as
``bytes(buf)`` — no pyo3 ChunkedArray plumbing required. The same path
works in reverse for reassembly via ``pa.Array.from_buffers``.

The plan-dict shape produced by ``_walker.walk()`` mirrors
``MetalPlanNode`` in ``crates/polars-metal-core/src/plan/mod.rs`` *plus*
walker-only side-channel keys on the Scan leaf:

- ``{"kind": "Scan", "columns": [(name, dtype_tag), ...], "df": <PyDataFrame>,
   "projection": list[str] | None}``
- ``{"kind": "Project", "input": <plan>, "columns": list[str]}``
- ``{"kind": "Filter", "input": <plan>, "predicate": <pred>}`` where ``<pred>``
  is one of:
    - ``{"kind": "Column", "name": ..., "dtype": "Bool"}`` (Phase 5),
    - ``{"kind": "Compare", "op": "Eq|Ne|Lt|Le|Gt|Ge", "lhs": <leaf>,
       "rhs": <leaf>, "dtype": "I64"|"F64"}`` (Phase 6, Task 18),
    - ``{"kind": "And"|"Or", "lhs": <pred>, "rhs": <pred>}`` (Phase 7,
       Task 20) — combinators over any Bool-typed sub-predicate, nested
       arbitrarily.
  Leaves are ``Column`` (as above) or one of ``LiteralI64 | LiteralF64
  | LiteralBool`` with a ``value`` field.

``df`` and ``projection`` are walker-only side channels; we strip them
before handing the plan to Rust.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pyarrow as pa

from polars_metal import _native


def build_udf(plan: dict) -> Any:
    """Return a callable suitable for ``nt.set_udf(...)``.

    The returned function matches the polars-mem-engine PythonScanSource::Cuda
    signature: ``(with_columns, predicate, n_rows, should_time)``. We ignore
    ``predicate`` (the optimizer hasn't pushed predicates into our subtree —
    we returned FallBack on DataFrameScan.selection) and ``with_columns``
    arrives as ``None`` because we never opt into column-projection pushdown;
    the walker handles projection internally.

    When ``should_time`` is true Polars expects a ``(df, timings)`` tuple.
    We don't measure kernel timings yet; emit an empty timing list.
    """
    if plan["kind"] == "GroupBy":
        return _build_groupby(plan)
    if plan["kind"] == "Sort":
        return _build_sort(plan)
    df_pydf, wire_plan = _extract_scan_df_and_wire_plan(plan)

    def udf(
        with_columns: list[str] | None,
        predicate: Any,
        n_rows: int | None,
        should_time: bool,
    ) -> Any:
        df = _dispatch(df_pydf, wire_plan)
        # Apply Polars-requested slice if any. Defensive: in Phase 4 the
        # optimizer should not push a slice into us, but if it does we honor
        # it rather than silently producing a too-large frame.
        if n_rows is not None:
            df = df.slice(0, n_rows)
        if should_time:
            return df, []
        return df

    return udf


def _dispatch(df_pydf: Any, wire_plan: dict) -> pl.DataFrame:
    """Route the wire plan to the appropriate Rust entry point.

    Filter at the plan root requires the dedicated compaction path
    (raw Arrow bytes in, raw Arrow bytes out); everything else goes
    through the generic ``execute_plan`` entry point.

    Note: Filter is currently only recognised at the *root* of the wire
    plan. Phase 5 emits plans of the form ``Filter(input=...)`` or
    ``Project(input=Filter(...))`` — the latter is unwrapped here to
    apply the post-filter projection.
    """
    if wire_plan["kind"] == "Filter":
        return _dispatch_filter(df_pydf, wire_plan)
    if wire_plan["kind"] == "Project" and wire_plan["input"]["kind"] == "Filter":
        # Filter followed by a column re-selection: compact first, then
        # project on the host (cheap synchronous DataFrame op).
        filtered = _dispatch_filter(df_pydf, wire_plan["input"])
        # CRITICAL: do not call `filtered.select(...)`; pl.DataFrame.select
        # routes through LazyFrame.collect which our monkey-patch
        # intercepts, causing infinite recursion. The underlying
        # PyDataFrame.select is the sync escape hatch (see
        # docs/open-questions.md, Walker / UDF integration #1).
        return pl.DataFrame._from_pydf(filtered._df.select(list(wire_plan["columns"])))
    if wire_plan["kind"] == "Project" and wire_plan["input"]["kind"] == "HStack":
        # Project wrapping an HStack chain — common when the optimizer
        # adds an output-column reorder above CSE-introduced HStacks
        # (haversine, Black-Scholes, any with_columns over a multi-col
        # frame). Dispatch the HStack input, then select on the result.
        # Same select-via-PyDataFrame trick as Project(Filter) — see
        # the comment above for why we can't use pl.DataFrame.select.
        inner = _dispatch(df_pydf, wire_plan["input"])
        return pl.DataFrame._from_pydf(inner._df.select(list(wire_plan["columns"])))
    if wire_plan["kind"] == "HStack":
        # M4 Phase 5: if every appended column has a fused MLX subgraph from
        # the Phase 3 analyzer, dispatch the whole HStack via the fused path.
        # Bindings without `_fused_scope` (analyzer rejected them, or they
        # came from the M3 conformance path) cause the HStack to fall through
        # to the existing _native.execute_plan path. This is the all-or-
        # nothing MVP; partial-fusion mixed dispatch can land in a follow-up.
        exprs = wire_plan.get("exprs", [])
        if exprs and all("_fused_scope" in e for e in exprs):
            return _dispatch_hstack_fused(df_pydf, wire_plan)
    result_pydf = _native.execute_plan(df_pydf, wire_plan)
    return pl.DataFrame._from_pydf(result_pydf)


def _dispatch_hstack_fused(df_pydf: Any, wire_plan: dict) -> pl.DataFrame:
    """Execute an HStack whose every binding carries a `_fused_scope`
    side-channel. Recurses on the upstream input, then runs each binding
    through `execute_fused_expr` and stitches the results onto the frame.

    For each binding, the analyzer returned a `_fused_columns` list of
    descriptors: ``("col", name)`` for real columns or ``("lit", value)``
    for literal scalars. Column inputs are passed as F32-cast NumPy bytes
    (n_rows-wide). Literal inputs are passed as a single F32 (4 bytes);
    the executor builds a shape=[1] MLX array and MLX broadcasts in any
    elementwise op — no need to materialize n_rows*4 bytes per literal.
    """
    import numpy as np

    upstream = _dispatch(df_pydf, wire_plan["input"])
    n_rows = upstream.height
    new_columns: list[pl.Series] = []
    for binding in wire_plan["exprs"]:
        scope = binding["_fused_scope"]
        descriptors: list[tuple[str, str | float]] = binding["_fused_columns"]
        name = binding["name"]

        input_arrays: list[np.ndarray] = []
        for kind, payload in descriptors:
            if kind == "col":
                series = upstream.get_column(payload)
                # Pass the column as a float32 numpy view. For a dense,
                # single-chunk F32 column `to_numpy()` is zero-copy and already
                # C-contiguous F32, so `ascontiguousarray` is a no-op view; the
                # Rust ingest then takes the zero-copy bytesNoCopy path when the
                # buffer is page-aligned (numpy-origin / large allocations) and
                # a single copy otherwise. Columns with nulls / multiple chunks
                # copy here (correctness preserved), then ingest-copy.
                input_arrays.append(np.ascontiguousarray(series.to_numpy(), dtype=np.float32))
            elif kind == "lit":
                # Single scalar — executor builds a shape=[1] MLX array, which
                # broadcasts in elementwise ops. No n_rows-wide materialization.
                input_arrays.append(np.asarray([payload], dtype=np.float32))
            else:
                raise RuntimeError(f"polars_metal: unknown input descriptor {kind!r}")

        # Output-zero-copy: pre-allocate the result array and let the executor
        # write the MLX output directly into it (one MLX->numpy copy, no PyBytes
        # round-trip). `pl.Series` then wraps it without copying.
        out_arr = np.empty(n_rows, dtype=np.float32)
        # The buffer protocol isn't available under abi3, so hand the executor
        # raw (ptr, n_elements) pairs. `input_arrays` and `out_arr` are held in
        # these locals across the fully synchronous call, keeping the borrowed
        # memory alive (the executor's safety contract).
        inputs = [(int(a.__array_interface__["data"][0]), int(a.size)) for a in input_arrays]
        written = _native.execute_fused_expr(
            scope=scope,
            inputs=inputs,
            out=(int(out_arr.__array_interface__["data"][0]), int(out_arr.size)),
        )
        # Literal-only expressions produce a length-1 output (scalar broadcast
        # never widened by an n_rows-shaped column). Polars requires the new
        # column's length to match the frame height, so broadcast the single
        # written value across the pre-allocated array.
        if written == 1 and n_rows != 1:
            out_arr.fill(out_arr[0])

        # Null handling (walker stamped `_fused_null_mode` only when an input
        # column may have nulls). MLX computed over NaN-filled nulls; Polars
        # propagates nulls instead. We compute the output null mask and mark
        # those rows null — the NaN data underneath is ignored by Arrow, and
        # the value compute already ran on the GPU.
        null_mask = _fused_null_mask(binding, descriptors, upstream, out_arr, n_rows)
        if null_mask is not None and null_mask.any():
            new_columns.append(pl.Series(name, pa.array(out_arr, mask=null_mask)))
        else:
            new_columns.append(pl.Series(name, out_arr))

    return upstream.with_columns(new_columns)


def _fused_null_mask(
    binding: dict,
    descriptors: list[tuple[str, str | float]],
    upstream: pl.DataFrame,
    out_arr: Any,
    n_rows: int,
) -> Any:
    """Return the boolean output null mask (True = null) for a fused binding,
    or ``None`` when the walker stamped no `_fused_null_mode` (inputs confirmed
    null-free → fast path).

    - ``elementwise``: output null iff any input column is null at that row →
      OR the input columns' null masks.
    - ``where``: data-dependent — evaluate the walker's validity subgraph (one
      extra MLX dispatch) producing an F32 1.0/0.0 mask; null where < 0.5.
    """
    import numpy as np

    null_mode = binding.get("_fused_null_mode")
    if null_mode is None:
        return None

    if null_mode == "elementwise":
        null_mask = np.zeros(n_rows, dtype=bool)
        for kind, payload in descriptors:
            if kind == "col":
                null_mask |= upstream.get_column(payload).is_null().to_numpy()
        return null_mask

    if null_mode == "where":
        v_scope = binding["_fused_validity_scope"]
        v_descriptors: list[tuple[str, str | float]] = binding["_fused_validity_columns"]
        v_arrays: list[np.ndarray] = []
        for kind, payload in v_descriptors:
            if kind == "valid":
                is_valid = ~upstream.get_column(payload).is_null().to_numpy()
                v_arrays.append(np.ascontiguousarray(is_valid, dtype=np.float32))
            elif kind == "col":
                series = upstream.get_column(payload)
                v_arrays.append(np.ascontiguousarray(series.to_numpy(), dtype=np.float32))
            elif kind == "lit":
                v_arrays.append(np.asarray([payload], dtype=np.float32))
            else:
                raise RuntimeError(f"polars_metal: unknown validity descriptor {kind!r}")
        mask_out = np.empty(n_rows, dtype=np.float32)
        v_inputs = [(int(a.__array_interface__["data"][0]), int(a.size)) for a in v_arrays]
        _native.execute_fused_expr(
            scope=v_scope,
            inputs=v_inputs,
            out=(int(mask_out.__array_interface__["data"][0]), int(mask_out.size)),
        )
        return mask_out < 0.5

    raise RuntimeError(f"polars_metal: unknown fused null mode {null_mode!r}")


def _filter_via_polars(df_pydf: Any, filter_plan: dict) -> pl.DataFrame:
    """Filter the upstream DataFrame via Polars CPU `filter`, using a
    boolean Series built from the predicate AST.

    The predicate evaluation reuses the Metal kernels in
    `_evaluate_predicate` to produce a bit-packed `(data, valid)` pair,
    then we wrap those bytes in a Polars Boolean Series via
    `pa.Array.from_buffers` and let Polars handle the surviving-row
    compaction. This bypasses `_native.execute_filter_compact`, which
    runs one scatter kernel per surviving column — at ~30 ms/column it
    dominated Q1's wall-clock once the walker engaged (Phase 10
    diagnosis). Polars CPU filter is single-pass over all columns
    (~25 ms regardless of n_cols at 10M rows).

    Falls back to `_dispatch_filter` only if a predicate type isn't
    supported by `_evaluate_predicate`; the walker rejects unsupported
    predicates earlier, so this fallback is defensive.
    """
    upstream_plan = filter_plan["input"]
    if upstream_plan["kind"] == "Scan":
        upstream = pl.DataFrame._from_pydf(df_pydf)
    else:
        upstream_pydf = _native.execute_plan(df_pydf, upstream_plan)
        upstream = pl.DataFrame._from_pydf(upstream_pydf)

    n_rows = upstream.height
    if n_rows == 0:
        return upstream

    try:
        pred_data, pred_valid = _evaluate_predicate(upstream, filter_plan["predicate"], n_rows)
    except Exception:
        # Defensive: fall back to the GPU-compaction path if predicate eval fails.
        return _dispatch_filter(df_pydf, filter_plan)

    min_valid_bytes = (n_rows + 7) // 8
    pred_data_trim = bytes(pred_data[:min_valid_bytes]) if min_valid_bytes > 0 else b""
    pred_valid_trim = bytes(pred_valid[:min_valid_bytes]) if min_valid_bytes > 0 else b""
    bool_arr = pa.Array.from_buffers(
        pa.bool_(),
        n_rows,
        [pa.py_buffer(pred_valid_trim), pa.py_buffer(pred_data_trim)],
        null_count=-1,
    )
    mask = pl.Series("__metal_filter_mask", bool_arr)
    # Eager DataFrame.filter doesn't go through LazyFrame.collect, so the
    # monkey-patched callback isn't re-entered here.
    return upstream.filter(mask)


def _dispatch_filter(df_pydf: Any, filter_plan: dict) -> pl.DataFrame:
    """Run the GPU compaction pipeline against ``df_pydf`` and return a
    Polars DataFrame containing the surviving rows of every input column.

    The predicate is read from the same upstream DataFrame as the survivors
    (it lives in the input subtree, which today is always a Scan/Project
    that resolves to a contiguous slice of ``df_pydf``). The walker stamps
    the predicate AST into ``filter_plan["predicate"]``; we evaluate it
    here:
    - ``Column(bool)`` (Phase 5) — read the precomputed bool column.
    - ``Compare`` (Phase 6, Task 18) — dispatch the matching ``cmp_*``
      kernel against the upstream column buffers, producing a fresh
      bool predicate (bit-packed data + validity).
    """
    # Resolve the upstream DataFrame (after running scan/project nodes
    # under the Filter). The Filter node's input plan is one of:
    #   - Scan (the root): df_pydf is the underlying frame, no rewrite.
    #   - Project(Scan): a column re-selection; route through execute_plan
    #     to get the upstream DataFrame in its post-projection form.
    upstream_plan = filter_plan["input"]
    if upstream_plan["kind"] == "Scan":
        upstream = pl.DataFrame._from_pydf(df_pydf)
    else:
        upstream_pydf = _native.execute_plan(df_pydf, upstream_plan)
        upstream = pl.DataFrame._from_pydf(upstream_pydf)

    # GPU column compaction doesn't speak Utf8 yet; fall back to the
    # Polars CPU filter path which compacts all dtypes in one pass.
    # (This mirrors the route used when a GroupBy sits above the Filter
    # — see `_filter_via_polars`.)
    if any(str(dt) in ("String", "Utf8") for dt in upstream.schema.values()):
        return _filter_via_polars(df_pydf, filter_plan)

    n_rows = upstream.height
    pred_data, pred_valid = _evaluate_predicate(upstream, filter_plan["predicate"], n_rows)

    # Build the column-input list. Polars semantics: ``df.filter(mask)``
    # keeps the predicate column in the output (unless the caller
    # explicitly drops/selects it later). Mirror that — include the
    # predicate column in the survivor list too.
    column_inputs: list[tuple[str, str, bytes, bytes]] = []
    for col_name, dtype in upstream.schema.items():
        dtype_tag = _DTYPE_TO_TAG.get(str(dtype))
        if dtype_tag is None:
            # Walker should have rejected this upstream; surface clearly
            # rather than silently producing wrong output.
            raise RuntimeError(
                f"polars_metal: unsupported dtype {dtype!s} on column {col_name!r} "
                f"(walker should have fallen back)"
            )
        col_arr = _materialize_arrow(upstream.get_column(col_name))
        data_bytes, valid_bytes = _data_and_valid_for_dtype(col_arr, n_rows, dtype_tag)
        column_inputs.append((col_name, dtype_tag, data_bytes, valid_bytes))

    # Call into Rust. Returns one (data_bytes, valid_bytes, n_out) per input.
    results = _native.execute_filter_compact(pred_data, pred_valid, n_rows, column_inputs)
    if len(results) != len(column_inputs):
        raise RuntimeError(
            f"polars_metal: execute_filter_compact returned {len(results)} columns, "
            f"expected {len(column_inputs)}"
        )

    # Reassemble. Every output column has the same n_out by construction;
    # we read it from the first column and validate the rest match.
    n_out = int(results[0][2])
    series_list: list[pl.Series] = []
    for (col_name, dtype_tag, _src_data, _src_valid), (out_data, out_valid, n_out_i) in zip(
        column_inputs, results, strict=True
    ):
        n_out_i_int = int(n_out_i)
        if n_out_i_int != n_out:
            raise RuntimeError(
                f"polars_metal: column {col_name!r} reports n_out={n_out_i_int}, "
                f"expected {n_out} (other columns); compaction is inconsistent"
            )
        series_list.append(_assemble_series(col_name, dtype_tag, out_data, out_valid, n_out))

    return pl.DataFrame(series_list)


def _build_sort(plan: dict) -> Any:
    """Return a UDF for a Sort plan; runs inner pipeline then sorts result."""
    inner_plan = plan["input"]
    by_columns: list[str] = list(plan.get("by_columns", []))
    descending: list[bool] = list(plan.get("descending", [False] * len(by_columns)))
    nulls_last: list[bool] = list(plan.get("nulls_last", [False] * len(by_columns)))

    if inner_plan["kind"] == "GroupBy":
        inner_udf = _build_groupby(inner_plan)
    else:
        inner_udf = build_udf(inner_plan)

    # M4 Phase 7 (Task 27): single-column F32 sort routed to MLX.
    fused_sort = plan.get("_fused_sort")
    if fused_sort is not None:
        return _build_sort_fused(inner_udf, fused_sort, by_columns)

    def udf(
        with_columns: list[str] | None,
        predicate: Any,
        n_rows: int | None,
        should_time: bool,
    ) -> Any:
        inner_result = inner_udf(None, None, None, False)
        df = inner_result if isinstance(inner_result, pl.DataFrame) else inner_result[0]
        if by_columns:
            df = df.sort(
                by_columns,
                descending=descending if any(descending) else False,
                nulls_last=nulls_last if any(nulls_last) else False,
            )
        if n_rows is not None:
            df = df.slice(0, n_rows)
        if should_time:
            return df, []
        return df

    return udf


def _build_sort_fused(inner_udf: Any, fused_sort: dict, by_columns: list[str]) -> Any:
    """UDF for a single-column F32 Sort routed to MLX (Task 27).

    Runs the inner pipeline, sorts the one F32 column via the MLX ``Sort`` op
    (ascending), then reverses for descending and slices for top_k/bottom_k on
    the host. Falls back to a Polars CPU sort when the column has nulls (MLX
    sorts NaN-filled nulls; Polars' null placement differs) or is empty.
    """
    import numpy as np

    scope = fused_sort["scope"]
    col: str = fused_sort["column"]
    descending: bool = fused_sort["descending"]
    nulls_last: bool = fused_sort["nulls_last"]
    sort_slice = fused_sort["slice"]
    drop_nulls: bool = fused_sort.get("drop_nulls", False)

    def udf(
        with_columns: list[str] | None,
        predicate: Any,
        n_rows: int | None,
        should_time: bool,
    ) -> Any:
        inner_result = inner_udf(None, None, None, False)
        df = inner_result if isinstance(inner_result, pl.DataFrame) else inner_result[0]
        series = df.get_column(col)
        if drop_nulls and series.null_count() > 0:
            # top_k / bottom_k drop nulls before ranking (we bypassed the
            # dynamic null-drop Filter in the walker). `null_count` is O(1)
            # (Arrow-tracked), so the common no-null case skips the O(n) copy.
            df = df.drop_nulls(col)
            series = df.get_column(col)

        if series.len() == 0 or series.null_count() > 0 or str(series.dtype) != "Float32":
            # CPU fallback preserves Polars null/empty semantics exactly.
            df = df.sort(by_columns, descending=descending, nulls_last=nulls_last)
            if sort_slice is not None:
                df = df.slice(sort_slice[0], sort_slice[1])
        else:
            arr = np.ascontiguousarray(series.to_numpy(), dtype=np.float32)
            out_arr = np.empty(arr.size, dtype=np.float32)
            _native.execute_fused_expr(
                scope=scope,
                inputs=[(int(arr.__array_interface__["data"][0]), int(arr.size))],
                out=(int(out_arr.__array_interface__["data"][0]), int(out_arr.size)),
            )
            # `out_arr` is ascending. Apply descending + slice at the array
            # level so a top_k (descending + slice=(0,k)) reverses only k
            # elements, not the whole 10M-row sort. MLX has no descending sort
            # binding, so a full descending sort still reverses all of out_arr.
            m = out_arr.size
            off, length = sort_slice if sort_slice is not None else (0, m)
            if descending:
                hi = m - off
                lo = max(0, hi - length)
                out_arr = np.ascontiguousarray(out_arr[lo:hi][::-1])
            else:
                out_arr = out_arr[off : off + length]
            df = pl.DataFrame([pl.Series(col, out_arr)])

        if n_rows is not None:
            df = df.slice(0, n_rows)
        if should_time:
            return df, []
        return df

    return udf


def _build_groupby(plan: dict) -> Any:
    """Return a UDF callable for a GroupBy plan.

    The plan dict's ``input`` subtree resolves to the upstream DataFrame.
    For Phase 8, the supported input shape is ``Scan`` directly under GroupBy
    (or any other scan/project/filter shape that ``_extract_scan_df_and_wire_plan``
    already handles). We extract the underlying ``df_pydf`` and the full wire
    plan (including the GroupBy root node) and dispatch to ``_dispatch_groupby``
    at call time.
    """
    # M4 Phase 7 (Task 26): empty-key reduction whose every agg the analyzer
    # accepted as a fused F32 reduction routes through the MLX subgraph path
    # instead of the GroupBy conformance kernel.
    if not plan.get("keys") and plan.get("_fused_aggs"):
        return _build_select_reduction_fused(plan)

    df_pydf, wire_plan = _extract_scan_df_and_wire_plan(plan)

    def udf(
        with_columns: list[str] | None,
        predicate: Any,
        n_rows: int | None,
        should_time: bool,
    ) -> Any:
        df = _dispatch_groupby(df_pydf, wire_plan)
        if n_rows is not None:
            df = df.slice(0, n_rows)
        if should_time:
            return df, []
        return df

    return udf


def _build_select_reduction_fused(plan: dict) -> Any:
    """UDF for an empty-key Select reduction routed through the fused MLX path.

    Each binding in ``plan['_fused_aggs']`` carries an analyzed scope whose
    single output is the scalar reduction, the ordered input descriptors, the
    output column name, and the lowercase agg kind. At call time we resolve the
    upstream frame, run each reduction (scalar output), apply the Bessel
    correction to std/var (MLX uses population variance, ddof=0; Polars
    defaults to sample, ddof=1), and assemble a one-row DataFrame.
    """
    import math

    import numpy as np

    fused_aggs: list[dict] = plan["_fused_aggs"]
    df_pydf, wire_plan = _extract_scan_df_and_wire_plan(plan)
    input_plan = wire_plan["input"]

    def udf(
        with_columns: list[str] | None,
        predicate: Any,
        n_rows: int | None,
        should_time: bool,
    ) -> Any:
        upstream = _dispatch(df_pydf, input_plan)
        n = upstream.height
        out_series: list[pl.Series] = []
        for agg in fused_aggs:
            scope = agg["_fused_scope"]
            descriptors: list[tuple[str, str | float]] = agg["_fused_columns"]
            name = agg["name"]
            kind = agg["_agg_kind"]
            # Bare single Float32 column (enforced by `analyze_ir_reduction`).
            col_name = descriptors[0][1]
            series = upstream.get_column(col_name)

            # MLX over `to_numpy()` turns nulls into NaN (Polars skips nulls),
            # and population std/var of <2 rows can't be Bessel-corrected. In
            # those cases compute the reduction with Polars on the source column
            # — exact (same null skipping, same n=0/1 semantics, same dtype).
            if n < 2 or series.null_count() > 0:
                value = getattr(series, kind)()
                out_series.append(pl.Series(name, [value], dtype=series.dtype))
                continue

            arr = np.ascontiguousarray(series.to_numpy(), dtype=np.float32)
            out_arr = np.empty(1, dtype=np.float32)
            _native.execute_fused_expr(
                scope=scope,
                inputs=[(int(arr.__array_interface__["data"][0]), int(arr.size))],
                out=(int(out_arr.__array_interface__["data"][0]), int(out_arr.size)),
            )
            val = float(out_arr[0])
            # Bessel: MLX uses population variance (ddof=0); Polars defaults to
            # sample (ddof=1). Only std/var need the correction.
            if kind == "var":
                val *= n / (n - 1)
            elif kind == "std":
                val *= math.sqrt(n / (n - 1))
            out_series.append(pl.Series(name, [val], dtype=pl.Float32))

        result = pl.DataFrame(out_series)
        if should_time:
            return result, []
        return result

    return udf


def _dispatch_groupby(df_pydf: Any, wire_plan: dict) -> pl.DataFrame:
    """Run the GroupBy pipeline against ``df_pydf`` and return a Polars
    DataFrame of (groups x (keys + aggs)) computed on Metal.

    Steps:
      1. Convert df_pydf → pl.DataFrame for column iteration.
      2. Determine the set of columns we need (union of keys + agg.input_col,
         skipping empty input_col which is for Len).
      3. Materialize each column as (name, dtype_tag, data_bytes, valid_bytes).
      4. Call ``_native.execute_groupby(wire_plan, n_rows, columns)``.
      5. Reassemble the returned (name, dtype_tag, data, valid) tuples into
         a Polars DataFrame.
    """
    # The wire_plan root is a GroupBy node; the upstream data lives in its
    # input subtree. Resolve it. For the Filter case we explicitly bypass
    # the GPU compaction path (which dispatches one scatter kernel per
    # surviving column — ~30ms x n_cols at 10M rows, dominating Q1's
    # engine-level wall-clock). Instead we compute the predicate via the
    # existing Metal kernels, build a Polars Boolean Series from the
    # bit-packed result, and let Polars' SIMD-tuned filter compact all
    # columns in one pass (~20-25ms regardless of n_cols).
    upstream_plan = wire_plan["input"]
    # HStack hoisted above the upstream (Polars CSE optimization) — peel it
    # off, dispatch the inner subtree, then evaluate the appended columns.
    hstack_exprs: list[dict] = []
    if upstream_plan["kind"] == "HStack":
        hstack_exprs = list(upstream_plan.get("exprs", []))
        upstream_plan = upstream_plan["input"]
    if upstream_plan["kind"] == "Scan":
        upstream = pl.DataFrame._from_pydf(df_pydf)
    elif upstream_plan["kind"] == "Filter":
        upstream = _filter_via_polars(df_pydf, upstream_plan)
    elif (
        upstream_plan["kind"] == "Project"
        and upstream_plan.get("input", {}).get("kind") == "Filter"
    ):
        filtered = _filter_via_polars(df_pydf, upstream_plan["input"])
        upstream = pl.DataFrame._from_pydf(filtered._df.select(list(upstream_plan["columns"])))
    else:
        upstream_pydf = _native.execute_plan(df_pydf, upstream_plan)
        upstream = pl.DataFrame._from_pydf(upstream_pydf)

    if hstack_exprs:
        new_cols = [_agg_expr_dict_to_polars(e["expr"]).alias(e["name"]) for e in hstack_exprs]
        upstream = upstream.lazy().with_columns(new_cols).collect()

    n_rows = upstream.height

    # M3 Phase 13b: F64-Expression aggs need pre-materialization.
    #
    # The Rust dispatcher has no path that accepts Expression specs over F64
    # inputs (fused kernel rejects F64; PerAgg rejects Expression). For any
    # Expression agg, evaluate the expression once on the upstream DataFrame
    # via Polars, then rewrite the agg as Simple-<op> referencing the tmp
    # column. The grouping + reduction still happens on Metal — only the
    # per-row arithmetic moves to CPU. Works for both empty-keys (Q6) and
    # multi-key (Q1 with CSE'd Expressions) GroupBy.
    if any(a.get("kind") == "Expression" for a in wire_plan["aggs"]):
        upstream, wire_plan = _materialize_expression_aggs(upstream, wire_plan)
        n_rows = upstream.height

    # Empty-keys short-circuit: a single-group reduction over the full
    # upstream is faster on CPU than on GPU on this hardware. The
    # per-agg kernels use atomic CAS-loops on a single output slot, which
    # collapses to fully-serialized execution when every thread targets
    # the same slot (n_groups == 1). At 1M+ rows that hits the Metal
    # watchdog (kIOGPUCommandBufferCallbackErrorImpactingInteractivity).
    # Polars CPU reductions are SIMD-tuned and finish in microseconds —
    # use them. Multi-group GroupBy still routes through the kernel
    # layer (low-cardinality contention is a separate M4 perf concern).
    if not wire_plan["keys"]:
        return _empty_keys_via_polars(upstream, wire_plan["aggs"])

    # Collect the column names we'll send to Rust (keys + agg inputs).
    # Simple aggs reference one column via `input_col`; Expression aggs
    # reference one or more via the recursive `expr` dict (M3 capability G).
    needed: list[str] = []
    seen: set[str] = set()

    def _collect_expr_cols(expr_dict: dict, out: list[str], seen_set: set[str]) -> None:
        """Walk an Expression-kind agg's `expr` sub-dict, collecting column refs."""
        kind = expr_dict.get("kind")
        if kind == "Column":
            name = expr_dict.get("name", "")
            if name and name not in seen_set:
                out.append(name)
                seen_set.add(name)
        elif kind == "Binary":
            lhs = expr_dict.get("lhs")
            rhs = expr_dict.get("rhs")
            if isinstance(lhs, dict):
                _collect_expr_cols(lhs, out, seen_set)
            if isinstance(rhs, dict):
                _collect_expr_cols(rhs, out, seen_set)
        # Literals (LiteralF64, LiteralI64) reference no columns.

    for key_entry in wire_plan["keys"]:
        key_name = key_entry[0]
        if key_name not in seen:
            needed.append(key_name)
            seen.add(key_name)
    for agg in wire_plan["aggs"]:
        kind = agg.get("kind", "Simple")
        if kind == "Expression":
            expr_dict = agg.get("expr")
            if isinstance(expr_dict, dict):
                _collect_expr_cols(expr_dict, needed, seen)
            continue
        ic = agg.get("input_col", "")
        if ic and ic not in seen:
            needed.append(ic)
            seen.add(ic)

    # Materialize each needed column as raw Arrow bytes.
    columns_in: list[tuple[str, str, bytes, bytes]] = []
    for col_name in needed:
        if col_name not in upstream.columns:
            raise RuntimeError(
                f"polars_metal: column {col_name!r} required by GroupBy not in upstream df"
            )
        dtype = upstream.schema[col_name]
        dtype_tag = _DTYPE_TO_TAG.get(str(dtype))
        if dtype_tag is None:
            raise RuntimeError(
                f"polars_metal: unsupported dtype {dtype!s} on column {col_name!r} "
                f"(walker should have fallen back)"
            )
        col_arr = _materialize_arrow(upstream.get_column(col_name))
        data_bytes, valid_bytes = _data_and_valid_for_dtype(col_arr, n_rows, dtype_tag)
        columns_in.append((col_name, dtype_tag, data_bytes, valid_bytes))

    # Build the GroupBy-only wire plan (strip the input subtree; Rust re-uses
    # just the keys/aggs/kind fields from the plan_dict).
    groupby_plan = {
        "kind": "GroupBy",
        "input": wire_plan["input"],
        "keys": wire_plan["keys"],
        "aggs": wire_plan["aggs"],
    }

    # Call Rust.
    out_columns = _native.execute_groupby(groupby_plan, n_rows, columns_in)

    # Reassemble. Two-pass: first find n_out from a non-Bool column (Bool's
    # bit-packed valid-bitmap length over-counts due to 4-byte alignment
    # padding); then assemble all columns with the established n_out.
    n_out: int | None = None
    for name, dtype_tag, data, _valid in out_columns:
        if dtype_tag in ("U32", "I32", "F32"):
            n_out = len(data) // 4
            break
        if dtype_tag in ("I64", "F64"):
            n_out = len(data) // 8
            break
        if dtype_tag == "Utf8":
            if len(data) < 4:
                raise RuntimeError(
                    f"polars_metal: Utf8 column {name!r} wire payload "
                    f"too short ({len(data)} B; need >= 4 for header)"
                )
            n_out = int.from_bytes(data[:4], "little")
            break
    if n_out is None:
        n_out = len(out_columns[0][3]) * 8 if out_columns else 0

    series_list: list[pl.Series] = [
        _assemble_series(name, dtype_tag, data, valid, n_out)
        for name, dtype_tag, data, valid in out_columns
    ]

    if not series_list:
        return pl.DataFrame()
    return pl.DataFrame(series_list)


def _agg_expr_dict_to_polars(expr_dict: dict) -> pl.Expr:
    """Recursively map a wire-form AggExpr sub-tree to a `pl.Expr`.

    Mirrors the shape produced by `_walker._walk_agg_expr_node`:

    - ``{"kind": "Column", "name": str}`` → ``pl.col(name)``
    - ``{"kind": "LiteralF64", "value": float}`` → ``pl.lit(float)``
    - ``{"kind": "LiteralI64", "value": int}`` → ``pl.lit(int)``
    - ``{"kind": "Binary", "op": "Add|Sub|Mul|Div", "lhs": ..., "rhs": ...}``
      → ``lhs +-*/ rhs``
    """
    kind = expr_dict.get("kind")
    if kind == "Column":
        return pl.col(str(expr_dict["name"]))
    if kind == "LiteralF64":
        return pl.lit(float(expr_dict["value"]))
    if kind == "LiteralI64":
        return pl.lit(int(expr_dict["value"]))
    if kind == "Binary":
        lhs = _agg_expr_dict_to_polars(expr_dict["lhs"])
        rhs = _agg_expr_dict_to_polars(expr_dict["rhs"])
        op = expr_dict.get("op")
        if op == "Add":
            return lhs + rhs
        if op == "Sub":
            return lhs - rhs
        if op == "Mul":
            return lhs * rhs
        if op == "Div":
            return lhs / rhs
        raise RuntimeError(f"polars_metal: unknown Binary op {op!r}")
    raise RuntimeError(f"polars_metal: unknown agg-expr kind {kind!r}")


def _materialize_expression_aggs(
    upstream: pl.DataFrame, wire_plan: dict
) -> tuple[pl.DataFrame, dict]:
    """Evaluate each Expression agg's expression via Polars into a tmp
    column, then rewrite the agg as Simple referencing that column.
    Returns the augmented DataFrame and a new wire_plan.

    Why this lives in Python:
        The Rust dispatch layer routes F64-Expression aggs nowhere — the
        fused kernel doesn't support F64 inputs (toolchain lacks 64-bit
        atomics) and the PerAgg loop has no Expression handler. Pre-
        materializing the expression on CPU lets the existing per-group
        Sum/Mean/Min/Max kernels consume the result. Per-row arithmetic
        runs on CPU; the (more expensive) grouping + reduction still
        runs on Metal. Works for empty-keys (Q6) and multi-key (Q1's
        CSE-residual Expression) shapes alike.
    """
    new_aggs: list[dict] = []
    new_columns: list[pl.Series] = []
    next_tmp_idx = 0
    for agg in wire_plan["aggs"]:
        if agg.get("kind") != "Expression":
            new_aggs.append(agg)
            continue
        expr = _agg_expr_dict_to_polars(agg["expr"])
        tmp_name = f"__pm_expr_agg_{next_tmp_idx}__"
        next_tmp_idx += 1
        # Evaluate the expression in isolation so we don't trigger a full
        # column re-materialization on `upstream` (which can be 10M+ rows).
        ser = upstream.lazy().select(expr.alias(tmp_name)).collect().to_series()
        new_columns.append(ser)
        new_aggs.append(
            {
                "kind": "Simple",
                "input_col": tmp_name,
                "op": agg["op"],
                "output_alias": agg.get("output_alias", tmp_name),
            }
        )

    if new_columns:
        upstream = upstream.with_columns(new_columns)

    new_wire_plan = dict(wire_plan)
    new_wire_plan["aggs"] = new_aggs
    return upstream, new_wire_plan


_AGG_OP_TO_POLARS: dict[str, str] = {
    "Sum": "sum",
    "Mean": "mean",
    "Min": "min",
    "Max": "max",
    "Count": "count",
}


def _empty_keys_via_polars(upstream: pl.DataFrame, aggs: list[dict]) -> pl.DataFrame:
    """Evaluate empty-keys aggs via Polars CPU. See call site for rationale.

    All Expression aggs are already materialized to Simple-<op> referencing
    a tmp column by `_materialize_expression_aggs` before we get here.
    """
    out_exprs: list[pl.Expr] = []
    for agg in aggs:
        kind = agg.get("kind")
        alias = agg.get("output_alias") or ""
        if kind == "Length":
            out_exprs.append(pl.len().alias(alias or "len"))
            continue
        if kind != "Simple":
            raise RuntimeError(
                f"polars_metal: empty-keys agg expected Simple/Length after "
                f"materialization, got kind={kind!r}"
            )
        col_name = agg.get("input_col") or ""
        op = agg.get("op") or ""
        polars_op = _AGG_OP_TO_POLARS.get(op)
        if polars_op is None:
            raise RuntimeError(f"polars_metal: unknown empty-keys agg op {op!r}")
        expr = getattr(pl.col(col_name), polars_op)().alias(alias or f"{col_name}_{op.lower()}")
        out_exprs.append(expr)

    return upstream.lazy().select(out_exprs).collect()


def _evaluate_predicate(upstream: pl.DataFrame, pred: dict, n_rows: int) -> tuple[bytes, bytes]:
    """Evaluate a serialized predicate AST into a bit-packed ``(data, valid)`` pair.

    Supported shapes (walker's closed set):

    - ``Column(name, dtype="Bool")`` — Phase 5: read the precomputed bool
      column straight off the upstream frame.
    - ``Compare { op, lhs, rhs, dtype }`` where ``lhs`` and ``rhs`` are
      each either ``Column(I64|F64)`` or matching-dtype ``Literal`` —
      Phase 6: dispatch the matching ``cmp_<dtype>_col_(col|scalar)``
      kernel against the upstream column buffers.
    - ``And { lhs, rhs }`` / ``Or { lhs, rhs }`` — Phase 7 (Task 20):
      recursively evaluate both sub-predicates, then combine via
      ``bool_and_dispatch`` / ``bool_or_dispatch`` (3-valued AND/OR
      against bit-packed bool + validity bitmaps). Arbitrary nesting
      unfolds via the recursion.

    Returns bit-packed bytes (each at least ``ceil(n_rows / 8)`` bytes).
    """
    kind = pred["kind"]
    if kind == "Column":
        if pred.get("dtype") != "Bool":
            raise RuntimeError(
                f"polars_metal: bare Column predicate must be Bool, got {pred.get('dtype')!r}"
            )
        pred_arr = _materialize_arrow(upstream.get_column(pred["name"]))
        return _bitpacked_data_and_valid(pred_arr, n_rows)

    if kind == "Compare":
        return _evaluate_compare(upstream, pred, n_rows)

    if kind in ("And", "Or"):
        return _evaluate_logical(upstream, pred, n_rows)

    raise RuntimeError(f"polars_metal: unsupported predicate kind {kind!r}")


def _evaluate_logical(upstream: pl.DataFrame, pred: dict, n_rows: int) -> tuple[bytes, bytes]:
    """Recursively evaluate both sub-predicates, then combine via the
    matching ``bool_<and|or>_dispatch`` pyfunction.

    ``bool_and`` / ``bool_or`` implement Polars' 3-valued (Kleene) logic:
    AND is dominated by ``false`` (so ``false AND null = false``), OR is
    dominated by ``true`` (so ``true OR null = true``); otherwise null
    propagates. The kernels read both data and validity bitmaps for each
    side and write a fresh ``(data, valid)`` pair for the combined
    predicate.

    The kernel input contract requires each bitmap to be at least
    ``ceil(n_rows / 8)`` bytes; sub-predicate outputs may be 4-byte-
    aligned padded (see ``cmp_out_min_bytes`` on the Rust side), which
    already satisfies the minimum. We don't trim further: extra bytes
    past row ``n_rows`` are read but ignored by the kernel.
    """
    lhs_data, lhs_valid = _evaluate_predicate(upstream, pred["lhs"], n_rows)
    rhs_data, rhs_valid = _evaluate_predicate(upstream, pred["rhs"], n_rows)
    min_bytes = (n_rows + 7) // 8
    # Defensive pad — every evaluator above already produces at least
    # `min_bytes` bytes, but keep the contract explicit so a future
    # evaluator that returns a tighter buffer doesn't silently break the
    # kernel's length check.
    lhs_data = _pad_to(lhs_data, min_bytes)
    lhs_valid = _pad_to(lhs_valid, min_bytes)
    rhs_data = _pad_to(rhs_data, min_bytes)
    rhs_valid = _pad_to(rhs_valid, min_bytes)
    if pred["kind"] == "And":
        return _native.bool_and_dispatch(lhs_data, lhs_valid, rhs_data, rhs_valid, n_rows)
    return _native.bool_or_dispatch(lhs_data, lhs_valid, rhs_data, rhs_valid, n_rows)


def _pad_to(buf: bytes, min_bytes: int) -> bytes:
    """Right-pad ``buf`` with zero bytes to at least ``min_bytes``."""
    if len(buf) >= min_bytes:
        return buf
    return buf + b"\x00" * (min_bytes - len(buf))


def _coerce_for_compare(series: pl.Series, kernel_dtype: str) -> pl.Series:
    """Cast `series` to the dtype the compare kernel expects.

    The walker widens narrow integer types and Date to I64 for the
    cmp_i64 kernel (see `_walker._PREDICATE_I64_WIDEN`); F32 widens to
    F64 for cmp_f64 (`_walker._PREDICATE_F64_WIDEN`). The underlying
    Polars Series is still in its narrow form here, so we cast on the
    way to materialization. Date → Int64 days-since-1970 (Polars'
    physical representation) matches the walker's literal encoding.
    """
    if kernel_dtype == "I64" and str(series.dtype) != "Int64":
        return series.cast(pl.Int64)
    if kernel_dtype == "F64" and str(series.dtype) != "Float64":
        return series.cast(pl.Float64)
    return series


def _evaluate_compare(upstream: pl.DataFrame, pred: dict, n_rows: int) -> tuple[bytes, bytes]:
    """Dispatch one of the four ``cmp_*`` pyfunctions based on the
    operand dtype and which side is a literal vs a column.

    The walker has already verified both leaves share an operand dtype
    (``Compare["dtype"]``). We additionally fast-path the
    ``column op scalar`` and ``column op column`` shapes — the walker
    rejects literal-vs-literal at validation time, so we don't see it
    here. Scalar-vs-column ``(literal op column)`` is rewritten into
    the canonical ``(column op_swap literal)`` form to hit the scalar
    kernel; the op must be swapped to preserve semantics
    (e.g. ``5 < a`` becomes ``a > 5``).
    """
    op = pred["op"]
    dtype = pred["dtype"]
    lhs = pred["lhs"]
    rhs = pred["rhs"]

    # Canonicalise so the Column is always on the LHS for scalar variants.
    if lhs["kind"].startswith("Literal") and rhs["kind"] == "Column":
        lhs, rhs = rhs, lhs
        op = _SWAP_COMPARE_OP[op]

    if lhs["kind"] != "Column":
        # Walker should reject; defensive check matches the cmp kernel
        # input contract (LHS is always a column).
        raise RuntimeError(f"polars_metal: cmp predicate LHS must be Column, got {lhs['kind']!r}")

    lhs_series = _coerce_for_compare(upstream.get_column(lhs["name"]), dtype)
    lhs_arr = _materialize_arrow(lhs_series)
    lhs_data, lhs_valid = _data_and_valid_for_dtype(lhs_arr, n_rows, dtype)

    if rhs["kind"] == "Column":
        rhs_series = _coerce_for_compare(upstream.get_column(rhs["name"]), dtype)
        rhs_arr = _materialize_arrow(rhs_series)
        rhs_data, rhs_valid = _data_and_valid_for_dtype(rhs_arr, n_rows, dtype)
        if dtype == "I64":
            return _native.cmp_i64_col_col(lhs_data, lhs_valid, rhs_data, rhs_valid, op, n_rows)
        if dtype == "F64":
            return _native.cmp_f64_col_col(lhs_data, lhs_valid, rhs_data, rhs_valid, op, n_rows)
        raise RuntimeError(f"polars_metal: cmp dtype {dtype!r} not wired")

    # rhs is a Literal — call the matching scalar variant.
    if dtype == "I64":
        if rhs["kind"] != "LiteralI64":
            raise RuntimeError(
                f"polars_metal: dtype I64 cmp expected LiteralI64 RHS, got {rhs['kind']!r}"
            )
        return _native.cmp_i64_col_scalar(lhs_data, lhs_valid, int(rhs["value"]), op, n_rows)
    if dtype == "F64":
        if rhs["kind"] != "LiteralF64":
            raise RuntimeError(
                f"polars_metal: dtype F64 cmp expected LiteralF64 RHS, got {rhs['kind']!r}"
            )
        return _native.cmp_f64_col_scalar(lhs_data, lhs_valid, float(rhs["value"]), op, n_rows)
    raise RuntimeError(f"polars_metal: cmp dtype {dtype!r} not wired")


# Op tags swapped when we canonicalise ``literal OP column`` into
# ``column OP_swap literal``. Eq and Ne are symmetric; the four inequality
# ops swap to their reflection (``a < b`` ↔ ``b > a``).
_SWAP_COMPARE_OP: dict[str, str] = {
    "Eq": "Eq",
    "Ne": "Ne",
    "Lt": "Gt",
    "Le": "Ge",
    "Gt": "Lt",
    "Ge": "Le",
}


_DTYPE_TO_TAG: dict[str, str] = {
    "Int64": "I64",
    "Float64": "F64",
    "Boolean": "Bool",
    "Int32": "I32",
    "Float32": "F32",
    # py-1.40.1 reports pl.Utf8 as "String"; older Polars reports "Utf8".
    # The Rust-side `MetalDtype::from_wire` accepts the "Utf8" tag.
    "String": "Utf8",
    "Utf8": "Utf8",
}

_TAG_TO_PA: dict[str, pa.DataType] = {
    "I64": pa.int64(),
    "F64": pa.float64(),
    "Bool": pa.bool_(),
    "U32": pa.uint32(),
    "I32": pa.int32(),
    "F32": pa.float32(),
    # M3 Phase 7: dictionary-encoded Utf8 keys round-trip through pa.string()
    # (i32 offsets). pl.Utf8 itself materializes as pa.large_string() in
    # py-1.40.1 — we cast on the way in (see `_data_and_valid_for_dtype`).
    "Utf8": pa.string(),
}


def _materialize_arrow(s: pl.Series) -> pa.Array:
    """Return a PyArrow array for ``s`` with ``offset == 0`` and a single chunk.

    Slicing a Polars frame produces Arrow arrays with a non-zero offset
    whose buffers still cover the *parent* extent. The kernels don't
    interpret offsets — they read from byte 0 — so we must materialise a
    fresh contiguous array before extracting buffers.
    ``pa.concat_arrays([arr])`` does this with a single copy when the
    offset is non-zero and is a no-op when the offset is already 0.
    """
    arr = s.to_arrow()
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks() if arr.num_chunks > 1 else arr.chunk(0)
    if arr.offset != 0:
        arr = pa.concat_arrays([arr])
    return arr


def _bitpacked_data_and_valid(arr: pa.Array, n_rows: int) -> tuple[bytes, bytes]:
    """Extract bit-packed data + validity bytes from a Boolean Arrow array.

    Arrow lays Boolean arrays out as ``[validity_bitmap, data_bitmap]``
    (buffers[0], buffers[1]). When all rows are valid, ``buffers()[0]``
    is ``None`` — materialise an all-ones bitmap so the Rust side always
    sees a real buffer.

    Returns ``(data, valid)`` each of length ``>= ceil(n_rows / 8)``.
    """
    bufs = arr.buffers()
    # buffers[0] is validity; buffers[1] is data.
    min_bytes = (n_rows + 7) // 8
    valid_buf = bufs[0]
    data_buf = bufs[1]
    valid = bytes(valid_buf) if valid_buf is not None else _all_ones_bitmap(n_rows)
    data = bytes(data_buf) if data_buf is not None else b""
    # Pad to the kernel's minimum so the Rust-side length checks pass.
    if len(valid) < min_bytes:
        valid = valid + b"\x00" * (min_bytes - len(valid))
    if len(data) < min_bytes:
        data = data + b"\x00" * (min_bytes - len(data))
    return data, valid


def _data_and_valid_for_dtype(arr: pa.Array, n_rows: int, dtype_tag: str) -> tuple[bytes, bytes]:
    """Extract ``(data, valid)`` bytes appropriate for ``dtype_tag``.

    For I64/F64 the data is dense (``n_rows * 8`` bytes); for Bool it is
    bit-packed (``ceil(n_rows / 8)`` bytes). Validity is always
    bit-packed and materialised to all-ones when Arrow elided it.

    For Utf8 (M3 Phase 7), the data buffer is the packed wire format
    ``[n_rows u32 le | offsets (n+1) x i32 le | concatenated bytes]``.
    pl.Utf8 materializes as ``pa.large_string()`` (i64 offsets) in
    py-1.40.1; we cast to ``pa.string()`` (i32 offsets) so the wire
    format matches the Rust-side parser.
    """
    if dtype_tag == "Utf8":
        # Cast Polars' native large_string → string (i32 offsets) so the
        # buffer layout matches the wire format the Rust side expects.
        # pa.array.cast is a no-op when the type already matches; for
        # large_string → string it materializes a fresh buffer set with
        # i32 offsets in one pass (Arrow C++).
        if arr.type != pa.string():
            arr = arr.cast(pa.string())
        bufs = arr.buffers()
        valid_buf = bufs[0]
        offsets_buf = bufs[1] if len(bufs) > 1 else None
        data_buf = bufs[2] if len(bufs) > 2 else None
        valid = bytes(valid_buf) if valid_buf is not None else _all_ones_bitmap(n_rows)
        min_valid_bytes = (n_rows + 7) // 8
        if len(valid) < min_valid_bytes:
            valid = valid + b"\x00" * (min_valid_bytes - len(valid))

        offsets_bytes = bytes(offsets_buf) if offsets_buf is not None else b""
        data_bytes = bytes(data_buf) if data_buf is not None else b""
        expected_offsets_len = (n_rows + 1) * 4
        if len(offsets_bytes) < expected_offsets_len:
            # Defensive: an empty array still has at least one offset (0).
            offsets_bytes = offsets_bytes + b"\x00" * (expected_offsets_len - len(offsets_bytes))
        else:
            offsets_bytes = offsets_bytes[:expected_offsets_len]

        # The data buffer is sized by offsets[n_rows]; trim safely.
        if n_rows == 0:
            total_data_len = 0
        else:
            total_data_len = int.from_bytes(
                offsets_bytes[n_rows * 4 : (n_rows + 1) * 4],
                "little",
                signed=True,
            )
        if total_data_len < 0:
            raise RuntimeError(
                f"polars_metal: Utf8 column has negative final offset {total_data_len}"
            )
        if total_data_len > len(data_bytes):
            # Pad up to the offset-promised length; should not occur with
            # well-formed Arrow buffers but matches the I64/F64 defensive
            # pattern.
            data_bytes = data_bytes + b"\x00" * (total_data_len - len(data_bytes))
        else:
            data_bytes = data_bytes[:total_data_len]

        out = bytearray()
        out.extend(n_rows.to_bytes(4, "little"))
        out.extend(offsets_bytes)
        out.extend(data_bytes)
        return bytes(out), valid

    bufs = arr.buffers()
    valid_buf = bufs[0]
    data_buf = bufs[1]
    valid = bytes(valid_buf) if valid_buf is not None else _all_ones_bitmap(n_rows)
    data = bytes(data_buf) if data_buf is not None else b""
    min_valid_bytes = (n_rows + 7) // 8
    if len(valid) < min_valid_bytes:
        valid = valid + b"\x00" * (min_valid_bytes - len(valid))
    if dtype_tag in ("I64", "F64"):
        expected_data = n_rows * 8
        if len(data) < expected_data:
            data = data + b"\x00" * (expected_data - len(data))
    elif dtype_tag in ("I32", "F32"):
        expected_data = n_rows * 4
        if len(data) < expected_data:
            data = data + b"\x00" * (expected_data - len(data))
    elif dtype_tag == "Bool":
        if len(data) < min_valid_bytes:
            data = data + b"\x00" * (min_valid_bytes - len(data))
    else:  # pragma: no cover — walker rejects other dtypes
        raise RuntimeError(f"polars_metal: unexpected dtype_tag {dtype_tag!r}")
    return data, valid


def _all_ones_bitmap(n_rows: int) -> bytes:
    """Bit-packed all-ones validity bitmap for ``n_rows`` rows.

    The kernel checks ``valid[i / 8] & (1 << (i % 8))`` per row, so any
    trailing bits past row ``n_rows`` are read but ignored — we can
    safely set every bit in the last byte without affecting correctness.
    """
    n_bytes = (n_rows + 7) // 8
    return b"\xff" * n_bytes


def _assemble_series(name: str, dtype_tag: str, data: bytes, valid: bytes, n_out: int) -> pl.Series:
    """Build a Polars Series from raw Arrow buffer bytes.

    For I64/F64: ``data`` is ``n_out * 8`` bytes (dense), ``valid`` is
    bit-packed at least ``ceil(n_out / 8)`` bytes (possibly 4-byte-aligned
    padded by the kernel — we trim).

    For Bool: both ``data`` and ``valid`` are bit-packed; same trim.

    For Utf8 (M3 Phase 7): ``data`` is the packed wire format
    ``[n_rows u32 le | offsets (n+1) x i32 le | concatenated bytes]``;
    we parse it back to three Arrow buffers for ``pa.string()``.

    The validity buffer is always present (the kernel writes one); if the
    column has no nulls we still hand a buffer to PyArrow rather than
    None — PyArrow accepts that and reports the correct null_count via
    ``null_count=-1`` (compute on demand).
    """
    if dtype_tag == "Utf8":
        # Wire: [n_rows u32 le | offsets (n+1) x i32 le | bytes].
        if len(data) < 4:
            raise RuntimeError(
                f"polars_metal: Utf8 output column {name!r} wire payload "
                f"too short ({len(data)} B; need >= 4 for header)"
            )
        n = int.from_bytes(data[:4], "little")
        if n != n_out:
            raise RuntimeError(
                f"polars_metal: Utf8 output column {name!r} n_rows mismatch: header={n} out={n_out}"
            )
        offsets_end = 4 + (n + 1) * 4
        if len(data) < offsets_end:
            raise RuntimeError(
                f"polars_metal: Utf8 output column {name!r} truncated in offsets "
                f"({len(data)} B; need >= {offsets_end})"
            )
        offsets_bytes = data[4:offsets_end]
        string_bytes = data[offsets_end:]
        min_valid_bytes = (n_out + 7) // 8
        valid_trim = valid[:min_valid_bytes] if min_valid_bytes > 0 else b""
        if n_out == 0:
            arr = pa.array([], type=pa.string())
            return pl.Series(name, arr)
        arr = pa.Array.from_buffers(
            pa.string(),
            n_out,
            [
                pa.py_buffer(valid_trim),
                pa.py_buffer(offsets_bytes),
                pa.py_buffer(string_bytes),
            ],
            null_count=-1,
        )
        return pl.Series(name, arr)

    pa_type = _TAG_TO_PA[dtype_tag]
    min_valid_bytes = (n_out + 7) // 8
    # The kernel pads valid (and bool data) to 4-byte alignment for the
    # atomic_uint cast; trim to the Arrow-canonical minimum so PyArrow
    # doesn't see trailing garbage as part of the buffer.
    valid_trim = valid[:min_valid_bytes] if min_valid_bytes > 0 else b""
    if dtype_tag == "Bool":
        data_trim = data[:min_valid_bytes] if min_valid_bytes > 0 else b""
    elif dtype_tag in ("U32", "I32", "F32"):
        # 32-bit types: dense, exactly n_out * 4 bytes.
        data_trim = data[: n_out * 4]
    else:
        # I64/F64: dense, exactly n_out * 8 bytes.
        data_trim = data[: n_out * 8]

    if n_out == 0:
        # PyArrow accepts buffers of length 0 with size=0; pass empty data.
        arr = pa.Array.from_buffers(
            pa_type, 0, [pa.py_buffer(b""), pa.py_buffer(b"")], null_count=0
        )
    else:
        arr = pa.Array.from_buffers(
            pa_type,
            n_out,
            [pa.py_buffer(valid_trim), pa.py_buffer(data_trim)],
            # null_count=-1 → recompute on demand (safe; we don't know it
            # cheaply from the bit-packed bitmap on the Python side).
            null_count=-1,
        )
    return pl.Series(name, arr)


def _extract_scan_df_and_wire_plan(plan: dict) -> tuple[Any, dict]:
    """Walk the plan dict, extract the captured ``PyDataFrame`` from its
    (unique) Scan leaf, and rewrite the tree into the Rust wire format.

    The walker stores ``df`` and ``projection`` on the Scan as side channels;
    Rust's ``MetalPlanNode::Scan`` wants ``{kind, n_rows, columns}`` only. If a
    ``projection`` is present we lift it into a synthetic ``Project`` wrapper
    above the Scan so the Rust dispatch handles it uniformly.

    Returns ``(df_pydf, wire_plan)``.

    Multiple-Scan plans are not expected in M1 (no joins/unions yet); if one
    appears we raise — better to surface than to silently mis-route.
    """
    captured: list[Any] = []

    def rewrite(node: dict) -> dict:
        kind = node["kind"]
        if kind == "Scan":
            df = node["df"]
            projection = node.get("projection")
            captured.append(df)
            cols = node["columns"]
            n_rows = df.height()
            # When PyDataFrame has no `height` attribute (older Polars), fall
            # back to len(); but py-1.40.1 exposes `.height()`.
            scan_wire = {
                "kind": "Scan",
                "n_rows": n_rows,
                # Rust extracts each entry as a 2-tuple (name, dtype); a list
                # of [name, dtype] also extract()s as a 2-tuple, but we
                # normalize to tuples for clarity.
                "columns": [(str(name), str(dtype)) for name, dtype in cols],
            }
            if projection is None:
                return scan_wire
            return {
                "kind": "Project",
                "columns": list(projection),
                "input": scan_wire,
            }
        if kind == "Project":
            return {
                "kind": "Project",
                "columns": list(node["columns"]),
                "input": rewrite(node["input"]),
            }
        if kind == "Filter":
            return {
                "kind": "Filter",
                "predicate": node["predicate"],
                "input": rewrite(node["input"]),
            }
        if kind == "GroupBy":
            return {
                "kind": "GroupBy",
                "input": rewrite(node["input"]),
                "keys": [list(k) for k in node["keys"]],
                "aggs": [dict(a) for a in node["aggs"]],
            }
        if kind == "HStack":
            return {
                "kind": "HStack",
                "input": rewrite(node["input"]),
                "exprs": [dict(e) for e in node["exprs"]],
            }
        raise ValueError(f"unknown plan kind: {kind!r}")

    wire = rewrite(plan)
    if len(captured) != 1:
        raise RuntimeError(
            f"polars_metal: expected exactly one Scan leaf in plan, got {len(captured)}. "
            "Multi-scan plans (joins/unions) are not supported in M1."
        )
    return captured[0], wire
