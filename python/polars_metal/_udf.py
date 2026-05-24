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
    result_pydf = _native.execute_plan(df_pydf, wire_plan)
    return pl.DataFrame._from_pydf(result_pydf)


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


def _build_groupby(plan: dict) -> Any:
    """Return a UDF callable for a GroupBy plan.

    The plan dict's ``input`` subtree resolves to the upstream DataFrame.
    For Phase 8, the supported input shape is ``Scan`` directly under GroupBy
    (or any other scan/project/filter shape that ``_extract_scan_df_and_wire_plan``
    already handles). We extract the underlying ``df_pydf`` and the full wire
    plan (including the GroupBy root node) and dispatch to ``_dispatch_groupby``
    at call time.
    """
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
    # input subtree. We run the input subtree first to get the actual
    # upstream DataFrame (which may involve scan/project/filter).
    upstream_plan = wire_plan["input"]
    if upstream_plan["kind"] == "Scan":
        upstream = pl.DataFrame._from_pydf(df_pydf)
    else:
        upstream_pydf = _native.execute_plan(df_pydf, upstream_plan)
        upstream = pl.DataFrame._from_pydf(upstream_pydf)

    n_rows = upstream.height

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

    # Reassemble. out_columns is a list of (name, dtype_tag, data, valid).
    # All output columns share the same n_out (number of groups); we determine
    # it from the first non-Bool column or from the valid bitmap length for
    # Bool columns.
    series_list: list[pl.Series] = []
    n_out: int | None = None

    for col_entry in out_columns:
        name, dtype_tag, data, valid = col_entry
        # Determine the row count from this column.
        if dtype_tag in ("U32", "I32", "F32"):
            n_this = len(data) // 4
        elif dtype_tag in ("I64", "F64"):
            n_this = len(data) // 8
        elif dtype_tag == "Bool":
            # Bool data is bit-packed; use n_out from a previous column if
            # available, otherwise estimate from valid-bitmap byte count
            # (conservative upper bound, refined once we see a non-Bool col).
            n_this = n_out if n_out is not None else len(valid) * 8
        else:
            raise RuntimeError(f"polars_metal: unexpected dtype_tag {dtype_tag!r}")

        if n_out is None:
            n_out = n_this

        series_list.append(_assemble_series(name, dtype_tag, data, valid, n_out))

    if not series_list:
        return pl.DataFrame()
    return pl.DataFrame(series_list)


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

    lhs_arr = _materialize_arrow(upstream.get_column(lhs["name"]))
    lhs_data, lhs_valid = _data_and_valid_for_dtype(lhs_arr, n_rows, dtype)

    if rhs["kind"] == "Column":
        rhs_arr = _materialize_arrow(upstream.get_column(rhs["name"]))
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
}

_TAG_TO_PA: dict[str, pa.DataType] = {
    "I64": pa.int64(),
    "F64": pa.float64(),
    "Bool": pa.bool_(),
    "U32": pa.uint32(),
    "I32": pa.int32(),
    "F32": pa.float32(),
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
    """
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

    The validity buffer is always present (the kernel writes one); if the
    column has no nulls we still hand a buffer to PyArrow rather than
    None — PyArrow accepts that and reports the correct null_count via
    ``null_count=-1`` (compute on demand).
    """
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
        raise ValueError(f"unknown plan kind: {kind!r}")

    wire = rewrite(plan)
    if len(captured) != 1:
        raise RuntimeError(
            f"polars_metal: expected exactly one Scan leaf in plan, got {len(captured)}. "
            "Multi-scan plans (joins/unions) are not supported in M1."
        )
    return captured[0], wire
