"""Property-based tests for the IR-arena program synthesizer.

The `analyze_ir_with_columns` walker is the layer that translates a Polars
IR expression tree into a `PyFusionScope` of MLX ops. A bug in this layer
silently produces wrong numeric results (the executor has no way to know
the synthesized scope doesn't match the intended expression). These tests
synthesise random F32 expressions from the analyzer's supported grammar,
build the scope via the analyzer, execute it via `execute_fused_expr`,
and compare against Polars CPU evaluation of the same expression on the
same data. Any mismatch beyond F32 tolerance is a synthesis bug.

This test specifically catches the kind of input/op NodeIdx collision the
two-pass `_gather_leaves_ir` + `_visit_ir_ops` design in `_fusion_analyzer.py`
exists to prevent.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import polars_metal._native as native
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from polars_metal._fusion_analyzer import analyze_ir_with_columns

_COLS = ("a", "b", "c")

# Each entry: (label, polars_op, numpy_op, well_defined_predicate).
# Polars CPU vs MLX-on-Metal can disagree on the exact F32 bit pattern at
# domain boundaries (e.g. arcsin(1.0+1e-7) → NaN in one but not the other);
# the predicate filters input rows where the answer is well-conditioned.
_UNARY_OPS: list[
    tuple[str, Callable[[pl.Expr], pl.Expr], Callable[[float], float], Callable[[float], bool]]
] = [
    ("sqrt", lambda e: e.sqrt(), np.sqrt, lambda x: x >= 0.0),
    ("abs", lambda e: e.abs(), np.abs, lambda _x: True),
    ("neg", lambda e: -e, np.negative, lambda _x: True),
    ("sin", lambda e: e.sin(), np.sin, lambda _x: True),
    ("cos", lambda e: e.cos(), np.cos, lambda _x: True),
    # tan(x) restricted to |tan(x)| <= 5: bigger values feed downstream sin/cos
    # with large arguments, where libm vs MLX modular-reduction differs in F32.
    ("tan", lambda e: e.tan(), np.tan, lambda x: abs(math.cos(x)) > 0.2),
    ("sinh", lambda e: e.sinh(), np.sinh, lambda x: abs(x) < 5.0),
    ("cosh", lambda e: e.cosh(), np.cosh, lambda x: abs(x) < 5.0),
    ("tanh", lambda e: e.tanh(), np.tanh, lambda _x: True),
    ("arcsin", lambda e: e.arcsin(), np.arcsin, lambda x: -0.99 <= x <= 0.99),
    ("arccos", lambda e: e.arccos(), np.arccos, lambda x: -0.99 <= x <= 0.99),
    ("arctan", lambda e: e.arctan(), np.arctan, lambda _x: True),
    # exp restricted so feeding into cos/sin keeps arguments small enough
    # that F32 modular reduction agrees between libm and MLX.
    ("exp", lambda e: e.exp(), np.exp, lambda x: x < 3.0),
    ("log", lambda e: e.log(), np.log, lambda x: x > 1e-3),
    ("log2", lambda e: e.log(2.0), np.log2, lambda x: x > 1e-3),
    ("log10", lambda e: e.log10(), np.log10, lambda x: x > 1e-3),
]

_BINARY_OPS: list[
    tuple[
        str,
        Callable[[pl.Expr, pl.Expr], pl.Expr],
        Callable[[float, float], float],
        Callable[[float, float], bool],
    ]
] = [
    ("add", lambda lhs, rhs: lhs + rhs, np.add, lambda _x, _y: True),
    ("sub", lambda lhs, rhs: lhs - rhs, np.subtract, lambda _x, _y: True),
    ("mul", lambda lhs, rhs: lhs * rhs, np.multiply, lambda _x, _y: True),
    ("div", lambda lhs, rhs: lhs / rhs, np.divide, lambda _x, y: abs(y) > 1e-2),
]


# An evaluator takes a row dict and returns (value, is_well_defined).
RowEval = Callable[[dict[str, float]], tuple[float, bool]]


@st.composite
def _expr_strategy(draw: Any, max_depth: int = 4) -> tuple[pl.Expr, RowEval]:
    """Generate a random F32 expression of bounded depth, paired with a
    Python-side evaluator that returns (value, is_well_defined) for any
    candidate input row. The well-defined flag lets the test filter rows
    where the expression would hit a domain boundary that Polars and MLX
    interpret with different NaN bit patterns."""

    def build(depth: int) -> tuple[pl.Expr, RowEval]:
        if depth <= 0:
            return _leaf(draw)

        kind = draw(st.sampled_from(["col", "lit", "unary", "binary"]))
        if kind == "col" or kind == "lit":
            return _leaf(draw)

        if kind == "unary":
            _label, polars_op, np_op, well_defined = draw(st.sampled_from(_UNARY_OPS))
            sub_expr, sub_eval = build(depth - 1)

            def eval_unary(
                row: dict[str, float],
                sub_eval: RowEval = sub_eval,
                np_op: Callable[[float], float] = np_op,
                well_defined: Callable[[float], bool] = well_defined,
            ) -> tuple[float, bool]:
                v, ok = sub_eval(row)
                if not ok or not well_defined(v):
                    return float("nan"), False
                result = float(np_op(v))
                if not math.isfinite(result):
                    return result, False
                return result, True

            return polars_op(sub_expr), eval_unary

        # binary
        _label, polars_op, np_op, well_defined = draw(st.sampled_from(_BINARY_OPS))
        left_expr, left_eval = build(depth - 1)
        right_expr, right_eval = build(depth - 1)

        def eval_binary(
            row: dict[str, float],
            left_eval: RowEval = left_eval,
            right_eval: RowEval = right_eval,
            np_op: Callable[[float, float], float] = np_op,
            well_defined: Callable[[float, float], bool] = well_defined,
        ) -> tuple[float, bool]:
            lv, lok = left_eval(row)
            rv, rok = right_eval(row)
            if not (lok and rok and well_defined(lv, rv)):
                return float("nan"), False
            result = float(np_op(lv, rv))
            if not math.isfinite(result):
                return result, False
            return result, True

        return polars_op(left_expr, right_expr), eval_binary

    depth = draw(st.integers(min_value=1, max_value=max_depth))
    return build(depth)


def _leaf(draw: Any) -> tuple[pl.Expr, RowEval]:
    choice = draw(st.sampled_from(["col", "lit"]))
    if choice == "col":
        name = draw(st.sampled_from(_COLS))
        return pl.col(name), lambda row, n=name: (row[n], True)
    val = float(draw(st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False)))
    return pl.lit(val), lambda _row, v=val: (v, True)


def _find_hstack_with_output(nt: Any, output_name: str) -> int | None:
    """DFS for the HStack whose first expression's output_name matches.

    Polars CSE optimization may introduce intermediate HStacks with
    `__POLARS_CSER_...` synthetic columns; we want the one carrying our
    actual target column, not the CSE wrappers."""
    node = nt.view_current_node()
    if type(node).__name__ == "HStack":
        try:
            if any(getattr(e, "output_name", None) == output_name for e in node.exprs):
                return nt.get_node()
        except AttributeError:
            pass
    for child in nt.get_inputs():
        nt.set_node(child)
        r = _find_hstack_with_output(nt, output_name)
        if r is not None:
            return r
    return None


def _analyzer_eval(
    expr: pl.Expr,
    inputs: dict[str, np.ndarray],
) -> np.ndarray | None:
    """Build a fusion scope from `expr` via the IR-arena analyzer and run
    it through `execute_fused_expr`. Returns None if the analyzer rejects
    the expression."""
    # Disable CSE so the IR walker doesn't hand us synthetic
    # `__POLARS_CSER_...` intermediates that our test inputs don't carry.
    # The analyzer must produce a correct scope on the user's original
    # expression tree regardless of CSE; CSE handling lives at the
    # walker/dispatcher boundary, not the synthesizer.
    df = pl.DataFrame(inputs).lazy().with_columns(y=expr)
    flags = pl.QueryOptFlags(comm_subexpr_elim=False)
    ldf = df._ldf.with_optimizations(flags._pyoptflags)
    nt = ldf.visit()
    hstack_id = _find_hstack_with_output(nt, "y")
    if hstack_id is None:
        return None
    nt.set_node(hstack_id)
    hstack = nt.view_current_node()
    # Find the binding whose output_name is 'y' (CSE-emitted intermediates
    # may share an HStack with our target column).
    expr_meta = next((e for e in hstack.exprs if getattr(e, "output_name", None) == "y"), None)
    if expr_meta is None:
        return None
    upstream_id = nt.get_inputs()[0]
    nt.set_node(upstream_id)
    schema = dict(nt.get_schema())
    nt.set_node(hstack_id)

    result = analyze_ir_with_columns(nt, expr_meta.node, schema)
    if result is None:
        return None
    scope, descriptors = result

    n_rows = len(next(iter(inputs.values())))
    input_buffers: list[bytes] = []
    for kind, payload in descriptors:
        if kind == "col":
            arr = inputs[payload].astype(np.float32, copy=False)
            input_buffers.append(arr.tobytes())
        elif kind == "lit":
            input_buffers.append(np.full(n_rows, payload, dtype=np.float32).tobytes())
        else:
            raise AssertionError(f"unknown descriptor kind {kind!r}")

    out_bytes = native.execute_fused_expr(scope=scope, input_buffers=input_buffers)
    return np.frombuffer(out_bytes, dtype=np.float32).copy()


@given(expr_and_eval=_expr_strategy(max_depth=5))
@settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_analyzer_matches_polars_cpu_on_random_f32_expressions(
    expr_and_eval: tuple[pl.Expr, RowEval],
) -> None:
    """For a randomly-generated F32 expression tree, the IR-arena analyzer's
    synthesized scope must produce the same numeric result as Polars CPU.
    """
    expr, well_defined = expr_and_eval
    rng = np.random.default_rng(0xC0FFEE)
    n_rows = 64
    a = rng.uniform(-2.0, 2.0, size=n_rows).astype(np.float32)
    b = rng.uniform(-2.0, 2.0, size=n_rows).astype(np.float32)
    c = rng.uniform(-2.0, 2.0, size=n_rows).astype(np.float32)

    mask = np.array(
        [
            well_defined({"a": float(a[i]), "b": float(b[i]), "c": float(c[i])})[1]
            for i in range(n_rows)
        ]
    )
    assume(mask.sum() >= 8)
    a, b, c = a[mask], b[mask], c[mask]
    inputs = {"a": a, "b": b, "c": c}

    cpu_df = pl.DataFrame(inputs).lazy().with_columns(y=expr).collect()
    cpu_y = cpu_df["y"].to_numpy().astype(np.float32, copy=False)

    analyzer_y = _analyzer_eval(expr, inputs)
    if analyzer_y is None:
        # Analyzer rejected — fine, no claim to make.
        return

    finite = np.isfinite(cpu_y) & np.isfinite(analyzer_y)
    assume(finite.sum() >= 8)
    # F32 transcendentals can differ by ~1-2 ulps between libm (Polars) and
    # MLX/Metal; rtol covers proportional error, atol catches near-zero
    # noise. Keep the bar tight enough to catch real synthesis bugs.
    np.testing.assert_allclose(
        analyzer_y[finite],
        cpu_y[finite],
        rtol=1e-3,
        atol=1e-4,
        err_msg=(
            f"Analyzer-built scope diverged from Polars CPU.\n"
            f"  expr: {expr}\n"
            f"  cpu[0:8]={cpu_y[:8]}\n"
            f"  analyzer[0:8]={analyzer_y[:8]}\n"
        ),
    )


@given(expr_and_eval=_expr_strategy(max_depth=4))
@settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_engine_metal_matches_polars_cpu_on_random_f32_expressions(
    expr_and_eval: tuple[pl.Expr, RowEval],
) -> None:
    """End-to-end property test: `df.collect(engine='metal')` must match
    `df.collect()` on random F32 expressions.

    This is broader than `test_analyzer_matches_polars_cpu_*` — it
    exercises the walker, the router, the UDF dispatcher, and the
    fused-binding path together. The analyzer test would have caught
    the synthesis bug in isolation; this test catches dispatcher /
    routing regressions that only manifest when the whole pipeline runs.
    """
    import polars_metal  # local import to avoid pulling Metal at module load

    expr, well_defined = expr_and_eval
    rng = np.random.default_rng(0xFA11C0DE)
    n_rows = 64
    a = rng.uniform(-2.0, 2.0, size=n_rows).astype(np.float32)
    b = rng.uniform(-2.0, 2.0, size=n_rows).astype(np.float32)
    c = rng.uniform(-2.0, 2.0, size=n_rows).astype(np.float32)

    mask = np.array(
        [
            well_defined({"a": float(a[i]), "b": float(b[i]), "c": float(c[i])})[1]
            for i in range(n_rows)
        ]
    )
    assume(mask.sum() >= 8)
    a, b, c = a[mask], b[mask], c[mask]
    df = pl.DataFrame({"a": a, "b": b, "c": c})

    cpu_y = df.lazy().with_columns(y=expr).collect()["y"].to_numpy().astype(np.float32, copy=False)
    metal_y = (
        df.lazy()
        .with_columns(y=expr)
        .collect(engine=polars_metal.MetalEngine())["y"]
        .to_numpy()
        .astype(np.float32, copy=False)
    )

    finite = np.isfinite(cpu_y) & np.isfinite(metal_y)
    assume(finite.sum() >= 8)
    np.testing.assert_allclose(
        metal_y[finite],
        cpu_y[finite],
        rtol=1e-3,
        atol=1e-4,
        err_msg=(
            f"engine='metal' diverged from Polars CPU.\n"
            f"  expr: {expr}\n"
            f"  cpu[0:8]={cpu_y[:8]}\n"
            f"  metal[0:8]={metal_y[:8]}\n"
        ),
    )


def test_engine_metal_handles_haversine_e2e() -> None:
    """End-to-end regression: the haversine expression triggers CSE which
    introduces a `Project(HStack(HStack(...)))` plan shape. The dispatcher
    must peel the Project root and recurse through chained HStacks.

    Originally caught while running the headline M4 haversine benchmark
    (10M F32 rows) at the engine='metal' path; left in as a regression
    against `_udf._dispatch` losing the Project(HStack(...)) peel."""
    import polars_metal

    N = 1024
    rng = np.random.default_rng(0)
    df = pl.DataFrame(
        {
            "pickup_lat": rng.uniform(40.6, 40.9, size=N).astype(np.float32),
            "pickup_lon": rng.uniform(-74.05, -73.7, size=N).astype(np.float32),
            "drop_lat": rng.uniform(40.6, 40.9, size=N).astype(np.float32),
            "drop_lon": rng.uniform(-74.05, -73.7, size=N).astype(np.float32),
        }
    )
    R = 6371.0
    d2r = float(np.pi / 180.0)
    pla = pl.col("pickup_lat") * d2r
    dla = pl.col("drop_lat") * d2r
    dlat = (dla - pla) / 2.0
    dlon = (pl.col("drop_lon") - pl.col("pickup_lon")) * d2r / 2.0
    a = dlat.sin() ** 2 + pla.cos() * dla.cos() * dlon.sin() ** 2
    expr = 2.0 * R * a.sqrt().arcsin()

    cpu_d = df.lazy().with_columns(d=expr).collect()["d"].to_numpy()
    metal_d = (
        df.lazy().with_columns(d=expr).collect(engine=polars_metal.MetalEngine())["d"].to_numpy()
    )
    np.testing.assert_allclose(metal_d, cpu_d, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize(
    "expr_builder, expected_value, description",
    [
        # The original Phase 5 bug: cos(sqrt(a) + sqrt(b)).
        # Inputs interleaved with ops in single-pass DFS → NodeIdx collision.
        (
            lambda: (pl.col("a").sqrt() + pl.col("b").sqrt()).cos(),
            lambda a, b: np.cos(np.sqrt(a) + np.sqrt(b)),
            "cos(sqrt(a)+sqrt(b)) — original Phase 5 regression",
        ),
        # Deeper nesting that exercises >2 inputs interleaved with ops.
        (
            lambda: (pl.col("a").sqrt() + pl.col("b")).cos() * pl.col("c").sin(),
            lambda a, b, c: np.cos(np.sqrt(a) + b) * np.sin(c),
            "(cos(sqrt(a)+b)*sin(c)) — three-input interleave",
        ),
        # Literal between two columns to ensure literal idx allocation works.
        (
            lambda: (pl.col("a") + pl.lit(3.0)) * pl.col("b"),
            lambda a, b: (a + 3.0) * b,
            "(a+3)*b — literal between columns",
        ),
    ],
)
def test_analyzer_known_regressions(
    expr_builder: Any, expected_value: Any, description: str
) -> None:
    """Regression cases that exercise the analyzer's input/op ordering."""
    a = np.array([4.0, 1.0, 0.25, 9.0], dtype=np.float32)
    b = np.array([9.0, 1.0, 4.0, 16.0], dtype=np.float32)
    c = np.array([0.5, 1.0, 1.5, 0.25], dtype=np.float32)
    inputs = {"a": a, "b": b, "c": c}
    expr = expr_builder()
    analyzer_y = _analyzer_eval(expr, inputs)
    assert analyzer_y is not None, f"analyzer rejected expr: {description}"
    n_args = expected_value.__code__.co_argcount
    args = [a, b, c][:n_args]
    expected = expected_value(*args).astype(np.float32)
    np.testing.assert_allclose(analyzer_y, expected, rtol=1e-3, atol=1e-4, err_msg=description)
