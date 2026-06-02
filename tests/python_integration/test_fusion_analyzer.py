"""M4 Phase 3 Task 16: fusion analyzer recognizes supported expression shapes."""

import polars as pl

from polars_metal._fusion_analyzer import analyze_expression


def test_simple_arithmetic_chain_is_recognized():
    expr = (pl.col("a") + pl.col("b")) * pl.col("c")
    schema = {"a": pl.Float32, "b": pl.Float32, "c": pl.Float32}
    scope = analyze_expression(expr, schema)
    assert scope is not None
    assert scope.n_inputs() == 3
    assert scope.n_ops() == 2  # Add, Mul


def test_transcendental_chain_is_recognized():
    expr = pl.col("a").log().sqrt()
    schema = {"a": pl.Float32}
    scope = analyze_expression(expr, schema)
    assert scope is not None
    # Log is 2-arg (input + base literal); the literal becomes a synthetic
    # input. 2 ops: Log, Sqrt.
    assert scope.n_ops() == 2


def test_string_op_is_rejected():
    expr = pl.col("s").str.len_chars()
    schema = {"s": pl.Utf8}
    scope = analyze_expression(expr, schema)
    assert scope is None


def test_sum_reduction_is_recognized():
    expr = pl.col("a").sum()
    schema = {"a": pl.Float32}
    scope = analyze_expression(expr, schema)
    assert scope is not None
    assert scope.n_ops() == 1


def test_mixed_dtype_with_cast_is_recognized():
    expr = pl.col("a").cast(pl.Float32).sin()
    schema = {"a": pl.Float64}
    scope = analyze_expression(expr, schema)
    assert scope is not None
    assert scope.n_ops() == 2  # CastF32, Sin


def test_unsupported_op_in_middle_truncates_scope():
    expr = pl.col("s").str.to_lowercase().str.len_chars().cast(pl.Float32).sqrt()
    schema = {"s": pl.Utf8}
    scope = analyze_expression(expr, schema)
    assert scope is None


def test_when_then_otherwise():
    expr = pl.when(pl.col("a") > 0).then(pl.col("b")).otherwise(pl.col("c"))
    schema = {"a": pl.Float32, "b": pl.Float32, "c": pl.Float32}
    scope = analyze_expression(expr, schema)
    assert scope is not None
    # Predicate: Gt + lit; then/else: cols. Top-level: Where.
    # The Gt and Where ops are the two compute ops here.
    assert scope.n_ops() >= 2


def test_negate_via_unary_minus():
    expr = -pl.col("a")
    schema = {"a": pl.Float32}
    scope = analyze_expression(expr, schema)
    assert scope is not None
    assert scope.n_ops() == 1  # Neg


def test_log2_routes_to_log2_op():
    expr = pl.col("a").log(2.0)
    schema = {"a": pl.Float32}
    scope = analyze_expression(expr, schema)
    assert scope is not None
    assert scope.n_ops() == 1  # Log2 (base 2 detected from literal)


def test_log10_routes_to_log10_op():
    expr = pl.col("a").log10()
    schema = {"a": pl.Float32}
    scope = analyze_expression(expr, schema)
    assert scope is not None
    assert scope.n_ops() == 1  # Log10


def test_logical_and():
    expr = (pl.col("a") > 0) & (pl.col("b") < 1)
    schema = {"a": pl.Float32, "b": pl.Float32}
    scope = analyze_expression(expr, schema)
    assert scope is not None
    # Gt, Lt, LogicalAnd
    assert scope.n_ops() == 3


def test_abs():
    expr = pl.col("a").abs()
    schema = {"a": pl.Float32}
    scope = analyze_expression(expr, schema)
    assert scope is not None
    assert scope.n_ops() == 1
