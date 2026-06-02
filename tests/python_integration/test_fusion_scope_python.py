"""M4 Phase 3 Task 15: Python can construct a FusionScope via the PyO3 binding."""

import polars_metal._native as native


def test_construct_simple_scope():
    scope = native.PyFusionScope()
    a = scope.add_input("a", "F32")
    b = scope.add_input("b", "F32")
    add = scope.push_op("Add", [a, b])
    scope.mark_output(add)
    assert scope.n_inputs() == 2
    assert scope.n_ops() == 1
    assert scope.est_flops(10_000_000) == 10_000_000


def test_unsupported_op_raises():
    scope = native.PyFusionScope()
    a = scope.add_input("a", "F32")
    try:
        scope.push_op("NotARealOp", [a])
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "NotARealOp" in str(e)


def test_arg_count_mismatch_raises():
    scope = native.PyFusionScope()
    a = scope.add_input("a", "F32")
    try:
        scope.push_op("Add", [a])  # Add needs 2 args
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "expects 2 args, got 1" in str(e)


def test_array_dtype_parses_dim():
    scope = native.PyFusionScope()
    emb = scope.add_input("emb", "ArrayF32(768)")
    assert scope.n_inputs() == 1
    assert emb == 0


def test_route_decision_strings():
    scope = native.PyFusionScope()
    a = scope.add_input("a", "F32")
    b = scope.add_input("b", "F32")
    scope.push_op("Add", [a, b])
    # 1 FLOP/row * 1_000 = 1e3 - below rows threshold
    assert "BelowRowsThreshold" in scope.route_decision(1_000)
    # 1 FLOP/row * 1M = 1e6 - above rows, below flops
    assert "BelowFlopsThreshold" in scope.route_decision(1_000_000)
