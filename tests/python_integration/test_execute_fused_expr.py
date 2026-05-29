"""M4 Phase 5: end-to-end execute_fused_expr."""

import numpy as np
import polars as pl
import polars_metal._native as native


def _f32_to_bytes(series: pl.Series) -> bytes:
    return series.to_numpy().astype(np.float32).tobytes()


def _bytes_to_f32_series(buf: bytes, name: str) -> pl.Series:
    arr = np.frombuffer(buf, dtype=np.float32)
    return pl.Series(name, arr)


def test_sqrt_one_million_rows():
    n = 1_000_000
    input_col = pl.Series("a", [float((i % 100) ** 2) for i in range(n)], dtype=pl.Float32)

    scope = native.PyFusionScope()
    a = scope.add_input("a", "F32")
    s = scope.push_op("Sqrt", [a])
    scope.mark_output(s)

    result_bytes = native.execute_fused_expr(
        scope=scope,
        input_buffers=[_f32_to_bytes(input_col)],
    )
    result = _bytes_to_f32_series(result_bytes, "result")
    assert result.dtype == pl.Float32
    assert result.len() == n
    # sqrt of (0, 1, 4, 9, 16) at positions 0, 1, 2, 3, 4.
    expected_head = [0.0, 1.0, 2.0, 3.0, 4.0]
    for i, exp in enumerate(expected_head):
        assert abs(result[i] - exp) < 1e-5, f"row {i}: got {result[i]} expected {exp}"


def test_arithmetic_chain():
    n = 1000
    a = pl.Series("a", np.arange(n, dtype=np.float32) * 0.01)
    b = pl.Series("b", np.arange(n, dtype=np.float32) * 0.02)

    scope = native.PyFusionScope()
    ai = scope.add_input("a", "F32")
    bi = scope.add_input("b", "F32")
    sum_ab = scope.push_op("Add", [ai, bi])
    sq = scope.push_op("Square", [sum_ab])
    scope.mark_output(sq)

    result_bytes = native.execute_fused_expr(
        scope=scope,
        input_buffers=[_f32_to_bytes(a), _f32_to_bytes(b)],
    )
    result = _bytes_to_f32_series(result_bytes, "y")
    expected = (a.to_numpy() + b.to_numpy()) ** 2
    np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-5)


def test_transcendental_chain():
    n = 1000
    a = pl.Series("a", np.linspace(0.1, 10.0, n, dtype=np.float32))

    scope = native.PyFusionScope()
    ai = scope.add_input("a", "F32")
    log_a = scope.push_op("Log", [ai])
    sq = scope.push_op("Sqrt", [log_a])
    scope.mark_output(sq)

    result_bytes = native.execute_fused_expr(
        scope=scope,
        input_buffers=[_f32_to_bytes(a)],
    )
    result = _bytes_to_f32_series(result_bytes, "y")
    expected = np.sqrt(np.log(a.to_numpy()))
    np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-4)
