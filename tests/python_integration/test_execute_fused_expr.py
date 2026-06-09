"""M4 Phase 5 + Phase 6: end-to-end execute_fused_expr.

Phase 6 evolved the FFI to zero-copy I/O: inputs are passed as float32 numpy
arrays (zero-copy ingest when their backing buffer is page-aligned) and the
output is written directly into a caller-allocated float32 numpy array. The
call returns the number of elements written.
"""

import numpy as np
import polars as pl
import polars_metal._native as native


def _f32(series: pl.Series) -> np.ndarray:
    return np.ascontiguousarray(series.to_numpy(), dtype=np.float32)


def _ptr(arr: np.ndarray) -> tuple[int, int, int]:
    """(data pointer, element count, dtype tag) — the executor's zero-copy
    input form. These tests are all F32; tag 0 = MlxDtype::F32."""
    return (int(arr.__array_interface__["data"][0]), int(arr.size), 0)


def _run(scope, in_arrays, out):
    """Call the executor with (ptr, len, tag) triples, keeping arrays alive."""
    return native.execute_fused_expr(
        scope=scope,
        inputs=[_ptr(a) for a in in_arrays],
        out=_ptr(out),
    )


def test_sqrt_one_million_rows():
    n = 1_000_000
    input_col = pl.Series("a", [float((i % 100) ** 2) for i in range(n)], dtype=pl.Float32)

    scope = native.PyFusionScope()
    a = scope.add_input("a", "F32")
    s = scope.push_op("Sqrt", [a])
    scope.mark_output(s)

    out = np.empty(n, dtype=np.float32)
    written = _run(scope, [_f32(input_col)], out)
    assert written == n
    # sqrt of (0, 1, 4, 9, 16) at positions 0, 1, 2, 3, 4.
    expected_head = [0.0, 1.0, 2.0, 3.0, 4.0]
    for i, exp in enumerate(expected_head):
        assert abs(out[i] - exp) < 1e-5, f"row {i}: got {out[i]} expected {exp}"


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

    out = np.empty(n, dtype=np.float32)
    written = _run(scope, [_f32(a), _f32(b)], out)
    assert written == n
    expected = (a.to_numpy() + b.to_numpy()) ** 2
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_transcendental_chain():
    n = 1000
    a = pl.Series("a", np.linspace(0.1, 10.0, n, dtype=np.float32))

    scope = native.PyFusionScope()
    ai = scope.add_input("a", "F32")
    log_a = scope.push_op("Log", [ai])
    sq = scope.push_op("Sqrt", [log_a])
    scope.mark_output(sq)

    out = np.empty(n, dtype=np.float32)
    written = _run(scope, [_f32(a)], out)
    assert written == n
    expected = np.sqrt(np.log(a.to_numpy()))
    np.testing.assert_allclose(out, expected, atol=1e-4)
