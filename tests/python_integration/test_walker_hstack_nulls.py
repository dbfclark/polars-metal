"""M4 Phase 7: null-safety of the fused HStack (`with_columns`) path.

The fused MLX subgraph ingests each input column via `series.to_numpy()`,
which turns Polars nulls into NaN. Polars elementwise ops propagate nulls
(output is null wherever an input is null), so a fused `with_columns` over a
column containing nulls produced NaN where CPU Polars produces null — a
correctness divergence in the already-shipped headline path
(haversine / black-scholes shape).

The fused reduction path (Task 26) already guards this by falling back to a
Polars reduction when the source column has nulls. The HStack path can't
replay an arbitrary transcendental chain on CPU from the wire plan, so the
walker refuses fusion when an input column has nulls and the whole subtree
falls back to CPU (correct null semantics, conformance-style assertion below).
"""

import polars as pl
from polars.testing import assert_frame_equal

import polars_metal
from polars_metal import _native


def test_fused_hstack_with_null_input_matches_cpu():
    """A transcendental `with_columns` over an F32 column containing nulls
    must equal the CPU result (nulls preserved, not turned into NaN)."""
    df = pl.DataFrame(
        {
            "a": pl.Series([1.0, 2.0, None, 4.0, None, 6.0, 7.0, 8.0], dtype=pl.Float32),
        }
    )
    expr = pl.col("a").sqrt() + pl.col("a").sin()
    cpu_result = df.lazy().with_columns(y=expr).collect()
    metal_result = df.lazy().with_columns(y=expr).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu_result, metal_result, check_exact=False, abs_tol=1e-4)


def test_fused_hstack_with_null_input_does_not_use_mlx(monkeypatch):
    """When an input column has nulls, the fused MLX path must NOT run — the
    walker falls the whole subtree back to CPU so null semantics are exact."""
    df = pl.DataFrame(
        {
            "a": pl.Series([1.0, 2.0, None, 4.0], dtype=pl.Float32),
        }
    )
    expr = pl.col("a").sqrt() + pl.col("a").sin()

    dispatch_count = 0
    orig = _native.execute_fused_expr

    def counting(scope, inputs, out):
        nonlocal dispatch_count
        dispatch_count += 1
        return orig(scope=scope, inputs=inputs, out=out)

    monkeypatch.setattr(_native, "execute_fused_expr", counting)

    df.lazy().with_columns(y=expr).collect(engine=polars_metal.MetalEngine())
    assert dispatch_count == 0, f"expected CPU fallback (0 fused dispatches), got {dispatch_count}"


def test_fused_hstack_null_free_still_uses_mlx(monkeypatch):
    """Regression guard: a null-free F32 column still routes through the fused
    MLX path (the null guard must not over-fall-back on clean inputs)."""
    df = pl.DataFrame(
        {
            "a": pl.Series([1.0, 2.0, 3.0, 4.0], dtype=pl.Float32),
        }
    )
    expr = pl.col("a").sqrt() + pl.col("a").sin()

    dispatch_count = 0
    orig = _native.execute_fused_expr

    def counting(scope, inputs, out):
        nonlocal dispatch_count
        dispatch_count += 1
        return orig(scope=scope, inputs=inputs, out=out)

    monkeypatch.setattr(_native, "execute_fused_expr", counting)

    cpu_result = df.lazy().with_columns(y=expr).collect()
    metal_result = df.lazy().with_columns(y=expr).collect(engine=polars_metal.MetalEngine())
    assert dispatch_count == 1, f"expected a single fused dispatch, got {dispatch_count}"
    assert_frame_equal(cpu_result, metal_result, check_exact=False, abs_tol=1e-4)
