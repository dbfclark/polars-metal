"""M4 Phase 7: null-safety of the fused HStack (`with_columns`) path.

The fused MLX subgraph ingests each input column via `series.to_numpy()`,
which turns Polars nulls into NaN. Polars elementwise ops propagate nulls
(output is null wherever an input is null), so a fused `with_columns` over a
column containing nulls produced NaN where CPU Polars produces null.

Resolution (two modes):
  - **elementwise** (arithmetic / transcendental / cast / comparison chains):
    output is null iff *any* input column is null at that row. We combine the
    input columns' null masks and attach the result to the output Series — the
    transcendental chain stays on the GPU (still one dispatch).
  - **where** (`pl.when/then/otherwise`): data-dependent null mask, handled by
    a validity subgraph (see the where-specific tests once that lands).
  - everything else (Kleene And/Or, null-skipping reductions, scans): the
    walker refuses fusion and the subtree falls back to CPU (exact semantics).
"""

import polars as pl
from polars.testing import assert_frame_equal

import polars_metal
from polars_metal import _native


def _count_fused_dispatches(monkeypatch):
    """Install a counter over `execute_fused_expr`; returns a 0-arg getter."""
    state = {"n": 0}
    orig = _native.execute_fused_expr

    def counting(scope, inputs, out):
        state["n"] += 1
        return orig(scope=scope, inputs=inputs, out=out)

    monkeypatch.setattr(_native, "execute_fused_expr", counting)
    return lambda: state["n"]


def test_elementwise_null_input_matches_cpu():
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


def test_elementwise_null_input_still_uses_mlx(monkeypatch):
    """A null-bearing elementwise chain stays on the GPU: nulls are handled by
    combining input null masks, not by falling the subtree back to CPU."""
    df = pl.DataFrame(
        {
            "a": pl.Series([1.0, 2.0, None, 4.0], dtype=pl.Float32),
        }
    )
    expr = pl.col("a").sqrt() + pl.col("a").sin()
    n_dispatches = _count_fused_dispatches(monkeypatch)
    df.lazy().with_columns(y=expr).collect(engine=polars_metal.MetalEngine())
    assert n_dispatches() == 1, f"expected a single fused dispatch, got {n_dispatches()}"


def test_multicol_elementwise_null_input_matches_cpu():
    """Nulls in different rows of different input columns: output is null where
    *any* operand is null (union of the input null masks)."""
    df = pl.DataFrame(
        {
            "a": pl.Series([1.0, None, 3.0, 4.0, 5.0], dtype=pl.Float32),
            "b": pl.Series([10.0, 20.0, None, 40.0, 50.0], dtype=pl.Float32),
        }
    )
    expr = (pl.col("a").sqrt() * pl.col("b").cos()) + pl.col("a")
    cpu_result = df.lazy().with_columns(y=expr).collect()
    metal_result = df.lazy().with_columns(y=expr).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu_result, metal_result, check_exact=False, abs_tol=1e-4)


def test_reduction_with_null_input_falls_back(monkeypatch):
    """A fused chain with an embedded null-skipping reduction (`sum` skips
    nulls; MLX over NaN would not) must fall back to CPU when inputs have
    nulls — the elementwise null-mask rule does not apply to reductions."""
    df = pl.DataFrame(
        {
            "a": pl.Series([1.0, 2.0, None, 4.0], dtype=pl.Float32),
        }
    )
    expr = pl.col("a") / pl.col("a").sum()
    n_dispatches = _count_fused_dispatches(monkeypatch)
    cpu_result = df.lazy().with_columns(y=expr).collect()
    metal_result = df.lazy().with_columns(y=expr).collect(engine=polars_metal.MetalEngine())
    assert n_dispatches() == 0, f"expected CPU fallback (0 fused dispatches), got {n_dispatches()}"
    assert_frame_equal(cpu_result, metal_result, check_exact=False, abs_tol=1e-4)


def test_fused_hstack_null_free_still_uses_mlx(monkeypatch):
    """Regression guard: a null-free F32 column still routes through the fused
    MLX path with no per-row null-mask overhead."""
    df = pl.DataFrame(
        {
            "a": pl.Series([1.0, 2.0, 3.0, 4.0], dtype=pl.Float32),
        }
    )
    expr = pl.col("a").sqrt() + pl.col("a").sin()
    n_dispatches = _count_fused_dispatches(monkeypatch)
    cpu_result = df.lazy().with_columns(y=expr).collect()
    metal_result = df.lazy().with_columns(y=expr).collect(engine=polars_metal.MetalEngine())
    assert n_dispatches() == 1, f"expected a single fused dispatch, got {n_dispatches()}"
    assert_frame_equal(cpu_result, metal_result, check_exact=False, abs_tol=1e-4)
