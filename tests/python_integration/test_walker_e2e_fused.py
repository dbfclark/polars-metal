"""M4 Phase 5 Task 23: end-to-end fused subgraph execution.

`df.with_columns(...).collect(engine="metal")` runs through the fused MLX
subgraph and returns a result equal to `engine="cpu"`.
"""

import polars as pl
from polars.testing import assert_frame_equal

import polars_metal
from polars_metal import _native


def test_sqrt_chain_e2e():
    n = 1024
    df = pl.DataFrame(
        {
            "a": pl.Series([float(i % 100) for i in range(n)], dtype=pl.Float32),
            "b": pl.Series([float((i * 7) % 256) for i in range(n)], dtype=pl.Float32),
        }
    )
    expr = (pl.col("a").sqrt() + pl.col("b").sqrt()).cos()
    cpu_result = df.lazy().with_columns(y=expr).collect()
    metal_result = df.lazy().with_columns(y=expr).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu_result, metal_result, check_exact=False, abs_tol=1e-4)


def test_transcendental_chain_e2e():
    n = 2048
    df = pl.DataFrame(
        {
            "a": pl.Series([0.1 + float(i) * 0.001 for i in range(n)], dtype=pl.Float32),
        }
    )
    expr = pl.col("a").log().sqrt()
    cpu_result = df.lazy().with_columns(y=expr).collect()
    metal_result = df.lazy().with_columns(y=expr).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu_result, metal_result, check_exact=False, abs_tol=1e-4)


def test_cse_heavy_expr_is_single_dispatch(monkeypatch):
    """A compute subtree with shared subexpressions must fuse to ONE MLX
    dispatch, not fragment into one-per-CSE-temp (CLAUDE.md principle #1).

    The engine forces `comm_subexpr_elim` off for Metal-routed plans so
    Polars' CSE pass doesn't hoist `pickup_lat * d2r` / `drop_lat * d2r`
    into `__POLARS_CSER_*` temp columns — each of which would otherwise
    become its own `execute_fused_expr` call with its result round-tripping
    Series→Metal→Series between dispatches. Regression guard for the
    haversine de-fragmentation (3 dispatches → 1).
    """
    n = 4096
    df = pl.DataFrame(
        {
            "pickup_lat": pl.Series([40.6 + (i % 100) * 0.001 for i in range(n)], dtype=pl.Float32),
            "pickup_lon": pl.Series(
                [-74.0 + (i % 100) * 0.001 for i in range(n)], dtype=pl.Float32
            ),
            "drop_lat": pl.Series([40.7 + (i % 100) * 0.001 for i in range(n)], dtype=pl.Float32),
            "drop_lon": pl.Series([-73.9 + (i % 100) * 0.001 for i in range(n)], dtype=pl.Float32),
        }
    )
    d2r = 0.017453292519943295
    pla = pl.col("pickup_lat") * d2r  # reused in dlat and cos -> CSE candidate
    dla = pl.col("drop_lat") * d2r  # reused in dlat and cos -> CSE candidate
    dlat = (dla - pla) / 2.0
    dlon = (pl.col("drop_lon") - pl.col("pickup_lon")) * d2r / 2.0
    expr = dlat.sin() ** 2 + pla.cos() * dla.cos() * dlon.sin() ** 2

    dispatch_count = 0
    orig = _native.execute_fused_expr

    def counting(scope, input_buffers):
        nonlocal dispatch_count
        dispatch_count += 1
        return orig(scope=scope, input_buffers=input_buffers)

    monkeypatch.setattr(_native, "execute_fused_expr", counting)

    cpu_result = df.lazy().with_columns(d=expr).collect()
    metal_result = df.lazy().with_columns(d=expr).collect(engine=polars_metal.MetalEngine())

    assert dispatch_count == 1, f"expected a single fused dispatch, got {dispatch_count}"
    assert_frame_equal(cpu_result, metal_result, check_exact=False, abs_tol=1e-4)
