"""M4 Phase 7 (Task 28): cum_sum routes through the fused MLX scan path.

`with_columns(cs=pl.col("x").cum_sum())` is an HStack binding, so it reuses the
Phase 6 `_dispatch_hstack_fused` path. The Rust side already wires OpId::CumSum
to mlx_cumsum; this exercises the analyzer recognizing the `cum_sum` Function
node. Reverse cumsum (no MLX forward-only binding) must fall back to CPU.
"""

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

import polars_metal
from polars_metal import _native


def _make_floats(n, seed=0xC57):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"x": rng.standard_normal(n).astype(np.float32)})


def _count_dispatches(monkeypatch):
    count = {"n": 0}
    orig = _native.execute_fused_expr

    def counting(scope, inputs, out):
        count["n"] += 1
        return orig(scope=scope, inputs=inputs, out=out)

    monkeypatch.setattr(_native, "execute_fused_expr", counting)
    return count


def test_cumsum_matches_cpu_single_dispatch(monkeypatch):
    df = _make_floats(50_000)
    count = _count_dispatches(monkeypatch)

    cpu = df.lazy().with_columns(cs=pl.col("x").cum_sum()).collect()
    metal = (
        df.lazy().with_columns(cs=pl.col("x").cum_sum()).collect(engine=polars_metal.MetalEngine())
    )

    assert count["n"] == 1, f"cum_sum should route to one fused dispatch, got {count['n']}"
    # cumsum accumulates 50k F32 adds; tolerance scales with magnitude.
    assert_frame_equal(cpu, metal, check_exact=False, abs_tol=1e-2, rel_tol=1e-4)


def test_cumsum_in_a_chain_matches_cpu(monkeypatch):
    # cum_sum as a terminus of an elementwise chain: (x*2).cum_sum()
    df = _make_floats(20_000)
    count = _count_dispatches(monkeypatch)
    expr = (pl.col("x") * 2.0).cum_sum()

    cpu = df.lazy().with_columns(cs=expr).collect()
    metal = df.lazy().with_columns(cs=expr).collect(engine=polars_metal.MetalEngine())

    assert count["n"] == 1, f"chain+cum_sum should be one dispatch, got {count['n']}"
    assert_frame_equal(cpu, metal, check_exact=False, abs_tol=1e-2, rel_tol=1e-4)


def test_reverse_cumsum_falls_back_to_cpu(monkeypatch):
    df = _make_floats(10_000)
    count = _count_dispatches(monkeypatch)

    cpu = df.lazy().with_columns(cs=pl.col("x").cum_sum(reverse=True)).collect()
    metal = (
        df.lazy()
        .with_columns(cs=pl.col("x").cum_sum(reverse=True))
        .collect(engine=polars_metal.MetalEngine())
    )

    assert count["n"] == 0, "reverse cum_sum has no MLX binding; must fall back to CPU"
    assert_frame_equal(cpu, metal)
