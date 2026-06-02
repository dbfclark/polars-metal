"""M4 Phase 7 Task 27: route single-column F32 Sort / top_k through MLX.

`df.sort("x")` and `df.top_k(k, by="x")` both lower to a Polars `Sort` IR node
(top_k = sort descending + `slice=(0,k)`). For a single F32 column the walker
routes the sort to an MLX `Sort` op (one fused dispatch); the host then applies
descending (reverse) and any slice.

MLX sort is unstable, so correctness is asserted by sorted *values*, never row
identity. Multi-column sorts (a bandwidth-bound gather), non-F32 columns, and
null-bearing columns fall back to CPU — correct, just not on the GPU.
"""

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

import polars_metal
from polars_metal import _native


def _count_fused_dispatches(monkeypatch):
    state = {"n": 0}
    orig = _native.execute_fused_expr

    def counting(scope, inputs, out):
        state["n"] += 1
        return orig(scope=scope, inputs=inputs, out=out)

    monkeypatch.setattr(_native, "execute_fused_expr", counting)
    return lambda: state["n"]


def _floats(n, seed=0x501):
    rng = np.random.default_rng(seed)
    return pl.DataFrame({"x": rng.standard_normal(n).astype(np.float32)})


def test_sort_uses_mlx(monkeypatch):
    df = _floats(10_000)
    n_dispatches = _count_fused_dispatches(monkeypatch)
    metal = df.lazy().sort("x").collect(engine=polars_metal.MetalEngine())
    assert n_dispatches() == 1, f"expected one fused sort dispatch, got {n_dispatches()}"
    assert metal["x"].to_list() == df.lazy().sort("x").collect()["x"].to_list()


def test_sort_ascending_matches_cpu():
    df = pl.DataFrame({"x": pl.Series([3.0, 1.0, 2.0, 2.0, 5.0, -1.0], dtype=pl.Float32)})
    cpu = df.lazy().sort("x").collect()
    metal = df.lazy().sort("x").collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)


def test_sort_descending_matches_cpu():
    df = _floats(5_000)
    cpu = df.lazy().sort("x", descending=True).collect()
    metal = df.lazy().sort("x", descending=True).collect(engine=polars_metal.MetalEngine())
    assert cpu["x"].to_list() == metal["x"].to_list()


def test_topk_uses_mlx_and_matches(monkeypatch):
    df = _floats(10_000)
    n_dispatches = _count_fused_dispatches(monkeypatch)
    metal = df.lazy().top_k(100, by="x").collect(engine=polars_metal.MetalEngine())
    assert n_dispatches() == 1, f"expected one fused sort dispatch, got {n_dispatches()}"
    cpu = df.lazy().top_k(100, by="x").collect()
    # top_k returns the 100 largest in descending order.
    assert cpu["x"].to_list() == metal["x"].to_list()


def test_bottom_k_matches_cpu():
    df = _floats(10_000)
    cpu = df.lazy().bottom_k(100, by="x").collect()
    metal = df.lazy().bottom_k(100, by="x").collect(engine=polars_metal.MetalEngine())
    assert cpu["x"].to_list() == metal["x"].to_list()


def test_topk_with_nulls_matches_cpu():
    """top_k drops nulls before ranking; the walker bypasses the dynamic
    null-drop filter and the dispatch re-drops nulls (still on MLX)."""
    df = pl.DataFrame({"x": pl.Series([3.0, None, 1.0, 5.0, None, 2.0, 4.0], dtype=pl.Float32)})
    cpu = df.lazy().top_k(3, by="x").collect()
    metal = df.lazy().top_k(3, by="x").collect(engine=polars_metal.MetalEngine())
    assert cpu["x"].to_list() == metal["x"].to_list()


def test_bottom_k_with_nulls_matches_cpu():
    df = pl.DataFrame({"x": pl.Series([3.0, None, 1.0, 5.0, None, 2.0, 4.0], dtype=pl.Float32)})
    cpu = df.lazy().bottom_k(3, by="x").collect()
    metal = df.lazy().bottom_k(3, by="x").collect(engine=polars_metal.MetalEngine())
    assert cpu["x"].to_list() == metal["x"].to_list()


def test_sort_nulls_falls_back_and_matches(monkeypatch):
    df = pl.DataFrame({"x": pl.Series([3.0, None, 1.0, None, 2.0], dtype=pl.Float32)})
    n_dispatches = _count_fused_dispatches(monkeypatch)
    cpu = df.lazy().sort("x").collect()
    metal = df.lazy().sort("x").collect(engine=polars_metal.MetalEngine())
    assert n_dispatches() == 0, f"null column must fall back to CPU, got {n_dispatches()}"
    assert_frame_equal(cpu, metal)


def test_multicolumn_sort_falls_back(monkeypatch):
    df = pl.DataFrame(
        {
            "x": pl.Series([3.0, 1.0, 2.0], dtype=pl.Float32),
            "y": pl.Series([10, 20, 30], dtype=pl.Int64),
        }
    )
    n_dispatches = _count_fused_dispatches(monkeypatch)
    cpu = df.lazy().sort("x").collect()
    metal = df.lazy().sort("x").collect(engine=polars_metal.MetalEngine())
    assert n_dispatches() == 0, f"multi-column sort must fall back, got {n_dispatches()}"
    assert_frame_equal(cpu, metal)


def test_non_f32_sort_falls_back(monkeypatch):
    df = pl.DataFrame({"x": pl.Series([3, 1, 2, 5, 4], dtype=pl.Int64)})
    n_dispatches = _count_fused_dispatches(monkeypatch)
    cpu = df.lazy().sort("x").collect()
    metal = df.lazy().sort("x").collect(engine=polars_metal.MetalEngine())
    assert n_dispatches() == 0, f"non-F32 sort must fall back, got {n_dispatches()}"
    assert_frame_equal(cpu, metal)
