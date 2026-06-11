"""M4 Phase 7: route reductions on compute intensity, not op identity.

A bare bandwidth-bound reduction (sum/min/max/mean over a plain column) is
~3x SLOWER on Metal than Polars CPU — the ~1 ms fused-dispatch floor dwarfs a
0.35 ms memory scan. Only compute-bound reductions (std/var) clear that floor
on their own (6-7x). So a *lone* bare sum/min/max/mean must stay on CPU; it may
ride to GPU only when the select also contains a GPU-worthy reduction (the
dispatch overhead is already paid). Compute-chain-terminated reductions land in
a follow-up increment.
"""

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

import polars_metal
from polars_metal import _native, _walker


def _dispatches(lf, eng):
    n = {"c": 0}
    orig = _native.execute_fused_expr

    def cnt(scope, inputs, out):
        n["c"] += 1
        return orig(scope=scope, inputs=inputs, out=out)

    _native.execute_fused_expr = cnt
    try:
        lf.collect(engine=eng)
    finally:
        _native.execute_fused_expr = orig
    return n["c"]


def _df():
    rng = np.random.default_rng(0x5151)
    return pl.DataFrame({"x": rng.standard_normal(50_000).astype(np.float32)})


def test_bare_sum_stays_on_cpu():
    eng = polars_metal.MetalEngine()
    lf = _df().lazy().select(pl.col("x").sum().alias("r"))
    assert _dispatches(lf, eng) == 0, "bare sum is bandwidth-bound; must stay on CPU"
    assert_frame_equal(lf.collect(engine=eng), lf.collect())


def test_bare_min_max_stay_on_cpu():
    eng = polars_metal.MetalEngine()
    for op in ("min", "max", "mean"):
        lf = _df().lazy().select(getattr(pl.col("x"), op)().alias("r"))
        assert _dispatches(lf, eng) == 0, f"bare {op} must stay on CPU"
        assert_frame_equal(lf.collect(engine=eng), lf.collect(), check_exact=False, abs_tol=1e-4)


def test_bare_std_var_use_gpu():
    eng = polars_metal.MetalEngine()
    for op in ("std", "var"):
        lf = _df().lazy().select(getattr(pl.col("x"), op)().alias("r"))
        assert _dispatches(lf, eng) == 1, f"bare {op} is compute-bound; should use GPU"
        assert_frame_equal(lf.collect(engine=eng), lf.collect(), check_exact=False, abs_tol=1e-3)


def test_bare_sum_rides_along_with_std():
    """A bare sum in the same select as a std still routes to GPU (the
    dispatch overhead is already paid by the std)."""
    eng = polars_metal.MetalEngine()
    lf = _df().lazy().select(pl.col("x").sum().alias("s"), pl.col("x").std().alias("d"))
    assert _dispatches(lf, eng) == 2, "sum+std select should route both to GPU"
    assert_frame_equal(lf.collect(engine=eng), lf.collect(), check_exact=False, abs_tol=1e-3)


def _int_df(dtype: pl.DataType) -> pl.DataFrame:
    rng = np.random.default_rng(0xB4)
    return pl.DataFrame({"x": pl.Series(rng.integers(-1_000_000, 1_000_000, 50_000), dtype=dtype)})


def test_bare_gpu_worthy_set_is_locked():
    """TRIPWIRE: bare bandwidth-bound reductions must stay on CPU.

    The B4 end-to-end spike (2026-06-10) measured the in-engine GPU bare-
    reduction path losing 2-5x at every size 1M->100M with no crossover — a
    bare reduction is bandwidth-bound and the host->MLX ingest alone exceeds
    Polars' multithreaded SIMD scan. Only the compute-bound std/var clear the
    dispatch floor (5-9x wins). If you are widening this set, you must first
    re-measure end-to-end and update this test deliberately. See the memory
    `m6-b4-reduction-routing-spike` and `tests/bench/m4_survey/bench_reductions.py`.
    """
    assert frozenset({"std", "var"}) == _walker._BARE_GPU_WORTHY_REDUCTIONS


def test_bare_int_reductions_stay_on_cpu():
    """Bare int sum/min/max are GPU-admissible (B2) but bandwidth-bound, so
    they must route to CPU exactly like F32 (the B4 spike confirmed the loss
    holds for Int32 and Int64)."""
    eng = polars_metal.MetalEngine()
    for dtype in (pl.Int32, pl.Int64):
        df = _int_df(dtype)
        for op in ("sum", "min", "max", "mean"):
            lf = df.lazy().select(getattr(pl.col("x"), op)().alias("r"))
            assert _dispatches(lf, eng) == 0, f"bare {op} {dtype} must stay on CPU"
            assert_frame_equal(lf.collect(engine=eng), lf.collect())
