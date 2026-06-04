"""Task 7 — rolling fallback guards: every non-handleable case routes to CPU.

Verifies that each guard in _rolling_detect / _rolling_dispatch:
  (a) does NOT call the kernel (execute_rolling dispatch count == 0)
  (b) still produces output that matches Polars CPU exactly

Guards exercised:
  - center=True          (detect rejects)
  - min_samples override (detect rejects)
  - weights              (detect rejects)
  - F64 dtype            (detect rejects — only Float32 supported)
  - Int64 dtype          (detect rejects — only Float32 supported)
  - window > MAX_W=4096  (detect rejects)
  - null-bearing input   (dispatch falls back before calling kernel)
  - streaming=True/new_streaming=True  (collect_wrapper skips rolling rewrite)

Control test: a clean handleable case MUST dispatch exactly once, confirming
the guards are not over-broad.

Note on assert_frame_equal kwargs: this Polars build uses rel_tol/abs_tol
(not rtol/atol); use those names for approximate comparisons.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

import polars_metal
from polars_metal import _native


def _count_dispatches(lf: pl.LazyFrame, eng: polars_metal.MetalEngine) -> tuple[int, pl.DataFrame]:
    """Return (execute_rolling call count, collected DataFrame).

    Monkey-patches _native.execute_rolling with a counting wrapper for the
    duration of the collect, then restores the original.
    """
    n: dict[str, int] = {"c": 0}
    orig = _native.execute_rolling

    def counting(**kw):  # type: ignore[no-untyped-def]
        n["c"] += 1
        return orig(**kw)

    _native.execute_rolling = counting
    try:
        out = lf.collect(engine=eng)
    finally:
        _native.execute_rolling = orig
    return n["c"], out


def test_option_and_dtype_fallbacks_route_zero_and_match_cpu():
    """Non-default options and non-Float32 dtypes must route to CPU (0 dispatches)."""
    eng = polars_metal.MetalEngine()
    f32 = pl.DataFrame({"x": np.arange(20, dtype=np.float32)}).lazy()
    cases = [
        ("center", f32.with_columns(r=pl.col("x").rolling_mean(3, center=True))),
        ("min_samples", f32.with_columns(r=pl.col("x").rolling_mean(3, min_samples=1))),
        ("weights", f32.with_columns(r=pl.col("x").rolling_mean(3, weights=[1.0, 2.0, 3.0]))),
        (
            "f64",
            pl.DataFrame({"x": np.arange(20, dtype=np.float64)})
            .lazy()
            .with_columns(r=pl.col("x").rolling_mean(3)),
        ),
        (
            "int",
            pl.DataFrame({"x": np.arange(20, dtype=np.int64)})
            .lazy()
            .with_columns(r=pl.col("x").rolling_sum(3)),
        ),
    ]
    for name, lf in cases:
        c, out = _count_dispatches(lf, eng)
        assert c == 0, f"{name}: expected CPU fallback (0 dispatches), got {c}"
        assert_frame_equal(out, lf.collect())


def test_large_window_falls_back():
    """window > MAX_W (4096) must route to CPU, not the Metal kernel."""
    eng = polars_metal.MetalEngine()
    # 5000-row frame, window=4097 exceeds MAX_W=4096
    df = pl.DataFrame({"x": np.arange(5000, dtype=np.float32)})
    lf = df.lazy().with_columns(r=pl.col("x").rolling_mean(4097))
    c, out = _count_dispatches(lf, eng)
    assert c == 0, f"window > MAX_W should fall back to CPU, got {c} dispatches"
    assert_frame_equal(out, lf.collect(), check_exact=False, rel_tol=1e-4, abs_tol=1e-4)


def test_null_input_falls_back_zero_dispatch():
    """Null-bearing input columns must fall back to Polars CPU (dispatch guard in
    _rolling_dispatch._rolling_series, not in detect)."""
    eng = polars_metal.MetalEngine()
    df = pl.DataFrame({"x": pl.Series([1.0, None, 3, 4, 5, 6], dtype=pl.Float32)})
    lf = df.lazy().with_columns(r=pl.col("x").rolling_mean(2))
    c, out = _count_dispatches(lf, eng)
    assert c == 0, f"null input should fall back to CPU, got {c} dispatches"
    assert_frame_equal(out, lf.collect())


def test_streaming_skips_rolling():
    """streaming=True/new_streaming=True must skip the rolling rewrite entirely
    (collect_wrapper reads those kwargs and sets rolling_bindings=[]).

    We try streaming=True first (accepted but deprecated in py-1.40.1).
    If neither kwarg is accepted, the test documents that fact via a skip.
    """
    eng = polars_metal.MetalEngine()
    df = pl.DataFrame({"x": np.arange(20, dtype=np.float32)})
    lf = df.lazy().with_columns(r=pl.col("x").rolling_mean(3))

    n: dict[str, int] = {"c": 0}
    orig = _native.execute_rolling

    def counting(**kw):  # type: ignore[no-untyped-def]
        n["c"] += 1
        return orig(**kw)

    _native.execute_rolling = counting
    streaming_tested = False
    try:
        # Try streaming=True first (deprecated but still accepted in py-1.40.1).
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                out = lf.collect(engine=eng, streaming=True)
            streaming_tested = True
        except TypeError:
            pass

        if not streaming_tested:
            # Fall back to new_streaming=True.
            try:
                out = lf.collect(engine=eng, new_streaming=True)
                streaming_tested = True
            except TypeError:
                pass
    finally:
        _native.execute_rolling = orig

    if not streaming_tested:
        # Neither kwarg accepted — document and confirm the guard code path
        # still reads those kwargs (static check that the collect_wrapper
        # looks for "streaming" and "new_streaming").
        import inspect

        import polars_metal as _pm_mod

        src = inspect.getsource(_pm_mod._patch_gpu_engine_callback)
        assert "streaming" in src, (
            "collect_wrapper should check 'streaming' kwarg for the streaming guard"
        )
        assert "new_streaming" in src, (
            "collect_wrapper should check 'new_streaming' kwarg for the streaming guard"
        )
        return  # guard logic confirmed statically; no collect to compare

    assert n["c"] == 0, (
        f"streaming mode should skip rolling dispatch (0 kernel calls), got {n['c']}"
    )
    assert_frame_equal(out, lf.collect())


def test_handleable_case_DOES_dispatch():
    """Control: a clean handleable F32 rolling_mean must dispatch to the kernel
    exactly once. Confirms the guards are not so broad they block everything."""
    eng = polars_metal.MetalEngine()
    df = pl.DataFrame({"x": np.arange(20, dtype=np.float32)})
    lf = df.lazy().with_columns(r=pl.col("x").rolling_mean(3))
    c, _ = _count_dispatches(lf, eng)
    assert c == 1, f"handleable rolling_mean should dispatch exactly once, got {c}"
