"""M5 rolling Task 5: analyzer recognizes ``shift`` (offset param) and the
structural head-null validity that makes a raw ``pl.col(x).shift(w)`` match
Polars CPU byte-for-byte under ``engine="metal"``.

Empirically pinned IR shapes (py-1.40.1, see _fusion_analyzer.py comments):
  - ``shift``: Function ``function_data=('shift',)`` with
    ``input=[value_child, Literal(offset: Int32)]``. The offset is a child
    Literal, NOT a kwarg. ``shift_and_fill`` is a DISTINCT function name
    (3-input) so plain ``shift`` never carries a fill_value.
  - ``int_range(len())``: OPAQUE — ``nt.view_expression`` raises
    ``NotImplementedError('range')``, so the walker cannot recognize it. The
    RowIndex code paths exist defensively but are unreachable via the walker
    in this Polars version (NodeTraverser-opacity wall). No test exercises it.
"""

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

import polars_metal
from polars_metal import _native


def _count_dispatches(monkeypatch):
    count = {"n": 0}
    orig = _native.execute_fused_expr

    def counting(scope, inputs, out):
        count["n"] += 1
        return orig(scope=scope, inputs=inputs, out=out)

    monkeypatch.setattr(_native, "execute_fused_expr", counting)
    return count


def test_shift_in_fused_expr_dispatches(monkeypatch):
    # cumsum-diff shape: the value graph (cum_sum, sub, shift) and the
    # structural-null validity graph (Shift of the all-valid mask) must both
    # be constructible, so the binding routes to MLX.
    df = pl.DataFrame({"x": np.arange(16, dtype=np.float32)})
    eng = polars_metal.MetalEngine()
    count = _count_dispatches(monkeypatch)
    expr = pl.col("x").cum_sum() - pl.col("x").cum_sum().shift(3)
    lf = df.lazy().with_columns(r=expr)
    metal = lf.collect(engine=eng)
    assert count["n"] >= 1, f"shift chain should route to MLX, got {count['n']} dispatches"
    # value+null correctness vs CPU (the rolling e2e is a later task, but the
    # raw chain must already match).
    assert_frame_equal(metal, lf.collect(), check_exact=False, abs_tol=1e-2, rel_tol=1e-4)


def test_raw_shift_matches_cpu_with_leading_nulls(monkeypatch):
    # The critical correctness test: a bare forward shift must reproduce
    # Polars' structural leading nulls (first w positions null) exactly.
    eng = polars_metal.MetalEngine()
    count = _count_dispatches(monkeypatch)
    df = pl.DataFrame({"x": np.arange(1, 9, dtype=np.float32)})
    lf = df.lazy().with_columns(s=pl.col("x").shift(2))
    metal = lf.collect(engine=eng)
    assert count["n"] >= 1, f"raw shift should route to MLX, got {count['n']} dispatches"
    assert_frame_equal(metal, lf.collect())
    assert metal["s"][:2].null_count() == 2  # first 2 structurally null


def test_negative_shift_falls_back_to_cpu(monkeypatch):
    # MLX Shift is forward-only; a negative offset must fall back to CPU.
    eng = polars_metal.MetalEngine()
    count = _count_dispatches(monkeypatch)
    df = pl.DataFrame({"x": np.arange(1, 9, dtype=np.float32)})
    lf = df.lazy().with_columns(s=pl.col("x").shift(-2))
    metal = lf.collect(engine=eng)
    assert count["n"] == 0, "negative shift has no forward MLX binding; must fall back"
    assert_frame_equal(metal, lf.collect())
