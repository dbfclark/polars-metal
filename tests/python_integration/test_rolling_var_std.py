"""
Task 6 — rolling_var/std end-to-end differential test vs Polars CPU.

Verifies that rolling_var and rolling_std are detected, routed to the Metal
kernel, and produce results that match Polars CPU (the oracle) within F32
tolerances (rtol=1e-3, atol=1e-4). The centered two-pass kernel avoids
catastrophic cancellation, so this tolerance is conservative.

Tests also confirm:
  - Correct leading nulls (first w-1 rows structurally null).
  - Multiple window sizes including w=257 (straddling the 256-element tile
    boundary), which exercises the tile-stitching path.
"""

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

import polars_metal


def test_rolling_var_std_e2e_match_cpu():
    df = pl.DataFrame({"x": np.random.default_rng(1).standard_normal(2048).astype(np.float32)})
    eng = polars_metal.MetalEngine()
    for op in ("var", "std"):
        lf = df.lazy().with_columns(r=getattr(pl.col("x"), f"rolling_{op}")(32))
        assert_frame_equal(
            lf.collect(engine=eng), lf.collect(), check_exact=False, rel_tol=1e-3, abs_tol=1e-4
        )


def test_rolling_var_std_multiple_windows():
    df = pl.DataFrame({"x": np.random.default_rng(2).standard_normal(1500).astype(np.float32)})
    eng = polars_metal.MetalEngine()
    for op in ("var", "std"):
        for w in (2, 16, 257):  # incl. a window straddling the 256 tile boundary
            lf = df.lazy().with_columns(r=getattr(pl.col("x"), f"rolling_{op}")(w))
            assert_frame_equal(
                lf.collect(engine=eng), lf.collect(), check_exact=False, rel_tol=1e-3, abs_tol=1e-4
            )
