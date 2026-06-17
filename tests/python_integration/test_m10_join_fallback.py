import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from polars_metal import MetalEngine


def _check_cpu_parity(lf):
    assert_frame_equal(
        lf.collect(),
        lf.collect(engine=MetalEngine()),
        check_dtypes=True,
        rel_tol=1e-3,
        abs_tol=1e-3,
    )


def test_f64_chain_falls_back():
    f = pl.DataFrame({"id": np.arange(100, dtype=np.int64), "v": np.arange(100, dtype=np.float64)})
    d = pl.DataFrame(
        {
            "id": np.arange(100, dtype=np.int64),
            "vol": np.arange(100, dtype=np.float64),
        }
    )
    _check_cpu_parity(
        f.lazy().join(d.lazy(), on="id").with_columns((pl.col("vol") * pl.col("v")).alias("o"))
    )


def test_string_key_falls_back():
    f = pl.DataFrame({"id": ["a", "b"], "v": np.float32([1, 2])})
    d = pl.DataFrame({"id": ["a", "b"], "vol": np.float32([3, 4])})
    _check_cpu_parity(
        f.lazy().join(d.lazy(), on="id").with_columns((pl.col("vol") * pl.col("v")).alias("o"))
    )


def test_outer_join_falls_back():
    f = pl.DataFrame({"id": np.int64([0, 1, 2]), "v": np.float32([1, 2, 3])})
    d = pl.DataFrame({"id": np.int64([0, 1]), "vol": np.float32([3, 4])})
    _check_cpu_parity(
        f.lazy()
        .join(d.lazy(), on="id", how="full")
        .with_columns((pl.col("vol") * pl.col("v")).alias("o"))
    )


def test_bare_join_no_chain_falls_back():
    # A join with NO fused F32 chain consuming it: `_walk_join` recognizes the
    # join but no `_parent_chain` is attached, so the engine must run it on CPU
    # (regression guard — this KeyError'd the whole Polars join conformance suite).
    f = pl.DataFrame({"id": np.int64([0, 1, 2]), "v": np.float32([1, 2, 3])})
    d = pl.DataFrame({"id": np.int64([0, 1, 2]), "vol": np.float32([3, 4, 5])})
    _check_cpu_parity(f.lazy().join(d.lazy(), on="id"))


def test_join_then_groupby_falls_back():
    # Join feeding a group_by (not an F32 HStack chain) — no `_parent_chain`.
    f = pl.DataFrame({"id": np.int64([0, 1, 0, 1]), "v": np.float32([1, 2, 3, 4])})
    d = pl.DataFrame({"id": np.int64([0, 1]), "vol": np.float32([10, 20])})
    _check_cpu_parity(
        f.lazy().join(d.lazy(), on="id").group_by("id").agg(pl.col("v").sum()).sort("id")
    )


def test_join_then_non_f32_chain_falls_back():
    # Join feeding an integer-output expression (not F32) — chain not fused.
    f = pl.DataFrame({"id": np.int64([0, 1, 2]), "v": np.int64([1, 2, 3])})
    d = pl.DataFrame({"id": np.int64([0, 1, 2]), "w": np.int64([10, 20, 30])})
    _check_cpu_parity(
        f.lazy().join(d.lazy(), on="id").with_columns((pl.col("v") + pl.col("w")).alias("o"))
    )
