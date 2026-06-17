"""Unit-level: _walk_join recognizes equi-join(int key, inner/left) -> fused F32
chain and returns a Handled plan dict; falls back otherwise."""

from __future__ import annotations

import numpy as np
import polars as pl

from polars_metal._walker import FallBack, Handled, walk


def _walk_plan(lf: pl.LazyFrame):
    out = {}

    def cb(nt, _d=None):
        out["res"] = walk(nt)

    lf.collect(engine="cpu", post_opt_callback=cb)
    return out["res"]


def _frames(key_dtype=np.int64, how="left"):
    rng = np.random.default_rng(1)
    fact = pl.DataFrame(
        {
            "id": rng.integers(0, 500, 2000).astype(key_dtype),
            "value": rng.uniform(50, 150, 2000).astype(np.float32),
        }
    )
    dim = pl.DataFrame(
        {
            "id": np.arange(500, dtype=key_dtype),
            "vol": rng.uniform(0.1, 0.5, 500).astype(np.float32),
        }
    )
    return (
        fact.lazy()
        .join(dim.lazy(), on="id", how=how)
        .with_columns((pl.col("vol").tanh() * pl.col("value")).alias("out"))
    )


def test_walk_join_handled_for_int_key_fused_chain():
    res = _walk_plan(_frames())
    assert isinstance(res, Handled), res
    plan = res.plan

    def find_join(p):
        if p.get("kind") == "Join":
            return p
        inner = p.get("input")
        return find_join(inner) if isinstance(inner, dict) else None

    jp = find_join(plan)
    assert jp is not None, plan
    assert jp["how"] in ("left", "inner"), jp
    assert jp["key"] == "id", jp
    assert jp["left"]["kind"] == "Scan" and jp["right"]["kind"] == "Scan", jp


def test_walk_join_fallback_on_f64_chain():
    rng = np.random.default_rng(2)
    fact = pl.DataFrame(
        {
            "id": rng.integers(0, 100, 500).astype(np.int64),
            "value": rng.uniform(1, 2, 500).astype(np.float64),
        }
    )
    dim = pl.DataFrame(
        {"id": np.arange(100, dtype=np.int64), "vol": rng.uniform(0.1, 0.5, 100).astype(np.float64)}
    )
    lf = (
        fact.lazy()
        .join(dim.lazy(), on="id", how="left")
        .with_columns((pl.col("vol") * pl.col("value")).alias("out"))
    )
    assert isinstance(_walk_plan(lf), FallBack)


def test_walk_join_fallback_on_string_key():
    fact = pl.DataFrame({"id": ["a", "b", "a"], "value": np.float32([1, 2, 3])})
    dim = pl.DataFrame({"id": ["a", "b"], "vol": np.float32([0.1, 0.2])})
    lf = (
        fact.lazy()
        .join(dim.lazy(), on="id", how="left")
        .with_columns((pl.col("vol") * pl.col("value")).alias("out"))
    )
    assert isinstance(_walk_plan(lf), FallBack)
