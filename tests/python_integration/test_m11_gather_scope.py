"""Isolated: analyze_ir_with_columns_gather with MULTIPLE dim value columns.
The spliced scope (Take(price,key), Take(rating,key) feeding an F32 chain) must
match a numpy gather+chain."""

from __future__ import annotations
import numpy as np
import polars as pl
from polars_metal import _native
from polars_metal._fusion_analyzer import analyze_ir_with_columns_gather


def _capture(lf, gather_cols, key_col):
    out = {}

    def cb(nt, _d=None):
        def visit(nid):
            nt.set_node(nid)
            node = nt.view_current_node()
            if type(node).__name__ == "HStack":
                inputs = nt.get_inputs()
                parent = nt.get_node()
                nt.set_node(inputs[0])
                schema = dict(nt.get_schema())
                nt.set_node(parent)
                e = node.exprs[0]
                out["res"] = analyze_ir_with_columns_gather(
                    nt, e.node, schema, gather_cols, key_col
                )
                return True
            for i in nt.get_inputs():
                if visit(i):
                    return True
            return False

        visit(nt.get_node())

    lf.collect(engine="cpu", post_opt_callback=cb)
    return out["res"]


def test_multi_col_gather_scope_matches_numpy():
    rng = np.random.default_rng(7)
    n, dim_n = 4096, 256
    fact_id = rng.integers(0, dim_n, n).astype(np.int64)
    sc = rng.uniform(0, 1, n).astype(np.float32)
    price = rng.uniform(0.1, 2.0, dim_n).astype(np.float32)
    rating = rng.uniform(1, 5, dim_n).astype(np.float32)

    fact = pl.DataFrame({"id": fact_id, "sc": sc})
    dim = pl.DataFrame({"id": np.arange(dim_n, dtype=np.int64), "price": price, "rating": rating})
    lf = (
        fact.lazy()
        .join(dim.lazy(), on="id", how="left")
        .with_columns((pl.col("sc") * pl.col("price").exp() * pl.col("rating").log()).alias("out"))
    )

    res = _capture(lf, ["price", "rating"], "id")
    assert res is not None
    scope, descriptors, out_dtype = res
    assert out_dtype == "F32"
    kinds = [d[0] for d in descriptors]
    assert kinds.count("gather_key") == 1, descriptors
    gv = [p for k, p in descriptors if k == "gather_value"]
    assert set(gv) == {"price", "rating"}, descriptors

    arrays, tags = [], []
    short = {"price": price, "rating": rating}
    for kind, payload in descriptors:
        if kind == "gather_key":
            a = np.ascontiguousarray(fact_id, dtype=np.int32)
            t = 2
        elif kind == "gather_value":
            a = np.ascontiguousarray(short[payload], dtype=np.float32)
            t = 0
        elif kind == "col":
            assert payload == "sc"
            a = np.ascontiguousarray(sc, dtype=np.float32)
            t = 0
        elif kind == "lit":
            a = np.asarray([payload], dtype=np.float32)
            t = 0
        else:
            raise AssertionError(kind)
        arrays.append(a)
        tags.append(t)
    out = np.empty(n, dtype=np.float32)
    inputs = [(int(a.__array_interface__["data"][0]), int(a.size), t) for a, t in zip(arrays, tags)]
    assert (
        _native.execute_fused_expr(
            scope=scope,
            inputs=inputs,
            out=(int(out.__array_interface__["data"][0]), int(out.size), 0),
        )
        == n
    )
    expect = sc * np.exp(price[fact_id]) * np.log(rating[fact_id])
    np.testing.assert_allclose(out, expect, rtol=1e-3, atol=1e-3)


def test_single_col_still_works():
    rng = np.random.default_rng(8)
    n, dim_n = 1024, 64
    fact = pl.DataFrame(
        {
            "id": rng.integers(0, dim_n, n).astype(np.int64),
            "sc": rng.uniform(0, 1, n).astype(np.float32),
        }
    )
    vol = rng.uniform(0.1, 0.5, dim_n).astype(np.float32)
    dim = pl.DataFrame({"id": np.arange(dim_n, dtype=np.int64), "vol": vol})
    lf = (
        fact.lazy()
        .join(dim.lazy(), on="id", how="left")
        .with_columns((pl.col("sc") * pl.col("vol").log()).alias("out"))
    )
    res = _capture(lf, ["vol"], "id")
    assert res is not None
    _, descriptors, _ = res
    assert [k for k, _ in descriptors].count("gather_key") == 1
    assert [p for k, p in descriptors if k == "gather_value"] == ["vol"]
