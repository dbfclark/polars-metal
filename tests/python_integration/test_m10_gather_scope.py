"""Isolated validation of analyze_ir_with_columns_gather: the spliced scope
(Take(dim_value,key) feeding an F32 chain) must match numpy gather+chain."""
from __future__ import annotations

import numpy as np
import polars as pl

from polars_metal import _native
from polars_metal._fusion_analyzer import analyze_ir_with_columns_gather


def _capture_binding(lf):
    """Return the gather-analyzer result, built inside post_opt_callback where
    nt is valid. Walks the IR to find the HStack chain expr node + input schema."""
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
                    nt, e.node, schema, gather_col="vol", key_col="id"
                )
                return True
            return any(visit(i) for i in nt.get_inputs())

        visit(nt.get_node())

    lf.collect(engine="cpu", post_opt_callback=cb)
    return out["res"]


def test_gather_scope_matches_numpy():
    rng = np.random.default_rng(7)
    n, dim_n = 4096, 256
    fact_id = rng.integers(0, dim_n, n).astype(np.int64)
    fact_value = rng.uniform(50, 150, n).astype(np.float32)
    dim_vol_by_key = rng.uniform(0.1, 0.5, dim_n).astype(np.float32)  # position == key

    fact = pl.DataFrame({"id": fact_id, "value": fact_value})
    dim = pl.DataFrame({"id": np.arange(dim_n, dtype=np.int64), "vol": dim_vol_by_key})
    lf = (
        fact.lazy()
        .join(dim.lazy(), on="id", how="left")
        .with_columns(
            (
                pl.col("value")
                * 0.5
                * (1.0 + (0.7978845608 * pl.col("vol").log()).tanh())
            ).alias("out")
        )
    )

    res = _capture_binding(lf)
    assert res is not None
    scope, descriptors, out_dtype = res
    assert out_dtype == "F32"
    # descriptors must contain the two gather kinds + the fact 'value' col
    kinds = [d[0] for d in descriptors]
    assert "gather_key" in kinds and "gather_value" in kinds and "col" in kinds, descriptors

    # Build inputs in descriptor order.
    arrays = []
    tags = []
    for kind, payload in descriptors:
        if kind == "gather_key":
            a = np.ascontiguousarray(fact_id, dtype=np.int32)
            tag = 2
        elif kind == "gather_value":
            a = np.ascontiguousarray(dim_vol_by_key, dtype=np.float32)
            tag = 0  # SHORT (dim_n)
        elif kind == "col":
            assert payload == "value", payload
            a = np.ascontiguousarray(fact_value, dtype=np.float32)
            tag = 0
        elif kind == "lit":
            a = np.asarray([payload], dtype=np.float32)
            tag = 0
        else:
            raise AssertionError(kind)
        arrays.append(a)
        tags.append(tag)

    out = np.empty(n, dtype=np.float32)
    inputs = [
        (int(a.__array_interface__["data"][0]), int(a.size), t)
        for a, t in zip(arrays, tags, strict=True)
    ]
    written = _native.execute_fused_expr(
        scope=scope,
        inputs=inputs,
        out=(int(out.__array_interface__["data"][0]), int(out.size), 0),
    )
    assert written == n

    # numpy oracle: vol = dim_vol_by_key[fact_id]; chain
    vol = dim_vol_by_key[fact_id]
    expect = fact_value * 0.5 * (1.0 + np.tanh(0.7978845608 * np.log(vol)))
    np.testing.assert_allclose(out, expect, rtol=1e-3, atol=1e-3)
