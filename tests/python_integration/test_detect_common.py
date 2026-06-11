import polars as pl

from polars_metal import _detect_common as dc


def test_alias_name_extracts():
    node = {"Alias": [{"Literal": {"Scalar": {"Int64": 7}}}, "__pm_x__"]}
    assert dc._alias_name(node) == "__pm_x__"
    assert dc._alias_name({"Column": "a"}) is None


def test_literal_int_extracts():
    node = {"Alias": [{"Literal": {"Scalar": {"Int64": 42}}}, "tag"]}
    assert dc._literal_int(node) == 42


def test_install_patch_captures_exprs():
    cache = {}
    dc.install_with_columns_capture("_test_attr_c4", cache)
    dc.install_with_columns_capture("_test_attr_c4", cache)  # idempotent, no double-wrap
    lf = pl.DataFrame({"a": [1]}).lazy().with_columns((pl.col("a") + 1).alias("b"))
    assert id(lf) in cache
    assert isinstance(cache[id(lf)], list)
