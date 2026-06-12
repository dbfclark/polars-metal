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
    # cache values are now (weakref, exprs) tuples
    entry = cache[id(lf)]
    assert isinstance(entry, tuple) and len(entry) == 2
    _, exprs = entry
    assert isinstance(exprs, list)


def test_lookup_does_not_pop_and_evicts_on_gc():
    import gc

    cache = {}
    dc.install_with_columns_capture("_test_attr_m7a", cache)
    lf = pl.DataFrame({"a": [1]}).lazy().with_columns((pl.col("a") + 1).alias("b"))
    assert dc.lookup(cache, lf) is not None  # found
    assert dc.lookup(cache, lf) is not None  # STILL found (no pop)
    del lf
    gc.collect()
    assert len(cache) == 0  # weakref callback evicted on GC


def test_capture_cache_roundtrip():
    cache = dc.CaptureCache()
    h1 = cache.capture("spec-a")
    h2 = cache.capture("spec-b")
    assert h1 != h2
    assert cache.get(h1) == "spec-a"
    assert cache.get(h2) == "spec-b"
    cache.evict(h1)
    assert cache.get(h1) is None
    assert cache.get(h2) == "spec-b"
    cache.evict(99999)  # evicting an absent handle is a no-op


def test_sentinel_binding_fields():
    b = dc.SentinelBinding(out_name="o", col="c", payload=7)
    assert (b.out_name, b.col, b.payload) == ("o", "c", 7)


def test_make_sentinel_parser_prefix():
    tag = "__pm_test__"
    parse = dc.make_sentinel_parser(tag)
    fields = [
        {"Alias": [{"Column": "x"}, "__pm_in"]},
        {"Alias": [{"Literal": {"Scalar": {"Int64": 42}}}, f"{tag}myCol"]},
    ]
    node = {"Function": {"input": fields, "function": {"AsStruct": None}}}
    b = parse(node, "out")
    assert b == dc.SentinelBinding(out_name="out", col="myCol", payload=42)


def test_make_sentinel_parser_exact():
    tag = "__pm_corr__"
    parse = dc.make_sentinel_parser(tag, exact=True)
    fields = [{"Alias": [{"Literal": {"Scalar": {"Int64": 5}}}, tag]}]
    node = {"Function": {"input": fields, "function": {"AsStruct": None}}}
    b = parse(node, "out")
    assert b == dc.SentinelBinding(out_name="out", col="", payload=5)


def test_make_sentinel_parser_no_tag_returns_none():
    parse = dc.make_sentinel_parser("__pm_test__")
    assert parse({"Column": "x"}, "out") is None
