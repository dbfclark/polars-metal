import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_metal import MetalEngine, _native
from polars_metal._fusion_analyzer import _int_reduction_out_dtype

# (agg_kind, polars dtype) -> expected wire output-dtype string, or None for
# "not GPU-admissible → CPU fallback". This table IS the B2 scope decision.
_ADMIT = [
    # sum: admitted only for the four widths where MLX-native == Polars.
    ("sum", pl.Int32, "I32"),
    ("sum", pl.Int64, "I64"),
    ("sum", pl.UInt32, "U32"),
    ("sum", pl.UInt64, "U64"),
    # sum of narrow ints → Polars upcasts to Int64/UInt64, MLX widens to
    # int32/uint32 → MISMATCH → CPU.
    ("sum", pl.Int8, None),
    ("sum", pl.Int16, None),
    ("sum", pl.UInt8, None),
    ("sum", pl.UInt16, None),
    # min / max: preserve input width for all 8 → admitted everywhere.
    ("min", pl.Int8, "I8"),
    ("min", pl.Int16, "I16"),
    ("min", pl.Int32, "I32"),
    ("min", pl.Int64, "I64"),
    ("min", pl.UInt8, "U8"),
    ("min", pl.UInt16, "U16"),
    ("min", pl.UInt32, "U32"),
    ("min", pl.UInt64, "U64"),
    ("max", pl.Int8, "I8"),
    ("max", pl.Int16, "I16"),
    ("max", pl.Int32, "I32"),
    ("max", pl.Int64, "I64"),
    ("max", pl.UInt8, "U8"),
    ("max", pl.UInt16, "U16"),
    ("max", pl.UInt32, "U32"),
    ("max", pl.UInt64, "U64"),
    # mean/std/var of int → MLX→f32, Polars→Float64 → MISMATCH → CPU.
    ("mean", pl.Int32, None),
    ("mean", pl.Int64, None),
    ("std", pl.Int64, None),
    ("var", pl.Int64, None),
]


@pytest.mark.parametrize("kind,dtype,expected", _ADMIT)
def test_int_reduction_out_dtype_matrix(kind, dtype, expected):
    assert _int_reduction_out_dtype(kind, dtype) == expected


def test_int64_sum_returns_int64_dtype_via_engine():
    # An int sum that is admitted (Int64) must come back as Int64, byte-exact.
    df = pl.DataFrame({"x": pl.Series([1, 2, 3, 4], dtype=pl.Int64)})
    lf = df.lazy().select(pl.col("x").sum().alias("s"))
    got = lf.collect(engine=MetalEngine())
    want = lf.collect()
    assert got.equals(want)
    assert got["s"].dtype == pl.Int64


def test_f32_sum_still_works_after_arity_change():
    # The analyzer-tuple arity grew from 5 to 6; the F32 path must be intact.
    df = pl.DataFrame({"x": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float32)})
    lf = df.lazy().select((pl.col("x") * 2.0).sum().alias("s"))  # chain → GPU
    got = lf.collect(engine=MetalEngine())
    want = lf.collect()
    assert_frame_equal(got, want, check_exact=False, abs_tol=1e-4)


def _reduction_dispatches(lf, eng) -> int:
    """Count GPU fused-reduction dispatches via the execute_fused_expr hook."""
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


def test_narrow_int_sum_never_routes_to_gpu():
    # Int8 sum → Polars upcasts to Int64; MLX→int32. Not admitted; must stay
    # CPU (0 GPU dispatches) AND match Polars exactly.
    eng = MetalEngine()
    df = pl.DataFrame({"x": pl.Series([1, 2, 3, 100], dtype=pl.Int8)})
    lf = df.lazy().select(pl.col("x").sum().alias("s"))
    assert _reduction_dispatches(lf, eng) == 0
    got, want = lf.collect(engine=eng), lf.collect()
    assert got.equals(want)
    assert got["s"].dtype == pl.Int64  # Polars upcast preserved on CPU


def test_int64_chain_sum_routes_to_gpu_and_matches():
    # A compute chain ending in sum is always GPU-worthy (is_chain=True), so
    # this genuinely exercises the GPU int-reduction path.
    eng = MetalEngine()
    df = pl.DataFrame({"x": pl.Series([1, 2, 3, 4, 5], dtype=pl.Int64)})
    lf = df.lazy().select(((pl.col("x") * 2) + 1).sum().alias("s"))
    assert _reduction_dispatches(lf, eng) == 1, "int chain sum should use GPU"
    got, want = lf.collect(engine=eng), lf.collect()
    assert got.equals(want)
    assert got["s"].dtype == pl.Int64


_SUM_DTYPES = [pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]
_MINMAX_DTYPES = [
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
]


def _vals_for(dtype) -> list[int]:
    # Small, in-range, mixed-sign where signed. Chosen so +/* in a chain stay
    # in range for the narrowest min/max types (Int8/UInt8).
    if dtype in (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
        return [0, 1, 2, 7, 9, 3]
    return [-3, 0, 1, 2, 7, -2]


@pytest.mark.parametrize("dtype", _SUM_DTYPES)
def test_int_sum_byte_exact_no_nulls(dtype):
    eng = MetalEngine()
    df = pl.DataFrame({"x": pl.Series(_vals_for(dtype), dtype=dtype)})
    lf = df.lazy().select(pl.col("x").sum().alias("r"))
    got, want = lf.collect(engine=eng), lf.collect()
    assert got.equals(want), f"{dtype}: {got} != {want}"
    assert got["r"].dtype == dtype


@pytest.mark.parametrize("dtype", _MINMAX_DTYPES)
@pytest.mark.parametrize("op", ["min", "max"])
def test_int_minmax_byte_exact_no_nulls(dtype, op):
    eng = MetalEngine()
    df = pl.DataFrame({"x": pl.Series(_vals_for(dtype), dtype=dtype)})
    lf = df.lazy().select(getattr(pl.col("x"), op)().alias("r"))
    got, want = lf.collect(engine=eng), lf.collect()
    assert got.equals(want), f"{op} {dtype}: {got} != {want}"
    assert got["r"].dtype == dtype


# M6: `_walker._map_dtype` now maps all 8 integer widths (UInt64 → "U64"), and
# the Rust `MetalDtype` enum + `from_wire` gained the matching "U64" arm, so a
# UInt64 input column is recognized at the DataFrameScan gate and reaches the
# fused GPU reduction path (the reduction analyzer already admits UInt64
# sum/min/max). All of _SUM_DTYPES now genuinely exercise the GPU int path.
@pytest.mark.parametrize("dtype", _SUM_DTYPES)
def test_int_chain_sum_gpu_path(dtype):
    # Chain → is_chain=True → routes to GPU. Proves the GPU int-reduction path
    # (dispatch count == 1) AND byte-exact dtype/value.
    eng = MetalEngine()
    df = pl.DataFrame({"x": pl.Series(_vals_for(dtype), dtype=dtype)})
    lf = df.lazy().select(((pl.col("x") + 1) * 2).sum().alias("r"))
    assert _reduction_dispatches(lf, eng) == 1, f"chain sum {dtype} should use GPU"
    got, want = lf.collect(engine=eng), lf.collect()
    assert got.equals(want), f"chain {dtype}: {got} != {want}"


@pytest.mark.parametrize("op", ["min", "max"])
def test_int_chain_minmax_gpu_path(op):
    eng = MetalEngine()
    df = pl.DataFrame({"x": pl.Series([-3, 0, 1, 2, 7], dtype=pl.Int64)})
    lf = df.lazy().select(getattr((pl.col("x") + 1), op)().alias("r"))
    assert _reduction_dispatches(lf, eng) == 1, f"chain {op} should use GPU"
    got, want = lf.collect(engine=eng), lf.collect()
    assert got.equals(want)


@pytest.mark.parametrize("dtype", _SUM_DTYPES)
def test_int_chain_sum_with_nulls_drop_nulls_path(dtype):
    # Null-bearing chain → walker stamps _drop_nulls → CPU drop + GPU reduce of
    # survivors. Byte-exact vs Polars (which skips nulls).
    eng = MetalEngine()
    vals = _vals_for(dtype)
    vals_n = [*vals[:2], None, *vals[2:]]
    df = pl.DataFrame({"x": pl.Series(vals_n, dtype=dtype)})
    lf = df.lazy().select(((pl.col("x") + 1) * 2).sum().alias("r"))
    got, want = lf.collect(engine=eng), lf.collect()
    assert got.equals(want), f"null chain {dtype}: {got} != {want}"


@pytest.mark.parametrize("dtype", _SUM_DTYPES)
def test_int_sum_empty(dtype):
    eng = MetalEngine()
    df = pl.DataFrame({"x": pl.Series([], dtype=dtype)})
    lf = df.lazy().select(pl.col("x").sum().alias("r"))
    got, want = lf.collect(engine=eng), lf.collect()
    assert got.equals(want), f"empty {dtype}: {got} != {want}"


@pytest.mark.parametrize("dtype", _SUM_DTYPES)
def test_int_sum_single_element(dtype):
    eng = MetalEngine()
    df = pl.DataFrame({"x": pl.Series([5], dtype=dtype)})
    lf = df.lazy().select(pl.col("x").sum().alias("r"))
    got, want = lf.collect(engine=eng), lf.collect()
    assert got.equals(want), f"single {dtype}: {got} != {want}"


@pytest.mark.parametrize("op", ["min", "max"])
def test_int_minmax_empty_and_single(op):
    eng = MetalEngine()
    for vals in ([], [5]):
        df = pl.DataFrame({"x": pl.Series(vals, dtype=pl.Int64)})
        lf = df.lazy().select(getattr(pl.col("x"), op)().alias("r"))
        got, want = lf.collect(engine=eng), lf.collect()
        assert got.equals(want), f"{op} {vals}: {got} != {want}"


@pytest.mark.parametrize("dtype", [pl.Int8, pl.Int16, pl.UInt8, pl.UInt16])
def test_narrow_int_sum_falls_back_to_cpu_and_matches(dtype):
    eng = MetalEngine()
    df = pl.DataFrame({"x": pl.Series([1, 2, 3, 4], dtype=dtype)})
    # Bare AND chain forms: narrow sum is never admitted (analyzer aborts).
    for lf in (
        df.lazy().select(pl.col("x").sum().alias("r")),
        df.lazy().select(((pl.col("x") + 1)).sum().alias("r")),
    ):
        assert _reduction_dispatches(lf, eng) == 0, f"narrow sum {dtype} must stay CPU"
        got, want = lf.collect(engine=eng), lf.collect()
        assert got.equals(want), f"narrow sum {dtype}: {got} != {want}"


@pytest.mark.parametrize("dtype", [pl.Int32, pl.Int64, pl.UInt32, pl.UInt64])
def test_int_mean_falls_back_to_cpu_and_matches(dtype):
    eng = MetalEngine()
    df = pl.DataFrame({"x": pl.Series([1, 2, 3, 4], dtype=dtype)})
    for lf in (
        df.lazy().select(pl.col("x").mean().alias("r")),
        df.lazy().select(((pl.col("x") + 1)).mean().alias("r")),
    ):
        assert _reduction_dispatches(lf, eng) == 0, f"int mean {dtype} must stay CPU"
        got, want = lf.collect(engine=eng), lf.collect()
        assert got.equals(want), f"int mean {dtype}: {got} != {want}"


def test_wire_str_to_polars_matches_walker_table():
    # Drift guard: _udf._WIRE_STR_TO_POLARS must agree with
    # _walker._INT_TAG_TO_POLARS (both map wire str -> the same Polars dtype).
    from polars_metal._udf import _WIRE_STR_TO_POLARS
    from polars_metal._walker import _INT_TAG_TO_POLARS

    for wire, pl_str in _INT_TAG_TO_POLARS.items():
        assert wire in _WIRE_STR_TO_POLARS, f"{wire} missing from _WIRE_STR_TO_POLARS"
        assert str(_WIRE_STR_TO_POLARS[wire]) == pl_str, (
            f"{wire}: {_WIRE_STR_TO_POLARS[wire]} != {pl_str}"
        )
