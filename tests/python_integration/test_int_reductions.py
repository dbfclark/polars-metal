import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_metal import MetalEngine
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
