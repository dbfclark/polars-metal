import polars as pl
import pytest

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
