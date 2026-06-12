"""M6 A3: FFT `.metal` sentinel builder (op encoded in an Int64 literal, like vector search).

`build_fft_sentinel(input_expr, input_col, op)` returns an as_struct expression that:
  - is serialize-detectable (carries the input column name in a tagged alias + op as Int64),
  - raises on plain-CPU collect (an opaque map_batches marker),
  - is recognized + dispatched to the GPU by collect(engine="metal").
There is no external capture (the input column IS the data), so unlike vector search this module
holds no module-global cache.
"""

from __future__ import annotations

import polars as pl

from polars_metal._detect_common import sentinel_fields

# op codes carried in the sentinel's Int64 literal (must match _fft_dispatch's inverse check).
OP_FFT = 0
OP_IFFT = 1

# Magic prefix on the Int64-literal field alias so the detector finds our bindings unambiguously.
# Distinct from the vector-search tag so the two detectors never cross-match.
FFT_SENTINEL_TAG = "__pm_fft__"


def _raise_cpu(_s: pl.Series) -> pl.Series:
    raise pl.exceptions.ComputeError(
        "polars_metal: .metal.fft/.ifft require collect(engine='metal'); "
        "they have no CPU implementation."
    )


def build_fft_sentinel(input_expr: pl.Expr, input_col: str, op: int) -> pl.Expr:
    """Build the recognizable, CPU-raising sentinel struct expression.

    Serialized shape: an as_struct with three fields:
      - field 0: the input column (so the detector can read the input column name),
      - field 1: an Int64 op literal tagged with FFT_SENTINEL_TAG{input_col} via its alias,
      - field 2: an opaque map_batches(_raise) over the input column → raises on plain CPU.
    Under engine="metal", dispatch DROPS this output column before the CPU collect, so the
    map_batches never executes; on plain CPU it executes and raises.
    """
    return pl.struct(
        sentinel_fields(
            input_expr,
            tag=FFT_SENTINEL_TAG,
            payload=op,
            col=input_col,
            in_alias="__pm_fft_in",
            raise_alias="__pm_fft_raise",
            raise_fn=_raise_cpu,
        )
    )
