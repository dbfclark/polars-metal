"""F32 variant of the TPC-H Q6 fixture. Same shape and column count as
the F64 ``make_q6_fixture`` but ``l_extendedprice`` / ``l_discount`` are
Float32. ``l_quantity`` (Int32) and ``l_shipdate`` (Date) are unchanged
from the F64 version — already 32-bit / not affected by atomic primitive
availability.

See ``_canonical_q1_fixture_f32.py`` for the architectural rationale:
F32 numerics route through the fused multi-aggregation kernel, F64
doesn't (no 64-bit atomic_float on this chip).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

_SHIPDATE_LO = (date(1992, 1, 1) - date(1970, 1, 1)).days
_SHIPDATE_HI = (date(1998, 12, 31) - date(1970, 1, 1)).days


def make_q6_fixture_f32(n_rows: int = 10_000_000, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "l_extendedprice": rng.uniform(900.0, 105_000.0, size=n_rows).astype(np.float32),
            "l_discount": rng.uniform(0.0, 0.11, size=n_rows).astype(np.float32),
            "l_quantity": rng.integers(1, 51, size=n_rows, dtype=np.int32),
            "l_shipdate": pl.Series(
                "l_shipdate",
                rng.integers(_SHIPDATE_LO, _SHIPDATE_HI + 1, size=n_rows, dtype=np.int64),
                dtype=pl.Date,
            ),
        }
    )
