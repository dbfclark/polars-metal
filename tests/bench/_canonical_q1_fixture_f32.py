"""Canonical-shape TPC-H Q1 fixture with F32 numerics. Mirrors
``_canonical_q1_fixture.make_canonical_q1_fixture`` but every numeric
column is Float32 (or Int32 for `l_quantity`) — Utf8 keys and pl.Date
shipdate are preserved.

The F32 variant exists because the Metal toolchain on Apple Silicon has
``atomic_float`` for 32-bit floats but *not* for 64-bit. The fused
multi-aggregation kernel therefore covers F32-input aggs in a single
dispatch, while F64 inputs fall through to the M2 per-agg path (8 separate
dispatches × ~20–50 ms each — see ``docs/architecture.md`` § "F32 vs
F64 on this chip"). Comparing the F32 canonical Q1 vs CPU is the
architectural reading: "what could be on hardware with the right
primitives." The F64 number is a chip-limitation reading, not an engine
limitation.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

_RETURN_FLAGS = ["A", "N", "R"]
_LINE_STATUSES = ["F", "O"]

_SHIPDATE_LO = (date(1992, 1, 1) - date(1970, 1, 1)).days
_SHIPDATE_HI = (date(1998, 12, 31) - date(1970, 1, 1)).days


def make_canonical_q1_fixture_f32(n_rows: int = 10_000_000, seed: int = 42) -> pl.DataFrame:
    """Build a `n_rows`-row lineitem-shaped DataFrame matching TPC-H Q1's
    canonical schema, with Float32/Int32 numerics. Keys remain Utf8.
    """
    rng = np.random.default_rng(seed)
    rf_idx = rng.integers(0, len(_RETURN_FLAGS), size=n_rows)
    ls_idx = rng.integers(0, len(_LINE_STATUSES), size=n_rows)
    return pl.DataFrame(
        {
            "l_returnflag": [_RETURN_FLAGS[i] for i in rf_idx],
            "l_linestatus": [_LINE_STATUSES[i] for i in ls_idx],
            "l_quantity": rng.integers(1, 51, size=n_rows, dtype=np.int32),
            "l_extendedprice": rng.uniform(900.0, 105_000.0, size=n_rows).astype(np.float32),
            "l_discount": rng.uniform(0.0, 0.11, size=n_rows).astype(np.float32),
            "l_tax": rng.uniform(0.0, 0.09, size=n_rows).astype(np.float32),
            "l_shipdate": pl.Series(
                "l_shipdate",
                rng.integers(_SHIPDATE_LO, _SHIPDATE_HI + 1, size=n_rows, dtype=np.int64),
                dtype=pl.Date,
            ),
        }
    )
