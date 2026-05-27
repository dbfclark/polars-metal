"""Canonical TPC-H Q1 input fixture: lineitem with Utf8 l_returnflag and
l_linestatus, no encoding shortcuts. M3 Phase 7 (KeyDtype::Utf8) unlocks
this fixture as the true canonical Q1 workload — prior M2/M3 Q1 fixtures
used Bool keys to fit the 128-bit encoder budget when Utf8 wasn't an
input type the kernel layer accepted.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

# Three returnflag values x two linestatus values = 6 possible
# (returnflag, linestatus) combinations, of which TPC-H reports four
# rows in the canonical Q1 result (R/O is empty after the date filter).
_RETURN_FLAGS = ["A", "N", "R"]
_LINE_STATUSES = ["F", "O"]

_SHIPDATE_LO = (date(1992, 1, 1) - date(1970, 1, 1)).days
_SHIPDATE_HI = (date(1998, 12, 31) - date(1970, 1, 1)).days


def make_canonical_q1_fixture(n_rows: int = 10_000_000, seed: int = 42) -> pl.DataFrame:
    """Build a `n_rows`-row lineitem-shaped DataFrame matching TPC-H Q1's
    canonical schema. Keys are Utf8; numeric columns are Float64.
    """
    rng = np.random.default_rng(seed)
    rf_idx = rng.integers(0, len(_RETURN_FLAGS), size=n_rows)
    ls_idx = rng.integers(0, len(_LINE_STATUSES), size=n_rows)
    return pl.DataFrame(
        {
            "l_returnflag": [_RETURN_FLAGS[i] for i in rf_idx],
            "l_linestatus": [_LINE_STATUSES[i] for i in ls_idx],
            "l_quantity": rng.uniform(1.0, 51.0, size=n_rows).astype(np.float64),
            "l_extendedprice": rng.uniform(900.0, 105_000.0, size=n_rows).astype(np.float64),
            "l_discount": rng.uniform(0.0, 0.11, size=n_rows).astype(np.float64),
            "l_tax": rng.uniform(0.0, 0.09, size=n_rows).astype(np.float64),
            "l_shipdate": pl.Series(
                "l_shipdate",
                rng.integers(_SHIPDATE_LO, _SHIPDATE_HI + 1, size=n_rows, dtype=np.int64),
                dtype=pl.Date,
            ),
        }
    )
