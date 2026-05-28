"""TPC-H Q6 input fixture: filter-heavy single-group reduction. Q6 hits
a different shape than Q1 — no group-by keys, but a 4-predicate
conjunction over a sizeable table, finishing in a single `sum(expr)`.

Matches the canonical TPC-H lineitem dtypes for the columns Q6 reads:
  l_extendedprice  Float64
  l_discount       Float64
  l_quantity       Int32  (Decimal in spec; Int32 is the engine-relevant
                            substitute — Polars CPU treats both the same
                            for the `< 24` predicate)
  l_shipdate       Date
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

_SHIPDATE_LO = (date(1992, 1, 1) - date(1970, 1, 1)).days
_SHIPDATE_HI = (date(1998, 12, 31) - date(1970, 1, 1)).days


def make_q6_fixture(n_rows: int = 10_000_000, seed: int = 42) -> pl.DataFrame:
    """Build a `n_rows`-row lineitem-shaped DataFrame with only the four
    columns Q6 reads. Distributions match the canonical fixture so the
    selectivity of the Q6 date+discount+quantity predicate stays in the
    realistic 1–2% range.
    """
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "l_extendedprice": rng.uniform(900.0, 105_000.0, size=n_rows).astype(np.float64),
            "l_discount": rng.uniform(0.0, 0.11, size=n_rows).astype(np.float64),
            "l_quantity": rng.integers(1, 51, size=n_rows, dtype=np.int32),
            "l_shipdate": pl.Series(
                "l_shipdate",
                rng.integers(_SHIPDATE_LO, _SHIPDATE_HI + 1, size=n_rows, dtype=np.int64),
                dtype=pl.Date,
            ),
        }
    )
