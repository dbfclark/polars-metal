# tests/bench/_lineitem_fixture.py
"""Deterministic lineitem-shaped fixture for modified TPC-H Q1.

Matches spec § "Workload validated" with one deviation for the encoder's
128-bit composite-key budget:
  - l_returnflag (Boolean) — values {True, False} mapping to TPC-H's A/N
  - l_linestatus (Boolean) — values {True, False} mapping to TPC-H's F/O

Original TPC-H Q1 uses Utf8 keys; the M2 spec proposed i8 or i64 integer
encoding. With our M2 encoder, 2 x i64 = 130 bits exceeds the 128-bit
budget. We use Boolean (2 keys x 2 bits = 4 bits) which is the natural
fit for the binary cardinality and stays well under budget.

Other columns:
  - l_quantity   (i64 ∈ [1, 50])
  - l_extendedprice (f64 ∈ [1000, 100000])
  - l_discount   (f64 ∈ [0.00, 0.10])
  - l_tax        (f64 ∈ [0.00, 0.08])
  - l_shipdate   (i64, days since 1970-01-01, range 1992-01-01..1998-12-31)
  - disc_price   (f64 = l_extendedprice * (1 - l_discount))  pre-projected
  - charge       (f64 = disc_price * (1 + l_tax))            pre-projected

Pre-projected disc_price/charge: the multi-aggregation expression
unfolding (sum(extendedprice * (1 - discount))) is deferred to M3.

Reproducibility: numpy.random.default_rng(seed) ensures bit-identical
output across runs of the same seed.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

_SHIPDATE_LO = (date(1992, 1, 1) - date(1970, 1, 1)).days
_SHIPDATE_HI = (date(1998, 12, 31) - date(1970, 1, 1)).days


def make_lineitem(n_rows: int = 10_000_000, seed: int = 0xC0FFEE) -> pl.DataFrame:
    """Build an n_rows x 9-column lineitem-shaped DataFrame.

    Returns columns matching modified Q1's input shape. Bit-reproducible
    across runs at the same seed.
    """
    rng = np.random.default_rng(seed)

    returnflag = rng.integers(0, 2, size=n_rows, dtype=np.uint8).astype(bool)
    linestatus = rng.integers(0, 2, size=n_rows, dtype=np.uint8).astype(bool)
    quantity = rng.integers(1, 51, size=n_rows, dtype=np.int64)
    extendedprice = rng.uniform(1000.0, 100_000.0, size=n_rows).astype(np.float64)
    discount = rng.uniform(0.0, 0.10, size=n_rows).astype(np.float64)
    tax = rng.uniform(0.0, 0.08, size=n_rows).astype(np.float64)
    shipdate = rng.integers(_SHIPDATE_LO, _SHIPDATE_HI + 1, size=n_rows, dtype=np.int64)

    disc_price = extendedprice * (1.0 - discount)
    charge = disc_price * (1.0 + tax)

    return pl.DataFrame(
        {
            "l_returnflag": returnflag,
            "l_linestatus": linestatus,
            "l_quantity": quantity,
            "l_extendedprice": extendedprice,
            "l_discount": discount,
            "l_tax": tax,
            "l_shipdate": shipdate,
            "disc_price": disc_price,
            "charge": charge,
        }
    )


if __name__ == "__main__":
    df = make_lineitem(1_000_000)
    print(df.schema)
    print(df.head())
    n_unique = df.select(["l_returnflag", "l_linestatus"]).unique().height
    print(f"n_rows={df.height}, n_unique_keys={n_unique}")
