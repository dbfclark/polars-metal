"""Datetime decomposition: year/month/day/hour extraction over a column.

A common ETL pattern: parse a datetime, extract components, use them as
groupby keys or features. Polars stores datetimes as i64 (ns since epoch);
extraction requires integer division + modulo chains, no transcendentals.

Hypothesis: this is bandwidth-shaped (~24 bytes touched per row).
Likely no win, but worth measuring to nail down which "obvious"
ops actually lose.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import polars as pl
from datetime import datetime

from tests.bench.m4_survey._timing import time_callable


def main() -> None:
    N = 10_000_000
    rng = np.random.default_rng(0xCAFE)

    # ns since epoch, random across ~10 years
    ts_ns = rng.integers(0, 10 * 365 * 24 * 3600 * 1_000_000_000, size=N, dtype=np.int64)
    df = pl.DataFrame({"ts": pl.Series(ts_ns, dtype=pl.Datetime("ns"))})

    print(f"\n=== datetime decomposition ===  N={N:,}")
    print()

    time_callable(
        "polars.dt.year",
        lambda: df.select(pl.col("ts").dt.year()),
    )
    time_callable(
        "polars.dt.month",
        lambda: df.select(pl.col("ts").dt.month()),
    )
    time_callable(
        "polars.dt.weekday",
        lambda: df.select(pl.col("ts").dt.weekday()),
    )
    time_callable(
        "polars.dt.year+month+day+hour (chained)",
        lambda: df.with_columns(
            year=pl.col("ts").dt.year(),
            month=pl.col("ts").dt.month(),
            day=pl.col("ts").dt.day(),
            hour=pl.col("ts").dt.hour(),
        ),
    )

    # MLX equivalent: hour-of-day extraction is just integer modulo
    # (no calendar math). Year/month require calendar math, which MLX
    # doesn't have natively — would need a custom kernel.
    ts_mx = mx.array(ts_ns)
    mx.eval(ts_mx)

    def mlx_hour():
        # hour = (ts_ns / (3600 * 1e9)) % 24
        h = (ts_mx // (3600 * 1_000_000_000)) % 24
        mx.eval(h)
        return h

    time_callable("mlx.hour_via_modulo", mlx_hour)
    print("  (year/month/weekday require calendar math; MLX has no native gregorian; would need custom MSL)")


if __name__ == "__main__":
    main()
