"""NYC Taxi-style ETL: haversine distance + datetime + groupby.

Real-world ETL shape: drop-off / pick-up coordinates → haversine distance,
datetime parsing, then aggregations. Each row of "real compute" (haversine)
costs ~10 transcendentals = ~50 amortized ops. Memory is 6 F32 + 2 datetime
+ groupby keys per row.

Compute density for haversine over N rows:
  ops    = ~50 * N (sin/cos/atan2/sqrt, each ~10 cycles on CPU)
  bytes  = ~32 * N (6 F32 lat/lon, write 1 F32 distance)
  density ~= 1.5 ops/byte effective, but each op is much more expensive
  than an FMA, so wall-clock compute time per row is high relative to
  memory bandwidth.

This is the workload class where compute density appears low but per-op
latency is high. GPU plausibly wins because:
  - Transcendentals are pipelined per-SIMD-lane on M2 Ultra.
  - CPU sin/cos are 20-40 cycles each, killing apparent throughput.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from tests.bench.m4_survey._timing import time_callable


def make_taxi(n: int, *, seed: int = 0xCAB) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    # NYC coordinates roughly
    pickup_lat = rng.uniform(40.6, 40.9, size=n).astype(np.float32)
    pickup_lon = rng.uniform(-74.05, -73.7, size=n).astype(np.float32)
    drop_lat = rng.uniform(40.6, 40.9, size=n).astype(np.float32)
    drop_lon = rng.uniform(-74.05, -73.7, size=n).astype(np.float32)
    pickup_ts = rng.integers(0, 31_536_000, size=n).astype(np.int64)  # seconds in a year
    fare = rng.uniform(2.5, 100.0, size=n).astype(np.float32)
    n_passengers = rng.integers(1, 5, size=n).astype(np.int32)
    return pl.DataFrame(
        {
            "pickup_lat": pickup_lat,
            "pickup_lon": pickup_lon,
            "drop_lat": drop_lat,
            "drop_lon": drop_lon,
            "pickup_ts": pickup_ts,
            "fare": fare,
            "n_passengers": n_passengers,
        }
    )


def haversine_expr() -> pl.Expr:
    """Approximate haversine distance in km between (pickup, drop) lat/lon."""
    R = 6371.0
    deg2rad = float(np.pi / 180.0)
    p_lat = pl.col("pickup_lat") * deg2rad
    p_lon = pl.col("pickup_lon") * deg2rad
    d_lat = pl.col("drop_lat") * deg2rad
    d_lon = pl.col("drop_lon") * deg2rad
    dlat = (d_lat - p_lat) / 2.0
    dlon = (d_lon - p_lon) / 2.0
    a = dlat.sin() ** 2 + p_lat.cos() * d_lat.cos() * dlon.sin() ** 2
    return 2.0 * R * a.sqrt().arcsin()


def main() -> None:
    N = 10_000_000
    df = make_taxi(N)

    print(f"\n=== NYC Taxi ETL ===  N={N:,}")
    print()

    time_callable(
        "haversine_only",
        lambda: df.select(haversine_expr().alias("dist_km")),
    )

    # Full ETL: hour-of-day extraction (mod 3600), haversine, then groupby
    time_callable(
        "haversine + n_passengers groupby (sum fare, mean dist)",
        lambda: (
            df.with_columns(dist_km=haversine_expr())
            .group_by("n_passengers")
            .agg(
                pl.col("fare").sum().alias("total_fare"),
                pl.col("dist_km").mean().alias("avg_dist"),
                pl.len().alias("n_trips"),
            )
        ),
    )


if __name__ == "__main__":
    main()
