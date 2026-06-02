"""MLX GPU ceiling for haversine distance.

NYC Taxi haversine on Polars CPU takes 181ms for 10M rows (measured in
bench_nyc_taxi.py). The bottleneck is transcendental latency: sin/cos/
sqrt/arcsin each cost 20-40 CPU cycles vs ~1 cycle in dedicated GPU SFUs.

This bench measures what an idealized routed-to-MLX implementation would
take. It's the ceiling for a polars-metal engine that recognizes the
haversine expression shape.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from tests.bench.m4_survey._timing import time_callable


def haversine_mlx(p_lat, p_lon, d_lat, d_lon):
    R = 6371.0
    deg2rad = float(np.pi / 180.0)
    p_lat_r = p_lat * deg2rad
    d_lat_r = d_lat * deg2rad
    dlat = (d_lat_r - p_lat_r) / 2.0
    dlon = (d_lon - p_lon) * deg2rad / 2.0
    a = mx.sin(dlat) ** 2 + mx.cos(p_lat_r) * mx.cos(d_lat_r) * mx.sin(dlon) ** 2
    out = 2.0 * R * mx.arcsin(mx.sqrt(a))
    mx.eval(out)
    return out


def main() -> None:
    N = 10_000_000
    rng = np.random.default_rng(0xCAB)
    p_lat = rng.uniform(40.6, 40.9, size=N).astype(np.float32)
    p_lon = rng.uniform(-74.05, -73.7, size=N).astype(np.float32)
    d_lat = rng.uniform(40.6, 40.9, size=N).astype(np.float32)
    d_lon = rng.uniform(-74.05, -73.7, size=N).astype(np.float32)

    p_lat_mx = mx.array(p_lat)
    p_lon_mx = mx.array(p_lon)
    d_lat_mx = mx.array(d_lat)
    d_lon_mx = mx.array(d_lon)
    mx.eval(p_lat_mx, p_lon_mx, d_lat_mx, d_lon_mx)

    print(f"\n=== MLX haversine ceiling ===  N={N:,}")
    time_callable(
        "haversine_mlx",
        lambda: haversine_mlx(p_lat_mx, p_lon_mx, d_lat_mx, d_lon_mx),
    )

    # numpy reference (the lower-bound CPU implementation)
    def haversine_numpy():
        R = 6371.0
        d2r = np.pi / 180.0
        plr = p_lat * d2r
        dlr = d_lat * d2r
        dlat = (dlr - plr) / 2.0
        dlon = (d_lon - p_lon) * d2r / 2.0
        a = np.sin(dlat) ** 2 + np.cos(plr) * np.cos(dlr) * np.sin(dlon) ** 2
        return 2.0 * R * np.arcsin(np.sqrt(a))

    time_callable("haversine_numpy", haversine_numpy)


if __name__ == "__main__":
    main()
