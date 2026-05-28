"""String / regex workloads on Polars CPU at M2-Ultra scale.

Strings are interesting for Metal because:
  - Polars' string kernels are already highly optimized (SIMD where possible,
    but most string ops are intrinsically branchy and serial).
  - For regex specifically, Polars uses the `regex` crate; per-row eval is
    a DFA walk with ops/byte that varies wildly by pattern.

Workloads measured:
  - contains_literal: substring search; Polars uses memchr/SIMD.
  - contains_regex: same shape but with a small alternation.
  - to_lowercase: per-byte transform, SIMD-friendly.
  - len_chars: UTF-8 codepoint count per row.
  - split_count: count of occurrences of a delimiter.

Compute density is moderate for to_lowercase (1 op/byte), borderline for
contains_literal (~0.1 op/byte but SIMD-friendly), and HIGH for regex
(20-100 ops/byte) — the only realistic Metal-win category here.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from tests.bench.m4_survey._timing import time_callable


def make_strings(n: int, *, seed: int = 0xC0DE) -> pl.DataFrame:
    """Random strings drawn from a fixed corpus, length 20-80.

    Realistic structure for ETL workloads: mostly short tokens with
    occasional long descriptive strings.
    """
    rng = np.random.default_rng(seed)
    words = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
        "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi",
        "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
        "apple", "banana", "cherry", "date", "elderberry", "fig", "grape",
        "honeydew", "import", "export", "session-id-12345", "ABCDEFG",
        "metric.cpu.user", "metric.cpu.system", "metric.mem.used",
    ]
    out = []
    for _ in range(n):
        k = rng.integers(2, 8)
        out.append(" ".join(words[i] for i in rng.integers(0, len(words), size=k)))
    return pl.DataFrame({"s": out})


def main() -> None:
    N = 2_000_000
    print(f"\n=== string / regex benchmarks ===  N={N:,}")
    print(f"  (string generation takes ~10s; one-time)")
    df = make_strings(N)
    total_bytes = sum(len(s) for s in df["s"].to_list())
    print(f"  total string bytes: {total_bytes / 1e6:.1f} MB")
    print()

    time_callable(
        "str.contains[literal=alpha]",
        lambda: df.select(pl.col("s").str.contains("alpha", literal=True)),
    )

    time_callable(
        "str.contains[regex=alpha|beta|gamma]",
        lambda: df.select(pl.col("s").str.contains(r"alpha|beta|gamma")),
    )

    time_callable(
        "str.contains[regex=session-id-\\d+]",
        lambda: df.select(pl.col("s").str.contains(r"session-id-\d+")),
    )

    time_callable(
        "str.to_lowercase",
        lambda: df.select(pl.col("s").str.to_lowercase()),
    )

    time_callable(
        "str.len_chars",
        lambda: df.select(pl.col("s").str.len_chars()),
    )

    time_callable(
        "str.split[' '].list.len",
        lambda: df.select(pl.col("s").str.split(" ").list.len()),
    )

    time_callable(
        "str.replace_all[regex]",
        lambda: df.select(pl.col("s").str.replace_all(r"metric\.([a-z]+)\.([a-z]+)", r"$1_$2")),
    )


if __name__ == "__main__":
    main()
