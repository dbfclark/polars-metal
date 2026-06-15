"""Artifact emitters: rows -> markdown report + JSON twin. No measurement."""

from __future__ import annotations

import json
import platform
from collections import defaultdict
from dataclasses import asdict

import numpy as np
import polars as pl

from tests.bench.m8_report._timing import DEFAULT_ITERS, DEFAULT_WARMUP


def build_header() -> dict[str, str]:
    try:
        import mlx.core as mx

        mlx_version = getattr(mx, "__version__", "unknown")
    except Exception:
        mlx_version = "unavailable"
    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "polars_version": pl.__version__,
        "numpy_version": np.__version__,
        "mlx_version": mlx_version,
        "methodology": f"warmup={DEFAULT_WARMUP}, iters={DEFAULT_ITERS}, "
        "median reported, engine path includes ingest+fold-back",
    }


def _fmt(x) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def to_markdown(rows, header) -> str:
    lines = ["# polars-metal — honest perf report", ""]
    lines.append(
        "> Internal decision-input. Numbers are machine-specific (see header). "
        'Engine path is full `engine="metal"` wall-clock incl. ingest + fold-back.'
    )
    lines.append("")
    lines.append("## Environment")
    for k, v in header.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    ge10 = sum(1 for r in rows if r.engine_vs_cpu >= 10.0)
    wins = sum(1 for r in rows if r.engine_vs_cpu > 1.15)
    losses = sum(1 for r in rows if r.engine_vs_cpu < 0.85)
    lines += [
        "## Executive scorecard",
        "",
        f"- Rows clearing **≥10× vs CPU** (order-of-magnitude bar): **{ge10}** / {len(rows)}",  # noqa: RUF001
        f"- Rows that win (>1.15×): **{wins}** / {len(rows)}",  # noqa: RUF001
        f"- Rows that tie/lose: **{len(rows) - wins}** (losses: {losses})",
        "",
    ]

    by_cat: dict[str, list] = defaultdict(list)
    for r in rows:
        by_cat[r.category].append(r)
    for cat, crows in by_cat.items():
        lines.append(f"## {cat}")
        lines.append("")
        lines.append(
            "| op | size | engine ms | CPU ms | engine ×CPU | ceiling ms | ceiling ×CPU | tax | verdict |"  # noqa: RUF001
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for r in crows:
            lines.append(
                f"| {r.name} | {r.size:,} | {_fmt(r.engine_ms)} | {_fmt(r.cpu_ms)} | "
                f"{_fmt(r.engine_vs_cpu)}× | {_fmt(r.ceiling_ms)} | "  # noqa: RUF001
                f"{_fmt(r.ceiling_vs_cpu)} | {_fmt(r.tax)} | {r.verdict} |"
            )
        lines.append("")

    lines.append("<!-- VERDICT: filled in Task 13 after a real run -->")
    lines.append("")
    return "\n".join(lines)


def to_json(rows, header) -> str:
    return json.dumps(
        {"header": header, "rows": [asdict(r) for r in rows]},
        indent=2,
    )


def write_report(rows, *, md_path: str, json_path: str) -> None:
    header = build_header()
    with open(md_path, "w") as f:
        f.write(to_markdown(rows, header))
    with open(json_path, "w") as f:
        f.write(to_json(rows, header))
