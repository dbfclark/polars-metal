"""rows + cost model -> markdown report + JSON. No measurement."""

from __future__ import annotations

import json
import platform
from collections import defaultdict
from dataclasses import asdict

import polars as pl

from tests.bench.m9_crossing._crossing import CostModel


def _header() -> dict[str, str]:
    try:
        import mlx.core as mx

        mlxv = getattr(mx, "__version__", "unknown")
    except Exception:
        mlxv = "unavailable"
    return {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "polars_version": pl.__version__,
        "mlx_version": mlxv,
    }


def to_markdown(rows, cost: CostModel, header: dict[str, str]) -> str:
    lines = ["# polars-metal — crossing-tax benchmark (M9)", ""]
    lines.append(
        "> Internal decision-input. Sizes the CPU<->GPU crossing tax on mixed compute+join"
    )
    lines.append(
        '> pipelines. Ratios are vs the all-CPU path (what `engine="metal"` does today on a join).'
    )
    lines.append("")
    lines.append("## Environment")
    for k, v in header.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Crossing cost model")
    lines.append("")
    lines.append("`crossing_ms ≈ alpha · bytes_crossed + beta · n_crossings`")
    lines.append("")
    lines.append(
        f"- **alpha** = {cost.alpha_ms_per_byte:.3e} ms/byte "
        f"(≈ {1.0 / (cost.alpha_ms_per_byte * 1e9):.1f} GB/s round-trip)"
    )
    lines.append(f"- **beta** = {cost.beta_ms_per_crossing:.4f} ms/crossing (fixed dispatch/sync)")
    lines.append("")
    by_pipe: dict[str, list] = defaultdict(list)
    for r in rows:
        by_pipe[r.pipeline].append(r)
    for pipe, prows in by_pipe.items():
        lines.append(f"## {pipe}  ({prows[0].family})")
        lines.append("")
        lines.append("| size | path | ms | × vs all_cpu |")  # noqa: RUF001
        lines.append("|---:|---|---:|---:|")
        for r in prows:
            lines.append(f"| {r.size:,} | {r.path} | {r.ms:.2f} | {r.vs_all_cpu:.2f}× |")  # noqa: RUF001
        lines.append("")
    lines.append("<!-- VERDICT: filled in Task 9 after a real run -->")
    lines.append("")
    return "\n".join(lines)


def to_json(rows, cost: CostModel, header: dict[str, str]) -> str:
    return json.dumps(
        {
            "header": header,
            "cost_model": {
                "alpha_ms_per_byte": cost.alpha_ms_per_byte,
                "beta_ms_per_crossing": cost.beta_ms_per_crossing,
            },
            "rows": [asdict(r) for r in rows],
        },
        indent=2,
    )


def write_report(rows, cost: CostModel, *, md_path: str, json_path: str) -> None:
    header = _header()
    with open(md_path, "w") as f:
        f.write(to_markdown(rows, cost, header))
    with open(json_path, "w") as f:
        f.write(to_json(rows, cost, header))
