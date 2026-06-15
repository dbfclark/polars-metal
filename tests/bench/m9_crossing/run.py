"""Driver: time every pipeline path across sizes, fit the (alpha, beta) cost model,
emit the report. No timing logic of its own (delegates to m8 measure)."""

from __future__ import annotations

from dataclasses import dataclass

from tests.bench.m8_report._timing import measure
from tests.bench.m9_crossing._crossing import CostModel, fit_cost_model
from tests.bench.m9_crossing._pipelines import PIPELINES, PipelineSpec


@dataclass
class Row:
    pipeline: str
    family: str
    size: int
    path: str
    ms: float
    vs_all_cpu: float  # all_cpu_ms / this_ms  (>1 = faster than all-CPU)


def run(pipelines: list[PipelineSpec] = PIPELINES) -> tuple[list[Row], CostModel]:
    rows: list[Row] = []
    for p in pipelines:
        for size in p.sizes:
            inp = p.make_inputs(size)
            times = {
                name: measure(lambda fn=fn, inp=inp: fn(inp)).median_ms
                for name, fn in p.paths.items()
            }
            base = times["all_cpu"]
            for name, ms in times.items():
                rows.append(Row(p.name, p.family, size, name, ms, base / ms))
                print(
                    f"{p.name:18s} N={size:>10,} {name:14s} {ms:9.2f}ms  {base / ms:6.2f}x vs all_cpu"
                )
    cost = fit_cost_model()
    print(
        f"\ncost model: alpha={cost.alpha_ms_per_byte:.3e} ms/byte  beta={cost.beta_ms_per_crossing:.4f} ms/crossing"
    )
    return rows, cost


def main() -> None:
    from tests.bench.m9_crossing.emit import write_report

    rows, cost = run()
    write_report(rows, cost, md_path="docs/crossing-tax-report.md", json_path="crossing-tax.json")
    print(f"\nWrote docs/crossing-tax-report.md + crossing-tax.json ({len(rows)} rows)")


if __name__ == "__main__":
    main()
