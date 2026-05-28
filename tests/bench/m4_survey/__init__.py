"""M4 benchmark survey: CPU baselines for candidate workloads.

Each script in this directory measures a candidate workload at a meaningful
scale on Polars CPU. Findings are consolidated in docs/m4-benchmark-survey.md.

The goal is to identify workloads where Metal could plausibly outperform CPU
on M2 Ultra — i.e., compute-bound workloads with high FLOPs/byte, not the
bandwidth-bound shapes that TPC-H is dominated by.
"""
