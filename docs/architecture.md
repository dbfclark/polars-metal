# polars-metal architecture

See [CLAUDE.md](../CLAUDE.md) for the high-level architecture statement and [the master plan](superpowers/specs/2026-05-19-master-plan-design.md) for the milestone-by-milestone roadmap.

This file deepens as each milestone lands. As of M0:

- Cargo workspace with four crates: `polars-metal-buffer`, `polars-metal-mlx-sys`, `polars-metal-kernels`, `polars-metal-core`.
- Python entry point `polars_metal._callback.execute_with_metal` is registered with Polars via an import-time monkey-patch on `polars.lazyframe.frame._gpu_engine_callback`.
- The walker walks the Polars IR via `nt.view_current_node()` and does nothing in M0 — every node falls back to CPU.
