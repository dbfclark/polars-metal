# Draft: proposing an engine-registration hook in Polars

> **Status:** Draft only. Do not file until project owner approves.

## Use case

We're building `polars-metal`, an Apple Silicon Metal-backed execution engine for Polars. Architecturally it's analogous to cuDF-Polars but targets MLX + custom MSL kernels instead of CUDA + libcudf.

Today, `polars.lazyframe.frame._gpu_engine_callback` is the dispatch point for the GPU engine, and it has the string `"gpu"` and `isinstance(engine, GPUEngine)` hardcoded as routing keys. The valid engine strings (`auto`/`cpu`/`in-memory`/`streaming`/`gpu`) are a closed allow-list. Third-party engines aren't routable through the public `engine=` parameter.

In Polars 1.40, `LazyFrame.collect` also passes `engine=` directly through to the Rust `ldf.collect()` call, which accepts only strings. As of M0, polars-metal works around this with a two-site monkey-patch (one on `_gpu_engine_callback`, one on `LazyFrame.collect`), using the internal `post_opt_callback` hook to inject our callback into the execution path. That works but couples us tightly to private Polars internals.

## What we'd like

A generic engine-registration hook so third-party engines can register a name and a callback, and `df.collect(engine="<name>")` (or `engine=<ConfigObject>`) routes to them.

Concretely, something like:

```python
import polars as pl

def my_engine_callback(nt, duration_since_start, *, config):
    ...

pl.register_engine("metal", my_engine_callback, config_type=MetalEngine)
```

After registration:
- `df.collect(engine="metal")` routes to `my_engine_callback` with `config=MetalEngine()` (the default).
- `df.collect(engine=MetalEngine(debug=True))` routes to `my_engine_callback` with that config instance.

## Why this is worth doing

- Removes the hardcoded routing and lets other GPU backends (Metal, ROCm, etc.) register cleanly.
- Avoids the monkey-patching pattern that polars-metal uses today, which we'd retire as soon as the hook lands.
- Mirrors a registry pattern already common in other parts of the Polars ecosystem.

## What we're not asking for

- Any concession about engine quality or maintenance — that's our problem.
- Inclusion of polars-metal in the Polars repo or organization.
- Stability guarantees on the IR; we pin and bump deliberately.

## Implementation sketch

Add a module-level dict `_engine_registry: dict[str | type, EngineEntry]` and a `register_engine(name, callback, config_type=None)` function. Inside both `_gpu_engine_callback` and the `LazyFrame.collect` engine-handling branch, after the existing string and `GPUEngine` checks, look up the registry by `engine` string or by `type(engine)`. If found, route to the registered callback (with appropriate `post_opt_callback` plumbing if needed).

We're happy to draft a PR if there's interest. Open question for maintainers: should the registry live in `polars-python` (Rust) or in `polars/lazyframe/frame.py` (Python)? Our weak preference is Python, because every engine plugin so far has been Python-callback-shaped.

## Pointer

The polars-metal repo (private during early development): https://github.com/dbfclark/polars-metal
