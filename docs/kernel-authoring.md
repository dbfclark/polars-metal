# Writing kernels for polars-metal

(Stub — populated when the first kernel arrives in M1.)

Conventions:

- One MSL kernel per file in `shaders/`. Filename matches the entry-point name.
- Document threadgroup/grid assumptions at the top of the file.
- Query `MTLDevice` capabilities at runtime; do not hardcode threadgroup sizes.
- Always read the matching cuDF kernel in `references/cudf/cpp/src/` first.
- Always add a kernel-level differential test in `tests/kernel/` (CPU reference vs MSL output).
