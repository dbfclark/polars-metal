# Authoring kernels for polars-metal

This guide is the maintainer's reference for adding a new MSL kernel to
`shaders/` and its dispatcher to `crates/polars-metal-kernels/`. It
codifies the conventions established by M1's five shaders (`cmp_i64`,
`cmp_f64`, `logical_bool`, `filter_predicate`, `filter_scatter`). A new
kernel that follows these rules slots into the existing pipeline
without surprises; one that deviates will fail in subtle ways (silent
output corruption, mis-aligned atomic casts, race-induced bit loss).

For *what* M1 kernels do at the semantic level, see the
[M1 design spec](superpowers/specs/2026-05-20-m1-design.md). This
document covers *how* to write them.

## Reading list before you write code

1. **The matching cuDF C++/CUDA kernel.** Per CLAUDE.md, every MSL
   kernel has a cuDF ancestor whose algorithm we port. For filter
   compaction read
   `references/cudf/cpp/include/cudf/detail/copy_if.cuh` (the
   prefix-sum-then-scatter pipeline as a single template) and
   `references/cudf/cpp/src/copying/scatter.cu` (the per-thread
   parallel scatter step that lands each surviving row at its
   prefix-sum index). For comparison families read
   `references/cudf/cpp/src/binaryop/compiled/binary_ops.cuh` (the
   `apply_binary_op` template machinery and op-struct dispatcher) and
   `references/cudf/cpp/src/binaryop/compiled/operation.cuh` (the six
   `Equal` / `Less` / `Greater` / `LessEqual` / `GreaterEqual` /
   `NotEqual` op structs whose semantics M1's `cmp_i64` ports). Do
   not re-derive the algorithm — port it.
2. **`shaders/_validity.metal`.** The shared null-bitmap helpers
   (`get_valid`, `set_valid_nonatomic`, `set_valid_atomic_or`). Every
   null-aware kernel includes this header; the helpers' contracts are
   the contract every new kernel must honour.
3. **The closest existing kernel.** Match the shape of an existing
   one — `shaders/cmp_i64.metal` for arithmetic-driven elementwise
   producing bit-packed bool; `shaders/filter_scatter.metal` for an
   index-driven write into a compacted output;
   `shaders/logical_bool.metal` for bit-packed-bool-in,
   bit-packed-bool-out 3-valued operations.
4. **The matching dispatcher in `crates/polars-metal-kernels/src/`**
   (`cmp.rs`, `logical.rs`, `filter.rs`). Each dispatcher's
   top-of-file comment is the call-site contract: input length checks,
   output buffer sizing, zero-init expectations.
5. **The M1 design spec, § "Predicate compaction kernels" and
   § "Logical kernels".** Authoritative source for the null and NaN
   semantics — do not restate them, link.

## File layout and naming

One MSL kernel family per file. A *family* is the closed set of
related entry points sharing a kernel body: `cmp_i64.metal` defines
all twelve i64 comparison entry points (six ops × two arities) from
one `CMP_KERNEL_CC` / `CMP_KERNEL_CS` template;
`filter_scatter.metal` defines `filter_scatter_i64`,
`filter_scatter_f64`, and `filter_scatter_bool` because the three
variants share the prefix-sum + sentinel-overrun protocol but differ
in their per-slot copy. Do not split a family across files; do not
combine two unrelated families into one.

Filename matches the family name. The MSL entry-point names inside
the file extend it with a per-op suffix (`cmp_i64_lt`,
`filter_scatter_bool`). The Rust dispatcher's
`Op::entry_point_*` lookup table is the only place those strings
appear on the host side; keep it in sync.

**Leading-underscore = header-only.** Any `.metal` file whose stem
starts with `_` (e.g. `_validity.metal`) is treated as a header by
`crates/polars-metal-kernels/build.rs`: the build skips it for
`xcrun metal -c` but passes `-I <shaders_dir>` so `#include
"_validity.metal"` resolves from any sibling. Use this convention for
any future shared header. New per-kernel headers do *not* belong
here — they belong inline in the kernel file unless at least two
kernels need them.

## Null semantics (non-negotiable)

Polars' null rules are the spec; mismatches are bugs even if
"mathematically reasonable." The canonical truth tables for AND/OR
live in `shaders/logical_bool.metal`'s top-of-file comment and in
`crates/polars-metal-kernels/src/logical.rs`'s module-level doc.
Cite, don't restate.

For elementwise ops the contract is:

- Inputs carry a bit-packed validity bitmap (one bit per row, 8 rows
  per byte, little-endian Arrow layout — see
  `crates/polars-metal-buffer/src/null_bitmap.rs`).
- Output validity = AND of input validities (column-column) or just
  `lhs_valid` (column-scalar — scalars are always-valid; the walker
  only lowers non-null literals).
- Output data bit is set only at rows where output validity is set
  *and* the per-row computation is true. Null rows leave the output
  data bit at zero. This is the "writes are append-only" property
  the atomic-OR pattern depends on.

For `filter_scatter`, output validity is a *direct copy* of the
surviving rows' validity bits, not an AND.

## NaN semantics for f64

Polars CPU uses **TotalOrd** semantics for floating-point comparison:
NaN is greater than any non-NaN, `NaN == NaN` is `true`, `NaN != NaN`
is `false`. M1's `shaders/cmp_f64.metal` currently implements **IEEE
754** semantics — `NaN OP x` is `false` for `==/</<=/>/>=` and `true`
only for `!=`. This is a known divergence tracked in
`docs/open-questions.md` § "cmp_f64 NaN semantics".

Any future kernel touching f64 ordering or equality must implement
TotalOrd. The fix sketch in the open-questions entry is the design:
rework the six `f64_<op>` helpers in `cmp_f64.metal` to special-case
NaN-presence into the TotalOrd outcome rather than the IEEE one, and
extend `f64_total_order_key` to map NaN consistently above ±Inf.

MSL's `double` is unavailable on Apple Silicon compute kernels. f64
values are bound to MSL as `device const ulong*` and compared in
integer arithmetic on the 8-byte bit pattern; the encoding is exact.
See `shaders/cmp_f64.metal`'s top-of-file comment for the
`f64_total_order_key` derivation.

## The atomic-OR pattern (bit-packed outputs)

Whenever 8 output rows share a byte — every bit-packed bool data
output, every validity bitmap — multiple threads will race that byte.
The non-atomic `set_valid_nonatomic` in `_validity.metal` would
corrupt the bitmap *in that case*. The non-atomic variant is still
the correct choice when each thread owns a unique byte (sequential
fill into a pre-zeroed scratch buffer, for example) — see its
header comment in `_validity.metal`. No M1 kernel currently uses it;
every M1 output that touches a bitmap goes through atomic OR. The
fix everywhere there *is* sharing is:

1. Bind the output as `device atomic_uint* [[buffer(N)]]` in the MSL
   signature (see any kernel signature in `cmp_i64.metal`,
   `cmp_f64.metal`, `logical_bool.metal`, or
   `filter_scatter.metal`).
2. Set bits via `set_valid_atomic_or(out_buf, row_idx)` from
   `_validity.metal`. The helper does the row → u32-word + bit-in-word
   arithmetic and emits an `atomic_fetch_or_explicit` with
   `memory_order_relaxed`.
3. The host **must** zero-initialise the buffer before dispatch
   (atomic OR is append-only — it never clears bits — so any
   pre-existing bit becomes a spurious 1).
4. The host **must** allocate the buffer in multiples of 4 bytes
   (minimum 4) so the `atomic_uint*` cast is well-aligned.

Skip the atomic OR only when each thread writes a unique byte:
`filter_predicate_to_u8` writes a dense `u8` keep-flag (one byte per
row, no sharing), and `filter_scatter_i64` / `filter_scatter_f64`
write 8-byte slots in `dst_data` (no sharing). The validity output
of those scatters still goes through the atomic OR.

`filter_scatter_bool` is the only scatter where the *data* output
is also bit-packed and shares the same multi-thread-one-byte race;
it uses atomic OR for both `dst_data` and `dst_valid`.

## Zero-init outputs before dispatch

Mandatory whenever the kernel uses atomic OR on the output. The Rust
dispatchers handle this via `MetalDevice::new_buffer_zeroed`:

```rust
let out_data_buf = device.new_buffer_zeroed(min_out)?;
let out_valid_buf = device.new_buffer_zeroed(min_out)?;
```

A future kernel that adds an atomic-OR output buffer must follow
suit. Reusing a previously-written buffer without zeroing is a
silent-corruption bug — the OR will keep bits from the prior
dispatch's results.

## 4-byte alignment for `device atomic_uint*`

Apple Silicon requires `atomic_uint` accesses to be 4-byte aligned.
`MTLBuffer` allocations are already 16-byte aligned by Metal, so the
buffer's base address is fine — the constraint applies to the
*length*. The minimum-output-bytes helper in every dispatcher
(`cmp::out_min_bytes`, `logical::out_min_bytes`,
`filter::dst_valid_min_bytes` — same body, the `filter` variant is
named for the validity buffer it sizes) encodes the rule:

```rust
fn out_min_bytes(n_rows: usize) -> usize {
    let raw = (n_rows + 7) / 8;
    let padded = (raw + 3) & !3;   // round up to next 4-byte boundary
    padded.max(4)                  // minimum 4 bytes for any atomic_uint access
}
```

Use exactly this formula for any new atomic-OR output. Forgetting
the `.max(4)` allows zero-row dispatches to allocate a 0-byte buffer,
which Metal rejects with a confusing error.

## The macro-from-template pattern

When a kernel family has more than three entry points sharing the
same body, factor the body into a preprocessor macro. The canonical
example is `shaders/cmp_i64.metal`. Here is the actual macro body — a
new kernel family's macro should be shaped the same way: full
binding list, early-exit on `gid >= n_rows`, both `get_valid` reads,
combined null short-circuit, `set_valid_atomic_or` for validity,
then the conditional `set_valid_atomic_or` for the data bit.

```msl
#define CMP_KERNEL_CC(name, op)                                       \
kernel void name(                                                     \
    device const int64_t*    lhs_data    [[buffer(0)]],               \
    device const uint8_t*    lhs_valid   [[buffer(1)]],               \
    device const int64_t*    rhs_data    [[buffer(2)]],               \
    device const uint8_t*    rhs_valid   [[buffer(3)]],               \
    device       atomic_uint* out_data   [[buffer(4)]],               \
    device       atomic_uint* out_valid  [[buffer(5)]],               \
    constant     uint32_t&   n_rows      [[buffer(6)]],               \
    uint                     gid         [[thread_position_in_grid]]) \
{                                                                     \
    if (gid >= n_rows) return;                                        \
    bool lv = get_valid(lhs_valid, gid);                              \
    bool rv = get_valid(rhs_valid, gid);                              \
    if (!lv || !rv) return;                                           \
    set_valid_atomic_or(out_valid, gid);                              \
    if (lhs_data[gid] op rhs_data[gid]) {                             \
        set_valid_atomic_or(out_data, gid);                           \
    }                                                                 \
}

CMP_KERNEL_CC(cmp_i64_eq, ==)
CMP_KERNEL_CC(cmp_i64_ne, !=)
/* ... four more ... */
```

`cmp_f64.metal` uses the same pattern but parameterises on a helper
function name (`f64_eq`, `f64_lt`, …) rather than an MSL operator,
because the NaN-handling helpers can't be inlined as operators.

Don't reach for the macro pattern when there are only two or three
entry points (e.g. `logical_bool.metal`'s `bool_and` /
`bool_or`) — the duplication is fine and the macros become harder to
read than the explicit kernels.

## Threadgroup sizing

Never hardcode. M1, M2, M3, M4 have different
`maxThreadsPerThreadgroup`; a literal `256` works on M2 Ultra and
crashes on small M1 / iPad-class devices.
`crates/polars-metal-kernels/src/command.rs::dispatch_1d` is the
canonical helper: it queries `pso.maxTotalThreadsPerThreadgroup()`
at runtime and clamps to `DEFAULT_THREADGROUP_WIDTH = 256`. For
specialised kernels that have measured an optimal width, call
`dispatch_1d_with_tg` directly; the width is still clamped against
the PSO's runtime maximum.

`dispatchThreads:threadsPerThreadgroup:` (not
`dispatchThreadgroups:`) is used throughout so non-power-of-two grids
work — Metal pads the trailing threadgroup with no-op threads whose
`thread_position_in_grid` is out-of-range. Every kernel therefore
needs an `if (gid >= n_rows) return;` early-exit; the M1 kernels
follow this without exception.

## Dispatcher conventions

The Rust dispatcher is the host-side contract enforcement layer. Each
new dispatcher should:

- Live in a per-family module (`cmp.rs`, `logical.rs`, `filter.rs`)
  under `crates/polars-metal-kernels/src/`.
- Length-check all inputs and outputs *before* allocating any
  `MTLBuffer`. The errors are per-crate `*Error` enums
  (`CmpError`, `LogicalError`, `FilterError`) deriving
  `thiserror::Error` and `From<ShaderError>` / `From<DispatchError>`
  / `From<BufferError>`.
- Treat `n_rows == 0` as a no-op: zero the caller's output slices
  (so the post-dispatch read sees a clean buffer) and return
  `Ok(())`. Metal rejects zero-thread dispatches and zero-byte
  buffer allocations, so this branch is a host-side concern only.
- Cast typed input slices to `&[u8]` via
  `std::slice::from_raw_parts` for `new_buffer_from_bytes` (no
  alignment requirement on the source slice; the byte length is
  `std::mem::size_of_val(slice)`). Every such cast needs a `//
  SAFETY:` comment naming the invariant — every `iN` / `uN` bit
  pattern is valid, the slice outlives the synchronous copy.
- Bind scalar arguments as little-endian raw bytes
  (`n.to_le_bytes()`, `f.to_bits().to_le_bytes()`). MSL's `constant
  T&` reads exactly `sizeof(T)` bytes from the bound buffer.
- Copy the kernel outputs back from the device buffer into the
  caller's `&mut [u8]` via `copy_from_slice` on the
  `out_min_bytes(n_rows)` prefix — the buffer is allocated padded;
  the caller's slice is not.

The "dispatcher 1:1 with kernel" rule means: one Rust dispatch
function per MSL entry-point family, *not* per individual entry
point. `dispatch_cmp_i64(... op: CompareOp ...)` selects between the
six column-column entry points; `dispatch_cmp_i64_scalar(... op:
CompareOp ...)` selects between the six column-scalar entry points.

## Test discipline

Every new kernel must add `tests/test_<kernel>.rs` containing:

1. A deterministic CPU reference function (e.g. `cpu_cmp_cc` in
   `tests/test_cmp_i64.rs`) that mirrors the kernel's null-and-data
   contract bit-for-bit. The kernel is asserted byte-equal to the
   reference, not approximately equal.
2. A handful of fixed-shape unit tests covering: all-valid input,
   half-null input, all-null input, empty input, single-row input,
   and a multi-thread-same-byte stress (≥ 256 rows so multiple
   threadgroups race the same output bytes).
3. A `proptest!` block with `ProptestConfig::with_cases(256)` driving
   randomised inputs through the reference and the kernel.

Every test acquires the process-wide `METAL_TEST_LOCK` mutex first:

```rust
static METAL_TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn cmp_i64_lt_basic() {
    let _lock = METAL_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    /* ... */
}
```

This serialises Metal access across `cargo test`'s default
multi-threaded scheduler. Without it, parallel workers thrash the
system shader cache and trigger "Internal Error 00000206" — first
hit during Task 14, fixed by the lock pattern that every subsequent
test file copies. The `unwrap_or_else(|p| p.into_inner())` is
mandatory: a panic in one test poisons the mutex; the recovery
unpoisons it so the remaining tests still run.

## Criterion benchmarks

Each new kernel also needs `benches/<kernel>.rs` registered under
`crates/polars-metal-kernels/Cargo.toml`'s `[[bench]]` table. Mirror
`benches/cmp_i64.rs`:

- Three row counts: 1K, 100K, 10M.
- Three validity densities: 1.0 (all-valid), 0.5 (alternating), 0.0
  (all-null).
- `iter_batched` with fresh output buffers per iteration so the
  atomic-OR semantics never observe stale bits across runs.
- Deterministic input — no PRNG dependency; the bench measures
  shape, not statistical realism.

The M1 bar is "runs without errors and produces stable numbers per
run"; performance comparisons across machines and milestones live in
the M1 retrospective and M2 design, not in commit logs.

## Output sentinels (`filter_scatter` pattern)

`filter_scatter_i64` and `filter_scatter_f64` allocate `n_out + 1`
slots and write a known-bad value (`SCATTER_SENTINEL_I64`,
`SCATTER_SENTINEL_F64_BITS`) into the trailing slot whenever a buggy
prefix sum would produce `out_idx >= n_out`. The host checks the
sentinel post-dispatch and raises `FilterError::ScatterOverrun`
rather than letting silently-corrupt output reach Polars.

Adopt this pattern for any new index-driven scatter where a
recognizable magic value exists in the output type's domain. It is
not applicable when every value of the output type is legitimate
data — `filter_scatter_bool` has no sentinel because every bit is
meaningful; instead the host pre-verifies the prefix-sum invariant
`prefix_sum[n_rows - 1] == n_out` before dispatch, which the kernel
relies on to keep `out_idx < n_out` for every thread.

## Common pitfalls

- **Forgetting to zero-init an atomic-OR output.** Silent corruption:
  bits from a previous dispatch (or whatever the allocator returned)
  bleed into the result. Always allocate via
  `MetalDevice::new_buffer_zeroed`, not `new_buffer_from_bytes` over
  a stale slice.
- **Hardcoding threadgroup width.** Works on the M2 Ultra dev box;
  crashes or under-utilises on smaller M-series. Go through
  `CommandQueue::dispatch_1d` or pass a measured width to
  `dispatch_1d_with_tg`; both clamp to the PSO's runtime maximum.
- **Missing 4-byte alignment for `atomic_uint*`.** Undefined behaviour
  per Metal; usually crashes with "command buffer GPU error" but can
  also corrupt silently. Use the `out_min_bytes` helper verbatim.
- **Validity-bitmap off-by-one.** Bit-packed buffers are
  `(n_rows + 7) / 8` bytes long. The trailing byte's upper bits are
  unspecified — never assume zero, never assert non-zero. Tests
  should mask before comparing (`got_valid[last] & ((1 << n_in_last)
  - 1)`).
- **Mixing IEEE-754 and TotalOrd NaN semantics.** Polars CPU is the
  spec; M1's `cmp_f64` currently disagrees with it on NaN. Until
  the open-question fix lands, every f64 conformance test that
  could see a NaN must either (a) inject no NaNs, or (b) skip /
  `xfail` with a pointer to `docs/open-questions.md`.
- **Reading kernel output before `wait_until_complete`.** The
  command buffer is committed asynchronously; the CPU and GPU race
  the buffer. Every dispatcher in `cmp.rs`, `logical.rs`, and
  `filter.rs` ends with `queue.wait_until_complete()?` before
  `copy_from_slice` on the output.

## M2 additions

The sections below document patterns and constraints introduced in M2 that
every kernel author working on groupby, aggregation, or future hash-based
operations must know before writing code.

### Apple Silicon Metal atomic ops constraint

**This is the most important M2 constraint.** The Metal toolchain version
32023.883 (current as of M2) does not support the following operations:

- `atomic_fetch_add_explicit` on `atomic_long` / `atomic_ulong`
- `atomic_compare_exchange_weak_explicit` on `atomic_ulong`
- `double` in compute kernels (bind as `ulong` and compare on the bit pattern — see `cmp_f64.metal`)

Only `atomic_uint` and `atomic_int` (32-bit) have the full atomic op set.

Practical consequences for kernel authoring:

- **64-bit aggregation runs on CPU.** `crates/polars-metal-kernels/src/groupby.rs`
  contains `aggregate_sum_i64_cpu`, `aggregate_min_i64_cpu`,
  `aggregate_max_i64_cpu`, `aggregate_sum_f64_cpu`, etc. — Rayon-parallel
  CPU loops over the `row_to_group` array produced by the GPU build phase.
  These are not performance fallbacks; they are the correct dispatch for
  64-bit dtypes until Apple ships wider atomics.
- **32-bit aggregation has GPU kernels.** `shaders/aggregate.metal` provides
  `agg_sum_i32`, `agg_sum_u32`, `agg_sum_f32`, `agg_min_i32`, `agg_max_i32`,
  `agg_min_u32`, `agg_max_u32`, `agg_min_f32`, `agg_max_f32`, `agg_count`,
  and `agg_len`. Each uses `atomic_int*` or `atomic_uint*` exclusively.
- **Any kernel pattern that needs 64-bit fetch_add or 64-bit CAS is out of
  scope until Apple ships the wider atomic set.** Do not attempt to work
  around this with spin-wait loops — SIMD-group lockstep execution means one
  thread cannot wait for a sibling thread's write to advance. The M2 groupby
  build phase started as a GPU atomic-CAS design, hit exactly this deadlock,
  was redesigned to a 3-state non-spinning machine, and still deadlocked. It
  was ultimately moved to CPU (see
  [§ Two-pass groupby idiom](#two-pass-groupby-idiom-overview) below).
  This is a cautionary tale: the deadlock is structural, not a bug.
- **The dispatch matrix.** When adding a new aggregation op or dtype, first
  check whether the op requires 64-bit atomics. If yes, write a CPU finalise
  function in `groupby.rs` following the `aggregate_sum_i64_cpu` pattern.
  If no, write an MSL kernel in `shaders/aggregate.metal` following the
  `agg_sum_i32` pattern.

### Two-pass groupby idiom (overview)

The M2 hash groupby uses a two-pass discipline. The split is CPU / GPU,
not pass-1 / pass-2 in the traditional GPU sense:

**Pass 1 — build (CPU, `groupby.rs`).** A Rust `HashMap<u128, u32>`
find-or-insert over the encoded key vector (`Vec<u128>` produced by
`encode_keys`) identifies unique groups and assigns each row a group index.
Outputs:
- `row_to_group: Vec<u32>` — one entry per input row, value = group index.
- `group_count: Vec<u32>` — one entry per unique group, value = row count.
- `first_row_per_group: Vec<u32>` — one entry per unique group, value =
  index of the first row in that group (used for key decoding).

This build lives on CPU for the reasons described in § Apple Silicon Metal
atomic ops constraint above. Future M3 work may replace it with a GPU
sort-then-segment-reduce approach that avoids CAS entirely; see
[`docs/open-questions.md` § GroupBy build phase on CPU](open-questions.md).

**Pass 2 — aggregate (GPU for 32-bit dtypes, CPU for 64-bit).** For each
(value column, aggregation op) pair, dispatch one kernel:
- 32-bit dtypes: one of the `agg_*` kernels in `shaders/aggregate.metal`.
  Thread-per-row atomic-OP into `output[row_to_group[gid]]`.
- 64-bit dtypes: corresponding `aggregate_*_cpu` function in `groupby.rs`.

The caller must seed output buffers with the operator's identity element
before dispatch (0 for sum/count, INT32_MAX for min_i32, etc. — see the
`shaders/aggregate.metal` header comment for the full table).

### Per-aggregation MSL kernel pattern

`shaders/aggregate.metal` does not use the `#define` macro pattern because
the entry points differ in their buffer types (`atomic_int*` vs
`atomic_uint*`) and each is short enough to be explicit. This is the one
place where the macro pattern would reduce readability rather than improve it.

The structure of each kernel is identical:
1. Early-exit guard: `if (gid >= n_rows) return;`
2. Validity check: `if (!get_valid(valid, gid)) return;`
3. Group lookup: `uint g = row_to_group[gid];`
4. Atomic op: `atomic_fetch_add_explicit(&out[g], values[gid], memory_order_relaxed);`
   (or fetch_min / fetch_max / CAS-loop for f32).

New kernel families that follow a similar thread-per-row → atomic-OP-into-
output pattern should adopt this explicit-per-entry-point style (not the
macro pattern) when the number of entry points is small (≤ 3) or when
buffer types differ across entry points. Use the macro pattern from
`cmp_i64.metal` when there are ≥ 4 entry points with identical signatures
differing only by a single operator token.

Cite `shaders/aggregate.metal` for the full list of entry points and their
dispatcher contracts.

### Composite key encoding contract

CPU-side code packs N key columns into a single `u128` per row via
`crates/polars-metal-kernels/src/groupby.rs::encode_keys`. The contract:

- Each key column contributes `1 (null flag) + data_bits` bits to the
  packed u128, in column order from LSB to MSB.
- Supported key dtypes (`KeyDtype`): `Bool` (2 bits), `I64` / `F64` (65
  bits each).
- Total bit width must not exceed 128 bits. If it does, `encode_keys`
  returns `Err(KeyEncodeError::TooWide)` and the caller must fall back.
- The matching `decode_keys` function inverts the encoding to recover key
  column values for the output DataFrame.

The hash kernel (`shaders/groupby_hash.metal`) only ever sees `u128`
values. It does not know the `KeySchema`; that lives entirely on the Rust
side. Adding a new key dtype requires extending `KeyDtype`, updating the
`key_width_bits` helper in `cost.rs`, updating `encode_keys` /
`decode_keys` in `groupby.rs`, and adding round-trip proptest cases in
the test suite. The hash kernel does not need changes (it operates on
the raw `u128` bit pattern).

The 128-bit cap is tracked as a known limitation in
[`docs/open-questions.md` § Composite key 128-bit limit](open-questions.md).

## Process

1. **Read the cuDF source.** Don't reinvent the algorithm.
2. **Sketch the kernel signature** matching an existing one in
   shape (input layout, validity layout, output layout). If outputs
   are bit-packed, you're in the atomic-OR regime — see § The
   atomic-OR pattern (bit-packed outputs) for the binding + zero-init
   + 4-byte-alignment trio. Sentinel-guarded i64/f64 scatter outputs
   follow § Output sentinels (`filter_scatter` pattern).
3. **Write the dispatcher first**, with input/output length checks
   and the `n_rows == 0` no-op path. The dispatcher's error enum
   nails down what the kernel may assume.
4. **Write the MSL kernel.** Include `_validity.metal` if any
   null-bitmap access is involved.
5. **Write the unit + proptest file** with `METAL_TEST_LOCK` and a
   bit-for-bit CPU reference.
6. **Write the criterion benchmark.** Register it in `Cargo.toml`.
7. **`make lint`** — every rule above is checked, plus formatter
   conventions.
8. **`make gate`** — the unit tests, kernel tests, conformance
   suite, and differential suite must all pass before claiming the
   kernel is done.
