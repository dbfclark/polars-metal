// crates/polars-metal-kernels/tests/test_filter_proptest.rs
//
// Filter compaction property tests were migrated from
// `tests/diff/test_filter_random.py` (Python hypothesis-based) to Rust
// proptest. The relevant coverage lives in
// `tests/test_compaction_pipeline.rs` (see `proptest!` blocks there):
//
//   - `compact_i64_matches_cpu_reference` — random (mask, source-i64) pairs
//   - `compact_f64_matches_cpu_reference` — random (mask, source-f64) pairs
//     including NaN payloads, ±Inf, and ±0.0
//   - `compact_bool_matches_cpu_reference` — random (mask, source-bool) pairs
//
// Each proptest runs at 64 cases (bounded to avoid Metal resource exhaustion
// across sequential GPU allocations in a single test process). All three
// verify that `compute_keep_and_prefix` + `compact_{i64,f64,bool}` produce
// output byte-identical to the pure-Rust CPU reference.
//
// This file is a stable pointer for future kernel authors; no test code lives
// here.
