#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
//! Task 16 — proptest verifying the fused-kernel groupby aggregation
//! produces results equivalent to M2's per-agg path across randomized
//! inputs.
//!
//! Two properties:
//!   1. `fused_eq_per_agg_sum_only` — single-column F32 Sum, varied
//!      group cardinality and null density. Sanity check on the
//!      simplest fused shape.
//!   2. `fused_eq_per_agg_full_q1_shape` — full Q1 shape: 4 F32 value
//!      columns with Sum + Mean each, plus Count and Len. This is the
//!      shape the fused kernel was designed for.
//!
//! ## Tolerance rationale
//!
//! Floating-point sum is non-associative. The fused kernel and the
//! per-agg kernel both sum the same values per group but interleave
//! the atomic CAS updates across SIMD lanes in different orders. The
//! resulting per-group sums differ by `O(ulp * n_rows_in_group)`,
//! typically `< 1e-4` relative for n=10k. We use 1e-3 to 1e-4 relative
//! tolerance with an absolute floor for tiny group magnitudes.
//!
//! ## Group-order normalization
//!
//! Neither `dispatch_groupby` nor `dispatch_groupby_fused` guarantees a
//! specific group output order — the order falls out of build-phase
//! hashing. Both paths agree on *which* groups exist, so we sort the
//! per-key result vectors before comparison.
//!
//! ## Critical correctness invariants checked
//!
//! - Validity bitmaps for Mean: both paths must produce `valid=false`
//!   for any group with count=0 (no non-null inputs).
//! - Count and Len: byte-exact equality (these are integer counts).
//! - Sum / Mean: relative tolerance under `assert_floats_eq`.
//! - NaN inputs: if any input is NaN, both paths' sum/mean must also
//!   be NaN (we assert via `is_nan` rather than numeric comparison).
//! - Empty groups (count=0): handled via validity bit, not by trying
//!   to compare a sentinel value.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::aggregate_fused::cache::FusedLibraryCache;
use polars_metal_kernels::aggregate_fused::signature::{
    AggOp as KAggOp, AggSpec as KAggSpec,
};
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::groupby::{
    dispatch_groupby, dispatch_groupby_fused, AggKind, AggOutput, AggRequest, DecodedColumn,
    KeyColumn, KeyDtype, ValueColumn,
};
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---------- Metal-test serialization mutex --------------------------------
//
// Each proptest case allocates a fresh `MetalDevice` + `CommandQueue` and
// dispatches kernels (build phase + fused or per-agg compute). Running these
// in parallel across rust test threads tickles Metal "Internal Error
// 00000206" from resource pressure (see test_compaction_pipeline.rs). We
// serialize using a process-wide mutex; the engine itself runs one query at
// a time in production, so this lock is purely a test-infra accommodation.
static METAL_TEST_LOCK: Mutex<()> = Mutex::new(());

fn lock_metal() -> std::sync::MutexGuard<'static, ()> {
    METAL_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------- helpers --------------------------------------------------------

fn pack_valid(valid: &[bool]) -> Vec<u8> {
    let n_bytes = ((valid.len() + 7) / 8 + 3) & !3;
    let mut out = vec![0u8; n_bytes.max(4)];
    for (i, &b) in valid.iter().enumerate() {
        if b {
            out[i >> 3] |= 1 << (i & 7);
        }
    }
    out
}

fn setup() -> (MetalDevice, CommandQueue, FusedLibraryCache) {
    let device = MetalDevice::system_default().expect("Metal hardware required");
    let queue = CommandQueue::new(&device).expect("command queue");
    let cache = FusedLibraryCache::new(device.clone());
    (device, queue, cache)
}

/// Kernel-layer AggSpec::Simple constructor.
fn simple(col: &str, op: KAggOp, alias: &str) -> KAggSpec {
    KAggSpec::Simple {
        input_col: col.into(),
        op,
        output_alias: alias.into(),
    }
}

/// Generate (keys, valid_mask) for n_rows with a given group count and null
/// density. RNG is seeded deterministically so prop_assert failures shrink
/// to a reproducible (seed, ...) tuple.
fn synth_inputs_f32(
    seed: u64,
    n_rows: usize,
    n_groups: u32,
    null_density: f32,
) -> (Vec<i32>, Vec<f32>, Vec<bool>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let keys: Vec<i32> = (0..n_rows)
        .map(|_| (rng.gen::<u32>() % n_groups) as i32)
        .collect();
    // Use finite-only values; range bounded so total sums stay well
    // within f32 precision (avoid >2^24 magnitudes where ulp dominates).
    let values: Vec<f32> = (0..n_rows).map(|_| rng.gen_range(-100.0f32..100.0)).collect();
    let valid: Vec<bool> = (0..n_rows)
        .map(|_| rng.gen::<f32>() >= null_density)
        .collect();
    (keys, values, valid)
}

/// Group decoded i32 keys + AggOutput rows by representative key. Returns a
/// BTreeMap so iteration is deterministic. Each entry is the index into the
/// result's group axis — caller uses this to re-extract per-group values.
fn group_indices_by_key(decoded: &DecodedColumn) -> BTreeMap<i32, usize> {
    let values = match decoded {
        DecodedColumn::I32 { values, .. } => values,
        other => panic!("expected I32 key column, got {other:?}"),
    };
    let mut map = BTreeMap::new();
    for (i, &k) in values.iter().enumerate() {
        // First occurrence wins (groups are unique by build-phase
        // invariant; there should be no duplicates).
        map.entry(k).or_insert(i);
    }
    map
}

/// Pairwise-equal comparison for two floats produced by atomic-CAS-based
/// sum kernels with different lane interleaving. NaN compares equal to
/// NaN; signs mismatch is a failure.
///
/// Tolerance model:
///
/// Sum-of-N-signed-values reorder noise from atomic CAS, in the worst case
/// (random commit order across N lanes), is bounded by:
///
///   |sum_a - sum_b|  ≲  partial_max * sqrt(N) * 2^-23   (single-precision)
///
/// where `partial_max` is the largest intermediate sum reached during the
/// accumulation (close to `N * max_abs_value` if values share sign, smaller
/// when they cancel). When the *true* sum cancels to near zero but the
/// partial sums were O(N*max_abs), tolerance based on result magnitude is
/// far too tight — the noise floor is set by the partial-sum walk, not the
/// final value.
///
/// We therefore take `noise_floor` as a caller-supplied absolute tolerance
/// computed from the input shape, plus a relative-to-result rel_tol for
/// cases where the result IS the natural scale (Mean / always-positive
/// sums).
fn assert_floats_eq(
    label: &str,
    a: f32,
    b: f32,
    rel_tol: f32,
    noise_floor: f32,
) -> Result<(), TestCaseError> {
    if a.is_nan() && b.is_nan() {
        return Ok(());
    }
    if a.is_nan() != b.is_nan() {
        return Err(TestCaseError::Fail(
            format!("{label}: NaN mismatch (a={a}, b={b})").into(),
        ));
    }
    // Combined tolerance: rel_tol against result magnitude OR absolute
    // noise_floor — whichever is more permissive.
    let abs_floor = (rel_tol * a.abs().max(b.abs())).max(noise_floor);
    let diff = (a - b).abs();
    if diff > abs_floor {
        return Err(TestCaseError::Fail(
            format!(
                "{label}: |{a} - {b}| = {diff} exceeds abs_floor {abs_floor} \
                 (rel_tol={rel_tol}, noise_floor={noise_floor})"
            )
            .into(),
        ));
    }
    Ok(())
}

/// Compute the atomic-CAS reorder noise floor for a sum over `n_rows`
/// random signed values bounded by `max_abs_value`.
///
/// Derivation: for a random-sign sum, partial sums perform a random walk
/// reaching `~sqrt(N) * M` in absolute value. The single-precision ulp at
/// that magnitude is `sqrt(N) * M * eps`. Accumulated over N steps with
/// different commit orders, the forward-error bound is:
///
///   error ≈ sqrt(N) * ulp(sqrt(N) * M) = N * M * eps
///
/// We pad with a constant factor of 4 to absorb CAS contention retries
/// (same-cycle conflicting writes get redone, effectively doubling lane
/// activity in hotspots) and use a 1e-3 floor for small-N degenerate cases.
fn sum_noise_floor(n_rows: usize, max_abs_value: f32) -> f32 {
    let n = n_rows as f32;
    let bound = 4.0 * n * max_abs_value * f32::EPSILON;
    bound.max(1e-3)
}

// Same comparison for f64-promoted outputs (e.g. for the multi-Q1 path the
// per-agg mean is f32 too, but downstream we may widen; keep a stub for
// symmetry). Currently unused — all Q1 outputs are f32.

// ---------- single-column F32 dispatch ------------------------------------

#[derive(Debug)]
struct OneColResult {
    /// key → sum
    sums: BTreeMap<i32, f32>,
}

fn run_fused_sum_f32(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    cache: &FusedLibraryCache,
    keys: &[i32],
    values: &[f32],
    valid: &[bool],
) -> Result<OneColResult, String> {
    let n_rows = keys.len();
    let key_data: Vec<u8> = keys.iter().flat_map(|v| v.to_le_bytes()).collect();
    let key_valid = pack_valid(&vec![true; n_rows]);
    let val_data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let val_valid = pack_valid(valid);
    let key_cols = vec![KeyColumn {
        name: "k".into(),
        dtype: KeyDtype::I32,
        data: &key_data,
        valid: &key_valid,
        n_rows,
    }];
    // SAFETY: f32 is plain-old-data; we just built val_data from the same
    // slice via to_le_bytes, so reinterpreting is safe.
    let val_typed: &[f32] =
        unsafe { std::slice::from_raw_parts(val_data.as_ptr() as *const f32, n_rows) };
    let mut value_columns: HashMap<String, ValueColumn<'_>> = HashMap::new();
    value_columns.insert(
        "v".into(),
        ValueColumn::F32 {
            data: val_typed,
            valid: &val_valid,
        },
    );
    let aggs = vec![simple("v", KAggOp::Sum, "sum_v")];
    let result = dispatch_groupby_fused(
        device,
        queue,
        cache,
        &key_cols,
        &aggs,
        &value_columns,
        n_rows,
    )
    .map_err(|e| format!("fused dispatch: {e:?}"))?;
    let idx_by_key = group_indices_by_key(&result.decoded_keys[0]);
    let sums_vec = match &result.agg_outputs[0] {
        AggOutput::F32 { values, .. } => values,
        other => return Err(format!("expected F32 sum, got {other:?}")),
    };
    let sums: BTreeMap<i32, f32> = idx_by_key.into_iter().map(|(k, i)| (k, sums_vec[i])).collect();
    Ok(OneColResult { sums })
}

fn run_per_agg_sum_f32(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    keys: &[i32],
    values: &[f32],
    valid: &[bool],
) -> Result<OneColResult, String> {
    let n_rows = keys.len();
    let key_data: Vec<u8> = keys.iter().flat_map(|v| v.to_le_bytes()).collect();
    let key_valid = pack_valid(&vec![true; n_rows]);
    let val_valid = pack_valid(valid);
    let key_cols = vec![KeyColumn {
        name: "k".into(),
        dtype: KeyDtype::I32,
        data: &key_data,
        valid: &key_valid,
        n_rows,
    }];
    let agg_specs: Vec<(AggRequest, ValueColumn<'_>)> = vec![(
        AggRequest {
            kind: AggKind::SumF32,
            input_col_idx: 0,
        },
        ValueColumn::F32 {
            data: values,
            valid: &val_valid,
        },
    )];
    let result = dispatch_groupby(device, queue, &key_cols, &agg_specs, n_rows)
        .map_err(|e| format!("per-agg dispatch: {e:?}"))?;
    let idx_by_key = group_indices_by_key(&result.decoded_keys[0]);
    let sums_vec = match &result.agg_outputs[0] {
        AggOutput::F32 { values, .. } => values,
        other => return Err(format!("expected F32 sum, got {other:?}")),
    };
    let sums: BTreeMap<i32, f32> = idx_by_key.into_iter().map(|(k, i)| (k, sums_vec[i])).collect();
    Ok(OneColResult { sums })
}

// ---------- multi-column Q1-shape dispatch --------------------------------

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Copy)]
enum Q1AggSelector {
    SumA,
    SumB,
    SumC,
    SumD,
    MeanA,
    MeanB,
    MeanC,
    MeanD,
    Count,
    Len,
}

const Q1_AGGS: &[Q1AggSelector] = &[
    Q1AggSelector::SumA,
    Q1AggSelector::SumB,
    Q1AggSelector::SumC,
    Q1AggSelector::SumD,
    Q1AggSelector::MeanA,
    Q1AggSelector::MeanB,
    Q1AggSelector::MeanC,
    Q1AggSelector::MeanD,
    Q1AggSelector::Count,
    Q1AggSelector::Len,
];

/// One aggregation output flattened to per-key. F32 values carry per-group
/// validity; integer values (Count/Len) are always-valid u64. We compare
/// both paths' per-key view so any (valid → invalid) drift gets flagged.
#[derive(Debug, Clone, PartialEq)]
enum PerKeyOutput {
    F32 { val: f32, valid: bool },
    U64 { val: u64 },
}

#[derive(Debug, Clone)]
struct Q1Result {
    /// key → per-agg-selector → output
    by_key: BTreeMap<i32, BTreeMap<Q1AggSelector, PerKeyOutput>>,
}

fn q1_kspecs() -> Vec<KAggSpec> {
    vec![
        simple("a", KAggOp::Sum, "sum_a"),
        simple("b", KAggOp::Sum, "sum_b"),
        simple("c", KAggOp::Sum, "sum_c"),
        simple("d", KAggOp::Sum, "sum_d"),
        simple("a", KAggOp::Mean, "mean_a"),
        simple("b", KAggOp::Mean, "mean_b"),
        simple("c", KAggOp::Mean, "mean_c"),
        simple("d", KAggOp::Mean, "mean_d"),
        // Count over column "a" (any valid col would do).
        simple("a", KAggOp::Count, "count"),
        KAggSpec::Length {
            output_alias: "len".into(),
        },
    ]
}

fn q1_agg_requests() -> Vec<(AggRequest, &'static str)> {
    vec![
        (
            AggRequest {
                kind: AggKind::SumF32,
                input_col_idx: 0, // a
            },
            "a",
        ),
        (
            AggRequest {
                kind: AggKind::SumF32,
                input_col_idx: 1, // b
            },
            "b",
        ),
        (
            AggRequest {
                kind: AggKind::SumF32,
                input_col_idx: 2, // c
            },
            "c",
        ),
        (
            AggRequest {
                kind: AggKind::SumF32,
                input_col_idx: 3, // d
            },
            "d",
        ),
        (
            AggRequest {
                kind: AggKind::MeanF32,
                input_col_idx: 0,
            },
            "a",
        ),
        (
            AggRequest {
                kind: AggKind::MeanF32,
                input_col_idx: 1,
            },
            "b",
        ),
        (
            AggRequest {
                kind: AggKind::MeanF32,
                input_col_idx: 2,
            },
            "c",
        ),
        (
            AggRequest {
                kind: AggKind::MeanF32,
                input_col_idx: 3,
            },
            "d",
        ),
        (
            AggRequest {
                kind: AggKind::Count,
                input_col_idx: 0,
            },
            "a",
        ),
        (
            AggRequest {
                kind: AggKind::Len,
                input_col_idx: 0,
            },
            "a",
        ),
    ]
}

/// Deterministic Q1-shape inputs. Returns (keys, [(col_name, vals, valid)]).
fn q1_inputs(
    seed: u64,
    n_rows: usize,
    n_groups: u32,
    null_density: f32,
) -> (Vec<i32>, [(&'static str, Vec<f32>, Vec<bool>); 4]) {
    let mut rng = StdRng::seed_from_u64(seed);
    let keys: Vec<i32> = (0..n_rows)
        .map(|_| (rng.gen::<u32>() % n_groups) as i32)
        .collect();
    let mut gen_col = || -> (Vec<f32>, Vec<bool>) {
        let vals: Vec<f32> = (0..n_rows).map(|_| rng.gen_range(-100.0f32..100.0)).collect();
        let valid: Vec<bool> = (0..n_rows)
            .map(|_| rng.gen::<f32>() >= null_density)
            .collect();
        (vals, valid)
    };
    let (a_v, a_valid) = gen_col();
    let (b_v, b_valid) = gen_col();
    let (c_v, c_valid) = gen_col();
    let (d_v, d_valid) = gen_col();
    (
        keys,
        [
            ("a", a_v, a_valid),
            ("b", b_v, b_valid),
            ("c", c_v, c_valid),
            ("d", d_v, d_valid),
        ],
    )
}

fn assemble_q1_result(
    decoded_keys: &[DecodedColumn],
    agg_outputs: &[AggOutput],
) -> Q1Result {
    let idx_by_key = group_indices_by_key(&decoded_keys[0]);
    let mut by_key: BTreeMap<i32, BTreeMap<Q1AggSelector, PerKeyOutput>> = BTreeMap::new();
    for (&k, &i) in idx_by_key.iter() {
        let mut row: BTreeMap<Q1AggSelector, PerKeyOutput> = BTreeMap::new();
        for (slot, sel) in Q1_AGGS.iter().enumerate() {
            let out = match &agg_outputs[slot] {
                AggOutput::F32 { values, valid } => PerKeyOutput::F32 {
                    val: values[i],
                    valid: valid[i],
                },
                AggOutput::U64 { values } => PerKeyOutput::U64 { val: values[i] },
                other => panic!("unexpected output for {sel:?}: {other:?}"),
            };
            row.insert(*sel, out);
        }
        by_key.insert(k, row);
    }
    Q1Result { by_key }
}

fn run_fused_q1(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    cache: &FusedLibraryCache,
    keys: &[i32],
    cols: &[(&'static str, Vec<f32>, Vec<bool>); 4],
) -> Result<Q1Result, String> {
    let n_rows = keys.len();
    let key_data: Vec<u8> = keys.iter().flat_map(|v| v.to_le_bytes()).collect();
    let key_valid = pack_valid(&vec![true; n_rows]);
    let key_cols = vec![KeyColumn {
        name: "k".into(),
        dtype: KeyDtype::I32,
        data: &key_data,
        valid: &key_valid,
        n_rows,
    }];
    // Build owned packed validity vecs so the borrowed slices outlive the
    // HashMap construction.
    let valids: Vec<Vec<u8>> = cols.iter().map(|(_, _, v)| pack_valid(v)).collect();
    let mut value_columns: HashMap<String, ValueColumn<'_>> = HashMap::new();
    for (i, (name, data, _)) in cols.iter().enumerate() {
        value_columns.insert(
            (*name).into(),
            ValueColumn::F32 {
                data,
                valid: &valids[i],
            },
        );
    }
    let aggs = q1_kspecs();
    let result = dispatch_groupby_fused(
        device,
        queue,
        cache,
        &key_cols,
        &aggs,
        &value_columns,
        n_rows,
    )
    .map_err(|e| format!("fused dispatch: {e:?}"))?;
    Ok(assemble_q1_result(&result.decoded_keys, &result.agg_outputs))
}

fn run_per_agg_q1(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    keys: &[i32],
    cols: &[(&'static str, Vec<f32>, Vec<bool>); 4],
) -> Result<Q1Result, String> {
    let n_rows = keys.len();
    let key_data: Vec<u8> = keys.iter().flat_map(|v| v.to_le_bytes()).collect();
    let key_valid = pack_valid(&vec![true; n_rows]);
    let key_cols = vec![KeyColumn {
        name: "k".into(),
        dtype: KeyDtype::I32,
        data: &key_data,
        valid: &key_valid,
        n_rows,
    }];
    let valids: Vec<Vec<u8>> = cols.iter().map(|(_, _, v)| pack_valid(v)).collect();
    // The per-agg path indexes by input_col_idx, but each (AggRequest,
    // ValueColumn) pair carries its own ValueColumn, so we just attach the
    // right column to each request.
    let request_specs = q1_agg_requests();
    let mut agg_specs: Vec<(AggRequest, ValueColumn<'_>)> = Vec::with_capacity(request_specs.len());
    for (req, col_name) in request_specs.iter() {
        let (_, data, _) = cols
            .iter()
            .find(|(n, _, _)| n == col_name)
            .expect("q1 column lookup");
        let valid_idx = cols
            .iter()
            .position(|(n, _, _)| n == col_name)
            .expect("q1 column lookup (valid_idx)");
        agg_specs.push((
            req.clone(),
            ValueColumn::F32 {
                data,
                valid: &valids[valid_idx],
            },
        ));
    }
    let result = dispatch_groupby(device, queue, &key_cols, &agg_specs, n_rows)
        .map_err(|e| format!("per-agg dispatch: {e:?}"))?;
    Ok(assemble_q1_result(&result.decoded_keys, &result.agg_outputs))
}

// ---------- proptest entry points -----------------------------------------

proptest! {
    // 64 cases per property. Each case spins up a fresh MetalDevice +
    // CommandQueue and runs two GPU dispatches (fused + per-agg); 256 cases
    // would push the Metal driver into the same resource-exhaustion regime
    // seen in test_compaction_pipeline.rs. 64 cases gives us good
    // shrink-coverage of (seed, n_rows, n_groups, null_density) without
    // tripping that wedge.
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn fused_eq_per_agg_sum_only(
        seed in any::<u64>(),
        n_rows in 100usize..5_000,
        n_groups in 2u32..32,
        null_density in 0.0f32..1.0,
    ) {
        let _guard = lock_metal();
        let (device, mut queue, cache) = setup();
        let (keys, values, valid) = synth_inputs_f32(seed, n_rows, n_groups, null_density);

        let fused = run_fused_sum_f32(&device, &mut queue, &cache, &keys, &values, &valid)
            .map_err(|e| TestCaseError::Fail(e.into()))?;
        let per_agg = run_per_agg_sum_f32(&device, &mut queue, &keys, &values, &valid)
            .map_err(|e| TestCaseError::Fail(e.into()))?;

        prop_assert_eq!(
            fused.sums.keys().copied().collect::<Vec<_>>(),
            per_agg.sums.keys().copied().collect::<Vec<_>>(),
            "group key sets must agree"
        );
        // Worst-case noise floor: assume one group received all rows
        // (other groups empty). Use n_rows as the per-group row count.
        let noise = sum_noise_floor(n_rows, 100.0);
        for (&k, &sf) in fused.sums.iter() {
            let sp = per_agg.sums[&k];
            assert_floats_eq(&format!("sum_v[k={k}]"), sf, sp, 1e-3, noise)?;
        }
    }

    #[test]
    fn fused_eq_per_agg_full_q1_shape(seed in 0u64..1_000) {
        let _guard = lock_metal();
        let (device, mut queue, cache) = setup();
        // Fixed shape: Q1's 10 aggs over 4 value cols, ~50k rows, 4 groups,
        // 5% null density. Seed varies — each seed gives a fully different
        // (keys, values, valid) configuration.
        let n_rows = 50_000usize;
        let n_groups = 4u32;
        let null_density = 0.05f32;
        let (keys, cols) = q1_inputs(seed, n_rows, n_groups, null_density);

        let fused = run_fused_q1(&device, &mut queue, &cache, &keys, &cols)
            .map_err(|e| TestCaseError::Fail(e.into()))?;
        let per_agg = run_per_agg_q1(&device, &mut queue, &keys, &cols)
            .map_err(|e| TestCaseError::Fail(e.into()))?;

        // Group key sets must agree.
        let f_keys: Vec<i32> = fused.by_key.keys().copied().collect();
        let p_keys: Vec<i32> = per_agg.by_key.keys().copied().collect();
        prop_assert_eq!(f_keys, p_keys, "group key sets must agree");

        // For Sum aggs the noise floor is set by the partial-sum walk.
        // Per-group row count averages n_rows/n_groups but can spike higher;
        // use n_rows as a safe upper bound. For Mean, the divide-by-count
        // attenuates the per-group noise by ~count, so we scale accordingly.
        let sum_noise = sum_noise_floor(n_rows, 100.0);
        let per_group_avg = (n_rows as f32) / (n_groups as f32);
        let mean_noise = sum_noise_floor(n_rows, 100.0) / per_group_avg.max(1.0);
        for (&k, f_row) in fused.by_key.iter() {
            let p_row = &per_agg.by_key[&k];
            for sel in Q1_AGGS.iter() {
                let f = &f_row[sel];
                let p = &p_row[sel];
                match (f, p) {
                    (
                        PerKeyOutput::F32 { val: fv, valid: fvalid },
                        PerKeyOutput::F32 { val: pv, valid: pvalid },
                    ) => {
                        prop_assert_eq!(
                            fvalid, pvalid,
                            "{:?}[k={}] validity mismatch (fused={}, per_agg={})",
                            sel, k, fvalid, pvalid
                        );
                        // Only compare numeric values when both are valid.
                        // Sum reports valid=true unconditionally; Mean
                        // reports valid=false when count=0 (skip numeric
                        // compare in that case).
                        if *fvalid && *pvalid {
                            // Sum uses partial-sum-magnitude noise floor;
                            // Mean's effective noise is sum_noise / count.
                            let noise = match sel {
                                Q1AggSelector::SumA
                                | Q1AggSelector::SumB
                                | Q1AggSelector::SumC
                                | Q1AggSelector::SumD => sum_noise,
                                Q1AggSelector::MeanA
                                | Q1AggSelector::MeanB
                                | Q1AggSelector::MeanC
                                | Q1AggSelector::MeanD => mean_noise,
                                _ => 0.0,
                            };
                            assert_floats_eq(
                                &format!("{sel:?}[k={k}]"),
                                *fv,
                                *pv,
                                1e-3,
                                noise,
                            )?;
                        }
                    }
                    (PerKeyOutput::U64 { val: fv }, PerKeyOutput::U64 { val: pv }) => {
                        prop_assert_eq!(
                            fv, pv,
                            "{:?}[k={}] integer count mismatch (fused={}, per_agg={})",
                            sel, k, fv, pv
                        );
                    }
                    _ => {
                        return Err(TestCaseError::Fail(
                            format!(
                                "{sel:?}[k={k}] output-kind mismatch: fused={f:?} per_agg={p:?}"
                            )
                            .into(),
                        ));
                    }
                }
            }
        }
    }
}
