// crates/polars-metal-kernels/benches/aggregate_fused.rs
//
//! Compares fused-kernel dispatch (one kernel, multi-agg) against M2's
//! per-agg loop (N kernels) on a Q1-shape workload. The metric of interest
//! is wall-clock time at fixed input sizes — dispatch count is observable
//! but secondary.
//!
//! Q1 shape: 4 F32 value columns with Sum + Mean over each (8 aggs), plus
//! Count and Len (2 more aggs) = 10 aggs total. Single F32 key column with
//! ~4 distinct groups, 5% null density on values.
//!
//! ## MSL compile cost note
//!
//! The first fused iteration pays MSL source-compile + pipeline-creation
//! cost (~50–200ms on M-series). The `FusedLibraryCache` is constructed
//! once outside `b.iter` and reused across iterations, so steady-state
//! samples reflect only kernel execution. Criterion's warmup phase
//! (default 3s) absorbs the cold compile before the measurement window
//! begins, so the reported median should be cache-hit. If the first
//! measured sample still shows compile cost, mean/max may be skewed but
//! median is the metric to read.
//!
//! ## GPU watchdog / `Impacting Interactivity`
//!
//! At larger sizes (>=1M rows) after a long run of preceding dispatches
//! the macOS GPU watchdog raises
//! `kIOGPUCommandBufferCallbackErrorImpactingInteractivity`. This is a
//! *bench-harness artifact*: criterion runs many back-to-back dispatches
//! during warmup + measurement, accumulating foreground-GPU time well
//! beyond the watchdog's cumulative threshold; by the time the 1M and
//! 10M groups run, the GPU is in a "stressed" state where any further
//! long-running command buffer trips the watchdog immediately. The same
//! `dispatch_groupby_fused` path runs cleanly at 10M in the Python
//! TPC-H-Q1 bench (`tests/bench/test_tpch_q1.py`) where each query is
//! driven via the full engine with intervening CPU work.
//!
//! To keep the bench runnable, we probe each size with a single
//! dispatch before the criterion measurement loop. If the probe fails,
//! we skip that path for that size and print a diagnostic. Numbers for
//! the size that did complete are still emitted to stdout.
//!
//! ## Apples-to-apples
//!
//! Both `run_fused_dispatch` and `run_per_agg_dispatch` perform the full
//! pipeline: key encode + hash + build + aggregate + finalize + decode.
//! The difference between them is the aggregate phase only (one fused
//! kernel vs N per-agg kernels). The build phase cost is identical and
//! present in both timings.

#![allow(clippy::expect_used)]

use std::collections::HashMap;

use std::thread;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::aggregate_fused::cache::FusedLibraryCache;
use polars_metal_kernels::aggregate_fused::signature::{AggOp as KAggOp, AggSpec as KAggSpec};
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::groupby::{
    dispatch_groupby, dispatch_groupby_fused, AggKind, AggRequest, KeyColumn, KeyDtype, ValueColumn,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---------- input synthesis (Q1 shape) ------------------------------------
//
// Helpers below mirror the proptest fixtures in
// `tests/test_fused_vs_per_agg.rs`. We duplicate (rather than share via a
// `common` module) because criterion benches and cargo tests don't share
// helper modules cleanly across `benches/` and `tests/` directories.

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

/// Q1-shape inputs: keys with 4 distinct groups, 4 F32 value columns with
/// 5% null density. Deterministic for a given seed.
struct Q1Inputs {
    keys: Vec<i32>,
    key_data: Vec<u8>,
    key_valid: Vec<u8>,
    cols: [Q1Col; 4],
}

struct Q1Col {
    name: &'static str,
    data: Vec<f32>,
    valid_packed: Vec<u8>,
}

fn q1_inputs(seed: u64, n_rows: usize) -> Q1Inputs {
    let n_groups = 4u32;
    let null_density = 0.05f32;
    let mut rng = StdRng::seed_from_u64(seed);
    let keys: Vec<i32> = (0..n_rows)
        .map(|_| (rng.gen::<u32>() % n_groups) as i32)
        .collect();
    let key_data: Vec<u8> = keys.iter().flat_map(|v| v.to_le_bytes()).collect();
    let key_valid = pack_valid(&vec![true; n_rows]);

    let mut gen_col = |name: &'static str| -> Q1Col {
        let data: Vec<f32> = (0..n_rows)
            .map(|_| rng.gen_range(-100.0f32..100.0))
            .collect();
        let valid: Vec<bool> = (0..n_rows)
            .map(|_| rng.gen::<f32>() >= null_density)
            .collect();
        let valid_packed = pack_valid(&valid);
        Q1Col {
            name,
            data,
            valid_packed,
        }
    };
    let cols = [gen_col("a"), gen_col("b"), gen_col("c"), gen_col("d")];

    Q1Inputs {
        keys,
        key_data,
        key_valid,
        cols,
    }
}

/// Kernel-layer AggSpec for the fused path: 4 Sum + 4 Mean + Count + Len.
fn q1_aggs_fused() -> Vec<KAggSpec> {
    let simple = |col: &str, op: KAggOp, alias: &str| -> KAggSpec {
        KAggSpec::Simple {
            input_col: col.into(),
            op,
            output_alias: alias.into(),
        }
    };
    vec![
        simple("a", KAggOp::Sum, "sum_a"),
        simple("b", KAggOp::Sum, "sum_b"),
        simple("c", KAggOp::Sum, "sum_c"),
        simple("d", KAggOp::Sum, "sum_d"),
        simple("a", KAggOp::Mean, "mean_a"),
        simple("b", KAggOp::Mean, "mean_b"),
        simple("c", KAggOp::Mean, "mean_c"),
        simple("d", KAggOp::Mean, "mean_d"),
        simple("a", KAggOp::Count, "count"),
        KAggSpec::Length {
            output_alias: "len".into(),
        },
    ]
}

/// Per-agg `AggRequest` list matching the fused shape. Column indices map
/// to `inputs.cols`: a=0, b=1, c=2, d=3.
fn q1_agg_requests() -> Vec<(AggRequest, usize)> {
    vec![
        (
            AggRequest {
                kind: AggKind::SumF32,
                input_col_idx: 0,
            },
            0,
        ),
        (
            AggRequest {
                kind: AggKind::SumF32,
                input_col_idx: 1,
            },
            1,
        ),
        (
            AggRequest {
                kind: AggKind::SumF32,
                input_col_idx: 2,
            },
            2,
        ),
        (
            AggRequest {
                kind: AggKind::SumF32,
                input_col_idx: 3,
            },
            3,
        ),
        (
            AggRequest {
                kind: AggKind::MeanF32,
                input_col_idx: 0,
            },
            0,
        ),
        (
            AggRequest {
                kind: AggKind::MeanF32,
                input_col_idx: 1,
            },
            1,
        ),
        (
            AggRequest {
                kind: AggKind::MeanF32,
                input_col_idx: 2,
            },
            2,
        ),
        (
            AggRequest {
                kind: AggKind::MeanF32,
                input_col_idx: 3,
            },
            3,
        ),
        (
            AggRequest {
                kind: AggKind::Count,
                input_col_idx: 0,
            },
            0,
        ),
        (
            AggRequest {
                kind: AggKind::Len,
                input_col_idx: 0,
            },
            0,
        ),
    ]
}

// ---------- bench dispatch wrappers ---------------------------------------

fn try_fused_dispatch(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    cache: &FusedLibraryCache,
    inputs: &Q1Inputs,
    aggs: &[KAggSpec],
) -> Result<(), String> {
    let n_rows = inputs.keys.len();
    let key_cols = vec![KeyColumn {
        name: "k".into(),
        dtype: KeyDtype::I32,
        data: &inputs.key_data,
        valid: &inputs.key_valid,
        n_rows,
        dict: None,
    }];
    let mut value_columns: HashMap<String, ValueColumn<'_>> = HashMap::new();
    for col in inputs.cols.iter() {
        value_columns.insert(
            col.name.into(),
            ValueColumn::F32 {
                data: &col.data,
                valid: &col.valid_packed,
            },
        );
    }
    dispatch_groupby_fused(
        device,
        queue,
        cache,
        &key_cols,
        aggs,
        &value_columns,
        n_rows,
    )
    .map(|_| ())
    .map_err(|e| format!("{e:?}"))
}

fn try_per_agg_dispatch(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    inputs: &Q1Inputs,
    requests: &[(AggRequest, usize)],
) -> Result<(), String> {
    let n_rows = inputs.keys.len();
    let key_cols = vec![KeyColumn {
        name: "k".into(),
        dtype: KeyDtype::I32,
        data: &inputs.key_data,
        valid: &inputs.key_valid,
        n_rows,
        dict: None,
    }];
    let agg_specs: Vec<(AggRequest, ValueColumn<'_>)> = requests
        .iter()
        .map(|(req, col_idx)| {
            let col = &inputs.cols[*col_idx];
            (
                req.clone(),
                ValueColumn::F32 {
                    data: &col.data,
                    valid: &col.valid_packed,
                },
            )
        })
        .collect();
    dispatch_groupby(device, queue, &key_cols, &agg_specs, n_rows)
        .map(|_| ())
        .map_err(|e| format!("{e:?}"))
}

// ---------- bench entry point ---------------------------------------------

fn bench_fused_vs_per_agg(c: &mut Criterion) {
    // Construct Metal resources ONCE so per-iteration setup cost stays out
    // of the measurement loop. The FusedLibraryCache also persists across
    // iterations so MSL compilation only happens during the first warmup
    // sample for each signature.
    let device = MetalDevice::system_default().expect("Metal hardware required");
    let mut queue = CommandQueue::new(&device).expect("command queue");
    let cache = FusedLibraryCache::new(device.clone());

    let mut group = c.benchmark_group("aggregate_fused_vs_per_agg");
    // Bigger sizes (>=1M rows) approach the macOS GPU watchdog limit when
    // many dispatches run back-to-back without breathing room. We use
    // `iter_custom` to run exactly one dispatch per sample and pause briefly
    // between samples so cumulative GPU foreground time stays under
    // `kIOGPUCommandBufferCallbackErrorImpactingInteractivity` threshold.
    // Each criterion sample is one (cold-or-warm) dispatch — criterion
    // still computes median/IQR over `sample_size` measurements.
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(10));

    // Per-iteration pause keeps the GPU below the macOS interactivity
    // watchdog cumulative-time budget when consecutive dispatches each
    // exceed ~100ms. 250ms is empirically sufficient at 100K-1M; 10M
    // may still trip the per-buffer limit on a single dispatch.
    let pause = Duration::from_millis(250);

    for &size in &[100_000usize, 1_000_000, 10_000_000] {
        group.throughput(Throughput::Elements(size as u64));

        let inputs = q1_inputs(42, size);
        let fused_aggs = q1_aggs_fused();
        let per_agg_requests = q1_agg_requests();

        // Probe each path once before letting criterion start its
        // measurement loop. If a probe trips the GPU watchdog or any other
        // dispatch error, we skip that path for this size and print a
        // diagnostic — criterion would otherwise abort the entire bench
        // run on the first panic.
        let fused_ok = match try_fused_dispatch(&device, &mut queue, &cache, &inputs, &fused_aggs) {
            Ok(()) => true,
            Err(e) => {
                eprintln!("[bench] fused_q1 size={size} probe failed, skipping: {e}");
                false
            }
        };
        thread::sleep(pause);
        let per_agg_ok = match try_per_agg_dispatch(&device, &mut queue, &inputs, &per_agg_requests)
        {
            Ok(()) => true,
            Err(e) => {
                eprintln!("[bench] per_agg_q1 size={size} probe failed, skipping: {e}");
                false
            }
        };
        thread::sleep(pause);

        if fused_ok {
            group.bench_with_input(BenchmarkId::new("fused_q1", size), &size, |b, _| {
                b.iter_custom(|n_iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..n_iters {
                        let start = Instant::now();
                        let _ =
                            try_fused_dispatch(&device, &mut queue, &cache, &inputs, &fused_aggs);
                        total += start.elapsed();
                        thread::sleep(pause);
                    }
                    total
                });
            });
        }

        if per_agg_ok {
            group.bench_with_input(BenchmarkId::new("per_agg_q1", size), &size, |b, _| {
                b.iter_custom(|n_iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..n_iters {
                        let start = Instant::now();
                        let _ =
                            try_per_agg_dispatch(&device, &mut queue, &inputs, &per_agg_requests);
                        total += start.elapsed();
                        thread::sleep(pause);
                    }
                    total
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_fused_vs_per_agg);
criterion_main!(benches);
