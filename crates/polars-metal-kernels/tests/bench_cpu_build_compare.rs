//! Manual perf comparison: A1 GPU build vs A2 GPU build vs M2's CPU HashMap.
//! Run with: cargo test -p polars-metal-kernels --test bench_cpu_build_compare \
//!     --release -- --nocapture --test-threads=1
//! This is a one-shot perf-data collector; not run by `make test`.

#![allow(
    clippy::expect_used,
    clippy::print_stdout,
    clippy::unwrap_used,
    clippy::panic
)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::groupby::dispatch_build;
use polars_metal_kernels::groupby_build_partitioned::gpu::{
    partition_and_build, partition_and_build_with_scratch,
};
use polars_metal_kernels::groupby_build_partitioned::BuildScratch;
use polars_metal_kernels::groupby_build_partitioned::PartitionedBuildError;
use polars_metal_kernels::groupby_build_sort::gpu::sort_and_segment;
use std::time::Instant;

fn time_fn<F: FnMut()>(mut f: F, iters: u32) -> f64 {
    // Warm-up.
    f();
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

#[test]
#[ignore = "perf data collector, run manually"]
fn print_build_comparison_table() {
    let device = MetalDevice::system_default().expect("metal device");
    println!();
    println!("=== Build-phase perf comparison (median ms / call) ===");
    println!(
        "{:>10} {:>10} {:>11} {:>11} {:>11} {:>11} {:>11}",
        "n_rows", "n_groups", "A1 fresh", "A1 scratch", "A2 (GPU)", "CPU Hash", "A1s/CPU"
    );

    // Persistent scratch reused across all entries (capacity grows with max).
    let mut scratch = BuildScratch::new(&device).expect("scratch");

    for &(n_rows, n_groups, iters) in &[
        (100_000usize, 4u32, 50),
        (100_000, 1024, 50),
        (100_000, 16_384, 50),
        (1_000_000, 4, 20),
        (1_000_000, 1024, 20),
        (1_000_000, 16_384, 20),
        (1_000_000, 65_536, 20),
        (10_000_000, 4, 5),
        (10_000_000, 1024, 5),
        (10_000_000, 65_536, 5),
    ] {
        let keys: Vec<u128> = (0..n_rows)
            .map(|i| (i % n_groups as usize) as u128)
            .collect();

        // A1 — fresh scratch per call (worst case).
        let a1_fresh_ms = match partition_and_build(&device, &keys, 16) {
            Ok(_) => time_fn(
                || {
                    partition_and_build(&device, &keys, 16).expect("a1");
                },
                iters,
            ),
            Err(PartitionedBuildError::Overflow) => f64::NAN,
            Err(e) => panic!("a1 dispatch err: {e}"),
        };

        // A1 — persistent scratch (production target).
        let a1_scratch_ms = if a1_fresh_ms.is_nan() {
            f64::NAN
        } else {
            time_fn(
                || {
                    partition_and_build_with_scratch(&device, &mut scratch, &keys, 16)
                        .expect("a1 scratch");
                },
                iters,
            )
        };

        // A2 — only run if n_rows × cost is reasonable (skip if it would take > 30s).
        let a2_ms = if n_rows >= 10_000_000 && iters >= 3 {
            // 10M × 1 iter ≈ 2s; do 3 iters minimum.
            time_fn(
                || {
                    sort_and_segment(&device, &keys).expect("a2");
                },
                3,
            )
        } else {
            time_fn(
                || {
                    sort_and_segment(&device, &keys).expect("a2");
                },
                iters.min(5),
            )
        };

        // CPU HashMap (M2's existing build).
        let mut queue = CommandQueue::new(&device).expect("queue");
        let hashes: Vec<u32> = vec![];
        let cpu_ms = time_fn(
            || {
                dispatch_build(&device, &mut queue, &keys, &hashes, n_rows)
                    .expect("cpu dispatch_build");
            },
            iters,
        );

        let ratio = if a1_scratch_ms.is_nan() {
            f64::NAN
        } else {
            a1_scratch_ms / cpu_ms
        };

        println!(
            "{:>10} {:>10} {:>11.2} {:>11.2} {:>11.2} {:>11.2} {:>11.2}",
            n_rows, n_groups, a1_fresh_ms, a1_scratch_ms, a2_ms, cpu_ms, ratio
        );
    }
    println!();
}
