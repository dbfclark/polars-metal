// crates/polars-metal-kernels/benches/groupby_build_partitioned.rs
#![allow(clippy::expect_used)]
#![allow(clippy::panic)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::groupby_build_partitioned::gpu::partition_and_build;
use polars_metal_kernels::groupby_build_partitioned::PartitionedBuildError;

fn bench_a1(c: &mut Criterion) {
    let device = MetalDevice::system_default().expect("metal device");
    let mut group = c.benchmark_group("groupby_build_partitioned");
    for &n_rows in &[100_000usize, 1_000_000, 10_000_000] {
        for &n_groups in &[4u32, 1024, 16_384] {
            let keys: Vec<u128> = (0..n_rows)
                .map(|i| (i % n_groups as usize) as u128)
                .collect();
            // Smoke once: if overflow at this cardinality, skip the bench row
            // instead of crashing the harness.
            match partition_and_build(&device, &keys, 16) {
                Ok(_) => {}
                Err(PartitionedBuildError::Overflow) => {
                    eprintln!("skipping rows={n_rows} groups={n_groups}: A1 overflow");
                    continue;
                }
                Err(e) => panic!("dispatch failed: {e}"),
            }
            group.bench_with_input(
                BenchmarkId::new(format!("rows{n_rows}_groups{n_groups}"), n_rows),
                &keys,
                |b, keys| b.iter(|| partition_and_build(&device, keys, 16).expect("dispatch")),
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_a1);
criterion_main!(benches);
