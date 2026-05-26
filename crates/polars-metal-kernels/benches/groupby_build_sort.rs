// crates/polars-metal-kernels/benches/groupby_build_sort.rs
#![allow(clippy::expect_used, clippy::panic)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::groupby_build_partitioned::gpu::partition_and_build;
use polars_metal_kernels::groupby_build_partitioned::PartitionedBuildError;
use polars_metal_kernels::groupby_build_sort::gpu::sort_and_segment;

fn bench_build_modes(c: &mut Criterion) {
    let device = MetalDevice::system_default().expect("metal device");
    let mut group = c.benchmark_group("groupby_build_a1_vs_a2");
    for &n_rows in &[1_000_000usize, 10_000_000] {
        for &n_groups in &[1024u32, 65_536, 1_048_576] {
            let keys: Vec<u128> = (0..n_rows)
                .map(|i| (i % n_groups as usize) as u128)
                .collect();

            // A1: only benches when TGSM fits.
            match partition_and_build(&device, &keys, 16) {
                Ok(_) => {
                    group.bench_with_input(
                        BenchmarkId::new(format!("a1_rows{n_rows}_groups{n_groups}"), n_rows),
                        &keys,
                        |b, keys| {
                            b.iter(|| partition_and_build(&device, keys, 16).expect("dispatch"))
                        },
                    );
                }
                Err(PartitionedBuildError::Overflow) => {
                    eprintln!(
                        "skipping A1 rows={n_rows} groups={n_groups}: TGSM overflow (expected)"
                    );
                }
                Err(e) => panic!("A1 dispatch failed: {e}"),
            }

            // A2: runs for every cardinality.
            group.bench_with_input(
                BenchmarkId::new(format!("a2_rows{n_rows}_groups{n_groups}"), n_rows),
                &keys,
                |b, keys| b.iter(|| sort_and_segment(&device, keys).expect("dispatch")),
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_build_modes);
criterion_main!(benches);
