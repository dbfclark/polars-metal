// crates/polars-metal-kernels/benches/groupby_hash.rs
//
// Criterion microbench for the standalone hash kernel. Sizes 100K, 1M,
// 10M. Inputs are pre-encoded random u128 values; we don't measure the
// encoder here (pure CPU, effectively free).

#![allow(clippy::expect_used)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::groupby::dispatch_hash;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn make_keys(n: usize, seed: u64) -> Vec<u128> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.gen::<u128>()).collect()
}

fn bench_hash(c: &mut Criterion) {
    let device = MetalDevice::system_default().expect("device");
    let mut queue = CommandQueue::new(&device).expect("queue");

    let mut group = c.benchmark_group("groupby_hash");
    group.sample_size(10);
    for &n in &[100_000usize, 1_000_000, 10_000_000] {
        group.throughput(Throughput::Elements(n as u64));
        let keys = make_keys(n, 0xC0FFEE);
        let mut out = vec![0u32; n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                dispatch_hash(&device, &mut queue, black_box(&keys), n, &mut out)
                    .expect("dispatch_hash");
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_hash);
criterion_main!(benches);
