// crates/polars-metal-kernels/benches/filter_predicate.rs
//
// Criterion benchmark for `dispatch_predicate_to_u8` — pass 1 of the
// filter compaction pipeline. Reads one bit-packed predicate + its
// validity bitmap, writes a dense `u8[n]` keep mask. The "null_density"
// axis here means the validity-bitmap density (predicate rows null vs
// valid).
#![allow(clippy::expect_used)]

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::filter::dispatch_predicate_to_u8;

fn packed_bytes(n: usize) -> usize {
    ((n + 7) / 8).max(1)
}

fn make_validity(n: usize, density: f64) -> Vec<u8> {
    let mut v = vec![0u8; packed_bytes(n)];
    if density >= 1.0 {
        for byte in v.iter_mut() {
            *byte = 0xFF;
        }
        let trailing = n & 7;
        if trailing != 0 {
            let last = (n + 7) / 8 - 1;
            v[last] = (1u8 << trailing) - 1;
        }
    } else if density > 0.0 {
        for i in 0..n {
            if i % 2 == 0 {
                v[i >> 3] |= 1u8 << (i & 7);
            }
        }
    }
    v
}

/// Bit-packed predicate: deterministic pattern so roughly half the rows
/// are true and half are false. Mirrors the cmp/logical inputs in
/// spirit.
fn make_predicate(n: usize) -> Vec<u8> {
    let mut v = vec![0u8; packed_bytes(n)];
    for i in 0..n {
        if i % 2 == 0 {
            v[i >> 3] |= 1u8 << (i & 7);
        }
    }
    v
}

fn bench_filter_predicate(c: &mut Criterion) {
    let device = MetalDevice::system_default().expect("Metal-capable hardware");
    let mut group = c.benchmark_group("filter_predicate");
    group.sample_size(10);

    for &n in &[1_000usize, 100_000, 10_000_000] {
        let pred_data = make_predicate(n);

        for &null_density in &[0.0_f64, 0.5, 1.0] {
            let pred_valid = make_validity(n, null_density);

            group.bench_with_input(
                BenchmarkId::new(format!("nulls={null_density}"), n),
                &n,
                |b, &n| {
                    b.iter_batched(
                        || {
                            let queue = CommandQueue::new(&device).expect("queue creation");
                            // Output is exactly `n` bytes (one keep flag
                            // per row); fresh per iteration.
                            let out = vec![0u8; n];
                            (queue, out)
                        },
                        |(mut queue, mut out)| {
                            dispatch_predicate_to_u8(
                                &device,
                                &mut queue,
                                &pred_data,
                                &pred_valid,
                                n,
                                &mut out,
                            )
                            .expect("dispatch succeeds");
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_filter_predicate);
criterion_main!(benches);
