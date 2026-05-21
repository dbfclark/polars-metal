// crates/polars-metal-kernels/benches/logical_bool.rs
//
// Criterion benchmark for `dispatch_bool_and` — the 3-valued AND
// kernel. Sweep (n, null_density) on the same grid as the cmp benches.
// Inputs are two bit-packed bool columns + their validity bitmaps;
// output is bit-packed data + validity. The AND/OR kernels share a
// dispatch path, so benching one is representative of both.
#![allow(clippy::expect_used)]

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::logical::dispatch_bool_and;

fn out_bytes(n: usize) -> usize {
    let raw = (n + 7) / 8;
    let padded = (raw + 3) & !3;
    padded.max(4)
}

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

/// Bit-packed bool data: a deterministic pattern across the range so the
/// kernel sees a mix of (true, true), (true, false), (false, true),
/// (false, false) row pairs.
fn make_data(n: usize, parity_shift: usize) -> Vec<u8> {
    let mut v = vec![0u8; packed_bytes(n)];
    for i in 0..n {
        if (i + parity_shift) % 3 != 0 {
            v[i >> 3] |= 1u8 << (i & 7);
        }
    }
    v
}

fn bench_bool_and(c: &mut Criterion) {
    let device = MetalDevice::system_default().expect("Metal-capable hardware");
    let mut group = c.benchmark_group("bool_and");
    group.sample_size(10);

    for &n in &[1_000usize, 100_000, 10_000_000] {
        let lhs_data = make_data(n, 0);
        let rhs_data = make_data(n, 1);

        for &null_density in &[0.0_f64, 0.5, 1.0] {
            let lhs_valid = make_validity(n, null_density);
            let rhs_valid = make_validity(n, null_density);
            let out_len = out_bytes(n);

            group.bench_with_input(
                BenchmarkId::new(format!("nulls={null_density}"), n),
                &n,
                |b, &_n| {
                    b.iter_batched(
                        || {
                            let queue = CommandQueue::new(&device).expect("queue creation");
                            let out_data = vec![0u8; out_len];
                            let out_valid = vec![0u8; out_len];
                            (queue, out_data, out_valid)
                        },
                        |(mut queue, mut out_data, mut out_valid)| {
                            dispatch_bool_and(
                                &device,
                                &mut queue,
                                &lhs_data,
                                &lhs_valid,
                                &rhs_data,
                                &rhs_valid,
                                n,
                                &mut out_data,
                                &mut out_valid,
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

criterion_group!(benches, bench_bool_and);
criterion_main!(benches);
