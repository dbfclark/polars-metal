// crates/polars-metal-kernels/benches/cmp_f64.rs
//
// Criterion benchmark for `dispatch_cmp_f64` with `CompareOp::Lt`.
//
// Mirrors `cmp_i64.rs` in structure. We use ONLY finite floats — no NaN
// — because the kernel has a known TotalOrd vs IEEE-NaN gap (see
// docs/open-questions.md); benchmarking pathological inputs is
// misleading. The (n, null_density) grid matches the i64 bench.
#![allow(clippy::expect_used)]

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::cmp::{dispatch_cmp_f64, CompareOp};
use polars_metal_kernels::command::CommandQueue;

fn out_bytes(n: usize) -> usize {
    let raw = (n + 7) / 8;
    let padded = (raw + 3) & !3;
    padded.max(4)
}

fn valid_bytes(n: usize) -> usize {
    ((n + 7) / 8).max(1)
}

fn make_validity(n: usize, density: f64) -> Vec<u8> {
    let mut v = vec![0u8; valid_bytes(n)];
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

fn bench_cmp_f64(c: &mut Criterion) {
    let device = MetalDevice::system_default().expect("Metal-capable hardware");
    let mut group = c.benchmark_group("cmp_f64_lt");
    group.sample_size(10);

    for &n in &[1_000usize, 100_000, 10_000_000] {
        // Finite floats only. `i` is mapped through a deterministic
        // formula that stays well inside the f64 finite range and is
        // dense across both sides of zero so `<` exercises both signs.
        let lhs: Vec<f64> = (0..n).map(|i| (i as f64) * 1.5 - 1024.0).collect();
        let rhs: Vec<f64> = (0..n).map(|i| (i as f64) * 1.25 - 1024.0).collect();

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
                            dispatch_cmp_f64(
                                &device,
                                &mut queue,
                                &lhs,
                                &lhs_valid,
                                &rhs,
                                &rhs_valid,
                                n,
                                CompareOp::Lt,
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

criterion_group!(benches, bench_cmp_f64);
criterion_main!(benches);
