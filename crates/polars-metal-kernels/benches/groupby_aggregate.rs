// crates/polars-metal-kernels/benches/groupby_aggregate.rs
//
// Criterion microbench for the 32-bit GPU aggregation dispatchers.
// Sweeps over (kernel, n_rows, null_density). 64-bit aggregation uses
// the CPU-finalize path (see groupby.rs::aggregate_*_cpu) and is timed
// implicitly via the Q1 bench in tests/bench/test_tpch_q1.py.

#![allow(clippy::expect_used)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::groupby::{
    dispatch_count_u32, dispatch_len_u32, dispatch_sum_f32, dispatch_sum_i32,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn make_row_to_group(n: usize, n_groups: u32) -> Vec<u32> {
    let mut rng = StdRng::seed_from_u64(0xC0FFEE);
    (0..n).map(|_| rng.gen_range(0..n_groups)).collect()
}

fn make_valid(n: usize, null_density: f64) -> Vec<u8> {
    let mut rng = StdRng::seed_from_u64(0xD00D);
    let mut v = vec![0u8; ((n + 7) / 8 + 3) & !3];
    for i in 0..n {
        if rng.gen::<f64>() > null_density {
            v[i >> 3] |= 1 << (i & 7);
        }
    }
    v
}

fn bench_aggregate(c: &mut Criterion) {
    let device = MetalDevice::system_default().expect("device");
    let mut queue = CommandQueue::new(&device).expect("queue");

    let n_groups: u32 = 100;
    let mut group = c.benchmark_group("groupby_aggregate");
    group.sample_size(10);

    for &n in &[100_000usize, 1_000_000, 10_000_000] {
        group.throughput(Throughput::Elements(n as u64));
        let row_to_group = make_row_to_group(n, n_groups);

        let i32_vals: Vec<i32> = (0..n).map(|i| (i % 10_000) as i32).collect();
        for &nd in &[0.0f64, 0.5, 1.0] {
            let valid = make_valid(n, nd);
            let mut out = vec![0i32; n_groups as usize];
            group.bench_with_input(
                BenchmarkId::new(format!("sum_i32_nulls={nd:.1}"), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        dispatch_sum_i32(
                            &device,
                            &mut queue,
                            black_box(&i32_vals),
                            &valid,
                            &row_to_group,
                            n,
                            n_groups as usize,
                            &mut out,
                        )
                        .expect("dispatch_sum_i32");
                    });
                },
            );
        }

        let f32_vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let valid = make_valid(n, 0.5);
        let mut out_f = vec![0.0f32; n_groups as usize];
        group.bench_with_input(BenchmarkId::new("sum_f32_nulls=0.5", n), &n, |b, _| {
            b.iter(|| {
                dispatch_sum_f32(
                    &device,
                    &mut queue,
                    black_box(&f32_vals),
                    &valid,
                    &row_to_group,
                    n,
                    n_groups as usize,
                    &mut out_f,
                )
                .expect("dispatch_sum_f32");
            });
        });

        let mut out_c = vec![0u32; n_groups as usize];
        group.bench_with_input(BenchmarkId::new("count_u32_nulls=0.5", n), &n, |b, _| {
            b.iter(|| {
                dispatch_count_u32(
                    &device,
                    &mut queue,
                    &valid,
                    &row_to_group,
                    n,
                    n_groups as usize,
                    &mut out_c,
                )
                .expect("dispatch_count_u32");
            });
        });

        let mut out_l = vec![0u32; n_groups as usize];
        group.bench_with_input(BenchmarkId::new("len_u32", n), &n, |b, _| {
            b.iter(|| {
                dispatch_len_u32(
                    &device,
                    &mut queue,
                    &row_to_group,
                    n,
                    n_groups as usize,
                    &mut out_l,
                )
                .expect("dispatch_len_u32");
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_aggregate);
criterion_main!(benches);
