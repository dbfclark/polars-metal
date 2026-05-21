// crates/polars-metal-kernels/benches/filter_scatter.rs
//
// Criterion benchmark for `dispatch_scatter_i64` — pass 3 of the filter
// compaction pipeline. The i64 case stands in for the kernel family
// (f64 and bool variants share the same algorithm and dispatch shape).
//
// We synthesise the keep mask and CPU-compute the inclusive prefix sum
// directly — bypassing MLX cumsum — so the bench isolates the scatter
// kernel's wall-time, which is the point of a per-kernel bench. The
// "null_density" axis here means the validity bitmap of the SOURCE
// column; selectivity (fraction of rows kept) is held fixed at 0.5
// (alternating parity) per the task plan, which does not ask us to
// vary it.
#![allow(clippy::expect_used)]

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::filter::dispatch_scatter_i64;

/// Minimum bytes for the scatter dispatcher's `dst_valid` (mirrors
/// `dst_valid_min_bytes`: `ceil(n_out/8)` padded to 4 bytes, minimum 4).
fn dst_valid_bytes(n_out: usize) -> usize {
    let raw = (n_out + 7) / 8;
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

/// Dense keep mask: keep even rows, drop odd rows. Selectivity ~0.5;
/// the task plan does not ask us to vary selectivity in this bench.
fn make_keep(n: usize) -> Vec<u8> {
    (0..n).map(|i| if i % 2 == 0 { 1u8 } else { 0 }).collect()
}

/// Inclusive prefix sum on the keep mask. Mirrors what MLX cumsum
/// produces in the real pipeline.
fn prefix_sum_inclusive(keep: &[u8]) -> Vec<u32> {
    let mut prefix = Vec::with_capacity(keep.len());
    let mut acc: u32 = 0;
    for &k in keep {
        acc += k as u32;
        prefix.push(acc);
    }
    prefix
}

fn bench_filter_scatter_i64(c: &mut Criterion) {
    let device = MetalDevice::system_default().expect("Metal-capable hardware");
    let mut group = c.benchmark_group("filter_scatter_i64");
    group.sample_size(10);

    for &n in &[1_000usize, 100_000, 10_000_000] {
        let src: Vec<i64> = (0..n).map(|i| i as i64).collect();
        let keep = make_keep(n);
        let prefix = prefix_sum_inclusive(&keep);
        let n_out = *prefix.last().copied().get_or_insert(0) as usize;
        let dst_valid_len = dst_valid_bytes(n_out);

        for &null_density in &[0.0_f64, 0.5, 1.0] {
            let src_valid = make_validity(n, null_density);

            group.bench_with_input(
                BenchmarkId::new(format!("nulls={null_density}"), n),
                &n,
                |b, &_n| {
                    b.iter_batched(
                        || {
                            let queue = CommandQueue::new(&device).expect("queue creation");
                            // dst_data: n_out + 1 slots (the extra slot
                            // is the sentinel overrun guard).
                            let dst_data = vec![0i64; n_out + 1];
                            let dst_valid = vec![0u8; dst_valid_len];
                            (queue, dst_data, dst_valid)
                        },
                        |(mut queue, mut dst_data, mut dst_valid)| {
                            dispatch_scatter_i64(
                                &device,
                                &mut queue,
                                &src,
                                &src_valid,
                                &keep,
                                &prefix,
                                n_out,
                                &mut dst_data,
                                &mut dst_valid,
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

criterion_group!(benches, bench_filter_scatter_i64);
criterion_main!(benches);
