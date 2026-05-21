// crates/polars-metal-kernels/benches/cmp_i64.rs
//
// Criterion benchmark for `dispatch_cmp_i64` with `CompareOp::Lt`.
//
// The M1 bar for this bench is "runs without errors" — numbers vary per
// machine. We sweep three row counts (1K, 100K, 10M) crossed with three
// validity densities (all-valid, half-null, all-null). Inputs are
// deterministic (no PRNG dep); each iteration receives fresh output
// buffers via `iter_batched` so the kernel's atomic-OR semantics never
// observe stale bits across runs.
#![allow(clippy::expect_used)]

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::cmp::{dispatch_cmp_i64, CompareOp};
use polars_metal_kernels::command::CommandQueue;

/// Minimum bytes for a bit-packed output bitmap (matches the dispatcher's
/// `out_min_bytes`: `ceil(n/8)` padded to 4 bytes, minimum 4).
fn out_bytes(n: usize) -> usize {
    let raw = (n + 7) / 8;
    let padded = (raw + 3) & !3;
    padded.max(4)
}

/// Bytes for a bit-packed validity bitmap of `n` rows. Matches the
/// dispatcher's `ceil(n / 8)` minimum; we round up so callers can pass
/// any over-large buffer without tripping the length check.
fn valid_bytes(n: usize) -> usize {
    ((n + 7) / 8).max(1)
}

/// Build a validity bitmap with approximately `density * n` bits set.
/// `density` of 0.0 produces an all-null bitmap; 1.0 produces an
/// all-valid bitmap; 0.5 alternates by row index parity. Deterministic;
/// the benchmark just needs sample data shape, not statistical realism.
fn make_validity(n: usize, density: f64) -> Vec<u8> {
    let mut v = vec![0u8; valid_bytes(n)];
    if density >= 1.0 {
        for byte in v.iter_mut() {
            *byte = 0xFF;
        }
        // Trim any trailing bits beyond `n` for tidiness; the kernel
        // ignores them but a clean bitmap is nicer to debug.
        let trailing = n & 7;
        if trailing != 0 {
            let last = (n + 7) / 8 - 1;
            v[last] = (1u8 << trailing) - 1;
        }
    } else if density > 0.0 {
        // 0 < density < 1 → alternate: even rows valid, odd rows null.
        // Density argument is only used to label the bench, not to tune
        // selectivity precisely.
        for i in 0..n {
            if i % 2 == 0 {
                v[i >> 3] |= 1u8 << (i & 7);
            }
        }
    }
    v
}

fn bench_cmp_i64(c: &mut Criterion) {
    let device = MetalDevice::system_default().expect("Metal-capable hardware");
    let mut group = c.benchmark_group("cmp_i64_lt");
    // Tame the 10M case: full statistical convergence on that pass would
    // dominate wall-time without changing the "runs without errors"
    // signal. Criterion still produces stable median estimates at low
    // sample counts.
    group.sample_size(10);

    for &n in &[1_000usize, 100_000, 10_000_000] {
        // Deterministic, reproducible inputs. Half the rows satisfy
        // `lhs < rhs` and half do not, so we exercise both branches of
        // the comparison kernel.
        let lhs: Vec<i64> = (0..n).map(|i| i as i64).collect();
        let rhs: Vec<i64> = (0..n).map(|i| (i as i64) ^ 0x5555_5555).collect();

        for &null_density in &[0.0_f64, 0.5, 1.0] {
            let lhs_valid = make_validity(n, null_density);
            let rhs_valid = make_validity(n, null_density);
            let out_len = out_bytes(n);

            group.bench_with_input(
                BenchmarkId::new(format!("nulls={null_density}"), n),
                &n,
                |b, &_n| {
                    b.iter_batched(
                        // Setup (untimed): fresh queue + zeroed output
                        // buffers per iteration. The kernel's atomic OR
                        // never clears bits, so re-using outputs across
                        // iterations would accumulate stale state.
                        || {
                            let queue = CommandQueue::new(&device).expect("queue creation");
                            let out_data = vec![0u8; out_len];
                            let out_valid = vec![0u8; out_len];
                            (queue, out_data, out_valid)
                        },
                        // Routine (timed): one dispatch call.
                        |(mut queue, mut out_data, mut out_valid)| {
                            dispatch_cmp_i64(
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

criterion_group!(benches, bench_cmp_i64);
criterion_main!(benches);
