// crates/polars-metal-buffer/benches/placeholder.rs
//
// Placeholder benchmark. Real microbenches arrive when M1 lands the filter kernel.

use criterion::{criterion_group, criterion_main, Criterion};

fn placeholder(c: &mut Criterion) {
    c.bench_function("placeholder/noop", |b| b.iter(|| 42_usize));
}

criterion_group!(benches, placeholder);
criterion_main!(benches);
