#![allow(clippy::expect_used)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::groupby_build_sort::gpu::sort_u128;
use proptest::prelude::*;

#[test]
fn empty_input_yields_empty_output() {
    let device = MetalDevice::system_default().expect("metal device");
    let (k, i) = sort_u128(&device, &[]).expect("sort");
    assert!(k.is_empty());
    assert!(i.is_empty());
}

#[test]
fn single_element_returns_self() {
    let device = MetalDevice::system_default().expect("metal device");
    let (k, i) = sort_u128(&device, &[42u128]).expect("sort");
    assert_eq!(k, vec![42u128]);
    assert_eq!(i, vec![0u32]);
}

#[test]
fn small_distinct_sort() {
    let device = MetalDevice::system_default().expect("metal device");
    let keys: Vec<u128> = vec![300, 100, 200, 500, 400];
    let (sk, si) = sort_u128(&device, &keys).expect("sort");
    assert_eq!(sk, vec![100, 200, 300, 400, 500]);
    assert_eq!(si, vec![1, 2, 0, 4, 3]);
}

#[test]
fn duplicates_preserve_input_order_by_idx() {
    let device = MetalDevice::system_default().expect("metal device");
    // 10 copies of each of 3 distinct keys, in interleaved order.
    let keys: Vec<u128> = (0..30u128).map(|i| (i % 3) * 100).collect();
    let (sk, si) = sort_u128(&device, &keys).expect("sort");
    // 0,0,0,...,100,100,100,...,200,200,200,...
    for w in sk.windows(2) {
        assert!(w[0] <= w[1]);
    }
    // si is a permutation of 0..30.
    let mut perm = si.clone();
    perm.sort_unstable();
    assert_eq!(perm, (0u32..30).collect::<Vec<_>>());
    // Each output key consistent with input via sorted_idx.
    for i in 0..sk.len() {
        assert_eq!(sk[i], keys[si[i] as usize]);
    }
}

// Reduced budget: 16 cases × 256 max-len = 16 × 16 × 2 × ~256 dispatches.
// If this works cleanly, scale up later. Keep an eye on GPU error 00000206.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn full_sort_matches_cpu_sort(
        keys in proptest::collection::vec(any::<u128>(), 1..256usize),
    ) {
        let device = MetalDevice::system_default().expect("metal device");
        let (sorted_keys, sorted_idx) = sort_u128(&device, &keys).expect("sort");

        // 1. Sorted ascending.
        for w in sorted_keys.windows(2) {
            prop_assert!(w[0] <= w[1]);
        }
        // 2. sorted_idx is a permutation of 0..n.
        let mut perm = sorted_idx.clone();
        perm.sort_unstable();
        prop_assert_eq!(&perm, &(0u32..keys.len() as u32).collect::<Vec<_>>());
        // 3. sorted_keys[i] == keys[sorted_idx[i]].
        for i in 0..keys.len() {
            prop_assert_eq!(sorted_keys[i], keys[sorted_idx[i] as usize]);
        }
        // 4. Stability: for two equal sorted_keys, sorted_idx is increasing.
        for i in 1..sorted_keys.len() {
            if sorted_keys[i] == sorted_keys[i - 1] {
                prop_assert!(sorted_idx[i - 1] < sorted_idx[i]);
            }
        }
    }
}
