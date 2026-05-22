// crates/polars-metal-kernels/tests/test_groupby_hash.rs
//
// Tests for `dispatch_hash` (GPU) and `hash_u128_reference` (Rust), and
// a proptest asserting they produce identical output for all random inputs.
#![allow(clippy::expect_used, clippy::panic)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::groupby::{dispatch_hash, hash_u128_reference};
use proptest::prelude::*;

/// Run the GPU kernel on `encoded` and return the resulting hash vector.
fn run_kernel(encoded: &[u128]) -> Vec<u32> {
    let device = MetalDevice::system_default().expect("Metal-capable hardware required");
    let mut queue = CommandQueue::new(&device).expect("command queue");
    let mut out = vec![0u32; encoded.len()];
    dispatch_hash(&device, &mut queue, encoded, encoded.len(), &mut out)
        .expect("dispatch_hash should succeed");
    out
}

#[test]
fn single_row_hashes_and_is_deterministic() {
    let encoded = vec![0x1234_5678_9abc_def0u128];
    let out1 = run_kernel(&encoded);
    let out2 = run_kernel(&encoded);
    assert_eq!(out1.len(), 1);
    assert_eq!(out1, out2, "same input must yield same hash on two runs");
}

#[test]
fn equal_keys_produce_equal_hashes() {
    let encoded = vec![42u128, 99u128, 42u128, 99u128];
    let out = run_kernel(&encoded);
    assert_eq!(out[0], out[2], "row 0 and row 2 hold the same key");
    assert_eq!(out[1], out[3], "row 1 and row 3 hold the same key");
}

#[test]
fn kernel_matches_reference_implementation() {
    let encoded: Vec<u128> = (0..1024u128).map(|i| i * 1_000_003).collect();
    let kernel_out = run_kernel(&encoded);
    let ref_out: Vec<u32> = encoded
        .iter()
        .map(|&k| {
            let lo = k as u64;
            let hi = (k >> 64) as u64;
            hash_u128_reference(lo, hi)
        })
        .collect();
    assert_eq!(
        kernel_out, ref_out,
        "GPU kernel output must match Rust reference for all 1024 rows"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn kernel_equals_reference_for_random_keys(
        ks in prop::collection::vec(any::<u128>(), 1..=512)
    ) {
        let kernel_out = run_kernel(&ks);
        let ref_out: Vec<u32> = ks
            .iter()
            .map(|&k| {
                let lo = k as u64;
                let hi = (k >> 64) as u64;
                hash_u128_reference(lo, hi)
            })
            .collect();
        prop_assert_eq!(kernel_out, ref_out);
    }
}
