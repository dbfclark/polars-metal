// crates/polars-metal-mlx-sys/src/lib.rs

// Allow unwrap() in tests; forbid it in production code.
#![cfg_attr(not(test), forbid(clippy::unwrap_used))]

//! FFI bindings to MLX C++.
//!
//! In M0 this crate exposes one trivial operation (`add_one` for the cxx
//! smoke test) and `add_f32` (real MLX elementwise add) to validate the FFI
//! strategy end to end. Real op coverage arrives in later milestones.

mod error;
pub use error::FfiError;

#[cxx::bridge(namespace = "polars_metal_mlx")]
mod ffi {
    unsafe extern "C++" {
        include!("polars-metal-mlx-sys/cxx/mlx_bridge.h");

        fn add_one(x: i64) -> i64;
        fn add_f32(a: &CxxVector<f32>, b: &CxxVector<f32>) -> Result<UniquePtr<CxxVector<f32>>>;
        fn add_f32_on_gpu(
            a: &CxxVector<f32>,
            b: &CxxVector<f32>,
        ) -> Result<UniquePtr<CxxVector<f32>>>;
    }
}

pub fn add_one(x: i64) -> i64 {
    ffi::add_one(x)
}

/// Elementwise add forced to the Metal GPU device.
///
/// Uses MLX's `StreamContext` RAII to switch the default device to
/// `Device::gpu` for the duration of the call. If Metal is unavailable on
/// this host (e.g. the linked `libmlx.a` was built without
/// `-DMLX_BUILD_METAL=ON`), MLX throws `std::invalid_argument` and this
/// function returns `Err`. Callers can use this to gate GPU-only code paths.
pub fn add_f32_on_gpu(a: &[f32], b: &[f32]) -> Result<Vec<f32>, FfiError> {
    if a.len() != b.len() {
        return Err(FfiError::ShapeMismatch {
            lhs: a.len(),
            rhs: b.len(),
        });
    }
    let mut va = cxx::CxxVector::new();
    let mut vb = cxx::CxxVector::new();
    for &x in a {
        va.pin_mut().push(x);
    }
    for &x in b {
        vb.pin_mut().push(x);
    }
    let result = ffi::add_f32_on_gpu(&va, &vb).map_err(FfiError::from)?;
    Ok(result.iter().copied().collect())
}

pub fn add_f32(a: &[f32], b: &[f32]) -> Result<Vec<f32>, FfiError> {
    if a.len() != b.len() {
        return Err(FfiError::ShapeMismatch {
            lhs: a.len(),
            rhs: b.len(),
        });
    }
    let mut va = cxx::CxxVector::new();
    let mut vb = cxx::CxxVector::new();
    for &x in a {
        va.pin_mut().push(x);
    }
    for &x in b {
        vb.pin_mut().push(x);
    }
    let result = ffi::add_f32(&va, &vb).map_err(FfiError::from)?;
    Ok(result.iter().copied().collect())
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn add_one_smoke() {
        assert_eq!(add_one(41), 42);
    }

    #[test]
    fn add_f32_on_gpu_dispatches_to_metal() {
        // 4096 elements: large enough that MLX would route to GPU even without
        // explicit forcing. We force explicitly via StreamContext so that a
        // broken Metal toolchain produces a hard error rather than a silent
        // CPU fallback.
        let len = 4096;
        let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..len).map(|i| i as f32 * -0.25).collect();
        let result = add_f32_on_gpu(&a, &b).expect("Metal GPU dispatch should work on this host");
        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        assert_eq!(result.len(), expected.len());
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-4, "GPU result {r} != CPU expected {e}");
        }
    }

    proptest! {
        #[test]
        fn add_f32_matches_rust(
            len in 0usize..256,
            seed_a: u64,
            seed_b: u64,
        ) {
            let a: Vec<f32> = (0..len)
                .map(|i| ((seed_a.wrapping_add(i as u64) & 0xffff) as f32) / 1024.0)
                .collect();
            let b: Vec<f32> = (0..len)
                .map(|i| ((seed_b.wrapping_add(i as u64) & 0xffff) as f32) / 1024.0)
                .collect();

            let result = add_f32(&a, &b).expect("add_f32 must succeed for same-length inputs");
            let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

            prop_assert_eq!(result.len(), expected.len());
            for (r, e) in result.iter().zip(expected.iter()) {
                prop_assert!((r - e).abs() < 1e-5);
            }
        }
    }
}
