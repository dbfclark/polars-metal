// crates/polars-metal-mlx-sys/src/lib.rs

//! FFI bindings to MLX C++.
//!
//! In M0 this crate exposes one trivial operation (`add_one` for the cxx
//! smoke test) and `mlx_add_f32` (real MLX) to validate the FFI strategy
//! end to end. Real op coverage arrives in later milestones.

#[cxx::bridge(namespace = "polars_metal_mlx")]
mod ffi {
    unsafe extern "C++" {
        include!("polars-metal-mlx-sys/cxx/hello.h");

        fn add_one(x: i64) -> i64;
    }
}

pub fn add_one(x: i64) -> i64 {
    ffi::add_one(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn add_one_matches_rust(x in i64::MIN..i64::MAX) {
            prop_assume!(x != i64::MAX);
            prop_assert_eq!(add_one(x), x + 1);
        }
    }
}
