//! M4 Phase 1 Task 11: smoke test for the full FFI surface.
//!
//! Builds a Black-Scholes-shaped graph (log/exp/sqrt/tanh chain) using the
//! Phase 1 bindings end to end. The goal is structural: verify that a
//! multi-op chain of MLX operations composed via the Rust binding surface
//! produces a finite scalar result. This is the kind of graph the Phase 4
//! MlxSubgraph builder will emit from a Polars expression tree.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use polars_metal_mlx_sys::array::*;
use polars_metal_mlx_sys::elementwise::*;
use polars_metal_mlx_sys::reduce::*;

#[test]
fn black_scholes_shape_via_bindings() {
    // Synthetic option-pricing inputs.
    let n = 1000;
    let spot: Vec<f32> = (0..n).map(|i| 80.0 + (i as f32 * 0.07)).collect();
    let strike: Vec<f32> = (0..n).map(|i| 100.0 - (i as f32 * 0.03)).collect();
    let ttm: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32 * 0.001)).collect();

    let sigma_val: f32 = 0.2;
    let r_val: f32 = 0.05;

    let s = mlx_array_from_f32_slice(&spot).unwrap();
    let k = mlx_array_from_f32_slice(&strike).unwrap();
    let t = mlx_array_from_f32_slice(&ttm).unwrap();
    let sig = mlx_array_from_f32_slice(&vec![sigma_val; n]).unwrap();
    let r_arr = mlx_array_from_f32_slice(&vec![r_val; n]).unwrap();
    let two = mlx_array_from_f32_slice(&vec![2.0_f32; n]).unwrap();
    let half = mlx_array_from_f32_slice(&vec![0.5_f32; n]).unwrap();
    let one = mlx_array_from_f32_slice(&vec![1.0_f32; n]).unwrap();
    let coef = mlx_array_from_f32_slice(&vec![0.797_884_5_f32; n]).unwrap();

    // d1 = (log(s/k) + (r + sigma^2/2)*t) / (sigma * sqrt(t))
    let sk = mlx_div(&s, &k).unwrap();
    let log_sk = mlx_log(&sk).unwrap();
    let sigma2 = mlx_mul(&sig, &sig).unwrap();
    let half_sigma2 = mlx_div(&sigma2, &two).unwrap();
    let r_plus = mlx_add(&r_arr, &half_sigma2).unwrap();
    let r_plus_t = mlx_mul(&r_plus, &t).unwrap();
    let num = mlx_add(&log_sk, &r_plus_t).unwrap();
    let sqrt_t = mlx_sqrt(&t).unwrap();
    let denom = mlx_mul(&sig, &sqrt_t).unwrap();
    let d1 = mlx_div(&num, &denom).unwrap();

    // CDF approximation: 0.5 * (1 + tanh(0.7978845608 * d1))
    let scaled = mlx_mul(&coef, &d1).unwrap();
    let tanh_s = mlx_tanh(&scaled).unwrap();
    let cdf_d1_inner = mlx_add(&one, &tanh_s).unwrap();
    let cdf_d1 = mlx_mul(&half, &cdf_d1_inner).unwrap();

    // Approximate call price: s * cdf(d1)
    let price = mlx_mul(&s, &cdf_d1).unwrap();
    let total = mlx_sum(&price).unwrap();

    mlx_array_eval(&[total.clone()]).unwrap();

    let v = mlx_array_to_f32_vec(&total).unwrap();
    assert_eq!(v.len(), 1);
    assert!(v[0].is_finite(), "total = {} (expected finite)", v[0]);
    // Sanity bound: prices are positive and bounded by sum(spot) ~= 115k
    assert!(v[0] > 0.0 && v[0] < 200_000.0, "total = {}", v[0]);
}
