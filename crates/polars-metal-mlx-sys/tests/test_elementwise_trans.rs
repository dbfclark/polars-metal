//! Transcendental + rounding + cast bindings.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use polars_metal_mlx_sys::array::*;
use polars_metal_mlx_sys::elementwise::*;

const TOL: f32 = 1e-5;
const TOL_TRANS: f32 = 1e-4; // looser for log/exp roundtrip

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol || ((a - b).abs() / b.abs().max(1.0) < tol)
}

#[test]
fn sin_cos_at_canonical_angles() {
    use std::f32::consts::PI;
    let a = mlx_array_from_f32_slice(&[0.0, PI / 6.0, PI / 4.0, PI / 3.0, PI / 2.0]).unwrap();
    let s = mlx_sin(&a).unwrap();
    let c = mlx_cos(&a).unwrap();
    mlx_array_eval(&[s.clone(), c.clone()]).unwrap();
    let sv = mlx_array_to_f32_vec(&s).unwrap();
    let cv = mlx_array_to_f32_vec(&c).unwrap();
    assert!(close(sv[0], 0.0, TOL));
    assert!(close(sv[4], 1.0, TOL));
    assert!(close(cv[0], 1.0, TOL));
    assert!(close(cv[4], 0.0, TOL));
}

#[test]
fn tan_tanh_sinh_cosh_basic() {
    use std::f32::consts::PI;
    let a = mlx_array_from_f32_slice(&[0.0, PI / 4.0]).unwrap();
    let t = mlx_tan(&a).unwrap();
    mlx_array_eval(&[t.clone()]).unwrap();
    let tv = mlx_array_to_f32_vec(&t).unwrap();
    assert!(close(tv[0], 0.0, TOL));
    assert!(close(tv[1], 1.0, TOL));

    let b = mlx_array_from_f32_slice(&[0.0, 1.0]).unwrap();
    let sh = mlx_sinh(&b).unwrap();
    let ch = mlx_cosh(&b).unwrap();
    let th = mlx_tanh(&b).unwrap();
    mlx_array_eval(&[sh.clone(), ch.clone(), th.clone()]).unwrap();
    let shv = mlx_array_to_f32_vec(&sh).unwrap();
    let chv = mlx_array_to_f32_vec(&ch).unwrap();
    let thv = mlx_array_to_f32_vec(&th).unwrap();
    // sinh(0) = 0, cosh(0) = 1, tanh(0) = 0
    assert!(close(shv[0], 0.0, TOL));
    assert!(close(chv[0], 1.0, TOL));
    assert!(close(thv[0], 0.0, TOL));
    // tanh(1) = 0.7616
    assert!(close(thv[1], 0.761_594_2_f32, TOL_TRANS));
}

#[test]
fn asin_acos_atan_basic() {
    use std::f32::consts::PI;
    let a = mlx_array_from_f32_slice(&[0.0, 1.0, -1.0]).unwrap();
    let as_ = mlx_asin(&a).unwrap();
    let ac = mlx_acos(&a).unwrap();
    let at = mlx_atan(&a).unwrap();
    mlx_array_eval(&[as_.clone(), ac.clone(), at.clone()]).unwrap();
    let asv = mlx_array_to_f32_vec(&as_).unwrap();
    let acv = mlx_array_to_f32_vec(&ac).unwrap();
    let atv = mlx_array_to_f32_vec(&at).unwrap();
    // asin(0) = 0, asin(1) = pi/2
    assert!(close(asv[0], 0.0, TOL));
    assert!(close(asv[1], PI / 2.0, TOL));
    // acos(1) = 0, acos(0) = pi/2
    assert!(close(acv[0], PI / 2.0, TOL));
    assert!(close(acv[1], 0.0, TOL));
    // atan(0) = 0, atan(1) = pi/4
    assert!(close(atv[0], 0.0, TOL));
    assert!(close(atv[1], PI / 4.0, TOL));
}

#[test]
fn log_exp_round_trip() {
    let a = mlx_array_from_f32_slice(&[1.0, 2.0, 5.0, 10.0, 100.0]).unwrap();
    let log_a = mlx_log(&a).unwrap();
    let exp_log_a = mlx_exp(&log_a).unwrap();
    mlx_array_eval(&[exp_log_a.clone()]).unwrap();
    let out = mlx_array_to_f32_vec(&exp_log_a).unwrap();
    for (a, b) in out.iter().zip([1.0, 2.0, 5.0, 10.0, 100.0].iter()) {
        assert!(
            close(*a, *b, TOL_TRANS),
            "exp(log({})) = {}, expected {}",
            b,
            a,
            b
        );
    }
}

#[test]
fn log2_log10_log1p_basic() {
    let a = mlx_array_from_f32_slice(&[1.0, 2.0, 10.0]).unwrap();
    let l2 = mlx_log2(&a).unwrap();
    let l10 = mlx_log10(&a).unwrap();
    mlx_array_eval(&[l2.clone(), l10.clone()]).unwrap();
    let l2v = mlx_array_to_f32_vec(&l2).unwrap();
    let l10v = mlx_array_to_f32_vec(&l10).unwrap();
    assert!(close(l2v[0], 0.0, TOL));
    assert!(close(l2v[1], 1.0, TOL));
    assert!(close(l10v[2], 1.0, TOL));

    // log1p(x) = log(1 + x): log1p(0) = 0, log1p(e-1) ~ 1
    let b = mlx_array_from_f32_slice(&[0.0, std::f32::consts::E - 1.0]).unwrap();
    let lp = mlx_log1p(&b).unwrap();
    mlx_array_eval(&[lp.clone()]).unwrap();
    let lpv = mlx_array_to_f32_vec(&lp).unwrap();
    assert!(close(lpv[0], 0.0, TOL));
    assert!(close(lpv[1], 1.0, TOL_TRANS));
}

#[test]
fn sqrt_cbrt() {
    let a = mlx_array_from_f32_slice(&[1.0, 4.0, 9.0, 16.0]).unwrap();
    let s = mlx_sqrt(&a).unwrap();
    mlx_array_eval(&[s.clone()]).unwrap();
    assert_eq!(mlx_array_to_f32_vec(&s).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

    // cbrt: implemented as pow(x, 1/3)
    let b = mlx_array_from_f32_slice(&[1.0, 8.0, 27.0, 64.0]).unwrap();
    let c = mlx_cbrt(&b).unwrap();
    mlx_array_eval(&[c.clone()]).unwrap();
    let cv = mlx_array_to_f32_vec(&c).unwrap();
    assert!(close(cv[0], 1.0, TOL));
    assert!(close(cv[1], 2.0, TOL));
    assert!(close(cv[2], 3.0, TOL));
    assert!(close(cv[3], 4.0, TOL));
}

#[test]
fn exp2_basic() {
    // exp2(x) = 2^x: implemented as exp(x * ln(2))
    let a = mlx_array_from_f32_slice(&[0.0, 1.0, 2.0, 3.0, 10.0]).unwrap();
    let r = mlx_exp2(&a).unwrap();
    mlx_array_eval(&[r.clone()]).unwrap();
    let rv = mlx_array_to_f32_vec(&r).unwrap();
    assert!(close(rv[0], 1.0, TOL));
    assert!(close(rv[1], 2.0, TOL));
    assert!(close(rv[2], 4.0, TOL));
    assert!(close(rv[3], 8.0, TOL));
    assert!(close(rv[4], 1024.0, TOL_TRANS));
}

#[test]
fn floor_ceil_round() {
    let a = mlx_array_from_f32_slice(&[1.5, -1.5, 2.7, -2.7]).unwrap();
    let f = mlx_floor(&a).unwrap();
    let c = mlx_ceil(&a).unwrap();
    let r = mlx_round(&a).unwrap();
    mlx_array_eval(&[f.clone(), c.clone(), r.clone()]).unwrap();
    assert_eq!(
        mlx_array_to_f32_vec(&f).unwrap(),
        vec![1.0, -2.0, 2.0, -3.0]
    );
    assert_eq!(
        mlx_array_to_f32_vec(&c).unwrap(),
        vec![2.0, -1.0, 3.0, -2.0]
    );
    let rv = mlx_array_to_f32_vec(&r).unwrap();
    // MLX round: half-to-even (banker's rounding). 1.5 -> 2, -1.5 -> -2,
    // 2.7 -> 3, -2.7 -> -3.
    assert_eq!(rv, vec![2.0, -2.0, 3.0, -3.0]);
}

#[test]
fn atan2_basic() {
    use std::f32::consts::PI;
    let y = mlx_array_from_f32_slice(&[0.0, 1.0, 0.0, -1.0]).unwrap();
    let x = mlx_array_from_f32_slice(&[1.0, 0.0, -1.0, 0.0]).unwrap();
    let r = mlx_atan2(&y, &x).unwrap();
    mlx_array_eval(&[r.clone()]).unwrap();
    let rv = mlx_array_to_f32_vec(&r).unwrap();
    assert!(close(rv[0], 0.0, TOL));
    assert!(close(rv[1], PI / 2.0, TOL));
    assert!(close(rv[2], PI, TOL));
    assert!(close(rv[3], -PI / 2.0, TOL));
}

#[test]
fn cast_f32_to_i32() {
    let a = mlx_array_from_f32_slice(&[1.7, -1.7, 0.5, 100.0]).unwrap();
    let i = mlx_cast(&a, MlxDtype::I32).unwrap();
    mlx_array_eval(&[i.clone()]).unwrap();
    // We can't read back i32 directly (no mlx_array_to_i32_vec yet);
    // round-trip cast back to F32 and verify.
    let f = mlx_cast(&i, MlxDtype::F32).unwrap();
    mlx_array_eval(&[f.clone()]).unwrap();
    // Truncation toward zero: 1.7->1, -1.7->-1, 0.5->0, 100.0->100
    assert_eq!(
        mlx_array_to_f32_vec(&f).unwrap(),
        vec![1.0, -1.0, 0.0, 100.0]
    );
}
