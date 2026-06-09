//! M6 A3: hand-rolled MSL FFT — planner + dispatcher. See
//! docs/superpowers/specs/2026-06-09-m6-a3-msl-fft-design.md.

use crate::command::{CommandQueue, DispatchError};
use crate::shader_lib::{shared_library, ShaderError};
use polars_metal_buffer::{BufferError, MetalBuffer, MetalDevice};

#[derive(Debug, thiserror::Error)]
pub enum FftError {
    #[error("shader library: {0}")]
    Shader(#[from] ShaderError),
    #[error("dispatch: {0}")]
    Dispatch(#[from] DispatchError),
    #[error("buffer: {0}")]
    Buffer(#[from] BufferError),
    #[error("unsupported fft size {0}")]
    Unsupported(i64),
}

/// Cap on the base radix-2 Stockham kernel — matches `FFT_BASE_MAX` in
/// `shaders/fft.metal`. 1024 complex points × 8 B × 2 ping-pong threadgroup
/// buffers = 16 KB, safely under Apple Silicon's 32 KB threadgroup-memory
/// limit. Sizes above this route through the four-step / Bluestein paths
/// (later tasks); for now they return [`FftError::Unsupported`].
pub const FFT_BASE_MAX: i64 = 1024;

/// Decompose `n` into an ordered list of radices, each in `{2,3,4,5,6,7,8}`,
/// whose product is `n`. Returns `None` if any prime factor `> 7` remains
/// (such sizes route to Bluestein in a later task) or for `n <= 0`.
///
/// True powers of two are handled by the dedicated radix-2 path in `fft_gpu`
/// *before* `factorize` is called; `factorize` is used only for composite
/// (non-pow2) `n`. Greedy largest-first over `{8,7,6,5,4,3,2}` is a perf
/// preference for the radix factors it emits — any valid factorization
/// (product == n, every factor <= 8) is correct for the mixed-radix kernel;
/// the differential test confirms.
pub fn factorize(n: i64) -> Option<Vec<u32>> {
    if n <= 0 {
        return None;
    }
    if n == 1 {
        return Some(vec![1]);
    }
    let mut rem = n;
    let mut out = Vec::new();
    for &r in &[8i64, 7, 6, 5, 4, 3, 2] {
        while rem % r == 0 {
            out.push(r as u32);
            rem /= r;
        }
    }
    if rem != 1 {
        // leftover prime factor > 7 (e.g. 11, 13) — not representable here.
        return None;
    }
    Some(out)
}

/// Compute a 1-D FFT over an interleaved-complex host slice `input`
/// (`[re,im,...]`, length `2*n`). Returns the transform interleaved, length
/// `2*n`. `inverse` applies 1/N scaling.
///
/// Phase 0 (this task): a single threadgroup cooperatively transforms one
/// length-`n` pow2 signal via iterative Stockham radix-2 in threadgroup
/// memory. Only `n` that is a power of two with `n <= FFT_BASE_MAX` is
/// supported; everything else returns [`FftError::Unsupported`].
pub fn fft_gpu(
    device: &MetalDevice,
    input: &[f32],
    n: i64,
    inverse: bool,
) -> Result<Vec<f32>, FftError> {
    debug_assert_eq!(input.len() as i64, 2 * n);
    if n <= 0 {
        return Err(FftError::Unsupported(n));
    }
    if n > FFT_BASE_MAX {
        // Larger sizes route to the four-step path (this task) when pow2 and
        // both factors fit; otherwise unsupported until later tasks.
        if (n & (n - 1)) == 0 {
            return fft_fourstep(device, input, n, inverse);
        }
        return Err(FftError::Unsupported(n));
    }

    let in_buf = MetalBuffer::from_f32_slice(device, input)?;
    let out_buf = device.new_buffer_zeroed((2 * n as usize) * std::mem::size_of::<f32>())?;
    let n_buf = device.new_buffer_from_bytes(&(n as u32).to_le_bytes())?;
    let inv_buf = device.new_buffer_from_bytes(&u32::from(inverse).to_le_bytes())?;

    let lib = shared_library(device)?;
    let mut queue = CommandQueue::new(device)?;
    // Single-threadgroup invariant: all stages run in per-threadgroup memory,
    // so a second threadgroup would transform uninitialized data.
    // dispatch_1d_with_tg sets grid_width = tg_width = tg, and n <= FFT_BASE_MAX
    // (1024) <= the PSO's maxTotalThreadsPerThreadgroup on Apple Silicon, so the
    // width clamp never splits into multiple threadgroups. The kernel strides by
    // tg_size to cover all n points.
    let tg = (n as usize).min(FFT_BASE_MAX as usize);

    if (n & (n - 1)) == 0 {
        // Power-of-two: keep the proven radix-2 Stockham path.
        let pso = lib.pipeline("fft_stockham_pow2_f32")?;
        queue.dispatch_1d_with_tg(&pso, &[&in_buf, &out_buf, &n_buf, &inv_buf], tg, tg)?;
    } else if let Some(radices) = factorize(n) {
        // Composite smooth n: mixed-radix (3..8) Stockham. Pass the per-stage
        // radix list as a u32 buffer plus its length scalar.
        let mut radix_bytes = Vec::with_capacity(radices.len() * 4);
        for r in &radices {
            radix_bytes.extend_from_slice(&r.to_le_bytes());
        }
        let radices_buf = device.new_buffer_from_bytes(&radix_bytes)?;
        let n_radices_buf = device.new_buffer_from_bytes(&(radices.len() as u32).to_le_bytes())?;
        let pso = lib.pipeline("fft_mixed_radix_f32")?;
        queue.dispatch_1d_with_tg(
            &pso,
            &[
                &in_buf,
                &out_buf,
                &n_buf,
                &inv_buf,
                &radices_buf,
                &n_radices_buf,
            ],
            tg,
            tg,
        )?;
    } else {
        // leftover prime factor > 7 → Bluestein (Task 6).
        return Err(FftError::Unsupported(n));
    }

    queue.wait_until_complete()?;
    Ok(out_buf.to_f32_vec())
}

/// Bailey's four-step FFT for power-of-two `n` with `FFT_BASE_MAX < n <=
/// FFT_BASE_MAX^2` (= 2^20). Factor `n = 2^p` as `n1 = 2^(p/2)`,
/// `n2 = 2^(p - p/2)`, so both factors are `<= FFT_BASE_MAX` for `p <= 20`.
///
/// View the input as an `n1 x n2` row-major matrix and run:
/// (1) length-`n1` column FFTs, (2) the `W_N^{i*j}` cross-twiddle,
/// (3) length-`n2` row FFTs, (4) an `n1 x n2 -> n2 x n1` transpose. The
/// sub-FFTs are forward-only; inverse is handled by conjugating the input
/// host-side, running the forward four-step, then conjugating and scaling by
/// `1/N` on readback (`ifft(x) = conj(fft(conj(x)))/N`).
///
/// The four passes are data-dependent, so we `wait_until_complete` after each
/// dispatch before issuing the next — both for correctness (later passes read
/// the prior pass's output) and so per-pass GPU errors surface (the queue only
/// tracks the most-recently-committed command buffer; see `command.rs`).
fn fft_fourstep(
    device: &MetalDevice,
    input: &[f32],
    n: i64,
    inverse: bool,
) -> Result<Vec<f32>, FftError> {
    // p = log2(n); split p into p1 = p/2, p2 = p - p1.
    let p = (n as u64).trailing_zeros();
    let p1 = p / 2;
    let p2 = p - p1;
    let n1 = 1u32 << p1;
    let n2 = 1u32 << p2;
    // n1*n2 == n by construction (p1 + p2 == p); assert it to document the
    // invariant the column/row/transpose passes rely on.
    debug_assert_eq!((n1 as i64) * (n2 as i64), n);
    if i64::from(n1) > FFT_BASE_MAX || i64::from(n2) > FFT_BASE_MAX {
        // p > 20: a factor exceeds the base cap; recursive path is Task 5.
        return Err(FftError::Unsupported(n));
    }
    let ntot = n as usize;

    // Stage the (optionally conjugated) input into the data buffer.
    let mut staged = input.to_vec();
    if inverse {
        for c in staged.chunks_exact_mut(2) {
            c[1] = -c[1];
        }
    }
    let data_buf = MetalBuffer::from_f32_slice(device, &staged)?;
    let trans_buf = device.new_buffer_zeroed(2 * ntot * std::mem::size_of::<f32>())?;
    let n1_buf = device.new_buffer_from_bytes(&n1.to_le_bytes())?;
    let n2_buf = device.new_buffer_from_bytes(&n2.to_le_bytes())?;
    let ntot_buf = device.new_buffer_from_bytes(&(ntot as u32).to_le_bytes())?;

    let lib = shared_library(device)?;
    let mut queue = CommandQueue::new(device)?;

    // Pass 1: column FFTs — n2 threadgroups, each width <= n1 (the column len).
    let cols_pso = lib.pipeline("fft_fourstep_cols")?;
    let col_tg = n1 as usize;
    queue.dispatch_1d_with_tg(
        &cols_pso,
        &[&data_buf, &n1_buf, &n2_buf],
        n2 as usize * col_tg,
        col_tg,
    )?;
    queue.wait_until_complete()?;

    // Pass 2: elementwise cross-twiddle over all N elements.
    let tw_pso = lib.pipeline("fft_twiddle_mul")?;
    queue.dispatch_1d(&tw_pso, &[&data_buf, &n2_buf, &ntot_buf], ntot)?;
    queue.wait_until_complete()?;

    // Pass 3: row FFTs — n1 threadgroups, each width <= n2 (the row len).
    let rows_pso = lib.pipeline("fft_fourstep_rows")?;
    let row_tg = n2 as usize;
    queue.dispatch_1d_with_tg(
        &rows_pso,
        &[&data_buf, &n1_buf, &n2_buf],
        n1 as usize * row_tg,
        row_tg,
    )?;
    queue.wait_until_complete()?;

    // Pass 4: transpose n1 x n2 -> n2 x n1 into the scratch buffer.
    let tr_pso = lib.pipeline("fft_transpose")?;
    queue.dispatch_1d(&tr_pso, &[&data_buf, &trans_buf, &n1_buf, &n2_buf], ntot)?;
    queue.wait_until_complete()?;

    let mut out = trans_buf.to_f32_vec();
    if inverse {
        // ifft = conj(fft(conj(x)))/N: conjugate + scale by 1/N on readback.
        let scale = 1.0f32 / n as f32;
        for c in out.chunks_exact_mut(2) {
            c[0] *= scale;
            c[1] = -c[1] * scale;
        }
    }
    Ok(out)
}

/// Naive O(N^2) DFT over interleaved-complex input `[re0,im0,re1,im1,...]`.
/// Reference oracle for tests ONLY (never a runtime path). `inverse` applies
/// +sign twiddles and 1/N scaling.
///
/// `#[doc(hidden)]` rather than `#[cfg(test)]`: integration tests in `tests/`
/// compile against the library without `--cfg test`, so a `cfg(test)` gate
/// would hide this oracle from them. It is hidden from docs and carries no
/// runtime cost (callers in `src/` never reference it).
#[doc(hidden)]
pub fn dft_reference(input: &[f32], n: usize, inverse: bool) -> Vec<f32> {
    let mut out = vec![0f32; 2 * n];
    let sign = if inverse { 1.0f64 } else { -1.0f64 };
    for k in 0..n {
        let (mut sre, mut sim) = (0f64, 0f64);
        for t in 0..n {
            let ang = sign * 2.0 * std::f64::consts::PI * (k as f64) * (t as f64) / (n as f64);
            let (c, s) = (ang.cos(), ang.sin());
            let (re, im) = (input[2 * t] as f64, input[2 * t + 1] as f64);
            sre += re * c - im * s;
            sim += re * s + im * c;
        }
        let scale = if inverse { 1.0 / n as f64 } else { 1.0 };
        out[2 * k] = (sre * scale) as f32;
        out[2 * k + 1] = (sim * scale) as f32;
    }
    out
}

/// L2 relative error between two interleaved-complex vectors (test helper).
/// `#[doc(hidden)]` for the same reason as [`dft_reference`].
#[doc(hidden)]
pub fn l2_rel_err(got: &[f32], exp: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for i in 0..exp.len() {
        let d = got[i] as f64 - exp[i] as f64;
        num += d * d;
        den += (exp[i] as f64) * (exp[i] as f64);
    }
    (num / den.max(1e-300)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::factorize;

    #[test]
    fn factorize_products_match_n() {
        for &n in &[3i64, 5, 6, 7, 9, 12, 15, 24, 35, 120, 360, 1000, 1024] {
            let f = factorize(n).expect("factorable");
            let prod: i64 = f.iter().map(|&r| r as i64).product();
            assert_eq!(prod, n, "n={n} factors={f:?}");
            assert!(f.iter().all(|&r| (2..=8).contains(&r)), "n={n} radix>8");
        }
    }

    #[test]
    fn factorize_rejects_large_prime() {
        assert!(factorize(11).is_none());
        assert!(factorize(13).is_none());
        assert!(factorize(22).is_none()); // 2 * 11
        assert!(factorize(0).is_none());
    }
}
