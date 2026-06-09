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
        // Larger sizes route to the recursive batched four-step path when pow2;
        // otherwise unsupported until later tasks. The recursion handles both
        // the single-level band (2^11..2^20) and the previously-broken
        // 2^21..2^25 band (where MLX returns garbage, ml-explore/mlx#1800).
        if (n & (n - 1)) == 0 {
            return fft_recursive_fourstep(device, input, n, inverse);
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

/// Maximum four-step recursion depth before bailing to [`FftError::Unsupported`].
/// With `l1 = FFT_BASE_MAX = 2^10` chosen at every level, each level reduces the
/// signal length by `2^10`, so depth `d` covers up to `2^(10*(d+1))`. Depth 6
/// covers `2^70` — astronomically beyond any size that fits in memory; the guard
/// exists only to make accidental non-termination a clean error, not a hang.
const FFT_MAX_RECURSION_DEPTH: u32 = 6;

/// Recursive batched Bailey four-step FFT for power-of-two `n > FFT_BASE_MAX`.
///
/// Stages the (optionally conjugated) input into a data buffer, allocates a
/// same-size scratch for the non-in-place per-signal transpose, runs
/// [`fft_pass`] (forward-only) over the single length-`n` signal, then reads
/// back. Inverse is handled at this boundary: `ifft(x) = conj(fft(conj(x)))/N`
/// — conjugate the input host-side, run the forward recursion, then conjugate
/// and scale by `1/N` on readback. This UNIFIES the old single-level
/// (2^11..2^20) four-step with the recursive (2^21..2^25) path: the former is
/// one recursion level, the latter two or more.
fn fft_recursive_fourstep(
    device: &MetalDevice,
    input: &[f32],
    n: i64,
    inverse: bool,
) -> Result<Vec<f32>, FftError> {
    debug_assert_eq!((n & (n - 1)), 0, "fft_recursive_fourstep requires pow2 n");
    let ntot = n as usize;

    // Stage the (optionally conjugated) input into the data buffer.
    let mut staged = input.to_vec();
    if inverse {
        for c in staged.chunks_exact_mut(2) {
            c[1] = -c[1];
        }
    }
    let mut data_buf = MetalBuffer::from_f32_slice(device, &staged)?;
    let mut scratch_buf = device.new_buffer_zeroed(2 * ntot * std::mem::size_of::<f32>())?;

    let lib = shared_library(device)?;
    let mut queue = CommandQueue::new(device)?;

    fft_pass(
        lib,
        &mut queue,
        device,
        &mut data_buf,
        &mut scratch_buf,
        ntot,
        1,
        0,
    )?;

    let mut out = data_buf.to_f32_vec();
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

/// Forward-only recursive batched four-step over `batch` contiguous signals,
/// each length `len` (pow2), packed back-to-back in `data` (signal `s` occupies
/// `data[s*len .. (s+1)*len]`). Result is written in place into `data`.
///
/// Base case (`len <= FFT_BASE_MAX`): dispatch `batch` threadgroups, one per
/// signal, running the batched single-threadgroup Stockham.
///
/// Recursive case: four-step each signal as `l1 x l2` row-major (`l1 =
/// FFT_BASE_MAX`, `l2 = len/l1`, both pow2, `l2 >= 2`):
///   1. column FFTs (length `l1`, strided) — `batch*l2` threadgroups,
///   2. cross-twiddle `W_len^{i*j}` (modulus = `len`, the CURRENT sub-length),
///   3. row FFTs (length `l2`) — RECURSE: the `l1` contiguous rows of every
///      signal form `batch*l1` contiguous length-`l2` signals at `data[0]`,
///   4. per-signal transpose `l1 x l2 -> l2 x l1` into `scratch`, then copy
///      `scratch -> data`.
///
/// Passes are data-dependent: `wait_until_complete` after each so later passes
/// read the prior output and per-pass GPU errors surface (the queue tracks only
/// the last command buffer; see `command.rs`).
///
/// `scratch` must be the same byte size as `data`; it is reused at every
/// recursion level for the transpose (the recursive row call runs in place on
/// `data`, using `scratch` for its own deeper transposes — disjoint in time).
#[allow(clippy::too_many_arguments)]
fn fft_pass(
    lib: &crate::shader_lib::ShaderLibrary,
    queue: &mut CommandQueue,
    device: &MetalDevice,
    data: &mut MetalBuffer,
    scratch: &mut MetalBuffer,
    len: usize,
    batch: usize,
    depth: u32,
) -> Result<(), FftError> {
    if depth > FFT_MAX_RECURSION_DEPTH {
        return Err(FftError::Unsupported(len as i64));
    }

    if len as i64 <= FFT_BASE_MAX {
        // Base case: one threadgroup per signal, contiguous length-len Stockham.
        let len_buf = device.new_buffer_from_bytes(&(len as u32).to_le_bytes())?;
        let pso = lib.pipeline("fft_stockham_pow2_batched")?;
        // Threadgroup width = len (the signal cooperatively transformed by its
        // group); grid = batch * len so each group covers one signal. Both
        // <= FFT_BASE_MAX <= maxTotalThreadsPerThreadgroup on Apple Silicon.
        queue.dispatch_1d_with_tg(&pso, &[data, &len_buf], batch * len, len)?;
        queue.wait_until_complete()?;
        return Ok(());
    }

    // Four-step: l1 = FFT_BASE_MAX (= 2^10), l2 = len / l1 (pow2, >= 2). If
    // l2 > FFT_BASE_MAX the recursion four-steps it again.
    let l1 = FFT_BASE_MAX as usize;
    let l2 = len / l1;
    debug_assert_eq!(l1 * l2, len, "l1*l2 must equal len");
    debug_assert!(l2 >= 2, "l2 must be >= 2 (len > FFT_BASE_MAX)");

    let l1_buf = device.new_buffer_from_bytes(&(l1 as u32).to_le_bytes())?;
    let l2_buf = device.new_buffer_from_bytes(&(l2 as u32).to_le_bytes())?;
    let len_buf = device.new_buffer_from_bytes(&(len as u32).to_le_bytes())?;
    let batch_buf = device.new_buffer_from_bytes(&(batch as u32).to_le_bytes())?;

    // Pass 1: column FFTs — batch*l2 threadgroups, width l1 (<= FFT_BASE_MAX).
    let cols_pso = lib.pipeline("fft_fourstep_cols")?;
    queue.dispatch_1d_with_tg(
        &cols_pso,
        &[data, &l1_buf, &l2_buf, &batch_buf],
        batch * l2 * l1,
        l1,
    )?;
    queue.wait_until_complete()?;

    // Pass 2: cross-twiddle over all batch*len elements (modulus = len).
    let tw_pso = lib.pipeline("fft_twiddle_mul")?;
    queue.dispatch_1d(&tw_pso, &[data, &l2_buf, &len_buf, &batch_buf], batch * len)?;
    queue.wait_until_complete()?;

    // Pass 3: row FFTs — RECURSE. The l1 contiguous rows of length l2 across all
    // batch signals are batch*l1 contiguous length-l2 signals starting at data[0].
    fft_pass(lib, queue, device, data, scratch, l2, batch * l1, depth + 1)?;

    // Pass 4: per-signal transpose l1 x l2 -> l2 x l1 into scratch, then copy
    // scratch -> data. Transpose is NOT in-place, hence the separate buffer.
    let tr_pso = lib.pipeline("fft_transpose")?;
    queue.dispatch_1d(
        &tr_pso,
        &[data, scratch, &l1_buf, &l2_buf, &batch_buf],
        batch * len,
    )?;
    queue.wait_until_complete()?;

    let count_buf = device.new_buffer_from_bytes(&((batch * len) as u32).to_le_bytes())?;
    let copy_pso = lib.pipeline("fft_copy")?;
    queue.dispatch_1d(&copy_pso, &[scratch, data, &count_buf], batch * len)?;
    queue.wait_until_complete()?;

    Ok(())
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
