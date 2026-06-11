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
            // All kernel scalars (len, batch, count) and the in-kernel twiddle
            // index i*j (< len <= n) are 32-bit. Cap n at 2^30 so neither the
            // u32 scalars (batch*len == n) nor the i32 twiddle index can
            // overflow; larger N would silently truncate. (The differential
            // tests run to 2^25; this leaves margin.)
            if n > (1 << 30) {
                return Err(FftError::Unsupported(n));
            }
            return fft_recursive_fourstep(device, input, n, inverse);
        }
        // Composite smooth n <= FFT_BASE_MAX is handled below; composite smooth
        // n > FFT_BASE_MAX and primes both fall through to Bluestein. Smooth
        // sizes > 1024 would also work via Bluestein here; the four-step path
        // already claimed pow2, and the mixed-radix kernel is single-threadgroup
        // (n <= 1024), so anything non-pow2 above 1024 routes to Bluestein.
        return bluestein_dispatch(device, input, n, inverse);
    }
    if (n & (n - 1)) != 0 && factorize(n).is_none() {
        // Small non-smooth n (prime factor > 7, e.g. 101 .. 1024) → Bluestein.
        return bluestein_dispatch(device, input, n, inverse);
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
        // Unreachable: small non-smooth n was routed to Bluestein above before
        // reaching the single-threadgroup base path. Kept as a defensive guard.
        return Err(FftError::Unsupported(n));
    }

    queue.wait_until_complete()?;
    Ok(out_buf.to_f32_vec())
}

/// Smallest power of two `>= x` (with `next_pow2(0) == 1`). Used to size the
/// Bluestein convolution length `M = next_pow2(2N-1)`.
fn next_pow2(x: u64) -> u64 {
    if x <= 1 {
        return 1;
    }
    1u64 << (64 - (x - 1).leading_zeros())
}

/// Bluestein boundary wrapper: dispatches the FORWARD chirp-z [`bluestein`]
/// and handles inverse the same way the four-step path does —
/// `ifft(x) = conj(fft(conj(x)))/N`. Conjugate the input host-side, run the
/// forward Bluestein, then conjugate + scale by `1/N` on readback.
///
/// Guards `M = next_pow2(2N-1) <= 2^30` so the internal pow2 FFTs stay within
/// the four-step path's supported range; otherwise [`FftError::Unsupported`].
fn bluestein_dispatch(
    device: &MetalDevice,
    input: &[f32],
    n: i64,
    inverse: bool,
) -> Result<Vec<f32>, FftError> {
    if n < 2 {
        return Err(FftError::Unsupported(n));
    }
    let m = next_pow2(2 * n as u64 - 1);
    if m > (1 << 30) {
        return Err(FftError::Unsupported(n));
    }
    let nn = n as usize;

    if !inverse {
        return bluestein(device, input, nn);
    }

    // ifft via forward Bluestein on the conjugated input.
    let mut staged = input.to_vec();
    for c in staged.chunks_exact_mut(2) {
        c[1] = -c[1];
    }
    let mut out = bluestein(device, &staged, nn)?;
    let scale = 1.0f32 / n as f32;
    for c in out.chunks_exact_mut(2) {
        c[0] *= scale;
        c[1] = -c[1] * scale;
    }
    Ok(out)
}

/// Forward length-`n` DFT via Bluestein's chirp-z transform, for arbitrary
/// `n >= 2` (in particular primes and large-prime-factor composites the smooth
/// paths reject). Reduces the DFT to a length-`M = next_pow2(2n-1)` circular
/// convolution evaluated by the (verified) pow2 four-step FFT.
///
/// Math: `X[k] = b[k] * (a ∗ h)[k]` where the chirp `b[m] = e^{-iπ m²/n}`,
/// `a[n_] = x[n_]·b[n_]`, and the filter `h[m] = conj(b[m]) = e^{+iπ m²/n}`
/// (even: `h[-m]=h[m]`). The linear convolution `a ∗ h` is computed as a
/// zero-padded length-`M` circular convolution:
/// `conv = IFFT_M(FFT_M(A) · FFT_M(H))`.
///
/// The chirp is built host-side in f64 using the period-`2n` reduction of `m²`
/// (`mm = (m·m) mod 2n`) so `m²` never overflows for large `n`. All O(M) work
/// (chirp/filter build, pointwise product, post-multiply) is host-side; only the
/// three length-`M` pow2 FFTs run on the GPU. `M` is pow2, so each FFT routes to
/// the four-step path and never re-enters Bluestein (no recursion).
///
/// Perf note (future optimization): the chirp/filter build, the `FFT·FFT`
/// pointwise product, and the final post-multiply are done host-side for
/// simplicity, which costs extra host↔device round-trips around the three GPU
/// FFTs. A fused premul/postmul MSL kernel (à la the plan's `fft_chirp_premul` /
/// `fft_chirp_postmul`) would keep this O(M) work on-device and cut those
/// round-trips. Bluestein is the non-headline (prime / non-smooth N) path, so
/// this is deferred; the large-pow2 four-step regime is already fully on-GPU.
///
/// Input `input` is interleaved-complex length `2n`; output is interleaved
/// length `2n`.
fn bluestein(device: &MetalDevice, input: &[f32], n: usize) -> Result<Vec<f32>, FftError> {
    use std::f64::consts::PI;
    let m = next_pow2(2 * n as u64 - 1) as usize;
    let two_n = 2 * n as u64;

    // Chirp b[m_] = e^{-iπ m_²/n}, built in f64 via mm = m_² mod 2n.
    let mut b_re = vec![0f64; n];
    let mut b_im = vec![0f64; n];
    for (mi, (br, bi)) in b_re.iter_mut().zip(b_im.iter_mut()).enumerate() {
        let mm = (mi as u64 * mi as u64) % two_n;
        let angle = -PI * (mm as f64) / (n as f64);
        *br = angle.cos();
        *bi = angle.sin();
    }

    // A: a[n_] = x[n_]·b[n_], zero-padded to M (interleaved length 2M).
    let mut a = vec![0f32; 2 * m];
    for i in 0..n {
        let (xr, xi) = (input[2 * i] as f64, input[2 * i + 1] as f64);
        let (br, bi) = (b_re[i], b_im[i]);
        a[2 * i] = (xr * br - xi * bi) as f32;
        a[2 * i + 1] = (xr * bi + xi * br) as f32;
    }

    // H: filter h[m_] = conj(b[m_]) = (b_re, -b_im). H[0]=h[0]; for m_ in [1,n):
    // H[m_]=h[m_] and H[M-m_]=h[m_] (even symmetry, negative lags wrap to the top).
    let mut h = vec![0f32; 2 * m];
    h[0] = b_re[0] as f32; // h[0] = conj(b[0]); b[0] = 1, imag 0.
    h[1] = -b_im[0] as f32;
    for mi in 1..n {
        let (hr, hi) = (b_re[mi] as f32, -b_im[mi] as f32);
        h[2 * mi] = hr;
        h[2 * mi + 1] = hi;
        let w = m - mi;
        h[2 * w] = hr;
        h[2 * w + 1] = hi;
    }

    // FFT_M(A) · FFT_M(H), then IFFT_M. M is pow2 → four-step path.
    let fa = fft_gpu(device, &a, m as i64, false)?;
    let fh = fft_gpu(device, &h, m as i64, false)?;
    let mut c = vec![0f32; 2 * m];
    for i in 0..m {
        let (ar, ai) = (fa[2 * i] as f64, fa[2 * i + 1] as f64);
        let (hr, hi) = (fh[2 * i] as f64, fh[2 * i + 1] as f64);
        c[2 * i] = (ar * hr - ai * hi) as f32;
        c[2 * i + 1] = (ar * hi + ai * hr) as f32;
    }
    let conv = fft_gpu(device, &c, m as i64, true)?;

    // X[k] = b[k] · conv[k] for k in [0,n).
    let mut out = vec![0f32; 2 * n];
    for k in 0..n {
        let (cr, ci) = (conv[2 * k] as f64, conv[2 * k + 1] as f64);
        let (br, bi) = (b_re[k], b_im[k]);
        out[2 * k] = (br * cr - bi * ci) as f32;
        out[2 * k + 1] = (br * ci + bi * cr) as f32;
    }
    Ok(out)
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

    // Stage into the data buffer. Forward: no clone — from_f32_slice copies into
    // Metal memory anyway. Inverse: clone to conjugate (input is borrowed).
    let mut data_buf = if inverse {
        let mut staged = input.to_vec();
        for c in staged.chunks_exact_mut(2) {
            c[1] = -c[1];
        }
        MetalBuffer::from_f32_slice(device, &staged)?
    } else {
        MetalBuffer::from_f32_slice(device, input)?
    };
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

/// Build the `constant uint& n` scalar buffer for the pack/unpack kernels,
/// guarding the `usize`→`u32` narrowing (mirrors the 2^30 cap in `fft_gpu`; a
/// larger `n` would silently wrap to a bogus thread count).
fn make_n_buf(device: &MetalDevice, n: usize) -> Result<MetalBuffer, FftError> {
    if n > (1 << 30) {
        return Err(FftError::Unsupported(n as i64));
    }
    Ok(device.new_buffer_from_bytes(&(n as u32).to_le_bytes())?)
}

/// Pack a real host signal into an interleaved-complex host `Vec` (length `2n`)
/// on the GPU. Each element `i` of the output satisfies `out[2i] = re[i]` and
/// `out[2i+1] = 0`. Mirrors the CPU scatter in `fft_core` so M5b can keep the
/// whole FFT pipeline on-device.
pub fn dispatch_pack_real(
    device: &MetalDevice,
    re: &[f32],
    n: usize,
) -> Result<Vec<f32>, FftError> {
    let in_buf = MetalBuffer::from_f32_slice(device, re)?;
    let out_buf = device.new_buffer_zeroed(2 * n * std::mem::size_of::<f32>())?;
    let n_buf = make_n_buf(device, n)?;

    let lib = shared_library(device)?;
    let pso = lib.pipeline("fft_pack_real_to_interleaved")?;
    let mut queue = CommandQueue::new(device)?;
    // One thread per sample; dispatch_1d handles threadgroup sizing at runtime.
    queue.dispatch_1d(&pso, &[&in_buf, &out_buf, &n_buf], n)?;
    queue.wait_until_complete()?;
    Ok(out_buf.to_f32_vec())
}

/// Pack separate real and imaginary host planes into an interleaved-complex host
/// `Vec` (length `2n`) on the GPU. Each element `i` of the output satisfies
/// `out[2i] = re[i]` and `out[2i+1] = im[i]`. Mirrors the CPU scatter in
/// `fft_core`'s complex branch so M5b can keep the whole FFT pipeline on-device.
pub fn dispatch_pack_complex(
    device: &MetalDevice,
    re: &[f32],
    im: &[f32],
    n: usize,
) -> Result<Vec<f32>, FftError> {
    let re_buf = MetalBuffer::from_f32_slice(device, re)?;
    let im_buf = MetalBuffer::from_f32_slice(device, im)?;
    let out_buf = device.new_buffer_zeroed(2 * n * std::mem::size_of::<f32>())?;
    let n_buf = make_n_buf(device, n)?;

    let lib = shared_library(device)?;
    let pso = lib.pipeline("fft_pack_complex_to_interleaved")?;
    let mut queue = CommandQueue::new(device)?;
    // One thread per sample; dispatch_1d handles threadgroup sizing at runtime.
    queue.dispatch_1d(&pso, &[&re_buf, &im_buf, &out_buf, &n_buf], n)?;
    queue.wait_until_complete()?;
    Ok(out_buf.to_f32_vec())
}

/// Unpack an interleaved-complex host slice (length `2n`) into `(re_out,
/// im_out)` host `Vec`s on the GPU. Each element `i` satisfies
/// `re_out[i] = inter[2i]` and `im_out[i] = inter[2i+1]`. Mirrors the CPU
/// gather in `fft_core` so M5b can keep the whole FFT pipeline on-device.
pub fn dispatch_unpack(
    device: &MetalDevice,
    inter: &[f32],
    n: usize,
) -> Result<(Vec<f32>, Vec<f32>), FftError> {
    let in_buf = MetalBuffer::from_f32_slice(device, inter)?;
    let re_out_buf = device.new_buffer_zeroed(n * std::mem::size_of::<f32>())?;
    let im_out_buf = device.new_buffer_zeroed(n * std::mem::size_of::<f32>())?;
    let n_buf = make_n_buf(device, n)?;

    let lib = shared_library(device)?;
    let pso = lib.pipeline("fft_unpack_interleaved_to_planar")?;
    let mut queue = CommandQueue::new(device)?;
    // One thread per sample; dispatch_1d handles threadgroup sizing at runtime.
    queue.dispatch_1d(&pso, &[&in_buf, &re_out_buf, &im_out_buf, &n_buf], n)?;
    queue.wait_until_complete()?;
    Ok((re_out_buf.to_f32_vec(), im_out_buf.to_f32_vec()))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
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
