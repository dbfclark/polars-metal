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
/// limit. Sizes above this route through the four-step / Bluestein paths.
pub const FFT_BASE_MAX: i64 = 1024;

/// Decompose `n` into an ordered list of radices, each in `{2,3,4,5,6,7,8}`,
/// whose product is `n`. Returns `None` if any prime factor `> 7` remains
/// (such sizes route to Bluestein in a later task) or for `n <= 0`.
///
/// True powers of two are handled by the dedicated radix-2 path in `fft_gpu_buf`
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
/// Thin host-slice wrapper over [`fft_gpu_buf`]: stage `input` into a Metal
/// buffer, run the buffer-level core (which routes among radix-2/mixed-radix,
/// the four-step path, and Bluestein by size), and read the result back. All
/// supported sizes (pow2, smooth composite, and non-smooth/prime via Bluestein)
/// go through here; routing + the size caps live in [`fft_gpu_buf`].
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
    // Thin host-slice wrapper over the buffer-in/buffer-out core: stage `input`
    // into `in_buf`, allocate `out_buf`, transform, read back. All routing and
    // validation lives in `fft_gpu_buf` so the GPU-resident pipeline (M5b-2) can
    // reuse it without round-tripping through host slices.
    let in_buf = MetalBuffer::from_f32_slice(device, input)?;
    let out_buf = device.new_buffer_zeroed((2 * n as usize) * std::mem::size_of::<f32>())?;
    fft_gpu_buf(device, &in_buf, &out_buf, n, inverse)?;
    Ok(out_buf.to_f32_vec())
}

/// Buffer-in / buffer-out FFT core. `in_buf` holds the interleaved-complex input
/// (`2n` f32) and the transform result lands in `out_buf` (`2n` f32). All FFT
/// paths route through here; [`fft_gpu`] is a thin host-slice wrapper that stages
/// `in_buf` and reads back `out_buf`.
///
/// Routing (identical to the former `fft_gpu`):
/// - `n <= FFT_BASE_MAX`, pow2 or smooth: single-threadgroup radix-2 / mixed-radix
///   Stockham dispatched directly into `out_buf`.
/// - `n > FFT_BASE_MAX`, pow2: recursive batched four-step via
///   [`fft_recursive_fourstep_buf`] (genuinely buffer-level, on-device).
/// - non-smooth `n` (prime factor > 7): Bluestein. Bluestein's numerics stay
///   host-internal (small-N, host-conjugation-heavy, ~zero on-device benefit);
///   this routes through a host bridge — read `in_buf` → host, call the existing
///   [`bluestein_dispatch`], memcpy its result Vec into `out_buf`. Only the API
///   is buffer-level for this path.
///
/// `inverse` applies 1/N scaling. The 2^30 caps (kernel scalars / twiddle index
/// are 32-bit) are enforced here, where the routing lives.
pub fn fft_gpu_buf(
    device: &MetalDevice,
    in_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    n: i64,
    inverse: bool,
) -> Result<(), FftError> {
    if n <= 0 {
        return Err(FftError::Unsupported(n));
    }
    if n > FFT_BASE_MAX {
        // Larger sizes route to the recursive batched four-step path when pow2;
        // otherwise Bluestein. The recursion handles both the single-level band
        // (2^11..2^20) and the previously-broken 2^21..2^25 band (where MLX
        // returns garbage, ml-explore/mlx#1800).
        if (n & (n - 1)) == 0 {
            // All kernel scalars (len, batch, count) and the in-kernel twiddle
            // index i*j (< len <= n) are 32-bit. Cap n at 2^30 so neither the
            // u32 scalars (batch*len == n) nor the i32 twiddle index can
            // overflow; larger N would silently truncate. (The differential
            // tests run to 2^25; this leaves margin.)
            if n > (1 << 30) {
                return Err(FftError::Unsupported(n));
            }
            return fft_recursive_fourstep_buf(device, in_buf, out_buf, n, inverse);
        }
        // Composite smooth n <= FFT_BASE_MAX is handled below; composite smooth
        // n > FFT_BASE_MAX and primes both fall through to Bluestein. Smooth
        // sizes > 1024 would also work via Bluestein here; the four-step path
        // already claimed pow2, and the mixed-radix kernel is single-threadgroup
        // (n <= 1024), so anything non-pow2 above 1024 routes to Bluestein.
        return bluestein_bridge(device, in_buf, out_buf, n, inverse);
    }
    if (n & (n - 1)) != 0 && factorize(n).is_none() {
        // Small non-smooth n (prime factor > 7, e.g. 101 .. 1024) → Bluestein.
        return bluestein_bridge(device, in_buf, out_buf, n, inverse);
    }

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
        queue.dispatch_1d_with_tg(&pso, &[in_buf, out_buf, &n_buf, &inv_buf], tg, tg)?;
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
                in_buf,
                out_buf,
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
    Ok(())
}

/// On-device PLANAR (SoA) FFT core: `re_in`/`im_in` (each length `n` f32) →
/// `re_out`/`im_out` (each length `n` f32). No interleaving — the kernels read
/// and write separate re/im global buffers directly, eliminating the host-side
/// planar→interleaved pack and interleaved→planar unpack that wrap [`fft_gpu_buf`].
///
/// This is the parallel planar path to the proven interleaved [`fft_gpu_buf`];
/// the planar kernels differ from the interleaved ones ONLY at global buffer I/O
/// (the threadgroup butterfly math is identical), and a differential test
/// verifies they match.
///
/// (M5c-1: only the single-threadgroup base case is implemented — `n <=
/// FFT_BASE_MAX` that is pow2 or smooth-composite. Larger pow2 (four-step) returns
/// [`FftError::Unsupported`] until M5c-2; non-smooth (Bluestein) until M5c-3.)
///
/// `inverse` applies 1/N scaling. Mirrors [`fft_gpu_buf`]'s `n <= FFT_BASE_MAX`
/// dispatch exactly — same scalar-buffer construction and `dispatch_1d_with_tg`
/// shape — only the buffer LIST gains the two extra planes and the pipeline name
/// is the `_planar_` variant.
pub fn fft_gpu_planar_core(
    device: &MetalDevice,
    re_in: &MetalBuffer,
    im_in: &MetalBuffer,
    re_out: &MetalBuffer,
    im_out: &MetalBuffer,
    n: i64,
    inverse: bool,
) -> Result<(), FftError> {
    if n <= 0 {
        return Err(FftError::Unsupported(n));
    }
    if n > FFT_BASE_MAX {
        // Larger pow2 routes to the planar recursive four-step path (M5c-2),
        // mirroring fft_gpu_buf's large-pow2 branch. The 2^30 cap guards the
        // u32 kernel scalars (batch*len == n) and the i32 twiddle index i*j.
        if (n & (n - 1)) == 0 {
            if n > (1 << 30) {
                return Err(FftError::Unsupported(n));
            }
            return fft_recursive_fourstep_planar(device, re_in, im_in, re_out, im_out, n, inverse);
        }
        // Non-smooth large n → planar Bluestein bridge.
        return bluestein_bridge_planar(device, re_in, im_in, re_out, im_out, n, inverse);
    }
    if (n & (n - 1)) != 0 && factorize(n).is_none() {
        // Small non-smooth n (prime factor > 7) → planar Bluestein bridge.
        return bluestein_bridge_planar(device, re_in, im_in, re_out, im_out, n, inverse);
    }

    let n_buf = device.new_buffer_from_bytes(&(n as u32).to_le_bytes())?;
    let inv_buf = device.new_buffer_from_bytes(&u32::from(inverse).to_le_bytes())?;

    let lib = shared_library(device)?;
    let mut queue = CommandQueue::new(device)?;
    // Single-threadgroup invariant: identical to fft_gpu_buf — grid_width =
    // tg_width = tg, n <= FFT_BASE_MAX (1024) <= maxTotalThreadsPerThreadgroup, so
    // the dispatch never splits into multiple threadgroups; the kernel strides by
    // tg_size to cover all n points.
    let tg = (n as usize).min(FFT_BASE_MAX as usize);

    if (n & (n - 1)) == 0 {
        // Power-of-two: planar radix-2 Stockham.
        let pso = lib.pipeline("fft_stockham_pow2_planar_f32")?;
        queue.dispatch_1d_with_tg(
            &pso,
            &[re_in, im_in, re_out, im_out, &n_buf, &inv_buf],
            tg,
            tg,
        )?;
    } else if let Some(radices) = factorize(n) {
        // Composite smooth n: planar mixed-radix (3..8) Stockham.
        let mut radix_bytes = Vec::with_capacity(radices.len() * 4);
        for r in &radices {
            radix_bytes.extend_from_slice(&r.to_le_bytes());
        }
        let radices_buf = device.new_buffer_from_bytes(&radix_bytes)?;
        let n_radices_buf = device.new_buffer_from_bytes(&(radices.len() as u32).to_le_bytes())?;
        let pso = lib.pipeline("fft_mixed_radix_planar_f32")?;
        queue.dispatch_1d_with_tg(
            &pso,
            &[
                re_in,
                im_in,
                re_out,
                im_out,
                &n_buf,
                &inv_buf,
                &radices_buf,
                &n_radices_buf,
            ],
            tg,
            tg,
        )?;
    } else {
        // Unreachable: small non-smooth n was rejected above before reaching the
        // single-threadgroup base path. Kept as a defensive guard.
        return Err(FftError::Unsupported(n));
    }

    queue.wait_until_complete()?;
    Ok(())
}

/// Planar Bluestein bridge: interleave the planar (re,im) inputs to host, run the
/// existing interleaved Bluestein, split the result back to the planar outputs.
/// Bluestein is small-N and rare, so the host interleave/split here is negligible
/// (the planar rewrite targets the four-step large-N path, not this one).
fn bluestein_bridge_planar(
    device: &MetalDevice,
    re_in: &MetalBuffer,
    im_in: &MetalBuffer,
    re_out: &MetalBuffer,
    im_out: &MetalBuffer,
    n: i64,
    inverse: bool,
) -> Result<(), FftError> {
    let re = re_in.to_f32_vec();
    let im = im_in.to_f32_vec();
    let nn = n as usize;
    let mut inter = vec![0.0f32; 2 * nn];
    for i in 0..nn {
        inter[2 * i] = re[i];
        inter[2 * i + 1] = im[i];
    }
    let result = bluestein_dispatch(device, &inter, n, inverse)?;
    let mut ro = vec![0.0f32; nn];
    let mut io = vec![0.0f32; nn];
    for i in 0..nn {
        ro[i] = result[2 * i];
        io[i] = result[2 * i + 1];
    }
    write_f32_into(re_out, &ro);
    write_f32_into(im_out, &io);
    Ok(())
}

/// Bluestein host bridge for [`fft_gpu_buf`]: read `in_buf` → host, run the
/// existing host-internal [`bluestein_dispatch`], memcpy its result into
/// `out_buf`. Bluestein is the non-headline (prime / non-smooth N) path — its
/// chirp build, pointwise product, and post-multiply are all small-N host work,
/// so keeping it host-internal here costs nothing the on-device pipeline cares
/// about (M5b-2's GPU-resident win is for the large-pow2 four-step regime).
fn bluestein_bridge(
    device: &MetalDevice,
    in_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    n: i64,
    inverse: bool,
) -> Result<(), FftError> {
    let host_input = in_buf.to_f32_vec();
    let result = bluestein_dispatch(device, &host_input, n, inverse)?;
    write_f32_into(out_buf, &result);
    Ok(())
}

/// Copy an interleaved-complex host `Vec` into a Shared-storage `MetalBuffer`
/// (bytewise memcpy through its CPU-coherent backing store). Used by the
/// host-internal Bluestein bridge to land a host result into a caller-provided
/// `out_buf`. `result` must be exactly `out_buf.len()` bytes of f32 (`2n`
/// floats).
fn write_f32_into(out_buf: &MetalBuffer, result: &[f32]) {
    let dst = out_buf.as_slice();
    let nbytes = std::mem::size_of_val(result);
    debug_assert_eq!(dst.len(), nbytes);
    // SAFETY: out_buf is StorageModeShared and CPU-writable; `dst.as_ptr()`
    // points at its backing store, valid for `dst.len()` bytes. No GPU command
    // buffer is in-flight against out_buf at this call site (the Bluestein
    // bridge never dispatches into out_buf — it produces the result host-side
    // and only then copies it in), so `&MetalBuffer` (not `&mut`) is sound for
    // the write. `result` has the same byte length (debug-asserted). The
    // bytewise copy is alignment-agnostic (mirrors `to_f32_vec`).
    unsafe {
        std::ptr::copy_nonoverlapping(
            result.as_ptr().cast::<u8>(),
            dst.as_ptr() as *mut u8,
            nbytes,
        );
    }
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

/// Buffer-in / buffer-out recursive batched Bailey four-step FFT for power-of-two
/// `n > FFT_BASE_MAX`. This UNIFIES the old single-level (2^11..2^20) four-step
/// with the recursive (2^21..2^25) path: the former is one recursion level, the
/// latter two or more.
///
/// Runs the four-step recursion GENUINELY on-device: `out_buf` doubles as the
/// in-place `data` buffer for [`fft_pass`] (forward-only, over the single
/// length-`n` signal), so the transform result lands in `out_buf` with no final
/// blit. Only a same-size `scratch` is allocated (for the non-in-place
/// per-signal transpose).
///
/// Input staging into `out_buf`:
/// - Forward: copy `in_buf` → `out_buf` (host bytewise memcpy through the Shared
///   backing store; `fft_pass` then runs in place).
/// - Inverse: `ifft(x) = conj(fft(conj(x)))/N`. Conjugate the input host-side
///   (read `in_buf` → host, negate imag) and stage into `out_buf`; after the
///   forward recursion, conjugate + scale by `1/N` in place on `out_buf`'s
///   backing store. The inverse path reads `in_buf` to host ONLY to conjugate —
///   the rarer path, and this preserves the exact prior behavior.
fn fft_recursive_fourstep_buf(
    device: &MetalDevice,
    in_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    n: i64,
    inverse: bool,
) -> Result<(), FftError> {
    debug_assert_eq!(
        (n & (n - 1)),
        0,
        "fft_recursive_fourstep_buf requires pow2 n"
    );
    let ntot = n as usize;

    // Seed out_buf (the in-place data buffer). Forward: byte-copy in_buf as-is.
    // Inverse: read in_buf → host, conjugate, write into out_buf.
    if inverse {
        let mut staged = in_buf.to_f32_vec();
        for c in staged.chunks_exact_mut(2) {
            c[1] = -c[1];
        }
        write_f32_into(out_buf, &staged);
    } else {
        copy_buf_into(out_buf, in_buf);
    }

    let mut data_buf = out_buf.shallow_clone();
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

    if inverse {
        // ifft = conj(fft(conj(x)))/N: conjugate + scale by 1/N in place on
        // out_buf's backing store. The pass has completed (fft_pass's last stage
        // waits), so no GPU work is in-flight against out_buf.
        let mut out = out_buf.to_f32_vec();
        let scale = 1.0f32 / n as f32;
        for c in out.chunks_exact_mut(2) {
            c[0] *= scale;
            c[1] = -c[1] * scale;
        }
        write_f32_into(out_buf, &out);
    }
    Ok(())
}

/// Bytewise copy `src` (a Shared-storage `MetalBuffer`) into `dst` (same size,
/// Shared storage). Both must be `2n` f32. CPU-side memcpy through the coherent
/// backing stores — used to seed the four-step `out_buf` from `in_buf` on the
/// forward path without a host round-trip Vec.
fn copy_buf_into(dst: &MetalBuffer, src: &MetalBuffer) {
    let d = dst.as_slice();
    let s = src.as_slice();
    debug_assert_eq!(d.len(), s.len());
    // copy_nonoverlapping is UB if dst/src alias the same MTLBuffer (possible now
    // that shallow_clone hands out aliased handles); catch it in debug builds.
    debug_assert_ne!(
        d.as_ptr(),
        s.as_ptr(),
        "copy_buf_into: dst and src alias the same MTLBuffer"
    );
    // SAFETY: both are StorageModeShared; `d`/`s` point at their backing stores,
    // valid for `len` bytes. dst is a freshly-allocated out_buf with no GPU work
    // in-flight; src is the caller's input buffer (also not mid-dispatch at this
    // seeding point). Disjoint allocations, so copy_nonoverlapping is sound.
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr(), d.as_ptr() as *mut u8, d.len());
    }
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

/// PLANAR (SoA) twin of [`fft_recursive_fourstep_buf`] for power-of-two
/// `n > FFT_BASE_MAX`. Separate re/im input planes (`re_in`/`im_in`, each `n`
/// f32) → separate re/im output planes (`re_out`/`im_out`, each `n` f32) — no
/// interleaving. Mirrors the interleaved driver exactly, with every buffer
/// doubled (data → data_re/data_im, scratch → scratch_re/scratch_im).
///
/// Input staging into the out planes (which double as the in-place `data`
/// planes for [`fft_pass_planar`]):
/// - Forward: byte-copy `re_in` → `re_out` and `im_in` → `im_out`.
/// - Inverse: `ifft(x) = conj(fft(conj(x)))/N`. PLANAR conjugation is just
///   negating the WHOLE im plane (no stride, unlike interleaved's every-other
///   element): copy `re_in` → `re_out` as-is, and stage the negated `im_in`
///   into `im_out`. After the forward recursion, conjugate + scale by `1/N` in
///   place — `re_out *= 1/N`, `im_out = -im_out/N` — on the backing stores.
///
/// Scratch planes are length `n` (`n` f32 each), NOT `2n` — that's the planar
/// saving vs the interleaved `2n` scratch.
fn fft_recursive_fourstep_planar(
    device: &MetalDevice,
    re_in: &MetalBuffer,
    im_in: &MetalBuffer,
    re_out: &MetalBuffer,
    im_out: &MetalBuffer,
    n: i64,
    inverse: bool,
) -> Result<(), FftError> {
    debug_assert_eq!(
        (n & (n - 1)),
        0,
        "fft_recursive_fourstep_planar requires pow2 n"
    );
    let ntot = n as usize;

    // Seed the out planes (which double as the in-place data planes). Forward:
    // byte-copy both planes as-is. Inverse: copy re as-is, stage negated im.
    if inverse {
        copy_buf_into(re_out, re_in);
        let mut staged_im = im_in.to_f32_vec();
        for v in staged_im.iter_mut() {
            *v = -*v;
        }
        write_f32_into(im_out, &staged_im);
    } else {
        copy_buf_into(re_out, re_in);
        copy_buf_into(im_out, im_in);
    }

    let mut data_re = re_out.shallow_clone();
    let mut data_im = im_out.shallow_clone();
    let mut scratch_re = device.new_buffer_zeroed(ntot * std::mem::size_of::<f32>())?;
    let mut scratch_im = device.new_buffer_zeroed(ntot * std::mem::size_of::<f32>())?;

    let lib = shared_library(device)?;
    let mut queue = CommandQueue::new(device)?;

    fft_pass_planar(
        lib,
        &mut queue,
        device,
        &mut data_re,
        &mut data_im,
        &mut scratch_re,
        &mut scratch_im,
        ntot,
        1,
        0,
    )?;

    if inverse {
        // ifft = conj(fft(conj(x)))/N: conjugate + scale by 1/N in place on the
        // out planes' backing stores. The pass has completed (fft_pass_planar's
        // last stage waits), so no GPU work is in-flight against the out planes.
        let scale = 1.0f32 / n as f32;
        let mut ro = re_out.to_f32_vec();
        for v in ro.iter_mut() {
            *v *= scale;
        }
        write_f32_into(re_out, &ro);
        let mut io = im_out.to_f32_vec();
        for v in io.iter_mut() {
            *v = -*v * scale;
        }
        write_f32_into(im_out, &io);
    }
    Ok(())
}

/// PLANAR twin of [`fft_pass`]: forward-only recursive batched four-step over
/// `batch` contiguous signals, each length `len` (pow2), packed back-to-back in
/// the `data_re`/`data_im` planes (signal `s` occupies `[s*len .. (s+1)*len]` in
/// each plane). Result is written in place into the data planes.
///
/// Mirrors [`fft_pass`] dispatch-for-dispatch, threading BOTH data planes and
/// BOTH scratch planes through the recursion (the planar kernels split each
/// interleaved float2 buffer into its re/im planes). Same per-pass
/// `wait_until_complete` (passes are data-dependent) and same
/// `FFT_MAX_RECURSION_DEPTH` guard.
#[allow(clippy::too_many_arguments)]
fn fft_pass_planar(
    lib: &crate::shader_lib::ShaderLibrary,
    queue: &mut CommandQueue,
    device: &MetalDevice,
    data_re: &mut MetalBuffer,
    data_im: &mut MetalBuffer,
    scratch_re: &mut MetalBuffer,
    scratch_im: &mut MetalBuffer,
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
        let pso = lib.pipeline("fft_stockham_pow2_batched_planar")?;
        queue.dispatch_1d_with_tg(&pso, &[data_re, data_im, &len_buf], batch * len, len)?;
        queue.wait_until_complete()?;
        return Ok(());
    }

    // Four-step: l1 = FFT_BASE_MAX (= 2^10), l2 = len / l1 (pow2, >= 2).
    let l1 = FFT_BASE_MAX as usize;
    let l2 = len / l1;
    debug_assert_eq!(l1 * l2, len, "l1*l2 must equal len");
    debug_assert!(l2 >= 2, "l2 must be >= 2 (len > FFT_BASE_MAX)");

    let l1_buf = device.new_buffer_from_bytes(&(l1 as u32).to_le_bytes())?;
    let l2_buf = device.new_buffer_from_bytes(&(l2 as u32).to_le_bytes())?;
    let len_buf = device.new_buffer_from_bytes(&(len as u32).to_le_bytes())?;
    let batch_buf = device.new_buffer_from_bytes(&(batch as u32).to_le_bytes())?;

    // Pass 1: column FFTs — batch*l2 threadgroups, width l1 (<= FFT_BASE_MAX).
    let cols_pso = lib.pipeline("fft_fourstep_cols_planar")?;
    queue.dispatch_1d_with_tg(
        &cols_pso,
        &[data_re, data_im, &l1_buf, &l2_buf, &batch_buf],
        batch * l2 * l1,
        l1,
    )?;
    queue.wait_until_complete()?;

    // Pass 2: cross-twiddle over all batch*len elements (modulus = len).
    let tw_pso = lib.pipeline("fft_twiddle_mul_planar")?;
    queue.dispatch_1d(
        &tw_pso,
        &[data_re, data_im, &l2_buf, &len_buf, &batch_buf],
        batch * len,
    )?;
    queue.wait_until_complete()?;

    // Pass 3: row FFTs — RECURSE. The l1 contiguous rows of length l2 across all
    // batch signals are batch*l1 contiguous length-l2 signals at data[0].
    fft_pass_planar(
        lib,
        queue,
        device,
        data_re,
        data_im,
        scratch_re,
        scratch_im,
        l2,
        batch * l1,
        depth + 1,
    )?;

    // Pass 4: per-signal transpose l1 x l2 -> l2 x l1 into scratch planes, then
    // copy scratch -> data. Transpose is NOT in-place, hence the separate planes.
    let tr_pso = lib.pipeline("fft_transpose_planar")?;
    queue.dispatch_1d(
        &tr_pso,
        &[
            data_re, data_im, scratch_re, scratch_im, &l1_buf, &l2_buf, &batch_buf,
        ],
        batch * len,
    )?;
    queue.wait_until_complete()?;

    let count_buf = device.new_buffer_from_bytes(&((batch * len) as u32).to_le_bytes())?;
    let copy_pso = lib.pipeline("fft_copy_planar")?;
    queue.dispatch_1d(
        &copy_pso,
        &[scratch_re, scratch_im, data_re, data_im, &count_buf],
        batch * len,
    )?;
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

/// Buffer-level core: GPU-pack a real input buffer (`re_buf`, `n` f32) into the
/// interleaved-complex `out_buf` (`2n` f32: `out[2i]=re[i]`, `out[2i+1]=0`).
/// Dispatches `fft_pack_real_to_interleaved` with `n` threads, commit+wait. The
/// host-slice [`dispatch_pack_real`] is a thin wrapper over this.
pub fn pack_real_buf(
    device: &MetalDevice,
    re_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    n: usize,
) -> Result<(), FftError> {
    let n_buf = make_n_buf(device, n)?;
    let lib = shared_library(device)?;
    let pso = lib.pipeline("fft_pack_real_to_interleaved")?;
    let mut queue = CommandQueue::new(device)?;
    // One thread per sample; dispatch_1d handles threadgroup sizing at runtime.
    queue.dispatch_1d(&pso, &[re_buf, out_buf, &n_buf], n)?;
    queue.wait_until_complete()?;
    Ok(())
}

/// Buffer-level core: GPU-pack separate real/imag input buffers (each `n` f32)
/// into the interleaved-complex `out_buf` (`2n` f32: `out[2i]=re[i]`,
/// `out[2i+1]=im[i]`). Dispatches `fft_pack_complex_to_interleaved`. The
/// host-slice [`dispatch_pack_complex`] is a thin wrapper over this.
pub fn pack_complex_buf(
    device: &MetalDevice,
    re_buf: &MetalBuffer,
    im_buf: &MetalBuffer,
    out_buf: &MetalBuffer,
    n: usize,
) -> Result<(), FftError> {
    let n_buf = make_n_buf(device, n)?;
    let lib = shared_library(device)?;
    let pso = lib.pipeline("fft_pack_complex_to_interleaved")?;
    let mut queue = CommandQueue::new(device)?;
    // One thread per sample; dispatch_1d handles threadgroup sizing at runtime.
    queue.dispatch_1d(&pso, &[re_buf, im_buf, out_buf, &n_buf], n)?;
    queue.wait_until_complete()?;
    Ok(())
}

/// Buffer-level core: GPU-unpack the interleaved-complex `in_buf` (`2n` f32)
/// into separate planar buffers `re_out_buf` / `im_out_buf` (each `n` f32:
/// `re_out[i]=in[2i]`, `im_out[i]=in[2i+1]`). Dispatches
/// `fft_unpack_interleaved_to_planar`. The host-slice [`dispatch_unpack`] is a
/// thin wrapper over this.
pub fn unpack_buf(
    device: &MetalDevice,
    in_buf: &MetalBuffer,
    re_out_buf: &MetalBuffer,
    im_out_buf: &MetalBuffer,
    n: usize,
) -> Result<(), FftError> {
    let n_buf = make_n_buf(device, n)?;
    let lib = shared_library(device)?;
    let pso = lib.pipeline("fft_unpack_interleaved_to_planar")?;
    let mut queue = CommandQueue::new(device)?;
    // One thread per sample; dispatch_1d handles threadgroup sizing at runtime.
    queue.dispatch_1d(&pso, &[in_buf, re_out_buf, im_out_buf, &n_buf], n)?;
    queue.wait_until_complete()?;
    Ok(())
}

/// On-device 1-D FFT: GPU-pack the real (or real+imag) input into interleaved
/// complex, run [`fft_gpu_buf`], GPU-unpack into planar `(re, im)`. Only the
/// input stage and the two planar readbacks cross the host boundary — no CPU
/// interleave/split, no intermediate interleaved host `Vec`. Returns
/// `(re_out, im_out)`, each length `n`.
///
/// M5b-2 finding: this does NOT beat the host interleave/split path at the
/// engine level. Pack and unpack each pay a full command-buffer submit +
/// `wait_until_complete` barrier + buffer alloc (~13ms / ~21ms at 2^23 / 2^24),
/// which is 2-3x SLOWER than the cache-friendly CPU scatter/gather they replace
/// (~4-5ms / ~8-10ms), while the host input-stage + planar readback transfers
/// are unchanged. `fft_core` therefore stays on the host path; this fn is
/// retained as the correct building block IF a future fully-fused
/// single-command-buffer pack->fft->unpack pipeline (no intermediate barriers)
/// is built. See the `fft_core` comment in
/// `crates/polars-metal-core/src/fft.rs`.
pub fn fft_gpu_planar(
    device: &MetalDevice,
    re: &[f32],
    im: Option<&[f32]>,
    n: i64,
    inverse: bool,
) -> Result<(Vec<f32>, Vec<f32>), FftError> {
    if n <= 0 {
        return Err(FftError::Unsupported(n));
    }
    let nn = n as usize;
    let inter_buf = device.new_buffer_zeroed(2 * nn * std::mem::size_of::<f32>())?;
    match im {
        None => {
            let re_buf = MetalBuffer::from_f32_slice(device, re)?;
            pack_real_buf(device, &re_buf, &inter_buf, nn)?;
        }
        Some(imag) => {
            let re_buf = MetalBuffer::from_f32_slice(device, re)?;
            let im_buf = MetalBuffer::from_f32_slice(device, imag)?;
            pack_complex_buf(device, &re_buf, &im_buf, &inter_buf, nn)?;
        }
    }
    let out_buf = device.new_buffer_zeroed(2 * nn * std::mem::size_of::<f32>())?;
    fft_gpu_buf(device, &inter_buf, &out_buf, n, inverse)?;
    let re_out_buf = device.new_buffer_zeroed(nn * std::mem::size_of::<f32>())?;
    let im_out_buf = device.new_buffer_zeroed(nn * std::mem::size_of::<f32>())?;
    unpack_buf(device, &out_buf, &re_out_buf, &im_out_buf, nn)?;
    Ok((re_out_buf.to_f32_vec(), im_out_buf.to_f32_vec()))
}

/// Pack a real host signal into an interleaved-complex host `Vec` (length `2n`)
/// on the GPU. Each element `i` of the output satisfies `out[2i] = re[i]` and
/// `out[2i+1] = 0`. Thin host-slice wrapper over [`pack_real_buf`].
pub fn dispatch_pack_real(
    device: &MetalDevice,
    re: &[f32],
    n: usize,
) -> Result<Vec<f32>, FftError> {
    let in_buf = MetalBuffer::from_f32_slice(device, re)?;
    let out_buf = device.new_buffer_zeroed(2 * n * std::mem::size_of::<f32>())?;
    pack_real_buf(device, &in_buf, &out_buf, n)?;
    Ok(out_buf.to_f32_vec())
}

/// Pack separate real and imaginary host planes into an interleaved-complex host
/// `Vec` (length `2n`) on the GPU. Each element `i` of the output satisfies
/// `out[2i] = re[i]` and `out[2i+1] = im[i]`. Thin host-slice wrapper over
/// [`pack_complex_buf`].
pub fn dispatch_pack_complex(
    device: &MetalDevice,
    re: &[f32],
    im: &[f32],
    n: usize,
) -> Result<Vec<f32>, FftError> {
    let re_buf = MetalBuffer::from_f32_slice(device, re)?;
    let im_buf = MetalBuffer::from_f32_slice(device, im)?;
    let out_buf = device.new_buffer_zeroed(2 * n * std::mem::size_of::<f32>())?;
    pack_complex_buf(device, &re_buf, &im_buf, &out_buf, n)?;
    Ok(out_buf.to_f32_vec())
}

/// Unpack an interleaved-complex host slice (length `2n`) into `(re_out,
/// im_out)` host `Vec`s on the GPU. Each element `i` satisfies
/// `re_out[i] = inter[2i]` and `im_out[i] = inter[2i+1]`. Thin host-slice
/// wrapper over [`unpack_buf`].
pub fn dispatch_unpack(
    device: &MetalDevice,
    inter: &[f32],
    n: usize,
) -> Result<(Vec<f32>, Vec<f32>), FftError> {
    let in_buf = MetalBuffer::from_f32_slice(device, inter)?;
    let re_out_buf = device.new_buffer_zeroed(n * std::mem::size_of::<f32>())?;
    let im_out_buf = device.new_buffer_zeroed(n * std::mem::size_of::<f32>())?;
    unpack_buf(device, &in_buf, &re_out_buf, &im_out_buf, n)?;
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
