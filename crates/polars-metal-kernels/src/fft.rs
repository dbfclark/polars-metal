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
    if n <= 0 || (n & (n - 1)) != 0 || n > FFT_BASE_MAX {
        return Err(FftError::Unsupported(n));
    }
    let in_buf = MetalBuffer::from_f32_slice(device, input)?;
    let out_buf = device.new_buffer_zeroed((2 * n as usize) * std::mem::size_of::<f32>())?;
    let n_buf = device.new_buffer_from_bytes(&(n as u32).to_le_bytes())?;
    let inv_buf = device.new_buffer_from_bytes(&u32::from(inverse).to_le_bytes())?;

    let lib = shared_library(device)?;
    let pso = lib.pipeline("fft_stockham_pow2_f32")?;
    let mut queue = CommandQueue::new(device)?;
    // This kernel REQUIRES exactly one threadgroup: all log2(n) butterfly
    // stages run in per-threadgroup memory, so a second threadgroup would
    // transform uninitialized data. dispatch_1d_with_tg sets grid_width =
    // tg_width = tg here, and n <= FFT_BASE_MAX (1024) <= the PSO's
    // maxTotalThreadsPerThreadgroup on Apple Silicon, so the width clamp
    // never splits this into multiple threadgroups. The kernel strides by
    // tg_size to cover all n points.
    let tg = (n as usize).min(FFT_BASE_MAX as usize);
    queue.dispatch_1d_with_tg(&pso, &[&in_buf, &out_buf, &n_buf, &inv_buf], tg, tg)?;
    queue.wait_until_complete()?;
    Ok(out_buf.to_f32_vec())
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
