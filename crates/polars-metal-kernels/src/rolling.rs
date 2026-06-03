//! Rolling windowed-statistics kernel dispatchers (M5).
//!
//! Tile-blocked rolling sum/mean over F32 columns. Scalar parameters (n, w,
//! op flags) are passed as 1-element MetalBuffers, matching the
//! `constant uint& x [[buffer(k)]]` convention used across `shaders/`.
//!
//! The shader (`shaders/rolling.metal`) keeps per-threadgroup accumulation
//! magnitudes at ~w·mean rather than ~N·mean, avoiding the F32 cancellation
//! that the cumsum-diff approach would suffer for small windows over large N.
//!
//! The first `w-1` output elements are zero-filled (structural nulls). The
//! caller is responsible for setting the validity bitmap to null for those
//! positions before handing the result to Polars.

use crate::command::{CommandQueue, DispatchError};
use crate::shader_lib::{shared_library, ShaderError};
use polars_metal_buffer::{BufferError, MetalDevice};

/// Outputs per threadgroup — kept in sync with `TG_SIZE` in
/// `shaders/rolling.metal`.
pub const TG_SIZE: usize = 256;

/// Maximum supported window size — kept in sync with `MAX_W` in
/// `shaders/rolling.metal`. Windows larger than this must be dispatched on
/// CPU; the dispatcher returns [`RollingError::WindowOutOfRange`] if `w`
/// exceeds this value.
pub const MAX_W: usize = 4096;

/// Errors raised by the rolling-statistics kernel dispatchers.
#[derive(Debug, thiserror::Error)]
pub enum RollingError {
    /// Failure loading the metallib or building the pipeline state for the
    /// kernel.
    #[error("shader library: {0}")]
    Shader(#[from] ShaderError),
    /// Failure dispatching the kernel onto the command queue.
    #[error("dispatch: {0}")]
    Dispatch(#[from] DispatchError),
    /// Failure allocating a Metal buffer.
    #[error("buffer: {0}")]
    Buffer(#[from] BufferError),
    /// `w` is zero or exceeds `MAX_W`. The kernel's threadgroup memory is
    /// sized for `TG_SIZE + MAX_W` floats; larger windows require the O(N)
    /// prefix-scan optimisation (a later task) or CPU fallback.
    #[error("window {w} out of range 1..={MAX_W}")]
    WindowOutOfRange { w: usize },
    /// `output.len()` does not equal `input.len()`.
    #[error("output length {got} does not match input length {expected}")]
    OutputLengthMismatch { got: usize, expected: usize },
    /// `n_rows` exceeds `u32::MAX`. The kernel's grid and scalar argument
    /// are both `uint`; refuse outsized inputs at the boundary rather than
    /// truncating silently.
    #[error("n_rows {n_rows} exceeds u32::MAX")]
    RowCountOverflow { n_rows: usize },
}

/// Dispatch `rolling_sum_f32` (sum) or mean (when `is_mean` is `true`) over a
/// 1-D F32 column.
///
/// `input` is the source column (`n` F32 values). `output` receives the
/// kernel results (`n` F32 values). The first `w-1` output elements are
/// zero-filled (structural nulls); the caller must set the Arrow validity
/// bitmap to null for those positions before handing the result to Polars.
///
/// ## Caller contract
///
/// - `output.len() == input.len()`.
/// - `1 <= w <= MAX_W` (4096). Larger windows must use CPU; this function
///   returns [`RollingError::WindowOutOfRange`].
/// - `n <= u32::MAX` (enforced; values larger than ~4 billion rows should
///   use a streaming variant not yet implemented).
///
/// ## n == 0
///
/// Zero-row input is a no-op. Metal rejects zero-byte buffers and zero-grid
/// dispatches; both are caught here and `Ok(())` is returned without
/// touching Metal.
///
/// ## Performance notes
///
/// The kernel is O(w) per output — appropriate for small-to-medium windows.
/// The O(N) prefix-scan optimisation (large w) is a planned follow-up task.
/// For now, windows > 4096 are rejected; the caller should fall back to CPU.
pub fn dispatch_rolling_sum_f32(
    device: &MetalDevice,
    input: &[f32],
    output: &mut [f32],
    w: usize,
    is_mean: bool,
) -> Result<(), RollingError> {
    let n = input.len();

    if output.len() != n {
        return Err(RollingError::OutputLengthMismatch {
            got: output.len(),
            expected: n,
        });
    }
    if w == 0 || w > MAX_W {
        return Err(RollingError::WindowOutOfRange { w });
    }
    if n == 0 {
        // Nothing to compute; Metal rejects zero-byte buffers and zero-grid
        // dispatches, so handle this case entirely on the host.
        return Ok(());
    }

    let n_u32 = u32::try_from(n).map_err(|_| RollingError::RowCountOverflow { n_rows: n })?;
    // w <= MAX_W <= 4096 <= u32::MAX, so this conversion always succeeds.
    // Using map_err rather than expect to satisfy the no-expect-in-non-test
    // workspace lint.
    let w_u32 =
        u32::try_from(w).map_err(|_| RollingError::WindowOutOfRange { w })?;
    let is_mean_u32: u32 = u32::from(is_mean);

    let lib = shared_library(device)?;
    let pso = lib.pipeline("rolling_sum_f32")?;

    // Reinterpret the F32 input slice as bytes for `new_buffer_from_bytes`.
    // F32 has no invalid bit patterns; all bit patterns are valid IEEE 754
    // values. `new_buffer_from_bytes` copies into a freshly allocated
    // MTLBuffer synchronously; the slice does not need to outlive the call.
    //
    // SAFETY: `input` is alive for the duration of this call; its pointer is
    // non-null. The byte length `n * 4` fits in usize on all supported
    // targets (we just bounds-checked n against u32::MAX). `f32` has no
    // invalid bit patterns, so reinterpreting as `[u8]` is sound.
    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(input.as_ptr() as *const u8, std::mem::size_of_val(input))
    };
    let input_buf = device.new_buffer_from_bytes(input_bytes)?;

    // Output: allocate a zeroed device buffer. The kernel writes every
    // element, but zero-fill ensures the first w-1 structural-null positions
    // are deterministically 0.0.
    let output_buf = device.new_buffer_zeroed(std::mem::size_of_val(input))?;

    // Scalar arguments: each is a 1-element MetalBuffer bound as
    // `constant uint& x [[buffer(k)]]`. Little-endian encoding matches the
    // ARM host and the GPU (both little-endian on Apple Silicon).
    let n_buf = device.new_buffer_from_bytes(&n_u32.to_le_bytes())?;
    let w_buf = device.new_buffer_from_bytes(&w_u32.to_le_bytes())?;
    let is_mean_buf = device.new_buffer_from_bytes(&is_mean_u32.to_le_bytes())?;

    // Round n up to a multiple of TG_SIZE so every threadgroup is fully
    // populated. The tile-blocked kernel relies on cooperative loading: each
    // thread owns one or more tile slots at stride TG_SIZE. If the last
    // threadgroup is partial (n % TG_SIZE != 0) then the "padding" slots
    // [n%TG_SIZE .. TG_SIZE) are never filled, leaving tile elements for the
    // active threads' windows undefined. Dispatching n_padded threads ensures
    // every threadgroup is full; the `if (gid >= n) return;` guard in the
    // kernel skips output writes for the surplus threads (they still
    // participate in tile loading).
    let n_padded = n.div_ceil(TG_SIZE) * TG_SIZE;

    let mut queue = CommandQueue::new(device)?;
    queue.dispatch_1d_with_tg(
        &pso,
        &[&input_buf, &output_buf, &n_buf, &w_buf, &is_mean_buf],
        n_padded,
        TG_SIZE,
    )?;
    queue.wait_until_complete()?;

    // Copy kernel outputs back into the caller's slice. The GPU buffer is
    // exactly `n * 4` bytes; read that prefix as F32.
    //
    // SAFETY: `output_buf.as_slice()` returns a `&[u8]` of length `n * 4`.
    // The MTLBuffer contents pointer is at least 256-byte aligned (Metal
    // resource alignment guarantee on Apple Silicon), so casting to `*const
    // f32` is well-aligned. `f32` has no invalid bit patterns. The source
    // lives in the MTLBuffer, which is alive for the duration of this call.
    let out_bytes = &output_buf.as_slice()[..std::mem::size_of_val(output)];
    let out_f32: &[f32] =
        unsafe { std::slice::from_raw_parts(out_bytes.as_ptr() as *const f32, n) };
    output.copy_from_slice(out_f32);

    Ok(())
}
