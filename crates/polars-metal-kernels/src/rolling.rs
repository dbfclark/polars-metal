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
use polars_metal_buffer::{BufferError, MetalBuffer, MetalDevice};

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

/// Core dispatch over pre-staged MetalBuffers (zero-copy when the caller
/// staged via `MetalBuffer::from_borrowed_f32` on page-aligned memory).
/// `input` and `output` are length-`n` F32 buffers; `output` is written in
/// place. This is the path the PyO3 `execute_rolling` binding uses.
///
/// ## Caller contract
///
/// - `input` must hold exactly `n * 4` bytes of F32 data.
/// - `output` must hold exactly `n * 4` bytes; it is fully overwritten by
///   the kernel (the first `w-1` positions are zero-filled as structural
///   nulls; the caller must set the Arrow validity bitmap accordingly).
/// - `1 <= w <= MAX_W` (4096). Larger windows must use CPU; this function
///   returns [`RollingError::WindowOutOfRange`].
/// - `n <= u32::MAX` (enforced).
///
/// ## n == 0
///
/// Zero-row input is a no-op. Metal rejects zero-byte buffers and zero-grid
/// dispatches; both are caught here and `Ok(())` is returned without
/// touching Metal. The `input` and `output` buffers are not accessed.
pub fn dispatch_rolling_sum_f32_buf(
    device: &MetalDevice,
    input: &MetalBuffer,
    output: &MetalBuffer,
    n: u32,
    w: u32,
    is_mean: bool,
) -> Result<(), RollingError> {
    let w_usize = w as usize;
    if w == 0 || w_usize > MAX_W {
        return Err(RollingError::WindowOutOfRange { w: w_usize });
    }
    if n == 0 {
        // Nothing to compute; Metal rejects zero-byte buffers and zero-grid
        // dispatches, so handle this case entirely on the host.
        return Ok(());
    }

    let is_mean_u32: u32 = u32::from(is_mean);

    let lib = shared_library(device)?;
    let pso = lib.pipeline("rolling_sum_f32")?;

    // Scalar arguments: each is a 1-element MetalBuffer bound as
    // `constant uint& x [[buffer(k)]]`. Little-endian encoding matches the
    // ARM host and the GPU (both little-endian on Apple Silicon).
    let n_buf = device.new_buffer_from_bytes(&n.to_le_bytes())?;
    let w_buf = device.new_buffer_from_bytes(&w.to_le_bytes())?;
    let is_mean_buf = device.new_buffer_from_bytes(&is_mean_u32.to_le_bytes())?;

    // Round n up to a multiple of TG_SIZE so every threadgroup is fully
    // populated. `dispatch_1d_with_tg` issues `dispatchThreads:`, whose
    // trailing threadgroup is partial (it does NOT pad the grid to a full
    // threadgroup). The tile-blocked kernel relies on cooperative loading:
    // each thread owns one or more tile slots at stride TG_SIZE. A partial
    // trailing threadgroup leaves tile slots [n%TG_SIZE .. TG_SIZE) unwritten,
    // producing undefined halo values for active threads' windows.
    // Rounding n_padded up to a multiple of TG_SIZE keeps every threadgroup
    // full so all cooperative tile slots are written. Surplus threads
    // (gid >= n) still load tile data, then bail via the kernel's
    // `gid >= n` guard before writing output.
    let n_padded = (n as usize).div_ceil(TG_SIZE) * TG_SIZE;

    let mut queue = CommandQueue::new(device)?;
    queue.dispatch_1d_with_tg(
        &pso,
        &[input, output, &n_buf, &w_buf, &is_mean_buf],
        n_padded,
        TG_SIZE,
    )?;
    queue.wait_until_complete()?;

    Ok(())
}

/// Dispatch `rolling_sum_f32` (sum) or mean (when `is_mean` is `true`) over a
/// 1-D F32 column.
///
/// This is a **test-ergonomics wrapper** that stages the caller's slices into
/// Metal buffers, calls [`dispatch_rolling_sum_f32_buf`], and copies the
/// result back. For the zero-copy path used by the PyO3 `execute_rolling`
/// binding, call [`dispatch_rolling_sum_f32_buf`] directly with pre-staged
/// [`MetalBuffer`]s.
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
    // Range and zero-row checks are also enforced in `dispatch_rolling_sum_f32_buf`,
    // but we validate here too so that errors reference the slice-level caller.
    if w == 0 || w > MAX_W {
        return Err(RollingError::WindowOutOfRange { w });
    }
    if n == 0 {
        return Ok(());
    }

    let n_u32 = u32::try_from(n).map_err(|_| RollingError::RowCountOverflow { n_rows: n })?;
    // w <= MAX_W <= 4096 <= u32::MAX, so this conversion always succeeds.
    // Using map_err rather than expect to satisfy the no-expect-in-non-test
    // workspace lint.
    let w_u32 = u32::try_from(w).map_err(|_| RollingError::WindowOutOfRange { w })?;

    // Stage the input into a Metal buffer (copy; the slice may not be
    // page-aligned, and this wrapper is for test ergonomics only).
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

    dispatch_rolling_sum_f32_buf(device, &input_buf, &output_buf, n_u32, w_u32, is_mean)?;

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

/// Core dispatch for windowed variance/std over pre-staged MetalBuffers
/// (zero-copy when staged via `MetalBuffer::from_borrowed_f32` on
/// page-aligned memory). `is_std` takes the square root of the variance
/// before writing each output element.
///
/// ## Caller contract
///
/// - `input` must hold exactly `n * 4` bytes of F32 data.
/// - `output` must hold exactly `n * 4` bytes; it is fully overwritten by
///   the kernel (the first `w-1` positions are zero-filled as structural
///   nulls; the caller must set the Arrow validity bitmap accordingly).
/// - `1 <= w <= MAX_W` (4096). Larger windows must use CPU; this function
///   returns [`RollingError::WindowOutOfRange`].
/// - `w > ddof` — the kernel computes `ss / (w - ddof)`; `w <= ddof` would
///   divide by zero or produce nonsensical results. Detection / rejection of
///   that case is the caller's responsibility.
/// - `n <= u32::MAX` (enforced).
///
/// ## n == 0
///
/// Zero-row input is a no-op; `Ok(())` is returned without touching Metal.
pub fn dispatch_rolling_var_f32_buf(
    device: &MetalDevice,
    input: &MetalBuffer,
    output: &MetalBuffer,
    n: u32,
    w: u32,
    ddof: u32,
    is_std: bool,
) -> Result<(), RollingError> {
    let w_usize = w as usize;
    if w == 0 || w_usize > MAX_W {
        return Err(RollingError::WindowOutOfRange { w: w_usize });
    }
    if n == 0 {
        return Ok(());
    }

    let is_std_u32: u32 = u32::from(is_std);

    let lib = shared_library(device)?;
    let pso = lib.pipeline("rolling_var_f32")?;

    // Scalar arguments: each is a 1-element MetalBuffer bound as
    // `constant uint& x [[buffer(k)]]`. Little-endian, matching the ARM host
    // and the GPU on Apple Silicon.
    let n_buf = device.new_buffer_from_bytes(&n.to_le_bytes())?;
    let w_buf = device.new_buffer_from_bytes(&w.to_le_bytes())?;
    let ddof_buf = device.new_buffer_from_bytes(&ddof.to_le_bytes())?;
    let is_std_buf = device.new_buffer_from_bytes(&is_std_u32.to_le_bytes())?;

    // Round n up to a multiple of TG_SIZE — same rationale as
    // `dispatch_rolling_sum_f32_buf`: the tile-blocked kernel relies on
    // cooperative loading, so every threadgroup must be fully populated.
    // Surplus threads (gid >= n) still participate in the tile load, then
    // exit before writing output via the kernel's `gid >= n` guard.
    let n_padded = (n as usize).div_ceil(TG_SIZE) * TG_SIZE;

    let mut queue = CommandQueue::new(device)?;
    queue.dispatch_1d_with_tg(
        &pso,
        &[input, output, &n_buf, &w_buf, &ddof_buf, &is_std_buf],
        n_padded,
        TG_SIZE,
    )?;
    queue.wait_until_complete()?;

    Ok(())
}

/// Dispatch `rolling_var_f32` (variance) or std (when `is_std` is `true`)
/// over a 1-D F32 column, using the centered two-pass algorithm (ddof=1 for
/// Polars sample-variance default).
///
/// This is a **test-ergonomics wrapper** that stages the caller's slices into
/// Metal buffers, calls [`dispatch_rolling_var_f32_buf`], and copies the
/// result back. For the zero-copy path used by the PyO3 binding, call
/// [`dispatch_rolling_var_f32_buf`] directly with pre-staged
/// [`MetalBuffer`]s.
///
/// The first `w-1` output elements are zero-filled (structural nulls); the
/// caller must set the Arrow validity bitmap to null for those positions.
///
/// ## Caller contract
///
/// - `output.len() == input.len()`.
/// - `1 <= w <= MAX_W` (4096). Larger windows must use CPU; returns
///   [`RollingError::WindowOutOfRange`].
/// - `w > ddof` (caller's responsibility; typically `ddof=1`, so `w >= 2`).
/// - `n <= u32::MAX` (enforced).
///
/// ## n == 0
///
/// Zero-row input is a no-op; `Ok(())` is returned without touching Metal.
///
/// ## Performance notes
///
/// The kernel is O(w) per output (two passes over the tile window). This is
/// appropriate for small-to-medium windows. For `w > 4096`, fall back to CPU.
pub fn dispatch_rolling_var_f32(
    device: &MetalDevice,
    input: &[f32],
    output: &mut [f32],
    w: usize,
    ddof: usize,
    is_std: bool,
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
        return Ok(());
    }

    let n_u32 = u32::try_from(n).map_err(|_| RollingError::RowCountOverflow { n_rows: n })?;
    // w <= MAX_W <= 4096 <= u32::MAX, so this conversion always succeeds.
    let w_u32 = u32::try_from(w).map_err(|_| RollingError::WindowOutOfRange { w })?;
    // ddof is typically 1; bounded by w which is <= u32::MAX.
    let ddof_u32 = u32::try_from(ddof).map_err(|_| RollingError::WindowOutOfRange { w: ddof })?;

    // Stage the input into a Metal buffer (copy; the slice may not be
    // page-aligned, and this wrapper is for test ergonomics only).
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

    dispatch_rolling_var_f32_buf(
        device,
        &input_buf,
        &output_buf,
        n_u32,
        w_u32,
        ddof_u32,
        is_std,
    )?;

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
