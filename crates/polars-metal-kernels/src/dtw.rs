//! Banded Euclidean DTW kernel dispatcher (M6 A4).
//!
//! One threadgroup per query pair; the L*L DP runs in threadgroup memory.
//! See `shaders/dtw.metal` for the kernel and its threadgroup/grid assumptions.

use crate::command::{CommandQueue, DispatchError};
use crate::shader_lib::{shared_library, ShaderError};
use polars_metal_buffer::{BufferError, MetalBuffer, MetalDevice};

/// Threads per threadgroup (cooperative load; thread 0 runs the DP). Kept in
/// sync with the dispatch in this file (the kernel works for any TG_SIZE).
pub const TG_SIZE: usize = 32;

/// Maximum supported sequence length — kept in sync with `MAX_L` in
/// `shaders/dtw.metal`. Longer sequences must use CPU.
pub const MAX_L: usize = 1024;

/// Errors raised by the DTW kernel dispatchers.
#[derive(Debug, thiserror::Error)]
pub enum DtwError {
    /// Failure loading the metallib or building the pipeline state.
    #[error("shader library: {0}")]
    Shader(#[from] ShaderError),
    /// Failure dispatching the kernel onto the command queue.
    #[error("dispatch: {0}")]
    Dispatch(#[from] DispatchError),
    /// Failure allocating a Metal buffer.
    #[error("buffer: {0}")]
    Buffer(#[from] BufferError),
    /// `seq_len` is zero or exceeds `MAX_L`; the kernel's threadgroup memory is
    /// sized for `MAX_L`, so longer sequences must fall back to CPU.
    #[error("sequence length {l} out of range 1..={MAX_L}")]
    SeqLenOutOfRange { l: usize },
    /// `queries.len()` does not equal `n_pairs * L`.
    #[error("queries length {got} is not n_pairs*L = {expected}")]
    QueriesLenMismatch { got: usize, expected: usize },
    /// `reference.len()` does not equal `L`.
    #[error("reference length {got} does not match L = {expected}")]
    ReferenceLenMismatch { got: usize, expected: usize },
    /// `output.len()` does not equal `n_pairs`.
    #[error("output length {got} does not match n_pairs = {expected}")]
    OutputLenMismatch { got: usize, expected: usize },
    /// `n_pairs` exceeds `u32::MAX`; the kernel's grid and scalar argument are
    /// both `uint`, so refuse outsized inputs rather than truncating silently.
    #[error("n_pairs {n} exceeds u32::MAX")]
    PairCountOverflow { n: usize },
}

/// Core dispatch over pre-staged buffers (zero-copy when staged via
/// `from_borrowed_f32` on page-aligned memory). `queries` = n_pairs*L f32
/// (pair-major), `reference` = L f32, `output` = n_pairs f32 (overwritten).
/// `window < 0` => unconstrained DTW; else Sakoe-Chiba radius (|i-j| <= window).
///
/// ## n_pairs == 0
/// No-op (Metal rejects zero-grid dispatch); returns Ok without touching Metal.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_dtw_buf(
    device: &MetalDevice,
    queries: &MetalBuffer,
    reference: &MetalBuffer,
    output: &MetalBuffer,
    n_pairs: u32,
    seq_len: u32,
    window: i32,
) -> Result<(), DtwError> {
    let l = seq_len as usize;
    if seq_len == 0 || l > MAX_L {
        return Err(DtwError::SeqLenOutOfRange { l });
    }
    if n_pairs == 0 {
        return Ok(());
    }
    let lib = shared_library(device)?;
    let pso = lib.pipeline("dtw_banded")?;
    let n_buf = device.new_buffer_from_bytes(&n_pairs.to_le_bytes())?;
    let l_buf = device.new_buffer_from_bytes(&seq_len.to_le_bytes())?;
    let w_buf = device.new_buffer_from_bytes(&window.to_le_bytes())?;

    // One threadgroup per pair: dispatch n_pairs*TG_SIZE threads in groups of
    // TG_SIZE, so threadgroup_position_in_grid == pair index.
    let total_threads = (n_pairs as usize) * TG_SIZE;
    let mut queue = CommandQueue::new(device)?;
    queue.dispatch_1d_with_tg(
        &pso,
        &[queries, reference, output, &n_buf, &l_buf, &w_buf],
        total_threads,
        TG_SIZE,
    )?;
    queue.wait_until_complete()?;
    Ok(())
}

/// Test-ergonomics wrapper: stages slices, dispatches, copies the result back.
/// For the zero-copy PyO3 path call `dispatch_dtw_buf` with pre-staged buffers.
pub fn dispatch_dtw(
    device: &MetalDevice,
    queries: &[f32],
    reference: &[f32],
    output: &mut [f32],
    seq_len: usize,
    window: i32,
) -> Result<(), DtwError> {
    if seq_len == 0 || seq_len > MAX_L {
        return Err(DtwError::SeqLenOutOfRange { l: seq_len });
    }
    if reference.len() != seq_len {
        return Err(DtwError::ReferenceLenMismatch {
            got: reference.len(),
            expected: seq_len,
        });
    }
    let n_pairs = output.len();
    if queries.len() != n_pairs * seq_len {
        return Err(DtwError::QueriesLenMismatch {
            got: queries.len(),
            expected: n_pairs * seq_len,
        });
    }
    if n_pairs == 0 {
        return Ok(());
    }
    let n_u32 = u32::try_from(n_pairs).map_err(|_| DtwError::PairCountOverflow { n: n_pairs })?;
    // seq_len <= MAX_L <= u32::MAX.
    let l_u32 = u32::try_from(seq_len).map_err(|_| DtwError::SeqLenOutOfRange { l: seq_len })?;

    // SAFETY: slices are alive for the call; f32 has no invalid bit patterns,
    // so reinterpreting as [u8] is sound.
    let q_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            queries.as_ptr() as *const u8,
            std::mem::size_of_val(queries),
        )
    };
    let r_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            reference.as_ptr() as *const u8,
            std::mem::size_of_val(reference),
        )
    };
    let q_buf = device.new_buffer_from_bytes(q_bytes)?;
    let r_buf = device.new_buffer_from_bytes(r_bytes)?;
    let out_buf = device.new_buffer_zeroed(std::mem::size_of_val(output))?;

    dispatch_dtw_buf(device, &q_buf, &r_buf, &out_buf, n_u32, l_u32, window)?;

    // SAFETY: out_buf holds exactly n_pairs*4 bytes; Metal Shared allocations
    // are >=256-byte aligned; f32 has no invalid bit patterns.
    let out_bytes = &out_buf.as_slice()[..std::mem::size_of_val(output)];
    let out_f32: &[f32] =
        unsafe { std::slice::from_raw_parts(out_bytes.as_ptr() as *const f32, n_pairs) };
    output.copy_from_slice(out_f32);
    Ok(())
}
