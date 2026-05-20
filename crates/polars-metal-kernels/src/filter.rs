//! Filter compaction kernel wrappers.
//!
//! M1 Phase 5 — Pass 1: predicate evaluation. Reads a bit-packed boolean
//! column plus its validity bitmap and writes a dense `u8[n_rows]` where
//! each byte is exactly `1` (keep this row) or `0` (drop it). The dense
//! u8 form is the input to MLX cumsum, which produces the scatter indices
//! consumed by the per-dtype scatter kernels (Tasks 11-13).
//!
//! Subsequent passes — `filter_scatter_i64`, `filter_scatter_f64`,
//! `filter_scatter_bool` — land in the same module.

use crate::command::{CommandQueue, DispatchError};
use crate::shader_lib::{shared_library, ShaderError};
use polars_metal_buffer::{BufferError, MetalDevice};

/// Errors raised by the filter-compaction kernel dispatchers.
#[derive(Debug, thiserror::Error)]
pub enum FilterError {
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
    /// One of the bit-packed input buffers is shorter than
    /// `ceil(n_rows / 8)`.
    #[error(
        "input length mismatch: predicate buffer is {pred_bytes} B, expected \
         at least {min_bytes} B for {n_rows} rows"
    )]
    InputLengthMismatch {
        pred_bytes: usize,
        min_bytes: usize,
        n_rows: usize,
    },
    /// The output slice is not exactly `n_rows` bytes long.
    #[error("output length mismatch: keep_flags={got}, expected {expected}")]
    OutputLengthMismatch { got: usize, expected: usize },
    /// `n_rows` exceeds `u32::MAX`. The kernel's grid size and the
    /// `n_rows` constant are both `u32`; refuse outsized inputs at the
    /// boundary rather than truncating silently.
    #[error("n_rows {n_rows} exceeds u32::MAX")]
    RowCountOverflow { n_rows: usize },
}

/// Dispatch `filter_predicate_to_u8` — pass 1 of the filter compaction
/// pipeline.
///
/// `pred_data` and `pred_valid` are bit-packed (Arrow validity format):
/// each row's bit lives at byte `row / 8`, bit `row % 8`. `n_rows` is the
/// number of rows under consideration; both input buffers must contain at
/// least `ceil(n_rows / 8)` bytes. `out` must be exactly `n_rows` bytes
/// long; each output byte is `1` iff the row's predicate bit is set AND
/// its validity bit is set, else `0`.
///
/// The function returns once the kernel has completed on the GPU and the
/// output bytes have been copied back into `out`. It waits on the command
/// queue internally; callers must NOT issue further dispatches before
/// this call returns.
///
/// `n_rows == 0` is accepted as a no-op: the function returns `Ok(())`
/// without touching Metal (Metal rejects zero-byte buffers and zero-grid
/// dispatches; both are caught here).
pub fn dispatch_predicate_to_u8(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    pred_data: &[u8],
    pred_valid: &[u8],
    n_rows: usize,
    out: &mut [u8],
) -> Result<(), FilterError> {
    if out.len() != n_rows {
        return Err(FilterError::OutputLengthMismatch {
            got: out.len(),
            expected: n_rows,
        });
    }
    if n_rows == 0 {
        return Ok(());
    }

    let min_bytes = (n_rows + 7) / 8;
    if pred_data.len() < min_bytes {
        return Err(FilterError::InputLengthMismatch {
            pred_bytes: pred_data.len(),
            min_bytes,
            n_rows,
        });
    }
    if pred_valid.len() < min_bytes {
        return Err(FilterError::InputLengthMismatch {
            pred_bytes: pred_valid.len(),
            min_bytes,
            n_rows,
        });
    }

    let n: u32 = u32::try_from(n_rows).map_err(|_| FilterError::RowCountOverflow { n_rows })?;

    let lib = shared_library(device)?;
    let pso = lib.pipeline("filter_predicate_to_u8")?;

    // The metallib reads `pred_data`/`pred_valid` from
    // `ceil(n_rows / 8)` bytes; allocate exactly that to keep the
    // device side honest, copying the trimmed prefix from the caller's
    // slice.
    let in_data = device.new_buffer_from_bytes(&pred_data[..min_bytes])?;
    let in_valid = device.new_buffer_from_bytes(&pred_valid[..min_bytes])?;
    let out_buf = device.new_buffer_zeroed(n_rows)?;
    let n_buf = device.new_buffer_from_bytes(&n.to_le_bytes())?;

    queue.dispatch_1d(&pso, &[&in_data, &in_valid, &out_buf, &n_buf], n_rows)?;
    queue.wait_until_complete()?;

    out.copy_from_slice(&out_buf.as_slice()[..n_rows]);
    Ok(())
}
