//! Logical-kernel wrappers.
//!
//! M1 Phase 6 sibling to [`crate::cmp`] — the `bool_and` / `bool_or`
//! entry points in `shaders/logical_bool.metal`. Two entry points, one
//! per 3-valued operator.
//!
//! Polars uses 3-valued logic on nullable bool columns:
//!
//! - AND: false dominates. `false ∧ null = false (valid)`. Result is
//!   null only when both sides are not-false (true or null) AND at
//!   least one side is null.
//! - OR : true dominates. `true ∨ null = true (valid)`. Result is null
//!   only when both sides are not-true (false or null) AND at least one
//!   side is null.
//!
//! Each kernel reads two bit-packed bool columns + their bit-packed
//! validity bitmaps and writes a bit-packed bool result + its validity
//! bitmap. 8 output rows share one byte and multiple threads can race
//! the same byte, so the kernel uses atomic OR (same pattern as
//! `filter_scatter_bool` and the comparison kernels). Callers must
//! zero-initialise the output buffers and allocate them in multiples of
//! 4 bytes so the kernel's `device atomic_uint*` cast is well-aligned.

use crate::command::{CommandQueue, DispatchError};
use crate::shader_lib::{shared_library, ShaderError};
use polars_metal_buffer::{BufferError, MetalDevice};

/// Errors raised by the logical-kernel dispatchers.
#[derive(Debug, thiserror::Error)]
pub enum LogicalError {
    /// Failure loading the metallib or building the pipeline state.
    #[error("shader library: {0}")]
    Shader(#[from] ShaderError),
    /// Failure dispatching the kernel onto the command queue.
    #[error("dispatch: {0}")]
    Dispatch(#[from] DispatchError),
    /// Failure allocating a Metal buffer.
    #[error("buffer: {0}")]
    Buffer(#[from] BufferError),
    /// An input bit-packed buffer (data or validity) is shorter than
    /// `ceil(n_rows / 8)` bytes.
    #[error(
        "input buffer too short: got {got} bytes, need at least {min_bytes} for {n_rows} rows"
    )]
    InputTooShort {
        got: usize,
        min_bytes: usize,
        n_rows: usize,
    },
    /// An output buffer is shorter than the kernel's alignment-padded
    /// minimum (`ceil(n_rows / 8)` rounded up to 4 bytes, min 4).
    #[error("output buffer too short: got {got} bytes, need at least {min_bytes}")]
    OutputTooShort { got: usize, min_bytes: usize },
    /// `n_rows` exceeds `u32::MAX`. The kernel's grid size and `n_rows`
    /// constant are both `u32`; refuse outsized inputs at the boundary.
    #[error("n_rows {n_rows} exceeds u32::MAX")]
    RowCountOverflow { n_rows: usize },
}

/// Minimum bytes for a bit-packed output bitmap of `n_rows` rows.
/// Rounded up to 4 bytes so the kernel's `device atomic_uint*` cast is
/// well-aligned; minimum 4 bytes for the same reason. Mirrors the
/// helper in [`crate::cmp`].
fn out_min_bytes(n_rows: usize) -> usize {
    let raw = (n_rows + 7) / 8;
    let padded = (raw + 3) & !3;
    padded.max(4)
}

/// Bit-packed input minimum: `ceil(n_rows / 8)` bytes.
fn in_min_bytes(n_rows: usize) -> usize {
    (n_rows + 7) / 8
}

/// Dispatch the `bool_and` kernel.
///
/// Reads two nullable bit-packed bool columns (`lhs_data` + `lhs_valid`,
/// `rhs_data` + `rhs_valid`) and writes the 3-valued AND into
/// `out_data` + `out_valid`.
///
/// Caller contract:
///   - `lhs_data.len() >= ceil(n_rows / 8)`. Same for `lhs_valid`,
///     `rhs_data`, `rhs_valid`.
///   - `out_data.len() >= out_min_bytes(n_rows)`. Same for `out_valid`.
///   - The dispatcher allocates fresh device-side output buffers
///     (zeroed by `new_buffer_zeroed`) for the atomic OR and copies
///     them back over the caller's slices, so any pre-existing bits in
///     the caller's slices are overwritten.
///
/// `n_rows == 0` is a no-op (Metal rejects zero-byte buffers and
/// zero-grid dispatches; both are caught here). The output slices are
/// cleared so the "no kernel ran, no stale data" contract holds.
// 9 arguments — same shape as `dispatch_cmp_i64`: device + queue, 4 input
// slices, n_rows, 2 output slices. A struct wrapper would not improve
// readability; every argument maps 1:1 to a kernel binding or a
// host-side check input.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_bool_and(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    lhs_data: &[u8],
    lhs_valid: &[u8],
    rhs_data: &[u8],
    rhs_valid: &[u8],
    n_rows: usize,
    out_data: &mut [u8],
    out_valid: &mut [u8],
) -> Result<(), LogicalError> {
    dispatch_logical(
        device, queue, "bool_and", lhs_data, lhs_valid, rhs_data, rhs_valid, n_rows, out_data,
        out_valid,
    )
}

/// Dispatch the `bool_or` kernel.
///
/// Mirrors [`dispatch_bool_and`] for 3-valued OR (true dominates).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_bool_or(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    lhs_data: &[u8],
    lhs_valid: &[u8],
    rhs_data: &[u8],
    rhs_valid: &[u8],
    n_rows: usize,
    out_data: &mut [u8],
    out_valid: &mut [u8],
) -> Result<(), LogicalError> {
    dispatch_logical(
        device, queue, "bool_or", lhs_data, lhs_valid, rhs_data, rhs_valid, n_rows, out_data,
        out_valid,
    )
}

/// Shared dispatch path for both AND and OR.
///
/// The two entry points have identical buffer layouts (4 in-bitmaps, 2
/// out-bitmaps, `n_rows`), so all the host-side validation and the
/// device-buffer plumbing collapses into one function parameterised by
/// the kernel entry-point name. The kernel-internal logic differs in
/// the truth table only.
#[allow(clippy::too_many_arguments)]
fn dispatch_logical(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    entry_point: &str,
    lhs_data: &[u8],
    lhs_valid: &[u8],
    rhs_data: &[u8],
    rhs_valid: &[u8],
    n_rows: usize,
    out_data: &mut [u8],
    out_valid: &mut [u8],
) -> Result<(), LogicalError> {
    let min_in = in_min_bytes(n_rows);
    for (got_len, label_n) in [
        (lhs_data.len(), "lhs_data"),
        (lhs_valid.len(), "lhs_valid"),
        (rhs_data.len(), "rhs_data"),
        (rhs_valid.len(), "rhs_valid"),
    ] {
        let _ = label_n; // not threaded into the error; matches existing style
        if got_len < min_in {
            return Err(LogicalError::InputTooShort {
                got: got_len,
                min_bytes: min_in,
                n_rows,
            });
        }
    }
    let min_out = out_min_bytes(n_rows);
    if out_data.len() < min_out {
        return Err(LogicalError::OutputTooShort {
            got: out_data.len(),
            min_bytes: min_out,
        });
    }
    if out_valid.len() < min_out {
        return Err(LogicalError::OutputTooShort {
            got: out_valid.len(),
            min_bytes: min_out,
        });
    }

    if n_rows == 0 {
        // Mirror the "no kernel ran, no writes" contract used by the
        // comparison dispatchers: zero the caller's output prefix so we
        // don't leak data from a previous reuse.
        for b in &mut out_data[..min_out] {
            *b = 0;
        }
        for b in &mut out_valid[..min_out] {
            *b = 0;
        }
        return Ok(());
    }

    let n_rows_u32 =
        u32::try_from(n_rows).map_err(|_| LogicalError::RowCountOverflow { n_rows })?;

    let lib = shared_library(device)?;
    let pso = lib.pipeline(entry_point)?;

    // Inputs: copy only the first `min_in` bytes — anything beyond is
    // padding the caller may not have initialised.
    let lhs_data_buf = device.new_buffer_from_bytes(&lhs_data[..min_in])?;
    let lhs_valid_buf = device.new_buffer_from_bytes(&lhs_valid[..min_in])?;
    let rhs_data_buf = device.new_buffer_from_bytes(&rhs_data[..min_in])?;
    let rhs_valid_buf = device.new_buffer_from_bytes(&rhs_valid[..min_in])?;
    let out_data_buf = device.new_buffer_zeroed(min_out)?;
    let out_valid_buf = device.new_buffer_zeroed(min_out)?;
    let n_rows_buf = device.new_buffer_from_bytes(&n_rows_u32.to_le_bytes())?;

    queue.dispatch_1d(
        &pso,
        &[
            &lhs_data_buf,
            &lhs_valid_buf,
            &rhs_data_buf,
            &rhs_valid_buf,
            &out_data_buf,
            &out_valid_buf,
            &n_rows_buf,
        ],
        n_rows,
    )?;
    queue.wait_until_complete()?;

    out_data[..min_out].copy_from_slice(&out_data_buf.as_slice()[..min_out]);
    out_valid[..min_out].copy_from_slice(&out_valid_buf.as_slice()[..min_out]);
    Ok(())
}
