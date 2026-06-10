//! Gregorian civil-from-days kernel dispatcher (M6 B3).
//!
//! Element-wise extraction of year / month / day from an Int32 column of
//! days-since-1970. `Date` columns feed their physical i32 directly; the host
//! converts `Datetime` to days (floor-div by units-per-day) before dispatch.
//!
//! The kernel (`shaders/dt_gregorian.metal`) computes every field in Int32;
//! the host narrows month/day to Int8 and restores the validity bitmap.

use crate::command::{CommandQueue, DispatchError};
use crate::shader_lib::{shared_library, ShaderError};
use polars_metal_buffer::{BufferError, MetalBuffer, MetalDevice};

/// Threadgroup width — kept in sync with `TG_SIZE` in `shaders/dt_gregorian.metal`.
pub const TG_SIZE: usize = 256;

/// Which gregorian field the kernel extracts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtField {
    Year,
    Month,
    Day,
}

impl DtField {
    /// Scalar selector passed to the kernel (`buffer(3)`).
    fn code(self) -> u32 {
        match self {
            DtField::Year => 0,
            DtField::Month => 1,
            DtField::Day => 2,
        }
    }
}

/// Errors raised by the gregorian kernel dispatcher.
#[derive(Debug, thiserror::Error)]
pub enum DtError {
    #[error("shader library: {0}")]
    Shader(#[from] ShaderError),
    #[error("dispatch: {0}")]
    Dispatch(#[from] DispatchError),
    #[error("buffer: {0}")]
    Buffer(#[from] BufferError),
    #[error("output length {got} does not match input length {expected}")]
    OutputLengthMismatch { got: usize, expected: usize },
    #[error("n_rows {n_rows} exceeds u32::MAX")]
    RowCountOverflow { n_rows: usize },
}

/// Core dispatch over pre-staged Int32 `MetalBuffer`s (zero-copy when the
/// caller staged via `MetalBuffer::from_borrowed_i32` on page-aligned
/// memory). `input` and `output` are length-`n` Int32 buffers; `output` is
/// written in place. This is the path the PyO3 `execute_dt` binding uses.
///
/// ## Caller contract
/// - `input` holds exactly `n * 4` bytes of Int32 days-since-1970.
/// - `output` holds exactly `n * 4` bytes; fully overwritten.
/// - `n <= u32::MAX` (enforced).
///
/// ## n == 0
/// No-op; Metal rejects zero-byte buffers / zero-grid dispatches, so this is
/// handled on the host without touching Metal.
pub fn dispatch_dt_field_buf(
    device: &MetalDevice,
    input: &MetalBuffer,
    output: &MetalBuffer,
    n: u32,
    field: DtField,
) -> Result<(), DtError> {
    if n == 0 {
        return Ok(());
    }
    let lib = shared_library(device)?;
    let pso = lib.pipeline("dt_field_from_days")?;

    let n_buf = device.new_buffer_from_bytes(&n.to_le_bytes())?;
    let field_buf = device.new_buffer_from_bytes(&field.code().to_le_bytes())?;

    let n_padded = (n as usize).div_ceil(TG_SIZE) * TG_SIZE;
    let mut queue = CommandQueue::new(device)?;
    queue.dispatch_1d_with_tg(
        &pso,
        &[input, output, &n_buf, &field_buf],
        n_padded,
        TG_SIZE,
    )?;
    queue.wait_until_complete()?;
    Ok(())
}

/// Test-ergonomics wrapper: stages the caller's slices into Metal buffers,
/// calls [`dispatch_dt_field_buf`], copies the result back. For the zero-copy
/// PyO3 path call [`dispatch_dt_field_buf`] directly with pre-staged buffers.
///
/// ## Caller contract
/// - `output.len() == input.len()`.
/// - `n <= u32::MAX` (enforced).
///
/// ## n == 0
/// No-op; `Ok(())` without touching Metal.
pub fn dispatch_dt_field(
    device: &MetalDevice,
    input: &[i32],
    output: &mut [i32],
    field: DtField,
) -> Result<(), DtError> {
    let n = input.len();
    if output.len() != n {
        return Err(DtError::OutputLengthMismatch {
            got: output.len(),
            expected: n,
        });
    }
    if n == 0 {
        return Ok(());
    }
    let n_u32 = u32::try_from(n).map_err(|_| DtError::RowCountOverflow { n_rows: n })?;

    let input_buf = MetalBuffer::from_i32_slice(device, input)?;
    let output_buf = device.new_buffer_zeroed(std::mem::size_of_val(output))?;

    dispatch_dt_field_buf(device, &input_buf, &output_buf, n_u32, field)?;

    let out_vec = output_buf.to_i32_vec();
    output.copy_from_slice(&out_vec[..n]);
    Ok(())
}
