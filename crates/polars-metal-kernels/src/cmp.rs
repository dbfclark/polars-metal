//! Comparison kernel wrappers.
//!
//! M1 Phase 6 — the i64 and f64 comparison kernels (`cmp_i64_*` and
//! `cmp_f64_*` in `shaders/cmp_i64.metal` and `shaders/cmp_f64.metal`).
//! Six ops × two variants (column-column and column-scalar) × two
//! dtypes = twenty-four entry points, generated from MSL macros.
//!
//! Each kernel reads one or two columns plus their validity bitmaps
//! and writes a bit-packed bool column plus its validity bitmap. The
//! output validity is `lhs_valid & rhs_valid` (column-column) or just
//! `lhs_valid` (column-scalar — the scalar is treated as always-valid).
//! The output data bit is set only where both inputs are valid AND the
//! comparison succeeds; null rows leave the output data bit at zero.
//!
//! Output writes use atomic OR because 8 rows share a byte and multiple
//! threads can race the same byte (same pattern as `filter_scatter_bool`
//! in Task 13). Callers must zero-initialise both output buffers and
//! allocate them in multiples of 4 bytes so the kernel's
//! `device atomic_uint*` cast is well-aligned.
//!
//! f64 NaN semantics (Polars-conformant, IEEE 754-ordered):
//!   - `NaN <op> x` is **false** for `==`, `<`, `<=`, `>`, `>=` (any x).
//!   - `NaN != x` is **true** (any x, including another NaN).
//!   - NaN's validity bit is unaffected — NaN is a valid f64 value.
//!
//! `cmp_f64.metal` implements these in integer arithmetic on the raw
//! 8-byte bit pattern because Apple Silicon MSL compute kernels do not
//! support the `double` type; see the kernel file's comment for the
//! total-order-key encoding.

use crate::command::{CommandQueue, DispatchError};
use crate::shader_lib::{shared_library, ShaderError};
use polars_metal_buffer::{BufferError, MetalDevice};

/// Comparison operator, one variant per MSL entry point.
///
/// Defined here rather than re-using the IR enum from `polars-metal-core`
/// to keep the dependency direction one-way (core depends on kernels, not
/// vice versa). `polars-metal-core` maps its IR `CompareOp` onto this enum
/// at dispatch time, just as it maps `MetalDtype`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CompareOp {
    /// Entry-point name for the column-column i64 kernel of this op.
    fn entry_point_i64_cc(self) -> &'static str {
        match self {
            CompareOp::Eq => "cmp_i64_eq",
            CompareOp::Ne => "cmp_i64_ne",
            CompareOp::Lt => "cmp_i64_lt",
            CompareOp::Le => "cmp_i64_le",
            CompareOp::Gt => "cmp_i64_gt",
            CompareOp::Ge => "cmp_i64_ge",
        }
    }

    /// Entry-point name for the column-scalar i64 kernel of this op.
    fn entry_point_i64_cs(self) -> &'static str {
        match self {
            CompareOp::Eq => "cmp_i64_eq_scalar",
            CompareOp::Ne => "cmp_i64_ne_scalar",
            CompareOp::Lt => "cmp_i64_lt_scalar",
            CompareOp::Le => "cmp_i64_le_scalar",
            CompareOp::Gt => "cmp_i64_gt_scalar",
            CompareOp::Ge => "cmp_i64_ge_scalar",
        }
    }

    /// Entry-point name for the column-column f64 kernel of this op.
    fn entry_point_f64_cc(self) -> &'static str {
        match self {
            CompareOp::Eq => "cmp_f64_eq",
            CompareOp::Ne => "cmp_f64_ne",
            CompareOp::Lt => "cmp_f64_lt",
            CompareOp::Le => "cmp_f64_le",
            CompareOp::Gt => "cmp_f64_gt",
            CompareOp::Ge => "cmp_f64_ge",
        }
    }

    /// Entry-point name for the column-scalar f64 kernel of this op.
    fn entry_point_f64_cs(self) -> &'static str {
        match self {
            CompareOp::Eq => "cmp_f64_eq_scalar",
            CompareOp::Ne => "cmp_f64_ne_scalar",
            CompareOp::Lt => "cmp_f64_lt_scalar",
            CompareOp::Le => "cmp_f64_le_scalar",
            CompareOp::Gt => "cmp_f64_gt_scalar",
            CompareOp::Ge => "cmp_f64_ge_scalar",
        }
    }
}

/// Errors raised by the comparison kernel dispatchers.
#[derive(Debug, thiserror::Error)]
pub enum CmpError {
    /// Failure loading the metallib or building the pipeline state.
    #[error("shader library: {0}")]
    Shader(#[from] ShaderError),
    /// Failure dispatching the kernel onto the command queue.
    #[error("dispatch: {0}")]
    Dispatch(#[from] DispatchError),
    /// Failure allocating a Metal buffer.
    #[error("buffer: {0}")]
    Buffer(#[from] BufferError),
    /// `lhs_data` and `rhs_data` have different lengths or do not match
    /// `n_rows`.
    #[error("input length mismatch: lhs={lhs}, rhs={rhs}, n_rows={n_rows}")]
    InputLengthMismatch {
        lhs: usize,
        rhs: usize,
        n_rows: usize,
    },
    /// A validity bitmap is shorter than `ceil(n_rows / 8)`.
    #[error(
        "validity buffer too short: got {got} bytes, need at least {min_bytes} for {n_rows} rows"
    )]
    ValidityTooShort {
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
/// well-aligned; minimum 4 bytes for the same reason.
fn out_min_bytes(n_rows: usize) -> usize {
    let raw = (n_rows + 7) / 8;
    let padded = (raw + 3) & !3;
    padded.max(4)
}

/// Dispatch a column-column i64 comparison.
///
/// Reads `lhs_data` and `rhs_data` (both `&[i64]` of length `n_rows`) plus
/// their validity bitmaps `lhs_valid` and `rhs_valid` (bit-packed,
/// `ceil(n_rows / 8)` bytes minimum). Writes a bit-packed bool column to
/// `out_data` and a bit-packed validity bitmap to `out_valid`.
///
/// Output semantics:
///   - `out_valid[i]` is set iff `lhs_valid[i] AND rhs_valid[i]`.
///   - `out_data[i]` is set iff `out_valid[i] AND (lhs[i] OP rhs[i])`.
///     Data bits at null rows are left at zero.
///
/// Caller contract:
///   - `lhs_data.len() == rhs_data.len() == n_rows`.
///   - `lhs_valid.len() >= ceil(n_rows / 8)`.
///   - `rhs_valid.len() >= ceil(n_rows / 8)`.
///   - `out_data.len() >= out_min_bytes(n_rows)`.
///   - `out_valid.len() >= out_min_bytes(n_rows)`.
///   - Both output slices SHOULD be zero-initialised on input; the
///     dispatcher allocates fresh device-side buffers (zeroed by
///     `new_buffer_zeroed`) for the atomic OR and copies them back over
///     the caller's slices, so any pre-existing bits in the caller's
///     slices are overwritten.
///
/// `n_rows == 0` is a no-op (Metal rejects zero-byte buffers and
/// zero-grid dispatches; both are caught here).
// 10 arguments — the device + queue pair plus 4 in-buffers, 2 out-buffers,
// `n_rows`, and the op selector. A struct wrapper would not improve
// readability; every argument maps 1:1 to a kernel binding or a host-side
// check input.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_cmp_i64(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    lhs_data: &[i64],
    lhs_valid: &[u8],
    rhs_data: &[i64],
    rhs_valid: &[u8],
    n_rows: usize,
    op: CompareOp,
    out_data: &mut [u8],
    out_valid: &mut [u8],
) -> Result<(), CmpError> {
    if lhs_data.len() != n_rows || rhs_data.len() != n_rows {
        return Err(CmpError::InputLengthMismatch {
            lhs: lhs_data.len(),
            rhs: rhs_data.len(),
            n_rows,
        });
    }
    let min_valid = (n_rows + 7) / 8;
    if lhs_valid.len() < min_valid {
        return Err(CmpError::ValidityTooShort {
            got: lhs_valid.len(),
            min_bytes: min_valid,
            n_rows,
        });
    }
    if rhs_valid.len() < min_valid {
        return Err(CmpError::ValidityTooShort {
            got: rhs_valid.len(),
            min_bytes: min_valid,
            n_rows,
        });
    }
    let min_out = out_min_bytes(n_rows);
    if out_data.len() < min_out {
        return Err(CmpError::OutputTooShort {
            got: out_data.len(),
            min_bytes: min_out,
        });
    }
    if out_valid.len() < min_out {
        return Err(CmpError::OutputTooShort {
            got: out_valid.len(),
            min_bytes: min_out,
        });
    }

    if n_rows == 0 {
        // Zero-init the caller's output slices to mirror the
        // "no kernel ran, no writes" contract used by the filter
        // dispatchers. The slices are at least `min_out` bytes (4); we
        // clear that prefix to avoid leaking stale data from a previous
        // reuse.
        for b in &mut out_data[..min_out] {
            *b = 0;
        }
        for b in &mut out_valid[..min_out] {
            *b = 0;
        }
        return Ok(());
    }

    let n_rows_u32 = u32::try_from(n_rows).map_err(|_| CmpError::RowCountOverflow { n_rows })?;

    let lib = shared_library(device)?;
    let pso = lib.pipeline(op.entry_point_i64_cc())?;

    // SAFETY: `i64` has no invalid bit patterns, so reinterpreting a live
    // `&[i64]` as `&[u8]` of the same byte length is sound. The slice is
    // alive for the duration of this call; `new_buffer_from_bytes` copies
    // its contents into a freshly allocated MTLBuffer synchronously and
    // does not retain the reference.
    let lhs_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            lhs_data.as_ptr() as *const u8,
            std::mem::size_of_val(lhs_data),
        )
    };
    // SAFETY: identical reasoning to `lhs_bytes`.
    let rhs_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            rhs_data.as_ptr() as *const u8,
            std::mem::size_of_val(rhs_data),
        )
    };

    let lhs_buf = device.new_buffer_from_bytes(lhs_bytes)?;
    let lhs_valid_buf = device.new_buffer_from_bytes(&lhs_valid[..min_valid])?;
    let rhs_buf = device.new_buffer_from_bytes(rhs_bytes)?;
    let rhs_valid_buf = device.new_buffer_from_bytes(&rhs_valid[..min_valid])?;
    let out_data_buf = device.new_buffer_zeroed(min_out)?;
    let out_valid_buf = device.new_buffer_zeroed(min_out)?;
    let n_rows_buf = device.new_buffer_from_bytes(&n_rows_u32.to_le_bytes())?;

    queue.dispatch_1d(
        &pso,
        &[
            &lhs_buf,
            &lhs_valid_buf,
            &rhs_buf,
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

/// Dispatch a column-scalar i64 comparison.
///
/// Mirrors [`dispatch_cmp_i64`] but compares each row of `lhs_data` to a
/// single i64 scalar `rhs`. The scalar is treated as always-valid; output
/// validity is therefore `lhs_valid` and output data is set iff
/// `lhs_valid[i] AND (lhs[i] OP rhs)`.
///
/// Caller contract: identical to [`dispatch_cmp_i64`] but with no
/// `rhs_data` / `rhs_valid` slices to length-check.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_cmp_i64_scalar(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    lhs_data: &[i64],
    lhs_valid: &[u8],
    rhs: i64,
    n_rows: usize,
    op: CompareOp,
    out_data: &mut [u8],
    out_valid: &mut [u8],
) -> Result<(), CmpError> {
    if lhs_data.len() != n_rows {
        return Err(CmpError::InputLengthMismatch {
            lhs: lhs_data.len(),
            rhs: 0, // not applicable; report 0 to keep the field meaningful
            n_rows,
        });
    }
    let min_valid = (n_rows + 7) / 8;
    if lhs_valid.len() < min_valid {
        return Err(CmpError::ValidityTooShort {
            got: lhs_valid.len(),
            min_bytes: min_valid,
            n_rows,
        });
    }
    let min_out = out_min_bytes(n_rows);
    if out_data.len() < min_out {
        return Err(CmpError::OutputTooShort {
            got: out_data.len(),
            min_bytes: min_out,
        });
    }
    if out_valid.len() < min_out {
        return Err(CmpError::OutputTooShort {
            got: out_valid.len(),
            min_bytes: min_out,
        });
    }

    if n_rows == 0 {
        for b in &mut out_data[..min_out] {
            *b = 0;
        }
        for b in &mut out_valid[..min_out] {
            *b = 0;
        }
        return Ok(());
    }

    let n_rows_u32 = u32::try_from(n_rows).map_err(|_| CmpError::RowCountOverflow { n_rows })?;

    let lib = shared_library(device)?;
    let pso = lib.pipeline(op.entry_point_i64_cs())?;

    // SAFETY: `i64` has no invalid bit patterns; the slice is alive for
    // the duration of this call.
    let lhs_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            lhs_data.as_ptr() as *const u8,
            std::mem::size_of_val(lhs_data),
        )
    };

    let lhs_buf = device.new_buffer_from_bytes(lhs_bytes)?;
    let lhs_valid_buf = device.new_buffer_from_bytes(&lhs_valid[..min_valid])?;
    // Bind the scalar as an 8-byte little-endian payload. The MSL kernel
    // declares it as `constant int64_t&`; Metal's `constant` binding
    // reads exactly `sizeof(T)` bytes from the buffer.
    let rhs_buf = device.new_buffer_from_bytes(&rhs.to_le_bytes())?;
    let out_data_buf = device.new_buffer_zeroed(min_out)?;
    let out_valid_buf = device.new_buffer_zeroed(min_out)?;
    let n_rows_buf = device.new_buffer_from_bytes(&n_rows_u32.to_le_bytes())?;

    queue.dispatch_1d(
        &pso,
        &[
            &lhs_buf,
            &lhs_valid_buf,
            &rhs_buf,
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

/// Dispatch a column-column f64 comparison.
///
/// Direct analogue of [`dispatch_cmp_i64`]: reads `lhs_data` and
/// `rhs_data` (both `&[f64]` of length `n_rows`) plus their validity
/// bitmaps `lhs_valid` and `rhs_valid` (bit-packed, `ceil(n_rows / 8)`
/// bytes minimum). Writes a bit-packed bool column to `out_data` and a
/// bit-packed validity bitmap to `out_valid`.
///
/// Output semantics under Polars/IEEE 754 NaN rules:
///   - `out_valid[i]` is set iff `lhs_valid[i] AND rhs_valid[i]`.
///     NaN's validity bit propagates unchanged (NaN is a value).
///   - `out_data[i]` is set iff `out_valid[i] AND (lhs[i] OP rhs[i])`,
///     where `OP` follows IEEE 754: `NaN OP x` is false for
///     `==/</<=/>/>=` and true for `!=`. Data bits at null rows are
///     left at zero.
///
/// Caller contract: identical to [`dispatch_cmp_i64`].
///
/// MSL note: the underlying kernel binds f64 data as `device const
/// ulong*` and runs the comparison in integer arithmetic on the bit
/// pattern, because Apple Silicon MSL compute kernels do not support
/// the `double` type. The encoding is exact (no rounding), so this is
/// indistinguishable from native double math for all finite values,
/// NaNs, infinities, and ±0.0.
// Argument count mirrors `dispatch_cmp_i64`; same justification.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_cmp_f64(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    lhs_data: &[f64],
    lhs_valid: &[u8],
    rhs_data: &[f64],
    rhs_valid: &[u8],
    n_rows: usize,
    op: CompareOp,
    out_data: &mut [u8],
    out_valid: &mut [u8],
) -> Result<(), CmpError> {
    if lhs_data.len() != n_rows || rhs_data.len() != n_rows {
        return Err(CmpError::InputLengthMismatch {
            lhs: lhs_data.len(),
            rhs: rhs_data.len(),
            n_rows,
        });
    }
    let min_valid = (n_rows + 7) / 8;
    if lhs_valid.len() < min_valid {
        return Err(CmpError::ValidityTooShort {
            got: lhs_valid.len(),
            min_bytes: min_valid,
            n_rows,
        });
    }
    if rhs_valid.len() < min_valid {
        return Err(CmpError::ValidityTooShort {
            got: rhs_valid.len(),
            min_bytes: min_valid,
            n_rows,
        });
    }
    let min_out = out_min_bytes(n_rows);
    if out_data.len() < min_out {
        return Err(CmpError::OutputTooShort {
            got: out_data.len(),
            min_bytes: min_out,
        });
    }
    if out_valid.len() < min_out {
        return Err(CmpError::OutputTooShort {
            got: out_valid.len(),
            min_bytes: min_out,
        });
    }

    if n_rows == 0 {
        for b in &mut out_data[..min_out] {
            *b = 0;
        }
        for b in &mut out_valid[..min_out] {
            *b = 0;
        }
        return Ok(());
    }

    let n_rows_u32 = u32::try_from(n_rows).map_err(|_| CmpError::RowCountOverflow { n_rows })?;

    let lib = shared_library(device)?;
    let pso = lib.pipeline(op.entry_point_f64_cc())?;

    // SAFETY: `f64` has no invalid bit patterns; every 8-byte sequence
    // is a legitimate `f64` (some bit patterns represent NaN payloads,
    // but those are still valid `f64` values per IEEE 754). Reading
    // those bytes as `ulong` on the GPU is the documented contract of
    // the kernel — see `cmp_f64.metal`'s top-of-file comment. The
    // slice is alive for the duration of this call;
    // `new_buffer_from_bytes` copies its contents synchronously.
    let lhs_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            lhs_data.as_ptr() as *const u8,
            std::mem::size_of_val(lhs_data),
        )
    };
    // SAFETY: identical reasoning to `lhs_bytes`.
    let rhs_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            rhs_data.as_ptr() as *const u8,
            std::mem::size_of_val(rhs_data),
        )
    };

    let lhs_buf = device.new_buffer_from_bytes(lhs_bytes)?;
    let lhs_valid_buf = device.new_buffer_from_bytes(&lhs_valid[..min_valid])?;
    let rhs_buf = device.new_buffer_from_bytes(rhs_bytes)?;
    let rhs_valid_buf = device.new_buffer_from_bytes(&rhs_valid[..min_valid])?;
    let out_data_buf = device.new_buffer_zeroed(min_out)?;
    let out_valid_buf = device.new_buffer_zeroed(min_out)?;
    let n_rows_buf = device.new_buffer_from_bytes(&n_rows_u32.to_le_bytes())?;

    queue.dispatch_1d(
        &pso,
        &[
            &lhs_buf,
            &lhs_valid_buf,
            &rhs_buf,
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

/// Dispatch a column-scalar f64 comparison.
///
/// Mirrors [`dispatch_cmp_f64`] but compares each row of `lhs_data` to
/// a single f64 scalar `rhs`. The scalar is treated as always-valid;
/// output validity is therefore `lhs_valid` and output data is set iff
/// `lhs_valid[i] AND (lhs[i] OP rhs)` under IEEE 754 / Polars NaN rules.
///
/// The scalar is bound as a `constant ulong&` — its `f64::to_bits()`
/// payload — matching the kernel's `as_type` style of treating f64 as
/// opaque 8 bytes. A scalar NaN propagates the NaN rules row-by-row
/// (e.g. `col == NaN` → all-false, `col != NaN` → all-true at valid
/// rows; both at zero at null rows).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_cmp_f64_scalar(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    lhs_data: &[f64],
    lhs_valid: &[u8],
    rhs: f64,
    n_rows: usize,
    op: CompareOp,
    out_data: &mut [u8],
    out_valid: &mut [u8],
) -> Result<(), CmpError> {
    if lhs_data.len() != n_rows {
        return Err(CmpError::InputLengthMismatch {
            lhs: lhs_data.len(),
            rhs: 0,
            n_rows,
        });
    }
    let min_valid = (n_rows + 7) / 8;
    if lhs_valid.len() < min_valid {
        return Err(CmpError::ValidityTooShort {
            got: lhs_valid.len(),
            min_bytes: min_valid,
            n_rows,
        });
    }
    let min_out = out_min_bytes(n_rows);
    if out_data.len() < min_out {
        return Err(CmpError::OutputTooShort {
            got: out_data.len(),
            min_bytes: min_out,
        });
    }
    if out_valid.len() < min_out {
        return Err(CmpError::OutputTooShort {
            got: out_valid.len(),
            min_bytes: min_out,
        });
    }

    if n_rows == 0 {
        for b in &mut out_data[..min_out] {
            *b = 0;
        }
        for b in &mut out_valid[..min_out] {
            *b = 0;
        }
        return Ok(());
    }

    let n_rows_u32 = u32::try_from(n_rows).map_err(|_| CmpError::RowCountOverflow { n_rows })?;

    let lib = shared_library(device)?;
    let pso = lib.pipeline(op.entry_point_f64_cs())?;

    // SAFETY: `f64` has no invalid bit patterns; the slice is alive for
    // the duration of this call. See `dispatch_cmp_f64` for the full
    // GPU-side bit-pattern contract.
    let lhs_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            lhs_data.as_ptr() as *const u8,
            std::mem::size_of_val(lhs_data),
        )
    };

    let lhs_buf = device.new_buffer_from_bytes(lhs_bytes)?;
    let lhs_valid_buf = device.new_buffer_from_bytes(&lhs_valid[..min_valid])?;
    // Pass the scalar's raw bit pattern as a little-endian u64. The MSL
    // kernel binds it as `constant ulong&` and uses it directly as the
    // RHS bit pattern in the IEEE-emulated comparison helpers.
    let rhs_buf = device.new_buffer_from_bytes(&rhs.to_bits().to_le_bytes())?;
    let out_data_buf = device.new_buffer_zeroed(min_out)?;
    let out_valid_buf = device.new_buffer_zeroed(min_out)?;
    let n_rows_buf = device.new_buffer_from_bytes(&n_rows_u32.to_le_bytes())?;

    queue.dispatch_1d(
        &pso,
        &[
            &lhs_buf,
            &lhs_valid_buf,
            &rhs_buf,
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
