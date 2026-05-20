//! Filter compaction kernel wrappers.
//!
//! M1 Phase 5 — Pass 1: predicate evaluation. Reads a bit-packed boolean
//! column plus its validity bitmap and writes a dense `u8[n_rows]` where
//! each byte is exactly `1` (keep this row) or `0` (drop it). The dense
//! u8 form is the input to MLX cumsum, which produces the scatter indices
//! consumed by the per-dtype scatter kernels (Tasks 11-13).
//!
//! Pass 3 (this module's `dispatch_scatter_i64`, mirrored for f64/bool in
//! later tasks) consumes the keep flags, the inclusive prefix sum, and
//! the source column, and writes the surviving rows into a dense output.
//! Validity bits are OR'd in via an atomic word-level op (see
//! `_validity.metal::set_valid_atomic_or`) because 8 output rows share a
//! byte and multiple surviving rows can race the same byte.

use crate::command::{CommandQueue, DispatchError};
use crate::shader_lib::{shared_library, ShaderError};
use polars_metal_buffer::{BufferError, MetalDevice};

/// Sentinel written to `dst_data[n_out]` by `filter_scatter_i64` if the
/// kernel ever computes `out_idx >= n_out` (which would only happen given
/// a buggy prefix sum or a programmer error). The host checks the
/// sentinel slot after the dispatch returns; if it matches, we raise
/// [`FilterError::ScatterOverrun`] rather than handing the caller silently
/// truncated output. Kept in sync with the MSL constant of the same name
/// in `shaders/filter_scatter.metal`.
pub const SCATTER_SENTINEL_I64: i64 = 0xDEADBEEFCAFEBABE_u64 as i64;

/// f64 scatter overrun sentinel, expressed as raw bits. We expose the
/// `u64` bit pattern (rather than an `f64` constant) because Rust does
/// not permit floating-point bit-pattern equality at the type level —
/// `f64` does not implement `Eq`, and two NaN values are never `==` even
/// if their bit patterns match. Callers compare via
/// `dst_data[n_out].to_bits() == SCATTER_SENTINEL_F64_BITS`.
///
/// The bit pattern is a NaN (exponent bits 52-62 all 1, non-zero
/// mantissa containing the recognisable `DEADBEEFCAFE` marker) so it is
/// distinguishable from every finite value the kernel could legitimately
/// copy. NaN itself is a valid f64 value, hence the bit-level (rather
/// than `is_nan`) comparison: users may have their own NaN payloads in
/// the data; only this exact pattern means "kernel overrun". Kept in
/// sync with the MSL constant of the same name in
/// `shaders/filter_scatter.metal`.
pub const SCATTER_SENTINEL_F64_BITS: u64 = 0x7FFD_EADB_EEFC_AFE0_u64;

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
    /// The scatter kernel computed an output index >= `n_out` and tripped
    /// the sentinel. Indicates a kernel-logic bug (most often a buggy
    /// prefix-sum input or an `n_out` smaller than what the prefix sum
    /// implies); the kernel's output is not safe to consume.
    #[error("scatter overrun: kernel produced an out-of-range output index (sentinel tripped)")]
    ScatterOverrun,
    /// `dst_data` must allocate `n_out + 1` slots (the extra slot is the
    /// sentinel overrun guard). The caller passed a shorter slice.
    #[error("dst_data too small: got {got} slots, expected at least {expected} (n_out + 1)")]
    DstDataTooSmall { got: usize, expected: usize },
    /// `dst_valid` must hold at least `ceil(n_out / 8)` bytes, rounded up
    /// to a multiple of 4 so the kernel's `device atomic_uint*` cast is
    /// well-aligned.
    #[error("dst_valid too small: got {got} bytes, expected at least {expected} (4-byte aligned)")]
    DstValidTooSmall { got: usize, expected: usize },
    /// `filter_scatter_bool` requires `dst_data` to be at least
    /// `ceil(n_out / 8)` bytes rounded up to 4-byte alignment (for the
    /// u32 atomic cast). The bool scatter has NO sentinel slot (every
    /// bit value is legitimate), so the allocation requirement is
    /// `dst_valid_min_bytes(n_out)` instead of `(n_out + 1) * sizeof(T)`.
    #[error("dst_data (bool, bit-packed) too small: got {got} bytes, expected at least {expected} (4-byte aligned)")]
    DstDataBoolTooSmall { got: usize, expected: usize },
    /// Host-side prefix-sum invariant tripped: `prefix_sum[n_rows - 1]`
    /// must equal `n_out` (the prefix sum is inclusive over the dense
    /// keep flags, so its final element is the total number of
    /// survivors). When this invariant holds, the kernel can never
    /// compute `out_idx >= n_out`, which is the only safety net the
    /// bool scatter has (it cannot write a sentinel because every bit
    /// is a legitimate data value). If this fires, the caller's
    /// compaction pipeline produced inconsistent state — most often a
    /// mis-sized `n_out` or a buggy cumsum.
    #[error(
        "prefix-sum invariant violated: last_prefix={last_prefix} but n_out={n_out}; \
         scatter would risk silent corruption"
    )]
    ScatterPrefixSumMismatch { last_prefix: u32, n_out: usize },
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

/// Minimum number of bytes required for an output validity bitmap holding
/// `n_out` rows. Returns at least 4 because the kernel reinterprets the
/// bitmap as `device atomic_uint*` and requires 4-byte alignment; smaller
/// outputs would otherwise leave the trailing partial word unbacked.
fn dst_valid_min_bytes(n_out: usize) -> usize {
    let raw = (n_out + 7) / 8;
    let padded = (raw + 3) & !3;
    padded.max(4)
}

/// Dispatch `filter_scatter_i64` — pass 3 of the filter compaction
/// pipeline.
///
/// Given the source column (`src_data` + `src_valid`), the dense keep
/// flags from pass 1 (`keep`), and the inclusive prefix sum of the keep
/// flags from pass 2 (`prefix_sum`), write the surviving rows into
/// `dst_data` and OR their validity bits into `dst_valid`. `n_out` is the
/// total count of surviving rows (equal to `*prefix_sum.last()` for
/// non-empty inputs).
///
/// Caller contract:
///   - `keep.len() == src_data.len() == prefix_sum.len() == n_rows`.
///   - `src_valid.len() >= ceil(n_rows / 8)`.
///   - `dst_data.len() >= n_out + 1`. The extra slot is the sentinel
///     overrun guard; the dispatcher pre-zeros it and checks it
///     post-dispatch, returning [`FilterError::ScatterOverrun`] if the
///     kernel computed an out-of-range output index.
///   - `dst_valid.len() >= ceil(n_out / 8)`, rounded up to a multiple of
///     4 bytes (minimum 4) so the kernel's `device atomic_uint*` cast is
///     well-aligned. The dispatcher zero-initialises `dst_valid` before
///     dispatch (the kernel's atomic OR never clears bits).
///
/// The function returns once the kernel has completed on the GPU and the
/// output buffers have been copied back into the caller's slices.
///
/// `n_rows == 0` is a no-op (Metal rejects zero-byte buffers and
/// zero-grid dispatches; both are caught here). `n_out == 0` is allowed
/// for non-empty inputs: the kernel runs but never writes (every thread
/// short-circuits on `keep[gid] == 0`).
// 9 arguments: the device + queue pair plus 4 in-buffers, 2 out-buffers,
// and `n_out`. A struct wrapper would not improve readability — every
// argument is a distinct kernel binding — and would obscure the
// 1:1 mapping to the MSL kernel's argument list.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_scatter_i64(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    src_data: &[i64],
    src_valid: &[u8],
    keep: &[u8],
    prefix_sum: &[u32],
    n_out: usize,
    dst_data: &mut [i64],
    dst_valid: &mut [u8],
) -> Result<(), FilterError> {
    let n_rows = src_data.len();
    if keep.len() != n_rows {
        return Err(FilterError::OutputLengthMismatch {
            got: keep.len(),
            expected: n_rows,
        });
    }
    if prefix_sum.len() != n_rows {
        return Err(FilterError::OutputLengthMismatch {
            got: prefix_sum.len(),
            expected: n_rows,
        });
    }
    let expected_dst = n_out
        .checked_add(1)
        .ok_or(FilterError::RowCountOverflow { n_rows: n_out })?;
    if dst_data.len() < expected_dst {
        return Err(FilterError::DstDataTooSmall {
            got: dst_data.len(),
            expected: expected_dst,
        });
    }
    let min_valid = dst_valid_min_bytes(n_out);
    if dst_valid.len() < min_valid {
        return Err(FilterError::DstValidTooSmall {
            got: dst_valid.len(),
            expected: min_valid,
        });
    }
    let min_src_valid = (n_rows + 7) / 8;
    if src_valid.len() < min_src_valid {
        return Err(FilterError::InputLengthMismatch {
            pred_bytes: src_valid.len(),
            min_bytes: min_src_valid,
            n_rows,
        });
    }
    if n_rows == 0 {
        // Mirror `dispatch_predicate_to_u8`: zero rows is a no-op. The
        // caller's `dst_data` / `dst_valid` are not touched, matching the
        // "no kernel ran, no writes" contract.
        return Ok(());
    }

    let n_rows_u32 = u32::try_from(n_rows).map_err(|_| FilterError::RowCountOverflow { n_rows })?;
    let n_out_u32 =
        u32::try_from(n_out).map_err(|_| FilterError::RowCountOverflow { n_rows: n_out })?;

    let lib = shared_library(device)?;
    let pso = lib.pipeline("filter_scatter_i64")?;

    // Reinterpret typed slices as byte slices for the buffer constructors.
    // i64 and u32 both have well-defined byte representations (all bit
    // patterns are valid), so this is a transmute-by-pointer-cast with
    // no UB. We avoid the bytemuck dep per the workspace "no new
    // dependency without justification" rule.
    //
    // SAFETY: `src_data` is alive for the duration of this call; its
    // pointer is non-null and the length `src_data.len() * 8` fits in
    // usize on all supported targets (we just bounds-checked n_rows
    // against u32::MAX above). `i64` has no invalid bit patterns, so
    // reinterpreting as `[u8]` is sound. `new_buffer_from_bytes` copies
    // the bytes into a freshly allocated MTLBuffer synchronously, so the
    // slice does not need to live past the call.
    let src_data_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            src_data.as_ptr() as *const u8,
            std::mem::size_of_val(src_data),
        )
    };
    // SAFETY: identical reasoning to `src_data_bytes`; `u32` has no
    // invalid bit patterns.
    let prefix_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            prefix_sum.as_ptr() as *const u8,
            std::mem::size_of_val(prefix_sum),
        )
    };

    let src_data_buf = device.new_buffer_from_bytes(src_data_bytes)?;
    let src_valid_buf = device.new_buffer_from_bytes(&src_valid[..min_src_valid])?;
    let keep_buf = device.new_buffer_from_bytes(keep)?;
    let prefix_buf = device.new_buffer_from_bytes(prefix_bytes)?;

    // `dst_data` is allocated as `n_out + 1` slots; the extra slot is the
    // sentinel guard. `new_buffer_zeroed` zero-fills, so the sentinel
    // starts at 0 and only the kernel can write `SCATTER_SENTINEL_I64` to
    // it.
    let dst_data_buf = device.new_buffer_zeroed(expected_dst * std::mem::size_of::<i64>())?;
    // `dst_valid` is zero-initialised by `new_buffer_zeroed`, satisfying
    // the atomic-OR-only contract documented in `_validity.metal`.
    let dst_valid_buf = device.new_buffer_zeroed(min_valid)?;

    let n_rows_buf = device.new_buffer_from_bytes(&n_rows_u32.to_le_bytes())?;
    let n_out_buf = device.new_buffer_from_bytes(&n_out_u32.to_le_bytes())?;

    queue.dispatch_1d(
        &pso,
        &[
            &src_data_buf,
            &src_valid_buf,
            &keep_buf,
            &prefix_buf,
            &dst_data_buf,
            &dst_valid_buf,
            &n_rows_buf,
            &n_out_buf,
        ],
        n_rows,
    )?;
    queue.wait_until_complete()?;

    // Copy outputs back. `dst_data` is `expected_dst = n_out + 1` slots
    // (including the sentinel slot); we copy all of them so callers can
    // inspect the sentinel if they wish, then re-verify it here.
    let dst_data_bytes_out = &dst_data_buf.as_slice()[..expected_dst * std::mem::size_of::<i64>()];
    // SAFETY: `dst_data_bytes_out` lives in the shared-memory MTLBuffer
    // we just allocated; its length is exactly `expected_dst * 8`, which
    // is `expected_dst` valid i64s. `i64` has no invalid bit patterns and
    // the source pointer is aligned because `MTLBuffer::contents()` is
    // always aligned to at least 256 bytes on Apple Silicon (Metal
    // resource alignment guarantee).
    let dst_data_typed: &[i64] = unsafe {
        std::slice::from_raw_parts(dst_data_bytes_out.as_ptr() as *const i64, expected_dst)
    };
    dst_data[..expected_dst].copy_from_slice(dst_data_typed);

    let valid_out = &dst_valid_buf.as_slice()[..min_valid];
    dst_valid[..min_valid].copy_from_slice(valid_out);

    if dst_data[n_out] == SCATTER_SENTINEL_I64 {
        return Err(FilterError::ScatterOverrun);
    }

    Ok(())
}

/// Dispatch `filter_scatter_f64` — pass 3 of the filter compaction
/// pipeline, f64 variant.
///
/// Shape-isomorphic to [`dispatch_scatter_i64`]: same buffer layout,
/// same kernel arguments, same overrun-sentinel protocol. The only
/// differences are the source/destination element type (`f64` instead
/// of `i64`) and the sentinel value (a NaN bit pattern instead of an
/// integer constant).
///
/// Internally, the kernel binds `dst_data` / `src_data` as `device
/// ulong*` (8-byte opaque chunks) rather than `device double*`. Apple
/// Silicon MSL compute kernels do not support `double`, but the scatter
/// performs no floating-point arithmetic — it just copies 8-byte slots
/// — so the `ulong` reinterpretation is bit-identical to an f64 copy.
/// NaN payloads, ±Inf, ±0.0, subnormals, and ordinary finites all
/// round-trip exactly.
///
/// Caller contract: identical to [`dispatch_scatter_i64`]. See that
/// function's doc-comment for the full enumeration of length / alignment
/// requirements.
///
/// On overrun the host compares
/// `dst_data[n_out].to_bits() == SCATTER_SENTINEL_F64_BITS`. The
/// bit-level comparison is required because two NaN values are never
/// `==` in IEEE 754, and we also need to distinguish the sentinel's
/// exact bit pattern from any other NaN that may appear in user data.
// 9 arguments mirror `dispatch_scatter_i64`; see that function for the
// rationale for not packaging them into a struct.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_scatter_f64(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    src_data: &[f64],
    src_valid: &[u8],
    keep: &[u8],
    prefix_sum: &[u32],
    n_out: usize,
    dst_data: &mut [f64],
    dst_valid: &mut [u8],
) -> Result<(), FilterError> {
    let n_rows = src_data.len();
    if keep.len() != n_rows {
        return Err(FilterError::OutputLengthMismatch {
            got: keep.len(),
            expected: n_rows,
        });
    }
    if prefix_sum.len() != n_rows {
        return Err(FilterError::OutputLengthMismatch {
            got: prefix_sum.len(),
            expected: n_rows,
        });
    }
    let expected_dst = n_out
        .checked_add(1)
        .ok_or(FilterError::RowCountOverflow { n_rows: n_out })?;
    if dst_data.len() < expected_dst {
        return Err(FilterError::DstDataTooSmall {
            got: dst_data.len(),
            expected: expected_dst,
        });
    }
    let min_valid = dst_valid_min_bytes(n_out);
    if dst_valid.len() < min_valid {
        return Err(FilterError::DstValidTooSmall {
            got: dst_valid.len(),
            expected: min_valid,
        });
    }
    let min_src_valid = (n_rows + 7) / 8;
    if src_valid.len() < min_src_valid {
        return Err(FilterError::InputLengthMismatch {
            pred_bytes: src_valid.len(),
            min_bytes: min_src_valid,
            n_rows,
        });
    }
    if n_rows == 0 {
        return Ok(());
    }

    let n_rows_u32 = u32::try_from(n_rows).map_err(|_| FilterError::RowCountOverflow { n_rows })?;
    let n_out_u32 =
        u32::try_from(n_out).map_err(|_| FilterError::RowCountOverflow { n_rows: n_out })?;

    let lib = shared_library(device)?;
    let pso = lib.pipeline("filter_scatter_f64")?;

    // Reinterpret typed slices as byte slices for the buffer
    // constructors. `f64` has well-defined byte representations for
    // every bit pattern (including NaN payloads, ±Inf, ±0.0,
    // subnormals), so the host-side reinterpret is bit-identical with
    // the `ulong` view the GPU takes. We avoid the bytemuck dep per the
    // workspace "no new dependency without justification" rule.
    //
    // SAFETY: `src_data` is alive for the duration of this call; its
    // pointer is non-null and the byte length `src_data.len() * 8` fits
    // in usize on all supported targets (we just bounds-checked n_rows
    // against u32::MAX above). `f64` has no invalid bit patterns, so
    // reinterpreting as `[u8]` is sound. `new_buffer_from_bytes` copies
    // the bytes into a freshly allocated MTLBuffer synchronously, so
    // the slice does not need to live past the call.
    let src_data_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            src_data.as_ptr() as *const u8,
            std::mem::size_of_val(src_data),
        )
    };
    // SAFETY: identical reasoning to `src_data_bytes`; `u32` has no
    // invalid bit patterns.
    let prefix_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            prefix_sum.as_ptr() as *const u8,
            std::mem::size_of_val(prefix_sum),
        )
    };

    let src_data_buf = device.new_buffer_from_bytes(src_data_bytes)?;
    let src_valid_buf = device.new_buffer_from_bytes(&src_valid[..min_src_valid])?;
    let keep_buf = device.new_buffer_from_bytes(keep)?;
    let prefix_buf = device.new_buffer_from_bytes(prefix_bytes)?;

    // `dst_data` is allocated as `n_out + 1` slots; the extra slot is
    // the sentinel guard. `new_buffer_zeroed` zero-fills, so the
    // sentinel starts at +0.0 (bit pattern 0) and only the kernel can
    // write the SCATTER_SENTINEL_F64_BITS NaN to it.
    let dst_data_buf = device.new_buffer_zeroed(expected_dst * std::mem::size_of::<f64>())?;
    // `dst_valid` is zero-initialised by `new_buffer_zeroed`, satisfying
    // the atomic-OR-only contract documented in `_validity.metal`.
    let dst_valid_buf = device.new_buffer_zeroed(min_valid)?;

    let n_rows_buf = device.new_buffer_from_bytes(&n_rows_u32.to_le_bytes())?;
    let n_out_buf = device.new_buffer_from_bytes(&n_out_u32.to_le_bytes())?;

    queue.dispatch_1d(
        &pso,
        &[
            &src_data_buf,
            &src_valid_buf,
            &keep_buf,
            &prefix_buf,
            &dst_data_buf,
            &dst_valid_buf,
            &n_rows_buf,
            &n_out_buf,
        ],
        n_rows,
    )?;
    queue.wait_until_complete()?;

    // Copy outputs back. `dst_data` is `expected_dst = n_out + 1` slots
    // (including the sentinel slot); we copy all of them so callers can
    // inspect the sentinel if they wish, then re-verify it here.
    let dst_data_bytes_out = &dst_data_buf.as_slice()[..expected_dst * std::mem::size_of::<f64>()];
    // SAFETY: `dst_data_bytes_out` lives in the shared-memory MTLBuffer
    // we just allocated; its length is exactly `expected_dst * 8`,
    // which is `expected_dst` valid f64s. `f64` has no invalid bit
    // patterns (every 64-bit value is some f64, including NaN with any
    // payload, ±Inf, subnormals, etc.) and the source pointer is
    // aligned because `MTLBuffer::contents()` is always aligned to at
    // least 256 bytes on Apple Silicon (Metal resource alignment
    // guarantee).
    let dst_data_typed: &[f64] = unsafe {
        std::slice::from_raw_parts(dst_data_bytes_out.as_ptr() as *const f64, expected_dst)
    };
    dst_data[..expected_dst].copy_from_slice(dst_data_typed);

    let valid_out = &dst_valid_buf.as_slice()[..min_valid];
    dst_valid[..min_valid].copy_from_slice(valid_out);

    // Bit-level comparison: NaN != NaN under IEEE 754, so we cannot
    // use `==` on the f64 directly. The exact bit pattern check also
    // distinguishes the sentinel from any user-supplied NaN payload
    // that may legitimately appear in `src_data`.
    if dst_data[n_out].to_bits() == SCATTER_SENTINEL_F64_BITS {
        return Err(FilterError::ScatterOverrun);
    }

    Ok(())
}

/// Dispatch `filter_scatter_bool` — pass 3 of the filter compaction
/// pipeline, bool variant.
///
/// Bool is unique among the scatter variants: BOTH the data buffer and
/// the validity buffer are bit-packed (one bit per row, 8 rows per
/// byte). Multiple surviving rows therefore race the same data byte
/// just as they race the same validity byte. The kernel uses
/// `set_valid_atomic_or` for both buffers; the host enforces the
/// same caller contract on `dst_data` as on `dst_valid` (4-byte
/// alignment, zero-initialised).
///
/// There is NO sentinel slot for `dst_data`. Every bit pattern is a
/// legitimate bool value (0 or 1), so we cannot reserve a "this means
/// overrun" sentinel as we do for the i64/f64 variants. The safety net
/// is instead the host-side prefix-sum invariant
/// `prefix_sum[n_rows - 1] == n_out`: if this holds, the kernel's
/// `out_idx = prefix_sum[gid] - 1` can never exceed `n_out - 1`. The
/// dispatcher checks the invariant before binding any buffers; a
/// violation raises [`FilterError::ScatterPrefixSumMismatch`] instead
/// of risking silent corruption.
///
/// Because the bit-packed bytes do not carry a row count, `n_rows`
/// is passed explicitly (e.g. 13 rows fit in 2 bytes with 3 padding
/// bits).
///
/// Caller contract:
///   - `keep.len() == prefix_sum.len() == n_rows`.
///   - `src_data.len() >= ceil(n_rows / 8)`.
///   - `src_valid.len() >= ceil(n_rows / 8)`.
///   - `dst_data.len() >= dst_valid_min_bytes(n_out)` (4-byte aligned,
///     minimum 4 bytes).
///   - `dst_valid.len() >= dst_valid_min_bytes(n_out)`.
///   - If `n_rows > 0`, `prefix_sum[n_rows - 1]` must equal `n_out`.
///   - The dispatcher zero-initialises both `dst_data` and `dst_valid`
///     on the device side (the kernel's atomic OR never clears bits);
///     the caller's slices are filled with the device-side values on
///     return.
///
/// `n_rows == 0` is a no-op (Metal rejects zero-byte buffers and
/// zero-grid dispatches; both are caught here). `n_out == 0` is allowed
/// for non-empty inputs: the kernel runs but never writes (every thread
/// short-circuits on `keep[gid] == 0`).
// 10 arguments mirror the i64/f64 dispatchers but with an explicit
// `n_rows` (bit-packed bytes don't encode row count). A struct wrapper
// would not improve readability — every argument is a distinct kernel
// binding or a host-side safety check input — and would obscure the
// 1:1 mapping to the MSL kernel's argument list.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_scatter_bool(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    src_data: &[u8],
    src_valid: &[u8],
    keep: &[u8],
    prefix_sum: &[u32],
    n_rows: usize,
    n_out: usize,
    dst_data: &mut [u8],
    dst_valid: &mut [u8],
) -> Result<(), FilterError> {
    if keep.len() != n_rows {
        return Err(FilterError::OutputLengthMismatch {
            got: keep.len(),
            expected: n_rows,
        });
    }
    if prefix_sum.len() != n_rows {
        return Err(FilterError::OutputLengthMismatch {
            got: prefix_sum.len(),
            expected: n_rows,
        });
    }
    let min_src_bytes = (n_rows + 7) / 8;
    if src_data.len() < min_src_bytes {
        return Err(FilterError::InputLengthMismatch {
            pred_bytes: src_data.len(),
            min_bytes: min_src_bytes,
            n_rows,
        });
    }
    if src_valid.len() < min_src_bytes {
        return Err(FilterError::InputLengthMismatch {
            pred_bytes: src_valid.len(),
            min_bytes: min_src_bytes,
            n_rows,
        });
    }
    let min_out_bytes = dst_valid_min_bytes(n_out);
    if dst_data.len() < min_out_bytes {
        return Err(FilterError::DstDataBoolTooSmall {
            got: dst_data.len(),
            expected: min_out_bytes,
        });
    }
    if dst_valid.len() < min_out_bytes {
        return Err(FilterError::DstValidTooSmall {
            got: dst_valid.len(),
            expected: min_out_bytes,
        });
    }

    // Host-side prefix-sum invariant. The bool scatter has no sentinel
    // overrun guard on the data buffer — every bit value is a
    // legitimate bool — so we must catch a mismatched (prefix_sum,
    // n_out) pair before any thread writes to a bit-packed buffer that
    // we cannot post-hoc audit. If `n_rows > 0`, the last element of
    // the inclusive prefix sum is the total survivor count, which the
    // caller has told us is `n_out`. A discrepancy means the caller's
    // compaction pipeline (predicate kernel or cumsum) is producing
    // inconsistent state — fail loudly here rather than producing
    // silently-corrupt output.
    if n_rows > 0 {
        let last_prefix = prefix_sum[n_rows - 1];
        let n_out_u32 =
            u32::try_from(n_out).map_err(|_| FilterError::RowCountOverflow { n_rows: n_out })?;
        if last_prefix != n_out_u32 {
            return Err(FilterError::ScatterPrefixSumMismatch { last_prefix, n_out });
        }
    }

    if n_rows == 0 {
        // Mirror the i64/f64 dispatchers: zero rows is a no-op. The
        // caller's `dst_data` / `dst_valid` are not touched.
        return Ok(());
    }

    let n_rows_u32 = u32::try_from(n_rows).map_err(|_| FilterError::RowCountOverflow { n_rows })?;
    let n_out_u32 =
        u32::try_from(n_out).map_err(|_| FilterError::RowCountOverflow { n_rows: n_out })?;

    let lib = shared_library(device)?;
    let pso = lib.pipeline("filter_scatter_bool")?;

    // `src_data` and `src_valid` are already `&[u8]`; pass the
    // exact `min_src_bytes`-prefix so the device-side allocation
    // matches the kernel's read pattern.
    let src_data_buf = device.new_buffer_from_bytes(&src_data[..min_src_bytes])?;
    let src_valid_buf = device.new_buffer_from_bytes(&src_valid[..min_src_bytes])?;
    let keep_buf = device.new_buffer_from_bytes(keep)?;

    // SAFETY: `u32` has no invalid bit patterns; reinterpreting a
    // `&[u32]` as `&[u8]` is sound. `new_buffer_from_bytes` copies the
    // bytes into a freshly allocated MTLBuffer synchronously, so the
    // slice does not need to live past the call.
    let prefix_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            prefix_sum.as_ptr() as *const u8,
            std::mem::size_of_val(prefix_sum),
        )
    };
    let prefix_buf = device.new_buffer_from_bytes(prefix_bytes)?;

    // BOTH dst buffers are zero-initialised and 4-byte aligned (the
    // kernel binds them as `device atomic_uint*`). The atomic OR is
    // append-only; bits left at 0 stay 0.
    let dst_data_buf = device.new_buffer_zeroed(min_out_bytes)?;
    let dst_valid_buf = device.new_buffer_zeroed(min_out_bytes)?;

    let n_rows_buf = device.new_buffer_from_bytes(&n_rows_u32.to_le_bytes())?;
    let n_out_buf = device.new_buffer_from_bytes(&n_out_u32.to_le_bytes())?;

    queue.dispatch_1d(
        &pso,
        &[
            &src_data_buf,
            &src_valid_buf,
            &keep_buf,
            &prefix_buf,
            &dst_data_buf,
            &dst_valid_buf,
            &n_rows_buf,
            &n_out_buf,
        ],
        n_rows,
    )?;
    queue.wait_until_complete()?;

    // Copy the bit-packed outputs back. Both buffers are exactly
    // `min_out_bytes` long; copy that prefix into the caller's slices.
    dst_data[..min_out_bytes].copy_from_slice(&dst_data_buf.as_slice()[..min_out_bytes]);
    dst_valid[..min_out_bytes].copy_from_slice(&dst_valid_buf.as_slice()[..min_out_bytes]);

    Ok(())
}
