//! Three-pass filter compaction pipeline.
//!
//! Orchestrates Tasks 10-13 + the MLX cumsum (Task 5) into a single
//! per-dtype entry point. Given a predicate column (bit-packed data +
//! bit-packed validity) and a source column, returns the compacted
//! survivors as a fresh `Vec<T>` (data) and `Vec<u8>` (bit-packed
//! validity) plus the survivor count.
//!
//! The three passes are:
//!   1. `dispatch_predicate_to_u8`   — bit-packed predicate (data ∧ valid)
//!                                     → dense `u8[n_rows]` keep flags.
//!   2. `cumsum_u8_to_u32` (MLX)     — inclusive prefix sum → output offsets.
//!   3. `dispatch_scatter_<dtype>`   — scatter surviving rows into a dense
//!                                     output, OR-ing validity bits in.
//!
//! Each call allocates fresh intermediate `keep` and `prefix` buffers.
//! M1's perf is not gated on this; M2 will revisit via the per-query
//! BumpArena (see `docs/open-questions.md`).
//!
//! Errors from any of the three passes (or the MLX FFI) are wrapped in
//! [`PipelineError`]. The pipeline is structurally identical across
//! dtypes; the only per-dtype variation is the data slot size and the
//! scatter dispatcher. Bool is special-cased because both its data and
//! validity are bit-packed; for bool, the returned `data` Vec carries
//! bit-packed bytes in the same layout as `valid`.

use crate::command::CommandQueue;
use crate::filter::{
    dispatch_predicate_to_u8, dispatch_scatter_bool, dispatch_scatter_f64, dispatch_scatter_i64,
    FilterError,
};
use polars_metal_buffer::MetalDevice;
use polars_metal_mlx_sys::cumsum_u8_to_u32;

/// Errors raised by the compaction pipeline.
///
/// Either a downstream kernel/dispatch failure (wrapped via
/// [`FilterError`]) or a failure inside the MLX cumsum FFI. We keep the
/// cumsum error as a `String` because `FfiError` is owned by a sibling
/// crate that does not (yet) need to be re-exported through
/// `polars-metal-kernels`'s public API.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// A filter kernel dispatcher returned an error (e.g. shader load,
    /// dispatch, buffer alloc, length mismatch).
    #[error(transparent)]
    Filter(#[from] FilterError),
    /// The MLX cumsum FFI returned an error (e.g. Metal unavailable,
    /// shape mismatch — though we guard against the latter here).
    #[error("MLX cumsum: {0}")]
    Cumsum(String),
}

/// Compacted output of a single source column.
///
/// `data` has exactly `n_out` elements. For i64/f64 these are the
/// surviving values in source-order; for the bool specialization
/// (`CompactionResult<u8>`), `data` is bit-packed (one bit per row,
/// 8 rows per byte) and its length in bytes is `dst_valid_min_bytes`.
///
/// `valid` is always bit-packed, 4-byte-aligned (rounded up from
/// `ceil(n_out / 8)`), minimum 4 bytes (the scatter kernel binds the
/// buffer as `device atomic_uint*`).
pub struct CompactionResult<T> {
    /// Surviving data values (or bit-packed bytes for bool).
    pub data: Vec<T>,
    /// Bit-packed validity bitmap. 4-byte-aligned, minimum 4 bytes.
    pub valid: Vec<u8>,
    /// Number of surviving rows (= `prefix_sum[n_rows - 1]`).
    pub n_out: usize,
}

/// Round up to the kernel's required validity-buffer size: `ceil(n / 8)`
/// rounded up to 4 bytes (for the u32 atomic cast), minimum 4 bytes.
/// Mirrors `filter::dst_valid_min_bytes`, which is private to that
/// module.
fn dst_valid_min_bytes(n_out: usize) -> usize {
    let raw = (n_out + 7) / 8;
    let padded = (raw + 3) & !3;
    padded.max(4)
}

/// Run the three-pass compaction pipeline on a single source column,
/// returning the compacted survivors. See module-level docs for the
/// pass-by-pass description.
///
/// Caller contract:
///   - `src_data.len() == n_rows`.
///   - `src_valid.len() >= ceil(n_rows / 8)`.
///   - `pred_data.len() >= ceil(n_rows / 8)`.
///   - `pred_valid.len() >= ceil(n_rows / 8)`.
///
/// `n_rows == 0` is accepted: returns an empty result without touching
/// Metal. Predicates that produce zero survivors short-circuit after
/// the prefix sum (no scatter dispatch).
pub fn compact_i64(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    src_data: &[i64],
    src_valid: &[u8],
    pred_data: &[u8],
    pred_valid: &[u8],
    n_rows: usize,
) -> Result<CompactionResult<i64>, PipelineError> {
    if n_rows == 0 {
        return Ok(CompactionResult {
            data: Vec::new(),
            valid: Vec::new(),
            n_out: 0,
        });
    }

    // Pass 1: bit-packed predicate (data ∧ valid) → dense u8 keep flags.
    let mut keep = vec![0u8; n_rows];
    dispatch_predicate_to_u8(device, queue, pred_data, pred_valid, n_rows, &mut keep)?;

    // Pass 2: MLX inclusive cumsum over the keep flags. The cumsum's
    // final element is the survivor count, which Pass 3 needs as
    // `n_out`.
    let mut prefix = vec![0u32; n_rows];
    cumsum_u8_to_u32(&keep, &mut prefix).map_err(|e| PipelineError::Cumsum(format!("{e:?}")))?;
    let n_out = prefix[n_rows - 1] as usize;
    if n_out == 0 {
        // No survivors — skip scatter, return an empty (well, empty
        // valid buffer minimum-aligned) result. Producing a `valid`
        // Vec at the kernel's alignment requirement would be lying:
        // there's no data to gate, so an empty `valid` is the honest
        // result.
        return Ok(CompactionResult {
            data: Vec::new(),
            valid: Vec::new(),
            n_out: 0,
        });
    }

    // Pass 3: scatter. The dispatcher requires `dst_data` to hold
    // `n_out + 1` slots (the extra slot is the overrun sentinel), and
    // `dst_valid` to be 4-byte-aligned (minimum 4 bytes).
    let mut dst_data = vec![0i64; n_out + 1];
    let valid_bytes = dst_valid_min_bytes(n_out);
    let mut dst_valid = vec![0u8; valid_bytes];

    dispatch_scatter_i64(
        device,
        queue,
        src_data,
        src_valid,
        &keep,
        &prefix,
        n_out,
        &mut dst_data,
        &mut dst_valid,
    )?;

    // Drop the sentinel slot so the caller sees a Vec of exactly
    // `n_out` survivors.
    dst_data.truncate(n_out);
    Ok(CompactionResult {
        data: dst_data,
        valid: dst_valid,
        n_out,
    })
}

/// f64 variant of [`compact_i64`]. Identical orchestration; the only
/// differences are the data slot size (still 8 bytes) and the scatter
/// dispatcher's NaN-bit-pattern sentinel. f64 bit patterns (including
/// NaN payloads, ±Inf, ±0.0, subnormals) round-trip exactly.
pub fn compact_f64(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    src_data: &[f64],
    src_valid: &[u8],
    pred_data: &[u8],
    pred_valid: &[u8],
    n_rows: usize,
) -> Result<CompactionResult<f64>, PipelineError> {
    if n_rows == 0 {
        return Ok(CompactionResult {
            data: Vec::new(),
            valid: Vec::new(),
            n_out: 0,
        });
    }

    let mut keep = vec![0u8; n_rows];
    dispatch_predicate_to_u8(device, queue, pred_data, pred_valid, n_rows, &mut keep)?;

    let mut prefix = vec![0u32; n_rows];
    cumsum_u8_to_u32(&keep, &mut prefix).map_err(|e| PipelineError::Cumsum(format!("{e:?}")))?;
    let n_out = prefix[n_rows - 1] as usize;
    if n_out == 0 {
        return Ok(CompactionResult {
            data: Vec::new(),
            valid: Vec::new(),
            n_out: 0,
        });
    }

    // Pre-fill with +0.0 (bit pattern 0) — the scatter dispatcher
    // requires the sentinel slot to start at 0 so its NaN-pattern
    // sentinel check has a deterministic baseline.
    let mut dst_data = vec![0.0f64; n_out + 1];
    let valid_bytes = dst_valid_min_bytes(n_out);
    let mut dst_valid = vec![0u8; valid_bytes];

    dispatch_scatter_f64(
        device,
        queue,
        src_data,
        src_valid,
        &keep,
        &prefix,
        n_out,
        &mut dst_data,
        &mut dst_valid,
    )?;

    dst_data.truncate(n_out);
    Ok(CompactionResult {
        data: dst_data,
        valid: dst_valid,
        n_out,
    })
}

/// Bool variant of [`compact_i64`]. Bool is the only scatter dtype
/// where the data buffer is itself bit-packed (one bit per row), so
/// the returned `data` field is a `Vec<u8>` carrying bit-packed bytes
/// in the same layout as `valid`. Its length in bytes is the same
/// 4-byte-aligned `dst_valid_min_bytes(n_out)` as the validity buffer
/// (the scatter kernel binds both as `device atomic_uint*`).
///
/// Caller contract additions over [`compact_i64`]:
///   - `src_data.len() >= ceil(n_rows / 8)` (instead of `== n_rows`),
///     because the source data is also bit-packed.
///
/// The bool scatter dispatcher carries no overrun sentinel (every bit
/// pattern is a legitimate bool value); the pipeline's invariant
/// `prefix[n_rows - 1] == n_out` is enforced by the dispatcher
/// itself.
pub fn compact_bool(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    src_data: &[u8],
    src_valid: &[u8],
    pred_data: &[u8],
    pred_valid: &[u8],
    n_rows: usize,
) -> Result<CompactionResult<u8>, PipelineError> {
    if n_rows == 0 {
        return Ok(CompactionResult {
            data: Vec::new(),
            valid: Vec::new(),
            n_out: 0,
        });
    }

    let mut keep = vec![0u8; n_rows];
    dispatch_predicate_to_u8(device, queue, pred_data, pred_valid, n_rows, &mut keep)?;

    let mut prefix = vec![0u32; n_rows];
    cumsum_u8_to_u32(&keep, &mut prefix).map_err(|e| PipelineError::Cumsum(format!("{e:?}")))?;
    let n_out = prefix[n_rows - 1] as usize;
    if n_out == 0 {
        return Ok(CompactionResult {
            data: Vec::new(),
            valid: Vec::new(),
            n_out: 0,
        });
    }

    // Both `dst_data` and `dst_valid` are 4-byte-aligned bit-packed
    // buffers in this variant — see the bool scatter dispatcher's
    // caller contract.
    let out_bytes = dst_valid_min_bytes(n_out);
    let mut dst_data = vec![0u8; out_bytes];
    let mut dst_valid = vec![0u8; out_bytes];

    dispatch_scatter_bool(
        device,
        queue,
        src_data,
        src_valid,
        &keep,
        &prefix,
        n_rows,
        n_out,
        &mut dst_data,
        &mut dst_valid,
    )?;

    Ok(CompactionResult {
        data: dst_data,
        valid: dst_valid,
        n_out,
    })
}
