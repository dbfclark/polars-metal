// crates/polars-metal-kernels/src/groupby.rs
//! Composite key encoding for the GroupBy hash kernel.
//!
//! Each key column contributes (1-bit null flag) + (dtype-width-bits) to
//! a u128 lane per row. Lane layout, from LSB to MSB:
//!
//! ```text
//! bit 0:                      key0.null
//! bits 1..1+w(key0):          key0.data
//! bit 1+w(key0):              key1.null
//! bits 2+w(key0)..2+w(key0)+w(key1):  key1.data
//! ...
//! ```
//!
//! The layout is deterministic given the input column list — the same
//! column order yields the same encoding, byte-for-byte. The hash kernel
//! consumes the raw u128; key equality is u128 equality.
//!
//! Why a single u128 lane:
//!   - One atomic-CAS op per row in the build phase (Metal supports 128-bit
//!     atomic CAS on Apple Silicon GPUs since M2; for M1 we fall back to a
//!     spinlock per slot — addressed in spec § "Risks").
//!   - No per-row dynamic allocation; each row is a single 16-byte read.
//!
//! Width budget: 128 bits per row. Common cases that fit:
//!   - 1 × i64 + up to 63 booleans
//!   - 2 × i32 (planned for M3) + null bits
//!   - 1 × i64 + 1 × bool (Q1's shape with the integer-encoded keys: l_returnflag, l_linestatus)
//!
//! Wider key sets must `Fallback` at plan time (router-side) or surface
//! `KeyEncodeError::TooWide` at dispatch time (defensive — router should
//! catch first).

use thiserror::Error;

/// Supported key dtypes. Mirrors `MetalDtype` but lives in this crate so
/// the kernel layer has no dependency on the engine-adapter crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyDtype {
    I64,
    F64,
    Bool,
}

impl KeyDtype {
    /// Width in bits of the data payload (excludes the null flag).
    pub fn data_bits(self) -> u32 {
        match self {
            KeyDtype::I64 | KeyDtype::F64 => 64,
            KeyDtype::Bool => 1,
        }
    }
}

/// One input column to the encoder. Carries the raw data + validity
/// bytes — the encoder doesn't own the buffers.
pub struct KeyColumn<'a> {
    pub name: String,
    pub dtype: KeyDtype,
    /// Little-endian packed values. For I64/F64: 8 bytes per row. For
    /// Bool: one bit per row, bit-packed, same convention as Arrow's
    /// validity bitmap (`bit i` of `byte i/8` at offset `i%8`).
    pub data: &'a [u8],
    /// Bit-packed validity bitmap, `ceil(n_rows / 8)` bytes minimum.
    pub valid: &'a [u8],
    pub n_rows: usize,
}

/// One field's position in the encoded u128 lane. Both fields and
/// schemas are immutable after construction.
#[derive(Debug, Clone)]
pub struct KeyField {
    pub name: String,
    pub dtype: KeyDtype,
    /// Bit position of this field's null flag in the u128 lane.
    pub null_bit_offset: u32,
    /// Bit position of this field's data, immediately following the null bit.
    pub data_bit_offset: u32,
}

/// Schema for a composite-key encoding. Sufficient to decode an encoded
/// u128 stream back to per-column values.
#[derive(Debug, Clone)]
pub struct KeySchema {
    fields: Vec<KeyField>,
    total_bits: u32,
    n_rows: usize,
}

impl KeySchema {
    pub fn fields(&self) -> &[KeyField] {
        &self.fields
    }
    pub fn total_bits(&self) -> u32 {
        self.total_bits
    }
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }
}

/// Error returned by `encode_keys`.
#[derive(Debug, Error)]
pub enum KeyEncodeError {
    #[error("no key columns provided")]
    NoKeys,
    #[error("composite key width {total_bits} bits exceeds 128-bit budget")]
    TooWide { total_bits: u32 },
    #[error("row count mismatch across key columns: first={first}, mismatched={mismatched}")]
    RowCountMismatch { first: usize, mismatched: usize },
    #[error("data buffer for {col:?} too short: got {got} bytes, need {need}")]
    DataTooShort {
        col: String,
        got: usize,
        need: usize,
    },
    #[error("validity buffer for {col:?} too short: got {got} bytes, need {need}")]
    ValidityTooShort {
        col: String,
        got: usize,
        need: usize,
    },
}

/// Encode `cols` to a `Vec<u128>` (one u128 per row). Returns the
/// encoded data and the schema needed to decode.
pub fn encode_keys(cols: &[KeyColumn<'_>]) -> Result<(Vec<u128>, KeySchema), KeyEncodeError> {
    if cols.is_empty() {
        return Err(KeyEncodeError::NoKeys);
    }
    let n_rows = cols[0].n_rows;
    for c in cols.iter().skip(1) {
        if c.n_rows != n_rows {
            return Err(KeyEncodeError::RowCountMismatch {
                first: n_rows,
                mismatched: c.n_rows,
            });
        }
    }

    let mut fields = Vec::with_capacity(cols.len());
    let mut offset: u32 = 0;
    let min_valid_bytes = (n_rows + 7) / 8;
    for c in cols {
        let need_data = match c.dtype {
            KeyDtype::I64 | KeyDtype::F64 => n_rows * 8,
            KeyDtype::Bool => min_valid_bytes,
        };
        if c.data.len() < need_data {
            return Err(KeyEncodeError::DataTooShort {
                col: c.name.clone(),
                got: c.data.len(),
                need: need_data,
            });
        }
        if c.valid.len() < min_valid_bytes {
            return Err(KeyEncodeError::ValidityTooShort {
                col: c.name.clone(),
                got: c.valid.len(),
                need: min_valid_bytes,
            });
        }

        let null_bit_offset = offset;
        let data_bit_offset = offset + 1;
        let field_bits = 1 + c.dtype.data_bits();
        if offset.saturating_add(field_bits) > 128 {
            return Err(KeyEncodeError::TooWide {
                total_bits: offset + field_bits,
            });
        }
        fields.push(KeyField {
            name: c.name.clone(),
            dtype: c.dtype,
            null_bit_offset,
            data_bit_offset,
        });
        offset += field_bits;
    }
    let total_bits = offset;

    let mut encoded = vec![0u128; n_rows];
    for (field_idx, c) in cols.iter().enumerate() {
        let field = &fields[field_idx];
        for (row, lane) in encoded.iter_mut().enumerate() {
            let valid_byte = c.valid[row >> 3];
            let valid_bit = (valid_byte >> (row & 7)) & 1;
            if valid_bit == 0 {
                *lane |= 1u128 << field.null_bit_offset;
                continue;
            }
            let data_value: u128 = match c.dtype {
                KeyDtype::I64 => {
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&c.data[row * 8..(row + 1) * 8]);
                    i64::from_le_bytes(bytes) as u64 as u128
                }
                KeyDtype::F64 => {
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&c.data[row * 8..(row + 1) * 8]);
                    f64::from_le_bytes(bytes).to_bits() as u128
                }
                KeyDtype::Bool => {
                    let byte = c.data[row >> 3];
                    let bit = (byte >> (row & 7)) & 1;
                    bit as u128
                }
            };
            *lane |= data_value << field.data_bit_offset;
        }
    }

    Ok((
        encoded,
        KeySchema {
            fields,
            total_bits,
            n_rows,
        },
    ))
}

// -----------------------------------------------------------------------
// dispatch_hash — Rust dispatcher for the `groupby_hash` MSL kernel
// -----------------------------------------------------------------------

use crate::command::{CommandQueue, DispatchError};
use crate::shader_lib::{shared_library, ShaderError};
use polars_metal_buffer::{BufferError, MetalDevice};

/// Errors raised by the groupby dispatchers.
#[derive(Debug, thiserror::Error)]
pub enum GroupByError {
    #[error("key encoding: {0}")]
    KeyEncode(#[from] KeyEncodeError),
    #[error("shader library: {0}")]
    Shader(#[from] ShaderError),
    #[error("dispatch: {0}")]
    Dispatch(#[from] DispatchError),
    #[error("buffer: {0}")]
    Buffer(#[from] BufferError),
    #[error("output buffer too short: got {got}, need {need}")]
    OutputTooShort { got: usize, need: usize },
    #[error("n_rows {n_rows} exceeds u32::MAX")]
    RowCountOverflow { n_rows: usize },
}

/// Dispatch the `groupby_hash` kernel.
///
/// Reads `encoded[0..n_rows]` (one u128 per row), writes one u32 hash per
/// row to `hashes[0..n_rows]`.
///
/// `n_rows == 0` is a no-op (Metal rejects zero-byte buffers and zero-grid
/// dispatches; we short-circuit cleanly here).
pub fn dispatch_hash(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    encoded: &[u128],
    n_rows: usize,
    hashes: &mut [u32],
) -> Result<(), GroupByError> {
    if n_rows == 0 {
        return Ok(());
    }

    let n_rows_u32: u32 =
        u32::try_from(n_rows).map_err(|_| GroupByError::RowCountOverflow { n_rows })?;

    if encoded.len() < n_rows {
        return Err(GroupByError::OutputTooShort {
            got: encoded.len(),
            need: n_rows,
        });
    }
    if hashes.len() < n_rows {
        return Err(GroupByError::OutputTooShort {
            got: hashes.len(),
            need: n_rows,
        });
    }

    // SAFETY: `u128` is plain-old-data with no invalid bit patterns.
    // We reinterpret a live `&[u128]` as a `&[u8]` of length `n_rows * 16`.
    // The slice is alive for the duration of this call; `new_buffer_from_bytes`
    // copies bytes synchronously into a freshly allocated MTLBuffer and does
    // not retain the reference past the call.
    let key_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            encoded.as_ptr() as *const u8,
            n_rows * std::mem::size_of::<u128>(),
        )
    };

    let lib = shared_library(device)?;
    let pso = lib.pipeline("groupby_hash")?;

    let keys_buf = device.new_buffer_from_bytes(key_bytes)?;
    let hashes_buf = device.new_buffer_zeroed(n_rows * std::mem::size_of::<u32>())?;
    let n_rows_buf = device.new_buffer_from_bytes(&n_rows_u32.to_le_bytes())?;

    queue.dispatch_1d(&pso, &[&keys_buf, &hashes_buf, &n_rows_buf], n_rows)?;
    queue.wait_until_complete()?;

    // Copy results back. `as_slice()` returns `&[u8]`; decode to `u32`.
    let out_bytes = hashes_buf.as_slice();
    let need_bytes = n_rows * std::mem::size_of::<u32>();
    if out_bytes.len() < need_bytes {
        return Err(GroupByError::OutputTooShort {
            got: out_bytes.len(),
            need: need_bytes,
        });
    }
    for (i, h) in hashes.iter_mut().take(n_rows).enumerate() {
        let b = &out_bytes[i * 4..(i + 1) * 4];
        *h = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
    Ok(())
}

/// Initial load factor for the build-phase hash table.
/// table_size = next_pow2(n_rows * BUILD_LOAD_FACTOR_DEN / BUILD_LOAD_FACTOR_NUM).
pub const BUILD_LOAD_FACTOR_NUM: usize = 1;
pub const BUILD_LOAD_FACTOR_DEN: usize = 2;

fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let bits = usize::BITS - (n - 1).leading_zeros();
    1usize << bits
}

/// Output of the build phase.
pub struct BuildOutput {
    /// `row_to_group[i]` = group ID for row i.
    pub row_to_group: Vec<u32>,
    /// Total number of distinct groups produced by the build.
    pub group_count: u32,
    /// `first_row_per_group[g]` = a representative source-row index for
    /// group g, used to reconstruct key columns in the result.
    pub first_row_per_group: Vec<u32>,
}

/// Dispatch the `groupby_build` kernel.
///
/// Slots use a 3-state machine (`atomic_uint`): 0=EMPTY, 1=CLAIMED, 2=READY.
/// Keys are stored as four `atomic_uint` words (lo_lo, lo_hi, hi_lo, hi_hi).
/// This layout avoids `atomic_ulong` CAS which is unsupported on Apple Silicon
/// Metal compute kernels (verified: Apple metal 32023.883).
pub fn dispatch_build(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    encoded: &[u128],
    hashes: &[u32],
    n_rows: usize,
) -> Result<BuildOutput, GroupByError> {
    if n_rows == 0 {
        return Ok(BuildOutput {
            row_to_group: vec![],
            group_count: 0,
            first_row_per_group: vec![],
        });
    }
    let n_rows_u32: u32 =
        u32::try_from(n_rows).map_err(|_| GroupByError::RowCountOverflow { n_rows })?;

    if encoded.len() < n_rows {
        return Err(GroupByError::OutputTooShort {
            got: encoded.len(),
            need: n_rows,
        });
    }
    if hashes.len() < n_rows {
        return Err(GroupByError::OutputTooShort {
            got: hashes.len(),
            need: n_rows,
        });
    }

    // table_size = next_pow2(n_rows * 2), minimum 2.
    let raw_size = n_rows
        .checked_mul(BUILD_LOAD_FACTOR_DEN)
        .and_then(|n| n.checked_div(BUILD_LOAD_FACTOR_NUM))
        .ok_or(GroupByError::RowCountOverflow { n_rows })?;
    let table_size = next_pow2(raw_size).max(2);
    let table_size_u32: u32 = u32::try_from(table_size)
        .map_err(|_| GroupByError::RowCountOverflow { n_rows: table_size })?;

    // SAFETY: `u128` and `u32` are plain-old-data with no invalid bit patterns.
    // We reinterpret live slices as `&[u8]` for the duration of this call.
    // `new_buffer_from_bytes` copies bytes synchronously; the references do not
    // escape this function.
    let key_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(encoded.as_ptr() as *const u8, n_rows * 16) };
    let hash_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(hashes.as_ptr() as *const u8, n_rows * 4) };

    let lib = shared_library(device)?;
    let pso = lib.pipeline("groupby_build")?;

    let keys_buf = device.new_buffer_from_bytes(key_bytes)?;
    let hashes_buf = device.new_buffer_from_bytes(hash_bytes)?;
    // slot_state:    table_size × atomic_uint (4 bytes each)
    let slot_state_buf = device.new_buffer_zeroed(table_size * 4)?;
    // slot_key:      table_size × 4 × atomic_uint (16 bytes per slot)
    let slot_key_buf = device.new_buffer_zeroed(table_size * 16)?;
    // slot_group_id: table_size × atomic_uint (4 bytes each)
    let slot_gid_buf = device.new_buffer_zeroed(table_size * 4)?;
    // group_count:   1 × atomic_uint (4 bytes)
    let group_count_buf = device.new_buffer_zeroed(4)?;
    // first_row_per_group: n_rows × u32 — overallocated; truncated on output
    let first_row_buf = device.new_buffer_zeroed(n_rows * 4)?;
    // row_to_group:  n_rows × u32
    let row_to_group_buf = device.new_buffer_zeroed(n_rows * 4)?;
    let n_rows_buf = device.new_buffer_from_bytes(&n_rows_u32.to_le_bytes())?;
    let table_size_buf = device.new_buffer_from_bytes(&table_size_u32.to_le_bytes())?;

    queue.dispatch_1d(
        &pso,
        &[
            &keys_buf,
            &hashes_buf,
            &slot_state_buf,
            &slot_key_buf,
            &slot_gid_buf,
            &group_count_buf,
            &first_row_buf,
            &row_to_group_buf,
            &n_rows_buf,
            &table_size_buf,
        ],
        n_rows,
    )?;
    queue.wait_until_complete()?;

    let group_count = {
        let b = group_count_buf.as_slice();
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    };

    let row_to_group = read_u32_vec(&row_to_group_buf, n_rows)?;
    let mut first_row_full = read_u32_vec(&first_row_buf, n_rows)?;
    first_row_full.truncate(group_count as usize);

    Ok(BuildOutput {
        row_to_group,
        group_count,
        first_row_per_group: first_row_full,
    })
}

/// Copy back `count` u32 values from a Metal buffer.
fn read_u32_vec(
    buf: &polars_metal_buffer::MetalBuffer,
    count: usize,
) -> Result<Vec<u32>, GroupByError> {
    let bytes = buf.as_slice();
    let need = count * 4;
    if bytes.len() < need {
        return Err(GroupByError::OutputTooShort {
            got: bytes.len(),
            need,
        });
    }
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let b = &bytes[i * 4..(i + 1) * 4];
        out.push(u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    }
    Ok(out)
}

// -----------------------------------------------------------------------
// 32-bit GPU aggregation dispatchers
// -----------------------------------------------------------------------

/// Shared helper: seed + dispatch an i32 aggregation kernel, copy results back.
#[allow(clippy::too_many_arguments)]
fn dispatch_agg_i32(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    values: &[i32],
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [i32],
    kernel_name: &str,
    init_value: i32,
) -> Result<(), GroupByError> {
    if n_rows == 0 {
        return Ok(());
    }
    let n_rows_u32: u32 =
        u32::try_from(n_rows).map_err(|_| GroupByError::RowCountOverflow { n_rows })?;
    let valid_bytes = (n_rows + 7) / 8;
    if values.len() < n_rows {
        return Err(GroupByError::OutputTooShort {
            got: values.len(),
            need: n_rows,
        });
    }
    if row_to_group.len() < n_rows {
        return Err(GroupByError::OutputTooShort {
            got: row_to_group.len(),
            need: n_rows,
        });
    }
    if valid.len() < valid_bytes {
        return Err(GroupByError::OutputTooShort {
            got: valid.len(),
            need: valid_bytes,
        });
    }
    if out.len() < n_groups {
        return Err(GroupByError::OutputTooShort {
            got: out.len(),
            need: n_groups,
        });
    }

    // SAFETY: `i32` is plain-old-data; reinterpret as bytes for the duration
    // of this call. `new_buffer_from_bytes` copies synchronously.
    let values_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, n_rows * 4) };
    let r2g_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(row_to_group.as_ptr() as *const u8, n_rows * 4) };

    let init_buf: Vec<i32> = vec![init_value; n_groups];
    // SAFETY: `i32` plain-old-data; same as above.
    let init_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(init_buf.as_ptr() as *const u8, n_groups * 4) };

    let lib = shared_library(device)?;
    let pso = lib.pipeline(kernel_name)?;

    let vals_buf = device.new_buffer_from_bytes(values_bytes)?;
    let valid_buf = device.new_buffer_from_bytes(&valid[..valid_bytes])?;
    let r2g_buf = device.new_buffer_from_bytes(r2g_bytes)?;
    let out_buf = device.new_buffer_from_bytes(init_bytes)?;
    let n_rows_buf = device.new_buffer_from_bytes(&n_rows_u32.to_le_bytes())?;

    queue.dispatch_1d(
        &pso,
        &[&vals_buf, &valid_buf, &r2g_buf, &out_buf, &n_rows_buf],
        n_rows,
    )?;
    queue.wait_until_complete()?;

    let out_bytes = out_buf.as_slice();
    let need_bytes = n_groups * 4;
    if out_bytes.len() < need_bytes {
        return Err(GroupByError::OutputTooShort {
            got: out_bytes.len(),
            need: need_bytes,
        });
    }
    for (i, v) in out.iter_mut().take(n_groups).enumerate() {
        let b = &out_bytes[i * 4..(i + 1) * 4];
        *v = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
    Ok(())
}

/// Dispatch `agg_sum_i32`.
///
/// `values`, `valid`, `row_to_group` are slices of length ≥ `n_rows`.
/// `out` is a slice of length ≥ `n_groups`, value on entry is overwritten
/// with the per-group sum (null rows contribute 0, all-null group → 0).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_sum_i32(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    values: &[i32],
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [i32],
) -> Result<(), GroupByError> {
    dispatch_agg_i32(
        device,
        queue,
        values,
        valid,
        row_to_group,
        n_rows,
        n_groups,
        out,
        "agg_sum_i32",
        0,
    )
}

/// Dispatch `agg_min_i32`. Seeded with `i32::MAX`; groups with no valid rows
/// retain `i32::MAX` — callers must apply a validity mask (see
/// `dispatch_count_u32`) to detect all-null groups.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_min_i32(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    values: &[i32],
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [i32],
) -> Result<(), GroupByError> {
    dispatch_agg_i32(
        device,
        queue,
        values,
        valid,
        row_to_group,
        n_rows,
        n_groups,
        out,
        "agg_min_i32",
        i32::MAX,
    )
}

/// Dispatch `agg_max_i32`. Seeded with `i32::MIN`; groups with no valid rows
/// retain `i32::MIN`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_max_i32(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    values: &[i32],
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [i32],
) -> Result<(), GroupByError> {
    dispatch_agg_i32(
        device,
        queue,
        values,
        valid,
        row_to_group,
        n_rows,
        n_groups,
        out,
        "agg_max_i32",
        i32::MIN,
    )
}

// ---- u32 GPU aggregation ----

/// Shared helper for u32 aggregation kernels.
#[allow(clippy::too_many_arguments)]
fn dispatch_agg_u32(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    values: &[u32],
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [u32],
    kernel_name: &str,
    init_value: u32,
) -> Result<(), GroupByError> {
    if n_rows == 0 {
        return Ok(());
    }
    let n_rows_u32: u32 =
        u32::try_from(n_rows).map_err(|_| GroupByError::RowCountOverflow { n_rows })?;
    let valid_bytes = (n_rows + 7) / 8;
    if values.len() < n_rows {
        return Err(GroupByError::OutputTooShort {
            got: values.len(),
            need: n_rows,
        });
    }
    if row_to_group.len() < n_rows {
        return Err(GroupByError::OutputTooShort {
            got: row_to_group.len(),
            need: n_rows,
        });
    }
    if valid.len() < valid_bytes {
        return Err(GroupByError::OutputTooShort {
            got: valid.len(),
            need: valid_bytes,
        });
    }
    if out.len() < n_groups {
        return Err(GroupByError::OutputTooShort {
            got: out.len(),
            need: n_groups,
        });
    }

    // SAFETY: `u32` is plain-old-data.
    let values_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, n_rows * 4) };
    let r2g_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(row_to_group.as_ptr() as *const u8, n_rows * 4) };
    let init_buf: Vec<u32> = vec![init_value; n_groups];
    // SAFETY: `u32` plain-old-data.
    let init_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(init_buf.as_ptr() as *const u8, n_groups * 4) };

    let lib = shared_library(device)?;
    let pso = lib.pipeline(kernel_name)?;

    let vals_buf = device.new_buffer_from_bytes(values_bytes)?;
    let valid_buf = device.new_buffer_from_bytes(&valid[..valid_bytes])?;
    let r2g_buf = device.new_buffer_from_bytes(r2g_bytes)?;
    let out_buf = device.new_buffer_from_bytes(init_bytes)?;
    let n_rows_buf = device.new_buffer_from_bytes(&n_rows_u32.to_le_bytes())?;

    queue.dispatch_1d(
        &pso,
        &[&vals_buf, &valid_buf, &r2g_buf, &out_buf, &n_rows_buf],
        n_rows,
    )?;
    queue.wait_until_complete()?;

    let out_bytes = out_buf.as_slice();
    let need_bytes = n_groups * 4;
    if out_bytes.len() < need_bytes {
        return Err(GroupByError::OutputTooShort {
            got: out_bytes.len(),
            need: need_bytes,
        });
    }
    for (i, v) in out.iter_mut().take(n_groups).enumerate() {
        let b = &out_bytes[i * 4..(i + 1) * 4];
        *v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
    Ok(())
}

/// Dispatch `agg_sum_u32`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_sum_u32(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    values: &[u32],
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [u32],
) -> Result<(), GroupByError> {
    dispatch_agg_u32(
        device,
        queue,
        values,
        valid,
        row_to_group,
        n_rows,
        n_groups,
        out,
        "agg_sum_u32",
        0,
    )
}

/// Dispatch `agg_min_u32`. Seeded with `u32::MAX`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_min_u32(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    values: &[u32],
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [u32],
) -> Result<(), GroupByError> {
    dispatch_agg_u32(
        device,
        queue,
        values,
        valid,
        row_to_group,
        n_rows,
        n_groups,
        out,
        "agg_min_u32",
        u32::MAX,
    )
}

/// Dispatch `agg_max_u32`. Seeded with `0`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_max_u32(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    values: &[u32],
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [u32],
) -> Result<(), GroupByError> {
    dispatch_agg_u32(
        device,
        queue,
        values,
        valid,
        row_to_group,
        n_rows,
        n_groups,
        out,
        "agg_max_u32",
        0,
    )
}

// ---- f32 GPU aggregation ----

/// Shared helper for f32 aggregation kernels.
/// The MSL kernels use `atomic_uint` as a bit-pattern container for f32;
/// the init value is passed as the bit pattern of the f32 identity element.
#[allow(clippy::too_many_arguments)]
fn dispatch_agg_f32(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    values: &[f32],
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [f32],
    kernel_name: &str,
    init_bits: u32,
) -> Result<(), GroupByError> {
    if n_rows == 0 {
        return Ok(());
    }
    let n_rows_u32: u32 =
        u32::try_from(n_rows).map_err(|_| GroupByError::RowCountOverflow { n_rows })?;
    let valid_bytes = (n_rows + 7) / 8;
    if values.len() < n_rows {
        return Err(GroupByError::OutputTooShort {
            got: values.len(),
            need: n_rows,
        });
    }
    if row_to_group.len() < n_rows {
        return Err(GroupByError::OutputTooShort {
            got: row_to_group.len(),
            need: n_rows,
        });
    }
    if valid.len() < valid_bytes {
        return Err(GroupByError::OutputTooShort {
            got: valid.len(),
            need: valid_bytes,
        });
    }
    if out.len() < n_groups {
        return Err(GroupByError::OutputTooShort {
            got: out.len(),
            need: n_groups,
        });
    }

    // SAFETY: `f32` is plain-old-data; bit patterns are stable.
    let values_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, n_rows * 4) };
    let r2g_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(row_to_group.as_ptr() as *const u8, n_rows * 4) };
    // Seed as u32 bit patterns; the kernel interprets them as f32 via as_type<float>.
    let init_buf: Vec<u32> = vec![init_bits; n_groups];
    // SAFETY: `u32` plain-old-data.
    let init_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(init_buf.as_ptr() as *const u8, n_groups * 4) };

    let lib = shared_library(device)?;
    let pso = lib.pipeline(kernel_name)?;

    let vals_buf = device.new_buffer_from_bytes(values_bytes)?;
    let valid_buf = device.new_buffer_from_bytes(&valid[..valid_bytes])?;
    let r2g_buf = device.new_buffer_from_bytes(r2g_bytes)?;
    let out_buf = device.new_buffer_from_bytes(init_bytes)?;
    let n_rows_buf = device.new_buffer_from_bytes(&n_rows_u32.to_le_bytes())?;

    queue.dispatch_1d(
        &pso,
        &[&vals_buf, &valid_buf, &r2g_buf, &out_buf, &n_rows_buf],
        n_rows,
    )?;
    queue.wait_until_complete()?;

    let out_bytes = out_buf.as_slice();
    let need_bytes = n_groups * 4;
    if out_bytes.len() < need_bytes {
        return Err(GroupByError::OutputTooShort {
            got: out_bytes.len(),
            need: need_bytes,
        });
    }
    for (i, v) in out.iter_mut().take(n_groups).enumerate() {
        let b = &out_bytes[i * 4..(i + 1) * 4];
        *v = f32::from_bits(u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    }
    Ok(())
}

/// Dispatch `agg_sum_f32`. Seeded with 0.0.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_sum_f32(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    values: &[f32],
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [f32],
) -> Result<(), GroupByError> {
    dispatch_agg_f32(
        device,
        queue,
        values,
        valid,
        row_to_group,
        n_rows,
        n_groups,
        out,
        "agg_sum_f32",
        0u32, // 0.0f32.to_bits()
    )
}

/// Dispatch `agg_min_f32`. Seeded with +INFINITY.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_min_f32(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    values: &[f32],
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [f32],
) -> Result<(), GroupByError> {
    dispatch_agg_f32(
        device,
        queue,
        values,
        valid,
        row_to_group,
        n_rows,
        n_groups,
        out,
        "agg_min_f32",
        f32::INFINITY.to_bits(),
    )
}

/// Dispatch `agg_max_f32`. Seeded with -INFINITY.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_max_f32(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    values: &[f32],
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [f32],
) -> Result<(), GroupByError> {
    dispatch_agg_f32(
        device,
        queue,
        values,
        valid,
        row_to_group,
        n_rows,
        n_groups,
        out,
        "agg_max_f32",
        f32::NEG_INFINITY.to_bits(),
    )
}

// ---- count / len GPU dispatchers ----

/// Dispatch `agg_count`: per-group count of non-null rows.
pub fn dispatch_count_u32(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    valid: &[u8],
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [u32],
) -> Result<(), GroupByError> {
    if n_rows == 0 {
        return Ok(());
    }
    let n_rows_u32: u32 =
        u32::try_from(n_rows).map_err(|_| GroupByError::RowCountOverflow { n_rows })?;
    let valid_bytes = (n_rows + 7) / 8;
    if row_to_group.len() < n_rows {
        return Err(GroupByError::OutputTooShort {
            got: row_to_group.len(),
            need: n_rows,
        });
    }
    if valid.len() < valid_bytes {
        return Err(GroupByError::OutputTooShort {
            got: valid.len(),
            need: valid_bytes,
        });
    }
    if out.len() < n_groups {
        return Err(GroupByError::OutputTooShort {
            got: out.len(),
            need: n_groups,
        });
    }

    // SAFETY: `u32` plain-old-data.
    let r2g_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(row_to_group.as_ptr() as *const u8, n_rows * 4) };
    let init_bytes: Vec<u8> = vec![0u8; n_groups * 4];

    let lib = shared_library(device)?;
    let pso = lib.pipeline("agg_count")?;

    let valid_buf = device.new_buffer_from_bytes(&valid[..valid_bytes])?;
    let r2g_buf = device.new_buffer_from_bytes(r2g_bytes)?;
    let out_buf = device.new_buffer_from_bytes(&init_bytes)?;
    let n_rows_buf = device.new_buffer_from_bytes(&n_rows_u32.to_le_bytes())?;

    queue.dispatch_1d(&pso, &[&valid_buf, &r2g_buf, &out_buf, &n_rows_buf], n_rows)?;
    queue.wait_until_complete()?;

    let out_bytes = out_buf.as_slice();
    let need_bytes = n_groups * 4;
    if out_bytes.len() < need_bytes {
        return Err(GroupByError::OutputTooShort {
            got: out_bytes.len(),
            need: need_bytes,
        });
    }
    for (i, v) in out.iter_mut().take(n_groups).enumerate() {
        let b = &out_bytes[i * 4..(i + 1) * 4];
        *v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
    Ok(())
}

/// Dispatch `agg_len`: per-group total row count (includes null rows).
pub fn dispatch_len_u32(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    row_to_group: &[u32],
    n_rows: usize,
    n_groups: usize,
    out: &mut [u32],
) -> Result<(), GroupByError> {
    if n_rows == 0 {
        return Ok(());
    }
    let n_rows_u32: u32 =
        u32::try_from(n_rows).map_err(|_| GroupByError::RowCountOverflow { n_rows })?;
    if row_to_group.len() < n_rows {
        return Err(GroupByError::OutputTooShort {
            got: row_to_group.len(),
            need: n_rows,
        });
    }
    if out.len() < n_groups {
        return Err(GroupByError::OutputTooShort {
            got: out.len(),
            need: n_groups,
        });
    }

    // SAFETY: `u32` plain-old-data.
    let r2g_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(row_to_group.as_ptr() as *const u8, n_rows * 4) };
    let init_bytes: Vec<u8> = vec![0u8; n_groups * 4];

    let lib = shared_library(device)?;
    let pso = lib.pipeline("agg_len")?;

    let r2g_buf = device.new_buffer_from_bytes(r2g_bytes)?;
    let out_buf = device.new_buffer_from_bytes(&init_bytes)?;
    let n_rows_buf = device.new_buffer_from_bytes(&n_rows_u32.to_le_bytes())?;

    queue.dispatch_1d(&pso, &[&r2g_buf, &out_buf, &n_rows_buf], n_rows)?;
    queue.wait_until_complete()?;

    let out_bytes = out_buf.as_slice();
    let need_bytes = n_groups * 4;
    if out_bytes.len() < need_bytes {
        return Err(GroupByError::OutputTooShort {
            got: out_bytes.len(),
            need: need_bytes,
        });
    }
    for (i, v) in out.iter_mut().take(n_groups).enumerate() {
        let b = &out_bytes[i * 4..(i + 1) * 4];
        *v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
    Ok(())
}

// -----------------------------------------------------------------------
// CPU-finalize aggregators for 64-bit dtypes
//
// GPU produced `row_to_group` in Phase 5. For i64/f64/u64 dtypes the
// aggregation runs in Rust+Rayon because the Metal toolchain 32023.883
// does not support 64-bit atomic ops in compute kernels.
//
// Null semantics match Polars CPU exactly:
//   - sum:   null rows skipped; all-null group → 0 (i64) / 0.0 (f64).
//   - count: null rows skipped; all-null group → 0.
//   - len:   all rows counted (ignores validity).
//   - min/max: null rows skipped; all-null group → None.
// -----------------------------------------------------------------------

/// Sum f64 values by group. Null rows skipped. All-null group → 0.0.
///
/// Uses AtomicU64 CAS loops (Rayon parallel) because AtomicF64 doesn't
/// exist in stable `std`.
pub fn aggregate_sum_f64_cpu(
    values: &[f64],
    valid: &[u8],
    row_to_group: &[u32],
    n_groups: usize,
) -> Vec<f64> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    let slots: Vec<AtomicU64> = (0..n_groups).map(|_| AtomicU64::new(0)).collect();
    let n = values.len().min(row_to_group.len());
    values[..n].par_iter().enumerate().for_each(|(i, &v)| {
        let byte = valid.get(i >> 3).copied().unwrap_or(0);
        if (byte >> (i & 7)) & 1 == 0 {
            return;
        }
        let g = row_to_group[i] as usize;
        if g >= n_groups {
            return;
        }
        let mut old_bits = slots[g].load(Ordering::Relaxed);
        loop {
            let cur = f64::from_bits(old_bits);
            let next_bits = (cur + v).to_bits();
            match slots[g].compare_exchange_weak(
                old_bits,
                next_bits,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(latest) => old_bits = latest,
            }
        }
    });
    slots
        .into_iter()
        .map(|a| f64::from_bits(a.into_inner()))
        .collect()
}

/// Sum i64 values by group. Null rows skipped. All-null group → 0.
pub fn aggregate_sum_i64_cpu(
    values: &[i64],
    valid: &[u8],
    row_to_group: &[u32],
    n_groups: usize,
) -> Vec<i64> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicI64, Ordering};

    let slots: Vec<AtomicI64> = (0..n_groups).map(|_| AtomicI64::new(0)).collect();
    let n = values.len().min(row_to_group.len());
    values[..n].par_iter().enumerate().for_each(|(i, &v)| {
        let byte = valid.get(i >> 3).copied().unwrap_or(0);
        if (byte >> (i & 7)) & 1 == 0 {
            return;
        }
        let g = row_to_group[i] as usize;
        if g >= n_groups {
            return;
        }
        slots[g].fetch_add(v, Ordering::Relaxed);
    });
    slots.into_iter().map(|a| a.into_inner()).collect()
}

/// Min i64 values by group. Null rows skipped.
/// Returns `(values, valid)`: `valid[g]` is false when group `g` had no non-null rows.
pub fn aggregate_min_i64_cpu(
    values: &[i64],
    valid: &[u8],
    row_to_group: &[u32],
    n_groups: usize,
) -> (Vec<i64>, Vec<bool>) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};

    let slots: Vec<AtomicI64> = (0..n_groups).map(|_| AtomicI64::new(i64::MAX)).collect();
    let has_value: Vec<AtomicBool> = (0..n_groups).map(|_| AtomicBool::new(false)).collect();
    let n = values.len().min(row_to_group.len());
    values[..n].par_iter().enumerate().for_each(|(i, &v)| {
        let byte = valid.get(i >> 3).copied().unwrap_or(0);
        if (byte >> (i & 7)) & 1 == 0 {
            return;
        }
        let g = row_to_group[i] as usize;
        if g >= n_groups {
            return;
        }
        has_value[g].store(true, Ordering::Relaxed);
        let mut old = slots[g].load(Ordering::Relaxed);
        while v < old {
            match slots[g].compare_exchange_weak(old, v, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(cur) => old = cur,
            }
        }
    });
    let vals: Vec<i64> = slots.into_iter().map(|a| a.into_inner()).collect();
    let valid_out: Vec<bool> = has_value.into_iter().map(|a| a.into_inner()).collect();
    (vals, valid_out)
}

/// Max i64 values by group. Null rows skipped.
/// Returns `(values, valid)`: `valid[g]` is false when group `g` had no non-null rows.
pub fn aggregate_max_i64_cpu(
    values: &[i64],
    valid: &[u8],
    row_to_group: &[u32],
    n_groups: usize,
) -> (Vec<i64>, Vec<bool>) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};

    let slots: Vec<AtomicI64> = (0..n_groups).map(|_| AtomicI64::new(i64::MIN)).collect();
    let has_value: Vec<AtomicBool> = (0..n_groups).map(|_| AtomicBool::new(false)).collect();
    let n = values.len().min(row_to_group.len());
    values[..n].par_iter().enumerate().for_each(|(i, &v)| {
        let byte = valid.get(i >> 3).copied().unwrap_or(0);
        if (byte >> (i & 7)) & 1 == 0 {
            return;
        }
        let g = row_to_group[i] as usize;
        if g >= n_groups {
            return;
        }
        has_value[g].store(true, Ordering::Relaxed);
        let mut old = slots[g].load(Ordering::Relaxed);
        while v > old {
            match slots[g].compare_exchange_weak(old, v, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(cur) => old = cur,
            }
        }
    });
    let vals: Vec<i64> = slots.into_iter().map(|a| a.into_inner()).collect();
    let valid_out: Vec<bool> = has_value.into_iter().map(|a| a.into_inner()).collect();
    (vals, valid_out)
}

/// Min f64 values by group. Null rows skipped.
/// NaN poisoning: if any non-null value in the group is NaN, result is NaN
/// (Polars CPU behaviour for min on a group containing NaN).
/// Returns `(values, valid)`: `valid[g]` is false when group `g` had no non-null rows.
pub fn aggregate_min_f64_cpu(
    values: &[f64],
    valid: &[u8],
    row_to_group: &[u32],
    n_groups: usize,
) -> (Vec<f64>, Vec<bool>) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    // Seeded with +INFINITY bit pattern so any real value wins.
    let slots: Vec<AtomicU64> = (0..n_groups)
        .map(|_| AtomicU64::new(f64::INFINITY.to_bits()))
        .collect();
    let has_value: Vec<AtomicBool> = (0..n_groups).map(|_| AtomicBool::new(false)).collect();
    let n = values.len().min(row_to_group.len());
    values[..n].par_iter().enumerate().for_each(|(i, &v)| {
        let byte = valid.get(i >> 3).copied().unwrap_or(0);
        if (byte >> (i & 7)) & 1 == 0 {
            return;
        }
        let g = row_to_group[i] as usize;
        if g >= n_groups {
            return;
        }
        has_value[g].store(true, Ordering::Relaxed);
        // NaN poisons: if we see a NaN, force NaN into the slot permanently.
        if v.is_nan() {
            slots[g].store(v.to_bits(), Ordering::Relaxed);
            return;
        }
        let mut old_bits = slots[g].load(Ordering::Relaxed);
        loop {
            let cur = f64::from_bits(old_bits);
            // If the slot is already NaN (from a prior NaN row), leave it.
            if cur.is_nan() {
                break;
            }
            // v is non-NaN (checked above) and cur is non-NaN (checked here),
            // so plain >= is safe and well-defined.
            if v >= cur {
                break;
            }
            match slots[g].compare_exchange_weak(
                old_bits,
                v.to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(latest) => old_bits = latest,
            }
        }
    });
    let vals: Vec<f64> = slots
        .into_iter()
        .map(|a| f64::from_bits(a.into_inner()))
        .collect();
    let valid_out: Vec<bool> = has_value.into_iter().map(|a| a.into_inner()).collect();
    (vals, valid_out)
}

/// Max f64 values by group. Null rows skipped.
/// NaN poisoning: same as `aggregate_min_f64_cpu`.
/// Returns `(values, valid)`: `valid[g]` is false when group `g` had no non-null rows.
pub fn aggregate_max_f64_cpu(
    values: &[f64],
    valid: &[u8],
    row_to_group: &[u32],
    n_groups: usize,
) -> (Vec<f64>, Vec<bool>) {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    // Seeded with -INFINITY bit pattern so any real value wins.
    let slots: Vec<AtomicU64> = (0..n_groups)
        .map(|_| AtomicU64::new(f64::NEG_INFINITY.to_bits()))
        .collect();
    let has_value: Vec<AtomicBool> = (0..n_groups).map(|_| AtomicBool::new(false)).collect();
    let n = values.len().min(row_to_group.len());
    values[..n].par_iter().enumerate().for_each(|(i, &v)| {
        let byte = valid.get(i >> 3).copied().unwrap_or(0);
        if (byte >> (i & 7)) & 1 == 0 {
            return;
        }
        let g = row_to_group[i] as usize;
        if g >= n_groups {
            return;
        }
        has_value[g].store(true, Ordering::Relaxed);
        if v.is_nan() {
            slots[g].store(v.to_bits(), Ordering::Relaxed);
            return;
        }
        let mut old_bits = slots[g].load(Ordering::Relaxed);
        loop {
            let cur = f64::from_bits(old_bits);
            if cur.is_nan() {
                break;
            }
            // v is non-NaN (checked above) and cur is non-NaN (checked here).
            if v <= cur {
                break;
            }
            match slots[g].compare_exchange_weak(
                old_bits,
                v.to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(latest) => old_bits = latest,
            }
        }
    });
    let vals: Vec<f64> = slots
        .into_iter()
        .map(|a| f64::from_bits(a.into_inner()))
        .collect();
    let valid_out: Vec<bool> = has_value.into_iter().map(|a| a.into_inner()).collect();
    (vals, valid_out)
}

/// Count of non-null rows per group (CPU path).
pub fn aggregate_count_cpu(valid: &[u8], row_to_group: &[u32], n_groups: usize) -> Vec<u64> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    let n = row_to_group.len();
    let slots: Vec<AtomicU64> = (0..n_groups).map(|_| AtomicU64::new(0)).collect();
    (0..n).into_par_iter().for_each(|i| {
        let byte = valid.get(i >> 3).copied().unwrap_or(0);
        if (byte >> (i & 7)) & 1 == 0 {
            return;
        }
        let g = row_to_group[i] as usize;
        if g >= n_groups {
            return;
        }
        slots[g].fetch_add(1, Ordering::Relaxed);
    });
    slots.into_iter().map(|a| a.into_inner()).collect()
}

/// Total row count per group, ignoring validity (CPU path).
pub fn aggregate_len_cpu(row_to_group: &[u32], n_groups: usize) -> Vec<u64> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    let slots: Vec<AtomicU64> = (0..n_groups).map(|_| AtomicU64::new(0)).collect();
    row_to_group.par_iter().for_each(|&g| {
        let g = g as usize;
        if g < n_groups {
            slots[g].fetch_add(1, Ordering::Relaxed);
        }
    });
    slots.into_iter().map(|a| a.into_inner()).collect()
}

// -----------------------------------------------------------------------
// compute_mean — host-side post-processor
// -----------------------------------------------------------------------

/// Compute per-group mean from sum + count.
/// Returns `None` for groups where `count == 0` (empty / all-null).
/// Polars semantic: mean of empty/all-null group is null.
pub fn compute_mean_f64(sums: &[f64], counts: &[u64]) -> Vec<Option<f64>> {
    sums.iter()
        .zip(counts.iter())
        .map(|(s, c)| {
            if *c == 0 {
                None
            } else {
                Some(*s / (*c as f64))
            }
        })
        .collect()
}

/// Compute per-group mean from i64 sum + count, returning f64.
/// Returns `None` for groups where `count == 0`.
pub fn compute_mean_i64(sums: &[i64], counts: &[u64]) -> Vec<Option<f64>> {
    sums.iter()
        .zip(counts.iter())
        .map(|(s, c)| {
            if *c == 0 {
                None
            } else {
                Some(*s as f64 / *c as f64)
            }
        })
        .collect()
}

// -----------------------------------------------------------------------
// Unit tests for compute_mean (T25)
// -----------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::expect_used)]
mod mean_tests {
    use super::{compute_mean_f64, compute_mean_i64};

    #[test]
    fn compute_mean_f64_handles_empty_group() {
        let m = compute_mean_f64(&[10.0, 0.0], &[2, 0]);
        assert_eq!(m, vec![Some(5.0), None]);
    }

    #[test]
    fn compute_mean_i64_returns_f64_with_division() {
        let m = compute_mean_i64(&[10, 7], &[4, 2]);
        assert_eq!(m, vec![Some(2.5), Some(3.5)]);
    }

    #[test]
    fn compute_mean_f64_all_null_groups_are_none() {
        let m = compute_mean_f64(&[0.0, 0.0, 0.0], &[0, 0, 0]);
        assert_eq!(m, vec![None, None, None]);
    }

    #[test]
    fn compute_mean_i64_single_row_groups() {
        let m = compute_mean_i64(&[3, -6, 0], &[1, 1, 1]);
        assert_eq!(m, vec![Some(3.0), Some(-6.0), Some(0.0)]);
    }

    #[test]
    fn compute_mean_f64_single_element() {
        let m = compute_mean_f64(&[7.5], &[1]);
        assert_eq!(m, vec![Some(7.5)]);
    }

    #[test]
    fn compute_mean_i64_integer_division_produces_f64() {
        // 1 / 3 = 0.333... (not truncated to 0)
        let m = compute_mean_i64(&[1], &[3]);
        let v = m[0].expect("count=3 must yield Some");
        assert!((v - 1.0 / 3.0).abs() < 1e-15);
    }
}

// -----------------------------------------------------------------------

/// Pure-Rust reference implementation of `hash_u128` from `shaders/_groupby.metal`.
///
/// Must stay byte-for-byte in sync with the MSL `hash_u128` function. Used by
/// the proptest in `tests/test_groupby_hash.rs` to verify GPU output.
///
/// Inputs: `lo` and `hi` are the low and high 64-bit halves of a u128 key,
/// exactly as the kernel reads them from `keys[gid*2]` and `keys[gid*2+1]`.
pub fn hash_u128_reference(lo: u64, hi: u64) -> u32 {
    fn rotl_u64(x: u64, r: u32) -> u64 {
        (x << r) | (x >> (64 - r))
    }
    fn xxhash_finalize_u64(v: u64) -> u32 {
        const PRIME32_2: u32 = 2_246_822_519;
        const PRIME32_3: u32 = 3_266_489_917;
        let mut h: u32 = (v ^ (v >> 32)) as u32;
        h ^= h >> 15;
        h = h.wrapping_mul(PRIME32_2);
        h ^= h >> 13;
        h = h.wrapping_mul(PRIME32_3);
        h ^= h >> 16;
        h
    }
    let combined = lo ^ rotl_u64(hi, 27);
    xxhash_finalize_u64(combined)
}

// -----------------------------------------------------------------------

/// Decoded representation of one key column, used to reconstruct result
/// DataFrames after the kernel returns indices.
#[derive(Debug, Clone, PartialEq)]
pub enum DecodedColumn {
    I64 { values: Vec<i64>, valid: Vec<bool> },
    F64 { values: Vec<f64>, valid: Vec<bool> },
    Bool { values: Vec<bool>, valid: Vec<bool> },
}

/// Decode a u128-encoded composite-key stream back to per-column values.
pub fn decode_keys(encoded: &[u128], schema: &KeySchema) -> Vec<DecodedColumn> {
    let mut out: Vec<DecodedColumn> = schema
        .fields()
        .iter()
        .map(|f| match f.dtype {
            KeyDtype::I64 => DecodedColumn::I64 {
                values: Vec::with_capacity(encoded.len()),
                valid: Vec::with_capacity(encoded.len()),
            },
            KeyDtype::F64 => DecodedColumn::F64 {
                values: Vec::with_capacity(encoded.len()),
                valid: Vec::with_capacity(encoded.len()),
            },
            KeyDtype::Bool => DecodedColumn::Bool {
                values: Vec::with_capacity(encoded.len()),
                valid: Vec::with_capacity(encoded.len()),
            },
        })
        .collect();

    for &lane in encoded {
        for (field_idx, field) in schema.fields().iter().enumerate() {
            let null_bit = (lane >> field.null_bit_offset) & 1u128;
            let is_valid = null_bit == 0;
            match (&mut out[field_idx], field.dtype) {
                (DecodedColumn::I64 { values, valid }, KeyDtype::I64) => {
                    let raw = (lane >> field.data_bit_offset) & ((1u128 << 64) - 1);
                    let v = if is_valid { raw as u64 as i64 } else { 0 };
                    values.push(v);
                    valid.push(is_valid);
                }
                (DecodedColumn::F64 { values, valid }, KeyDtype::F64) => {
                    let raw = (lane >> field.data_bit_offset) & ((1u128 << 64) - 1);
                    let v = if is_valid {
                        f64::from_bits(raw as u64)
                    } else {
                        0.0
                    };
                    values.push(v);
                    valid.push(is_valid);
                }
                (DecodedColumn::Bool { values, valid }, KeyDtype::Bool) => {
                    let raw = (lane >> field.data_bit_offset) & 1u128;
                    let v = if is_valid { raw == 1 } else { false };
                    values.push(v);
                    valid.push(is_valid);
                }
                _ => unreachable!("decoded column dtype must match field dtype"),
            }
        }
    }

    out
}
