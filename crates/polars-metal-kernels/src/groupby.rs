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
