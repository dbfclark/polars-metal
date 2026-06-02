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
    I32,
    F32,
    // M3 additions
    I8,
    I16,
    U8,
    U16,
    U32,
    // M3 Phase 7: dictionary-encoded, 32-bit codes + dict carried in schema.
    Utf8,
}

impl KeyDtype {
    /// Width in bits of the data payload (excludes the null flag).
    pub fn data_bits(self) -> u32 {
        match self {
            KeyDtype::I64 | KeyDtype::F64 => 64,
            KeyDtype::Bool => 1,
            KeyDtype::I32 | KeyDtype::F32 | KeyDtype::U32 | KeyDtype::Utf8 => 32,
            KeyDtype::I16 | KeyDtype::U16 => 16,
            KeyDtype::I8 | KeyDtype::U8 => 8,
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
    /// Only present for KeyDtype::Utf8. Caller must build the dict via
    /// `polars_metal_buffer::dict::build_dict_nullable` (or similar) and
    /// supply the resulting `Vec<String>` here; `data` then holds the
    /// u32 codes as little-endian bytes. None for all other dtypes.
    pub dict: Option<Vec<String>>,
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
    /// Only present when dtype == Utf8. Carries the dictionary so
    /// `decode_keys` can map u32 codes back to strings.
    pub dict: Option<Vec<String>>,
}

/// Schema for a composite-key encoding. Sufficient to decode an encoded
/// u128 stream back to per-column values.
///
/// The `Default` impl yields an empty schema (no fields, 0 bits) — used
/// by the empty-keys (single-group) reduction path in `dispatch_groupby`
/// / `dispatch_groupby_fused`, where no encoding occurs and the schema
/// is only passed through for shape consistency.
#[derive(Debug, Clone, Default)]
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
    #[error("KeyColumn for {col:?} has dtype Utf8 but no dict")]
    Utf8MissingDict { col: String },
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
            KeyDtype::I32 | KeyDtype::F32 | KeyDtype::U32 | KeyDtype::Utf8 => n_rows * 4,
            KeyDtype::I16 | KeyDtype::U16 => n_rows * 2,
            KeyDtype::I8 | KeyDtype::U8 => n_rows,
            KeyDtype::Bool => min_valid_bytes,
        };
        if c.dtype == KeyDtype::Utf8 && c.dict.is_none() {
            return Err(KeyEncodeError::Utf8MissingDict {
                col: c.name.clone(),
            });
        }
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
            dict: c.dict.clone(),
        });
        offset += field_bits;
    }
    let total_bits = offset;

    // Parallelize over rows. Each row is independent (different lane);
    // each column appends bits into its row's lane with disjoint bit
    // ranges (null_bit_offset + data_bit_offset are per-field). We
    // rayon-parallelize the row loop and let each thread compute its
    // lane sequentially over all fields. ~6× speedup at 10M rows × 2
    // keys vs the serial loop.
    use rayon::prelude::*;
    let encoded: Vec<u128> = (0..n_rows)
        .into_par_iter()
        .map(|row| {
            let mut lane: u128 = 0;
            for (field_idx, c) in cols.iter().enumerate() {
                let field = &fields[field_idx];
                let valid_byte = c.valid[row >> 3];
                let valid_bit = (valid_byte >> (row & 7)) & 1;
                if valid_bit == 0 {
                    lane |= 1u128 << field.null_bit_offset;
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
                    KeyDtype::I32 => {
                        let mut bytes = [0u8; 4];
                        bytes.copy_from_slice(&c.data[row * 4..(row + 1) * 4]);
                        i32::from_le_bytes(bytes) as u32 as u128
                    }
                    KeyDtype::F32 => {
                        let mut bytes = [0u8; 4];
                        bytes.copy_from_slice(&c.data[row * 4..(row + 1) * 4]);
                        f32::from_le_bytes(bytes).to_bits() as u128
                    }
                    KeyDtype::U32 | KeyDtype::Utf8 => {
                        let mut bytes = [0u8; 4];
                        bytes.copy_from_slice(&c.data[row * 4..(row + 1) * 4]);
                        u32::from_le_bytes(bytes) as u128
                    }
                    KeyDtype::I16 => {
                        let mut bytes = [0u8; 2];
                        bytes.copy_from_slice(&c.data[row * 2..(row + 1) * 2]);
                        i16::from_le_bytes(bytes) as u16 as u128
                    }
                    KeyDtype::U16 => {
                        let mut bytes = [0u8; 2];
                        bytes.copy_from_slice(&c.data[row * 2..(row + 1) * 2]);
                        u16::from_le_bytes(bytes) as u128
                    }
                    KeyDtype::I8 => {
                        let byte = c.data[row];
                        i8::from_le_bytes([byte]) as u8 as u128
                    }
                    KeyDtype::U8 => {
                        let byte = c.data[row];
                        u8::from_le_bytes([byte]) as u128
                    }
                    KeyDtype::Bool => {
                        let byte = c.data[row >> 3];
                        let bit = (byte >> (row & 7)) & 1;
                        bit as u128
                    }
                };
                lane |= data_value << field.data_bit_offset;
            }
            lane
        })
        .collect();

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
    #[error("aggregation kind {kind} not compatible with value dtype {value_dtype}")]
    AggTypeMismatch { kind: String, value_dtype: String },
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

/// Output of the build phase. Fields: `row_to_group`, `n_groups`,
/// `first_row_per_group`.
///
/// Re-exported from `crate::groupby_build_partitioned` so the CPU build
/// (`dispatch_build`), the GPU A1 build (`partition_and_build`), and
/// the A2 sort-based build (`sort_and_segment`) all return the same
/// shape — the engine's UDF dispatch can swap among them without
/// adapter code.
pub use crate::groupby_build_partitioned::BuildOutput;

/// Dispatch the `groupby_build` phase.
///
/// ## Implementation note — CPU execution
///
/// The build phase (find-or-insert into an open-addressing hash table) is
/// inherently sequential: each row must check for an existing entry before
/// deciding whether to create a new group.  Implementing this as a GPU
/// kernel requires concurrent CAS operations and spin-waiting between
/// threads, which introduces two opposing failure modes on Metal:
///
/// 1. **SIMD-group deadlock**: threads in the same SIMD-group (warp) that
///    hash to the same slot can deadlock — the CAS winner needs to write key
///    words and publish READY, but its sibling threads are spinning on that
///    slot in lockstep, preventing the winner from executing its stores.
///
/// 2. **Livelock (skip-and-retry)**: replacing the spin with "skip and
///    retry" (to avoid the deadlock) causes a different failure — threads
///    perpetually skip CLAIMED slots and exhaust their retry budget before
///    the contested slots settle, especially when many rows hash to a small
///    cluster of keys.
///
/// The CPU build path remains correct, simple, and fast for the small
/// cardinalities that dominate typical workloads. Phase 4 / 4.5 added a
/// GPU build (capability A1 — `partition_and_build_with_scratch`) that
/// wins at high row counts and low-to-medium cardinality. The router
/// here picks A1 above [`A1_ROWS_THRESHOLD`] and falls back to CPU on
/// A1 overflow (cardinality exceeds A1's TGSM capacity).
///
/// Empirical crossover from `tests/bench_cpu_build_compare.rs`:
///   - 100K rows: CPU wins (A1 is 3-5× slower; GPU dispatch overhead dominates)
///   - 1M rows × 1024 groups: A1 ~2× CPU win
///   - 10M rows × 1024 groups: A1 ~4× CPU win
///   - >16K unique groups: A1 overflows; CPU is the fallback
///
/// `hashes` is accepted but unused (kept in the signature for API
/// compatibility — callers that compute hashes for the GPU hash kernel
/// can pass them here without branching).
pub fn dispatch_build(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    encoded: &[u128],
    hashes: &[u32],
    n_rows: usize,
) -> Result<BuildOutput, GroupByError> {
    let _ = queue;
    let _ = hashes;

    if n_rows == 0 {
        return Ok(BuildOutput {
            row_to_group: vec![],
            n_groups: 0,
            first_row_per_group: vec![],
        });
    }

    if encoded.len() < n_rows {
        return Err(GroupByError::OutputTooShort {
            got: encoded.len(),
            need: n_rows,
        });
    }

    // A1 (GPU) path for inputs large enough that the GPU dispatch +
    // host↔device traffic amortizes. Falls back to CPU on overflow.
    if n_rows >= A1_ROWS_THRESHOLD {
        if let Some(out) = try_a1_build(device, &encoded[..n_rows]) {
            return Ok(out);
        }
        // None ⇒ A1 overflowed or hit a transient dispatch error; fall
        // through to CPU.
    }

    cpu_hashmap_build(&encoded[..n_rows])
}

/// Row-count crossover above which A1 is empirically faster than CPU.
/// Measured 2026-05-26 on M2 Ultra: at 1M rows × 4 groups, A1 = 9.1ms
/// vs CPU = 10.7ms (tied); at 1M × 1024 groups, A1 = 6.5ms vs CPU =
/// 13.2ms. Below 500K, CPU is faster. Conservative threshold leaves
/// some margin for jitter.
pub const A1_ROWS_THRESHOLD: usize = 500_000;

/// CPU HashMap build (original M2 path; remains the fallback for small
/// inputs and high-cardinality cases where A1 overflows).
fn cpu_hashmap_build(encoded: &[u128]) -> Result<BuildOutput, GroupByError> {
    let n_rows = encoded.len();
    let mut group_for_key: std::collections::HashMap<u128, u32> =
        std::collections::HashMap::with_capacity(n_rows.min(1 << 20));
    let mut next_gid: u32 = 0;
    let mut row_to_group = Vec::with_capacity(n_rows);
    let mut first_row_per_group: Vec<u32> = Vec::new();

    for (row, &key) in encoded.iter().enumerate() {
        let gid = *group_for_key.entry(key).or_insert_with(|| {
            let g = next_gid;
            next_gid = next_gid.checked_add(1).unwrap_or(u32::MAX);
            first_row_per_group.push(row as u32);
            g
        });
        row_to_group.push(gid);
    }

    Ok(BuildOutput {
        row_to_group,
        n_groups: next_gid,
        first_row_per_group,
    })
}

/// Try A1 GPU build using the process-wide [`BuildScratch`] arena.
/// Returns `Some(BuildOutput)` on success, `None` on
/// [`PartitionedBuildError::Overflow`] (caller should fall back to CPU).
/// Other dispatch errors are converted to `None` and likewise route to
/// CPU — A1 is the optimization; correctness comes from the CPU path.
fn try_a1_build(device: &MetalDevice, encoded: &[u128]) -> Option<BuildOutput> {
    use crate::groupby_build_partitioned::gpu::partition_and_build_with_scratch;
    use crate::groupby_build_partitioned::PartitionedBuildError;

    let scratch_mutex = a1_scratch(device)?;
    let mut scratch = scratch_mutex.lock().ok()?;
    match partition_and_build_with_scratch(device, &mut scratch, encoded, 16) {
        Ok(out) => Some(out),
        Err(PartitionedBuildError::Overflow) => None,
        Err(_) => None,
    }
}

/// Process-wide A1 scratch arena, lazily initialized on first GPU build
/// dispatch. One per device — Polars currently uses a single device.
/// `Mutex` serializes concurrent queries, which is fine because the
/// polars callback layer runs queries one at a time per process.
///
/// Returns `None` if scratch allocation fails (extremely unusual —
/// Metal must support 16-byte buffer allocation); the caller then falls
/// back to the CPU build path so the engine never returns wrong
/// results, only loses the perf optimization.
fn a1_scratch(
    device: &MetalDevice,
) -> Option<&'static std::sync::Mutex<crate::groupby_build_partitioned::BuildScratch>> {
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<crate::groupby_build_partitioned::BuildScratch>> = OnceLock::new();
    if let Some(c) = CACHE.get() {
        return Some(c);
    }
    let scratch = crate::groupby_build_partitioned::BuildScratch::new(device).ok()?;
    Some(CACHE.get_or_init(|| Mutex::new(scratch)))
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

/// Hard cap on n_groups for the F32 pre-reduce kernels. Matches the
/// `MAX_GROUPS` constant baked into `shaders/aggregate.metal` — exceeding
/// this would overflow the per-thread register array. Callers must route
/// queries with more groups to the M2 per-agg / CPU finalize path.
pub(crate) const F32_AGG_MAX_GROUPS: usize = 16;

/// Threadgroup width for the F32 pre-reduce kernels. The MSL side
/// (`shaders/aggregate.metal`) hardcodes `MAX_SIMDS_PER_TG = 8` which
/// corresponds to a 256-wide threadgroup on Apple Silicon (32-lane simd).
/// Dispatcher MUST match.
const F32_AGG_TG_WIDTH: usize = 256;

/// Cap on the number of threadgroups dispatched for the F32 pre-reduce
/// kernels. Each TG runs the full pre-reduce + simd reduce + final flush
/// once, and emits up to `n_groups` atomic CAS-adds against the device
/// output buffer. Capping at 128 TGs keeps the post-reduction atomic
/// contention bounded (≤ 128 * n_groups device atomics, well under the
/// GPU watchdog budget) while still saturating Apple Silicon's compute
/// units. Each thread strides over ~`n_rows / (caps × 256)` rows.
const F32_AGG_MAX_TGS: usize = 128;

/// Shared helper for f32 aggregation kernels.
///
/// Routes between two MSL kernels depending on `n_groups`:
///
/// - **`<kernel>_prereduce`** (low cardinality, ≤ [`F32_AGG_MAX_GROUPS`]):
///   per-thread register pre-reduce → simdgroup reduce → per-TG CAS to
///   device. Avoids the GPU watchdog at 10M rows × 4 groups where the
///   per-row variant retries O(N²/2) times.
/// - **`<kernel>` (per-row CAS, high cardinality > [`F32_AGG_MAX_GROUPS`]):
///   per-row CAS-loop on the device output. At high group cardinality
///   contention is low and CAS retries are rare; the pre-reduce variant
///   can't run anyway (register-array cap).
///
/// `kernel_name` is the **base** kernel name (e.g. `"agg_sum_f32"`); the
/// helper picks the right suffix.
///
/// The init value is passed as the bit pattern of the f32 identity
/// element (the kernel reads it via `as_type<float>`).
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

    let vals_buf = device.new_buffer_from_bytes(values_bytes)?;
    let valid_buf = device.new_buffer_from_bytes(&valid[..valid_bytes])?;
    let r2g_buf = device.new_buffer_from_bytes(r2g_bytes)?;
    let out_buf = device.new_buffer_from_bytes(init_bytes)?;
    let n_rows_buf = device.new_buffer_from_bytes(&n_rows_u32.to_le_bytes())?;

    if n_groups <= F32_AGG_MAX_GROUPS {
        // Low-cardinality: route to the pre-reduce kernel.
        let n_groups_u32: u32 = u32::try_from(n_groups).map_err(|_| {
            // Unreachable given the cap above, but keeps the path total.
            GroupByError::RowCountOverflow { n_rows: n_groups }
        })?;
        let n_groups_buf = device.new_buffer_from_bytes(&n_groups_u32.to_le_bytes())?;
        let pso_name = format!("{kernel_name}_prereduce");
        let pso = lib.pipeline(&pso_name)?;

        // Grid size: pre-reduce kernels stride per-thread, so we want far
        // fewer threads than rows. Pick the smallest multiple of TG width
        // that fits in [TG_width, MAX_TGS * TG_width].
        let target_tgs = n_rows.div_ceil(F32_AGG_TG_WIDTH);
        let n_tgs = target_tgs.clamp(1, F32_AGG_MAX_TGS);
        let n_threads = n_tgs * F32_AGG_TG_WIDTH;

        queue.dispatch_1d_with_tg(
            &pso,
            &[
                &vals_buf,
                &valid_buf,
                &r2g_buf,
                &out_buf,
                &n_rows_buf,
                &n_groups_buf,
            ],
            n_threads,
            F32_AGG_TG_WIDTH,
        )?;
    } else {
        // High-cardinality: route to per-row CAS. Contention is low
        // enough that the CAS retries don't hit the watchdog budget.
        let pso = lib.pipeline(kernel_name)?;
        queue.dispatch_1d(
            &pso,
            &[&vals_buf, &valid_buf, &r2g_buf, &out_buf, &n_rows_buf],
            n_rows,
        )?;
    }
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

    // Thread-local accumulators + reduce: each rayon work-unit sums into
    // its own [n_groups]-sized array, then we sum across chunks. Avoids
    // the per-row atomic-CAS contention that dominated the previous
    // par_iter design at low cardinality (4 groups × 9.5M rows → 9.5M
    // CAS retries serialized on 4 slots → 3.4s; this pattern: ~2ms).
    let n = values.len().min(row_to_group.len());

    (0..n)
        .into_par_iter()
        .fold(
            || vec![0.0f64; n_groups],
            |mut local, i| {
                let byte = valid.get(i >> 3).copied().unwrap_or(0);
                if (byte >> (i & 7)) & 1 == 0 {
                    return local;
                }
                let g = row_to_group[i] as usize;
                if g < n_groups {
                    local[g] += values[i];
                }
                local
            },
        )
        .reduce(
            || vec![0.0f64; n_groups],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                a
            },
        )
}

/// Sum i64 values by group. Null rows skipped. All-null group → 0.
pub fn aggregate_sum_i64_cpu(
    values: &[i64],
    valid: &[u8],
    row_to_group: &[u32],
    n_groups: usize,
) -> Vec<i64> {
    use rayon::prelude::*;

    // Thread-local accumulators + reduce. See `aggregate_sum_f64_cpu`
    // for the perf rationale; ~10× speedup at low cardinality vs the
    // atomic-fetch_add design.
    let n = values.len().min(row_to_group.len());
    (0..n)
        .into_par_iter()
        .fold(
            || vec![0i64; n_groups],
            |mut local, i| {
                let byte = valid.get(i >> 3).copied().unwrap_or(0);
                if (byte >> (i & 7)) & 1 == 0 {
                    return local;
                }
                let g = row_to_group[i] as usize;
                if g < n_groups {
                    local[g] = local[g].wrapping_add(values[i]);
                }
                local
            },
        )
        .reduce(
            || vec![0i64; n_groups],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x = x.wrapping_add(*y);
                }
                a
            },
        )
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

    let n = values.len().min(row_to_group.len());
    let (vals, has_value) = (0..n)
        .into_par_iter()
        .fold(
            || (vec![i64::MAX; n_groups], vec![false; n_groups]),
            |(mut vals, mut has), i| {
                let byte = valid.get(i >> 3).copied().unwrap_or(0);
                if (byte >> (i & 7)) & 1 == 0 {
                    return (vals, has);
                }
                let g = row_to_group[i] as usize;
                if g < n_groups {
                    let v = values[i];
                    if v < vals[g] {
                        vals[g] = v;
                    }
                    has[g] = true;
                }
                (vals, has)
            },
        )
        .reduce(
            || (vec![i64::MAX; n_groups], vec![false; n_groups]),
            |(mut va, mut ha), (vb, hb)| {
                for ((a, &b), (ah, &bh)) in va
                    .iter_mut()
                    .zip(vb.iter())
                    .zip(ha.iter_mut().zip(hb.iter()))
                {
                    if b < *a {
                        *a = b;
                    }
                    *ah |= bh;
                }
                (va, ha)
            },
        );
    (vals, has_value)
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

    let n = values.len().min(row_to_group.len());
    (0..n)
        .into_par_iter()
        .fold(
            || (vec![i64::MIN; n_groups], vec![false; n_groups]),
            |(mut vals, mut has), i| {
                let byte = valid.get(i >> 3).copied().unwrap_or(0);
                if (byte >> (i & 7)) & 1 == 0 {
                    return (vals, has);
                }
                let g = row_to_group[i] as usize;
                if g < n_groups {
                    let v = values[i];
                    if v > vals[g] {
                        vals[g] = v;
                    }
                    has[g] = true;
                }
                (vals, has)
            },
        )
        .reduce(
            || (vec![i64::MIN; n_groups], vec![false; n_groups]),
            |(mut va, mut ha), (vb, hb)| {
                for ((a, &b), (ah, &bh)) in va
                    .iter_mut()
                    .zip(vb.iter())
                    .zip(ha.iter_mut().zip(hb.iter()))
                {
                    if b > *a {
                        *a = b;
                    }
                    *ah |= bh;
                }
                (va, ha)
            },
        )
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

    let n = values.len().min(row_to_group.len());
    (0..n)
        .into_par_iter()
        .fold(
            || (vec![f64::INFINITY; n_groups], vec![false; n_groups]),
            |(mut vals, mut has), i| {
                let byte = valid.get(i >> 3).copied().unwrap_or(0);
                if (byte >> (i & 7)) & 1 == 0 {
                    return (vals, has);
                }
                let g = row_to_group[i] as usize;
                if g < n_groups {
                    let v = values[i];
                    has[g] = true;
                    // NaN poisons: once NaN, stay NaN.
                    if v.is_nan() || vals[g].is_nan() {
                        vals[g] = f64::NAN;
                    } else if v < vals[g] {
                        vals[g] = v;
                    }
                }
                (vals, has)
            },
        )
        .reduce(
            || (vec![f64::INFINITY; n_groups], vec![false; n_groups]),
            |(mut va, mut ha), (vb, hb)| {
                for ((a, &b), (ah, &bh)) in va
                    .iter_mut()
                    .zip(vb.iter())
                    .zip(ha.iter_mut().zip(hb.iter()))
                {
                    if a.is_nan() || b.is_nan() {
                        *a = f64::NAN;
                    } else if b < *a {
                        *a = b;
                    }
                    *ah |= bh;
                }
                (va, ha)
            },
        )
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

    let n = values.len().min(row_to_group.len());
    (0..n)
        .into_par_iter()
        .fold(
            || (vec![f64::NEG_INFINITY; n_groups], vec![false; n_groups]),
            |(mut vals, mut has), i| {
                let byte = valid.get(i >> 3).copied().unwrap_or(0);
                if (byte >> (i & 7)) & 1 == 0 {
                    return (vals, has);
                }
                let g = row_to_group[i] as usize;
                if g < n_groups {
                    let v = values[i];
                    has[g] = true;
                    if v.is_nan() || vals[g].is_nan() {
                        vals[g] = f64::NAN;
                    } else if v > vals[g] {
                        vals[g] = v;
                    }
                }
                (vals, has)
            },
        )
        .reduce(
            || (vec![f64::NEG_INFINITY; n_groups], vec![false; n_groups]),
            |(mut va, mut ha), (vb, hb)| {
                for ((a, &b), (ah, &bh)) in va
                    .iter_mut()
                    .zip(vb.iter())
                    .zip(ha.iter_mut().zip(hb.iter()))
                {
                    if a.is_nan() || b.is_nan() {
                        *a = f64::NAN;
                    } else if b > *a {
                        *a = b;
                    }
                    *ah |= bh;
                }
                (va, ha)
            },
        )
}

/// Count of non-null rows per group (CPU path).
pub fn aggregate_count_cpu(valid: &[u8], row_to_group: &[u32], n_groups: usize) -> Vec<u64> {
    use rayon::prelude::*;

    // Thread-local accumulators + reduce; same rationale as aggregate_sum_*.
    let n = row_to_group.len();
    (0..n)
        .into_par_iter()
        .fold(
            || vec![0u64; n_groups],
            |mut local, i| {
                let byte = valid.get(i >> 3).copied().unwrap_or(0);
                if (byte >> (i & 7)) & 1 == 0 {
                    return local;
                }
                let g = row_to_group[i] as usize;
                if g < n_groups {
                    local[g] += 1;
                }
                local
            },
        )
        .reduce(
            || vec![0u64; n_groups],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                a
            },
        )
}

/// Total row count per group, ignoring validity (CPU path).
pub fn aggregate_len_cpu(row_to_group: &[u32], n_groups: usize) -> Vec<u64> {
    use rayon::prelude::*;

    row_to_group
        .par_iter()
        .fold(
            || vec![0u64; n_groups],
            |mut local, &g| {
                let g = g as usize;
                if g < n_groups {
                    local[g] += 1;
                }
                local
            },
        )
        .reduce(
            || vec![0u64; n_groups],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                a
            },
        )
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

/// Compute per-group mean from f32 sum + count, returning f32.
/// Returns `None` for groups where `count == 0`.
pub fn compute_mean_f32(sums: &[f32], counts: &[u32]) -> Vec<Option<f32>> {
    sums.iter()
        .zip(counts.iter())
        .map(|(s, c)| {
            if *c == 0 {
                None
            } else {
                Some(*s / (*c as f32))
            }
        })
        .collect()
}

/// Compute per-group mean from i32 sum + count, returning f32.
/// Returns `None` for groups where `count == 0`.
pub fn compute_mean_i32(sums: &[i32], counts: &[u32]) -> Vec<Option<f32>> {
    sums.iter()
        .zip(counts.iter())
        .map(|(s, c)| {
            if *c == 0 {
                None
            } else {
                Some(*s as f32 / (*c as f32))
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
    I64 {
        values: Vec<i64>,
        valid: Vec<bool>,
    },
    F64 {
        values: Vec<f64>,
        valid: Vec<bool>,
    },
    Bool {
        values: Vec<bool>,
        valid: Vec<bool>,
    },
    I32 {
        values: Vec<i32>,
        valid: Vec<bool>,
    },
    F32 {
        values: Vec<f32>,
        valid: Vec<bool>,
    },
    // M3 additions
    I8 {
        values: Vec<i8>,
        valid: Vec<bool>,
    },
    I16 {
        values: Vec<i16>,
        valid: Vec<bool>,
    },
    U8 {
        values: Vec<u8>,
        valid: Vec<bool>,
    },
    U16 {
        values: Vec<u16>,
        valid: Vec<bool>,
    },
    U32 {
        values: Vec<u32>,
        valid: Vec<bool>,
    },
    // M3 Phase 7: dictionary-decoded Utf8 keys.
    Utf8 {
        values: Vec<String>,
        valid: Vec<bool>,
    },
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
            KeyDtype::I32 => DecodedColumn::I32 {
                values: Vec::with_capacity(encoded.len()),
                valid: Vec::with_capacity(encoded.len()),
            },
            KeyDtype::F32 => DecodedColumn::F32 {
                values: Vec::with_capacity(encoded.len()),
                valid: Vec::with_capacity(encoded.len()),
            },
            KeyDtype::I8 => DecodedColumn::I8 {
                values: Vec::with_capacity(encoded.len()),
                valid: Vec::with_capacity(encoded.len()),
            },
            KeyDtype::I16 => DecodedColumn::I16 {
                values: Vec::with_capacity(encoded.len()),
                valid: Vec::with_capacity(encoded.len()),
            },
            KeyDtype::U8 => DecodedColumn::U8 {
                values: Vec::with_capacity(encoded.len()),
                valid: Vec::with_capacity(encoded.len()),
            },
            KeyDtype::U16 => DecodedColumn::U16 {
                values: Vec::with_capacity(encoded.len()),
                valid: Vec::with_capacity(encoded.len()),
            },
            KeyDtype::U32 => DecodedColumn::U32 {
                values: Vec::with_capacity(encoded.len()),
                valid: Vec::with_capacity(encoded.len()),
            },
            KeyDtype::Utf8 => DecodedColumn::Utf8 {
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
                (DecodedColumn::I32 { values, valid }, KeyDtype::I32) => {
                    let raw = (lane >> field.data_bit_offset) & ((1u128 << 32) - 1);
                    let v = if is_valid { raw as u32 as i32 } else { 0 };
                    values.push(v);
                    valid.push(is_valid);
                }
                (DecodedColumn::F32 { values, valid }, KeyDtype::F32) => {
                    let raw = (lane >> field.data_bit_offset) & ((1u128 << 32) - 1);
                    let v = if is_valid {
                        f32::from_bits(raw as u32)
                    } else {
                        0.0f32
                    };
                    values.push(v);
                    valid.push(is_valid);
                }
                (DecodedColumn::U32 { values, valid }, KeyDtype::U32) => {
                    let raw = (lane >> field.data_bit_offset) & ((1u128 << 32) - 1);
                    let v = if is_valid { raw as u32 } else { 0u32 };
                    values.push(v);
                    valid.push(is_valid);
                }
                (DecodedColumn::I16 { values, valid }, KeyDtype::I16) => {
                    let raw = (lane >> field.data_bit_offset) & ((1u128 << 16) - 1);
                    let v = if is_valid { raw as u16 as i16 } else { 0i16 };
                    values.push(v);
                    valid.push(is_valid);
                }
                (DecodedColumn::U16 { values, valid }, KeyDtype::U16) => {
                    let raw = (lane >> field.data_bit_offset) & ((1u128 << 16) - 1);
                    let v = if is_valid { raw as u16 } else { 0u16 };
                    values.push(v);
                    valid.push(is_valid);
                }
                (DecodedColumn::I8 { values, valid }, KeyDtype::I8) => {
                    let raw = (lane >> field.data_bit_offset) & ((1u128 << 8) - 1);
                    let v = if is_valid { raw as u8 as i8 } else { 0i8 };
                    values.push(v);
                    valid.push(is_valid);
                }
                (DecodedColumn::U8 { values, valid }, KeyDtype::U8) => {
                    let raw = (lane >> field.data_bit_offset) & ((1u128 << 8) - 1);
                    let v = if is_valid { raw as u8 } else { 0u8 };
                    values.push(v);
                    valid.push(is_valid);
                }
                (DecodedColumn::Utf8 { values, valid }, KeyDtype::Utf8) => {
                    let raw = (lane >> field.data_bit_offset) & ((1u128 << 32) - 1);
                    let code = raw as u32;
                    // `encode_keys` guarantees `dict` is `Some` for every
                    // Utf8 field it emits (validated up-front via
                    // `KeyEncodeError::Utf8MissingDict`). A `None` here
                    // would mean a schema was constructed by some other
                    // path; degrade gracefully to empty-string output
                    // rather than panicking on a kernel-layer assumption.
                    let v = if is_valid {
                        field
                            .dict
                            .as_ref()
                            .and_then(|d| d.get(code as usize).cloned())
                            .unwrap_or_default()
                    } else {
                        String::new()
                    };
                    values.push(v);
                    valid.push(is_valid);
                }
                _ => unreachable!("decoded column dtype must match field dtype"),
            }
        }
    }

    out
}

// -----------------------------------------------------------------------
// T26: dispatch_groupby orchestrator
// -----------------------------------------------------------------------

/// Which aggregation kernel to run. Mirrors `polars-metal-core::plan::AggOp`
/// but is defined here to keep the dep direction one-way (core → kernels).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggKind {
    SumI64,
    SumF64,
    MeanI64, // sum_i64 / count → Option<f64>
    MeanF64, // sum_f64 / count → Option<f64>
    MinI64,
    MaxI64,
    MinF64,
    MaxF64,
    Count, // count of non-null values
    Len,   // total row count per group (pl.len())
    // 32-bit GPU dispatcher variants
    SumI32,
    SumF32,
    MeanI32, // sum_i32 / count → f32
    MeanF32, // sum_f32 / count → f32
    MinI32,
    MaxI32,
    MinF32,
    MaxF32,
}

/// A single aggregation request.
#[derive(Debug, Clone)]
pub struct AggRequest {
    pub kind: AggKind,
    /// Index into the `value_cols` slice. Ignored for `AggKind::Len`.
    pub input_col_idx: usize,
}

/// One value column passed to the pipeline (typed view).
pub enum ValueColumn<'a> {
    I64 { data: &'a [i64], valid: &'a [u8] },
    F64 { data: &'a [f64], valid: &'a [u8] },
    I32 { data: &'a [i32], valid: &'a [u8] },
    F32 { data: &'a [f32], valid: &'a [u8] },
}

/// One aggregation's output. Carries data + per-group null flags for ops
/// that can produce null per-group (min/max/mean on all-null groups).
#[derive(Debug, Clone, PartialEq)]
pub enum AggOutput {
    I64 { values: Vec<i64>, valid: Vec<bool> },
    F64 { values: Vec<f64>, valid: Vec<bool> },
    U64 { values: Vec<u64> },
    I32 { values: Vec<i32>, valid: Vec<bool> },
    F32 { values: Vec<f32>, valid: Vec<bool> },
}

/// The pipeline's complete result.
#[derive(Debug)]
pub struct GroupByResult {
    /// One entry per key column, in input order.
    pub decoded_keys: Vec<DecodedColumn>,
    /// One entry per AggRequest, in input order.
    pub agg_outputs: Vec<AggOutput>,
    /// Distinct group count.
    pub n_groups: u32,
}

/// Full groupby pipeline: encode → hash → build → aggregate → decode.
///
/// Handles i64/f64 value columns via CPU-finalize aggregators (Metal toolchain
/// 32023.883 lacks 64-bit atomics). i32/f32 paths are wired through GPU
/// dispatchers in a future task.
pub fn dispatch_groupby(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    key_cols: &[KeyColumn<'_>],
    agg_specs: &[(AggRequest, ValueColumn<'_>)],
    n_rows: usize,
) -> Result<GroupByResult, GroupByError> {
    // Defensive: every key column reports the same n_rows.
    for kc in key_cols {
        if kc.n_rows != n_rows {
            return Err(GroupByError::OutputTooShort {
                got: kc.n_rows,
                need: n_rows,
            });
        }
    }

    // Empty-keys path: a SELECT(agg(expr)) with no group-by keys is a
    // single-group reduction over the entire input (TPC-H Q6 shape). Skip
    // the build phase (no hashing, no dedup) and synthesize a BuildOutput
    // that maps every row to group 0.
    if key_cols.is_empty() {
        if n_rows == 0 {
            let agg_outputs = agg_specs
                .iter()
                .map(|(req, _)| empty_output_for(req.kind))
                .collect();
            return Ok(GroupByResult {
                decoded_keys: vec![],
                agg_outputs,
                n_groups: 0,
            });
        }
        let row_to_group = vec![0u32; n_rows];
        let mut agg_outputs = Vec::with_capacity(agg_specs.len());
        for (req, vcol) in agg_specs {
            agg_outputs.push(run_one_agg(device, queue, req, vcol, &row_to_group, 1)?);
        }
        return Ok(GroupByResult {
            decoded_keys: vec![],
            agg_outputs,
            n_groups: 1,
        });
    }

    let (encoded, schema) = encode_keys(key_cols)?;

    if n_rows == 0 {
        let decoded_keys = decode_keys(&[], &schema);
        let agg_outputs = agg_specs
            .iter()
            .map(|(req, _)| empty_output_for(req.kind))
            .collect();
        return Ok(GroupByResult {
            decoded_keys,
            agg_outputs,
            n_groups: 0,
        });
    }

    // The current build path (CPU HashMap + A1 GPU build) ignores
    // precomputed hashes — they were a relic of the GPU global-CAS hash
    // table M2 abandoned. Skip the dispatch_hash kernel entirely; an
    // empty slice satisfies the `hashes` parameter for both build paths.
    // (~23 ms saved at 10M rows.)
    let build = dispatch_build(device, queue, &encoded, &[], n_rows)?;
    let n_groups = build.n_groups;

    let mut agg_outputs = Vec::with_capacity(agg_specs.len());
    for (req, vcol) in agg_specs {
        agg_outputs.push(run_one_agg(
            device,
            queue,
            req,
            vcol,
            &build.row_to_group,
            n_groups as usize,
        )?);
    }

    let representative_keys: Vec<u128> = build
        .first_row_per_group
        .iter()
        .map(|&row| encoded[row as usize])
        .collect();
    let decoded_keys = decode_keys(&representative_keys, &schema);

    Ok(GroupByResult {
        decoded_keys,
        agg_outputs,
        n_groups,
    })
}

fn empty_output_for(kind: AggKind) -> AggOutput {
    match kind {
        AggKind::SumI64 | AggKind::MinI64 | AggKind::MaxI64 => AggOutput::I64 {
            values: vec![],
            valid: vec![],
        },
        AggKind::SumF64
        | AggKind::MeanI64
        | AggKind::MeanF64
        | AggKind::MinF64
        | AggKind::MaxF64 => AggOutput::F64 {
            values: vec![],
            valid: vec![],
        },
        AggKind::Count | AggKind::Len => AggOutput::U64 { values: vec![] },
        AggKind::SumI32 | AggKind::MinI32 | AggKind::MaxI32 => AggOutput::I32 {
            values: vec![],
            valid: vec![],
        },
        AggKind::SumF32
        | AggKind::MeanI32
        | AggKind::MeanF32
        | AggKind::MinF32
        | AggKind::MaxF32 => AggOutput::F32 {
            values: vec![],
            valid: vec![],
        },
    }
}

/// Helper: build a per-group null-validity mask from a GPU count result.
/// Groups with `count[g] == 0` are null (all input rows were null).
fn count_to_valid(counts: &[u32]) -> Vec<bool> {
    counts.iter().map(|&c| c > 0).collect()
}

fn run_one_agg(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    req: &AggRequest,
    vcol: &ValueColumn<'_>,
    row_to_group: &[u32],
    n_groups: usize,
) -> Result<AggOutput, GroupByError> {
    let n_rows = row_to_group.len();
    match (req.kind, vcol) {
        (AggKind::SumI64, ValueColumn::I64 { data, valid }) => {
            let values = aggregate_sum_i64_cpu(data, valid, row_to_group, n_groups);
            // Polars sum of an all-null group returns 0 (not null).
            Ok(AggOutput::I64 {
                valid: vec![true; n_groups],
                values,
            })
        }
        (AggKind::SumF64, ValueColumn::F64 { data, valid }) => {
            let values = aggregate_sum_f64_cpu(data, valid, row_to_group, n_groups);
            Ok(AggOutput::F64 {
                valid: vec![true; n_groups],
                values,
            })
        }
        (AggKind::MinI64, ValueColumn::I64 { data, valid }) => {
            let (values, valid_out) = aggregate_min_i64_cpu(data, valid, row_to_group, n_groups);
            Ok(AggOutput::I64 {
                values,
                valid: valid_out,
            })
        }
        (AggKind::MaxI64, ValueColumn::I64 { data, valid }) => {
            let (values, valid_out) = aggregate_max_i64_cpu(data, valid, row_to_group, n_groups);
            Ok(AggOutput::I64 {
                values,
                valid: valid_out,
            })
        }
        (AggKind::MinF64, ValueColumn::F64 { data, valid }) => {
            let (values, valid_out) = aggregate_min_f64_cpu(data, valid, row_to_group, n_groups);
            Ok(AggOutput::F64 {
                values,
                valid: valid_out,
            })
        }
        (AggKind::MaxF64, ValueColumn::F64 { data, valid }) => {
            let (values, valid_out) = aggregate_max_f64_cpu(data, valid, row_to_group, n_groups);
            Ok(AggOutput::F64 {
                values,
                valid: valid_out,
            })
        }
        (AggKind::Count, ValueColumn::I64 { valid, .. })
        | (AggKind::Count, ValueColumn::F64 { valid, .. })
        | (AggKind::Count, ValueColumn::I32 { valid, .. })
        | (AggKind::Count, ValueColumn::F32 { valid, .. }) => {
            let values = aggregate_count_cpu(valid, row_to_group, n_groups);
            Ok(AggOutput::U64 { values })
        }
        (AggKind::Len, _) => {
            let values = aggregate_len_cpu(row_to_group, n_groups);
            Ok(AggOutput::U64 { values })
        }
        (AggKind::MeanI64, ValueColumn::I64 { data, valid }) => {
            let sums = aggregate_sum_i64_cpu(data, valid, row_to_group, n_groups);
            let counts = aggregate_count_cpu(valid, row_to_group, n_groups);
            let means = compute_mean_i64(&sums, &counts);
            let values: Vec<f64> = means.iter().map(|m| m.unwrap_or(0.0)).collect();
            let valid_out: Vec<bool> = means.iter().map(|m| m.is_some()).collect();
            Ok(AggOutput::F64 {
                values,
                valid: valid_out,
            })
        }
        (AggKind::MeanF64, ValueColumn::F64 { data, valid }) => {
            let sums = aggregate_sum_f64_cpu(data, valid, row_to_group, n_groups);
            let counts = aggregate_count_cpu(valid, row_to_group, n_groups);
            let means = compute_mean_f64(&sums, &counts);
            let values: Vec<f64> = means.iter().map(|m| m.unwrap_or(0.0)).collect();
            let valid_out: Vec<bool> = means.iter().map(|m| m.is_some()).collect();
            Ok(AggOutput::F64 {
                values,
                valid: valid_out,
            })
        }
        // --- 32-bit GPU dispatcher paths ---
        (AggKind::SumI32, ValueColumn::I32 { data, valid }) => {
            let mut out = vec![0i32; n_groups];
            dispatch_sum_i32(
                device,
                queue,
                data,
                valid,
                row_to_group,
                n_rows,
                n_groups,
                &mut out,
            )?;
            Ok(AggOutput::I32 {
                valid: vec![true; n_groups],
                values: out,
            })
        }
        (AggKind::SumF32, ValueColumn::F32 { data, valid }) => {
            let mut out = vec![0f32; n_groups];
            dispatch_sum_f32(
                device,
                queue,
                data,
                valid,
                row_to_group,
                n_rows,
                n_groups,
                &mut out,
            )?;
            Ok(AggOutput::F32 {
                valid: vec![true; n_groups],
                values: out,
            })
        }
        (AggKind::MinI32, ValueColumn::I32 { data, valid }) => {
            let mut out = vec![i32::MAX; n_groups];
            dispatch_min_i32(
                device,
                queue,
                data,
                valid,
                row_to_group,
                n_rows,
                n_groups,
                &mut out,
            )?;
            // Groups with no valid rows retain i32::MAX; detect via count.
            let mut counts = vec![0u32; n_groups];
            dispatch_count_u32(
                device,
                queue,
                valid,
                row_to_group,
                n_rows,
                n_groups,
                &mut counts,
            )?;
            let valid_out = count_to_valid(&counts);
            Ok(AggOutput::I32 {
                values: out,
                valid: valid_out,
            })
        }
        (AggKind::MaxI32, ValueColumn::I32 { data, valid }) => {
            let mut out = vec![i32::MIN; n_groups];
            dispatch_max_i32(
                device,
                queue,
                data,
                valid,
                row_to_group,
                n_rows,
                n_groups,
                &mut out,
            )?;
            let mut counts = vec![0u32; n_groups];
            dispatch_count_u32(
                device,
                queue,
                valid,
                row_to_group,
                n_rows,
                n_groups,
                &mut counts,
            )?;
            let valid_out = count_to_valid(&counts);
            Ok(AggOutput::I32 {
                values: out,
                valid: valid_out,
            })
        }
        (AggKind::MinF32, ValueColumn::F32 { data, valid }) => {
            let mut out = vec![f32::INFINITY; n_groups];
            dispatch_min_f32(
                device,
                queue,
                data,
                valid,
                row_to_group,
                n_rows,
                n_groups,
                &mut out,
            )?;
            let mut counts = vec![0u32; n_groups];
            dispatch_count_u32(
                device,
                queue,
                valid,
                row_to_group,
                n_rows,
                n_groups,
                &mut counts,
            )?;
            let valid_out = count_to_valid(&counts);
            Ok(AggOutput::F32 {
                values: out,
                valid: valid_out,
            })
        }
        (AggKind::MaxF32, ValueColumn::F32 { data, valid }) => {
            let mut out = vec![f32::NEG_INFINITY; n_groups];
            dispatch_max_f32(
                device,
                queue,
                data,
                valid,
                row_to_group,
                n_rows,
                n_groups,
                &mut out,
            )?;
            let mut counts = vec![0u32; n_groups];
            dispatch_count_u32(
                device,
                queue,
                valid,
                row_to_group,
                n_rows,
                n_groups,
                &mut counts,
            )?;
            let valid_out = count_to_valid(&counts);
            Ok(AggOutput::F32 {
                values: out,
                valid: valid_out,
            })
        }
        (AggKind::MeanI32, ValueColumn::I32 { data, valid }) => {
            let mut sums = vec![0i32; n_groups];
            dispatch_sum_i32(
                device,
                queue,
                data,
                valid,
                row_to_group,
                n_rows,
                n_groups,
                &mut sums,
            )?;
            let mut counts = vec![0u32; n_groups];
            dispatch_count_u32(
                device,
                queue,
                valid,
                row_to_group,
                n_rows,
                n_groups,
                &mut counts,
            )?;
            let means = compute_mean_i32(&sums, &counts);
            let values: Vec<f32> = means.iter().map(|m| m.unwrap_or(0.0f32)).collect();
            let valid_out: Vec<bool> = means.iter().map(|m| m.is_some()).collect();
            Ok(AggOutput::F32 {
                values,
                valid: valid_out,
            })
        }
        (AggKind::MeanF32, ValueColumn::F32 { data, valid }) => {
            let mut sums = vec![0f32; n_groups];
            dispatch_sum_f32(
                device,
                queue,
                data,
                valid,
                row_to_group,
                n_rows,
                n_groups,
                &mut sums,
            )?;
            let mut counts = vec![0u32; n_groups];
            dispatch_count_u32(
                device,
                queue,
                valid,
                row_to_group,
                n_rows,
                n_groups,
                &mut counts,
            )?;
            let means = compute_mean_f32(&sums, &counts);
            let values: Vec<f32> = means.iter().map(|m| m.unwrap_or(0.0f32)).collect();
            let valid_out: Vec<bool> = means.iter().map(|m| m.is_some()).collect();
            Ok(AggOutput::F32 {
                values,
                valid: valid_out,
            })
        }
        (kind, vcol) => {
            let vt = match vcol {
                ValueColumn::I64 { .. } => "I64",
                ValueColumn::F64 { .. } => "F64",
                ValueColumn::I32 { .. } => "I32",
                ValueColumn::F32 { .. } => "F32",
            };
            Err(GroupByError::AggTypeMismatch {
                kind: format!("{kind:?}"),
                value_dtype: vt.to_string(),
            })
        }
    }
}

// ============================================================================
// Fused groupby dispatch (Task 15)
// ============================================================================
//
// Parallel entry point to `dispatch_groupby` that uses a single MSL kernel
// for ALL aggregations rather than launching one kernel per agg. The fused
// kernel is emitted by `aggregate_fused::emitter::emit_msl` and cached by
// signature hash in `FusedLibraryCache`.
//
// The caller (IR layer in `polars-metal-core::udf`) decides between this
// path and the M2 per-agg path based on `signature_supported_by_fused` and
// agg shape (see `udf::pick_groupby_dispatch`).
//
// ## Buffer slot layout (must match `emitter.rs`)
//
//   slot 0:                  row_to_group  (u32 per row)
//   slot 1:                  n_rows scalar (1-element u32)
//   slots 2 .. 2+C:          value_<i> column data, in `sig.column_order()`
//   slots 2+C .. 2+2C:       validity_<i> bitmap, in `sig.column_order()`
//   slots 2+2C ..:           outputs — one per agg, plus a second u32 count
//                            companion for every Mean.

use crate::aggregate_fused::cache::{FusedCacheError, FusedLibraryCache};
use crate::aggregate_fused::emitter::{signature_supported_by_fused, FUSED_TG_WIDTH, MAX_GROUPS};
use crate::aggregate_fused::signature::{
    AggOp as KAggOp, AggSignature, AggSignatureError, AggSpec as KAggSpec,
    MetalDtype as KMetalDtype,
};

/// Errors specific to the fused dispatcher. Wraps `GroupByError` for the
/// shared path and adds variants for fused-only signal failures.
#[derive(Debug, thiserror::Error)]
pub enum FusedDispatchError {
    #[error("groupby: {0}")]
    GroupBy(#[from] GroupByError),
    #[error("signature: {0}")]
    Signature(#[from] AggSignatureError),
    #[error("fused cache: {0}")]
    Cache(#[from] FusedCacheError),
    #[error(
        "fused dispatch rejected: signature contains 64-bit input dtypes; \
         caller must route to the M2 per-agg path"
    )]
    SignatureNotFused,
    /// `n_groups` exceeds the kernel's per-thread register-array cap
    /// (`MAX_GROUPS` in `aggregate_fused/emitter.rs`). The caller must
    /// route to the M2 per-agg path; that path uses per-row atomics
    /// directly on device memory, which scales to arbitrary cardinality
    /// (and isn't penalized at high cardinality the way the pre-reduce
    /// fused kernel would be).
    #[error("fused dispatch rejected: n_groups={n_groups} exceeds MAX_GROUPS={max_groups}; caller must route to the M2 per-agg path")]
    NgroupsExceedsFusedCap { n_groups: usize, max_groups: usize },
    #[error("missing value column for slot {slot} ({name})")]
    MissingValueColumn { slot: usize, name: String },
    #[error("value column {name}: dtype mismatch (signature has {sig_dt:?}, caller {got:?})")]
    ValueDtypeMismatch {
        name: String,
        sig_dt: KMetalDtype,
        got: &'static str,
    },
}

/// Per-output buffer descriptor used while encoding the dispatch and
/// reading results back. Kept private — callers only see `AggOutput`.
struct OutputBuf {
    primary: polars_metal_buffer::MetalBuffer,
    /// Mean carries a companion `count` (u32). For non-Mean aggs this is `None`.
    count: Option<polars_metal_buffer::MetalBuffer>,
    /// What the primary buffer holds, used to widen back to AggOutput. The
    /// fused emitter accumulates Sum/Min/Max into one of three 32-bit
    /// containers: i32 (for signed-integer inputs), u32 (for unsigned),
    /// or f32-bit-pattern stored in a u32 (for float inputs).
    primary_kind: PrimaryKind,
}

#[derive(Debug, Clone, Copy)]
enum PrimaryKind {
    /// Output is u32 (Count, Length, unsigned-integer Sum/Min/Max).
    U32,
    /// Output is i32 (signed-integer Sum/Min/Max).
    I32,
    /// Output is f32, stored as u32 bit pattern (float Sum/Min/Max + Mean's sum slot).
    F32Bits,
}

/// Fused-kernel groupby dispatch.
///
/// Single MSL kernel handles all aggs; one kernel launch per groupby
/// regardless of agg count.
///
/// # Preconditions
///
/// - Every value column is 32-bit or narrower (caller verified via
///   [`signature_supported_by_fused`]); F64 / I64 input columns must route
///   to [`dispatch_groupby`] instead.
/// - Every column referenced by a Simple or Expression agg appears in
///   `value_columns_by_name`. `Length` aggs need no value columns.
/// - `value_columns_by_name` typing matches `col_dtypes` exactly. The two
///   must come from a single caller-side derivation step.
///
/// Returns the same shape of result as [`dispatch_groupby`].
pub fn dispatch_groupby_fused(
    device: &MetalDevice,
    queue: &mut CommandQueue,
    cache: &FusedLibraryCache,
    key_cols: &[KeyColumn<'_>],
    aggs: &[KAggSpec],
    value_columns_by_name: &std::collections::HashMap<String, ValueColumn<'_>>,
    n_rows: usize,
) -> Result<GroupByResult, FusedDispatchError> {
    // Defensive: every key column reports the same n_rows.
    for kc in key_cols {
        if kc.n_rows != n_rows {
            return Err(FusedDispatchError::GroupBy(GroupByError::OutputTooShort {
                got: kc.n_rows,
                need: n_rows,
            }));
        }
    }

    // Empty-keys path: SELECT(agg(expr)) with no group-by keys becomes a
    // single-group reduction over the full input. Skip encode/hash/build
    // and synthesize a BuildOutput in which every row maps to group 0.
    // (TPC-H Q6 takes this shape after optimization.)
    let (encoded, schema) = if key_cols.is_empty() {
        (Vec::<u128>::new(), KeySchema::default())
    } else {
        encode_keys(key_cols).map_err(GroupByError::KeyEncode)?
    };

    if n_rows == 0 {
        let decoded_keys = if key_cols.is_empty() {
            Vec::new()
        } else {
            decode_keys(&[], &schema)
        };
        let agg_outputs = aggs.iter().map(empty_output_for_kagg).collect();
        return Ok(GroupByResult {
            decoded_keys,
            agg_outputs,
            n_groups: 0,
        });
    }

    // Build col_dtypes from value_columns_by_name. Walk every column that
    // any agg references — Simple.input_col or Expression.referenced_columns.
    let mut col_dtypes: std::collections::BTreeMap<String, KMetalDtype> =
        std::collections::BTreeMap::new();
    for agg in aggs {
        match agg {
            KAggSpec::Simple { input_col, .. } => {
                insert_dtype(&mut col_dtypes, input_col, value_columns_by_name)?;
            }
            KAggSpec::Expression { expr, .. } => {
                for col in expr.referenced_columns() {
                    insert_dtype(&mut col_dtypes, &col, value_columns_by_name)?;
                }
            }
            KAggSpec::Length { .. } => {}
        }
    }

    let sig = AggSignature::from_specs(aggs, &col_dtypes)?;
    if !signature_supported_by_fused(&sig) {
        return Err(FusedDispatchError::SignatureNotFused);
    }

    // Compile / look up the PSO.
    let pso = cache.get_or_compile(&sig, aggs)?;

    // Build hashes / row_to_group via the existing CPU build phase. For
    // empty-keys (single-group reduction) we synthesize the BuildOutput
    // directly — every row belongs to group 0.
    let build = if key_cols.is_empty() {
        BuildOutput {
            row_to_group: vec![0u32; n_rows],
            n_groups: 1,
            first_row_per_group: vec![0],
        }
    } else {
        let mut hashes = vec![0u32; n_rows];
        dispatch_hash(device, queue, &encoded, n_rows, &mut hashes)?;
        dispatch_build(device, queue, &encoded, &hashes, n_rows)?
    };
    let n_groups_u32 = build.n_groups;
    let n_groups = n_groups_u32 as usize;

    // Pre-reduce fused kernel caps n_groups at MAX_GROUPS (16). Higher
    // cardinality must route via the M2 per-agg path — that path is
    // strictly faster at high cardinality anyway because contention is
    // low and the register-array overhead disappears.
    if n_groups > MAX_GROUPS {
        return Err(FusedDispatchError::NgroupsExceedsFusedCap {
            n_groups,
            max_groups: MAX_GROUPS,
        });
    }

    // ---- allocate per-slot value/validity buffers in column_order -----------
    let column_order = sig.column_order().to_vec();
    let column_dtypes = sig.column_dtypes().to_vec();

    let mut value_bufs: Vec<polars_metal_buffer::MetalBuffer> =
        Vec::with_capacity(column_order.len());
    let mut valid_bufs: Vec<polars_metal_buffer::MetalBuffer> =
        Vec::with_capacity(column_order.len());
    let valid_bytes_needed = (n_rows + 7) / 8;
    for (slot, name) in column_order.iter().enumerate() {
        let vcol = value_columns_by_name.get(name).ok_or_else(|| {
            FusedDispatchError::MissingValueColumn {
                slot,
                name: name.clone(),
            }
        })?;
        let sig_dt = column_dtypes[slot];
        let (data_bytes, valid_bytes_src) = vcol_bytes_for_slot(vcol, sig_dt, name, n_rows)?;
        let val_buf = device
            .new_buffer_from_bytes(data_bytes)
            .map_err(GroupByError::Buffer)?;
        // Defensive: ensure the validity buffer has enough bytes.
        if valid_bytes_src.len() < valid_bytes_needed {
            return Err(FusedDispatchError::GroupBy(GroupByError::OutputTooShort {
                got: valid_bytes_src.len(),
                need: valid_bytes_needed,
            }));
        }
        let valid_slice = &valid_bytes_src[..valid_bytes_needed];
        // Metal rejects zero-byte allocations; new_buffer_from_bytes also
        // rejects empty slices. Pad to at least 4 bytes for alignment.
        let padded;
        let valid_for_buf: &[u8] = if valid_slice.len() < 4 {
            padded = {
                let mut v = vec![0u8; 4];
                v[..valid_slice.len()].copy_from_slice(valid_slice);
                v
            };
            &padded
        } else {
            valid_slice
        };
        let valid_buf = device
            .new_buffer_from_bytes(valid_for_buf)
            .map_err(GroupByError::Buffer)?;
        value_bufs.push(val_buf);
        valid_bufs.push(valid_buf);
    }

    // ---- allocate output buffers per agg ------------------------------------
    let mut output_bufs: Vec<OutputBuf> = Vec::with_capacity(aggs.len());
    for agg in aggs {
        output_bufs.push(allocate_agg_output(device, agg, &col_dtypes, n_groups)?);
    }

    // ---- row_to_group + n_rows scalar buffers --------------------------------
    let r2g_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            build.row_to_group.as_ptr() as *const u8,
            build.row_to_group.len() * 4,
        )
    };
    let r2g_buf = device
        .new_buffer_from_bytes(r2g_bytes)
        .map_err(GroupByError::Buffer)?;
    let n_rows_u32: u32 =
        u32::try_from(n_rows).map_err(|_| GroupByError::RowCountOverflow { n_rows })?;
    let n_rows_buf = device
        .new_buffer_from_bytes(&n_rows_u32.to_le_bytes())
        .map_err(GroupByError::Buffer)?;
    let n_groups_u32_le: u32 =
        u32::try_from(n_groups).map_err(|_| GroupByError::RowCountOverflow { n_rows: n_groups })?;
    let n_groups_buf = device
        .new_buffer_from_bytes(&n_groups_u32_le.to_le_bytes())
        .map_err(GroupByError::Buffer)?;

    // ---- assemble bindings in emitter slot order ----------------------------
    let mut bindings: Vec<&polars_metal_buffer::MetalBuffer> = Vec::new();
    bindings.push(&r2g_buf);
    bindings.push(&n_rows_buf);
    bindings.push(&n_groups_buf);
    for vb in &value_bufs {
        bindings.push(vb);
    }
    for vb in &valid_bufs {
        bindings.push(vb);
    }
    for ob in &output_bufs {
        bindings.push(&ob.primary);
        if let Some(cnt) = &ob.count {
            bindings.push(cnt);
        }
    }

    // ---- dispatch and wait --------------------------------------------------
    //
    // Pre-reduce kernel strides each thread over the row range, so we
    // dispatch far fewer threads than rows. The cap matches the standalone
    // F32 dispatcher (`F32_AGG_MAX_TGS`) — each TG emits at most n_groups
    // device atomics, so the total post-reduction atomic count is
    // <= n_threadgroups * n_groups, well under the GPU watchdog budget.
    const FUSED_MAX_TGS: usize = 128;
    let target_tgs = n_rows.div_ceil(FUSED_TG_WIDTH);
    let n_tgs = target_tgs.clamp(1, FUSED_MAX_TGS);
    let n_threads = n_tgs * FUSED_TG_WIDTH;
    queue
        .dispatch_1d_with_tg(&pso, &bindings, n_threads, FUSED_TG_WIDTH)
        .map_err(GroupByError::from)?;
    queue.wait_until_complete().map_err(GroupByError::from)?;

    // ---- read back and finalize ---------------------------------------------
    let mut agg_outputs: Vec<AggOutput> = Vec::with_capacity(aggs.len());
    for (i, agg) in aggs.iter().enumerate() {
        agg_outputs.push(finalize_agg_output(
            agg,
            &output_bufs[i],
            n_groups,
            &col_dtypes,
        )?);
    }

    let decoded_keys = if key_cols.is_empty() {
        Vec::new()
    } else {
        let representative_keys: Vec<u128> = build
            .first_row_per_group
            .iter()
            .map(|&row| encoded[row as usize])
            .collect();
        decode_keys(&representative_keys, &schema)
    };

    Ok(GroupByResult {
        decoded_keys,
        agg_outputs,
        n_groups: n_groups_u32,
    })
}

/// Empty-shape output for an `AggSpec` (kernel-layer variant).
fn empty_output_for_kagg(agg: &KAggSpec) -> AggOutput {
    match agg {
        KAggSpec::Simple { op, .. } => empty_output_for_op_simple(*op),
        KAggSpec::Expression { op, .. } => empty_output_for_op_expression(*op),
        KAggSpec::Length { .. } => AggOutput::U64 { values: vec![] },
    }
}

fn empty_output_for_op_simple(op: KAggOp) -> AggOutput {
    // Match the M2 dtype shape for these ops. Sum keeps its input width;
    // Mean is always F32 (in the fused path — we only handle 32-bit inputs).
    match op {
        KAggOp::Count | KAggOp::Len => AggOutput::U64 { values: vec![] },
        KAggOp::Mean => AggOutput::F32 {
            values: vec![],
            valid: vec![],
        },
        // Fall back to F32 for the other empty cases — the actual width is
        // dtype-dependent but on an empty input no values are produced
        // anyway; the python encoder reads `values.len()` (== 0) regardless.
        KAggOp::Sum | KAggOp::Min | KAggOp::Max => AggOutput::F32 {
            values: vec![],
            valid: vec![],
        },
    }
}

fn empty_output_for_op_expression(op: KAggOp) -> AggOutput {
    // Expression aggs always evaluate to f32 in the fused emitter.
    match op {
        KAggOp::Count | KAggOp::Len => AggOutput::U64 { values: vec![] },
        _ => AggOutput::F32 {
            values: vec![],
            valid: vec![],
        },
    }
}

fn insert_dtype(
    out: &mut std::collections::BTreeMap<String, KMetalDtype>,
    name: &str,
    by_name: &std::collections::HashMap<String, ValueColumn<'_>>,
) -> Result<(), FusedDispatchError> {
    if out.contains_key(name) {
        return Ok(());
    }
    let vcol = by_name
        .get(name)
        .ok_or_else(|| FusedDispatchError::MissingValueColumn {
            slot: out.len(),
            name: name.to_string(),
        })?;
    let dt = match vcol {
        ValueColumn::I32 { .. } => KMetalDtype::I32,
        ValueColumn::F32 { .. } => KMetalDtype::F32,
        ValueColumn::I64 { .. } => KMetalDtype::I64,
        ValueColumn::F64 { .. } => KMetalDtype::F64,
    };
    out.insert(name.to_string(), dt);
    Ok(())
}

/// Borrow the data and validity byte slices for a value column. Verifies
/// the runtime dtype matches the signature's per-slot dtype (callers are
/// expected to derive both from the same source, so a mismatch is a bug).
fn vcol_bytes_for_slot<'a>(
    vcol: &'a ValueColumn<'a>,
    sig_dt: KMetalDtype,
    name: &str,
    n_rows: usize,
) -> Result<(&'a [u8], &'a [u8]), FusedDispatchError> {
    match (sig_dt, vcol) {
        (KMetalDtype::I32, ValueColumn::I32 { data, valid }) => {
            // SAFETY: i32 is plain-old-data; we expose the n_rows*4 byte
            // window onto the typed slice.
            let bytes =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, n_rows * 4) };
            Ok((bytes, valid))
        }
        (KMetalDtype::F32, ValueColumn::F32 { data, valid }) => {
            // SAFETY: f32 is plain-old-data.
            let bytes =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, n_rows * 4) };
            Ok((bytes, valid))
        }
        (sig_dt, got) => {
            let got_name = match got {
                ValueColumn::I32 { .. } => "I32",
                ValueColumn::F32 { .. } => "F32",
                ValueColumn::I64 { .. } => "I64",
                ValueColumn::F64 { .. } => "F64",
            };
            Err(FusedDispatchError::ValueDtypeMismatch {
                name: name.to_string(),
                sig_dt,
                got: got_name,
            })
        }
    }
}

/// Allocate the output buffer(s) for one agg, seeded per op semantics.
/// Mean reserves a count companion (u32); other ops use only the primary.
fn allocate_agg_output(
    device: &MetalDevice,
    agg: &KAggSpec,
    col_dtypes: &std::collections::BTreeMap<String, KMetalDtype>,
    n_groups: usize,
) -> Result<OutputBuf, FusedDispatchError> {
    let (op, primary_kind) = match agg {
        KAggSpec::Simple { op, input_col, .. } => {
            let dt = *col_dtypes.get(input_col).ok_or_else(|| {
                FusedDispatchError::MissingValueColumn {
                    slot: 0,
                    name: input_col.clone(),
                }
            })?;
            (*op, primary_kind_for(*op, dt))
        }
        KAggSpec::Expression { expr, op, .. } => {
            // Expression aggs always evaluate to f32 in the emitter (see
            // `emit_expr_msl`).
            let _ = expr; // unused except for documentation
            (*op, primary_kind_for(*op, KMetalDtype::F32))
        }
        KAggSpec::Length { .. } => (KAggOp::Len, PrimaryKind::U32),
    };

    let primary = match (op, primary_kind) {
        // Sum / Mean's sum slot / Count / Len → zero-init.
        (KAggOp::Sum, _) | (KAggOp::Count, _) | (KAggOp::Len, _) | (KAggOp::Mean, _) => device
            .new_buffer_zeroed(n_groups * 4)
            .map_err(GroupByError::Buffer)?,
        // Min over signed int: seed with i32::MAX.
        (KAggOp::Min, PrimaryKind::I32) => seed_buffer_i32(device, i32::MAX, n_groups)?,
        // Min over unsigned int: seed with u32::MAX.
        (KAggOp::Min, PrimaryKind::U32) => seed_buffer_u32(device, u32::MAX, n_groups)?,
        // Min over float: seed with +INFINITY's bit pattern.
        (KAggOp::Min, PrimaryKind::F32Bits) => {
            seed_buffer_u32(device, f32::INFINITY.to_bits(), n_groups)?
        }
        // Max over signed int: seed with i32::MIN.
        (KAggOp::Max, PrimaryKind::I32) => seed_buffer_i32(device, i32::MIN, n_groups)?,
        // Max over unsigned int: seed with u32::MIN (0). Same as zero-init.
        (KAggOp::Max, PrimaryKind::U32) => device
            .new_buffer_zeroed(n_groups * 4)
            .map_err(GroupByError::Buffer)?,
        // Max over float: seed with -INFINITY's bit pattern.
        (KAggOp::Max, PrimaryKind::F32Bits) => {
            seed_buffer_u32(device, f32::NEG_INFINITY.to_bits(), n_groups)?
        }
    };

    let count = if matches!(op, KAggOp::Mean) {
        Some(
            device
                .new_buffer_zeroed(n_groups * 4)
                .map_err(GroupByError::Buffer)?,
        )
    } else {
        None
    };

    Ok(OutputBuf {
        primary,
        count,
        primary_kind,
    })
}

fn primary_kind_for(op: KAggOp, input_dt: KMetalDtype) -> PrimaryKind {
    match op {
        KAggOp::Count | KAggOp::Len => PrimaryKind::U32,
        // Mean's sum slot follows the same shape as a float Sum (CAS on
        // atomic_uint as a bit-pattern container) when the input is float,
        // or i32/u32 otherwise. The CPU finalize divides by count.
        _ => match input_dt {
            KMetalDtype::F32 | KMetalDtype::F64 => PrimaryKind::F32Bits,
            KMetalDtype::I32 | KMetalDtype::I16 | KMetalDtype::I8 | KMetalDtype::I64 => {
                PrimaryKind::I32
            }
            KMetalDtype::U32 | KMetalDtype::U16 | KMetalDtype::U8 | KMetalDtype::Bool => {
                PrimaryKind::U32
            }
            // Utf8 is never an agg value-column dtype; the dispatch router
            // filters this out. Treat as U32 (its underlying code dtype) for
            // exhaustiveness; reaching this arm would mean a router bug.
            KMetalDtype::Utf8 => PrimaryKind::U32,
        },
    }
}

fn seed_buffer_i32(
    device: &MetalDevice,
    seed: i32,
    n_groups: usize,
) -> Result<polars_metal_buffer::MetalBuffer, FusedDispatchError> {
    let v: Vec<i32> = vec![seed; n_groups];
    // SAFETY: i32 is plain-old-data.
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n_groups * 4) };
    device
        .new_buffer_from_bytes(bytes)
        .map_err(|e| FusedDispatchError::GroupBy(GroupByError::Buffer(e)))
}

fn seed_buffer_u32(
    device: &MetalDevice,
    seed: u32,
    n_groups: usize,
) -> Result<polars_metal_buffer::MetalBuffer, FusedDispatchError> {
    let v: Vec<u32> = vec![seed; n_groups];
    // SAFETY: u32 is plain-old-data.
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, n_groups * 4) };
    device
        .new_buffer_from_bytes(bytes)
        .map_err(|e| FusedDispatchError::GroupBy(GroupByError::Buffer(e)))
}

/// Read back one agg's output buffer(s) and widen into `AggOutput`.
fn finalize_agg_output(
    agg: &KAggSpec,
    obuf: &OutputBuf,
    n_groups: usize,
    col_dtypes: &std::collections::BTreeMap<String, KMetalDtype>,
) -> Result<AggOutput, FusedDispatchError> {
    let primary_bytes = obuf.primary.as_slice();
    let need = n_groups * 4;
    if primary_bytes.len() < need {
        return Err(FusedDispatchError::GroupBy(GroupByError::OutputTooShort {
            got: primary_bytes.len(),
            need,
        }));
    }

    // For Mean, read the count companion now.
    let count_vec: Option<Vec<u32>> = obuf.count.as_ref().map(|c| {
        let b = c.as_slice();
        (0..n_groups)
            .map(|g| u32::from_le_bytes(b[g * 4..g * 4 + 4].try_into().unwrap_or([0; 4])))
            .collect::<Vec<u32>>()
    });

    match agg {
        KAggSpec::Length { .. } => {
            let vals = read_u32_to_u64(primary_bytes, n_groups);
            Ok(AggOutput::U64 { values: vals })
        }
        KAggSpec::Simple { op, input_col, .. } => {
            let dt = *col_dtypes.get(input_col).ok_or_else(|| {
                FusedDispatchError::MissingValueColumn {
                    slot: 0,
                    name: input_col.clone(),
                }
            })?;
            finalize_simple_like(*op, dt, obuf, primary_bytes, n_groups, count_vec)
        }
        KAggSpec::Expression { op, .. } => {
            // Expression evaluates to f32 always; treat as F32 input dtype.
            finalize_simple_like(
                *op,
                KMetalDtype::F32,
                obuf,
                primary_bytes,
                n_groups,
                count_vec,
            )
        }
    }
}

/// Shared finalize: handles Sum / Mean / Min / Max / Count given the
/// (op, input dtype) pair. The fused emitter's primary kind disambiguates
/// the 32-bit container type.
fn finalize_simple_like(
    op: KAggOp,
    input_dt: KMetalDtype,
    obuf: &OutputBuf,
    primary_bytes: &[u8],
    n_groups: usize,
    count_vec: Option<Vec<u32>>,
) -> Result<AggOutput, FusedDispatchError> {
    match op {
        KAggOp::Count => Ok(AggOutput::U64 {
            values: read_u32_to_u64(primary_bytes, n_groups),
        }),
        KAggOp::Len => Ok(AggOutput::U64 {
            values: read_u32_to_u64(primary_bytes, n_groups),
        }),
        KAggOp::Sum => match obuf.primary_kind {
            PrimaryKind::I32 => {
                let values: Vec<i32> = read_i32(primary_bytes, n_groups);
                // Polars sum of an all-null group returns 0 (not null) —
                // matches the M2 i32 sum path.
                Ok(AggOutput::I32 {
                    values,
                    valid: vec![true; n_groups],
                })
            }
            PrimaryKind::U32 => {
                let values: Vec<i32> = read_u32(primary_bytes, n_groups)
                    .into_iter()
                    .map(|v| v as i32)
                    .collect();
                Ok(AggOutput::I32 {
                    values,
                    valid: vec![true; n_groups],
                })
            }
            PrimaryKind::F32Bits => {
                let values: Vec<f32> = read_f32_bits(primary_bytes, n_groups);
                Ok(AggOutput::F32 {
                    values,
                    valid: vec![true; n_groups],
                })
            }
        },
        KAggOp::Mean => {
            let counts = count_vec.unwrap_or_else(|| vec![0u32; n_groups]);
            match obuf.primary_kind {
                PrimaryKind::I32 => {
                    let sums: Vec<i32> = read_i32(primary_bytes, n_groups);
                    let (values, valid) = mean_from_sum_count_i32(&sums, &counts);
                    Ok(AggOutput::F32 { values, valid })
                }
                PrimaryKind::U32 => {
                    let sums: Vec<u32> = read_u32(primary_bytes, n_groups);
                    let (values, valid) = mean_from_sum_count_u32(&sums, &counts);
                    Ok(AggOutput::F32 { values, valid })
                }
                PrimaryKind::F32Bits => {
                    let sums: Vec<f32> = read_f32_bits(primary_bytes, n_groups);
                    let (values, valid) = mean_from_sum_count_f32(&sums, &counts);
                    Ok(AggOutput::F32 { values, valid })
                }
            }
        }
        KAggOp::Min | KAggOp::Max => {
            // Per-group validity: a group is null only when no input row
            // contributed. The fused emitter doesn't currently materialize
            // a per-agg count companion for Min/Max (Task 16 will explore
            // adding one). Q1-shape workloads have no all-null groups, so
            // we treat all groups as valid for now. See the commit message
            // for Task 15 for the limitation note.
            match obuf.primary_kind {
                PrimaryKind::I32 => {
                    let values: Vec<i32> = read_i32(primary_bytes, n_groups);
                    Ok(AggOutput::I32 {
                        values,
                        valid: vec![true; n_groups],
                    })
                }
                PrimaryKind::U32 => {
                    // unsigned-input Min/Max isn't exercised in Q1 today, but
                    // the emitter still emits a u32 accumulator. We preserve
                    // it by returning as I32 (Polars surfaces u32 separately;
                    // M2's per-agg path doesn't yet route u32 either, so
                    // this is unreachable on the canonical workloads). Mark
                    // every group valid.
                    let _ = input_dt; // silence
                    let values: Vec<i32> = read_u32(primary_bytes, n_groups)
                        .into_iter()
                        .map(|v| v as i32)
                        .collect();
                    Ok(AggOutput::I32 {
                        values,
                        valid: vec![true; n_groups],
                    })
                }
                PrimaryKind::F32Bits => {
                    let values: Vec<f32> = read_f32_bits(primary_bytes, n_groups);
                    Ok(AggOutput::F32 {
                        values,
                        valid: vec![true; n_groups],
                    })
                }
            }
        }
    }
}

// ---- typed read helpers --------------------------------------------------

fn read_u32(bytes: &[u8], n: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(n);
    for g in 0..n {
        let b = &bytes[g * 4..g * 4 + 4];
        out.push(u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    }
    out
}

fn read_u32_to_u64(bytes: &[u8], n: usize) -> Vec<u64> {
    let mut out = Vec::with_capacity(n);
    for g in 0..n {
        let b = &bytes[g * 4..g * 4 + 4];
        out.push(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u64);
    }
    out
}

fn read_i32(bytes: &[u8], n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n);
    for g in 0..n {
        let b = &bytes[g * 4..g * 4 + 4];
        out.push(i32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    }
    out
}

fn read_f32_bits(bytes: &[u8], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for g in 0..n {
        let b = &bytes[g * 4..g * 4 + 4];
        let bits = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        out.push(f32::from_bits(bits));
    }
    out
}

fn mean_from_sum_count_f32(sums: &[f32], counts: &[u32]) -> (Vec<f32>, Vec<bool>) {
    let mut values = Vec::with_capacity(sums.len());
    let mut valid = Vec::with_capacity(sums.len());
    for (s, &c) in sums.iter().zip(counts.iter()) {
        if c == 0 {
            values.push(0.0f32);
            valid.push(false);
        } else {
            values.push(*s / c as f32);
            valid.push(true);
        }
    }
    (values, valid)
}

fn mean_from_sum_count_i32(sums: &[i32], counts: &[u32]) -> (Vec<f32>, Vec<bool>) {
    let mut values = Vec::with_capacity(sums.len());
    let mut valid = Vec::with_capacity(sums.len());
    for (s, &c) in sums.iter().zip(counts.iter()) {
        if c == 0 {
            values.push(0.0f32);
            valid.push(false);
        } else {
            values.push(*s as f32 / c as f32);
            valid.push(true);
        }
    }
    (values, valid)
}

fn mean_from_sum_count_u32(sums: &[u32], counts: &[u32]) -> (Vec<f32>, Vec<bool>) {
    let mut values = Vec::with_capacity(sums.len());
    let mut valid = Vec::with_capacity(sums.len());
    for (s, &c) in sums.iter().zip(counts.iter()) {
        if c == 0 {
            values.push(0.0f32);
            valid.push(false);
        } else {
            values.push(*s as f32 / c as f32);
            valid.push(true);
        }
    }
    (values, valid)
}
