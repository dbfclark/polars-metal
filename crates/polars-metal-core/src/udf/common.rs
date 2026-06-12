use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Build a `(MetalDevice, CommandQueue)` pair for one comparison-kernel
/// dispatch. The kernel dispatcher reuses the queue across its three
/// internal passes (load + compute + readback) so callers don't share
/// queues across pyfunction invocations.
pub(crate) fn new_device_and_queue() -> PyResult<(MetalDevice, CommandQueue)> {
    let device = MetalDevice::system_default()
        .map_err(|e| crate::engine_err(crate::EngineError::Buffer(e)))?;
    let queue = CommandQueue::new(&device)
        .map_err(|e| crate::engine_err(crate::EngineError::Other(format!("command queue: {e}"))))?;
    Ok((device, queue))
}

/// Length-check the data and validity buffers for a numeric column input
/// to a comparison kernel. `data_bytes_per_row` is 8 for i64/f64 (the
/// only widths we support today). Validity is bit-packed.
pub(crate) fn check_numeric_buffers(
    data: &[u8],
    valid: &[u8],
    n_rows: usize,
    data_bytes_per_row: usize,
) -> PyResult<()> {
    let expected_data = n_rows * data_bytes_per_row;
    if data.len() < expected_data {
        return Err(PyValueError::new_err(format!(
            "polars_metal: data buffer is {got} B, need at least {expected} B for {n} rows",
            got = data.len(),
            expected = expected_data,
            n = n_rows,
        )));
    }
    let min_valid = (n_rows + 7) / 8;
    if valid.len() < min_valid {
        return Err(PyValueError::new_err(format!(
            "polars_metal: validity buffer is {got} B, need at least {expected} B for {n} rows",
            got = valid.len(),
            expected = min_valid,
            n = n_rows,
        )));
    }
    Ok(())
}

/// Length-check a bit-packed bool buffer (data or validity) for an
/// `n_rows`-long input to the `bool_and` / `bool_or` kernels.
pub(crate) fn check_bitpacked_buffer(buf: &[u8], n_rows: usize, label: &str) -> PyResult<()> {
    let min_bytes = (n_rows + 7) / 8;
    if buf.len() < min_bytes {
        return Err(PyValueError::new_err(format!(
            "polars_metal: {label} buffer is {got} B, need at least {expected} B for {n} rows",
            label = label,
            got = buf.len(),
            expected = min_bytes,
            n = n_rows,
        )));
    }
    Ok(())
}

/// Pack a `Vec<bool>` validity slice into a 4-byte-aligned little-endian
/// bit-packed validity bitmap, matching Arrow's convention.
pub(crate) fn pack_valid_bitmap(bits: &[bool]) -> Vec<u8> {
    let n_bytes = ((bits.len() + 7) / 8 + 3) & !3;
    let n_bytes = n_bytes.max(4);
    let mut out = vec![0u8; n_bytes];
    for (i, &b) in bits.iter().enumerate() {
        if b {
            out[i >> 3] |= 1 << (i & 7);
        }
    }
    out
}
