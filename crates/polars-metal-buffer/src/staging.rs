//! Reusable page-aligned staging buffer for kernel inputs (M6 B3b).
//!
//! Ingesting a Polars/Arrow column into Metal requires one copy: Polars' Arrow
//! buffers are 64-byte aligned, while `newBufferWithBytesNoCopy` hard-requires
//! 16 KB page alignment (see [`crate::alignment`]). A spike showed the cost of
//! the current per-call `newBufferWithBytes` path is the *allocation*, not the
//! copy — a reused Shared buffer + `memcpy` is ~5× faster than allocating a
//! fresh buffer each call. [`StagingPool`] holds one growable Shared
//! [`MetalBuffer`] and reallocates only when a larger input arrives.
//!
//! The pool buffer's capacity may exceed the staged length; kernels that take
//! an explicit element count (`n`) and read only `input[0..n)` are unaffected.
//! Callers must pass the true element count to the dispatcher, NOT
//! `buffer.len()`.

use crate::{BufferError, MetalBuffer, MetalDevice};

/// A single reusable, growable, page-aligned Shared staging buffer.
///
/// Not `Sync`/thread-safe on its own — wrap in a `Mutex` for cross-thread use
/// (Metal command submission serializes anyway). One pool holds at most one
/// buffer, sized to the largest input seen so far.
#[derive(Default)]
pub struct StagingPool {
    buf: Option<MetalBuffer>,
}

impl StagingPool {
    /// A pool with no buffer yet allocated.
    pub const fn new() -> Self {
        Self { buf: None }
    }

    /// Stage `src` into the pooled buffer and return it.
    ///
    /// Reallocates (via [`MetalDevice::new_buffer_uninit`]) only when `src` is
    /// larger than the current capacity; otherwise reuses the existing buffer.
    /// The first `src.len()` bytes are overwritten by `memcpy`; bytes beyond
    /// are left as-is (never read by a kernel that respects its `n`).
    ///
    /// Returns `BufferError::AllocationFailed { bytes: 0 }` for an empty input
    /// (Metal rejects zero-byte buffers).
    pub fn stage(&mut self, device: &MetalDevice, src: &[u8]) -> Result<&MetalBuffer, BufferError> {
        let need = src.len();
        if need == 0 {
            return Err(BufferError::AllocationFailed { bytes: 0 });
        }
        let grow = self.buf.as_ref().is_none_or(|b| b.len() < need);
        if grow {
            self.buf = Some(device.new_buffer_uninit(need)?);
        }
        let buf = self
            .buf
            .as_mut()
            .ok_or(BufferError::AllocationFailed { bytes: need })?;
        buf.as_mut_slice()[..need].copy_from_slice(src);
        Ok(buf)
    }
}
