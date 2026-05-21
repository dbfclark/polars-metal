// crates/polars-metal-core/src/arena.rs
//
// Per-query scratch arena.
//
// M0 shipped a `ScratchArena` trait + `StubArena` that allocated fresh
// MTLBuffers per `reserve` call. M1 introduces `BumpArena`, which
// pre-allocates a single large `StorageModeShared` MTLBuffer up front and
// serves allocations as 16-byte-aligned views into it. Each `alloc` returns
// a `MetalBuffer` that is a no-copy view onto the parent's bytes; dropping
// the view does not free the bytes (only dropping the arena does).
//
// The `StubArena` is kept around for now so existing M0 callers keep
// compiling. It will be deleted once nothing references it.

use std::sync::Arc;

use polars_metal_buffer::{BufferError, MetalBuffer, MetalDevice};

use crate::EngineError;

pub trait ScratchArena: Send {
    fn reserve(&mut self, bytes: usize) -> Result<MetalBuffer, EngineError>;
    fn reset(&mut self);
}

pub struct StubArena {
    device: MetalDevice,
}

// SAFETY: `MTLDevice` objects are thread-safe on Apple platforms: Metal
// explicitly documents that device and command-queue objects may be used from
// any thread. objc2 does not reflect this automatically because `ProtocolObject`
// is a dyn type and therefore conservatively not `Send`, but the underlying
// Objective-C object satisfies the invariant.
unsafe impl Send for StubArena {}

impl StubArena {
    pub fn new(device: MetalDevice) -> Self {
        Self { device }
    }
}

impl ScratchArena for StubArena {
    fn reserve(&mut self, bytes: usize) -> Result<MetalBuffer, EngineError> {
        // Stub: always allocates fresh. M1+ uses a free list.
        let dummy = std::sync::Arc::new(arrow_buffer::Buffer::from_vec(vec![0u8; bytes]));
        Ok(MetalBuffer::from_arrow(&self.device, dummy)?)
    }

    fn reset(&mut self) {
        // No-op in the stub.
    }
}

/// Alignment (in bytes) of every allocation handed out by [`BumpArena`].
///
/// 16 bytes is the standard MSL-friendly alignment (covers `float4`, `uint4`,
/// etc.) and is also conservative enough for any Arrow primitive layout.
const ARENA_ALIGN: usize = 16;

/// Per-query bump allocator over a single `StorageModeShared` MTLBuffer.
///
/// Allocations are served as 16-byte-aligned offsets into the backing buffer.
/// Each [`MetalBuffer`] returned by [`BumpArena::alloc`] is a Metal-level view
/// that shares storage with the arena (via `newBufferWithBytesNoCopy:` on the
/// parent's shared-memory mapping); the parent stays alive via an
/// `Arc<MetalBuffer>` keep-alive captured by the view.
///
/// The arena does *not* support per-allocation free — dropping the arena
/// releases every outstanding view (provided no `Arc<MetalBuffer>` clones of
/// the backing buffer remain elsewhere). This matches the
/// allocate-many/free-once lifetime of a single Polars UDF call.
pub struct BumpArena {
    backing: Arc<MetalBuffer>,
    cursor: usize,
}

impl BumpArena {
    /// Pre-allocate a single `StorageModeShared` MTLBuffer of `bytes` bytes
    /// and wrap it as the arena's backing store.
    ///
    /// Returns the underlying [`BufferError`] if Metal refuses the
    /// allocation (e.g. `bytes == 0` or process memory exhaustion).
    pub fn with_capacity(device: &MetalDevice, bytes: usize) -> Result<Self, BufferError> {
        let backing = Arc::new(device.new_buffer_zeroed(bytes)?);
        Ok(Self { backing, cursor: 0 })
    }

    /// Hand out a 16-byte-aligned view of `bytes` bytes from the backing
    /// buffer.
    ///
    /// The cursor is advanced past the returned region (including any
    /// alignment padding). On exhaustion or zero-length request, returns
    /// [`BufferError::AllocationFailed`]; the cursor is left unchanged in
    /// that case so callers may retry against a fresh arena.
    pub fn alloc(&mut self, bytes: usize) -> Result<MetalBuffer, BufferError> {
        if bytes == 0 {
            return Err(BufferError::AllocationFailed { bytes: 0 });
        }
        let aligned_start = self
            .cursor
            .checked_add(ARENA_ALIGN - 1)
            .ok_or(BufferError::AllocationFailed { bytes })?
            & !(ARENA_ALIGN - 1);
        let aligned_end = aligned_start
            .checked_add(bytes)
            .ok_or(BufferError::AllocationFailed { bytes })?;
        if aligned_end > self.backing.len() {
            return Err(BufferError::AllocationFailed { bytes });
        }
        // SAFETY: `aligned_start + bytes <= self.backing.len()` (bounds-checked
        // immediately above). The backing buffer is `StorageModeShared`, which
        // is the same mode `view_into` requires. The view captures an
        // `Arc<MetalBuffer>` keep-alive on `self.backing`, so the parent
        // outlives the view independently of the arena's own lifetime.
        let view = unsafe { MetalBuffer::view_into(&self.backing, aligned_start, bytes)? };
        self.cursor = aligned_end;
        Ok(view)
    }

    /// Returns an `Arc` clone of the backing buffer.
    ///
    /// Useful for keep-alive patterns where an output Arrow buffer needs to
    /// extend the arena's effective lifetime past the UDF call.
    pub fn shared(&self) -> Arc<MetalBuffer> {
        self.backing.clone()
    }

    /// Total capacity of the arena in bytes (size of the backing buffer).
    pub fn capacity(&self) -> usize {
        self.backing.len()
    }

    /// Number of bytes consumed (including alignment padding) from the
    /// backing buffer so far.
    pub fn used(&self) -> usize {
        self.cursor
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn stub_arena_reserves_and_resets() {
        let device = MetalDevice::system_default().expect("Metal device");
        let mut a = StubArena::new(device);
        let buf = a.reserve(4096).expect("reserve");
        assert_eq!(buf.len(), 4096);
        a.reset();
        let buf2 = a.reserve(2048).expect("reserve");
        assert_eq!(buf2.len(), 2048);
    }
}
