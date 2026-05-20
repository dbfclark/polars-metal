// crates/polars-metal-core/src/arena.rs
//
// Per-query scratch arena. M0 ships the trait + a stub that allocates
// fresh MTLBuffers; M1+ replaces the stub with a free-list-by-size-class.

use polars_metal_buffer::{MetalBuffer, MetalDevice};

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
