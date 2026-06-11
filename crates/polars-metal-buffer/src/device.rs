// crates/polars-metal-buffer/src/device.rs
//
// Need to link CoreGraphics to pull in MTLCreateSystemDefaultDevice.
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer as _, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};

use crate::{BufferError, MetalBuffer};

/// A handle to an `MTLDevice`. Constructed once per process; cloning is cheap
/// (it bumps an Objective-C refcount).
#[derive(Clone)]
pub struct MetalDevice {
    inner: Retained<ProtocolObject<dyn MTLDevice>>,
}

impl MetalDevice {
    /// Acquire the system-default Metal device.
    ///
    /// Returns `Err(BufferError::AllocationFailed { bytes: 0 })` when no
    /// Metal-capable GPU is present (e.g. running on a non-Apple-Silicon
    /// machine or inside a CI sandbox without GPU access).
    pub fn system_default() -> Result<Self, BufferError> {
        // SAFETY: MTLCreateSystemDefaultDevice is safe to call; it returns NULL
        // when no device is available (e.g., non-Metal hardware).
        let raw = unsafe { MTLCreateSystemDefaultDevice() };
        // SAFETY: If non-null, the pointer is a valid +1 Objective-C object
        // returned by a function whose name begins with "Create" (CF ownership
        // convention), which objc2's Retained takes ownership of.
        let inner =
            unsafe { Retained::from_raw(raw) }.ok_or(BufferError::AllocationFailed { bytes: 0 })?;
        Ok(Self { inner })
    }

    /// Human-readable name of the GPU (e.g. "Apple M2 Ultra").
    pub fn name(&self) -> String {
        // name() returns Retained<NSString>; NSString implements Display.
        self.inner.name().to_string()
    }

    /// Allocate a new shared-storage `MTLBuffer` of the given length, zeroed.
    ///
    /// Metal does not specify the initial contents of a freshly allocated
    /// `newBufferWithLength:options:` buffer, so we explicitly zero the
    /// shared-memory region before returning. Allocation is via
    /// `MTLResourceStorageModeShared`, which on Apple Silicon means the
    /// buffer lives in unified memory and is directly CPU-addressable.
    ///
    /// Returns `BufferError::AllocationFailed` when `bytes == 0` (Metal
    /// rejects zero-byte allocations) or when Metal otherwise refuses to
    /// allocate (e.g. process memory exhaustion).
    pub fn new_buffer_zeroed(&self, bytes: usize) -> Result<MetalBuffer, BufferError> {
        if bytes == 0 {
            return Err(BufferError::AllocationFailed { bytes: 0 });
        }
        // `newBufferWithLength:options:` is a safe fn in `objc2-metal` 0.2
        // (no raw pointers in or out). It returns `None` if Metal refuses
        // the allocation, which we map to `AllocationFailed`.
        let inner = self
            .inner
            .newBufferWithLength_options(bytes, MTLResourceOptions::MTLResourceStorageModeShared)
            .ok_or(BufferError::AllocationFailed { bytes })?;

        // Freshly allocated MTLBuffer contents are implementation-defined;
        // zero them via the StorageModeShared CPU mapping so callers can
        // rely on a known initial state.
        //
        // `contents()` returns `NonNull<c_void>` pointing at the unified-
        // memory backing store. We have just allocated this buffer, so no
        // GPU command is in-flight against it: writing from the CPU is
        // safe without an explicit synchronization barrier.
        let ptr = inner.contents().cast::<u8>();
        // SAFETY:
        // - `ptr` is non-null (NonNull invariant from `contents()`).
        // - The buffer is `bytes` long and StorageModeShared, so the entire
        //   range is CPU-addressable.
        // - No other thread or GPU command holds a reference to this buffer
        //   yet (we just allocated it), so there is no concurrent access.
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr(), 0u8, bytes);
        }

        Ok(MetalBuffer::from_metal_owned(inner))
    }

    /// Allocate a new shared-storage `MTLBuffer` and copy `bytes` into it.
    ///
    /// Mirrors [`new_buffer_zeroed`](Self::new_buffer_zeroed) but seeds the
    /// freshly allocated buffer with caller-supplied contents in a single
    /// `newBufferWithBytes:length:options:` call. Storage mode is
    /// `MTLResourceStorageModeShared`, so on Apple Silicon the buffer lives
    /// in unified memory and is directly CPU-addressable after construction.
    ///
    /// This is the standard kernel-input constructor: callers stage a
    /// `Vec<u8>` (or any `&[u8]`) of predicate / validity / scalar bytes and
    /// hand it off without a separate allocate-then-copy sequence.
    ///
    /// Returns `BufferError::AllocationFailed { bytes: 0 }` when `bytes` is
    /// empty (Metal rejects zero-byte allocations) or `AllocationFailed
    /// { bytes }` when Metal otherwise refuses to allocate.
    pub fn new_buffer_from_bytes(&self, bytes: &[u8]) -> Result<MetalBuffer, BufferError> {
        let len = bytes.len();
        if len == 0 {
            return Err(BufferError::AllocationFailed { bytes: 0 });
        }
        let ptr = std::ptr::NonNull::new(bytes.as_ptr() as *mut std::ffi::c_void)
            .ok_or(BufferError::AllocationFailed { bytes: len })?;

        // SAFETY:
        // - `ptr` is valid for `len` bytes for the duration of this call
        //   (the input slice is alive for the call's stack frame and Metal
        //   copies the bytes synchronously into its own allocation).
        // - `newBufferWithBytes:length:options:` performs an internal copy
        //   into the new MTLBuffer; we do not retain `bytes` past the call.
        // - StorageModeShared matches the rest of the buffer-bridge
        //   allocator; the resulting buffer is CPU- and GPU-addressable in
        //   unified memory.
        let inner = unsafe {
            self.inner.newBufferWithBytes_length_options(
                ptr,
                len,
                MTLResourceOptions::MTLResourceStorageModeShared,
            )
        }
        .ok_or(BufferError::AllocationFailed { bytes: len })?;

        Ok(MetalBuffer::from_metal_owned(inner))
    }

    /// Allocate a new shared-storage `MTLBuffer` of the given length, WITHOUT
    /// zeroing its contents.
    ///
    /// Mirrors [`new_buffer_zeroed`](Self::new_buffer_zeroed) but skips the
    /// `write_bytes` zero-fill. Intended for staging buffers whose full used
    /// prefix is overwritten by a `memcpy` before any read (e.g.
    /// [`crate::StagingPool`]); bytes beyond the staged length are never read.
    ///
    /// Storage mode is `MTLResourceStorageModeShared` (unified memory, CPU- and
    /// GPU-addressable, page-aligned base). Returns
    /// `BufferError::AllocationFailed` when `bytes == 0` or Metal refuses.
    pub fn new_buffer_uninit(&self, bytes: usize) -> Result<MetalBuffer, BufferError> {
        if bytes == 0 {
            return Err(BufferError::AllocationFailed { bytes: 0 });
        }
        let inner = self
            .inner
            .newBufferWithLength_options(bytes, MTLResourceOptions::MTLResourceStorageModeShared)
            .ok_or(BufferError::AllocationFailed { bytes })?;
        Ok(MetalBuffer::from_metal_owned(inner))
    }

    /// Raw borrow of the underlying `MTLDevice` protocol object.
    ///
    /// Exposed to sibling crates (`polars-metal-kernels`,
    /// `polars-metal-core`) so they can call Metal APIs such as
    /// `newLibraryWithURL`, `newCommandQueue`, and
    /// `newComputePipelineStateWithFunction` without re-implementing the
    /// device handle. Most of those calls are `unsafe fn`s in
    /// `objc2-metal`; callers must add a `// SAFETY:` comment per workspace
    /// convention.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.inner
    }
}

impl std::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalDevice")
            .field("name", &self.name())
            .finish()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn system_default_device_acquires_and_has_a_name() {
        let device =
            MetalDevice::system_default().expect("Metal-capable hardware required for this test");
        let name = device.name();
        assert!(!name.is_empty(), "device name should be non-empty");
        eprintln!("Metal device: {name}");
    }

    #[test]
    fn clone_is_cheap_and_gives_same_name() {
        let device =
            MetalDevice::system_default().expect("Metal-capable hardware required for this test");
        let clone = device.clone();
        assert_eq!(device.name(), clone.name());
    }

    #[test]
    fn new_buffer_zeroed_allocates_and_zeroes() {
        let device =
            MetalDevice::system_default().expect("Metal-capable hardware required for this test");
        let buf = device.new_buffer_zeroed(256).expect("allocation succeeds");
        assert_eq!(buf.len(), 256);
        assert!(
            buf.as_slice().iter().all(|b| *b == 0),
            "newly-allocated buffer should be all zero"
        );
    }

    #[test]
    fn new_buffer_zeroed_rejects_zero_length() {
        let device =
            MetalDevice::system_default().expect("Metal-capable hardware required for this test");
        match device.new_buffer_zeroed(0) {
            Ok(_) => panic!("zero-byte allocation must error"),
            Err(BufferError::AllocationFailed { bytes: 0 }) => {}
            Err(other) => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn new_buffer_zeroed_is_cpu_writable() {
        // Sanity check that StorageModeShared lets us mutate the buffer from
        // the CPU side after construction. This is the contract dispatch
        // tests rely on for asserting kernel outputs.
        let device =
            MetalDevice::system_default().expect("Metal-capable hardware required for this test");
        let buf = device.new_buffer_zeroed(8).expect("allocation succeeds");
        // SAFETY: `as_slice` exposes the shared-memory mapping; we cast to
        // `*mut u8` to scribble bytes the same way a kernel later would.
        // No GPU work is in flight, so the write is race-free.
        unsafe {
            let p = buf.as_slice().as_ptr() as *mut u8;
            for i in 0..8 {
                *p.add(i) = (i as u8) + 1;
            }
        }
        assert_eq!(buf.as_slice(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn new_buffer_from_bytes_copies_input() {
        let device =
            MetalDevice::system_default().expect("Metal-capable hardware required for this test");
        let payload: [u8; 6] = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE];
        let buf = device
            .new_buffer_from_bytes(&payload)
            .expect("allocation succeeds");
        assert_eq!(buf.len(), payload.len());
        assert_eq!(buf.as_slice(), &payload);
    }

    #[test]
    fn new_buffer_from_bytes_rejects_empty_input() {
        let device =
            MetalDevice::system_default().expect("Metal-capable hardware required for this test");
        match device.new_buffer_from_bytes(&[]) {
            Ok(_) => panic!("empty-input allocation must error"),
            Err(BufferError::AllocationFailed { bytes: 0 }) => {}
            Err(other) => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn debug_includes_name() {
        let device =
            MetalDevice::system_default().expect("Metal-capable hardware required for this test");
        let debug_str = format!("{device:?}");
        assert!(
            debug_str.contains("MetalDevice"),
            "debug should contain type name"
        );
        assert!(
            debug_str.contains("name"),
            "debug should contain field name"
        );
    }
}
