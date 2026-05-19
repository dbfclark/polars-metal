// crates/polars-metal-buffer/src/bridge.rs
//
// Arrow Buffer ↔ MTLBuffer bridge, implementing two regimes:
//   1. Zero-copy: when the Arrow buffer's pointer and length are both
//      page-aligned, we wrap the existing allocation via
//      `newBufferWithBytesNoCopy:length:options:deallocator:`.  The
//      deallocator block holds an `Arc<ArrowBuffer>` so the underlying
//      memory outlives the MTLBuffer.
//   2. Copy: all other cases — allocate a fresh MTLBuffer and let Metal
//      copy the bytes in.
//
// Storage mode is always `StorageModeShared` (Unified Memory), which
// makes the buffer CPU-readable without an explicit blit.

use std::ptr::NonNull;
use std::sync::Arc;

use arrow_buffer::Buffer as ArrowBuffer;
use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice as _, MTLResourceOptions};

use crate::{is_page_aligned, BufferError, MetalDevice};

/// A wrapper around an `MTLBuffer` in Shared storage mode.
///
/// Lifetime is managed via Objective-C reference counting (through
/// `Retained<…>`).  When constructed via the zero-copy path, the
/// `Arc<ArrowBuffer>` keep-alive ensures the underlying allocation
/// remains valid for the MTLBuffer's lifetime.
pub struct MetalBuffer {
    inner: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Keeps the source `ArrowBuffer` alive when we took the zero-copy path.
    /// `None` on the copy path (Metal owns its own allocation).
    _owner: Option<Arc<ArrowBuffer>>,
}

// MTLBuffer is backed by shared memory and is used from a single thread
// within a command submission; the underlying Objective-C object is
// thread-safe at the refcount level.
// SAFETY: We never mutate the buffer contents through Rust concurrently,
// and Metal's thread model guarantees safe concurrent GPU access.
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl MetalBuffer {
    /// Wrap an `ArrowBuffer` as a Metal buffer.
    ///
    /// Uses the zero-copy path when `arrow` is page-aligned (both pointer
    /// and length), otherwise copies the bytes into a new Metal allocation.
    pub fn from_arrow(device: &MetalDevice, arrow: Arc<ArrowBuffer>) -> Result<Self, BufferError> {
        let ptr = arrow.as_ptr() as usize;
        let len = arrow.len();
        if len > 0 && is_page_aligned(ptr, len) {
            Self::zero_copy(device, arrow)
        } else {
            Self::copy(device, &arrow)
        }
    }

    /// Zero-copy path: wrap the Arrow buffer's existing allocation.
    ///
    /// The deallocator block holds a clone of the `Arc` so the Arrow
    /// allocation stays alive until Metal releases the buffer.
    fn zero_copy(device: &MetalDevice, arrow: Arc<ArrowBuffer>) -> Result<Self, BufferError> {
        let len = arrow.len();
        // SAFETY: `arrow.as_ptr()` is non-null (len > 0 checked by caller).
        let ptr = NonNull::new(arrow.as_ptr() as *mut std::ffi::c_void)
            .ok_or(BufferError::AllocationFailed { bytes: len })?;

        // Clone the Arc into the block so it's dropped when Metal deallocates
        // the buffer (i.e., when the MTLBuffer is released).
        let owner_for_dealloc = arrow.clone();
        // The deallocator block signature is `(NonNull<c_void>, NSUInteger) -> ()`.
        // We ignore both arguments because we're dropping via Arc, not by freeing
        // the pointer directly.
        let deallocator: RcBlock<dyn Fn(NonNull<std::ffi::c_void>, usize)> =
            RcBlock::new(move |_ptr, _len| {
                drop(owner_for_dealloc.clone());
            });

        // SAFETY:
        // - `ptr` is valid for `len` bytes for the entire duration the MTLBuffer
        //   exists: the `Arc<ArrowBuffer>` in `_owner` (and in `deallocator`)
        //   ensures the backing allocation is not freed.
        // - Pointer and length are page-aligned (caller-checked).
        // - We never mutate through the Rust `&[u8]` slice and Metal accesses
        //   the buffer only via GPU commands submitted after construction.
        let inner = unsafe {
            device
                .raw()
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    ptr,
                    len,
                    MTLResourceOptions::MTLResourceStorageModeShared,
                    Some(&deallocator),
                )
        }
        .ok_or(BufferError::AllocationFailed { bytes: len })?;

        Ok(Self {
            inner,
            _owner: Some(arrow),
        })
    }

    /// Copy path: allocate a new MTLBuffer and copy the Arrow bytes in.
    fn copy(device: &MetalDevice, arrow: &ArrowBuffer) -> Result<Self, BufferError> {
        let len = arrow.len();
        if len == 0 {
            // Metal does not allow zero-byte buffers; treat this as a
            // caller-side misuse rather than silently succeeding.
            return Err(BufferError::AllocationFailed { bytes: 0 });
        }
        // SAFETY: `arrow.as_ptr()` is valid for `len` bytes (live ArrowBuffer).
        let ptr = NonNull::new(arrow.as_ptr() as *mut std::ffi::c_void)
            .ok_or(BufferError::AllocationFailed { bytes: len })?;

        // SAFETY: pointer is valid for `len` bytes (live ArrowBuffer).
        let inner = unsafe {
            device.raw().newBufferWithBytes_length_options(
                ptr,
                len,
                MTLResourceOptions::MTLResourceStorageModeShared,
            )
        }
        .ok_or(BufferError::AllocationFailed { bytes: len })?;

        Ok(Self {
            inner,
            _owner: None,
        })
    }

    /// Byte length of the buffer.
    pub fn len(&self) -> usize {
        self.inner.length()
    }

    /// Returns `true` if the buffer contains no bytes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// View the buffer's contents as a byte slice.
    ///
    /// Valid as long as `self` is alive and no GPU writes are in-flight.
    pub fn as_slice(&self) -> &[u8] {
        // contents() returns NonNull<c_void> pointing at the shared-memory
        // backing store, which is CPU-accessible in StorageModeShared.
        let ptr = self.inner.contents().cast::<u8>();
        let len = self.len();
        // SAFETY:
        // - `ptr` is non-null (NonNull invariant) and valid for `len` bytes.
        // - The lifetime of the slice is tied to `&self`, which keeps the
        //   Retained<MTLBuffer> alive.
        // - StorageModeShared guarantees CPU reads are coherent with GPU writes
        //   once any in-flight command buffer has completed.
        unsafe { std::slice::from_raw_parts(ptr.as_ptr(), len) }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::MetalDevice;

    fn device() -> MetalDevice {
        MetalDevice::system_default().expect("Metal-capable hardware required")
    }

    // ── Copy path ────────────────────────────────────────────────────────────

    #[test]
    fn copy_path_handles_misaligned_buffer() {
        let device = device();
        // A 5-byte Vec is almost certainly not page-aligned.
        let arrow = Arc::new(ArrowBuffer::from_vec(vec![1u8, 2, 3, 4, 5]));
        let metal = MetalBuffer::from_arrow(&device, arrow).expect("allocation must succeed");
        assert_eq!(metal.len(), 5);
        assert_eq!(metal.as_slice(), &[1u8, 2, 3, 4, 5]);
    }

    #[test]
    fn copy_path_large_buffer() {
        let device = device();
        let data: Vec<u8> = (0u8..=255).cycle().take(4096).collect();
        let expected = data.clone();
        let arrow = Arc::new(ArrowBuffer::from_vec(data));
        let metal = MetalBuffer::from_arrow(&device, arrow).expect("allocation must succeed");
        assert_eq!(metal.len(), 4096);
        assert_eq!(metal.as_slice(), expected.as_slice());
    }

    #[test]
    fn zero_len_returns_error() {
        let device = device();
        let arrow = Arc::new(ArrowBuffer::from_vec(vec![] as Vec<u8>));
        let result = MetalBuffer::from_arrow(&device, arrow);
        assert!(
            result.is_err(),
            "zero-length buffer should fail with AllocationFailed"
        );
    }

    // ── Zero-copy path ───────────────────────────────────────────────────────

    /// Allocate a page-aligned buffer using mmap and verify that
    /// `from_arrow` takes the zero-copy path (no crash, correct data).
    #[test]
    fn zero_copy_path_page_aligned_buffer() {
        use crate::page_size;
        let page = page_size();

        // Use mmap to allocate page-aligned memory.
        // SAFETY: standard mmap call with well-known flags.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                page,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED, "mmap must succeed");

        // Write a recognizable pattern.
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, page) };
        for (i, b) in slice.iter_mut().enumerate() {
            *b = (i % 256) as u8;
        }

        // Wrap in an ArrowBuffer without copying.
        // SAFETY: ptr is valid, page-aligned, and will be munmap'd only after
        // the ArrowBuffer (and therefore MetalBuffer) is dropped.
        let arrow = unsafe {
            Arc::new(ArrowBuffer::from_custom_allocation(
                NonNull::new(ptr as *mut u8).expect("ptr is non-null"),
                page,
                Arc::new(MmapDealloc { ptr, len: page }),
            ))
        };

        let device = device();
        let metal = MetalBuffer::from_arrow(&device, arrow.clone()).expect("zero-copy allocation");
        assert_eq!(metal.len(), page);
        // Verify round-trip byte-for-byte.
        let got = metal.as_slice();
        let expected: Vec<u8> = (0..page).map(|i| (i % 256) as u8).collect();
        assert_eq!(got, expected.as_slice());
    }

    /// Drop implementation that `munmap`s the allocation when done.
    struct MmapDealloc {
        ptr: *mut libc::c_void,
        len: usize,
    }
    // SAFETY: the pointer is valid until drop and not aliased after.
    unsafe impl Send for MmapDealloc {}
    unsafe impl Sync for MmapDealloc {}
    impl Drop for MmapDealloc {
        fn drop(&mut self) {
            // SAFETY: ptr/len were obtained from mmap.
            unsafe { libc::munmap(self.ptr, self.len) };
        }
    }

    // ── Proptest round-trip tests ────────────────────────────────────────

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn arrow_to_metal_round_trip(bytes in proptest::collection::vec(any::<u8>(), 1..4096)) {
            let device = device();
            let arrow = Arc::new(ArrowBuffer::from_vec(bytes.clone()));
            let metal = MetalBuffer::from_arrow(&device, arrow).expect("allocation must succeed");
            prop_assert_eq!(metal.as_slice(), bytes.as_slice());
        }
    }
}
