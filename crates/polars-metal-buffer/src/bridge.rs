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
use objc2_metal::{MTLBuffer, MTLDevice as _, MTLResource as _, MTLResourceOptions};

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
    /// Keeps a parent `MetalBuffer` alive when this buffer is a view into a
    /// larger arena-allocated buffer (see [`MetalBuffer::view_into`]). `None`
    /// when this buffer owns its bytes outright.
    _view_parent: Option<Arc<MetalBuffer>>,
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

        // Capture an Arc<ArrowBuffer> in the deallocator block. The Arc is owned
        // by the block; when Metal eventually releases the block (after the
        // MTLBuffer is freed), the Arc drops, dropping the ArrowBuffer if no
        // other references remain.
        //
        // The block body itself is a no-op. The block is `Fn` (RcBlock requires
        // it), so we cannot move-consume the captured Arc inside the body; the
        // release happens via block lifecycle, not via explicit drop here.
        let owner_for_dealloc = arrow.clone();
        // The deallocator block signature is `(NonNull<c_void>, NSUInteger) -> ()`.
        let deallocator: RcBlock<dyn Fn(NonNull<std::ffi::c_void>, usize)> =
            RcBlock::new(move |_ptr, _len| {
                // Keep `owner_for_dealloc` alive by referencing it here. The
                // compiler would otherwise be free to drop captures it sees as
                // unused.
                let _keep_alive = &owner_for_dealloc;
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
            _view_parent: None,
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
            _view_parent: None,
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

    /// Construct a `MetalBuffer` from an already-allocated MTLBuffer that
    /// Metal owns outright (no Arrow backing).
    ///
    /// Used by [`MetalDevice::new_buffer_zeroed`] and any future allocator
    /// that hands out fresh GPU-resident buffers. Crate-private so external
    /// code goes through the typed constructors above.
    pub(crate) fn from_metal_owned(inner: Retained<ProtocolObject<dyn MTLBuffer>>) -> Self {
        Self {
            inner,
            _owner: None,
            _view_parent: None,
        }
    }

    /// Raw borrow of the underlying `MTLBuffer` protocol object.
    ///
    /// Exposed to sibling crates (`polars-metal-kernels`,
    /// `polars-metal-core`) so they can bind buffers into compute encoders
    /// via `setBuffer:offset:atIndex:` without re-implementing the buffer
    /// handle. Mirrors `MetalDevice::raw()`; callers invoking unsafe
    /// `objc2-metal` APIs through this accessor must add the usual
    /// `// SAFETY:` comment.
    pub fn raw(&self) -> &Retained<ProtocolObject<dyn MTLBuffer>> {
        &self.inner
    }

    /// Construct a view onto a sub-range of an existing `MetalBuffer`'s
    /// contents.
    ///
    /// The view is realized as a *new* `MTLBuffer` produced via
    /// `newBufferWithBytesNoCopy:length:options:deallocator:` over the
    /// parent's shared-memory mapping at `parent_ptr + offset`, with length
    /// `len`. The returned buffer therefore has its own MTLBuffer identity
    /// (so dispatches see a buffer of exactly `len` bytes starting at the
    /// view's base) while still sharing storage with the parent.
    ///
    /// The deallocator block captures a clone of `parent` so the parent
    /// (and transitively any `_owner` / `_view_parent` keep-alives it
    /// holds) outlives the view. The returned `MetalBuffer` also stashes
    /// the same `Arc<MetalBuffer>` in `_view_parent` to make the lifetime
    /// dependency explicit on the Rust side.
    ///
    /// # Errors
    ///
    /// Returns [`BufferError::AllocationFailed`] if `offset + len` overflows
    /// or exceeds `parent.len()`, if `len == 0` (Metal rejects zero-byte
    /// buffers), or if Metal otherwise refuses the no-copy allocation.
    ///
    /// # Safety
    ///
    /// - `offset + len` must be `<= parent.len()`. The function bounds-checks
    ///   this before calling into Metal; the `unsafe` is for documenting that
    ///   the caller is responsible for *semantic* aliasing: the view shares
    ///   bytes with the parent and any other outstanding views on the same
    ///   parent, so concurrent CPU/GPU writes that overlap are the caller's
    ///   problem.
    /// - The parent's MTLBuffer must outlive the view; this is enforced by
    ///   the `Arc<MetalBuffer>` captured in the deallocator block and the
    ///   `_view_parent` field.
    /// - The view must not be used concurrently with a GPU command in flight
    ///   that writes to the same byte range — standard MTLBuffer rule.
    pub unsafe fn view_into(
        parent: &Arc<MetalBuffer>,
        offset: usize,
        len: usize,
    ) -> Result<MetalBuffer, BufferError> {
        // Bounds check before any unsafe work. Reject zero-length views up
        // front (Metal rejects zero-byte buffers).
        if len == 0 {
            return Err(BufferError::AllocationFailed { bytes: 0 });
        }
        let end = offset
            .checked_add(len)
            .ok_or(BufferError::AllocationFailed { bytes: len })?;
        if end > parent.len() {
            return Err(BufferError::AllocationFailed { bytes: len });
        }

        // Compute the start address inside the parent's shared-memory mapping.
        // `parent.inner.contents()` returns NonNull<c_void> for the entire
        // parent's backing store; the view sits at `+offset` into it.
        let base = parent.inner.contents().as_ptr() as *mut u8;
        // SAFETY: `offset` is `<= parent.len()` (bounds-checked above), so the
        // resulting pointer is within the parent's allocation (one past the
        // end is allowed for pointer arithmetic; we further restrict to
        // `offset + len <= parent.len()` so `len` bytes from this pointer are
        // valid).
        let view_ptr = unsafe { base.add(offset) } as *mut std::ffi::c_void;
        let view_ptr =
            NonNull::new(view_ptr).ok_or(BufferError::AllocationFailed { bytes: len })?;

        // Capture an Arc<MetalBuffer> clone in the deallocator block so the
        // parent (and its owners) survive until Metal releases the view.
        let parent_for_dealloc = parent.clone();
        let deallocator: RcBlock<dyn Fn(NonNull<std::ffi::c_void>, usize)> =
            RcBlock::new(move |_ptr, _len| {
                // Reference the captured Arc so the compiler cannot drop it
                // before block release. The actual `Arc::drop` runs when the
                // block itself is dropped by Metal.
                let _keep_alive = &parent_for_dealloc;
            });

        // The view inherits the parent's device; query it via MTLResource.
        let device = parent.inner.device();

        // SAFETY:
        // - `view_ptr` is valid for `len` bytes (bounds-checked above) and the
        //   parent's `Arc<MetalBuffer>` in `_view_parent`/deallocator keeps the
        //   memory alive.
        // - The parent was allocated `StorageModeShared`; the no-copy options
        //   here match. We do not change storage modes mid-allocation.
        // - Metal only accesses the buffer via GPU commands submitted later;
        //   no concurrent access is happening at construction time.
        let inner = unsafe {
            device.newBufferWithBytesNoCopy_length_options_deallocator(
                view_ptr,
                len,
                MTLResourceOptions::MTLResourceStorageModeShared,
                Some(&deallocator),
            )
        }
        .ok_or(BufferError::AllocationFailed { bytes: len })?;

        Ok(Self {
            inner,
            _owner: None,
            _view_parent: Some(parent.clone()),
        })
    }

    /// Construct a `MetalBuffer` from an `f32` slice.
    ///
    /// The input values are copied into a new Metal allocation via
    /// [`MetalDevice::new_buffer_from_bytes`]. This is the standard way to
    /// stage F32 host data for use as an MLX array input.
    ///
    /// Returns `BufferError::AllocationFailed` when `data` is empty (Metal
    /// rejects zero-byte allocations) or when Metal otherwise refuses the
    /// allocation.
    pub fn from_f32_slice(device: &MetalDevice, data: &[f32]) -> Result<Self, BufferError> {
        // SAFETY: Reinterpreting &[f32] as &[u8] is valid. f32 has no padding,
        // alignment of [u8] (1) is ≤ alignment of [f32] (4), and the resulting
        // byte length (data.len() * 4) is exact. This is a read-only view with
        // lifetime bounded by `data`.
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };
        device.new_buffer_from_bytes(bytes)
    }

    /// Return a raw ObjC pointer to the underlying `MTL::Buffer` object.
    ///
    /// This is the address that metal-cpp sees as `MTL::Buffer*`. The caller
    /// can pass it to C++ code that wraps it in `mlx::core::allocator::Buffer`
    /// for zero-copy MLX array construction.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid as long as `self` is alive. The caller
    /// is responsible for ensuring that any C++ use of the pointer (including
    /// MLX operations that retain it) is completed before `self` drops. When
    /// used with `mlx_array_view_metal_buffer`, the `Arc<MetalBuffer>` stored
    /// inside `MlxArrayHandle::_input_refs` enforces this invariant.
    pub fn as_mtl_buffer_raw_ptr(&self) -> *const std::ffi::c_void {
        // SAFETY: `self.inner` is a `Retained<ProtocolObject<dyn MTLBuffer>>`.
        // A reference to `ProtocolObject<dyn MTLBuffer>` IS the ObjC instance
        // pointer — the ProtocolObject wrapper is a zero-sized newtype around the
        // ObjC `id`. Taking a reference and casting it through `*const _` gives
        // the same address that metal-cpp would see as `MTL::Buffer*`.
        let proto_ref: &ProtocolObject<dyn MTLBuffer> = &self.inner;
        proto_ref as *const ProtocolObject<dyn MTLBuffer> as *const std::ffi::c_void
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

    /// View the buffer's contents as a mutable byte slice.
    ///
    /// Caller must guarantee no GPU command buffer is in-flight against
    /// this buffer; mutating bytes while the GPU is reading them is a
    /// data race. Used for scratch-arena patterns where the host re-seeds
    /// the buffer between dispatches (counts, cursors, scalar params,
    /// etc.).
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        let ptr = self.inner.contents().cast::<u8>();
        let len = self.len();
        // SAFETY:
        // - Same address validity as `as_slice`.
        // - `&mut self` (not `&self`) statically excludes concurrent
        //   reads through Rust references. Concurrent GPU access is the
        //   caller's responsibility (see doc comment).
        // - StorageModeShared backs the address; the CPU can write to it.
        unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), len) }
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

        #[test]
        fn validity_buffer_round_trip(
            row_count in 1usize..4096,
            seed in any::<u64>(),
        ) {
            use crate::null_bitmap::{set_valid, validity_bytes, get_valid};

            let mut bm = vec![0u8; validity_bytes(row_count)];
            for r in 0..row_count {
                set_valid(&mut bm, r, ((seed.rotate_left(r as u32 & 63)) & 1) == 1);
            }

            let device = device();
            let arrow = Arc::new(ArrowBuffer::from_vec(bm.clone()));
            let metal = MetalBuffer::from_arrow(&device, arrow).expect("allocation must succeed");
            let round_tripped: &[u8] = metal.as_slice();

            for r in 0..row_count {
                prop_assert_eq!(get_valid(&bm, r), get_valid(round_tripped, r));
            }
        }

        #[test]
        fn offset_buffer_i32_round_trip(values in proptest::collection::vec(0i32..=1_000_000, 1..256)) {
            let mut offsets = vec![0i32];
            let mut running = 0i32;
            for v in &values {
                running = running.saturating_add(*v);
                offsets.push(running);
            }
            let bytes: Vec<u8> = offsets.iter().flat_map(|o| o.to_le_bytes()).collect();
            let device = device();
            let arrow = Arc::new(ArrowBuffer::from_vec(bytes.clone()));
            let metal = MetalBuffer::from_arrow(&device, arrow).expect("allocation must succeed");
            prop_assert_eq!(metal.as_slice(), bytes.as_slice());
        }

        #[test]
        fn dictionary_column_round_trip(
            dict_values in proptest::collection::vec(any::<u8>(), 1..64),
            index_count in 1usize..256,
        ) {
            let dict_bytes: Vec<u8> = dict_values.clone();
            let indices: Vec<u32> = (0..index_count as u32)
                .map(|i| i % dict_values.len() as u32)
                .collect();
            let idx_bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();

            let device = device();
            let dict_metal = MetalBuffer::from_arrow(
                &device,
                Arc::new(ArrowBuffer::from_vec(dict_bytes.clone())),
            )
            .expect("allocation must succeed");
            let idx_metal = MetalBuffer::from_arrow(
                &device,
                Arc::new(ArrowBuffer::from_vec(idx_bytes.clone())),
            )
            .expect("allocation must succeed");

            prop_assert_eq!(dict_metal.as_slice(), dict_bytes.as_slice());
            prop_assert_eq!(idx_metal.as_slice(), idx_bytes.as_slice());
        }
    }
}
