// crates/polars-metal-buffer/src/alignment.rs

/// Returns the system page size in bytes.
///
/// On Apple Silicon this is typically 16384 (16 KiB). We don't cache because
/// it's a cheap libc call and avoiding statics keeps the crate test-friendly.
pub fn page_size() -> usize {
    // SAFETY: `sysconf` with `_SC_PAGESIZE` is safe to call; returns a positive
    // long on supported systems.
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    assert!(v > 0, "page_size sysconf returned non-positive value");
    v as usize
}

/// Returns true when `ptr` is aligned to `align` bytes. `align` must be a
/// power of two.
pub fn is_aligned(ptr: usize, align: usize) -> bool {
    debug_assert!(align.is_power_of_two(), "align must be a power of two");
    ptr & (align - 1) == 0
}

/// Returns true when `ptr` is page-aligned, regardless of length.
///
/// `MTLDevice.makeBuffer(bytesNoCopy:length:...)` requires only the *pointer*
/// to be page-aligned — the length may be any value (Metal maps whole pages
/// but reads only the requested length, and a page-aligned allocation's
/// trailing partial page is committed). Verified empirically against the live
/// Metal driver. This is the relaxed gate for [`crate::MetalBuffer::from_borrowed_f32`];
/// [`is_page_aligned`] keeps the stricter both-aligned contract used by
/// [`crate::MetalBuffer::from_arrow`].
pub fn is_ptr_page_aligned(ptr: usize) -> bool {
    is_aligned(ptr, page_size())
}

/// Returns true when both the pointer and length are page-aligned, which is
/// the requirement for `MTLDevice.makeBuffer(bytesNoCopy:length:...)`.
pub fn is_page_aligned(ptr: usize, len: usize) -> bool {
    let page = page_size();
    is_aligned(ptr, page) && is_aligned(len, page)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn page_size_is_power_of_two() {
        assert!(page_size().is_power_of_two());
    }

    proptest! {
        #[test]
        fn aligned_to_one_always_true(ptr: usize) {
            prop_assert!(is_aligned(ptr, 1));
        }

        #[test]
        fn aligned_to_power_of_two(shift in 0u32..16, base: usize) {
            let align = 1usize << shift;
            let ptr = base.wrapping_mul(align);
            prop_assert!(is_aligned(ptr, align));
        }

        #[test]
        fn page_aligned_round_trip(pages in 0usize..16) {
            let page = page_size();
            let len = pages * page;
            prop_assert!(is_aligned(len, page));
        }
    }
}
