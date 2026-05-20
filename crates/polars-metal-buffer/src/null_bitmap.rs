// crates/polars-metal-buffer/src/null_bitmap.rs
//
// Arrow validity bitmap helpers. The bitmap is bit-packed, little-endian
// within each byte: bit i of row n lives at byte (n / 8), bit (n % 8).
// A `1` means "valid"; a `0` means "null".

/// Returns the byte length needed to store `row_count` validity bits.
pub fn validity_bytes(row_count: usize) -> usize {
    row_count.div_ceil(8)
}

/// Reads the validity bit for `row` from `bitmap`. Out-of-range rows return
/// `false` (treat as null), which matches Arrow's convention for buffers
/// that have been padded.
pub fn get_valid(bitmap: &[u8], row: usize) -> bool {
    let byte_idx = row / 8;
    let bit_idx = row % 8;
    bitmap
        .get(byte_idx)
        .is_some_and(|&b| (b >> bit_idx) & 1 == 1)
}

/// Sets the validity bit for `row` in `bitmap` to `valid`.
///
/// Panics if `row` is out of range; callers must size the bitmap with
/// [`validity_bytes`] first.
pub fn set_valid(bitmap: &mut [u8], row: usize, valid: bool) {
    let byte_idx = row / 8;
    let bit_idx = row % 8;
    let mask = 1u8 << bit_idx;
    let byte = &mut bitmap[byte_idx];
    if valid {
        *byte |= mask;
    } else {
        *byte &= !mask;
    }
}

/// Returns the number of valid (set) bits in `bitmap[..row_count]`.
///
/// Bytes past the end of `bitmap` are treated as zero (no valid rows), matching
/// the convention used by [`get_valid`] and [`load_chunk_8`]. Callers should
/// still size `bitmap` with [`validity_bytes`] for correctness when the row
/// data extends to `row_count`; the saturating behaviour here is purely a
/// safety net against panics on truncated slices.
pub fn count_valid(bitmap: &[u8], row_count: usize) -> usize {
    let full_bytes = row_count / 8;
    let trailing_bits = row_count % 8;
    let end = full_bytes.min(bitmap.len());
    let mut sum: usize = bitmap[..end].iter().map(|b| b.count_ones() as usize).sum();
    if trailing_bits > 0 {
        let mask: u8 = (1u8 << trailing_bits) - 1;
        let trailing = bitmap.get(full_bytes).copied().unwrap_or(0);
        sum += (trailing & mask).count_ones() as usize;
    }
    sum
}

/// Loads 8 rows of validity, starting at `row_start`, as a single `u8`.
/// Bit `i` of the returned byte corresponds to row `row_start + i`.
///
/// `row_start` must be a multiple of 8 — call sites doing arbitrary offsets
/// should pad the bitmap first.
///
/// Rows past the end of `bitmap` are treated as invalid (bit = 0).
pub fn load_chunk_8(bitmap: &[u8], row_start: usize) -> u8 {
    debug_assert_eq!(row_start % 8, 0, "row_start must be a multiple of 8");
    let byte_idx = row_start / 8;
    bitmap.get(byte_idx).copied().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn naive_count_valid(bitmap: &[u8], row_count: usize) -> usize {
        (0..row_count).filter(|&r| get_valid(bitmap, r)).count()
    }

    proptest! {
        #[test]
        fn validity_bytes_matches_div_ceil(row_count in 0usize..10_000) {
            prop_assert_eq!(validity_bytes(row_count), (row_count + 7) / 8);
        }

        #[test]
        fn set_then_get_round_trip(
            row_count in 1usize..1024,
            valid in any::<Vec<bool>>(),
        ) {
            let row_count = row_count.min(valid.len().max(1));
            let mut bm = vec![0u8; validity_bytes(row_count)];
            for (r, &v) in valid.iter().take(row_count).enumerate() {
                set_valid(&mut bm, r, v);
            }
            for (r, &v) in valid.iter().take(row_count).enumerate() {
                prop_assert_eq!(get_valid(&bm, r), v);
            }
        }

        #[test]
        fn count_valid_matches_naive(
            row_count in 0usize..1024,
            seed in any::<u64>(),
        ) {
            let mut bm = vec![0u8; validity_bytes(row_count)];
            for r in 0..row_count {
                let valid = ((seed.rotate_left(r as u32 & 63)) & 1) == 1;
                set_valid(&mut bm, r, valid);
            }
            prop_assert_eq!(count_valid(&bm, row_count), naive_count_valid(&bm, row_count));
        }

        #[test]
        fn load_chunk_8_matches_individual_gets(
            bytes in proptest::collection::vec(any::<u8>(), 0..128),
            chunk in 0usize..128,
        ) {
            let row_start = chunk * 8;
            let chunk_byte = load_chunk_8(&bytes, row_start);
            for i in 0..8 {
                let from_individual = get_valid(&bytes, row_start + i);
                let from_chunk = (chunk_byte >> i) & 1 == 1;
                prop_assert_eq!(from_individual, from_chunk);
            }
        }
    }
}
