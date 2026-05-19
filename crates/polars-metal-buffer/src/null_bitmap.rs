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
pub fn count_valid(bitmap: &[u8], row_count: usize) -> usize {
    let full_bytes = row_count / 8;
    let trailing_bits = row_count % 8;
    let mut sum: usize = bitmap[..full_bytes].iter().map(|b| b.count_ones() as usize).sum();
    if trailing_bits > 0 {
        let mask: u8 = (1u8 << trailing_bits) - 1;
        sum += (bitmap[full_bytes] & mask).count_ones() as usize;
    }
    sum
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
    }
}
