//! Arrow ↔ MTLBuffer bridge.
//!
//! Two regimes (per the M0 spec):
//! - Zero-copy when an Arrow `Buffer` is page-aligned (both pointer and length).
//! - Copy fallback otherwise.

#![cfg_attr(not(test), forbid(clippy::unwrap_used))]

mod alignment;
mod device;
mod error;
mod null_bitmap;

pub use alignment::{is_aligned, is_page_aligned, page_size};
pub use device::MetalDevice;
pub use error::BufferError;
pub use null_bitmap::{validity_bytes, get_valid, set_valid, count_valid, load_chunk_8};
