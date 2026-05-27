//! Arrow ↔ MTLBuffer bridge.
//!
//! Two regimes (per the M0 spec):
//! - Zero-copy when an Arrow `Buffer` is page-aligned (both pointer and length).
//! - Copy fallback otherwise.

#![cfg_attr(not(test), forbid(clippy::unwrap_used))]

mod alignment;
mod bridge;
mod device;
pub mod dict;
mod error;
mod null_bitmap;

pub use alignment::{is_aligned, is_page_aligned, page_size};
pub use bridge::MetalBuffer;
pub use device::MetalDevice;
pub use error::BufferError;
pub use null_bitmap::{count_valid, get_valid, load_chunk_8, set_valid, validity_bytes};
