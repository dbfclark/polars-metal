//! Arrow ↔ MTLBuffer bridge.
//!
//! Two regimes (per the M0 spec):
//! - Zero-copy when an Arrow `Buffer` is page-aligned (both pointer and length).
//! - Copy fallback otherwise.

#![cfg_attr(not(test), forbid(clippy::unwrap_used))]
