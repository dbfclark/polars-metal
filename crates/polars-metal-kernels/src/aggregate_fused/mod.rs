//! Fused multi-aggregation kernel: MSL template engine + library cache.
//!
//! See spec § B (Fused multi-aggregation kernel template).
//!
//! Lifecycle per query:
//!   1. `AggSignature::from_specs(aggs, col_dtypes)` → cache key
//!   2. `emit_msl_for(signature)` → MSL source (if not in cache)
//!   3. compile via `MTLDevice::newLibraryWithSource` → cached pipeline
//!   4. dispatch with bound buffers per signature's column order
//!
//! Phase 3 lands these in tasks 11–18. Tasks 11–14 ship the signature
//! module, the MSL emitter (Simple + Expression aggs), and the lazy
//! library cache; Tasks 15–18 wire those into `dispatch_groupby` and
//! pre-compile common signatures at module import.

pub mod cache;
pub mod emitter;
pub mod signature;
