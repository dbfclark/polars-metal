//! `FusedLibraryCache`: maps `AggSignature` → compiled
//! `MTLComputePipelineState`.
//!
//! Phase 3 requires *runtime* MSL source compilation (one library per
//! signature shape). This module owns the lazy compile-and-cache pipeline:
//!
//! 1. [`AggSignature::hash64`] yields a stable 64-bit key. Two query plans
//!    with isomorphic agg shapes (same per-slot dtypes, same op set, same
//!    expression structure — aliases & column names ignored) hash equal.
//! 2. On a cache miss we emit the MSL via [`emit_msl`], compile via
//!    `MTLDevice::newLibraryWithSource:options:error:`, resolve the
//!    `aggregate_fused` entry point, and build the
//!    `MTLComputePipelineState`. The PSO is stored and reused for every
//!    subsequent matching signature.
//! 3. [`warmup`] drives Task 18's module-import pre-compile: best-effort,
//!    swallow errors so a partial warmup never breaks engine start-up.
//!
//! The locking pattern mirrors `ShaderLibrary::pipeline` in
//! `crates/polars-metal-kernels/src/shader_lib.rs`: take a short read
//! lock first, drop it, compile under no lock, then re-acquire and
//! `entry().or_insert_with(...)` to avoid double-inserting under a race.
//! A second thread that inserts the same key during our compile wins —
//! we discard the fresh PSO and clone the cached one to preserve pointer
//! identity for callers.
//!
//! Thread safety: `Retained<ProtocolObject<dyn MTLComputePipelineState>>`
//! is not auto-`Send`, but Apple documents `MTLLibrary` and
//! `MTLComputePipelineState` as thread-safe (encodable from any thread).
//! The HashMap itself is `Mutex`-guarded. See the `unsafe impl Send` /
//! `Sync` block at the bottom of the module.

use std::collections::HashMap;
use std::sync::Mutex;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState, MTLDevice as _, MTLFunction as _,
    MTLLibrary as _,
};
use polars_metal_buffer::MetalDevice;
use thiserror::Error;

use super::emitter::emit_msl;
use super::signature::{AggSignature, AggSpec};

/// MSL entry point emitted by [`emit_msl`].
const FUSED_ENTRY_POINT: &str = "aggregate_fused";

/// Errors raised while compiling and caching a fused-agg kernel.
#[derive(Debug, Error)]
pub enum FusedCacheError {
    /// `newLibraryWithSource:options:error:` returned an `NSError`. The
    /// MSL source is included verbatim so debugging does not require
    /// regenerating it. Field is named `msl_source` (not `source`) to
    /// avoid `thiserror`'s implicit `#[source]` treatment of fields
    /// literally named `source`.
    #[error("MSL compile failed: {message}\n--- source ---\n{msl_source}")]
    CompileFailed { message: String, msl_source: String },

    /// `library.newFunctionWithName("aggregate_fused")` returned nil. This
    /// is a contract violation by the emitter — every emitted source must
    /// contain the `aggregate_fused` entry point.
    #[error("entry point 'aggregate_fused' not found in compiled library")]
    EntryPointMissing,

    /// `device.newComputePipelineStateWithFunction:error:` returned an
    /// `NSError`. The most common cause on Apple Silicon is an atomic op
    /// the source uses that the runtime does not support (atomic_fetch_add
    /// on `atomic_float`, for example) — `newLibraryWithSource` accepts
    /// such source but PSO creation rejects it.
    #[error("pipeline state creation failed: {0}")]
    PipelineStateFailed(String),
}

/// Lazy MSL compile-and-cache for fused-agg kernels.
///
/// One instance per engine; shared across queries on the same device.
/// Compilation is performed under no lock so concurrent first-touch
/// queries do not serialize on the (slow) Metal compiler.
pub struct FusedLibraryCache {
    device: MetalDevice,
    by_hash: Mutex<HashMap<u64, Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
}

impl FusedLibraryCache {
    /// Build an empty cache bound to `device`. The device is `Clone`
    /// (cheap retained-pointer copy), so the cache owns its own handle.
    pub fn new(device: MetalDevice) -> Self {
        Self {
            device,
            by_hash: Mutex::new(HashMap::new()),
        }
    }

    /// Look up or compile the fused-agg pipeline for `sig`. Returns a
    /// retained PSO that can be encoded into command buffers.
    ///
    /// `specs` is forwarded to [`emit_msl`] on a cache miss; the signature
    /// alone is insufficient because emitted MSL embeds the original
    /// column-name strings for debug output and alias mapping (the
    /// canonical part — the actual op/dtype/expression *structure* — is
    /// what makes the signature equal across isomorphic specs, and is the
    /// only part used for the cache key).
    pub fn get_or_compile(
        &self,
        sig: &AggSignature,
        specs: &[AggSpec],
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, FusedCacheError> {
        let key = sig.hash64();

        // Fast path: cached. Hold the lock only for the lookup.
        {
            let cache = self.by_hash.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(pso) = cache.get(&key) {
                return Ok(pso.clone());
            }
        }

        // Slow path: compile under no lock so concurrent first-touch
        // queries with distinct signatures do not serialize on the Metal
        // compiler.
        let src = emit_msl(sig, specs);
        let pso = self.compile_pipeline(&src)?;

        // Re-acquire the lock. A second thread may have inserted the same
        // key between our drop and re-acquire (e.g. a parallel test or
        // two queries with the same shape racing). In that case we
        // discard the fresh PSO and use the already-cached one to
        // preserve pointer identity for the caller — Metal happily
        // caches duplicate PSOs internally, but our contract is "one PSO
        // per signature for the lifetime of the cache."
        let mut cache = self.by_hash.lock().unwrap_or_else(|e| e.into_inner());
        let entry = cache.entry(key).or_insert_with(|| pso.clone());
        Ok(entry.clone())
    }

    /// Pre-compile a list of signatures. Best-effort: errors are silently
    /// swallowed so a partial warmup never blocks engine start-up. Task 18
    /// drives this from module import with the canonical Q1-shape
    /// signatures.
    pub fn warmup(&self, signatures: &[(AggSignature, Vec<AggSpec>)]) {
        for (sig, specs) in signatures {
            // Discard the result — warmup is advisory.
            let _ = self.get_or_compile(sig, specs);
        }
    }

    /// Compile MSL source into a `MTLComputePipelineState` for the
    /// `aggregate_fused` entry point. The pattern mirrors
    /// `shader_lib.rs::build_pipeline_state` (lines 203–225) but takes
    /// the library from an in-process source string rather than a
    /// pre-compiled metallib.
    fn compile_pipeline(
        &self,
        src: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, FusedCacheError> {
        let ns_src = NSString::from_str(src);
        let opts = MTLCompileOptions::new();

        // `newLibraryWithSource_options_error` is a safe wrapper in
        // `objc2-metal` 0.2 (it takes `&NSString` + `Option<&...>` and
        // returns a `Result`). The Task 12 test helper at
        // `tests/test_aggregate_fused_emitter.rs` calls it without an
        // `unsafe` block; mirror that exactly.
        let library = self
            .device
            .raw()
            .newLibraryWithSource_options_error(&ns_src, Some(&opts))
            .map_err(|err| FusedCacheError::CompileFailed {
                message: err.localizedDescription().to_string(),
                msl_source: src.to_string(),
            })?;

        let func_name = NSString::from_str(FUSED_ENTRY_POINT);
        let function = library
            .newFunctionWithName(&func_name)
            .ok_or(FusedCacheError::EntryPointMissing)?;

        // Resolve the device from the function (same pattern as
        // `build_pipeline_state` in shader_lib.rs). This sidesteps
        // re-borrowing `self.device.raw()` and keeps the call symmetric
        // with the metallib loader.
        let device = function.device();
        device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|err| FusedCacheError::PipelineStateFailed(err.localizedDescription().to_string()))
    }
}

// The wrapped `Retained<…>` values are thread-safe Objective-C objects
// and the HashMap is guarded by a `Mutex`. `FusedLibraryCache` therefore
// satisfies `Send + Sync` even though `Retained<ProtocolObject<dyn …>>`
// is not auto-`Send`.
// SAFETY: `MTLComputePipelineState` (and the `MetalDevice` it was built
// against) are documented by Apple as thread-safe — they may be encoded
// into command buffers from multiple threads concurrently. The HashMap
// itself is `Mutex`-guarded. This mirrors the `Send + Sync` impl on
// `ShaderLibrary` in `shader_lib.rs`.
unsafe impl Send for FusedLibraryCache {}
// SAFETY: see Send impl above.
unsafe impl Sync for FusedLibraryCache {}
