//! Runtime loader for the metallib produced by `build.rs`.
//!
//! `build.rs` compiles every `*.metal` source under `<workspace>/shaders/`
//! into a single metallib and exports its absolute path as
//! `POLARS_METAL_METALLIB`. We `include_bytes!` that file so the metallib
//! travels with the compiled binary; consumers do not need to know the
//! filesystem path.
//!
//! Loading flow:
//! 1. The first call to [`ShaderLibrary::load`] in this process materialises
//!    the embedded bytes to a per-PID temp file via a `OnceLock`, so the
//!    write+load sequence is serialised even when several threads race to
//!    construct libraries concurrently (e.g. parallel `cargo test`). Metal's
//!    public Rust bindings on `objc2-metal` 0.2 expose
//!    `newLibraryWithURL:error:` but not the `dispatch_data_t`-based
//!    `newLibraryWithData:error:`, so a brief on-disk hop is required.
//! 2. We construct an `NSURL` for that path and call
//!    `device.raw().newLibraryWithURL_error_(url)`. `newLibraryWithURL:` reads
//!    the metallib eagerly, so the file is no longer needed once the call
//!    returns; the first successful loader removes it. Subsequent loaders see
//!    the cached path from the `OnceLock` and skip both the write and the
//!    delete.
//! 3. Compute pipeline states are looked up lazily by entry point and
//!    cached in a `Mutex<HashMap<…>>` so repeated `pipeline()` calls are
//!    cheap.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{MTLComputePipelineState, MTLDevice as _, MTLFunction as _, MTLLibrary as _};
use polars_metal_buffer::MetalDevice;

/// Bytes of the workspace's combined metallib, embedded at compile time.
const METALLIB_BYTES: &[u8] = include_bytes!(env!("POLARS_METAL_METALLIB"));

/// Loaded metallib plus a per-entry-point pipeline state cache.
///
/// Safe to share across threads: pipeline lookups take a short mutex on the
/// cache. Pipeline state objects themselves are thread-safe Objective-C
/// objects and can be encoded into command buffers from any thread.
pub struct ShaderLibrary {
    library: Retained<ProtocolObject<dyn objc2_metal::MTLLibrary>>,
    psos: Mutex<HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
}

// The wrapped `Retained<…>` values are thread-safe Objective-C objects and
// the cache is guarded by a Mutex. ShaderLibrary therefore satisfies
// `Send + Sync` even though `Retained<ProtocolObject<dyn …>>` is not
// auto-`Send` (the underlying objects are thread-safe at the Objective-C
// level for MTLLibrary and MTLComputePipelineState).
// SAFETY: `MTLLibrary` and `MTLComputePipelineState` are documented by
// Apple as thread-safe (they may be used concurrently from multiple
// threads). The cache itself is `Mutex`-guarded.
unsafe impl Send for ShaderLibrary {}
// SAFETY: see Send impl above.
unsafe impl Sync for ShaderLibrary {}

/// Errors raised while loading the metallib or building a pipeline state.
#[derive(Debug, thiserror::Error)]
pub enum ShaderError {
    #[error("failed to materialise embedded metallib to disk: {0}")]
    TempFile(String),
    #[error("failed to load metallib from URL: {0}")]
    LibraryLoad(String),
    #[error("unknown kernel entry point: {0}")]
    UnknownEntryPoint(String),
    #[error("compute pipeline state creation failed for '{entry_point}': {message}")]
    PipelineStateFailed {
        entry_point: String,
        message: String,
    },
}

impl ShaderLibrary {
    /// Materialise the embedded metallib bytes and load them on `device`.
    ///
    /// In normal usage, callers should prefer [`shared_library`] which
    /// memoises the result for the process. `load` is exposed primarily so
    /// tests can construct a library against an arbitrary device.
    pub fn load(device: &MetalDevice) -> Result<Self, ShaderError> {
        let metallib_path = materialize_metallib_once()?;
        let library = load_library_at_path(device, metallib_path)?;
        // The metallib has been read into `MTLLibrary` in-memory and the
        // on-disk file is no longer needed. Removing it eagerly keeps
        // `$TMPDIR` tidy across dev cycles. The remove is best-effort: a
        // sibling thread that won the `OnceLock` race may have already
        // unlinked it (or a previous `load` call did so), which is fine.
        // Any other error here is non-fatal — the library load already
        // succeeded — so we swallow it.
        let _ = std::fs::remove_file(metallib_path);
        Ok(Self {
            library,
            psos: Mutex::new(HashMap::new()),
        })
    }

    /// Look up (or build) the compute pipeline state for `entry_point`.
    ///
    /// Each entry point is built at most once per `ShaderLibrary`. Repeat
    /// calls clone a `Retained` pointing at the same Objective-C object.
    pub fn pipeline(
        &self,
        entry_point: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, ShaderError> {
        // Fast path: already cached. Hold the lock only for the lookup.
        {
            let cache = self.psos.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(pso) = cache.get(entry_point) {
                return Ok(pso.clone());
            }
        }

        let pso = build_pipeline_state(&self.library, entry_point)?;

        let mut cache = self.psos.lock().unwrap_or_else(|e| e.into_inner());
        // A second thread may have inserted the same key between our drop
        // and re-acquire of the lock; in that case we discard the freshly
        // built PSO (Metal happily caches duplicates internally) and use
        // the already-cached one to preserve pointer identity for the
        // caller.
        let entry = cache
            .entry(entry_point.to_owned())
            .or_insert_with(|| pso.clone());
        Ok(entry.clone())
    }
}

/// Process-wide, lazily-initialised shader library.
///
/// The first call loads the metallib on `device`; subsequent calls return
/// the cached library. This is the entry point most consumers should use.
pub fn shared_library(device: &MetalDevice) -> Result<&'static ShaderLibrary, ShaderError> {
    // We cache `Result` rather than `ShaderLibrary` so that a transient
    // failure on the first attempt is still surfaced (and not silently
    // retried). Returning `&Result<…>::Err` would be awkward, so we store
    // the cloneable error message.
    static INSTANCE: OnceLock<Result<ShaderLibrary, String>> = OnceLock::new();
    let result = INSTANCE.get_or_init(|| ShaderLibrary::load(device).map_err(|e| e.to_string()));
    result
        .as_ref()
        .map_err(|msg| ShaderError::LibraryLoad(msg.clone()))
}

/// Materialise the embedded metallib bytes to a per-PID temp path exactly
/// once per process.
///
/// A `OnceLock` serialises the write+load sequence so concurrent callers
/// (e.g. parallel `cargo test` workers in the same binary) cannot observe a
/// half-written file. The first thread performs the write; later threads
/// see the cached path (or the cached error) and skip the I/O entirely.
///
/// The outcome is stored as `Result<PathBuf, String>` rather than
/// `Result<PathBuf, ShaderError>` because `ShaderError` is not `Clone` and
/// callers need a fresh `Err` on every retrieval. The string is re-wrapped
/// in `ShaderError::TempFile` here so the public surface is unchanged.
fn materialize_metallib_once() -> Result<&'static Path, ShaderError> {
    static TEMP_METALLIB: OnceLock<Result<PathBuf, String>> = OnceLock::new();

    let outcome = TEMP_METALLIB.get_or_init(|| {
        let mut path = std::env::temp_dir();
        path.push(format!("polars_metal_{}.metallib", std::process::id()));

        let mut file =
            std::fs::File::create(&path).map_err(|e| format!("create({}): {e}", path.display()))?;
        file.write_all(METALLIB_BYTES)
            .map_err(|e| format!("write({}): {e}", path.display()))?;
        file.sync_all()
            .map_err(|e| format!("sync({}): {e}", path.display()))?;
        drop(file);
        Ok(path)
    });

    match outcome {
        Ok(path) => Ok(path.as_path()),
        Err(msg) => Err(ShaderError::TempFile(msg.clone())),
    }
}

/// Load a metallib from a filesystem path via `MTLDevice newLibraryWithURL`.
fn load_library_at_path(
    device: &MetalDevice,
    path: &std::path::Path,
) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLLibrary>>, ShaderError> {
    let path_str = path.to_string_lossy();
    let ns_path = NSString::from_str(&path_str);
    // SAFETY: `fileURLWithPath:` returns a non-null retained NSURL for any
    // non-null NSString input; the resulting URL is a value type that does
    // not need filesystem access at construction time.
    let url = unsafe { NSURL::fileURLWithPath(&ns_path) };

    // SAFETY: `url` is a non-null NSURL referencing a file we just wrote.
    // `newLibraryWithURL:error:` is unsafe in `objc2-metal` only because
    // its underlying ObjC signature traffics in `NSError**`; the Rust
    // wrapper translates that into a `Result`, so all preconditions
    // (well-formed NSURL pointing at a Mach-O metallib on disk) hold.
    let library = unsafe { device.raw().newLibraryWithURL_error(&url) }
        .map_err(|err| ShaderError::LibraryLoad(err.localizedDescription().to_string()))?;
    Ok(library)
}

/// Resolve `entry_point` against `library` and build a compute pipeline state.
fn build_pipeline_state(
    library: &ProtocolObject<dyn objc2_metal::MTLLibrary>,
    entry_point: &str,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, ShaderError> {
    let ns_name = NSString::from_str(entry_point);
    let function = library
        .newFunctionWithName(&ns_name)
        .ok_or_else(|| ShaderError::UnknownEntryPoint(entry_point.to_owned()))?;

    // newComputePipelineStateWithFunction:error: needs the device that
    // produced the function. We retrieve it from the function itself
    // rather than threading a `&MetalDevice` into every PSO build, so the
    // cache can satisfy lookups without re-borrowing the device handle.
    let device = function.device();
    let pso = device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|err| ShaderError::PipelineStateFailed {
            entry_point: entry_point.to_owned(),
            message: err.localizedDescription().to_string(),
        })?;
    Ok(pso)
}
