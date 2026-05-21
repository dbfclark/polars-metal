// crates/polars-metal-kernels/tests/test_shader_lib.rs
//
// End-to-end check that the metallib produced by `build.rs` can be loaded at
// runtime and that the hello-world kernel's entry point resolves to a usable
// compute pipeline state.
#![allow(clippy::expect_used)]

use objc2_metal::MTLComputePipelineState as _;
use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::shader_lib::ShaderLibrary;

#[test]
fn loads_metallib_and_finds_hello_kernel() {
    let device = MetalDevice::system_default().expect("Metal-capable hardware");
    let lib = ShaderLibrary::load(&device).expect("metallib must load");
    let pso = lib
        .pipeline("hello_write_constant")
        .expect("entry point must exist");
    // objc2-metal preserves the Objective-C selector name verbatim.
    assert!(pso.maxTotalThreadsPerThreadgroup() > 0);
}

#[test]
fn pipeline_lookup_caches_pso() {
    let device = MetalDevice::system_default().expect("Metal-capable hardware");
    let lib = ShaderLibrary::load(&device).expect("metallib must load");

    let first = lib
        .pipeline("hello_write_constant")
        .expect("entry point must exist");
    let second = lib
        .pipeline("hello_write_constant")
        .expect("entry point must exist");

    // Two lookups of the same entry point should yield the same underlying
    // Objective-C pipeline state object (the cache returns a clone of the
    // same Retained pointer).
    assert!(
        std::ptr::eq(
            &*first as *const _ as *const (),
            &*second as *const _ as *const ()
        ),
        "pipeline state object should be cached and shared between lookups"
    );
}

#[test]
fn unknown_entry_point_is_error() {
    let device = MetalDevice::system_default().expect("Metal-capable hardware");
    let lib = ShaderLibrary::load(&device).expect("metallib must load");
    let err = lib
        .pipeline("definitely_not_a_kernel")
        .expect_err("missing entry point must surface an error");
    let msg = format!("{err}");
    assert!(
        msg.contains("definitely_not_a_kernel"),
        "error message should name the missing entry point, got: {msg}"
    );
}

/// Regression test for the TOCTOU race in the per-PID metallib temp path:
/// many threads in the same process race to load the library. With the
/// pre-fix per-call `File::create`+write+load sequence, thread B could
/// truncate the file while thread A's Metal load read it, surfacing as a
/// flaky "load library failed" error. The OnceLock-based materialisation
/// must serialise these callers.
#[test]
fn shared_library_is_safe_under_concurrent_load() {
    use std::thread;

    let handles: Vec<_> = (0..8)
        .map(|_| {
            thread::spawn(|| {
                let device = MetalDevice::system_default().expect("Metal-capable hardware");
                let lib = polars_metal_kernels::shader_lib::shared_library(&device)
                    .expect("metallib must load");
                let _ = lib
                    .pipeline("hello_write_constant")
                    .expect("entry point must exist");
            })
        })
        .collect();

    for h in handles {
        h.join().expect("worker thread panicked");
    }
}
