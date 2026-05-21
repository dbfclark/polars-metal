// crates/polars-metal-kernels/build.rs
//
// Compile every `*.metal` file under `<workspace>/shaders/` into a single
// metallib, exposed to the crate at compile time via the
// `POLARS_METAL_METALLIB` env var. The crate `include_bytes!`es that path
// into the binary, so no runtime filesystem path needs to be provided.
//
// Build pipeline: `xcrun metal -c` produces a `.air` per source file, then
// `xcrun metallib` links the AIR files into a single `.metallib`.
//
// File-naming convention: any `.metal` file whose stem starts with `_` is a
// header-only file — it is `#include`d by other kernels and is NOT compiled
// into the metallib on its own. The `-I <shaders_dir>` flag passed to
// `xcrun metal` makes those includes resolve.
//
// Build scripts use panics as their failure mechanism (there is no recovery
// path; aborting the build with a diagnostic is the desired behaviour).
// Workspace-level lints would otherwise forbid `unwrap`/`expect`/`panic`.
#![allow(clippy::expect_used, clippy::panic)]

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let shaders_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../shaders");
    let out_dir = PathBuf::from(
        std::env::var("OUT_DIR").expect("cargo always sets OUT_DIR for build scripts"),
    );
    let metallib_path = out_dir.join("polars_metal.metallib");

    // Rebuild whenever the shaders directory changes (any file added/removed
    // /edited) or build.rs itself changes.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", shaders_dir.display());

    let read_dir = std::fs::read_dir(&shaders_dir)
        .unwrap_or_else(|e| panic!("shaders dir {} unreadable: {e}", shaders_dir.display()));

    // Collect .metal sources deterministically so the resulting metallib is
    // reproducible across builds. Files whose stem starts with `_` are
    // header-only (included by other kernels via #include) and must not be
    // fed to `xcrun metal -c` directly.
    let mut metal_sources: Vec<PathBuf> = read_dir
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().and_then(|s| s.to_str()) != Some("metal") {
                return None;
            }
            let stem = path.file_stem()?.to_string_lossy().to_string();
            if stem.starts_with('_') {
                // Header-only file; re-run on edits but do not compile.
                println!("cargo:rerun-if-changed={}", path.display());
                return None;
            }
            Some(path)
        })
        .collect();
    metal_sources.sort();

    assert!(
        !metal_sources.is_empty(),
        "no compilable .metal sources found under {} (header-only files \
         starting with `_` do not count)",
        shaders_dir.display()
    );

    let mut air_files: Vec<PathBuf> = Vec::with_capacity(metal_sources.len());
    for source in &metal_sources {
        // Explicitly re-run when an individual source changes too — covers
        // the case where read_dir watching might miss in-place edits.
        println!("cargo:rerun-if-changed={}", source.display());

        let stem = source
            .file_stem()
            .unwrap_or_else(|| panic!("shader source {} has no stem", source.display()))
            .to_string_lossy()
            .to_string();
        let air_path = out_dir.join(format!("{stem}.air"));

        let status = Command::new("xcrun")
            .args(["metal", "-c", "-frecord-sources", "-I"])
            .arg(&shaders_dir)
            .arg("-o")
            .arg(&air_path)
            .arg(source)
            .status()
            .unwrap_or_else(|e| panic!("failed to invoke `xcrun metal`: {e}"));
        assert!(
            status.success(),
            "`xcrun metal -c` failed for {}",
            source.display()
        );
        air_files.push(air_path);
    }

    let mut link_cmd = Command::new("xcrun");
    link_cmd.args(["metallib", "-o"]).arg(&metallib_path);
    for air in &air_files {
        link_cmd.arg(air);
    }
    let status = link_cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to invoke `xcrun metallib`: {e}"));
    assert!(status.success(), "`xcrun metallib` failed to link metallib");

    println!(
        "cargo:rustc-env=POLARS_METAL_METALLIB={}",
        metallib_path.display()
    );
}
