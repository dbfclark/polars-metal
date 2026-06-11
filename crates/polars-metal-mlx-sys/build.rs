// crates/polars-metal-mlx-sys/build.rs
use std::path::PathBuf;

const REQUIRED_MLX_VERSION: &str = "0.25.1";

fn check_mlx_version(version_h: &std::path::Path) {
    // MLX >=0.24 derives its version from `mlx/version.h` (#define MLX_VERSION_MAJOR/MINOR/PATCH)
    // rather than a literal `set(MLX_VERSION X.Y.Z)` in CMakeLists, so parse the header.
    let contents = match std::fs::read_to_string(version_h) {
        Ok(s) => s,
        Err(_) => {
            println!("cargo:warning=could not read vendor/mlx/mlx/version.h to verify MLX version");
            return;
        }
    };
    let field = |name: &str| -> Option<u32> {
        contents.lines().find_map(|line| {
            line.trim()
                .strip_prefix(name)
                .and_then(|rest| rest.trim().parse::<u32>().ok())
        })
    };
    match (
        field("#define MLX_VERSION_MAJOR "),
        field("#define MLX_VERSION_MINOR "),
        field("#define MLX_VERSION_PATCH "),
    ) {
        (Some(major), Some(minor), Some(patch)) => {
            let found = format!("{major}.{minor}.{patch}");
            if found != REQUIRED_MLX_VERSION {
                println!(
                    "cargo:warning=vendor/mlx version={found} but polars-metal-mlx-sys pins {REQUIRED_MLX_VERSION}; bump deliberately"
                );
            }
        }
        _ => println!("cargo:warning=could not parse MLX version from vendor/mlx/mlx/version.h"),
    }
}

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let version_h = manifest_dir.join("../../vendor/mlx/mlx/version.h");
    check_mlx_version(&version_h);
    println!("cargo:rerun-if-changed={}", version_h.display());

    // vendor/mlx is at <repo-root>/vendor/mlx; this crate is at
    // <repo-root>/crates/polars-metal-mlx-sys, so go up two levels.
    let mlx_root = manifest_dir.join("../../vendor/mlx");
    let mlx_build = mlx_root.join("build");

    cxx_build::bridge("src/lib.rs")
        .file("cxx/mlx_bridge.cc")
        // Add MLX headers so `#include "mlx/array.h"` etc. resolve.
        .include(&mlx_root)
        .flag_if_supported("-std=c++17")
        .compile("polars_metal_mlx_bridge");

    // Point the linker at the MLX build output. libmlx.a is the static
    // archive produced by `cmake -DMLX_BUILD_METAL=ON` under vendor/mlx/build.
    println!("cargo:rustc-link-search=native={}", mlx_build.display());
    println!("cargo:rustc-link-lib=static=mlx");

    // Frameworks MLX requires (per vendor/mlx/CMakeLists.txt):
    //  Accelerate — CPU/BLAS backend
    //  Metal      — GPU dispatch (when MLX_BUILD_METAL=ON)
    //  Foundation — NSString, NSData, basic Cocoa types used by metal-cpp
    //  QuartzCore — CAMetalLayer and friends, transitively required by Metal
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=QuartzCore");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cxx/mlx_bridge.h");
    println!("cargo:rerun-if-changed=cxx/mlx_bridge.cc");
}
