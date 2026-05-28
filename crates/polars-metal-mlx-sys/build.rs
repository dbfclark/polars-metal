// crates/polars-metal-mlx-sys/build.rs
use std::path::PathBuf;

const REQUIRED_MLX_VERSION: &str = "0.22.0";

fn check_mlx_version(cmake_lists: &std::path::Path) {
    let contents = match std::fs::read_to_string(cmake_lists) {
        Ok(s) => s,
        Err(_) => {
            println!("cargo:warning=could not read vendor/mlx/CMakeLists.txt to verify MLX version");
            return;
        }
    };
    // Look for `set(MLX_VERSION X.Y.Z)`
    for line in contents.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("set(MLX_VERSION ") {
            let found = rest.trim_end_matches(|c: char| c == ')' || c.is_whitespace());
            if found != REQUIRED_MLX_VERSION {
                println!(
                    "cargo:warning=vendor/mlx MLX_VERSION={found} but polars-metal-mlx-sys pins {REQUIRED_MLX_VERSION}; bump deliberately"
                );
            }
            return;
        }
    }
    println!("cargo:warning=could not parse MLX_VERSION from vendor/mlx/CMakeLists.txt");
}

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let cmake_lists = manifest_dir.join("../../vendor/mlx/CMakeLists.txt");
    check_mlx_version(&cmake_lists);
    println!("cargo:rerun-if-changed={}", cmake_lists.display());

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
