// crates/polars-metal-mlx-sys/build.rs
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
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

    // Point the linker at the MLX build output.
    // Built with -DMLX_BUILD_METAL=OFF (Metal toolchain absent on this host);
    // produces libmlx.a (static). Switch back to dylib once Metal toolchain
    // is available and MLX is rebuilt with -DMLX_BUILD_METAL=ON.
    println!("cargo:rustc-link-search=native={}", mlx_build.display());
    println!("cargo:rustc-link-lib=static=mlx");

    // MLX (CPU backend) depends on Accelerate.
    println!("cargo:rustc-link-lib=framework=Accelerate");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cxx/mlx_bridge.h");
    println!("cargo:rerun-if-changed=cxx/mlx_bridge.cc");
}
