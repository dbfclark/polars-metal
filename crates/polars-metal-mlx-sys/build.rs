// crates/polars-metal-mlx-sys/build.rs
fn main() {
    cxx_build::bridge("src/lib.rs")
        .file("cxx/hello.cc")
        .flag_if_supported("-std=c++17")
        .compile("polars_metal_mlx_bridge");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cxx/hello.h");
    println!("cargo:rerun-if-changed=cxx/hello.cc");
}
