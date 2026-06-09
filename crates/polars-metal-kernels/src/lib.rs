//! Custom MSL kernel wrappers.
//!
//! M1 introduces the shader build/load pipeline (`shader_lib`) and the
//! compute-dispatch primitives (`command`); individual kernel modules
//! (filter, comparison, logical) land in subsequent tasks.

pub mod aggregate_fused;
pub mod cmp;
pub mod command;
pub mod fft;
pub mod filter;
pub mod groupby;
pub mod groupby_build_partitioned;
pub mod groupby_build_sort;
pub mod groupby_global_hash;
pub mod logical;
pub mod pipeline;
pub mod rolling;
pub mod shader_lib;
