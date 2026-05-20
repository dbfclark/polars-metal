//! Custom MSL kernel wrappers.
//!
//! M1 introduces the shader build/load pipeline (`shader_lib`) and the
//! compute-dispatch primitives (`command`); individual kernel modules
//! (filter, comparison, logical) land in subsequent tasks.

pub mod command;
pub mod filter;
pub mod shader_lib;
