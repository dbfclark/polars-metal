//! Custom MSL kernel wrappers.
//!
//! M1 introduces the shader build/load pipeline (`shader_lib`); individual
//! kernel modules (filter, comparison, logical) land in subsequent tasks.

pub mod shader_lib;
