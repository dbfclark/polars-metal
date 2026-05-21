// crates/polars-metal-kernels/tests/test_dispatch.rs
//
// End-to-end validation of the dispatch pipeline: load the hello-world PSO,
// allocate an output buffer, dispatch a 1D grid, wait for completion, and
// assert the kernel wrote 42 into every slot.
//
// This is the smoke test for `CommandQueue` + `MetalDevice::new_buffer_zeroed`
// + the metallib loader from Task 2. If it passes we have a working dispatch
// path; subsequent kernels need only swap in their own PSO and buffers.
#![allow(clippy::expect_used)]

use polars_metal_buffer::MetalDevice;
use polars_metal_kernels::command::CommandQueue;
use polars_metal_kernels::shader_lib;

#[test]
fn hello_kernel_writes_42_into_every_slot() {
    let device = MetalDevice::system_default().expect("Metal-capable hardware");
    let lib = shader_lib::shared_library(&device).expect("library loads");
    let pso = lib
        .pipeline("hello_write_constant")
        .expect("entry point exists");

    let n: usize = 1024;
    let mut queue = CommandQueue::new(&device).expect("queue creation");
    let buf = device
        .new_buffer_zeroed(n * std::mem::size_of::<u32>())
        .expect("alloc");

    queue
        .dispatch_1d(&pso, &[&buf], n)
        .expect("dispatch succeeds");
    queue.wait_until_complete().expect("no GPU errors");

    // SAFETY: buf was allocated as u32 * n bytes; StorageModeShared makes it
    // CPU-readable, and `wait_until_complete` guarantees the GPU writes are
    // visible to the CPU. The lifetime of the slice is bounded by `buf`,
    // which is alive for the duration of the block.
    let slice: &[u32] =
        unsafe { std::slice::from_raw_parts(buf.as_slice().as_ptr() as *const u32, n) };
    for (i, v) in slice.iter().enumerate() {
        assert_eq!(*v, 42, "slot {i} should hold 42, got {v}");
    }
}

#[test]
fn dispatch_non_power_of_two_grid_size() {
    // dispatchThreads (vs dispatchThreadgroups) handles arbitrary grid sizes
    // by computing threadgroup bounds automatically. Verify a non-round size.
    let device = MetalDevice::system_default().expect("Metal-capable hardware");
    let lib = shader_lib::shared_library(&device).expect("library loads");
    let pso = lib
        .pipeline("hello_write_constant")
        .expect("entry point exists");

    let n: usize = 1000; // not a multiple of any plausible threadgroup width
    let mut queue = CommandQueue::new(&device).expect("queue creation");
    let buf = device
        .new_buffer_zeroed(n * std::mem::size_of::<u32>())
        .expect("alloc");

    queue
        .dispatch_1d(&pso, &[&buf], n)
        .expect("dispatch succeeds");
    queue.wait_until_complete().expect("no GPU errors");

    // SAFETY: see above.
    let slice: &[u32] =
        unsafe { std::slice::from_raw_parts(buf.as_slice().as_ptr() as *const u32, n) };
    for (i, v) in slice.iter().enumerate() {
        assert_eq!(*v, 42, "slot {i} should hold 42, got {v}");
    }
}

#[test]
fn wait_without_dispatch_is_no_op() {
    // A queue that has issued nothing yet should not error on
    // wait_until_complete; this mirrors how a query that bails out before
    // dispatching will still drop the queue cleanly.
    let device = MetalDevice::system_default().expect("Metal-capable hardware");
    let mut queue = CommandQueue::new(&device).expect("queue creation");
    queue.wait_until_complete().expect("no-op wait");
}
