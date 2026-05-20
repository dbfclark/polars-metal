//! Compute-dispatch primitives.
//!
//! A `CommandQueue` wraps a `MTLCommandQueue` and the single command buffer
//! currently in flight against it. Callers build a query's worth of
//! dispatches against the same queue and then call
//! [`CommandQueue::wait_until_complete`] before reading any outputs back
//! from the buffers.
//!
//! Threadgroup sizing follows the CLAUDE.md gotcha "Threadgroup sizing is
//! not portable across M1/M2/M3/M4. Query MTLDevice capabilities at
//! runtime; do not hardcode": [`CommandQueue::dispatch_1d`] reads
//! `maxTotalThreadsPerThreadgroup` off the pipeline state and clamps to
//! [`DEFAULT_THREADGROUP_WIDTH`] for kernels that have not been tuned.
//! Specialised kernels can pick their own threadgroup width via
//! [`dispatch_1d_with_tg`].
//!
//! We use `dispatchThreads:threadsPerThreadgroup:` (not
//! `dispatchThreadgroups:threadsPerThreadgroup:`) so non-power-of-two grid
//! sizes work out of the box; Metal pads the trailing threadgroup with
//! no-op threads and tells the kernel its `thread_position_in_grid` is
//! out-of-range via `threads_per_grid`.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder as _, MTLCommandQueue as _, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice as _, MTLSize,
};
use polars_metal_buffer::{MetalBuffer, MetalDevice};

/// Conservative cap on per-threadgroup width for the auto-sized `dispatch_1d`
/// entry point. Kernels with measured optimal widths should call
/// `dispatch_1d_with_tg` directly to bypass this cap. The cap is queried
/// against `pso.maxTotalThreadsPerThreadgroup` at runtime; we never exceed
/// what the PSO supports.
const DEFAULT_THREADGROUP_WIDTH: usize = 256;

/// Owns an `MTLCommandQueue` and tracks at most one in-flight command
/// buffer.
///
/// One queue per query is the expected usage pattern: dispatch a series of
/// kernels, then call [`wait_until_complete`](Self::wait_until_complete)
/// before reading results. Dropping the queue without waiting is safe —
/// Metal will still complete the work in the background — but reading
/// buffer contents from the CPU before the wait returns is a data race.
///
/// Reusing a queue across queries is allowed; each new
/// [`dispatch_1d`](Self::dispatch_1d) call replaces the previously
/// in-flight command buffer (the previous one continues to run; we just
/// stop tracking it for waits). For tighter control,
/// [`wait_until_complete`](Self::wait_until_complete) between queries.
pub struct CommandQueue {
    queue: Retained<ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
    in_flight: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}

// SAFETY: All state-changing methods take `&mut self`, so concurrent access
// through a shared reference is statically prevented. Send requires that
// `Retained<ProtocolObject<...>>` can move between threads; Objective-C
// retain/release on MTLCommandQueue and MTLCommandBuffer is atomic and
// Apple documents both types as safe to use from any thread (Metal Best
// Practices).
unsafe impl Send for CommandQueue {}
// SAFETY: see Send impl above. We do not expose shared interior mutability
// on `&self`; all state-changing methods take `&mut self`.
unsafe impl Sync for CommandQueue {}

/// Errors raised while constructing a queue or dispatching a kernel.
#[derive(Debug, thiserror::Error)]
pub enum DispatchError {
    /// `MTLDevice::newCommandQueue` returned nil. Typically indicates the
    /// device is in a bad state or Metal exhausted internal resources.
    #[error("command queue creation failed")]
    QueueCreation,
    /// `MTLCommandQueue::commandBuffer` returned nil. Same root causes as
    /// `QueueCreation`.
    #[error("command buffer creation failed")]
    CommandBufferCreation,
    /// `MTLCommandBuffer::computeCommandEncoder` returned nil. Rare;
    /// usually a sign the queue was created with the wrong descriptor.
    #[error("compute encoder creation failed")]
    EncoderCreation,
    /// The GPU reported an error on `waitUntilCompleted`. The message is
    /// the `NSError::localizedDescription` string.
    #[error("GPU error: {0}")]
    GpuError(String),
    /// Caller passed `n_threads == 0`. `dispatchThreads:` rejects a zero
    /// grid size; we catch it early with a clearer message.
    #[error("dispatch_1d called with n_threads == 0")]
    EmptyGrid,
}

impl CommandQueue {
    /// Construct a new command queue bound to `device`.
    pub fn new(device: &MetalDevice) -> Result<Self, DispatchError> {
        let queue = device
            .raw()
            .newCommandQueue()
            .ok_or(DispatchError::QueueCreation)?;
        Ok(Self {
            queue,
            in_flight: None,
        })
    }

    /// Dispatch a 1D compute kernel with the given pipeline state and
    /// buffer bindings.
    ///
    /// - `pso`: the compute pipeline state (from
    ///   `ShaderLibrary::pipeline(...)`).
    /// - `buffers`: bound to indices `0..buffers.len()` via
    ///   `setBuffer:offset:atIndex:`. Buffers may be aliased across slots;
    ///   Metal does not require uniqueness.
    /// - `n_threads`: total grid size in the X dimension. Y and Z are
    ///   fixed at 1.
    ///
    /// Threadgroup width =
    /// `min(pso.maxTotalThreadsPerThreadgroup, DEFAULT_THREADGROUP_WIDTH)`.
    /// `dispatchThreads:` handles non-aligned grids by padding with no-op
    /// threads; kernels should still range-check their
    /// `thread_position_in_grid` against the grid bound if they read or
    /// write outside `n_threads`.
    ///
    /// After the call returns, the command buffer has been `commit()`ed
    /// and is executing asynchronously. Callers must invoke
    /// [`wait_until_complete`](Self::wait_until_complete) before reading
    /// any output buffer from the CPU.
    ///
    /// # Multiple dispatches
    ///
    /// Issuing another dispatch before `wait_until_complete` is allowed —
    /// Metal executes both command buffers — but only the most recent
    /// buffer's completion and `error()` are tracked. Earlier dispatches'
    /// GPU errors will not surface. For chained dispatches within a single
    /// query, either interleave `wait_until_complete` calls between
    /// dispatches, or accept this limitation explicitly.
    pub fn dispatch_1d(
        &mut self,
        pso: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        buffers: &[&MetalBuffer],
        n_threads: usize,
    ) -> Result<(), DispatchError> {
        if n_threads == 0 {
            return Err(DispatchError::EmptyGrid);
        }

        // Width clamp: query the PSO at runtime, then cap at
        // `DEFAULT_THREADGROUP_WIDTH` for un-tuned kernels (CLAUDE.md
        // gotcha). Specialised kernels that have measured a better fit can
        // call dispatch_1d_with_tg. `maxTotalThreadsPerThreadgroup()`
        // returns NSUInteger (= usize on this platform); `clamp` is safe
        // because 1 <= DEFAULT_THREADGROUP_WIDTH.
        let max_tg = pso.maxTotalThreadsPerThreadgroup();
        let tg_width = max_tg.clamp(1, DEFAULT_THREADGROUP_WIDTH);
        self.dispatch_1d_with_tg(pso, buffers, n_threads, tg_width)
    }

    /// Like [`dispatch_1d`](Self::dispatch_1d) but lets the caller pick the
    /// threadgroup width directly. The width is clamped to
    /// `pso.maxTotalThreadsPerThreadgroup` so we never exceed what the
    /// pipeline supports.
    ///
    /// # Multiple dispatches
    ///
    /// Issuing another dispatch before `wait_until_complete` is allowed —
    /// Metal executes both command buffers — but only the most recent
    /// buffer's completion and `error()` are tracked. Earlier dispatches'
    /// GPU errors will not surface. For chained dispatches within a single
    /// query, either interleave `wait_until_complete` calls between
    /// dispatches, or accept this limitation explicitly.
    pub fn dispatch_1d_with_tg(
        &mut self,
        pso: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        buffers: &[&MetalBuffer],
        n_threads: usize,
        threadgroup_width: usize,
    ) -> Result<(), DispatchError> {
        if n_threads == 0 {
            return Err(DispatchError::EmptyGrid);
        }
        let max_tg = pso.maxTotalThreadsPerThreadgroup();
        let tg_width = threadgroup_width.min(max_tg).max(1);

        let command_buf = self
            .queue
            .commandBuffer()
            .ok_or(DispatchError::CommandBufferCreation)?;
        let encoder = command_buf
            .computeCommandEncoder()
            .ok_or(DispatchError::EncoderCreation)?;

        encoder.setComputePipelineState(pso);
        for (i, b) in buffers.iter().enumerate() {
            // SAFETY: `setBuffer:offset:atIndex:` is unsafe in `objc2-metal`
            // only because it takes raw protocol objects; we pass a valid
            // MTLBuffer obtained from `MetalBuffer::raw()` (still alive for
            // the duration of this call) and an in-range slot index.
            // Metal copies the binding into the encoder state, so the
            // buffer reference does not need to outlive this loop —
            // command-buffer commit retains the buffer for execution.
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(b.raw()), 0, i);
            }
        }

        let grid = MTLSize {
            width: n_threads,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: tg_width,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid, tg);
        encoder.endEncoding();
        command_buf.commit();

        self.in_flight = Some(command_buf);
        Ok(())
    }

    /// Block until the last-issued command buffer completes.
    ///
    /// Returns `Ok(())` if no command buffer is in flight or if the
    /// in-flight buffer completed successfully. Returns
    /// `Err(DispatchError::GpuError)` carrying the localized error
    /// description if Metal reported a failure (e.g. kernel page fault,
    /// resource limit exceeded).
    pub fn wait_until_complete(&mut self) -> Result<(), DispatchError> {
        let Some(buf) = self.in_flight.take() else {
            return Ok(());
        };
        // SAFETY: `waitUntilCompleted` is marked unsafe in `objc2-metal`
        // only because it blocks the current thread; there are no aliasing
        // or lifetime preconditions beyond having a live command buffer,
        // which we just took out of `self.in_flight`.
        unsafe {
            buf.waitUntilCompleted();
        }
        // SAFETY: `error()` is a safe read of the command buffer's error
        // property after completion; it is marked unsafe in `objc2-metal`
        // only because the property is generated as such. The buffer
        // outlives the call (we still hold `buf`).
        let err = unsafe { buf.error() };
        if let Some(err) = err {
            return Err(DispatchError::GpuError(
                err.localizedDescription().to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn queue_creation_succeeds_on_real_hardware() {
        let device =
            MetalDevice::system_default().expect("Metal-capable hardware required for this test");
        let _ = CommandQueue::new(&device).expect("queue must be creatable");
    }

    #[test]
    fn dispatch_with_zero_threads_returns_empty_grid() {
        let device =
            MetalDevice::system_default().expect("Metal-capable hardware required for this test");
        let lib = crate::shader_lib::shared_library(&device).expect("library loads");
        let pso = lib
            .pipeline("hello_write_constant")
            .expect("entry point exists");
        let mut queue = CommandQueue::new(&device).expect("queue creation");
        let buf = device.new_buffer_zeroed(64).expect("allocation succeeds");
        let err = queue
            .dispatch_1d(&pso, &[&buf], 0)
            .expect_err("zero-thread dispatch must error");
        assert!(matches!(err, DispatchError::EmptyGrid));
    }
}
