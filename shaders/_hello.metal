// shaders/_hello.metal
//
// Hello-world kernel used solely to validate the metallib build/load pipeline.
// One thread per grid position writes a known constant; tests can assert the
// kernel resolves and its PSO is usable. Real kernels arrive in later tasks.
#include <metal_stdlib>
using namespace metal;

kernel void hello_write_constant(
    device uint32_t* out [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
    out[gid] = 42u;
}
