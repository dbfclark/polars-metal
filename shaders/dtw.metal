#include <metal_stdlib>
using namespace metal;

// Banded Euclidean DTW. ONE THREADGROUP PER QUERY PAIR.
//
// Grid: dispatched as (n_pairs * TG_SIZE) threads in groups of TG_SIZE, so
//   threadgroup_position_in_grid == pair index. All TG_SIZE threads cooperate
//   to load the reference + init the DP rows; thread 0 then runs the L*L DP
//   serially across the two rolling rows held in threadgroup memory. (Intra-
//   pair anti-diagonal parallelism is a future optimization; the dominant win
//   is the N-way parallelism across pairs with the DP in fast threadgroup mem.)
//
// Threadgroup memory (static, sized by MAX_L): reference[MAX_L] + two rolling
//   DP rows row_a[MAX_L+1], row_b[MAX_L+1]. MAX_L=1024 => ~12 KB.
//
// Cost: squared difference; the kernel returns sqrt(D[L,L]) directly. Cell
//   cost (q_i - r_j)^2.
// Band: window < 0 => unconstrained; else cell (i,j) computed iff |i-j| <= window.

constant uint MAX_L = 1024;

kernel void dtw_banded(
    device const float* queries   [[buffer(0)]],   // n_pairs * L, pair-major
    device const float* reference [[buffer(1)]],   // L
    device float*       out       [[buffer(2)]],   // n_pairs
    constant uint&      n_pairs   [[buffer(3)]],
    constant uint&      seq_len   [[buffer(4)]],   // L (<= MAX_L)
    constant int&       window    [[buffer(5)]],   // <0 => full DTW
    uint pair_id [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tgsize  [[threads_per_threadgroup]])
{
    if (pair_id >= n_pairs) return;
    uint L = seq_len;

    threadgroup float ref_s[MAX_L];
    threadgroup float row_a[MAX_L + 1];
    threadgroup float row_b[MAX_L + 1];

    device const float* q = queries + (uint64_t)pair_id * L;

    // Cooperative load of the reference + DP-row init.
    for (uint j = tid; j < L; j += tgsize) ref_s[j] = reference[j];
    // prev row = D[0, *]: D[0,0]=0, else +inf.
    for (uint j = tid; j <= L; j += tgsize) row_a[j] = (j == 0) ? 0.0f : INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        threadgroup float* prev = row_a;
        threadgroup float* cur  = row_b;
        for (uint i = 1; i <= L; ++i) {
            cur[0] = INFINITY;
            float qi = q[i - 1];
            uint jlo = 1, jhi = L;
            if (window >= 0) {
                long lo = (long)i - (long)window;
                long hi = (long)i + (long)window;
                if (lo > 1) jlo = (uint)lo;
                if (hi < (long)L) jhi = (uint)hi;
                for (uint j = 1; j < jlo; ++j) cur[j] = INFINITY; // left of band
            }
            for (uint j = jlo; j <= jhi; ++j) {
                float d = qi - ref_s[j - 1];
                float cost = d * d;
                float m = min(min(prev[j], cur[j - 1]), prev[j - 1]);
                cur[j] = cost + m;
            }
            for (uint j = jhi + 1; j <= L; ++j) cur[j] = INFINITY; // right of band
            threadgroup float* t = prev; prev = cur; cur = t;       // ping-pong
        }
        out[pair_id] = sqrt(prev[L]);
    }
}
