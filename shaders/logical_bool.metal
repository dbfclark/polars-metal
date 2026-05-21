// shaders/logical_bool.metal
//
// 3-valued AND/OR for nullable bit-packed Boolean columns, matching
// Polars' CPU semantics exactly:
//
//   AND truth table (false dominates):
//     T ∧ T    = T (valid)
//     T ∧ F    = F (valid)
//     T ∧ null = null
//     F ∧ T    = F (valid)
//     F ∧ F    = F (valid)
//     F ∧ null = F (valid)   ← key: false dominates a null
//     null ∧ T = null
//     null ∧ F = F (valid)   ← symmetric
//     null ∧ null = null
//
//   OR truth table (true dominates):
//     T ∨ T    = T (valid)
//     T ∨ F    = T (valid)
//     T ∨ null = T (valid)   ← key: true dominates a null
//     F ∨ T    = T (valid)
//     F ∨ F    = F (valid)
//     F ∨ null = null
//     null ∨ T = T (valid)   ← symmetric
//     null ∨ F = null
//     null ∨ null = null
//
// Inputs and outputs are bit-packed: one bit per row, 8 rows per byte
// (Arrow layout, little-endian). Multiple threads share each output
// byte, so writes use `set_valid_atomic_or` from `_validity.metal`
// (same pattern as `cmp_i64.metal` / `filter_scatter.metal`). Callers
// must:
//   - Zero-initialise both `out_data` and `out_valid` (the OR is
//     append-only; it never clears bits).
//   - Allocate each output buffer as a multiple of 4 bytes (minimum 4)
//     so the kernel's `device atomic_uint*` cast is well-aligned.
//
// Threadgroup / grid: one thread per row (`n_rows` threads).
// `thread_position_in_grid` is bounds-checked against `n_rows` because
// `dispatchThreads:` pads the trailing threadgroup with no-op threads
// whose gid is out-of-range.

#include "_validity.metal"

kernel void bool_and(
    device const uint8_t*    lhs_data   [[buffer(0)]],
    device const uint8_t*    lhs_valid  [[buffer(1)]],
    device const uint8_t*    rhs_data   [[buffer(2)]],
    device const uint8_t*    rhs_valid  [[buffer(3)]],
    device       atomic_uint* out_data  [[buffer(4)]],
    device       atomic_uint* out_valid [[buffer(5)]],
    constant     uint32_t&   n_rows     [[buffer(6)]],
    uint                     gid        [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;

    bool ld = get_valid(lhs_data,  gid);  // reused helper: same "read a bit" op
    bool lv = get_valid(lhs_valid, gid);
    bool rd = get_valid(rhs_data,  gid);
    bool rv = get_valid(rhs_valid, gid);

    // 3-valued AND. Order matters: check the "false dominates" cases
    // before the "all true" case so a valid-false short-circuits even
    // when the other side is null.
    bool out_data_bit;
    bool out_valid_bit;
    if (lv && !ld) {                       // lhs is false-and-valid → F (valid)
        out_data_bit = false; out_valid_bit = true;
    } else if (rv && !rd) {                // rhs is false-and-valid → F (valid)
        out_data_bit = false; out_valid_bit = true;
    } else if (lv && ld && rv && rd) {     // both true-and-valid → T (valid)
        out_data_bit = true;  out_valid_bit = true;
    } else {                               // at least one null, no dominating false → null
        out_data_bit = false; out_valid_bit = false;
    }

    // Append-only writes. Skip both buffers when the result is null
    // (the OR semantics mean the bits stay at their zero-init values).
    if (out_valid_bit) {
        set_valid_atomic_or(out_valid, gid);
        if (out_data_bit) {
            set_valid_atomic_or(out_data, gid);
        }
    }
}

kernel void bool_or(
    device const uint8_t*    lhs_data   [[buffer(0)]],
    device const uint8_t*    lhs_valid  [[buffer(1)]],
    device const uint8_t*    rhs_data   [[buffer(2)]],
    device const uint8_t*    rhs_valid  [[buffer(3)]],
    device       atomic_uint* out_data  [[buffer(4)]],
    device       atomic_uint* out_valid [[buffer(5)]],
    constant     uint32_t&   n_rows     [[buffer(6)]],
    uint                     gid        [[thread_position_in_grid]])
{
    if (gid >= n_rows) return;

    bool ld = get_valid(lhs_data,  gid);
    bool lv = get_valid(lhs_valid, gid);
    bool rd = get_valid(rhs_data,  gid);
    bool rv = get_valid(rhs_valid, gid);

    // 3-valued OR. Order matters: check the "true dominates" cases
    // before the "both false" case so a valid-true short-circuits even
    // when the other side is null.
    bool out_data_bit;
    bool out_valid_bit;
    if (lv && ld) {                        // lhs is true-and-valid → T (valid)
        out_data_bit = true;  out_valid_bit = true;
    } else if (rv && rd) {                 // rhs is true-and-valid → T (valid)
        out_data_bit = true;  out_valid_bit = true;
    } else if (lv && !ld && rv && !rd) {   // both false-and-valid → F (valid)
        out_data_bit = false; out_valid_bit = true;
    } else {                               // at least one null, no dominating true → null
        out_data_bit = false; out_valid_bit = false;
    }

    if (out_valid_bit) {
        set_valid_atomic_or(out_valid, gid);
        if (out_data_bit) {
            set_valid_atomic_or(out_data, gid);
        }
    }
}
