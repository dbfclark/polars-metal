// shaders/cmp_i64.metal
//
// Six comparison kernels (eq, ne, lt, le, gt, ge) for i64 columns, each
// with a column-column and a column-scalar variant. Null-aware:
//   - The output validity bit at row i is set iff `lhs_valid[i] AND
//     rhs_valid[i]` (column-column) or `lhs_valid[i]` (column-scalar).
//   - The output data bit at row i is set iff (a) both inputs are valid
//     AND (b) `lhs[i] OP rhs[i]` is true.
//
// Output layout is bit-packed (one bit per row, 8 rows per byte), so
// multiple threads share each output byte. Writes use atomic OR (same
// pattern as `filter_scatter_bool`). The caller must:
//   - Zero-initialise both `out_data` and `out_valid`.
//   - Allocate each buffer as a multiple of 4 bytes (minimum 4) so the
//     `device atomic_uint*` cast is well-aligned.
//
// MSL note: we use `int64_t` (from `<metal_stdlib>`, included transitively
// via `_validity.metal`) rather than `long`. On Apple Silicon's MSL, the
// `long` keyword is 32-bit in compute kernels; `int64_t` is the portable
// signed 64-bit type.
//
// Macros generate the twelve entry points from two body templates. cuDF
// uses an analogous template-driven approach for its comparison kernels
// (see references/cudf/cpp/src/binaryop/compiled/).

#include "_validity.metal"

// Column-column variant: bind two i64 columns + their validity bitmaps,
// produce a bit-packed bool column + its validity bitmap.
#define CMP_KERNEL_CC(name, op)                                                                    \
kernel void name(                                                                                  \
    device const int64_t*    lhs_data    [[buffer(0)]],                                            \
    device const uint8_t*    lhs_valid   [[buffer(1)]],                                            \
    device const int64_t*    rhs_data    [[buffer(2)]],                                            \
    device const uint8_t*    rhs_valid   [[buffer(3)]],                                            \
    device       atomic_uint* out_data   [[buffer(4)]],                                            \
    device       atomic_uint* out_valid  [[buffer(5)]],                                            \
    constant     uint32_t&   n_rows      [[buffer(6)]],                                            \
    uint                     gid         [[thread_position_in_grid]])                              \
{                                                                                                  \
    if (gid >= n_rows) return;                                                                     \
    bool lv = get_valid(lhs_valid, gid);                                                           \
    bool rv = get_valid(rhs_valid, gid);                                                           \
    if (!lv || !rv) return;                                                                        \
    set_valid_atomic_or(out_valid, gid);                                                           \
    if (lhs_data[gid] op rhs_data[gid]) {                                                          \
        set_valid_atomic_or(out_data, gid);                                                        \
    }                                                                                              \
}

CMP_KERNEL_CC(cmp_i64_eq, ==)
CMP_KERNEL_CC(cmp_i64_ne, !=)
CMP_KERNEL_CC(cmp_i64_lt, <)
CMP_KERNEL_CC(cmp_i64_le, <=)
CMP_KERNEL_CC(cmp_i64_gt, >)
CMP_KERNEL_CC(cmp_i64_ge, >=)

// Column-scalar variant: bind one i64 column + its validity bitmap and a
// scalar i64, produce the same bit-packed bool column + validity bitmap.
// The scalar is treated as always-valid (the M1 walker only lowers
// non-null literals into `Compare(Column, LiteralI64)`).
#define CMP_KERNEL_CS(name, op)                                                                    \
kernel void name(                                                                                  \
    device const int64_t*    lhs_data    [[buffer(0)]],                                            \
    device const uint8_t*    lhs_valid   [[buffer(1)]],                                            \
    constant     int64_t&    rhs_scalar  [[buffer(2)]],                                            \
    device       atomic_uint* out_data   [[buffer(3)]],                                            \
    device       atomic_uint* out_valid  [[buffer(4)]],                                            \
    constant     uint32_t&   n_rows      [[buffer(5)]],                                            \
    uint                     gid         [[thread_position_in_grid]])                              \
{                                                                                                  \
    if (gid >= n_rows) return;                                                                     \
    if (!get_valid(lhs_valid, gid)) return;                                                        \
    set_valid_atomic_or(out_valid, gid);                                                           \
    if (lhs_data[gid] op rhs_scalar) {                                                             \
        set_valid_atomic_or(out_data, gid);                                                        \
    }                                                                                              \
}

CMP_KERNEL_CS(cmp_i64_eq_scalar, ==)
CMP_KERNEL_CS(cmp_i64_ne_scalar, !=)
CMP_KERNEL_CS(cmp_i64_lt_scalar, <)
CMP_KERNEL_CS(cmp_i64_le_scalar, <=)
CMP_KERNEL_CS(cmp_i64_gt_scalar, >)
CMP_KERNEL_CS(cmp_i64_ge_scalar, >=)
