// shaders/cmp_f64.metal
//
// Six comparison kernels (eq, ne, lt, le, gt, ge) for f64 columns, each
// with a column-column and a column-scalar variant. Null-aware (matches
// `cmp_i64.metal`'s null contract):
//   - The output validity bit at row i is set iff `lhs_valid[i] AND
//     rhs_valid[i]` (column-column) or `lhs_valid[i]` (column-scalar).
//   - The output data bit at row i is set iff (a) both inputs are valid
//     AND (b) `lhs[i] OP rhs[i]` evaluates true under Polars/IEEE 754
//     NaN semantics.
//
// Output layout is bit-packed (one bit per row, 8 rows per byte), so
// multiple threads share each output byte. Writes use atomic OR (same
// pattern as `cmp_i64.metal`). The caller must:
//   - Zero-initialise both `out_data` and `out_valid`.
//   - Allocate each buffer as a multiple of 4 bytes (minimum 4) so the
//     `device atomic_uint*` cast is well-aligned.
//
// MSL `double` note (this is the only difference from `cmp_i64.metal`):
// Apple Silicon compute kernels do NOT support `double` — the toolchain
// rejects `as_type<double>(...)` with "double is not supported in Metal".
// We therefore implement IEEE 754 ordered comparisons in pure integer
// arithmetic on the 8-byte raw bit pattern (`ulong`), the same way
// `filter_scatter_f64` treats f64 slots as opaque `ulong` 8-byte
// payloads. The Rust host (`dispatch_cmp_f64`) casts its `&[f64]` slices
// to byte slices before constructing the MTLBuffers; the GPU side reads
// those bytes as `ulong`.
//
// The integer-emulated comparison is exact (no rounding, no precision
// loss): we are comparing IEEE bit patterns, not performing floating-
// point math. NaN, ±Inf, ±0.0, and subnormals all behave as IEEE
// requires, which matches Polars exactly.

#include "_validity.metal"

// True iff the f64 bit pattern represents a NaN.
// NaN: exponent (bits 52..62) = all 1s, mantissa (bits 0..51) != 0.
inline bool f64_is_nan(ulong bits) {
    ulong exp = (bits >> 52) & 0x7FFul;
    ulong mantissa = bits & 0x000FFFFFFFFFFFFFul;
    return (exp == 0x7FFul) && (mantissa != 0ul);
}

// True iff both bit patterns represent ±0.0 (so they should compare
// equal under IEEE 754, even though their raw bits differ by the sign
// bit). `(a | b) << 1 == 0` is true iff every non-sign bit of both a
// and b is zero, i.e. both are +0.0 or -0.0.
inline bool both_are_zero(ulong a, ulong b) {
    return ((a | b) << 1) == 0ul;
}

// Map an f64 bit pattern to a monotonic unsigned integer key under IEEE
// 754 ordered comparison (NaN aside).
//   - Non-negative x: flip the sign bit, so positives end up above the
//     midpoint of the ulong range (0x8000... and above).
//   - Negative x: invert all bits, so the most negative finite (and -inf)
//     map to small keys and the least negative (just below -0.0) maps
//     just below 0x8000..., preserving the right order.
// After mapping: -inf < -finite < -0.0_key < +0.0_key < +finite < +inf.
// (-0.0 and +0.0 differ by 1 in the key space; callers must special-
// case them via `both_are_zero`.)
inline ulong f64_total_order_key(ulong bits) {
    ulong sign_mask = ((bits >> 63) == 0ul)
        ? 0x8000000000000000ul
        : 0xFFFFFFFFFFFFFFFFul;
    return bits ^ sign_mask;
}

// IEEE 754 ordered equality. NaN op anything is false (including
// NaN == NaN); ±0 == ±0 is true; otherwise raw-bit equality is correct
// (for all non-NaN, IEEE equality matches bitwise equality except for
// the ±0 case).
inline bool f64_eq(ulong a, ulong b) {
    if (f64_is_nan(a) || f64_is_nan(b)) return false;
    if (both_are_zero(a, b)) return true;
    return a == b;
}

// IEEE 754 ordered <. NaN-or-anything → false. ±0 < ±0 → false.
inline bool f64_lt(ulong a, ulong b) {
    if (f64_is_nan(a) || f64_is_nan(b)) return false;
    if (both_are_zero(a, b)) return false;
    return f64_total_order_key(a) < f64_total_order_key(b);
}

// IEEE 754 ordered <=. NaN-or-anything → false. ±0 <= ±0 → true.
inline bool f64_le(ulong a, ulong b) {
    if (f64_is_nan(a) || f64_is_nan(b)) return false;
    if (both_are_zero(a, b)) return true;
    return f64_total_order_key(a) <= f64_total_order_key(b);
}

// IEEE 754 ordered >. Symmetric to f64_lt.
inline bool f64_gt(ulong a, ulong b) {
    if (f64_is_nan(a) || f64_is_nan(b)) return false;
    if (both_are_zero(a, b)) return false;
    return f64_total_order_key(a) > f64_total_order_key(b);
}

// IEEE 754 ordered >=. Symmetric to f64_le.
inline bool f64_ge(ulong a, ulong b) {
    if (f64_is_nan(a) || f64_is_nan(b)) return false;
    if (both_are_zero(a, b)) return true;
    return f64_total_order_key(a) >= f64_total_order_key(b);
}

// IEEE 754 != (the "unordered or distinct" predicate). Crucially,
// `NaN != x` is TRUE for every x (including NaN itself) — this is the
// one case where the NaN early-return path produces `true` rather than
// `false`. Otherwise it's just the negation of f64_eq.
inline bool f64_ne(ulong a, ulong b) {
    if (f64_is_nan(a) || f64_is_nan(b)) return true;
    if (both_are_zero(a, b)) return false;
    return a != b;
}

// Column-column variant: bind two f64 columns (as opaque `ulong*`) +
// their validity bitmaps, produce a bit-packed bool column + its
// validity bitmap.
#define CMP_F64_KERNEL_CC(name, fn)                                                                \
kernel void name(                                                                                  \
    device const ulong*       lhs_data    [[buffer(0)]],                                           \
    device const uint8_t*     lhs_valid   [[buffer(1)]],                                           \
    device const ulong*       rhs_data    [[buffer(2)]],                                           \
    device const uint8_t*     rhs_valid   [[buffer(3)]],                                           \
    device       atomic_uint* out_data    [[buffer(4)]],                                           \
    device       atomic_uint* out_valid   [[buffer(5)]],                                           \
    constant     uint32_t&    n_rows      [[buffer(6)]],                                           \
    uint                      gid         [[thread_position_in_grid]])                             \
{                                                                                                  \
    if (gid >= n_rows) return;                                                                     \
    bool lv = get_valid(lhs_valid, gid);                                                           \
    bool rv = get_valid(rhs_valid, gid);                                                           \
    if (!lv || !rv) return;                                                                        \
    set_valid_atomic_or(out_valid, gid);                                                           \
    if (fn(lhs_data[gid], rhs_data[gid])) {                                                        \
        set_valid_atomic_or(out_data, gid);                                                        \
    }                                                                                              \
}

CMP_F64_KERNEL_CC(cmp_f64_eq, f64_eq)
CMP_F64_KERNEL_CC(cmp_f64_ne, f64_ne)
CMP_F64_KERNEL_CC(cmp_f64_lt, f64_lt)
CMP_F64_KERNEL_CC(cmp_f64_le, f64_le)
CMP_F64_KERNEL_CC(cmp_f64_gt, f64_gt)
CMP_F64_KERNEL_CC(cmp_f64_ge, f64_ge)

// Column-scalar variant: bind one f64 column (as `ulong*`) + its
// validity bitmap and an f64 scalar (also passed as `ulong` — its
// `to_bits()` payload), produce a bit-packed bool column + validity
// bitmap. The scalar is treated as always-valid (the walker only lowers
// non-null literals into `Compare(Column, LiteralF64)`).
//
// NaN scalar: if the user writes `col == NaN` or similar at the Polars
// level, the IEEE rules apply (every row, valid or not, gets `false`
// for ==/<= etc., `true` for !=). This matches CPU Polars exactly.
#define CMP_F64_KERNEL_CS(name, fn)                                                                \
kernel void name(                                                                                  \
    device const ulong*       lhs_data        [[buffer(0)]],                                       \
    device const uint8_t*     lhs_valid       [[buffer(1)]],                                       \
    constant     ulong&       rhs_scalar_bits [[buffer(2)]],                                       \
    device       atomic_uint* out_data        [[buffer(3)]],                                       \
    device       atomic_uint* out_valid       [[buffer(4)]],                                       \
    constant     uint32_t&    n_rows          [[buffer(5)]],                                       \
    uint                      gid             [[thread_position_in_grid]])                         \
{                                                                                                  \
    if (gid >= n_rows) return;                                                                     \
    if (!get_valid(lhs_valid, gid)) return;                                                        \
    set_valid_atomic_or(out_valid, gid);                                                           \
    if (fn(lhs_data[gid], rhs_scalar_bits)) {                                                      \
        set_valid_atomic_or(out_data, gid);                                                        \
    }                                                                                              \
}

CMP_F64_KERNEL_CS(cmp_f64_eq_scalar, f64_eq)
CMP_F64_KERNEL_CS(cmp_f64_ne_scalar, f64_ne)
CMP_F64_KERNEL_CS(cmp_f64_lt_scalar, f64_lt)
CMP_F64_KERNEL_CS(cmp_f64_le_scalar, f64_le)
CMP_F64_KERNEL_CS(cmp_f64_gt_scalar, f64_gt)
CMP_F64_KERNEL_CS(cmp_f64_ge_scalar, f64_ge)
