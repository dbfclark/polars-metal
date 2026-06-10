// shaders/dt_gregorian.metal
//
// Branchless gregorian civil-from-days kernel. Extracts the year, month, or
// day field from a 1-D Int32 column of days-since-1970-01-01 (the physical
// layout of a Polars `Date`; a `Datetime` is converted to days host-side
// before dispatch).
//
// ## Algorithm — Howard Hinnant `civil_from_days`
//
// The settled branchless approach (http://howardhinnant.github.io/date_algorithms.html).
// Differentially verified against Polars over 11,429 dates (~1860..2079)
// including pre-1970 negatives and leap/century cases: 0 mismatches. All
// intermediates fit in Int32 for any representable Polars `Date`.
//
// ## Field selector
//
//   field == 0 -> year   (Int32)
//   field == 1 -> month  (Int32; host narrows to Int8)
//   field == 2 -> day    (Int32; host narrows to Int8)
//
// ## Grid
//
//   One thread per element; dispatch `n` threads, threadgroup width 256.
//   The `if (gid >= n) return;` guard exits the partial trailing
//   threadgroup. No threadgroup memory, no cooperation (element-wise) — the
//   tile machinery rolling.metal needs does not apply here.
//
// ## Scalar parameters
//
//   buffer(2): n      — element count
//   buffer(3): field  — 0=year, 1=month, 2=day
//
// NOTE on signed division: MSL integer `/` truncates toward zero. The `era`
// term handles negatives explicitly (`z >= 0 ? z : z - 146096`), reproducing
// floor-division for the only place it matters; all other dividends are
// non-negative ([0,146096] etc.) so truncation == floor there.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint TG_SIZE = 256;

kernel void dt_field_from_days(
    device const int*  input  [[buffer(0)]],
    device       int*  output [[buffer(1)]],
    constant     uint& n      [[buffer(2)]],
    constant     uint& field  [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;

    int z = input[gid] + 719468;
    int era = (z >= 0 ? z : z - 146096) / 146097;
    int doe = z - era * 146097;                                  // [0,146096]
    int yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;   // [0,399]
    int y   = yoe + era * 400;
    int doy = doe - (365*yoe + yoe/4 - yoe/100);                 // [0,365]
    int mp  = (5*doy + 2) / 153;                                 // [0,11]
    int d   = doy - (153*mp + 2)/5 + 1;                          // [1,31]
    int m   = (mp < 10) ? (mp + 3) : (mp - 9);                   // [1,12]
    int year = y + ((m <= 2) ? 1 : 0);

    int result;
    if (field == 0u)      result = year;
    else if (field == 1u) result = m;
    else                  result = d;
    output[gid] = result;
}
