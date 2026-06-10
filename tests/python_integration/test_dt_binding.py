"""Direct smoke test of the execute_dt native binding (no engine/detect)."""

import numpy as np

from polars_metal import _native


def _hinnant(z: int) -> tuple[int, int, int]:
    z += 719468
    era = (z if z >= 0 else z - 146096) // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + 3 if mp < 10 else mp - 9
    return (y + (1 if m <= 2 else 0), m, d)


def test_execute_dt_year_month_day():
    days = np.array(
        [18336, -1, -25567, 0, 11016], dtype=np.int32
    )  # incl negatives, leap (2000-02-29)
    for field in (0, 1, 2):  # year, month, day
        inp = np.ascontiguousarray(days, dtype=np.int32)
        out = np.empty(inp.size, dtype=np.int32)
        _native.execute_dt(
            inp=(inp.ctypes.data, inp.size),
            out=(out.ctypes.data, out.size),
            field=field,
        )
        want = np.array([_hinnant(int(z))[field] for z in days], dtype=np.int32)
        np.testing.assert_array_equal(out, want)


def test_execute_dt_empty_is_noop():
    inp = np.empty(0, dtype=np.int32)
    out = np.empty(0, dtype=np.int32)
    _native.execute_dt(inp=(inp.ctypes.data, 0), out=(out.ctypes.data, 0), field=0)
