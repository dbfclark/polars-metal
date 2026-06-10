"""Engine-level differential tests for GPU-accelerated dt.year/month/day."""

import datetime

import polars as pl
import pytest

from polars_metal import MetalEngine, _native

_FIELDS = ["year", "month", "day"]
_OUT_DTYPE = {"year": pl.Int32, "month": pl.Int8, "day": pl.Int8}


def _dt_dispatches(lf, eng) -> int:
    """Count execute_dt dispatches (proves the GPU kernel path runs)."""
    n = {"c": 0}
    orig = _native.execute_dt

    def cnt(inp, out, field):
        n["c"] += 1
        return orig(inp=inp, out=out, field=field)

    _native.execute_dt = cnt
    try:
        lf.collect(engine=eng)
    finally:
        _native.execute_dt = orig
    return n["c"]


def _date_range(start: datetime.date, n: int, step: int = 1) -> list[datetime.date]:
    return [start + datetime.timedelta(days=i * step) for i in range(n)]


@pytest.mark.parametrize("field", _FIELDS)
def test_date_field_byte_exact_and_gpu(field):
    eng = MetalEngine()
    dates = _date_range(datetime.date(1900, 1, 1), 2000, step=33)
    df = pl.DataFrame({"d": dates, "v": list(range(len(dates)))})
    expr = getattr(pl.col("d").dt, field)().alias("o")
    lf = df.lazy().with_columns(expr)
    assert _dt_dispatches(lf, eng) == 1, f"{field} should use the GPU kernel"
    got = lf.collect(engine=eng)
    want = lf.collect()
    assert got.equals(want), f"{field}: mismatch"
    assert got["o"].dtype == _OUT_DTYPE[field]


@pytest.mark.parametrize("tu", ["ms", "us", "ns"])
@pytest.mark.parametrize("field", _FIELDS)
def test_datetime_all_time_units(tu, field):
    eng = MetalEngine()
    base = [
        datetime.datetime(2020, 3, 15, 12, 30),
        datetime.datetime(1969, 12, 31, 1, 0),  # pre-epoch (day -1)
        datetime.datetime(2000, 2, 29, 23, 59),  # leap
        datetime.datetime(1970, 1, 1, 0, 0),  # epoch
    ]
    s = pl.Series("t", base, dtype=pl.Datetime(tu))
    lf = pl.DataFrame({"t": s}).lazy().with_columns(getattr(pl.col("t").dt, field)().alias("o"))
    assert _dt_dispatches(lf, eng) == 1
    got, want = lf.collect(engine=eng), lf.collect()
    assert got.equals(want), f"{tu}/{field}: mismatch"
    assert got["o"].dtype == _OUT_DTYPE[field]


@pytest.mark.parametrize("field", _FIELDS)
def test_nulls_preserved(field):
    eng = MetalEngine()
    dates = [datetime.date(2020, 3, 15), None, datetime.date(1969, 12, 31), None]
    df = pl.DataFrame({"d": pl.Series("d", dates)})
    lf = df.lazy().with_columns(getattr(pl.col("d").dt, field)().alias("o"))
    got, want = lf.collect(engine=eng), lf.collect()
    assert got.equals(want), f"{field} nulls: mismatch"
    assert got["o"].dtype == _OUT_DTYPE[field]


def test_empty_frame():
    eng = MetalEngine()
    df = pl.DataFrame({"d": pl.Series("d", [], dtype=pl.Date)})
    lf = df.lazy().with_columns(pl.col("d").dt.year().alias("o"))
    got, want = lf.collect(engine=eng), lf.collect()
    assert got.equals(want)


def test_multiple_fields_one_collect():
    eng = MetalEngine()
    df = pl.DataFrame({"d": [datetime.date(2020, 3, 15), datetime.date(1999, 7, 4)]})
    lf = df.lazy().with_columns(
        pl.col("d").dt.year().alias("y"),
        pl.col("d").dt.month().alias("mo"),
        pl.col("d").dt.day().alias("da"),
    )
    assert _dt_dispatches(lf, eng) == 3  # one kernel call per field
    got, want = lf.collect(engine=eng), lf.collect()
    assert got.equals(want)


def test_unsupported_field_falls_back_and_matches():
    eng = MetalEngine()
    df = pl.DataFrame({"d": [datetime.date(2020, 3, 15)]})
    lf = df.lazy().with_columns(pl.col("d").dt.weekday().alias("wd"))
    assert _dt_dispatches(lf, eng) == 0  # not handled -> CPU
    got, want = lf.collect(engine=eng), lf.collect()
    assert got.equals(want)
