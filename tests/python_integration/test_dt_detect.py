"""Unit tests for dt.year/month/day serialize-detection (no engine call)."""

import datetime

import polars as pl

from polars_metal._dt_detect import DtBinding, find_dt_bindings


def test_detect_date_year_month_day():
    df = pl.DataFrame({"d": [datetime.date(2020, 3, 15)], "v": [1.0]})
    lf = df.lazy().with_columns(
        pl.col("d").dt.year().alias("y"),
        pl.col("d").dt.month().alias("mo"),
        pl.col("d").dt.day().alias("da"),
    )
    got = {(b.field, b.out_name, b.column) for b in find_dt_bindings(lf)}
    assert got == {("year", "y", "d"), ("month", "mo", "d"), ("day", "da", "d")}


def test_detect_datetime_carries_time_unit():
    s = pl.Series("t", [datetime.datetime(2020, 3, 15, 1, 0)], dtype=pl.Datetime("us"))
    lf = pl.DataFrame({"t": s}).lazy().with_columns(pl.col("t").dt.year().alias("y"))
    bindings = find_dt_bindings(lf)
    assert len(bindings) == 1
    b = bindings[0]
    assert isinstance(b, DtBinding)
    assert b.field == "year" and b.column == "t"
    assert b.units_per_day == 86_400_000_000  # us
    assert b.is_date is False


def test_detect_date_has_no_units():
    lf = (
        pl.DataFrame({"d": [datetime.date(2020, 1, 1)]})
        .lazy()
        .with_columns(pl.col("d").dt.day().alias("o"))
    )
    b = find_dt_bindings(lf)[0]
    assert b.is_date is True and b.units_per_day is None


def test_non_handleable_omitted():
    df = pl.DataFrame(
        {
            "d": [datetime.date(2020, 1, 1)],
            "v": [1.0],
            "t": pl.Series([datetime.datetime(2020, 1, 1)], dtype=pl.Datetime("us", "UTC")),
        }
    )
    # Unsupported accessor, sub-expression input, tz-aware datetime, non-date col.
    lf = df.lazy().with_columns(
        pl.col("d").dt.weekday().alias("wd"),       # unsupported field
        (pl.col("d") + pl.duration(days=1)).dt.year().alias("expr_in"),  # sub-expr input
        pl.col("t").dt.year().alias("tz"),          # tz-aware -> CPU
        (pl.col("v") * 2).alias("plain"),           # not a dt expr
    )
    assert find_dt_bindings(lf) == []


def test_out_name_shadowing_source_rejected():
    # An output that overwrites a source column the kernel must read -> []
    lf = (
        pl.DataFrame({"d": [datetime.date(2020, 1, 1)]})
        .lazy()
        .with_columns(pl.col("d").dt.year().alias("d"))
    )
    assert find_dt_bindings(lf) == []
