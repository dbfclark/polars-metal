"""Unit tests for dt.year/month/day serialize-detection (no engine call)."""

import datetime
import os
import tempfile

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
        pl.col("d").dt.weekday().alias("wd"),  # unsupported field
        (pl.col("d") + pl.duration(days=1)).dt.year().alias("expr_in"),  # sub-expr input
        pl.col("t").dt.year().alias("tz"),  # tz-aware -> CPU
        (pl.col("v") * 2).alias("plain"),  # not a dt expr
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


def _make_scan_dt_lf(exprs):
    """Return a scan_parquet-backed LazyFrame that exercises the SLOW detect path."""
    df = pl.DataFrame({"d": pl.Series([datetime.date(2020, 1, 1)], dtype=pl.Date)})
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmppath = f.name
    try:
        df.write_parquet(tmppath)
        return pl.scan_parquet(tmppath).with_columns(exprs), tmppath
    except Exception:
        os.unlink(tmppath)
        raise


def test_slow_path_bare_plus_aliased_dt_returns_empty():
    """Slow path: bare dt.year() (shadows 'd') alongside dt.year().alias('yr')
    returns [] for both old and new code.

    This is NOT the same regression as rolling: for dt the schema for the
    resulting LazyFrame already shows d as Int32 (the bare overwrite), so the
    column-dtype lookup inside _parse_dt_expr fails for ALL bindings regardless
    of whether the bare one is skipped first.  Both old and new behaviour = [].

    The test pins that find_dt_bindings does not raise and returns [] for this
    mixed in-place + aliased dt case."""
    lf, tmppath = _make_scan_dt_lf(
        [
            pl.col("d").dt.year(),  # bare — shadows "d" in schema
            pl.col("d").dt.year().alias("yr"),  # aliased — also broken by schema
        ]
    )
    try:
        assert find_dt_bindings(lf) == []
    finally:
        os.unlink(tmppath)


def test_slow_path_bare_dt_alone_returns_empty():
    """Slow path: a single bare dt.year() (shadows source) returns [] — no dispatch."""
    lf, tmppath = _make_scan_dt_lf([pl.col("d").dt.year()])
    try:
        assert find_dt_bindings(lf) == []
    finally:
        os.unlink(tmppath)


def test_slow_path_aliased_dt_alone_dispatches():
    """Slow path: a single aliased dt.year() is dispatched normally."""
    lf, tmppath = _make_scan_dt_lf([pl.col("d").dt.year().alias("yr")])
    try:
        result = find_dt_bindings(lf)
        assert len(result) == 1
        assert result[0].out_name == "yr"
    finally:
        os.unlink(tmppath)
