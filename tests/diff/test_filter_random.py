"""Differential property tests for the M1 scan/project/filter feature set.

For a randomly-generated (DataFrame, predicate, projection) triple drawn
from the closed M1 strategy set, the Metal engine must produce a
byte-identical result to Polars CPU.

The DataFrame strategy biases null density toward 0%/100% to stress the
bit-packing path. f64 columns intentionally exclude NaN — see the
cmp_f64 IEEE-754-vs-TotalOrd entry in ``docs/open-questions.md``; that
divergence is covered by a dedicated xfail-strict edge case in
``test_filter_edges.py``.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

import polars_metal
from tests.diff.strategies import (
    m1_null_density_dataframe,
    m1_predicate_expr,
    m1_projection_subset,
)


# We use ``st.data()`` so the predicate strategy can depend on the
# DataFrame's schema (which is only known after the frame has been drawn).
# This is the canonical hypothesis pattern; calling ``.example()`` would
# silently break hypothesis's shrinking/replaying machinery.
@given(df=m1_null_density_dataframe(), data=st.data())
@settings(max_examples=200, deadline=None)
def test_filter_select_random_inputs_match_cpu(df, data) -> None:  # type: ignore[no-untyped-def]
    schema = df.schema
    pred = data.draw(m1_predicate_expr(schema))
    cols = data.draw(m1_projection_subset(schema))
    cpu = df.lazy().filter(pred).select(cols).collect()
    metal = df.lazy().filter(pred).select(cols).collect(engine=polars_metal.MetalEngine())
    assert_frame_equal(cpu, metal)
