"""Smoke + correctness gate: for each pipeline, every path produces the IDENTICAL
result at the smallest size. A fast-but-wrong path is caught, and timing is only
ever apples-to-apples. Runs in `make test-unit`.
"""

from __future__ import annotations

import pytest

from tests.bench.m9_crossing._pipelines import PIPELINES


def _cases():
    for p in PIPELINES:
        for path in p.paths:
            yield pytest.param(p, path, id=f"{p.name}:{path}")


@pytest.mark.parametrize("pipeline,path", list(_cases()))
def test_paths_agree(pipeline, path):
    inp = pipeline.make_inputs(min(pipeline.sizes))
    base = pipeline.paths["all_cpu"](inp)
    pipeline.check(base, pipeline.paths[path](inp))
