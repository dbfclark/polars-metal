from __future__ import annotations

import json as _json

import numpy as np

from tests.bench.m9_crossing._crossing import (
    CostModel,
    fit_cost_model,
    to_cpu,
    to_gpu,
)
from tests.bench.m9_crossing.emit import to_json, to_markdown
from tests.bench.m9_crossing.run import Row


def test_crossing_roundtrip_preserves_data():
    a = np.arange(64, dtype=np.float32).reshape(8, 8)
    g = to_gpu(a)
    b = to_cpu(g)
    assert np.array_equal(a, b)


def test_fit_cost_model_returns_positive_coeffs():
    cm = fit_cost_model()
    assert isinstance(cm, CostModel)
    assert cm.alpha_ms_per_byte > 0
    assert cm.beta_ms_per_crossing > 0
    assert cm.predict(bytes_crossed=10_000_000, n_crossings=1) > cm.predict(
        bytes_crossed=1_000, n_crossings=1
    )
    assert cm.predict(bytes_crossed=1_000, n_crossings=10) > cm.predict(
        bytes_crossed=1_000, n_crossings=1
    )


def test_emit_wellformed():
    rows = [
        Row("retrieve_rerank", "gather", 1000, "resident", 5.0, 3.0),
        Row("retrieve_rerank", "gather", 1000, "all_cpu", 15.0, 1.0),
    ]
    cm = CostModel(alpha_ms_per_byte=1e-7, beta_ms_per_crossing=0.05)
    md = to_markdown(rows, cm, {"machine": "arm64"})
    assert "crossing cost model" in md.lower()
    assert "retrieve_rerank" in md and "resident" in md
    parsed = _json.loads(to_json(rows, cm, {"machine": "arm64"}))
    assert parsed["rows"][0]["path"] == "resident"
    assert parsed["cost_model"]["beta_ms_per_crossing"] == 0.05
