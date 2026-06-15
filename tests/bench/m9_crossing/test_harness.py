from __future__ import annotations

import numpy as np

from tests.bench.m9_crossing._crossing import (
    CostModel,
    fit_cost_model,
    to_cpu,
    to_gpu,
)


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
