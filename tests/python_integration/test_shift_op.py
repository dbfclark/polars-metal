"""M5 rolling Task 4: PyFusionScope.push_op carries optional scalar param."""

from polars_metal import _native


def test_push_op_accepts_param():
    s = _native.PyFusionScope()
    a = s.add_input("x", "F32")
    sh = s.push_op("Shift", [a], 3)  # third positional arg = scalar param
    s.mark_output(sh)
    assert s.n_ops() == 1
