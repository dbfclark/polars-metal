"""Placeholder pytest-benchmark scaffolding."""


def test_placeholder(benchmark) -> None:  # type: ignore[no-untyped-def]
    result = benchmark(lambda: 1 + 1)
    assert result == 2
