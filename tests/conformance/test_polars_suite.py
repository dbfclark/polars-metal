"""Run Polars' own lazyframe test suite with engine forced to MetalEngine().

This catches real regressions in our engine path: any Polars test that passes
on CPU but fails under our engine is a defect.

A baseline of version-skew failures (tests written for newer Polars features
that the pinned wheel doesn't have) lives at
`tests/conformance/_polars_known_failures.txt`. The test asserts that the
observed failure set is a subset of the baseline — new failures = real
regressions in our engine, fixed failures = baseline can shrink.

Skipped if the `references/polars/py-polars/tests/unit/lazyframe` directory
isn't present (e.g., references/ wasn't cloned). Run `make refresh-refs`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
POLARS_TEST_ROOT = REPO_ROOT / "references" / "polars" / "py-polars"
POLARS_LAZYFRAME_TESTS = POLARS_TEST_ROOT / "tests" / "unit" / "lazyframe"
KNOWN_FAILURES_FILE = Path(__file__).parent / "_polars_known_failures.txt"


def _load_known_failures() -> set[str]:
    if not KNOWN_FAILURES_FILE.exists():
        return set()
    return {
        line.strip()
        for line in KNOWN_FAILURES_FILE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def _parse_failures(stdout: str) -> set[str]:
    """Parse pytest --tb=no output for FAILED node ids."""
    failures: set[str] = set()
    for line in stdout.splitlines():
        if line.startswith("FAILED "):
            # `FAILED tests/unit/lazyframe/test_xyz.py::test_name - reason...`
            nodeid = line.removeprefix("FAILED ").split(" - ", 1)[0].strip()
            failures.add(nodeid)
    return failures


@pytest.mark.skipif(
    not POLARS_LAZYFRAME_TESTS.exists(),
    reason="references/polars not cloned; run `make refresh-refs`",
)
def test_polars_lazyframe_suite_under_metal() -> None:
    """Run polars/py-polars/tests/unit/lazyframe with engine forced to MetalEngine().

    Asserts no new failures vs the recorded baseline.
    """
    if shutil.which("pyarrow") is None:
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            pytest.skip("pyarrow not installed; required for Polars' own test suite")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/lazyframe",
        "-p",
        "polars_metal._pytest_plugin",
        "--ignore=tests/unit/lazyframe/cuda",
        "--no-header",
        "-q",
        "--tb=no",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(POLARS_TEST_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )

    actual = _parse_failures(result.stdout)
    known = _load_known_failures()

    new = sorted(actual - known)
    fixed = sorted(known - actual)

    msgs: list[str] = []
    if new:
        msgs.append(
            f"{len(new)} new failure(s) under engine=MetalEngine() "
            f"(regressions in our engine):\n  " + "\n  ".join(new)
        )
    if fixed:
        msgs.append(
            f"{len(fixed)} previously-failing test(s) now pass "
            f"(remove from _polars_known_failures.txt):\n  " + "\n  ".join(fixed)
        )

    assert not new, "\n\n".join(msgs) + "\n\nFull pytest output:\n" + result.stdout[-2000:]
