"""Run Polars' own test suites with engine forced to MetalEngine().

This catches real regressions in our engine path: any Polars test that passes
on CPU but fails under our engine is a defect.

Each upstream path has its own baseline of version-skew failures (tests
written for newer Polars features that the pinned wheel doesn't have, or
missing optional deps). The baseline files live next to this test as
`_polars_known_failures_<short>.txt`. For each path we assert that the
observed failure set is a subset of its baseline — new failures = real
regressions in our engine, fixed failures = the baseline can shrink.

Re-capture a baseline (preserving the one-line header):

    cd references/polars/py-polars && \\
      ( head -n 1 ../../../tests/conformance/_polars_known_failures_<short>.txt; \\
        python3 -m pytest <path> --no-header -q --tb=no 2>&1 \\
          | grep '^FAILED' | sed 's/^FAILED //; s/ - .*//' | sort ) \\
        > ../../../tests/conformance/_polars_known_failures_<short>.txt.new \\
      && mv ../../../tests/conformance/_polars_known_failures_<short>.txt{.new,}

Skipped if `references/polars/py-polars` isn't present (e.g., references/
wasn't cloned). Run `make refresh-refs`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
POLARS_TEST_ROOT = REPO_ROOT / "references" / "polars" / "py-polars"
CONFORMANCE_DIR = Path(__file__).parent

# (short-id, upstream-path-relative-to-py-polars, known-failures-filename, extra-pytest-args).
#
# Each upstream path is run as a separate subprocess with the
# polars_metal pytest plugin loaded so LazyFrame.collect() defaults to
# engine=MetalEngine(). Adding a new path is: append a tuple, capture its
# baseline (see module docstring), and check it in.
SUITE_PATHS: list[tuple[str, str, str, tuple[str, ...]]] = [
    (
        "lazyframe",
        "tests/unit/lazyframe",
        "_polars_known_failures.txt",
        ("--ignore=tests/unit/lazyframe/cuda",),
    ),
    (
        "operations_filter",
        "tests/unit/operations/test_filter.py",
        "_polars_known_failures_operations_filter.txt",
        (),
    ),
    (
        "operations_comparison",
        "tests/unit/operations/test_comparison.py",
        "_polars_known_failures_operations_comparison.txt",
        (),
    ),
    (
        "operations_select",
        "tests/unit/operations/test_select.py",
        "_polars_known_failures_operations_select.txt",
        (),
    ),
    (
        "expr_binary",
        "tests/unit/expr/test_binary.py",
        "_polars_known_failures_expr_binary.txt",
        (),
    ),
    (
        "operations_group_by",
        "tests/unit/operations/test_group_by.py",
        "_polars_known_failures_operations_group_by.txt",
        (),
    ),
    (
        "operations_aggregation",
        "tests/unit/operations/aggregation/test_aggregations.py",
        "_polars_known_failures_operations_aggregation.txt",
        (),
    ),
]


def _load_known_failures(filename: str) -> set[str]:
    path = CONFORMANCE_DIR / filename
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text().splitlines()
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
    not POLARS_TEST_ROOT.exists(),
    reason="references/polars not cloned; run `make refresh-refs`",
)
@pytest.mark.parametrize(
    ("short_id", "upstream_path", "known_failures_filename", "extra_args"),
    SUITE_PATHS,
    ids=[entry[0] for entry in SUITE_PATHS],
)
def test_polars_suite_under_metal(
    short_id: str,
    upstream_path: str,
    known_failures_filename: str,
    extra_args: tuple[str, ...],
) -> None:
    """Run an upstream Polars test path with engine forced to MetalEngine().

    Asserts no new failures vs the recorded baseline for that path.
    """
    if shutil.which("pyarrow") is None:
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            pytest.skip("pyarrow not installed; required for Polars' own test suite")

    full_path = POLARS_TEST_ROOT / upstream_path
    if not full_path.exists():
        pytest.skip(f"upstream path missing: {upstream_path}")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        upstream_path,
        "-p",
        "polars_metal._pytest_plugin",
        *extra_args,
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

    # pytest exit codes: 0 = all passed, 1 = some failed (expected when baseline
    # has entries). Anything else (2 interrupted, 3 internal error, 4 usage
    # error, 5 no tests collected) means the suite didn't actually run, so the
    # empty-`actual` set is meaningless and would silently pass the diff below.
    if result.returncode not in (0, 1):
        raise AssertionError(
            f"[{short_id}] pytest subprocess exited with code {result.returncode} "
            f"(expected 0 or 1); the test suite did not run to completion.\n\n"
            f"Last stdout lines:\n{result.stdout[-2000:]}\n\n"
            f"Stderr:\n{result.stderr[-2000:]}"
        )

    actual = _parse_failures(result.stdout)
    known = _load_known_failures(known_failures_filename)

    new = sorted(actual - known)
    fixed = sorted(known - actual)

    msgs: list[str] = []
    if new:
        msgs.append(
            f"[{short_id}] {len(new)} new failure(s) under engine=MetalEngine() "
            f"(regressions in our engine):\n  " + "\n  ".join(new)
        )
    if fixed:
        msgs.append(
            f"[{short_id}] {len(fixed)} previously-failing test(s) now pass — "
            f"shrink the baseline by removing these node ids from "
            f"tests/conformance/{known_failures_filename}:\n  " + "\n  ".join(fixed)
        )

    assert not (new or fixed), (
        "\n\n".join(msgs) + "\n\nFull pytest output:\n" + result.stdout[-2000:]
    )
