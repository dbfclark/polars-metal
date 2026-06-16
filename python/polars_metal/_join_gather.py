"""Dense-key detection for join->gather lowering. A left/inner equi-join on an
integer key equals ``value_reordered[fact_key]`` IFF the dim key is a permutation
of ``0..dim_height-1`` (unique, contiguous, no gaps, no nulls). Otherwise the
caller takes the CPU-lookup branch. Correctness-critical: a wrong 'dense' verdict
produces silently wrong results.

Note: this helper assumes the caller has already verified null_count()==0 on the
key column before converting to a numpy int array. A numpy int64 array cannot
carry Polars nulls, so no null-in-key check is needed here."""
from __future__ import annotations

import numpy as np


def dense_positions(
    key: np.ndarray, value: np.ndarray, dim_height: int
) -> tuple[bool, np.ndarray | None]:
    """Return ``(is_dense, value_reordered)`` where ``value_reordered[k]`` is the
    dim value for key ``k``.

    ``is_dense`` is ``False`` when ``key`` is not a ``0..dim_height-1``
    permutation (empty, gaps, duplicates, out-of-range, or nulls).
    When ``is_dense`` is ``False`` the second element is ``None`` and the caller
    must use the CPU-join branch.
    """
    if dim_height <= 0:
        return False, None
    if key.shape[0] != dim_height or value.shape[0] != dim_height:
        return False, None
    # Range check MUST come before any scatter to prevent out-of-bounds indexing.
    if int(key.min()) != 0 or int(key.max()) != dim_height - 1:
        return False, None
    # Permutation check: every slot hit exactly once (rejects duplicates).
    seen = np.zeros(dim_height, dtype=bool)
    seen[key] = True
    if not seen.all():
        return False, None
    reordered = np.empty(dim_height, dtype=value.dtype)
    reordered[key] = value
    return True, reordered
