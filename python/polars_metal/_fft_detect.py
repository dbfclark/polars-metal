"""M6 A3: detect FFT sentinel bindings from a LazyFrame's outermost with_columns layer.

Mirrors _vector_detect exactly, with an INDEPENDENT with_columns patch + cache so it chains
safely with the rolling and vector-search patches (each pops only its own cache). Reuses the
struct-json helpers from _vector_detect (general, not vector-specific).
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import polars as pl

from polars_metal import _detect_common as dc
from polars_metal._detect_common import _alias_name, _literal_int, _struct_fields
from polars_metal._fft_namespace import FFT_SENTINEL_TAG

# id(result LazyFrame) -> captured Expr objects (fast path). Get-not-pop; evicted on GC via weakref (see _detect_common.lookup).
_fft_lf_exprs_cache: dict = {}  # id(lf) -> (weakref.ref, exprs); managed by _detect_common
_PATCH_ATTR = "_polars_metal_fft_original_with_columns"

dc.install_with_columns_capture(_PATCH_ATTR, _fft_lf_exprs_cache)


@dataclass(frozen=True)
class FftBinding:
    out_name: str
    input_col: str
    op: int


def _binding_from_expr_json(expr_json: dict, out_name: str) -> FftBinding | None:
    try:
        if FFT_SENTINEL_TAG not in json.dumps(expr_json):
            return None
        input_col = None
        op = None
        for fld in _struct_fields(expr_json):
            alias_name = _alias_name(fld)
            if alias_name and alias_name.startswith(FFT_SENTINEL_TAG):
                input_col = alias_name[len(FFT_SENTINEL_TAG) :]
                op = _literal_int(fld)
        if input_col is None or op is None:
            return None
        return FftBinding(out_name=out_name, input_col=input_col, op=op)
    except Exception:
        return None


def find_fft_bindings(lf: pl.LazyFrame) -> list[FftBinding]:
    """Return FftBinding for each sentinel alias in the outermost with_columns layer."""
    try:
        cached = dc.lookup(_fft_lf_exprs_cache, lf)
        if cached is not None:
            out: list[FftBinding] = []
            for expr in cached:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    j = json.loads(expr.meta.serialize(format="json"))
                name = _alias_name(j)
                inner = j["Alias"][0] if name else j
                b = _binding_from_expr_json(inner, name or "")
                if b is not None and b.out_name:
                    out.append(b)
            return out

        # Slow fallback: pre-filter then bounded scan.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            if FFT_SENTINEL_TAG not in lf.explain():
                return []
            plan = lf.serialize(format="json")
        # Bounded parse of the exprs fragment (same rfind trick as _rolling_detect).
        key = '"exprs":['
        i = plan.rfind(key)
        if i == -1:
            return []
        start = i + len(key) - 1
        j = plan.rfind(',"options":', start)
        frag = plan[start:j] if j != -1 else plan[start:]
        nodes = json.loads(frag)
        out = []
        for node in nodes if isinstance(nodes, list) else []:
            name = _alias_name(node)
            inner = node["Alias"][0] if name else node
            b = _binding_from_expr_json(inner, name or "")
            if b is not None and b.out_name:
                out.append(b)
        return out
    except Exception:
        return []
