"""polars-metal: Metal-backed execution engine for Polars on Apple Silicon."""

from polars_metal import _native

__version__ = _native.version_string()
