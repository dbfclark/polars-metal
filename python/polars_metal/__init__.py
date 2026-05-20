"""polars-metal: Metal-backed execution engine for Polars on Apple Silicon."""

from polars_metal import _native
from polars_metal._engine import MetalEngine

__version__ = _native.version_string()

__all__ = ["MetalEngine", "__version__"]
