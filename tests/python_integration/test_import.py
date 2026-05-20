"""polars_metal can be imported, exposes its API, and loads its Rust extension."""

import polars_metal


def test_module_imports() -> None:
    assert polars_metal.__version__ == "0.0.0"


def test_metal_engine_exported() -> None:
    assert hasattr(polars_metal, "MetalEngine")


def test_native_extension_loaded() -> None:
    from polars_metal import _native

    assert _native.version_string() == "0.0.0"


def test_metal_device_acquires() -> None:
    from polars_metal import _native

    name = _native.device_name()
    assert isinstance(name, str)
    assert len(name) > 0


def test_mlx_add_f32_smoke() -> None:
    from polars_metal import _native

    result = _native.add_f32([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    assert result == [11.0, 22.0, 33.0]
