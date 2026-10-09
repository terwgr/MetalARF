"""Arwen Level-3 global spherical-harmonic dynamical-core prototype.

The package is deliberately standalone and research-only.  Its public door is
``python -m woof.globe.spectral``.
"""
from .compression import (
    CompressedScalarField,
    CompressedWindField,
    compress_scalar,
    compress_wind,
    decode_scalar,
    decode_wind,
)
from .config import GlobalSpectralRunConfig, load_config
from .export import export_checkpoint_latlon, read_latlon_export
from .grid import GaussianGrid
from .initial_conditions import primitive_rest_state, williamson2_state
from .pins import PINS_HASH, pins_receipt
from .primitive import HeldSuarezForcing, PrimitiveDryModel, SigmaCoordinate
from .runner import run
from .sampling import sample_gradient, sample_scalar, sample_wind
from .shallow_water import ShallowWaterModel
from .state import PrimitiveDryState, ShallowWaterState
from .transform import SphericalHarmonicTransform
from .vector import VorticityDivergenceOperator

__all__ = [
    "CompressedScalarField",
    "CompressedWindField",
    "GaussianGrid",
    "GlobalSpectralRunConfig",
    "HeldSuarezForcing",
    "PINS_HASH",
    "PrimitiveDryModel",
    "PrimitiveDryState",
    "ShallowWaterModel",
    "ShallowWaterState",
    "SigmaCoordinate",
    "SphericalHarmonicTransform",
    "VorticityDivergenceOperator",
    "compress_scalar",
    "compress_wind",
    "decode_scalar",
    "decode_wind",
    "export_checkpoint_latlon",
    "load_config",
    "pins_receipt",
    "primitive_rest_state",
    "read_latlon_export",
    "run",
    "sample_gradient",
    "sample_scalar",
    "sample_wind",
    "williamson2_state",
]
