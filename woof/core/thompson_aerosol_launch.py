"""Thompson aerosol-aware launch contracts and input validation."""

from __future__ import annotations

import contextlib
import contextvars

import numpy as np

from woof.core.kernels import get_kernel, get_kernel_int_defines
from woof.core.state import DTYPE


# ---------------------------------------------------------------------------
# Which Thompson generation the aerosol kernels compile (RunConfig.
# thompson_version).
# ---------------------------------------------------------------------------
#
# ``wrf_461`` compiles every unit exactly as before, through get_kernel, so
# its assembled source and binary are the ones every earlier gate measured.
# ``wrf_39_noaa`` compiles the SAME files with one integer define,
# THOMPSON_AA_WRF39, whose #if arms carry the operational WRF 3.9 fork's
# statements (NOAA-EMC/HRRR v4.1.21 module_mp_thompson.F, cited per arm).
# The define is inserted after thompson_aerosol_common.cuh (kernels/
# __init__.py module_source_int_defines), so the header carries no #if:
# its fork variants are separate functions the .cu arms call.
#
# The generation is a property of one microphysics call, set by the
# adapter around it (thompson_version_scope) and read by every launcher
# through aerosol_kernel, so no launcher signature changes.
THOMPSON_VERSIONS = ("wrf_461", "wrf_39_noaa")
WRF39_DEFINES = (("THOMPSON_AA_WRF39", 1),)

_ACTIVE_VERSION: contextvars.ContextVar[str] = contextvars.ContextVar(
    "gpuwm_thompson_version", default="wrf_461")


def active_thompson_version() -> str:
    """The generation the current microphysics call runs."""
    return _ACTIVE_VERSION.get()


@contextlib.contextmanager
def thompson_version_scope(version: str):
    """Run the enclosed launches as ``version`` (one of THOMPSON_VERSIONS)."""
    if version not in THOMPSON_VERSIONS:
        raise ValueError(
            f"thompson_version={version!r} is not one of {THOMPSON_VERSIONS}")
    token = _ACTIVE_VERSION.set(version)
    try:
        yield version
    finally:
        _ACTIVE_VERSION.reset(token)


def aerosol_kernel(module: str, func: str):
    """``get_kernel(module, func)`` compiled for the active generation."""
    if _ACTIVE_VERSION.get() == "wrf_39_noaa":
        return get_kernel_int_defines(module, func, WRF39_DEFINES)
    return get_kernel(module, func)


# ---------------------------------------------------------------------------
# CUDA module names.  These are the six .cu files that receive
# thompson_aerosol_common.cuh from woof/core/kernels/__init__.py's
# _EXTRA_HEADERS allow-list; the tuple below and that dict must agree, which
# tests/test_kernel_loader_inert.py asserts.
# ---------------------------------------------------------------------------

AEROSOL_COMMON_HEADER = "thompson_aerosol_common.cuh"

STATE_MODULE = "thompson_aerosol_state"
SAT_MODULE = "thompson_aerosol_sat"
COLD_MODULE = "thompson_aerosol_cold"
WARM_MODULE = "thompson_aerosol_warm"
SED_MODULE = "thompson_aerosol_sed"

AEROSOL_KERNEL_MODULES = (
    STATE_MODULE, SAT_MODULE, COLD_MODULE, WARM_MODULE, SED_MODULE,
)

#: The mp=8 translation unit.  It is byte-frozen and is never given a header.
CLASSIC_MODULE = "thompson"

#: woof's default 1-D launch width, matching woof/core/thompson.py.
DEFAULT_THREADS = 256

#: Fortran shape of ``tnccn_act`` (module_mp_thompson.F:247-251, :393).
CCN_ACTIVATION_SHAPE = (7, 9, 7, 5, 4)

#: Row order of ``thompson_aa_probe_constant_tables``'s output.  Part of the
#: published probe API; do not reorder.

#: Eff_aero collector species selectors; must match
#: THOMPSON_AA_SPECIES_* in thompson_aerosol_common.cuh.
SPECIES_RAIN = 0
SPECIES_SNOW = 1
SPECIES_GRAUPEL = 2
SPECIES_CODES = {"r": SPECIES_RAIN, "s": SPECIES_SNOW, "g": SPECIES_GRAUPEL}


# ---------------------------------------------------------------------------
# Argument validation.
# ---------------------------------------------------------------------------

def validate_fields(fields: dict[str, object]) -> tuple[tuple[int, ...], int]:
    """Require every named field to be one float32 C-contiguous shape.

    Transcribed from ``woof/core/thompson.py:50-64``.  Returns the common
    shape and its element count.
    """
    first = next(iter(fields.values()))
    shape = first.shape
    if not shape:
        raise ValueError("Thompson aerosol fields must be arrays")
    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    return shape, int(np.prod(shape, dtype=np.int64))


def validate_fp64_fortran_table(name, value, shape) -> None:
    """Require a float64 Fortran-ordered device table of the given shape.

    Transcribed from ``woof/core/thompson.py:64-71``.  mp=28's aerosol table
    contract (:mod:`woof.core.thompson_aerosol_runtime`) uploads float64
    Fortran arrays exactly like the classic one, so the same check applies.
    """
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != np.dtype(np.float64):
        raise TypeError(f"{name} must be float64, got {value.dtype}")
    if not value.flags.f_contiguous:
        raise ValueError(f"{name} must be Fortran-contiguous")


def validate_int_fields(fields: dict[str, object],
                        shape: tuple[int, ...]) -> None:
    """Require every named field to be one int32 C-contiguous ``shape``."""
    for name, value in fields.items():
        if value.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {value.shape}")
        if value.dtype != np.dtype(np.int32):
            raise TypeError(f"{name} must be int32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")


def launch_grid(size: int, threads: int = DEFAULT_THREADS
                ) -> tuple[tuple[int], tuple[int]]:
    """Return ``(grid, block)`` for a flat ``size``-element 1-D kernel."""
    if size <= 0:
        raise ValueError("launch size must be positive")
    blocks = (int(size) + threads - 1) // threads
    return (blocks,), (threads,)


# ---------------------------------------------------------------------------
# Pointwise probes of thompson_aerosol_common.cuh.
# ---------------------------------------------------------------------------









































__all__ = [
    "AEROSOL_COMMON_HEADER",
    "AEROSOL_KERNEL_MODULES",
    "CCN_ACTIVATION_SHAPE",
    "CLASSIC_MODULE",
    "COLD_MODULE",
    "DEFAULT_THREADS",
    "SAT_MODULE",
    "SED_MODULE",
    "SPECIES_CODES",
    "SPECIES_GRAUPEL",
    "SPECIES_RAIN",
    "SPECIES_SNOW",
    "STATE_MODULE",
    "WARM_MODULE",
    "launch_grid",
    "validate_fields",
    "validate_fp64_fortran_table",
    "validate_int_fields",
]
