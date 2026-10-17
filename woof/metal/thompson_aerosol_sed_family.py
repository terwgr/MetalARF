"""Typed Metal submissions for Thompson aerosol sedimentation stages.

These submitters define the MP28 production binding contract for the source
entrypoints in ``shaders/thompson_aerosol_sed_family.metal``.  They remain
fail-closed until those entrypoints and their reviewed math dependencies are
registered in the generated runtime ABI.
"""

from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from woof.metal.array import MetalArray
from woof.core.thompson_aerosol_contract import (
    AM_R, AV_C, NC_SEDIMENT_FLOOR,
)


_FLOAT = np.dtype(np.float32)
_THREADS = 32
_LEVEL_THREADS = 512
# The soft64 levels kernel is register-heavy: its pipeline cannot launch a
# 512-thread block (8 columns x 64 levels), which hung.  Use the column
# kernel until the levels kernel fits a threadgroup.
_LEVEL_PARALLEL_ENABLED = False
_KMAX = 256
_SHALLOW_KMAX = 64
_R1 = np.float32(1.0e-12)
_R2 = np.float32(1.0e-6)
_OBMR = np.float32(3.333333433e-1)
_SED_W_LIMIT = np.float32(1.0e-1)
_SED_HGT_AGL = np.float32(500.0)
_AM_I = np.float32(4.660029297e+2)


def _overlap(left: MetalArray, right: MetalArray) -> bool:
    if left.buffer.handle != right.buffer.handle:
        return False
    return (left.byte_offset < right.byte_offset + right.nbytes
            and right.byte_offset < left.byte_offset + left.nbytes)


def _validate_array(name: str, value: MetalArray, shape: tuple[int, ...],
                    context) -> None:
    if (not isinstance(value, MetalArray) or value.dtype != _FLOAT
            or value.shape != shape or not value.flags.c_contiguous):
        raise TypeError(
            f"Thompson aerosol sedimentation {name} must be contiguous "
            f"float32 with shape {shape}")
    if value.buffer.context is not context:
        raise ValueError(
            "Thompson aerosol sedimentation arrays must share one context")


def submit_cloud_sedimentation(
        qc, cloud_number_entry, cloud_number_tendency,
        temperature, pressure, qv, vertical_velocity, dz, *,
        reference_density, rain_active_columns, cloud_active_columns,
        gamma_moments, dt: float, level_parallel: bool):
    """Submit masked production cloud fallout for one MP28 column grid."""
    if not isinstance(qc, MetalArray) or qc.ndim != 3:
        raise TypeError("MP28 cloud fallout requires 3-D MetalArray state")
    nz, ny, nx = map(int, qc.shape)
    if not (2 <= nz <= _KMAX) or ny <= 0 or nx <= 0:
        raise ValueError(
            f"MP28 cloud fallout requires 2 <= nz <= {_KMAX} and a "
            "nonempty horizontal grid")
    size = nz * ny * nx
    columns = ny * nx
    if max(size, columns) > np.iinfo(np.int32).max:
        raise ValueError("MP28 cloud fallout dimensions must fit int32")
    context = qc.buffer.context
    volume_shape = (nz, ny, nx)
    surface_shape = (ny, nx)
    arrays = {
        "qc": qc, "cloud_number_entry": cloud_number_entry,
        "cloud_number_tendency": cloud_number_tendency,
        "temperature": temperature, "pressure": pressure, "qv": qv,
        "reference_density": reference_density,
        "rain_active_columns": rain_active_columns,
        "cloud_active_columns": cloud_active_columns,
        "vertical_velocity": vertical_velocity, "dz": dz,
    }
    for name, value in arrays.items():
        expected = (surface_shape if name.endswith("_columns")
                    else volume_shape)
        _validate_array(name, value, expected, context)
    if not isinstance(gamma_moments, Mapping) or set(gamma_moments) != {
            "ccg2", "ccg4", "ccg5", "ocg1", "ocg2"}:
        raise ValueError("MP28 cloud fallout needs all five gamma moments")
    for name, value in gamma_moments.items():
        _validate_array(name, value, (16,), context)
        arrays[name] = value

    # Source diagnostics are not part of the forecast contract. Their fixed
    # ABI slots bind a safe in-bounds array and diagnostic_mask=0 prevents all
    # reads and writes through those slots.
    arrays.update({
        "out_mass_velocity": pressure,
        "out_number_velocity": pressure,
        "out_cloud_mass": pressure,
        "out_cloud_number": pressure,
    })
    for left_index, (left_name, left) in enumerate(arrays.items()):
        if left_name.startswith("out_"):
            continue
        for right_name, right in list(arrays.items())[:left_index]:
            if right_name.startswith("out_"):
                continue
            if _overlap(left, right):
                raise ValueError(
                    f"restricted MP28 cloud fallout resources overlap: "
                    f"{left_name}, {right_name}")

    dt32 = np.float32(dt)
    if not np.isfinite(dt32) or dt32 <= 0.0:
        raise ValueError(f"MP28 cloud fallout dt must be positive, got {dt!r}")
    level_parallel = bool(_LEVEL_PARALLEL_ENABLED and level_parallel
                          and nz <= _SHALLOW_KMAX)
    if level_parallel:
        entrypoint = "thompson_aa_cloud_sediment_levels_64_source"
        threads = _LEVEL_THREADS
        grid = ((columns + 7) // 8) * threads
        parameters = {
            "dt": float(dt32), "r1": float(_R1), "r2": float(_R2),
            "am_r": float(np.float32(AM_R)), "obmr": float(_OBMR),
            "av_c": float(np.float32(AV_C)),
            "nc_sed_floor": float(np.float32(NC_SEDIMENT_FLOOR)),
            "sed_w_limit": float(_SED_W_LIMIT),
            "sed_hgt_agl": float(_SED_HGT_AGL),
            "nz": nz, "ny": ny, "nx": nx,
            "has_rain_mask": 1, "has_cloud_mask": 1,
        }
        arrays.pop("out_mass_velocity")
        arrays.pop("out_number_velocity")
        arrays.pop("out_cloud_mass")
        arrays.pop("out_cloud_number")
    else:
        suffix = "64" if nz <= _SHALLOW_KMAX else "256"
        entrypoint = f"thompson_aa_cloud_sediment_{suffix}_source"
        threads = _THREADS
        grid = columns
        parameters = {
            "dt": float(dt32), "r1": float(_R1), "r2": float(_R2),
            "am_r": float(np.float32(AM_R)), "obmr": float(_OBMR),
            "av_c": float(np.float32(AV_C)),
            "nc_sed_floor": float(np.float32(NC_SEDIMENT_FLOOR)),
            "sed_w_limit": float(_SED_W_LIMIT),
            "sed_hgt_agl": float(_SED_HGT_AGL),
            "nz": nz, "ny": ny, "nx": nx,
            "has_rain_mask": 1, "has_cloud_mask": 1,
            "diagnostic_mask": 0,
        }
    return context.submit_kernel(
        entrypoint, arrays, parameters, exact_policy=False,
        grid_size=grid, threads_per_group=threads)


def submit_final_phase_cleanup(
        qc, qi, ni, temperature, cloud_number_entry, ice_number_entry,
        cloud_number_tendency, pressure, qv, *, dt: float, wrf39: bool = False):
    """Submit the MP28 cloud/ice number-conserving phase cleanup."""
    if not isinstance(qc, MetalArray) or qc.ndim != 3:
        raise TypeError("MP28 phase cleanup requires 3-D MetalArray state")
    shape = tuple(map(int, qc.shape))
    if not all(shape):
        raise ValueError("MP28 phase-cleanup fields must be nonempty")
    size = int(np.prod(shape, dtype=np.int64))
    if size > np.iinfo(np.int32).max:
        raise ValueError("MP28 phase-cleanup size must fit int32")
    context = qc.buffer.context
    arrays = {
        "qc": qc, "qi": qi, "ni": ni, "temperature": temperature,
        "cloud_number_entry": cloud_number_entry,
        "ice_number_entry": ice_number_entry,
        "cloud_number_tendency": cloud_number_tendency,
        "pressure": pressure, "qv": qv,
    }
    for name, value in arrays.items():
        _validate_array(name, value, shape, context)
    names = list(arrays)
    for index, left_name in enumerate(names):
        for right_name in names[:index]:
            if _overlap(arrays[left_name], arrays[right_name]):
                raise ValueError(
                    f"restricted MP28 phase-cleanup resources overlap: "
                    f"{left_name}, {right_name}")
    dt32 = np.float32(dt)
    if not np.isfinite(dt32) or dt32 <= 0.0:
        raise ValueError(
            f"MP28 phase-cleanup dt must be positive, got {dt!r}")
    flat = {name: value.view((size,), byte_offset=value.byte_offset)
            for name, value in arrays.items()}
    return context.submit_kernel(
        "thompson_aa_final_phase_cleanup_source", flat,
        {"dt": float(dt32), "r1": float(_R1), "r2": float(_R2),
         "am_i": float(_AM_I), "size": size, "wrf39": int(wrf39)},
        exact_policy=False, grid_size=size, threads_per_group=256)


__all__ = ["submit_cloud_sedimentation", "submit_final_phase_cleanup"]
