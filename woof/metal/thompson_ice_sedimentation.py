"""Typed Metal submission for classic Thompson held-density ice fallout."""

from __future__ import annotations

import numpy as np

from .array import MetalArray


def _overlap(left: MetalArray, right: MetalArray) -> bool:
    if left.buffer is not right.buffer:
        return False
    a0, a1 = left.byte_offset, left.byte_offset + left.nbytes
    b0, b1 = right.byte_offset, right.byte_offset + right.nbytes
    return a0 < b1 and b0 < a1


def submit_thompson_ice_sedimentation_with_density(
        *, qi, ni, temperature, pressure, qv, reference_density, dz,
        rainnc, rainncv, snownc, snowncv, dt):
    """Submit diagnosis, sediment subcycling, and terminal update in order.

    Per-level work arrays are allocated on the same Metal context and retained
    by the submitted command buffers. They replace the large per-thread local
    arrays used by a single-pass transcription.
    """
    arrays = {
        "qi": qi, "ni": ni, "temperature": temperature,
        "pressure": pressure, "qv": qv,
        "reference_density": reference_density, "dz": dz,
        "rainnc": rainnc, "rainncv": rainncv,
        "snownc": snownc, "snowncv": snowncv,
    }
    if not isinstance(temperature, MetalArray) or temperature.ndim != 3:
        raise TypeError("Metal Thompson ice sedimentation temperature must be 3-D")
    nz, ny, nx = shape = tuple(temperature.shape)
    if min(shape) <= 0:
        raise ValueError("Metal Thompson ice sedimentation requires nonempty dimensions")
    if nz < 2 or nz > 256:
        raise ValueError(f"Metal Thompson ice sedimentation requires 2 <= nz <= 256, got {nz}")
    if ny * nx > 0x7FFFFFFF or nz * ny * nx > 0x7FFFFFFF:
        raise ValueError("Metal Thompson ice sedimentation dimensions exceed the int32 ABI")
    surface_shape = (ny, nx)
    for name, value in arrays.items():
        if not isinstance(value, MetalArray):
            raise TypeError(f"Metal Thompson ice sedimentation {name} must be a MetalArray")
        expected = surface_shape if name in {
            "rainnc", "rainncv", "snownc", "snowncv"} else shape
        if (value.shape != expected or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(
                f"Metal Thompson ice sedimentation {name} must be contiguous FP32 with shape {expected}")
    context = temperature.buffer.context
    if any(value.buffer.context is not context for value in arrays.values()):
        raise ValueError("Metal Thompson ice sedimentation fields must share one context")
    names = tuple(arrays)
    for i, left_name in enumerate(names):
        for right_name in names[i + 1:]:
            if _overlap(arrays[left_name], arrays[right_name]):
                raise ValueError(
                    f"Metal Thompson ice sedimentation {left_name} must not alias {right_name}")
    dt32 = np.float32(dt)
    if not np.isfinite(dt32) or dt32 <= 0.0:
        raise ValueError(f"Metal Thompson ice sedimentation dt must be finite and positive, got {dt!r}")

    level_shape = shape
    level_scratch_names = (
        "density", "ice_mass", "ice_number", "mass_velocity", "number_velocity",
        "mass_flux", "number_flux", "qi_tendency", "ni_tendency")
    scratch = {
        name: MetalArray(context.buffer(nz * ny * nx * 4), level_shape, np.float32)
        for name in level_scratch_names
    }
    surface_scratch = {
        "sediment_top": MetalArray(context.buffer(ny * nx * 4), surface_shape, np.int32),
        "nstep": MetalArray(context.buffer(ny * nx * 4), surface_shape, np.int32),
        "exported": MetalArray(context.buffer(ny * nx * 4), surface_shape, np.float32),
    }
    parameters = {"dt": float(dt32), "nz": nz, "ny": ny, "nx": nx}
    context.submit_kernel(
        "thompson_ice_sedimentation_seed",
        {"qi": qi, "ni": ni, "temperature": temperature,
         "pressure": pressure, "qv": qv, "reference_density": reference_density,
         **{name: scratch[name] for name in (
             "density", "ice_mass", "ice_number", "mass_velocity",
             "number_velocity", "qi_tendency", "ni_tendency")},
         "active_marker": scratch["mass_flux"]},
        parameters, exact_policy=False, grid_size=ny * nx, threads_per_group=128)
    context.submit_kernel(
        "thompson_ice_sedimentation_active_ice",
        {"qi": qi, "ni": ni, "reference_density": reference_density,
         "density": scratch["density"], "ice_mass": scratch["ice_mass"],
         "ice_number": scratch["ice_number"],
         "mass_velocity": scratch["mass_velocity"],
         "number_velocity": scratch["number_velocity"],
         "active_marker": scratch["mass_flux"]},
        parameters, exact_policy=False, grid_size=ny * nx, threads_per_group=128)
    context.submit_kernel(
        "thompson_ice_sedimentation_select_timestep",
        {"dz": dz, "mass_velocity": scratch["mass_velocity"],
         "number_velocity": scratch["number_velocity"],
         "active_marker": scratch["mass_flux"],
         "sediment_top": surface_scratch["sediment_top"],
         "nstep": surface_scratch["nstep"]},
        parameters, exact_policy=False, grid_size=ny * nx, threads_per_group=128)
    context.submit_kernel(
        "thompson_ice_sedimentation_subcycling",
        {"dz": dz, "density": scratch["density"],
         "ice_mass": scratch["ice_mass"], "ice_number": scratch["ice_number"],
         "mass_velocity": scratch["mass_velocity"],
         "number_velocity": scratch["number_velocity"],
         "mass_flux": scratch["mass_flux"], "number_flux": scratch["number_flux"],
         "qi_tendency": scratch["qi_tendency"], "ni_tendency": scratch["ni_tendency"],
         "sediment_top": surface_scratch["sediment_top"],
         "nstep": surface_scratch["nstep"], "exported": surface_scratch["exported"]},
        parameters, exact_policy=False, grid_size=ny * nx, threads_per_group=128)
    return context.submit_kernel(
        "thompson_ice_sedimentation_finalize",
        {"qi": qi, "ni": ni, "density": scratch["density"],
         "qi_tendency": scratch["qi_tendency"],
         "ni_tendency": scratch["ni_tendency"], "exported": surface_scratch["exported"],
         "rainnc": rainnc, "rainncv": rainncv,
         "snownc": snownc, "snowncv": snowncv},
        parameters, exact_policy=False, grid_size=ny * nx, threads_per_group=128)
