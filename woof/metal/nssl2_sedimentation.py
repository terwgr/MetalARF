"""Column-owned Metal submission for native NSSL sedimentation."""

from __future__ import annotations

import math
import numpy as np

from woof.metal.array import MetalArray, MetalArrayModule


_TPB = 32


def workspace_columns(context, nz, ncol):
    """Bound deep scratch by the actual device budget, shared with admission."""
    words = 15 * (int(nz) + 1)
    return min(int(ncol), max(1, int(context.recommended_working_set_size)
                             // 64 // (words * np.dtype(np.float32).itemsize)))


def workspace_bytes(context, nz, ncol):
    return (workspace_columns(context, nz, ncol) * 15 * (int(nz) + 1) * 4
            if int(nz) > 128 else 0)


def submit_nssl2_sediment_all(
        context, air_density, temperature_k, state, dz,
        ignored_accumulator, category_export, dt_s, *, nz, ny, nx):
    """Run all six default sedimentation categories in source order.

    The kernel uses one lane per horizontal column and runs the admitted
    single-column WRF sedimentation routines in order. Category exports are
    overwritten, while the ignored receipt is cleared before its staged use.
    """
    dims = (nz, ny, nx)
    if any(isinstance(value, (bool, np.bool_))
           or not isinstance(value, (int, np.integer)) or value < 1
           for value in dims):
        raise ValueError("NSSL sedimentation dimensions must be positive integers")
    nz, ny, nx = map(int, dims)
    if nz < 2 or nz > 256:
        raise ValueError("NSSL sedimentation requires 2 <= nz <= 256")
    if not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
        raise ValueError("NSSL dt_s must be positive and finite")
    dt = np.float32(dt_s)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("NSSL dt_s must be representable as positive float32")
    expected = (nz, ny, nx)
    for name, value in (("air_density", air_density),
                        ("temperature_k", temperature_k), ("dz", dz)):
        if not isinstance(value, MetalArray):
            raise TypeError(f"{name} must be a MetalArray")
        if value.shape != expected or value.dtype != np.dtype(np.float32):
            raise ValueError(f"{name} must be FP32 with shape {expected}")
        if value.buffer.context is not context or not value.flags.c_contiguous:
            raise ValueError(f"{name} must be contiguous in the submitting context")
    if not isinstance(state, MetalArray) or state.shape != (16, *expected):
        raise ValueError("state must have shape (16, nz, ny, nx)")
    if state.dtype != np.dtype(np.float32) or state.buffer.context is not context:
        raise ValueError("state must be FP32 storage in the submitting context")
    if not isinstance(ignored_accumulator, MetalArray) or ignored_accumulator.shape != (ny, nx):
        raise ValueError("ignored_accumulator must have shape (ny, nx)")
    if not isinstance(category_export, MetalArray) or category_export.shape != (5, ny, nx):
        raise ValueError("category_export must have shape (5, ny, nx)")
    for name, value in (("ignored_accumulator", ignored_accumulator),
                        ("category_export", category_export)):
        if value.dtype != np.dtype(np.float32) or value.buffer.context is not context:
            raise ValueError(f"{name} must be FP32 storage in the submitting context")
    ignored_accumulator.fill(0.0)
    arrays = {"air_density": air_density, "temperature_k": temperature_k,
              "state": state, "dz": dz,
              "ignored_accumulator": ignored_accumulator,
              "category_export": category_export}
    parameters = {"dt": dt, "nz": nz, "ny": ny, "nx": nx}
    if nz <= 128:
        tier = 64 if nz <= 64 else 128
        return context.submit_kernel(
            f"nssl2_sediment_all_{tier}", arrays, parameters,
            exact_policy=False, grid_size=ny*nx, threads_per_group=_TPB)
    # Keep deep-column scratch off the per-thread Metal stack and bound
    # its live allocation to a share of the device's working-set budget.
    columns = workspace_columns(context, nz, ny*nx)
    workspace = MetalArrayModule(context).empty((columns, 15, nz+1), np.float32)
    submission = None
    for start in range(0, ny*nx, columns):
        count = min(columns, ny*nx-start)
        arrays["workspace"] = workspace[:count]
        submission = context.submit_kernel(
            "nssl2_sediment_all_256", arrays,
            dict(parameters, column0=start, ncolumns=count),
            exact_policy=False, grid_size=count, threads_per_group=_TPB)
    return submission



__all__ = ["submit_nssl2_sediment_all"]


def submit_nssl2_cloud_sediment_exact(context, air_density, temperature_k, qc,
                                    qndrop, dz, cloud_surface_export, dt_s):
    """Submit the independently matched strict cloud sedimentation stage."""
    from woof.wrf_exact import ENABLED
    if not ENABLED:
        raise RuntimeError("strict NSSL cloud sedimentation requires WRF exact mode")
    shape = tuple(air_density.shape)
    if len(shape) != 3 or not 2 <= shape[0] <= 256:
        raise ValueError("NSSL cloud sedimentation requires 2 <= nz <= 256")
    for name, value in (("air_density", air_density), ("temperature_k", temperature_k),
                        ("qc", qc), ("qndrop", qndrop), ("dz", dz),
                        ("cloud_surface_export", cloud_surface_export)):
        expected = shape[1:] if name == "cloud_surface_export" else shape
        if (not isinstance(value, MetalArray) or value.shape != expected
                or value.dtype != np.dtype(np.float32) or not value.flags.c_contiguous
                or value.buffer.context is not context):
            raise ValueError(f"{name} must be contiguous FP32 with shape {expected} in context")
    dt = np.float32(dt_s)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("NSSL dt_s must be representable as positive float32")
    nz, ny, nx = shape
    return context.submit_kernel("nssl2_cloud_sediment_exact", {
        "air_density": air_density, "temperature_k": temperature_k,
        "qc": qc, "qndrop": qndrop, "dz": dz,
        "cloud_surface_export": cloud_surface_export,
    }, {"dt": dt, "nz": nz, "ny": ny, "nx": nx},
        exact_policy=True, grid_size=ny*nx, threads_per_group=_TPB)


def submit_nssl2_rain_sediment_exact(context, air_density, qr, qnr, dz,
                                   rainnc, rainncv, dt_s):
    """Submit the independently matched strict serial rain stage."""
    from woof.wrf_exact import ENABLED
    if not ENABLED:
        raise RuntimeError("strict NSSL rain sedimentation requires WRF exact mode")
    shape = tuple(air_density.shape)
    if len(shape) != 3 or not 2 <= shape[0] <= 256:
        raise ValueError("NSSL rain sedimentation requires 2 <= nz <= 256")
    for name, value in (("air_density", air_density), ("qr", qr), ("qnr", qnr),
                        ("dz", dz), ("rainnc", rainnc), ("rainncv", rainncv)):
        expected = shape[1:] if name in ("rainnc", "rainncv") else shape
        if (not isinstance(value, MetalArray) or value.shape != expected
                or value.dtype != np.dtype(np.float32) or not value.flags.c_contiguous
                or value.buffer.context is not context):
            raise ValueError(f"{name} must be contiguous FP32 with shape {expected} in context")
    dt = np.float32(dt_s)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("NSSL dt_s must be representable as positive float32")
    nz, ny, nx = shape
    return context.submit_kernel("nssl2_rain_sediment_exact", {
        "air_density": air_density, "qr": qr, "qnr": qnr,
        "dz": dz, "rainnc": rainnc, "rainncv": rainncv,
    }, {"dt": dt, "nz": nz, "ny": ny, "nx": nx},
        exact_policy=True, grid_size=ny*nx, threads_per_group=_TPB)


def submit_nssl2_ice_sediment_exact(context, air_density, qi, qni, dz,
                                  icenc, icencv, dt_s):
    """Submit the independently matched strict serial ice stage."""
    from woof.wrf_exact import ENABLED
    if not ENABLED:
        raise RuntimeError("strict NSSL ice sedimentation requires WRF exact mode")
    shape = tuple(air_density.shape)
    if len(shape) != 3 or not 2 <= shape[0] <= 256:
        raise ValueError("NSSL ice sedimentation requires 2 <= nz <= 256")
    for name, value in (("air_density", air_density), ("qi", qi), ("qni", qni),
                        ("dz", dz), ("icenc", icenc), ("icencv", icencv)):
        expected = shape[1:] if name in ("icenc", "icencv") else shape
        if (not isinstance(value, MetalArray) or value.shape != expected
                or value.dtype != np.dtype(np.float32) or not value.flags.c_contiguous
                or value.buffer.context is not context):
            raise ValueError(f"{name} must be contiguous FP32 with shape {expected} in context")
    dt = np.float32(dt_s)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("NSSL dt_s must be representable as positive float32")
    nz, ny, nx = shape
    return context.submit_kernel("nssl2_ice_sediment_exact", {
        "air_density": air_density, "qi": qi, "qni": qni,
        "dz": dz, "icenc": icenc, "icencv": icencv,
    }, {"dt": dt, "nz": nz, "ny": ny, "nx": nx},
        exact_policy=True, grid_size=ny*nx, threads_per_group=_TPB)


def submit_nssl2_snow_sediment_exact(context, air_density, qs, qns, dz,
                                  snownc, snowncv, dt_s):
    """Submit the independently matched strict serial snow stage."""
    from woof.wrf_exact import ENABLED
    if not ENABLED:
        raise RuntimeError("strict NSSL snow sedimentation requires WRF exact mode")
    shape = tuple(air_density.shape)
    if len(shape) != 3 or not 2 <= shape[0] <= 256:
        raise ValueError("NSSL snow sedimentation requires 2 <= nz <= 256")
    for name, value in (("air_density", air_density), ("qs", qs), ("qns", qns),
                        ("dz", dz), ("snownc", snownc), ("snowncv", snowncv)):
        expected = shape[1:] if name in ("snownc", "snowncv") else shape
        if (not isinstance(value, MetalArray) or value.shape != expected
                or value.dtype != np.dtype(np.float32) or not value.flags.c_contiguous
                or value.buffer.context is not context):
            raise ValueError(f"{name} must be contiguous FP32 with shape {expected} in context")
    dt = np.float32(dt_s)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("NSSL dt_s must be representable as positive float32")
    nz, ny, nx = shape
    return context.submit_kernel("nssl2_snow_sediment_exact", {
        "air_density": air_density, "qs": qs, "qns": qns,
        "dz": dz, "snownc": snownc, "snowncv": snowncv,
    }, {"dt": dt, "nz": nz, "ny": ny, "nx": nx},
        exact_policy=True, grid_size=ny*nx, threads_per_group=_TPB)
