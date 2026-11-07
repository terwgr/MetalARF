"""Typed Metal launchers for the single-moment WSM-family column kernels.

The shaders are translated from woof/core/kernels/wsm*.cu by
tools/wsm_metal/generate.py; each unit has a column kernel and, where WRF
provides one, a REFL_10CM kernel, at CUDA's two level tiers.
"""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray
from woof.wrf_exact import ENABLED as WRF_EXACT

#: CUDA's level tiers: the shallowest entry point that holds the column.
_TIERS = ((64, ""), (80, "_deep"))
_SURFACE = frozenset(("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
                      "graupelncv", "hailnc", "hailncv", "sr"))


def _entrypoint(base: str, nz: int) -> str:
    for kmax, suffix in _TIERS:
        if nz <= kmax:
            return base + suffix
    raise ValueError(f"Metal {base} supports nz <= {_TIERS[-1][0]}, got {nz}")


def _words(context, values: np.ndarray, dtype) -> MetalArray:
    values = np.ascontiguousarray(values, dtype=dtype)
    array = MetalArray(context.buffer(values.nbytes), values.shape, values.dtype)
    array.buffer.write(values.tobytes())
    return array


def _check(label, arrays, shape_of):
    context = None
    for name, array in arrays.items():
        if not isinstance(array, MetalArray):
            raise TypeError(f"Metal {label} {name} must be a MetalArray")
        if array.shape != shape_of(name):
            raise ValueError(
                f"Metal {label} {name} shape {array.shape} != {shape_of(name)}")
        if array.dtype != np.dtype(np.float32) or not array.flags.c_contiguous:
            raise TypeError(
                f"Metal {label} {name} requires contiguous float32 storage")
        context = context or array.buffer.context
        if array.buffer.context is not context:
            raise ValueError(f"Metal {label} resources must share one context")
    return context


def submit_wsm_column(unit: str, arrays: dict, coef: np.ndarray, dt: float,
                      conservative_surface: bool = False):
    """Submit one thread per column of ``unit``'s column kernel.

    ``arrays`` holds the kernel's float32 fields by CUDA parameter name;
    ``coef`` is the init routine's float32 words.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            f"Metal {unit} does not implement WRF exact-D arithmetic")
    theta = arrays["theta"]
    if not isinstance(theta, MetalArray) or theta.ndim != 3:
        raise TypeError(f"Metal {unit} theta must be a three-dimensional MetalArray")
    nz, ny, nx = map(int, theta.shape)
    if nz < 2:
        raise ValueError(f"Metal {unit} requires nz >= 2")
    entry = _entrypoint(unit + "_column", nz)
    if conservative_surface:
        entry += "_cs"
    if ny * nx > 0x7FFFFFFF:
        raise ValueError(f"Metal {unit} dimensions must fit CUDA int indexing")
    dt32 = np.float32(dt)
    if not np.isfinite(dt32) or dt32 <= 0:
        raise ValueError(f"{unit} dt must be finite and positive, got {dt!r}")
    context = _check(unit, arrays, lambda n: (ny, nx) if n in _SURFACE
                     else (nz, ny, nx))
    resources = dict(arrays)
    resources["coef"] = _words(context, coef, np.float32)
    return context.submit_kernel(
        entry, resources, {"delt": float(dt32), "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=32)


def submit_wsm_refl(unit: str, arrays: dict, rf: np.ndarray, rd: np.ndarray,
                    coef: np.ndarray | None = None):
    """Submit ``unit``'s REFL_10CM kernel (dBZ floored at -35)."""
    if WRF_EXACT:
        raise NotImplementedError(
            f"Metal {unit} reflectivity supports ordinary arithmetic only")
    refl = arrays["refl"]
    if not isinstance(refl, MetalArray) or refl.ndim != 3:
        raise TypeError(f"Metal {unit} refl must be a three-dimensional MetalArray")
    nz, ny, nx = map(int, refl.shape)
    entry = _entrypoint(unit + "_refl10cm_column", nz)
    context = _check(unit + " reflectivity", arrays, lambda n: (nz, ny, nx))
    resources = dict(arrays)
    resources["rf"] = _words(context, rf, np.float32)
    if coef is not None:
        resources["coef"] = _words(context, coef, np.float32)
    resources["rd"] = _words(context, rd, np.float64)
    return context.submit_kernel(
        entry, resources, {"nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=32)


def submit_wsm6r_column(arrays: dict, coef: np.ndarray, dt: float):
    """Submit WSM6R's column kernel (no level tiers; ``work`` holds fall speeds)."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal WSM6R does not implement WRF exact-D arithmetic")
    theta = arrays["theta"]
    if not isinstance(theta, MetalArray) or theta.ndim != 3:
        raise TypeError("Metal WSM6R theta must be a three-dimensional MetalArray")
    nz, ny, nx = map(int, theta.shape)
    if nz < 1 or max(nz, ny, nx, ny * nx, 3 * nz * ny * nx) > 0x7FFFFFFF:
        raise ValueError("Metal WSM6R dimensions must be positive and fit int indexing")
    dt32 = np.float32(dt)
    if not np.isfinite(dt32) or dt32 <= 0:
        raise ValueError(f"WSM6R dt must be finite and positive, got {dt!r}")
    context = _check("WSM6R", arrays, lambda n: (3, nz, ny, nx) if n == "work"
                     else (ny, nx) if n in _SURFACE else (nz, ny, nx))
    resources = dict(arrays)
    resources["cf"] = _words(context, coef, np.float32)
    return context.submit_kernel(
        "wsm6r_column", resources,
        {"delt": float(dt32), "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=32)


def submit_wsm6r_effective_radii(arrays: dict, pidnc: float, pidn0s: float):
    """Submit WSM6's effective radii on a WSM6R state."""
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal WSM6R does not implement WRF exact-D arithmetic")
    theta = arrays["theta"]
    nz, ny, nx = map(int, theta.shape)
    context = _check("WSM6R radii", arrays, lambda n: (nz, ny, nx))
    return context.submit_kernel(
        "wsm6r_effective_radii", arrays,
        {"pidnc": float(np.float32(pidnc)), "pidn0s": float(np.float32(pidn0s)),
         "nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=32)
