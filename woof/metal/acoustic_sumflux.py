"""Typed Metal submission for WRF acoustic sumflux bookkeeping."""

from __future__ import annotations

import numpy as np

from .array import MetalArray
from woof.wrf_exact import ENABLED as WRF_EXACT

_MAX_I32 = 2_147_483_647


def _overlap(left: MetalArray, right: MetalArray) -> bool:
    return (left.buffer.handle == right.buffer.handle
            and left.byte_offset < right.byte_offset + right.nbytes
            and right.byte_offset < left.byte_offset + left.nbytes)


def _validate(name: str, targets: tuple, sources: tuple, nsub: int):
    expected = {
        "zero_sumflux": (3, 0),
        "accumulate_sumflux": (3, 3),
        "finish_sumflux": (3, 3),
    }
    if name not in expected:
        raise ValueError(f"unknown Metal sumflux entrypoint {name!r}")
    if (len(targets), len(sources)) != expected[name]:
        raise ValueError(f"{name} needs {expected[name][0]} targets and "
                         f"{expected[name][1]} sources")
    arrays = (*targets, *sources)
    if not all(isinstance(array, MetalArray) for array in arrays):
        raise TypeError("WRF sumflux resources must all be MetalArray values")
    if any(array.dtype != np.dtype(np.float32)
           or not array.flags.c_contiguous for array in arrays):
        raise TypeError("WRF sumflux resources must be contiguous float32 arrays")
    context = arrays[0].buffer.context
    if any(array.buffer.context is not context for array in arrays):
        raise ValueError("WRF sumflux arrays belong to different Metal contexts")
    for i, array in enumerate(arrays):
        for other in arrays[i + 1:]:
            if _overlap(array, other):
                raise ValueError(
                    "WRF sumflux resources overlap; CUDA restrict alias "
                    "contracts require disjoint buffers")
    ru_shape = targets[0].shape
    if len(ru_shape) != 3:
        raise ValueError("WRF sumflux arrays must use their physical 3-D staggers")
    nz, ny, nxf = ru_shape
    nx = nxf - 1
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError("WRF sumflux grid dimensions must be positive")
    expected_shapes = ((nz, ny, nx + 1), (nz, ny + 1, nx),
                       (nz + 1, ny, nx))
    if tuple(array.shape for array in targets) != expected_shapes:
        raise ValueError(
            "WRF sumflux target shapes must be u=(nz,ny,nx+1), "
            "v=(nz,ny+1,nx), w=(nz+1,ny,nx)")
    if sources and tuple(array.shape for array in sources) != expected_shapes:
        raise ValueError(
            "WRF sumflux source shapes must match the corresponding "
            "staggered target shapes")
    if max(nx, ny, nz) > _MAX_I32:
        raise OverflowError("WRF sumflux dimension exceeds typed int32 indexing")
    sizes = tuple(int(array.size) for array in targets)
    if any(size > _MAX_I32 for size in sizes):
        raise OverflowError("WRF sumflux extent exceeds typed int32 indexing")
    if name == "finish_sumflux":
        if isinstance(nsub, (bool, np.bool_)) or not isinstance(
                nsub, (int, np.integer)) or int(nsub) <= 0:
            raise ValueError("finish_sumflux requires a positive integer nsub")
        if int(nsub) > 16_777_216:
            raise OverflowError("nsub is not exactly representable as float32")
    return context, sizes, max(sizes), nx, ny, nz


def prepare_sumflux(name: str, targets: tuple, sources: tuple = (),
                    nsub: int = 0):
    """Validate and bind one zero/accumulate/finish Metal launch.

    The kernel uses a union grid but each stagger's count remains independent.
    All resources are captured by the returned launch closure, matching the
    CUDA call's allocation lifetime through command-buffer completion.
    """
    context, call = prepare_sumflux_call(name, targets, sources, nsub)
    if call is None:
        return lambda: None

    def launch():
        context.submit_kernel_batch((call,), label=name)

    return launch


def prepare_sumflux_call(name: str, targets: tuple, sources: tuple = (),
                         nsub: int = 0):
    """Freeze one sumflux dispatch for composition with adjacent kernels."""
    targets, sources = tuple(targets), tuple(sources)
    context, sizes, nmax, nx, ny, nz = _validate(
        name, targets, sources, nsub)
    if nmax == 0:
        return context, None
    params = {"nx": nx, "ny": ny, "nz": nz,
              "nu": sizes[0], "nv": sizes[1], "nw": sizes[2],
              "nmax": nmax}
    if name == "finish_sumflux":
        params["nsub"] = float(np.float32(nsub))
    resources = dict(zip(("ru_m", "rv_m", "ww_m"), targets))
    if sources:
        resources.update(zip(("ru", "rv", "ww"), sources))

    call = context.prepare_kernel_call(
        name, resources, params, exact_policy=WRF_EXACT, grid_size=nmax)
    return context, call
