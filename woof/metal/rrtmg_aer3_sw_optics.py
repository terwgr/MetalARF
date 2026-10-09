"""Device-resident Thompson aerosol optics for legacy RRTMG SW."""
from __future__ import annotations

import numpy as np
from .array import MetalArray, MetalArrayModule


def aer3_sw_optics_metal(p3d, t3d, qv3d, dz8w, nwfa, nifa, nlayers, *,
                         smoke_aod=None, smoke_feedback=False):
    """Return ``(ztaua, zasya, zomga, taod)`` on the input Metal context.

    Every input is an explicit contiguous float32 MetalArray. This entrypoint
    deliberately refuses NumPy and CuPy arrays so no residency conversion can
    happen implicitly. Inputs are column-major ``(ncol,nz)``; spectral outputs
    use RRTMG SW's ``(ncol,14,nlayers)`` band-major layout.
    """
    if type(smoke_feedback) is not bool:
        raise ValueError("smoke_feedback must be an explicit boolean")
    first = p3d
    if not isinstance(first, MetalArray) or first.ndim != 2:
        raise TypeError("aer_opt=3 Metal optics require (ncol,nz) MetalArrays")
    nc, nz = map(int, first.shape)
    nlayers = int(nlayers)
    if nc < 1 or nz < 1 or nlayers < nz or max(nc, nz, nlayers, nc * nlayers) > 0x7fffffff:
        raise ValueError("aer_opt=3 Metal dimensions are unsupported")
    context = first.buffer.context
    arrays = {}
    for name, value in (("p3d", p3d), ("t3d", t3d), ("qv3d", qv3d),
                        ("dz8w", dz8w), ("nwfa", nwfa), ("nifa", nifa)):
        if (not isinstance(value, MetalArray) or value.shape != (nc, nz)
                or value.dtype != np.dtype(np.float32)
                or not value.flags.c_contiguous):
            raise TypeError(f"{name} must be contiguous float32 Metal {(nc,nz)}")
        if value.buffer.context is not context:
            raise ValueError("aer_opt=3 inputs must share one Metal context")
        arrays[name] = value
    if smoke_feedback:
        if (not isinstance(smoke_aod, MetalArray) or smoke_aod.shape != (nc, nz)
                or smoke_aod.dtype != np.dtype(np.float32)
                or not smoke_aod.flags.c_contiguous
                or smoke_aod.buffer.context is not context):
            raise TypeError("smoke feedback requires same-context float32 Metal smoke_aod")
        smoke = smoke_aod
    elif smoke_aod is not None:
        raise ValueError("smoke_aod was supplied while smoke feedback is disabled")
    else:
        smoke = MetalArrayModule(context).zeros((nc, nz), np.float32)
    xp = MetalArrayModule(context)
    out = {name: xp.empty(shape, np.float32) for name, shape in (
        ("ztaua", (nc, 14, nlayers)), ("zasya", (nc, 14, nlayers)),
        ("zomga", (nc, 14, nlayers)), ("taod", (nc, nz)))}
    arrays.update(smoke_aod=smoke, **out)
    context.submit_kernel(
        "rrtmg_aer3_sw_optics", arrays,
        {"nc": nc, "nz": nz, "nlayers": nlayers,
         "smoke_feedback": int(smoke_feedback)},
        exact_policy=False, grid_size=nc * nlayers, threads_per_group=128)
    return out["ztaua"], out["zasya"], out["zomga"], out["taod"]


__all__ = ["aer3_sw_optics_metal"]
