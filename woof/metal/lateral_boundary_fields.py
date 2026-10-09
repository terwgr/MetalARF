"""Metal implementation of WRF coupled snapshot fields for lateral BCs.

This helper is intentionally limited to the transformation used while a
``StateBoundaryFrames`` collector snapshots a Metal ``DomainState``.  It
keeps the full fields resident; extraction/readback of each perimeter side is
left to the I/O boundary that consumes the snapshot.
"""

from __future__ import annotations

import numpy as np

from woof.metal.array import MetalArray
from woof.wrf_exact import ENABLED as WRF_EXACT


_KIND = {"u": 0, "v": 1, "theta": 2, "phi": 3}


def coupled_device_fields(state) -> dict[str, MetalArray]:
    """Return the coupled WRF boundary inventory on Metal.

    The arithmetic follows ``_coupled_device_fields``: total column mass is
    formed first, u/v face masses are periodic averages with the physical
    domain boundary faces replaced by their adjacent mass cells, then the
    hybrid coefficient is applied.  Momentum map factors are divided after
    multiplying the prognostic field.  Scalars use half-level coupling;
    ``phi`` uses full-level coefficients.  ``mu`` is a shaped view of the
    resident perturbation mass, matching the source helper's broadcast.

    No field values are staged through NumPy.  The result arrays are typed
    MetalArray allocations except for ``mu``, which is an allocation-free
    Metal view; explicit host readback belongs to side extraction/I/O.
    """
    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("coupled Metal boundary snapshots require a Metal DomainState")
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal coupled boundary snapshots currently refuse WRF exact arithmetic")

    xp = state._array_module
    context = xp.context
    arrays = {name: getattr(state, name, None) for name in (
        "u", "v", "thp", "php", "mub2d", "mup", "thb", "c1h", "c2h",
        "c1f", "c2f", "msft", "msfu", "msfv")}
    for name, value in arrays.items():
        if not isinstance(value, MetalArray):
            raise TypeError(f"Metal boundary snapshot field {name!r} is not resident")

    mny, mnx = (int(v) for v in state.mup.shape)
    if tuple(state.mub2d.shape) != (mny, mnx):
        raise ValueError("base and perturbation column mass shapes differ")
    has_msf = bool(state.has_msf)
    external = tuple(getattr(state, "_external_scalar_boundary_fields", ("qv",)))

    result: dict[str, MetalArray] = {
        "mu": state.mup.view((1, mny, mnx)),
    }
    fields = {
        "u": (state.u, 0),
        "v": (state.v, 1),
        "theta": (state.thp, 2),
        "phi": (state.php, 3),
    }
    for name in external:
        if name == "qv" and getattr(state, name, None) is None:
            continue  # dry legacy/direct snapshot callers
        target = getattr(state, name, None)
        if not isinstance(target, MetalArray):
            raise ValueError(f"analysis boundary field {name!r} has no Metal state array")
        fields[name] = (target, 5)

    for name, (target, kind) in fields.items():
        if target.ndim != 3:
            raise ValueError(f"coupled snapshot field {name!r} must be 3-D")
        nz, ny, nx = (int(v) for v in target.shape)
        if kind == 0 and (ny != mny or nx != mnx + 1):
            raise ValueError("u boundary snapshot must use (nz, ny, nx+1) faces")
        if kind == 1 and (ny != mny + 1 or nx != mnx):
            raise ValueError("v boundary snapshot must use (nz, ny+1, nx) faces")
        if kind in (2, 3, 5) and (ny, nx) != (mny, mnx):
            raise ValueError(f"scalar boundary field {name!r} is not mass-grid shaped")
        out = xp.zeros((nz, ny, nx), dtype=np.float32)
        bound = {
            "target": target,
            "mub2d": state.mub2d,
            "mup": state.mup,
            "thb": state.thb,
            "c1h": state.c1h,
            "c2h": state.c2h,
            "c1f": state.c1f,
            "c2f": state.c2f,
            "msft": state.msft,
            "msfu": state.msfu,
            "msfv": state.msfv,
            "out": out,
        }
        params = {
            "kind": kind,
            "has_msf": int(has_msf),
            "thb_3d": int(state.thb.ndim == 3),
            "nz": nz,
            "ny": ny,
            "nx": nx,
            "mny": mny,
            "mnx": mnx,
        }
        context.submit_kernel("coupled_boundary_field", bound, params,
                              exact_policy=False, grid_size=out.size)
        result[name] = out
    return result


__all__ = ["coupled_device_fields"]
