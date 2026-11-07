"""Typed Metal launcher for the Purdue Lin column kernel."""
from __future__ import annotations

import numpy as np

from .array import MetalArray

_KMAX = 128
_VOLUMES = ("th", "qv", "qc", "qr", "qi", "qs", "qg", "rho", "pii",
            "pressure", "z", "dz")
_SURFACES = ("ht", "rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
             "graupelncv", "sr")


def submit_lin_column(theta, qv, qc, qr, qi, qs, qg, rho, pii, pressure, z, dz,
                      ht, rainnc, rainncv, snownc, snowncv, graupelnc,
                      graupelncv, sr, dt):
    """Submit one thread per column; the arrays are contiguous float32."""
    arrays = dict(zip(_VOLUMES + _SURFACES,
                      (theta, qv, qc, qr, qi, qs, qg, rho, pii, pressure, z, dz,
                       ht, rainnc, rainncv, snownc, snowncv, graupelnc,
                       graupelncv, sr)))
    nz, ny, nx = map(int, theta.shape)
    if not 2 <= nz <= _KMAX:
        raise ValueError(f"Metal Lin runs 2 to {_KMAX} levels, got {nz}")
    context = theta.buffer.context
    for name, array in arrays.items():
        shape = (nz, ny, nx) if name in _VOLUMES else (ny, nx)
        if (not isinstance(array, MetalArray) or array.shape != shape
                or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous
                or array.buffer.context is not context):
            raise TypeError(f"Metal Lin {name} must be contiguous float32 {shape} "
                            "in the shared context")
    ncol = ny * nx
    flat = {name: array.view((array.size,)) for name, array in arrays.items()}
    return context.submit_kernel(
        "lin_column", flat, {"dt": float(np.float32(dt)), "nz": nz, "ncol": ncol},
        exact_policy=False, grid_size=ncol, threads_per_group=32)


def launch_refl10cm_lin(qv, qr, qs, qg, t, p, refl):
    """Lin's REFL_10CM with its fixed intercepts, including melting particles."""
    from woof.core.refl import radar_init
    arrays = {"qv": qv, "qr": qr, "qs": qs, "qg": qg, "t": t, "pressure": p, "refl": refl}
    nz, ny, nx = map(int, refl.shape)
    if not 1 <= nz <= 256:
        raise ValueError(f"Metal Lin reflectivity runs 1 to 256 levels, got {nz}")
    context = refl.buffer.context
    for name, array in arrays.items():
        if (not isinstance(array, MetalArray) or array.shape != (nz, ny, nx)
                or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous
                or array.buffer.context is not context):
            raise TypeError(f"Metal Lin reflectivity {name} must be contiguous "
                            f"float32 {(nz, ny, nx)} in the shared context")
    rc = radar_init(0)

    def words(values):
        values = np.ascontiguousarray(np.concatenate(values), dtype=np.float64)
        out = MetalArray(context.buffer(values.nbytes), values.shape, np.float64)
        out.buffer.write(values.tobytes())
        return out

    flat = {name: array.view((array.size,)) for name, array in arrays.items()}
    flat["tables"] = words((rc.xxds, rc.xdts, rc.xxdg, rc.xdtg, rc.simpson))
    flat["radar_constants"] = words((np.asarray((
        rc.k_w, rc.m_w_0.real, rc.m_w_0.imag, rc.m_i_0.real, rc.m_i_0.imag,
        209.4393310546875, 4.0e6), dtype=np.float64),))
    return context.submit_kernel(
        "refl10cm_lin_column", flat, {"nz": nz, "ny": ny, "nx": nx},
        exact_policy=False, grid_size=ny * nx, threads_per_group=32)
