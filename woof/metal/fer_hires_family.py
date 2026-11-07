"""Typed Metal launcher for the Ferrier-Aligo column kernel and its tables."""
from __future__ import annotations

import numpy as np

from .array import MetalArray

_VOLUMES = ("th", "qv", "qc", "qr", "qi", "qt", "f_ice", "f_rain", "f_rimef",
            "pressure", "pii", "rho", "dz8w")
_SURFACES = ("rainnc", "rainncv", "sr")


def device_tables(context, dtph: float) -> MetalArray:
    """The kernel's table buffer for a domain whose init step is ``dtph``.

    The integral tables come from WRF's file; GPVS and the SAVE scalars are
    filled by fer_hires_init's kernels. Cached per context and binary32 DTPH.
    """
    from woof.core.fer_hires import NX, TABLE_WORDS, host_table_words, load_table
    cache = context.__dict__.setdefault("_fer_hires_tables", {})
    dtph32 = np.float32(dtph)
    if not np.isfinite(dtph32) or dtph32 <= 0:
        raise ValueError(f"Ferrier init step must be positive, got {dtph}")
    key = int(dtph32.view(np.uint32))
    tab = cache.get(key)
    if tab is None:
        words = host_table_words(load_table())
        tab = MetalArray(context.buffer(4 * TABLE_WORDS), (TABLE_WORDS,), np.float32)
        tab.buffer.write(words.tobytes())
        context.submit_kernel("fer_hires_init_tables", {"tab": tab}, {"threads": NX},
                              exact_policy=False, grid_size=NX, threads_per_group=32)
        context.submit_kernel("fer_hires_init_scalars", {"tab": tab},
                              {"dtph": float(dtph32)}, exact_policy=False,
                              grid_size=1, threads_per_group=32)
        cache[key] = tab
    return tab


def submit_fer_hires_column(th, qv, qc, qr, qi, qt, f_ice, f_rain, f_rimef, p,
                            pii, rho, dz8w, rainnc, rainncv, sr, tab, dt, gid,
                            refl=None, analysis_only=False):
    """Submit one thread per column over contiguous float32 MetalArrays."""
    arrays = dict(zip(_VOLUMES + _SURFACES,
                      (th, qv, qc, qr, qi, qt, f_ice, f_rain, f_rimef, p, pii,
                       rho, dz8w, rainnc, rainncv, sr)))
    nz, ny, nx = map(int, th.shape)
    context = th.buffer.context
    for name, array in arrays.items():
        shape = (nz, ny, nx) if name in _VOLUMES else (ny, nx)
        if (not isinstance(array, MetalArray) or array.shape != shape
                or array.dtype != np.dtype(np.float32)
                or not array.flags.c_contiguous
                or array.buffer.context is not context):
            raise TypeError(f"Metal Ferrier {name} must be contiguous float32 "
                            f"{shape} in the shared context")
    flat = {name: array.view((array.size,)) for name, array in arrays.items()}
    refl_due = refl is not None
    if refl is None:
        # The kernel writes refl only on output steps; any other buffer serves.
        refl = context.__dict__.setdefault("_fer_hires_unused", MetalArray(
            context.buffer(4), (1,), np.float32))
    flat["refl"] = refl.view((refl.size,))
    flat["tab"] = tab
    params = {"dt": float(np.float32(dt)), "domain_id": int(gid),
              "refl_due": int(refl_due), "nz": nz, "ny": ny, "nx": nx,
              "analysis_only": int(bool(analysis_only))}
    return context.submit_kernel(
        "fer_hires_column", flat, params, exact_policy=False,
        grid_size=ny * nx, threads_per_group=32)
