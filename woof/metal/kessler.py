"""Metal launcher for the ordinary float32 Kessler column kernel."""

from __future__ import annotations

import numpy as np

from woof.core import constants as c
from woof.wrf_exact import ENABLED as WRF_EXACT


KESSLER_VERTICAL_LEVEL_BOUNDS = (None, 256)


def submit_kessler(t, qv, qc, qr, rho, pii, z, dz8w, rainnc, rainncv,
                   dt: float):
    """Submit ``kessler_column`` for contiguous Metal float32 arrays.

    The four moisture/temperature fields and accumulated rain are updated in
    place.  WRF exact mode is refused because this implementation has no
    verified exact arithmetic/transcendental layer.
    """
    if WRF_EXACT:
        raise NotImplementedError(
            "Metal Kessler supports ordinary float32 arithmetic only; "
            "WRF exact arithmetic is not implemented")

    nz, ny, nx = t.shape
    if ny < 1 or nx < 1:
        raise ValueError("Kessler requires nonempty horizontal dimensions")
    if nz < 2:
        raise ValueError("Kessler requires nz >= 2 for its top rdzk spacing")
    if nz > KESSLER_VERTICAL_LEVEL_BOUNDS[1]:
        raise ValueError(
            f"nz={nz} exceeds Kessler per-column storage KESS_KMAX="
            f"{KESSLER_VERTICAL_LEVEL_BOUNDS[1]}")

    arrays = {
        "t": t, "qv": qv, "qc": qc, "qr": qr,
        "rho": rho, "pii": pii, "z": z, "dz8w": dz8w,
        "rainnc": rainnc, "rainncv": rainncv,
    }
    parameters = {
        "dt": float(np.float32(dt)), "nz": nz, "ny": ny, "nx": nx,
        "svp1": float(np.float32(c.SVP1)),
        "svp2": float(np.float32(c.SVP2)),
        "svp3": float(np.float32(c.SVP3)),
        "svpt0": float(np.float32(c.SVPT0)),
        "xlv": float(np.float32(c.XLV)),
        "cp": float(np.float32(c.CP)),
        "rhowater": float(np.float32(c.RHOWATER)),
        "ep2": float(np.float32(c.EP2)),
    }
    return t.buffer.context.submit_kernel(
        "kessler_column", arrays, parameters, exact_policy=False,
        grid_size=ny * nx)
