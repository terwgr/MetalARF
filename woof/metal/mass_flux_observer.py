"""Independent lateral acoustic flux observation for case mass closure."""
import numpy as np


def boundary_mass_tendency(state, cfg):
    if cfg.specified or cfg.nested:
        raise NotImplementedError("Metal case boundary-flux observation requires periodic or open boundaries")
    if not cfg.open_x and not cfg.open_y:
        return 0.0
    nz, ny, nx = state.u.shape[0], cfg.ny, cfg.nx
    out = state.scratch((nz, ny + nx), "verify_boundary_mass_flux")
    arrays = {name: getattr(state, name) for name in
              ("u", "v", "u_pp", "v_pp", "c1h", "c2h", "dnw", "msfu", "msfv")}
    arrays.update(mu=state.total_mu(), out=out)
    parameters = {"nz": nz, "ny": ny, "nx": nx, "nedge": ny + nx,
                  "has_msf": int(state.has_msf), "open_x": int(cfg.open_x),
                  "open_y": int(cfg.open_y),
                  "rdx": float(np.float32(1.0 / cfg.dx)),
                  "rdy": float(np.float32(1.0 / cfg.dy))}
    out.buffer.context.submit_kernel("case_boundary_mass_flux", arrays, parameters,
                                     exact_policy=False, grid_size=out.size)
    contributions = out.to_numpy().astype(np.float64)
    # Match the original independent FP64 sums for the two boundary axes.
    return float(np.sum(contributions[:, :ny], dtype=np.float64)
                 + np.sum(contributions[:, ny:], dtype=np.float64))
