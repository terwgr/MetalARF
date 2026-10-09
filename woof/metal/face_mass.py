"""Metal launchers for periodic and IEVA stage face column masses."""

from __future__ import annotations

from woof.wrf_exact import ENABLED as WRF_EXACT


def submit_average_mass_faces(state, mu, *, yface: bool, slot: str):
    """Average a mass-plane Metal array onto one periodic staggered axis."""
    from .array import MetalArray

    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal face mass averaging requires a Metal DomainState")
    if not isinstance(mu, MetalArray):
        raise TypeError("Metal face mass input must be a MetalArray")
    if mu.dtype.name != "float32" or len(mu.shape) != 2:
        raise TypeError("Metal face mass input must be a 2-D float32 array")
    ny, nx = mu.shape
    if not ny or not nx:
        raise ValueError("Metal face mass averaging requires nonempty extents")
    out = state.scratch((ny + int(yface), nx + int(not yface)), slot)
    state._array_module.context.submit_kernel(
        "average_mass_faces",
        {"mu": mu, "out": out},
        {"yface": int(yface), "boundary_x": 0, "boundary_y": 0,
         "ny": ny, "nx": nx},
        exact_policy=False, grid_size=out.size)
    return out


def submit_stage_face_masses(state, cfg, mu):
    """Build IEVA stage masses, keeping its WRF-exact branch separate."""
    from .array import MetalArray

    if not getattr(state, "_metal_setup_state", False):
        raise TypeError("Metal stage face masses require a Metal DomainState")
    ny, nx = state.mup.shape
    mux = state.scratch((ny, nx + 1), "ieva_mux")
    muy = state.scratch((ny + 1, nx), "ieva_muy")
    boundary_x = bool(cfg.open_x or getattr(cfg, "specified", False)
                      or getattr(cfg, "nested", False))
    boundary_y = bool(cfg.open_y or getattr(cfg, "specified", False)
                      or getattr(cfg, "nested", False))
    context = state._array_module.context

    if WRF_EXACT:
        # WRF exact computes directly from the perturbation and two base words;
        # using the ordinary stage-total input would change its expression tree.
        context.submit_kernel(
            "stage_face_masses",
            {"mup": state.mup, "mub2d": state.mub2d,
             "mux": mux, "muy": muy},
            {"wrf_exact": 1, "boundary_x": int(boundary_x),
             "boundary_y": int(boundary_y), "ny": ny, "nx": nx},
            exact_policy=True, grid_size=ny * nx)
        return mux, muy

    if not isinstance(mu, MetalArray):
        raise TypeError("ordinary Metal stage mass input must be a MetalArray")
    if mu.dtype.name != "float32" or mu.shape != (ny, nx):
        raise TypeError("ordinary Metal stage mass input must be float32 (ny, nx)")
    for yface, out in ((False, mux), (True, muy)):
        context.submit_kernel(
            "average_mass_faces",
            {"mu": mu, "out": out},
            {"yface": int(yface), "boundary_x": int(boundary_x),
             "boundary_y": int(boundary_y), "ny": ny, "nx": nx},
            exact_policy=False,
            grid_size=out.size)
    return mux, muy
